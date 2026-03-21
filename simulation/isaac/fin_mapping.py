"""Derived fin mapping utilities for Isaac runtime and PID mixing.

This module centralizes how canonical fin commands map to articulation joint
targets, and how controller pitch/roll/yaw terms mix into per-fin actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from simulation.config_loader import load_config


def _as_float_list(values: Sequence[float], *, expected: int, name: str) -> list[float]:
    out = [float(v) for v in values]
    if len(out) != expected:
        raise ValueError(f"{name} must have {expected} values, got {len(out)}.")
    return out


def _as_int_list(values: Sequence[int], *, expected: int, name: str) -> list[int]:
    out = [int(v) for v in values]
    if len(out) != expected:
        raise ValueError(f"{name} must have {expected} values, got {len(out)}.")
    return out


@dataclass(frozen=True, slots=True)
class FinMapping:
    """Canonical mapping between controller fins and Isaac articulation joints."""

    # Joint command mapping: joint_cmd[i] = sign[i] * canonical_delta[source_idx[i]]
    joint_source_indices: tuple[int, int, int, int]
    joint_signs: tuple[float, float, float, float]

    # Controller mixing (per fin) for canonical fin order:
    # delta = pitch_weights * pitch_cmd + roll_weights * roll_cmd + yaw_weights * yaw_cmd
    pitch_weights: tuple[float, float, float, float]
    roll_weights: tuple[float, float, float, float]
    yaw_weights: tuple[float, float, float, float]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FinMapping":
        return cls(
            joint_source_indices=tuple(
                _as_int_list(
                    data.get("joint_source_indices", [0, 1, 2, 3]),
                    expected=4,
                    name="joint_source_indices",
                )
            ),
            joint_signs=tuple(
                _as_float_list(
                    data.get("joint_signs", [1.0, 1.0, 1.0, 1.0]),
                    expected=4,
                    name="joint_signs",
                )
            ),
            pitch_weights=tuple(
                _as_float_list(
                    data.get("pitch_weights", [1.0, 1.0, 0.0, 0.0]),
                    expected=4,
                    name="pitch_weights",
                )
            ),
            roll_weights=tuple(
                _as_float_list(
                    data.get("roll_weights", [0.0, 0.0, 1.0, 1.0]),
                    expected=4,
                    name="roll_weights",
                )
            ),
            yaw_weights=tuple(
                _as_float_list(
                    data.get("yaw_weights", [-1.0, 1.0, -1.0, 1.0]),
                    expected=4,
                    name="yaw_weights",
                )
            ),
        )


def default_fin_mapping() -> FinMapping:
    """Identity mapping — USD hinge axes are authored to match controller convention.

    As of feature 006-refactor-fin-physics, the USD joint frames include a 180° localRot
    flip so positive drive target = positive deflection per controller convention.
    No runtime index swap or sign negation is needed.
    Ref: specs/006-refactor-fin-physics/research.md RQ-2
    """
    return FinMapping.from_dict({})


def load_fin_mapping(mapping_path: str | Path | None = None) -> FinMapping:
    """Load fin mapping from YAML; fallback to default if missing/disabled."""
    if mapping_path is None:
        return default_fin_mapping()
    path = Path(mapping_path)
    raw = load_config(path)
    if not isinstance(raw, Mapping):
        return default_fin_mapping()
    if "fin_mapping" in raw and isinstance(raw["fin_mapping"], Mapping):
        return FinMapping.from_dict(raw["fin_mapping"])
    return FinMapping.from_dict(raw)


def mix_controls(
    *,
    pitch_cmd: float,
    roll_cmd: float,
    yaw_cmd: float,
    delta_max: float,
    mapping: FinMapping,
) -> tuple[float, float, float, float]:
    """Map pitch/roll/yaw terms (rad) to normalized per-fin action in [-1, 1]."""
    if delta_max <= 0.0:
        raise ValueError("delta_max must be > 0.")
    p = float(pitch_cmd)
    r = float(roll_cmd)
    y = float(yaw_cmd)
    fins = []
    for i in range(4):
        delta = (
            mapping.pitch_weights[i] * p
            + mapping.roll_weights[i] * r
            + mapping.yaw_weights[i] * y
        )
        fins.append(float(np.clip(delta / delta_max, -1.0, 1.0)))
    return (fins[0], fins[1], fins[2], fins[3])


def derive_mapping_from_axis_response(
    dominant_axis: Sequence[str],
    dominant_sign: Sequence[float],
) -> FinMapping:
    """Derive controller mixing from per-fin dominant axis/sign measurements.

    Args:
        dominant_axis: sequence of 4 labels, each in {"roll","pitch","yaw"}.
        dominant_sign: sequence of 4 signs (+/-), one per fin.
    """
    if len(dominant_axis) != 4 or len(dominant_sign) != 4:
        raise ValueError("dominant_axis and dominant_sign must each have 4 values.")

    pitch = [0.0, 0.0, 0.0, 0.0]
    roll = [0.0, 0.0, 0.0, 0.0]
    yaw = [0.0, 0.0, 0.0, 0.0]
    for i, (axis, sign) in enumerate(zip(dominant_axis, dominant_sign)):
        s = 1.0 if float(sign) >= 0.0 else -1.0
        if axis == "pitch":
            pitch[i] = s
        elif axis == "roll":
            roll[i] = s
        elif axis == "yaw":
            yaw[i] = s
        else:
            raise ValueError(f"Unsupported dominant axis {axis!r}.")

    # If yaw was not observed as dominant, keep the current stable differential default.
    if not any(abs(v) > 0.0 for v in yaw):
        yaw = [-1.0, 1.0, -1.0, 1.0]

    return FinMapping(
        joint_source_indices=(0, 1, 2, 3),
        joint_signs=(1.0, 1.0, 1.0, 1.0),
        pitch_weights=tuple(pitch),  # type: ignore[arg-type]
        roll_weights=tuple(roll),  # type: ignore[arg-type]
        yaw_weights=tuple(yaw),  # type: ignore[arg-type]
    )
