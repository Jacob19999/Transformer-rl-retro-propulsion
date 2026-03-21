"""Shared Isaac Sim control and observation conventions.

This module centralizes the action layout, fin ordering, observation indices,
and a few Isaac-only actuator constants so scripts and runtime code do not
redefine them independently.
"""

from __future__ import annotations

from simulation.isaac.usd.parts_registry import FIN_PRIM_NAMES, frd_to_zup, zup_to_frd
from simulation.isaac.fin_mapping import default_fin_mapping

# ---------------------------------------------------------------------------
# Action layout
# ---------------------------------------------------------------------------
ACTION_DIM = 5
ACTION_THRUST_IDX = 0
ACTION_FIN_START = 1
ACTION_FIN_STOP = 5
ACTION_FIN_SLICE = slice(ACTION_FIN_START, ACTION_FIN_STOP)

# Fin order matches parts_registry.FIN_PRIM_NAMES and vehicle YAML fins_config:
# RightFin, LeftFin, FwdFin, AftFin.
FIN_RIGHT_IDX = 0
FIN_LEFT_IDX = 1
FIN_FWD_IDX = 2
FIN_AFT_IDX = 3
FIN_COUNT = 4
FIN_INDICES = tuple(range(FIN_COUNT))

PITCH_FIN_INDICES = (FIN_RIGHT_IDX, FIN_LEFT_IDX)
ROLL_FIN_INDICES = (FIN_FWD_IDX, FIN_AFT_IDX)
YAW_FIN_SIGNS = (+1.0, -1.0, +1.0, -1.0)

FIN_SHORT_LABELS = ("Fin_1", "Fin_2", "Fin_3", "Fin_4")
FIN_DISPLAY_NAMES = (
    "Fin_1 (right)",
    "Fin_2 (left)",
    "Fin_3 (forward)",
    "Fin_4 (aft)",
)
FIN_AXIS_LABELS = (
    "RightFin",
    "LeftFin",
    "FwdFin",
    "AftFin",
)

# ---------------------------------------------------------------------------
# Joint unit convention (IsaacLab / PhysX)
# ---------------------------------------------------------------------------
# IsaacLab uses DEGREES for both joint position targets and joint position readback:
#   - Write path: convert rad → deg, call set_joint_position_target(deg)
#   - Read path:  read joint_pos (deg), convert deg → rad unconditionally
# Do NOT use a heuristic threshold (e.g. max_abs > 3.5) to detect units —
# it misclassifies small deflections (< 3.5 deg) as radians and fails silently.
# USD angular limits and drive targets are also authored in degrees.
# Ref: specs/006-refactor-fin-physics/research.md RQ-1

# ---------------------------------------------------------------------------
# Sign convention (eliminated by USD hinge axis fix)
# ---------------------------------------------------------------------------
# FIN_JOINT_VISUAL_SIGN was (-1,-1,-1,-1) — a runtime sign hack to compensate
# for hinge axes being authored in the wrong direction.  As of feature 006,
# the USD joint frames are corrected at authoring time (postprocess_usd.py
# applies a 180° localRot flip to all four fin joints so positive drive target
# produces positive deflection per controller convention).  FinMapping is now
# identity; no runtime sign correction is needed.
# Ref: specs/006-refactor-fin-physics/research.md RQ-2

# Fin joint drive constants are authored in the USD postprocessor and mirrored
# by the runtime actuator config.
FIN_DRIVE_STIFFNESS = 20.0
FIN_DRIVE_DAMPING = 1.0
FIN_DRIVE_EFFORT_LIMIT = 2.0

# ---------------------------------------------------------------------------
# Observation layout
# ---------------------------------------------------------------------------
OBS_POS_ERROR = slice(0, 3)
OBS_VEL_BODY = slice(3, 6)
OBS_GRAVITY_BODY = slice(6, 9)
OBS_OMEGA = slice(9, 12)
OBS_TWR = 12
OBS_WIND_EMA = slice(13, 16)
OBS_H_AGL = 16
OBS_SPEED = 17
OBS_ANG_SPEED = 18
OBS_TIME_FRAC = 19
OBS_DIM = 20

OBS_OMEGA_X = OBS_OMEGA.start
OBS_OMEGA_Y = OBS_OMEGA.start + 1
OBS_OMEGA_Z = OBS_OMEGA.start + 2
OBS_WIND_X = OBS_WIND_EMA.start
OBS_WIND_Y = OBS_WIND_EMA.start + 1
OBS_WIND_Z = OBS_WIND_EMA.start + 2

FRD_BODY_FRAME_TEXT = "FRD (+X=fwd/nose, +Y=right, +Z=down)"


def fin_axis_command(axis: str, magnitude: float) -> tuple[float, float, float, float]:
    """Return common-mode fin commands for the requested body axis."""
    mapping = default_fin_mapping()
    mag = float(magnitude)
    if axis == "pitch":
        return tuple(float(w * mag) for w in mapping.pitch_weights)  # type: ignore[return-value]
    if axis == "roll":
        return tuple(float(w * mag) for w in mapping.roll_weights)  # type: ignore[return-value]
    if axis == "yaw":
        return yaw_fin_command(mag)
    raise ValueError(f"Unsupported fin axis {axis!r}")


def yaw_fin_command(magnitude: float) -> tuple[float, float, float, float]:
    """Return the differential yaw fin pattern using the canonical fin order."""
    mapping = default_fin_mapping()
    mag = float(magnitude)
    return tuple(float(w * mag) for w in mapping.yaw_weights)  # type: ignore[return-value]

