"""
Helpers for adapting hover-centered controller actions to Isaac thrust commands.
"""

from __future__ import annotations

import numpy as np


def map_hover_centered_thrust_to_isaac(
    thrust_action: float,
    *,
    hover_thrust_frac: float,
) -> float:
    """Map a hover-centered thrust action in [-1, 1] to Isaac's [0, 1] command.

    The classical PID controller used in this repo assumes:
    - `0.0` means hover thrust
    - `-1.0` means minimum thrust
    - `+1.0` means maximum thrust

    The Isaac task instead interprets thrust as:
    - `0.0` means zero thrust
    - `1.0` means max thrust

    This adapter preserves the controller contract while targeting Isaac:
    - `-1.0 -> 0.0`
    - ` 0.0 -> hover_thrust_frac`
    - `+1.0 -> 1.0`
    """

    u = float(np.clip(thrust_action, -1.0, 1.0))
    hover = float(np.clip(hover_thrust_frac, 0.0, 1.0))
    if u >= 0.0:
        return float(hover + u * (1.0 - hover))
    return float(hover * (1.0 + u))


def map_pid_action_to_isaac(
    action: np.ndarray,
    *,
    hover_thrust_frac: float,
) -> np.ndarray:
    """Map a full PID action vector to Isaac's action convention."""

    mapped = np.asarray(action, dtype=np.float32).copy()
    if mapped.shape != (5,):
        mapped = mapped.reshape(5)
    mapped[0] = map_hover_centered_thrust_to_isaac(
        float(mapped[0]),
        hover_thrust_frac=hover_thrust_frac,
    )
    return mapped.astype(np.float32, copy=False)


__all__ = [
    "map_hover_centered_thrust_to_isaac",
    "map_pid_action_to_isaac",
]
