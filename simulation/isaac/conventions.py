"""Shared Isaac Sim control and observation conventions.

This module centralizes the action layout, fin ordering, observation indices,
and a few Isaac-only actuator constants so scripts and runtime code do not
redefine them independently.
"""

from __future__ import annotations

from simulation.isaac.usd.parts_registry import FIN_PRIM_NAMES, frd_to_zup, zup_to_frd

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

# FwdFin/AftFin are mounted with opposite positive rotation sign relative to the
# control/aero convention used by the task and diagnostics.
FIN_JOINT_VISUAL_SIGN = (1.0, 1.0, -1.0, -1.0)

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
    mag = float(magnitude)
    if axis == "pitch":
        return (mag, mag, 0.0, 0.0)
    if axis == "roll":
        return (0.0, 0.0, mag, mag)
    if axis == "yaw":
        return yaw_fin_command(mag)
    raise ValueError(f"Unsupported fin axis {axis!r}")


def yaw_fin_command(magnitude: float) -> tuple[float, float, float, float]:
    """Return the differential yaw fin pattern using the canonical fin order."""
    mag = float(magnitude)
    return tuple(sign * mag for sign in YAW_FIN_SIGNS)  # type: ignore[return-value]

