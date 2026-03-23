"""
Physical constants and enumerations for the TVC environment.
"""

import torch
from enum import IntEnum


# Physical constants
GRAVITY: float = 9.81  # m/s², standard gravity
AIR_DENSITY: float = 1.225  # kg/m³, ISA sea level

# Gravity vector in Isaac world frame (y-up convention)
GRAVITY_VEC_ISAAC = torch.tensor([0.0, -GRAVITY, 0.0], dtype=torch.float32)

# Gravity vector in body-FRD frame (z-down = positive gravity)
GRAVITY_VEC_FRD = torch.tensor([0.0, 0.0, GRAVITY], dtype=torch.float32)


class ContactState(IntEnum):
    """4-state contact state machine per data-model ContactStateMachine."""
    AIRBORNE = 0
    GROUND_CONTACT_CANDIDATE = 1
    LANDED = 2
    CRASHED = 3


class DispatchMode(IntEnum):
    """Force dispatch mode: per-fin COP forces or collapsed body wrench."""
    PER_LINK_FORCE = 0
    COLLAPSED_BODY_WRENCH = 1
