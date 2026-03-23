"""
FRD ↔ Isaac frame conversion — SINGLE canonical conversion boundary.

Body-FRD convention: x=forward, y=right, z=down
Isaac/world convention: x=right, y=up, z=back (NED-like, per Isaac Sim default)

All frame conversions in the entire codebase MUST pass through this module.
No other module may implement frame transformations.
"""

import torch
from torch import Tensor


# Rotation matrix: body-FRD → Isaac world frame
# FRD (x_fwd, y_right, z_down) → Isaac (x_right, y_up, z_back)
# x_isaac =  y_frd   (right)
# y_isaac = -z_frd   (up = -down)
# z_isaac = -x_frd   (back = -forward)
_R_FRD_TO_ISAAC = torch.tensor([
    [0.0,  1.0,  0.0],
    [0.0,  0.0, -1.0],
    [-1.0, 0.0,  0.0],
], dtype=torch.float64)

_R_ISAAC_TO_FRD = _R_FRD_TO_ISAAC.T  # Orthogonal, so inverse = transpose


def frd_to_isaac(v: Tensor) -> Tensor:
    """Convert a 3-vector or batch of 3-vectors from body-FRD to Isaac world frame.

    Args:
        v: Tensor of shape (..., 3) in body-FRD frame.

    Returns:
        Tensor of shape (..., 3) in Isaac world frame.
    """
    R = _R_FRD_TO_ISAAC.to(v.device, v.dtype)
    return (R @ v.unsqueeze(-1)).squeeze(-1)


def isaac_to_frd(v: Tensor) -> Tensor:
    """Convert a 3-vector or batch of 3-vectors from Isaac world frame to body-FRD.

    Args:
        v: Tensor of shape (..., 3) in Isaac world frame.

    Returns:
        Tensor of shape (..., 3) in body-FRD frame.
    """
    R = _R_ISAAC_TO_FRD.to(v.device, v.dtype)
    return (R @ v.unsqueeze(-1)).squeeze(-1)


def frd_position_to_isaac(pos: Tensor) -> Tensor:
    """Convert position vector from body-FRD to Isaac frame (pure rotation, no translation)."""
    return frd_to_isaac(pos)


def isaac_position_to_frd(pos: Tensor) -> Tensor:
    """Convert position vector from Isaac frame to body-FRD (pure rotation, no translation)."""
    return isaac_to_frd(pos)


def frd_velocity_to_isaac(vel: Tensor) -> Tensor:
    """Convert velocity vector from body-FRD to Isaac frame."""
    return frd_to_isaac(vel)


def isaac_velocity_to_frd(vel: Tensor) -> Tensor:
    """Convert velocity vector from Isaac frame to body-FRD."""
    return isaac_to_frd(vel)


def frd_force_to_isaac(force: Tensor) -> Tensor:
    """Convert force vector from body-FRD to Isaac frame."""
    return frd_to_isaac(force)


def isaac_force_to_frd(force: Tensor) -> Tensor:
    """Convert force vector from Isaac frame to body-FRD."""
    return isaac_to_frd(force)


def get_frd_to_isaac_matrix(device: torch.device = None, dtype: torch.dtype = torch.float32) -> Tensor:
    """Return the 3x3 rotation matrix from body-FRD to Isaac world frame."""
    return _R_FRD_TO_ISAAC.to(device=device, dtype=dtype)


def get_isaac_to_frd_matrix(device: torch.device = None, dtype: torch.dtype = torch.float32) -> Tensor:
    """Return the 3x3 rotation matrix from Isaac world frame to body-FRD."""
    return _R_ISAAC_TO_FRD.to(device=device, dtype=dtype)
