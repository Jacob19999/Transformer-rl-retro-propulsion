"""
Isaac-facing quaternion helpers (wxyz).

This module centralizes all runtime quaternion operations that touch IsaacLab
articulations so that the convention is defined in one place.

Convention
----------
- IsaacLab uses scalar-first wxyz ordering: [qw, qx, qy, qz].
- These helpers assume quaternions are already normalized.
"""

from __future__ import annotations

import torch


def identity_quat_wxyz(n: int = 1, *, device: torch.device | None = None) -> torch.Tensor:
    """Return an (n, 4) tensor of identity quaternions in wxyz order."""
    q = torch.zeros(n, 4, device=device)
    q[:, 0] = 1.0  # qw = 1
    return q


def rotate_world_to_body_wxyz(v_world: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vectors into body frame using IsaacLab wxyz quaternions."""
    qw = quat_w[:, 0:1]
    q_vec = -quat_w[:, 1:4]  # conjugate imaginary part for world -> body

    t = 2.0 * torch.linalg.cross(q_vec, v_world)
    return v_world + qw * t + torch.linalg.cross(q_vec, t)


def rotate_body_to_world_wxyz(quat_w: torch.Tensor, v_body: torch.Tensor) -> torch.Tensor:
    """Rotate body-frame vectors into world frame using IsaacLab wxyz quaternions."""
    qw = quat_w[:, 0:1]
    q_vec = quat_w[:, 1:4]

    t = 2.0 * torch.linalg.cross(q_vec, v_body)
    return v_body + qw * t + torch.linalg.cross(q_vec, t)


__all__ = [
    "identity_quat_wxyz",
    "rotate_world_to_body_wxyz",
    "rotate_body_to_world_wxyz",
]

