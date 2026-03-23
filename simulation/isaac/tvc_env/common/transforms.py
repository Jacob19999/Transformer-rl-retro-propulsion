"""
Rotation and translation utilities for the TVC environment.

All operations use (w,x,y,z) quaternion convention per Isaac Lab 2.3.2.
"""

import torch
from torch import Tensor
from tvc_env.common.quaternions import rotate_vector, multiply, inverse, normalize


def quat_apply(q: Tensor, v: Tensor) -> Tensor:
    """Apply quaternion rotation to vector(s). Alias for rotate_vector.

    Args:
        q: Quaternion(s) of shape (..., 4) in (w,x,y,z).
        v: Vector(s) of shape (..., 3).

    Returns:
        Rotated vector(s) of shape (..., 3).
    """
    return rotate_vector(q, v)


def quat_conjugate(q: Tensor) -> Tensor:
    """Return quaternion conjugate (same as inverse for unit quaternions).

    Args:
        q: Tensor of shape (..., 4) in (w,x,y,z).

    Returns:
        Conjugate of shape (..., 4).
    """
    conj = q.clone()
    conj[..., 1:] = -conj[..., 1:]
    return conj


def transform_points(points: Tensor, q: Tensor, t: Tensor) -> Tensor:
    """Transform points from local frame to world frame.

    Applies: p_world = R(q) * p_local + t

    Args:
        points: Points of shape (..., 3) in local frame.
        q: Quaternion(s) of shape (..., 4) in (w,x,y,z).
        t: Translation(s) of shape (..., 3).

    Returns:
        Points of shape (..., 3) in world frame.
    """
    return quat_apply(q, points) + t


def local_to_world(v: Tensor, q: Tensor) -> Tensor:
    """Rotate vector from local (body) frame to world frame using quaternion.

    Args:
        v: Vector(s) of shape (..., 3) in local frame.
        q: Quaternion(s) of shape (..., 4) in (w,x,y,z) representing local→world rotation.

    Returns:
        Vector(s) of shape (..., 3) in world frame.
    """
    return rotate_vector(q, v)


def world_to_local(v: Tensor, q: Tensor) -> Tensor:
    """Rotate vector from world frame to local (body) frame using quaternion.

    Args:
        v: Vector(s) of shape (..., 3) in world frame.
        q: Quaternion(s) of shape (..., 4) in (w,x,y,z) representing local→world rotation.

    Returns:
        Vector(s) of shape (..., 3) in local frame.
    """
    q_inv = quat_conjugate(normalize(q))
    return rotate_vector(q_inv, v)


def compute_heading(q: Tensor) -> Tensor:
    """Compute heading angle (yaw) from quaternion.

    Args:
        q: Quaternion(s) of shape (..., 4) in (w,x,y,z).

    Returns:
        Heading angle(s) in radians, shape (...,).
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def axis_angle_to_quat(axis: Tensor, angle: Tensor) -> Tensor:
    """Convert axis-angle representation to quaternion.

    Args:
        axis: Unit vector(s) of shape (..., 3).
        angle: Rotation angle(s) in radians, shape (...,).

    Returns:
        Quaternion(s) of shape (..., 4) in (w,x,y,z).
    """
    half_angle = angle * 0.5
    w = torch.cos(half_angle)
    xyz = axis * torch.sin(half_angle).unsqueeze(-1)
    return torch.cat([w.unsqueeze(-1), xyz], dim=-1)
