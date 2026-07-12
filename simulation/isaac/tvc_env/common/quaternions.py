"""
Quaternion math using (w,x,y,z) convention — as used by Isaac Lab 2.3.2.

All quaternions in this codebase use (w,x,y,z) ordering internally.
Convention converters are provided for the single boundary with body-frame (xyzw) code.

IMPORTANT: Isaac Lab 2.3.2 uses (w,x,y,z). Isaac Lab 3.0 switches to (x,y,z,w).
Do not migrate convention without updating this module and all callers.
"""

import torch
from torch import Tensor
import math


def identity(num: int = 1, device: torch.device = None, dtype: torch.dtype = torch.float32) -> Tensor:
    """Return identity quaternion(s) [w=1, x=0, y=0, z=0].

    Args:
        num: Number of quaternions.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tensor of shape (num, 4) if num > 1, else (4,).
    """
    q = torch.zeros(num, 4, device=device, dtype=dtype)
    q[:, 0] = 1.0
    return q.squeeze(0) if num == 1 else q


def normalize(q: Tensor) -> Tensor:
    """Normalize quaternion(s) to unit length.

    Args:
        q: Tensor of shape (..., 4).

    Returns:
        Normalized quaternion(s) of same shape.
    """
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """Hamilton product of two quaternions q1 * q2.

    Args:
        q1: Tensor of shape (..., 4) in (w,x,y,z).
        q2: Tensor of shape (..., 4) in (w,x,y,z).

    Returns:
        Tensor of shape (..., 4) in (w,x,y,z).
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def inverse(q: Tensor) -> Tensor:
    """Quaternion inverse (conjugate for unit quaternions).

    Args:
        q: Tensor of shape (..., 4) in (w,x,y,z).

    Returns:
        Conjugate quaternion of same shape.
    """
    conj = q.clone()
    conj[..., 1:] = -conj[..., 1:]
    return conj


def rotate_vector(q: Tensor, v: Tensor) -> Tensor:
    """Rotate vector(s) v by quaternion(s) q.

    Computes: q * [0,v] * q^-1 (pure quaternion sandwich product).

    Args:
        q: Tensor of shape (..., 4) in (w,x,y,z).
        v: Tensor of shape (..., 3).

    Returns:
        Rotated vector(s) of shape (..., 3).
    """
    # Efficient Rodrigues' rotation formula
    # v' = v + 2*w*(t) + 2*(xyz × t)  where t = xyz × v
    xyz = q[..., 1:]
    w = q[..., 0:1]
    t = 2.0 * torch.linalg.cross(xyz, v)
    return v + w * t + torch.linalg.cross(xyz, t)


def to_rotation_matrix(q: Tensor) -> Tensor:
    """Convert quaternion(s) to 3x3 rotation matrix.

    Args:
        q: Tensor of shape (..., 4) in (w,x,y,z).

    Returns:
        Rotation matrix of shape (..., 3, 3).
    """
    q = normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.stack([
        1 - 2*(y2 + z2),   2*(xy - wz),    2*(xz + wy),
        2*(xy + wz),        1 - 2*(x2 + z2), 2*(yz - wx),
        2*(xz - wy),        2*(yz + wx),    1 - 2*(x2 + y2),
    ], dim=-1)
    return R.reshape(q.shape[:-1] + (3, 3))


def from_euler(roll: Tensor, pitch: Tensor, yaw: Tensor) -> Tensor:
    """Create quaternion from Euler angles (intrinsic ZYX / extrinsic XYZ).

    Args:
        roll:  Rotation around x-axis (rad), shape (...,).
        pitch: Rotation around y-axis (rad), shape (...,).
        yaw:   Rotation around z-axis (rad), shape (...,).

    Returns:
        Quaternion of shape (..., 4) in (w,x,y,z).
    """
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


def to_euler(q: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Convert quaternion to Euler angles (roll, pitch, yaw) in ZYX convention.

    Args:
        q: Tensor of shape (..., 4) in (w,x,y,z).

    Returns:
        Tuple (roll, pitch, yaw) each of shape (...,) in radians.
    """
    q = normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    sinp = sinp.clamp(-1.0, 1.0)
    pitch = torch.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def tilt_angle(q: Tensor) -> Tensor:
    """Return yaw-invariant body tilt from the world vertical in radians.

    Isaac quaternions rotate local axes into world axes. The dot product of
    local +Z with world +Z is the (2, 2) rotation-matrix element.
    """
    q = normalize(q)
    x, y = q[..., 1], q[..., 2]
    cos_tilt = 1.0 - 2.0 * (x * x + y * y)
    return torch.acos(cos_tilt.clamp(-1.0, 1.0))


# Convention converters — the SINGLE boundary between wxyz and xyzw
def isaac_wxyz_to_xyzw(q: Tensor) -> Tensor:
    """Convert from Isaac Lab 2.3.2 (w,x,y,z) to body-frame (x,y,z,w) ordering.

    Args:
        q: Tensor of shape (..., 4) in (w,x,y,z).

    Returns:
        Tensor of shape (..., 4) in (x,y,z,w).
    """
    return torch.cat([q[..., 1:], q[..., :1]], dim=-1)


def xyzw_to_isaac_wxyz(q: Tensor) -> Tensor:
    """Convert from body-frame (x,y,z,w) to Isaac Lab 2.3.2 (w,x,y,z) ordering.

    Args:
        q: Tensor of shape (..., 4) in (x,y,z,w).

    Returns:
        Tensor of shape (..., 4) in (w,x,y,z).
    """
    return torch.cat([q[..., -1:], q[..., :-1]], dim=-1)
