"""
Quaternion utilities for the simulation dynamics.

Conventions
-----------
- Frames: inertial NED (North-East-Down), body FRD (Forward-Right-Down).
- Quaternions are scalar-first: [q0, q1, q2, q3].
- Hamilton product convention for composition.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _as_quat_array(q: Iterable[float]) -> np.ndarray:
    """Convert input to a float64 numpy array of shape (4,)."""
    q_arr = np.asarray(q, dtype=float)
    if q_arr.shape != (4,):
        raise ValueError(f"Quaternion must have shape (4,), got {q_arr.shape}.")
    return q_arr


def quat_to_dcm(q: Iterable[float]) -> np.ndarray:
    """Convert quaternion to a 3x3 direction cosine matrix (body → inertial).

    Parameters
    ----------
    q:
        Quaternion in scalar-first form [q0, q1, q2, q3].

    Returns
    -------
    np.ndarray
        3x3 rotation matrix R such that v_inertial = R @ v_body.
    """
    q0, q1, q2, q3 = _as_quat_array(q)

    q0q0 = q0 * q0
    q1q1 = q1 * q1
    q2q2 = q2 * q2
    q3q3 = q3 * q3

    q0q1 = q0 * q1
    q0q2 = q0 * q2
    q0q3 = q0 * q3
    q1q2 = q1 * q2
    q1q3 = q1 * q3
    q2q3 = q2 * q3

    R = np.empty((3, 3), dtype=float)

    # vehicle.md §2.3, body → inertial DCM
    R[0, 0] = q0q0 + q1q1 - q2q2 - q3q3
    R[0, 1] = 2.0 * (q1q2 - q0q3)
    R[0, 2] = 2.0 * (q1q3 + q0q2)

    R[1, 0] = 2.0 * (q1q2 + q0q3)
    R[1, 1] = q0q0 - q1q1 + q2q2 - q3q3
    R[1, 2] = 2.0 * (q2q3 - q0q1)

    R[2, 0] = 2.0 * (q1q3 - q0q2)
    R[2, 1] = 2.0 * (q2q3 + q0q1)
    R[2, 2] = q0q0 - q1q1 - q2q2 + q3q3

    return R


def quat_mult(q1: Iterable[float], q2: Iterable[float]) -> np.ndarray:
    """Hamilton product of two quaternions.

    The result represents applying rotation q2 followed by q1.

    Parameters
    ----------
    q1, q2:
        Quaternions in scalar-first form [q0, q1, q2, q3].

    Returns
    -------
    np.ndarray
        Quaternion q = q1 ⊗ q2 in scalar-first form.
    """
    w1, x1, y1, z1 = _as_quat_array(q1)
    w2, x2, y2, z2 = _as_quat_array(q2)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z], dtype=float)


def quat_normalize(q: Iterable[float], eps: float = 1e-12) -> np.ndarray:
    """Return a unit-norm version of quaternion q.

    Parameters
    ----------
    q:
        Quaternion in scalar-first form [q0, q1, q2, q3].
    eps:
        Small threshold below which the norm is considered zero.

    Returns
    -------
    np.ndarray
        Normalized quaternion with unit norm.

    Raises
    ------
    ValueError
        If the quaternion norm is below `eps`.
    """
    q_arr = _as_quat_array(q)
    norm = np.linalg.norm(q_arr)
    if norm < eps:
        raise ValueError("Cannot normalize quaternion with near-zero norm.")
    return q_arr / norm


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles (roll, pitch, yaw) to a quaternion.

    Convention
    ----------
    - Intrinsic rotations applied in yaw (z), pitch (y), roll (x) order.
    - Right-handed rotations, angles in radians.

    Parameters
    ----------
    roll:
        Rotation about body x-axis (rad).
    pitch:
        Rotation about body y-axis (rad).
    yaw:
        Rotation about body z-axis (rad).

    Returns
    -------
    np.ndarray
        Quaternion [q0, q1, q2, q3] (scalar-first).
    """
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    q0 = cr * cp * cy + sr * sp * sy
    q1 = sr * cp * cy - cr * sp * sy
    q2 = cr * sp * cy + sr * cp * sy
    q3 = cr * cp * sy - sr * sp * cy

    return quat_normalize([q0, q1, q2, q3])


def quat_to_euler(q: Iterable[float]) -> tuple[float, float, float]:
    """Convert quaternion to Euler angles (roll, pitch, yaw).

    Uses the inverse of :func:`euler_to_quat` convention.

    Parameters
    ----------
    q:
        Quaternion in scalar-first form [q0, q1, q2, q3].

    Returns
    -------
    (roll, pitch, yaw):
        Tuple of Euler angles in radians.
    """
    q0, q1, q2, q3 = quat_normalize(q)

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (q0 * q2 - q3 * q1)
    if abs(sinp) >= 1.0:
        pitch = np.pi / 2.0 * np.sign(sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return float(roll), float(pitch), float(yaw)

