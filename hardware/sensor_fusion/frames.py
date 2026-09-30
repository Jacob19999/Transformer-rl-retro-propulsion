"""Frames and rotation helpers shared by the calibration, the EKF and the simulator.

    vehicle body  FRD  x forward, y right, z down            navigation  NED (yaw origin arbitrary: no compass)
    WTGAHRS1      X right, Y forward, Z up (datasheet 3.3), its onboard attitude is Z-Y-X Euler in an ENU frame

``R_vs`` maps sensor-frame vectors into the vehicle frame: ``v_vehicle = R_vs @ v_sensor``. A mount is fully described
by which sensor axis is the vehicle's forward, right and down axis, so there are 24 axis-aligned mounts.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

T_EN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])  # NED <-> ENU (its own inverse)
AXES = "XYZ"


def skew(v) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(R) -> np.ndarray:
    m = np.asarray(R)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def quat_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def rotvec_to_quat(v) -> np.ndarray:
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.array([1.0, 0.5 * v[0], 0.5 * v[1], 0.5 * v[2]])
    s = math.sin(th / 2) / th
    return np.array([math.cos(th / 2), v[0] * s, v[1] * s, v[2] * s])


def euler_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Z-Y-X: R = Rz(yaw) Ry(pitch) Rx(roll), radians (aerospace and the WTGAHRS1 datasheet convention)."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def matrix_to_euler(R) -> tuple[float, float, float]:
    """(roll, pitch, yaw) radians of a Z-Y-X rotation matrix."""
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    return math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0])


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def vee(S) -> np.ndarray:
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


# ---- mounts ---------------------------------------------------------------------------------------
def _mount_key(f, r, d) -> str:
    return "".join(f"{'+' if s > 0 else '-'}{AXES[i]}" for i, s in (f, r, d))


def _build_mounts() -> dict[str, np.ndarray]:
    out = {}
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for row, (ax, s) in enumerate(zip(perm, signs)):
                R[row, ax] = s
            if round(np.linalg.det(R)) == 1:
                out[_mount_key(*[(perm[i], signs[i]) for i in range(3)])] = R
    return out


MOUNTS: dict[str, np.ndarray] = _build_mounts()  # key "<F><R><D>", e.g. "+Y+X-Z" = forward +Y, right +X, down -Z
DEFAULT_MOUNT = "+Y+X-Z"                          # WTGAHRS1 flat, printed Y arrow forward, X right, Z up
assert len(MOUNTS) == 24


def mount_label(key: str) -> str:
    f, r, d = key[0:2], key[2:4], key[4:6]
    note = ""
    if key == DEFAULT_MOUNT:
        note = "  (flat, printed Y arrow forward)"
    elif key == "+X+Y+Z":
        note = "  (sensor axes = FRD, ArduPilot 'None')"
    return f"forward {f}, right {r}, down {d}{note}"


def nearest_mount(R_est) -> tuple[str, float]:
    """The axis-aligned mount closest to ``R_est`` and the residual angle in degrees."""
    best, best_tr = DEFAULT_MOUNT, -9.0
    for key, R in MOUNTS.items():
        tr = float(np.trace(R.T @ R_est))
        if tr > best_tr:
            best, best_tr = key, tr
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (best_tr - 1.0) / 2.0))))
    return best, ang


def sensor_attitude_enu(R_nb, R_vs) -> np.ndarray:
    """Rotation of the sensor body (its own axes) into the ENU frame: what the WTGAHRS1 reports as its attitude."""
    return T_EN @ R_nb @ R_vs
