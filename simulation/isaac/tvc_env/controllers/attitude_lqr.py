"""Gyroscopic attitude regulator: discrete LQR on the vehicle and vane-actuator model.

The EDF rotor's angular momentum H = I_rotor omega dominates roll/pitch:
I p' = K f^2 j_x - H q - c p and I q' = K f^2 j_y + H p - c q, with vane
authority K (torque per rad of effort at full rotor speed), rotor fraction f
and body damping c. A torque about one axis mostly precesses the body about
the other, and the unforced motion is a nutation at H / I (~2.5 Hz at hover),
right where the vane actuators lag: servo s' = (u - s) / tau_s, then the vane
joint j' = (s - j) / tau_j (~0.075 s in total). A PD law on the attitude
error, however tuned, pushes the rate loop past 90 deg of lag at the nutation
frequency. On momentum-bounded vanes every PD gain set that tracks attitude
is linearly unstable (2026-09-26 analysis: the mission gains grew a ~4 Hz
nutation at +1.1 to +2.8 /s and a ~1.6 Hz mode at +0.6 to +1.9 /s; best
retune: -1.0 /s with a 10x slower attitude loop).

This regulator feeds back the whole state: the attitude-error integral,
attitude error, body rates, and the servo and joint angles in effort space
(both measurable: the joint is the vane angle, and the servo is
j + tau_j j'). The actuator states give the phase lead a PD law lacks, and the
gain matrix rotates its torque toward the precession direction. Gains come
from the discrete algebraic Riccati equation for the zero-order-hold model at
the control period, scheduled on rotor fraction, which scales both K f^2 and H.

Yaw is a separate channel with the same actuator model,
I_zz r' = K_yaw f^2 j_yaw - c r, and an integral of the yaw rate: the jet's
residual swirl is a steady yaw torque, which rate feedback alone leaves as a
steady spin (the body then rotates the roll/pitch trim integral with it).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import expm, solve_discrete_are


@dataclass(frozen=True)
class AttitudePlant:
    """What the flight computer knows about the attitude dynamics (identified hardware)."""

    inertia_rp_kg_m2: float          # transverse (roll/pitch) moment of inertia
    inertia_yaw_kg_m2: float
    rp_authority_nm_per_rad: float   # vane torque per rad of effort at full rotor speed
    yaw_authority_nm_per_rad: float
    rotor_momentum_max_nms: float    # I_rotor omega_max
    servo_lag_s: float               # servo first-order time constant
    joint_lag_s: float               # vane linkage/joint lag behind the servo
    damping_nms_per_rad: float = 0.0


@dataclass(frozen=True)
class LQRWeights:
    """Quadratic weights: attitude error per rad^2, rates per (rad/s)^2, integrals per (rad s)^2.

    The input is weighted by the vane torque it makes at full rotor speed,
    per (N m)^2 (effort^2 x authority^2), so one set of weights gives the
    same torque loop on any vanes: weighting the effort itself made the
    design 72x more aggressive on the legacy vanes (8.5x the authority).
    """

    error: float
    rate: float
    integral: float
    torque: float
    yaw_rate: float
    yaw_torque: float
    yaw_integral: float = 0.0


def _zoh(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A * dt, B * dt
    E = expm(M)
    return E[:n, :n], E[:n, n:]


def _lqr(Ad, Bd, Q, R) -> np.ndarray:
    P = solve_discrete_are(Ad, Bd, Q, R)
    return -np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)


class GyroAttitudeLQR:
    """Scheduled full-state attitude regulator; efforts are in the mixer's roll/pitch/yaw units (rad)."""

    # Roll/pitch state: [z_x, z_y, e_x, e_y, p, q, s_x, s_y, j_x, j_y] (z: integral of e).
    def __init__(self, plant: AttitudePlant, weights: LQRWeights, dt: float,
                 fractions=tuple(np.round(np.arange(0.30, 1.0001, 0.05), 2))):
        if plant.inertia_rp_kg_m2 <= 0.0 or plant.rp_authority_nm_per_rad <= 0.0 or plant.servo_lag_s <= 0.0:
            raise ValueError('The attitude LQR needs positive inertia, vane authority and servo lag')
        self.plant, self.weights, self.dt = plant, weights, float(dt)
        self.fractions = np.asarray(fractions, dtype=float)
        self._rp = np.stack([self._design_rp(f) for f in self.fractions])
        self._yaw = np.stack([self._design_yaw(f) for f in self.fractions])

    # ---- models ----

    def _actuator(self, A, B, servo, joint, column):
        p = self.plant
        A[servo, servo] = -1.0 / p.servo_lag_s
        B[servo, column] = 1.0 / p.servo_lag_s
        if p.joint_lag_s > 0.0:
            A[joint, servo] = 1.0 / p.joint_lag_s
            A[joint, joint] = -1.0 / p.joint_lag_s

    @property
    def rigid(self) -> bool:
        """No joint lag: the vane is the servo, and its joint state drops out of the model."""
        return self.plant.joint_lag_s <= 0.0

    def roll_pitch_model(self, fraction: float) -> tuple[np.ndarray, np.ndarray]:
        """Continuous (A, B) of the roll/pitch state at this rotor fraction."""
        p = self.plant
        f = float(fraction)
        H, k, I, c = p.rotor_momentum_max_nms * f, p.rp_authority_nm_per_rad * f * f, p.inertia_rp_kg_m2, p.damping_nms_per_rad
        A, B = np.zeros((10, 10)), np.zeros((10, 2))
        A[0, 2] = A[1, 3] = 1.0                       # z' = e
        A[2, 4] = A[3, 5] = 1.0                       # e' = w
        A[4, 5], A[5, 4] = -H / I, H / I              # gyroscopic coupling
        A[4, 4] = A[5, 5] = -c / I
        vane = (6, 7) if self.rigid else (8, 9)       # vane torque from the joint (or servo) angle
        A[4, vane[0]] = A[5, vane[1]] = k / I
        self._actuator(A, B, 6, 8, 0)
        self._actuator(A, B, 7, 9, 1)
        return A, B

    def yaw_model(self, fraction: float) -> tuple[np.ndarray, np.ndarray]:
        """Continuous (A, B) of the yaw state [z_r, r, s, j] (z_r: integral of the yaw rate)."""
        p = self.plant
        f = float(fraction)
        A, B = np.zeros((4, 4)), np.zeros((4, 1))
        A[0, 1] = 1.0
        A[1, 1] = -p.damping_nms_per_rad / p.inertia_yaw_kg_m2
        A[1, 2 if self.rigid else 3] = p.yaw_authority_nm_per_rad * f * f / p.inertia_yaw_kg_m2
        self._actuator(A, B, 2, 3, 0)
        return A, B

    def _kept(self, size, integral_states, joint_states, integral_weight):
        """State indices the design keeps: an unweighted integrator (it sits on the
        unit circle, which the Riccati solve rejects) and a rigid joint drop out."""
        drop = set(joint_states) if self.rigid else set()
        if integral_weight <= 0.0:
            drop |= set(integral_states)
        return [i for i in range(size) if i not in drop]

    def _design_rp(self, fraction):
        w = self.weights
        Ad, Bd = _zoh(*self.roll_pitch_model(fraction), self.dt)
        Q = np.diag([w.integral] * 2 + [w.error] * 2 + [w.rate] * 2 + [0.0] * 4)
        keep = self._kept(10, (0, 1), (8, 9), w.integral)
        gain = np.zeros((2, 10))
        R = w.torque * self.plant.rp_authority_nm_per_rad ** 2 * np.eye(2)
        gain[:, keep] = _lqr(Ad[np.ix_(keep, keep)], Bd[keep], Q[np.ix_(keep, keep)], R)
        return gain

    def _design_yaw(self, fraction):
        w = self.weights
        gain = np.zeros((1, 4))
        if self.plant.yaw_authority_nm_per_rad <= 0.0 or self.plant.inertia_yaw_kg_m2 <= 0.0:
            return gain
        Ad, Bd = _zoh(*self.yaw_model(fraction), self.dt)
        Q = np.diag([w.yaw_integral, w.yaw_rate, 0.0, 0.0])
        R = np.array([[w.yaw_torque * self.plant.yaw_authority_nm_per_rad ** 2]])
        keep = self._kept(4, (0,), (3,), w.yaw_integral)
        gain[:, keep] = _lqr(Ad[np.ix_(keep, keep)], Bd[keep], Q[np.ix_(keep, keep)], R)
        return gain

    def _gain(self, table, fraction):
        f = min(max(float(fraction), self.fractions[0]), self.fractions[-1])
        return np.array([np.interp(f, self.fractions, table[:, i, j]) for i in range(table.shape[1])
                         for j in range(table.shape[2])]).reshape(table.shape[1:])

    # ---- control ----

    def roll_pitch_gain(self, fraction: float) -> np.ndarray:
        """(2, 10) gain on [z, e, w, s, j] at this rotor fraction."""
        return self._gain(self._rp, fraction)

    def yaw_gain(self, fraction: float) -> np.ndarray:
        """(1, 4) gain on [z_r, r, s, j] at this rotor fraction."""
        return self._gain(self._yaw, fraction)

    def modes(self, fraction: float, plant: AttitudePlant | None = None) -> list[tuple[float, float]]:
        """Closed-loop roll/pitch (decay 1/s, frequency Hz) at this rotor fraction, optionally on another plant."""
        other = self if plant is None else GyroAttitudeLQR.__new__(GyroAttitudeLQR)
        if plant is not None:
            other.plant, other.dt, other.weights = plant, self.dt, self.weights
        Ad, Bd = _zoh(*other.roll_pitch_model(fraction), self.dt)
        keep = other._kept(10, (0, 1), (8, 9), self.weights.integral)
        closed = Ad[np.ix_(keep, keep)] + Bd[keep] @ self.roll_pitch_gain(fraction)[:, keep]
        out = []
        for z in np.linalg.eigvals(closed):
            s = complex(np.log(complex(z))) / self.dt
            out.append((s.real, abs(s.imag) / (2.0 * math.pi)))
        return sorted(out, reverse=True)
