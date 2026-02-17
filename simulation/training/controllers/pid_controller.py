"""
PID controller baseline (Stage 17).

Implements a classical cascaded structure per training.md §7.3:
- Outer loop:
  - Altitude PID -> normalized thrust command a0 in [-1, 1]
  - Lateral PD -> desired roll/pitch angles (clamped)
- Inner loop:
  - Attitude PD -> fin deflections (mapped to fins 1..4, normalized to [-1, 1])

Frame conventions (consistent with the simulation codebase):
- Inertial frame: NED (z down)
- Body frame: FRD (z down)

Observation layout matches `simulation/training/observation.py` (20-dim).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from simulation.training.controllers.base import Controller


@dataclass(frozen=True, slots=True)
class _PidPhase:
    alt_lo: float
    alt_hi: float
    gain_scale: float


class PIDController(Controller):
    """Cascaded PID controller for the EDF landing task."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = dict(config)
        # Allow either a top-level {pid: {...}} or the inner dict itself.
        if "pid" in cfg and isinstance(cfg["pid"], Mapping):
            cfg = dict(cfg["pid"])

        self.outer: dict[str, Any] = dict(cfg["outer_loop"])
        self.inner: dict[str, Any] = dict(cfg["inner_loop"])

        self.dt: float = float(cfg.get("dt", 0.025))  # 40 Hz by default
        if self.dt <= 0.0:
            raise ValueError("PIDController dt must be > 0.")

        self.delta_max: float = float(cfg.get("delta_max", 0.26))  # rad (~15 deg)
        if self.delta_max <= 0.0:
            raise ValueError("PIDController delta_max must be > 0.")

        self.max_tilt: float = float(
            np.deg2rad(self.outer.get("max_tilt_cmd_deg", 20.0))
        )

        # Integral state for altitude loop (anti-windup via clamp).
        alt_cfg = dict(self.outer["altitude"])
        self.integral_limit: float = float(alt_cfg.get("integral_limit", 5.0))
        self.max_alt_error: float = float(alt_cfg.get("max_error", 2.0))
        self.alt_integral: float = 0.0
        self.prev_alt_error: float = 0.0

        # Thrust rate limiter: maximum change in normalised thrust action per
        # controller step.  Limits motor reaction torque that causes yaw spin-up.
        self.thrust_rate_limit: float = float(
            cfg.get("thrust_rate_limit", 0.15)
        )
        self.prev_thrust_action: float = 0.0
        self._thrust_delta: float = 0.0

        # Gain scheduling (optional, Stage 18 may enable this).
        sched = dict(cfg.get("gain_schedule", {}) or {})
        self._schedule_enabled: bool = bool(sched.get("enabled", False))
        phases_in = list((sched.get("phases", []) or []))
        phases: list[_PidPhase] = []
        for p in phases_in:
            if not isinstance(p, Mapping):
                continue
            alt_range = np.asarray(p.get("altitude_range", [0.0, 0.0]), dtype=float).reshape(
                -1
            )
            if alt_range.size != 2:
                continue
            alt_lo, alt_hi = float(alt_range[0]), float(alt_range[1])
            if alt_hi < alt_lo:
                alt_lo, alt_hi = alt_hi, alt_lo
            phases.append(
                _PidPhase(
                    alt_lo=alt_lo,
                    alt_hi=alt_hi,
                    gain_scale=float(p.get("gain_scale", 1.0)),
                )
            )
        self._phases: tuple[_PidPhase, ...] = tuple(phases)

    def reset(self) -> None:
        self.alt_integral = 0.0
        self.prev_alt_error = 0.0
        self.prev_thrust_action = 0.0
        self._thrust_delta = 0.0

    def _gain_scale(self, h_agl: float) -> float:
        if not self._schedule_enabled or not self._phases:
            return 1.0
        h = float(h_agl)
        for ph in self._phases:
            if ph.alt_lo <= h <= ph.alt_hi:
                return float(ph.gain_scale)
        return 1.0

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Map observation (20,) to normalized action in [-1,1]^5."""
        o = np.asarray(obs, dtype=float).reshape(-1)
        if o.size < 20:
            raise ValueError(f"obs must have at least 20 elements, got {o.size}.")

        # Unpack observation (training.md §3.2, observation.py docstring)
        target_body = o[0:3]
        v_b = o[3:6]
        g_body = o[6:9]
        omega = o[9:12]
        h_agl = float(o[16])

        s = self._gain_scale(h_agl)

        # --- Outer loop: altitude PID -> normalized throttle action a0 ---
        # h_agl is positive upwards; to descend, error is negative.
        # NOTE: altitude loop is NOT scaled by gain_schedule so that the
        # vehicle always retains full altitude control authority, even in the
        # terminal phase where inner-loop gains are reduced.
        Kz = dict(self.outer["altitude"])
        alt_target = float(Kz.get("target_h_agl", 0.0))
        alt_error = alt_target - h_agl

        # Clamp error to prevent thrust saturation at large altitude offsets.
        alt_error_clamped = float(
            np.clip(alt_error, -self.max_alt_error, self.max_alt_error)
        )

        self.alt_integral = float(
            np.clip(
                self.alt_integral + alt_error_clamped * self.dt,
                -self.integral_limit,
                self.integral_limit,
            )
        )

        # Use measured body vertical velocity for derivative term.
        # In FRD body frame v_b[2] > 0 = descending, which approximates
        # d(alt_error)/dt = -d(h_agl)/dt.  This avoids the large derivative
        # kick that occurs when differencing alt_error with prev_alt_error=0.
        alt_rate = float(v_b[2])
        self.prev_alt_error = float(alt_error)

        thrust_action = (
            float(Kz.get("Kp", 0.0)) * alt_error_clamped
            + float(Kz.get("Ki", 0.0)) * self.alt_integral
            + float(Kz.get("Kd", 0.0)) * alt_rate
        )
        thrust_action = float(np.clip(thrust_action, -1.0, 1.0))

        # Rate-limit thrust to reduce motor reaction torque (which causes
        # yaw spin-up on a single-fan vehicle with limited yaw authority).
        if self.thrust_rate_limit > 0:
            lo = self.prev_thrust_action - self.thrust_rate_limit
            hi = self.prev_thrust_action + self.thrust_rate_limit
            thrust_action = float(np.clip(thrust_action, lo, hi))
        # Record thrust change for motor-torque feedforward (used in yaw).
        self._thrust_delta = thrust_action - self.prev_thrust_action
        self.prev_thrust_action = thrust_action

        # --- Outer loop: lateral PD -> desired roll/pitch angles (rad) ---
        # target_body is R^T (p_target - p), expressed in body FRD.
        # For a target forward (+x body), we want negative pitch (nose down)
        # to accelerate toward it.  The Kd term uses body velocity to damp
        # the lateral motion: when moving toward the target (v_b > 0 in that
        # axis), we *reduce* the tilt command so the vehicle decelerates.
        Kx = dict(self.outer.get("lateral_x", {}))
        Ky = dict(self.outer.get("lateral_y", {}))

        pitch_des = -(
            s * float(Kx.get("Kp", 0.0)) * float(target_body[0])
            - s * float(Kx.get("Kd", 0.0)) * float(v_b[0])
        )
        roll_des = (
            s * float(Ky.get("Kp", 0.0)) * float(target_body[1])
            - s * float(Ky.get("Kd", 0.0)) * float(v_b[1])
        )
        pitch_des = float(np.clip(pitch_des, -self.max_tilt, self.max_tilt))
        roll_des = float(np.clip(roll_des, -self.max_tilt, self.max_tilt))

        # --- Inner loop: attitude PD -> fin deflections (rad) ---
        # Estimate roll/pitch from gravity direction (training.md §3.3 rationale).
        g2 = float(g_body[2]) if abs(float(g_body[2])) > 1e-9 else 1e-9
        roll_est = float(np.arctan2(float(g_body[1]), g2))
        pitch_est = float(np.arctan2(-float(g_body[0]), g2))

        roll_error = roll_des - roll_est
        pitch_error = pitch_des - pitch_est

        Kr = dict(self.inner.get("roll", {}))
        Kp_i = dict(self.inner.get("pitch", {}))

        # PD + gyroscopic feedforward.
        # The EOM has: I*w_dot = tau - w x (I*w) - w x h_fan
        # So the effective gyroscopic disturbance torque is:
        #   tau_eff_gyro_x = -omega_y * h_z  (negative when pitching positive)
        #   tau_eff_gyro_y = +omega_x * h_z  (positive when rolling positive)
        # To compensate, the feedforward adds:
        #   roll_cmd  += +Kff * omega_y  (positive roll torque to oppose negative gyro)
        #   pitch_cmd += -Kff * omega_x  (negative pitch torque to oppose positive gyro)
        Kff = float(self.inner.get("gyro_ff", 0.0))
        roll_cmd = (
            s * float(Kr.get("Kp", 0.0)) * roll_error
            - s * float(Kr.get("Kd", 0.0)) * float(omega[0])
            + s * Kff * float(omega[1])
        )
        pitch_cmd = (
            s * float(Kp_i.get("Kp", 0.0)) * pitch_error
            - s * float(Kp_i.get("Kd", 0.0)) * float(omega[1])
            - s * Kff * float(omega[0])
        )

        # Yaw rate damping via differential fin deflection.
        # The yaw deflection is clamped to a fraction of delta_max so that
        # yaw damping never consumes more than ~40 % of the available fin
        # authority, leaving the rest for pitch/roll attitude control.
        Kyaw = float(self.inner.get("yaw_Kd", 0.0))
        max_yaw_frac = float(self.inner.get("max_yaw_frac", 0.30))
        yaw_limit = max_yaw_frac * self.delta_max
        yaw_damp = -Kyaw * float(omega[2])

        # Motor-torque feedforward: when thrust changes, the fan accelerates
        # and the reaction torque spins the vehicle in yaw.  Pre-compensate
        # by commanding yaw fins proportional to the thrust change.
        #   tau_motor_z = -I_fan * d(omega_fan)/dt   (negative when speeding up)
        # A positive thrust_delta → negative yaw torque → we need positive yaw_ff
        K_motor_ff = float(self.inner.get("motor_yaw_ff", 0.0))
        yaw_ff = K_motor_ff * self._thrust_delta

        yaw_total = float(np.clip(yaw_damp + yaw_ff, -yaw_limit, yaw_limit))

        # Map to fin actions: common-mode deflection for pitch/roll torque,
        # differential deflection for yaw torque.
        #
        # Pitch: fins 1/2 (right/left, lift in body-x) produce pitch torque
        #   via common-mode deflection.
        # Roll:  fins 3/4 (fwd/aft, lift in body-y) produce roll torque
        #   via common-mode deflection (sign inverted per fin_model.compute).
        # Yaw:   the correct yaw-producing pattern is [-d,+d,+d,-d].
        #   Fins 1 at y>0 and fin 3 at x>0 must have *opposite* differential
        #   signs so that their yaw-torque contributions add rather than
        #   cancel.  Pattern verified via open-loop yaw diagnostic.
        fin1 = float(np.clip((+pitch_cmd - yaw_total) / self.delta_max, -1.0, 1.0))
        fin2 = float(np.clip((+pitch_cmd + yaw_total) / self.delta_max, -1.0, 1.0))
        fin3 = float(np.clip((-roll_cmd + yaw_total) / self.delta_max, -1.0, 1.0))
        fin4 = float(np.clip((-roll_cmd - yaw_total) / self.delta_max, -1.0, 1.0))

        return np.array([thrust_action, fin1, fin2, fin3, fin4], dtype=np.float32)


__all__ = ["PIDController"]

