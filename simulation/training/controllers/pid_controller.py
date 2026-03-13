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
        self.prev_h_agl: float | None = None

        # Thrust rate limiter: maximum change in normalised thrust action per
        # controller step.  Limits motor reaction torque that causes yaw spin-up.
        self.thrust_rate_limit: float = float(
            cfg.get("thrust_rate_limit", 0.15)
        )
        self.prev_thrust_action: float = 0.0
        self._thrust_delta: float = 0.0

        # Low-pass filter on yaw rate to prevent servo-lag instability.
        # The yaw oscillation (~7 Hz) is above the servo bandwidth (~4 Hz);
        # filtering omega_z reduces the effective gain at the oscillation
        # frequency so the delayed servo response doesn't amplify it.
        yaw_lp_hz = float(self.inner.get("yaw_lp_hz", 2.0))
        dt_ctrl = float(cfg.get("dt", 0.025))
        if yaw_lp_hz > 0:
            tau_lp = 1.0 / (2.0 * np.pi * yaw_lp_hz)
            self._yaw_lp_alpha = dt_ctrl / (dt_ctrl + tau_lp)
        else:
            self._yaw_lp_alpha = 1.0  # no filter
        self._omega_z_filt: float = 0.0

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
        self.prev_h_agl = None
        self.prev_thrust_action = 0.0
        self._thrust_delta = 0.0
        self._omega_z_filt = 0.0

    def _gain_scale(self, h_agl: float) -> float:
        if not self._schedule_enabled or not self._phases:
            return 1.0
        h = float(h_agl)
        for ph in self._phases:
            if ph.alt_lo <= h <= ph.alt_hi:
                return float(ph.gain_scale)
        return 1.0

    def _compute_action(
        self, obs: np.ndarray, *, return_debug: bool
    ) -> tuple[np.ndarray, dict[str, float] | None]:
        """Shared implementation for get_action and debug variants."""
        o = np.asarray(obs, dtype=float).reshape(-1)
        if o.size < 20:
            raise ValueError(f"obs must have at least 20 elements, got {o.size}.")

        target_body = o[0:3]
        v_b = o[3:6]
        g_body = o[6:9]
        omega = o[9:12]
        h_agl = float(o[16])

        s = self._gain_scale(h_agl)

        Kz = dict(self.outer["altitude"])
        alt_target = float(Kz.get("target_h_agl", 0.0))
        alt_error = alt_target - h_agl

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

        # Use measured altitude change instead of body-frame z velocity so the
        # thrust damping term stays meaningful when the vehicle tilts.
        if self.prev_h_agl is None:
            alt_rate = 0.0
        else:
            alt_rate = float((self.prev_h_agl - h_agl) / self.dt)
        self.prev_h_agl = h_agl
        self.prev_alt_error = float(alt_error)

        thrust_cmd_raw = (
            float(Kz.get("Kp", 0.0)) * alt_error_clamped
            + float(Kz.get("Ki", 0.0)) * self.alt_integral
            + float(Kz.get("Kd", 0.0)) * alt_rate
        )
        thrust_action = float(np.clip(thrust_cmd_raw, -1.0, 1.0))

        if self.thrust_rate_limit > 0:
            lo = self.prev_thrust_action - self.thrust_rate_limit
            hi = self.prev_thrust_action + self.thrust_rate_limit
            thrust_action = float(np.clip(thrust_action, lo, hi))
        self._thrust_delta = thrust_action - self.prev_thrust_action
        self.prev_thrust_action = thrust_action

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

        gx = float(g_body[0])
        gy = float(g_body[1])
        gz = float(g_body[2])
        # Use bounded gravity-vector tilt estimates ([-pi/2, +pi/2]) to avoid
        # branch flips when g_body[2] crosses zero during aggressive maneuvers.
        roll_den = max(float(np.hypot(gx, gz)), 1e-9)
        pitch_den = max(float(np.hypot(gy, gz)), 1e-9)
        roll_est = float(np.arctan2(gy, roll_den))
        pitch_est = float(np.arctan2(-gx, pitch_den))

        roll_error = roll_des - roll_est
        pitch_error = pitch_des - pitch_est

        Kr = dict(self.inner.get("roll", {}))
        Kp_i = dict(self.inner.get("pitch", {}))

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

        omega_z_raw = float(omega[2])
        self._omega_z_filt = (
            self._yaw_lp_alpha * omega_z_raw
            + (1.0 - self._yaw_lp_alpha) * self._omega_z_filt
        )

        Kyaw = float(self.inner.get("yaw_Kd", 0.0))
        max_yaw_frac = float(self.inner.get("max_yaw_frac", 0.30))
        yaw_limit = max_yaw_frac * self.delta_max
        yaw_total = float(
            np.clip(-Kyaw * self._omega_z_filt, -yaw_limit, yaw_limit)
        )

        fin1 = float(np.clip((+pitch_cmd - yaw_total) / self.delta_max, -1.0, 1.0))
        fin2 = float(np.clip((+pitch_cmd + yaw_total) / self.delta_max, -1.0, 1.0))
        fin3 = float(np.clip((-roll_cmd + yaw_total) / self.delta_max, -1.0, 1.0))
        fin4 = float(np.clip((-roll_cmd - yaw_total) / self.delta_max, -1.0, 1.0))

        action = np.array(
            [thrust_action, fin1, fin2, fin3, fin4], dtype=np.float32
        )

        if not return_debug:
            return action, None

        debug: dict[str, float] = {
            "gain_scale": float(s),
            "alt_target": alt_target,
            "alt_error": alt_error,
            "alt_error_clamped": alt_error_clamped,
            "alt_integral": self.alt_integral,
            "alt_rate": alt_rate,
            "thrust_cmd_raw": thrust_cmd_raw,
            "thrust_action": thrust_action,
            "pitch_des": pitch_des,
            "roll_des": roll_des,
            "pitch_est": pitch_est,
            "roll_est": roll_est,
            "pitch_error": pitch_error,
            "roll_error": roll_error,
            "omega_x": float(omega[0]),
            "omega_y": float(omega[1]),
            "omega_z": float(omega[2]),
            "roll_Kp": float(Kr.get("Kp", 0.0)),
            "roll_Kd": float(Kr.get("Kd", 0.0)),
            "pitch_Kp": float(Kp_i.get("Kp", 0.0)),
            "pitch_Kd": float(Kp_i.get("Kd", 0.0)),
            "gyro_ff": float(Kff),
            "pitch_cmd": pitch_cmd,
            "roll_cmd": roll_cmd,
            "omega_z_raw": omega_z_raw,
            "omega_z_filt": self._omega_z_filt,
            "yaw_total": yaw_total,
            "fin1": fin1,
            "fin2": fin2,
            "fin3": fin3,
            "fin4": fin4,
        }
        return action, debug

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Map observation (20,) to normalized action in [-1,1]^5."""
        action, _ = self._compute_action(obs, return_debug=False)
        return action

    def get_action_with_debug(
        self, obs: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Return action and a debug snapshot of internal PID terms."""
        action, debug = self._compute_action(obs, return_debug=True)
        assert debug is not None
        return action, debug


__all__ = ["PIDController"]

