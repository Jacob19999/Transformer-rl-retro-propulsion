"""
PID controller adapter for the TVC environment.

Maps PID output to 5-dim action vector per action_space contract:
  - Altitude loop â†’ throttle [0, 1]
  - Attitude loop â†’ roll/pitch/yaw commands â†’ fin angles via pid_fin_mixer.py

Observation indexing per observation_space contract:
  [0:3]   position_error  (x, y, z) â€” z component = altitude error
  [3:7]   attitude_quat_wxyz
  [7:10]  linear_vel_body_frd
  [10:13] angular_vel_body_frd
"""

from __future__ import annotations
import torch
from torch import Tensor
from typing import Any

from tvc_env.controllers.base import BaseController
from tvc_env.controllers.pid_fin_mixer import PIDFinMixer
from tvc_env.common.quaternions import to_euler
from tvc_env.common.quaternions import inverse as quat_inverse, normalize as quat_normalize, rotate_vector
from tvc_env.common.frames import isaac_velocity_to_frd


class PIDController(BaseController):
    """PID controller for hover stabilization with XY position hold."""

    def __init__(
        self,
        # Altitude PID gains
        kp_alt: float = 0.22,
        ki_alt: float = 0.01,
        kd_alt: float = 0.10,
        # Attitude gains (roll/pitch shared, yaw separate)
        kp_att: float = 0.24,
        ki_att: float = 0.00,
        kd_att: float = 0.36,
        kp_yaw: float = 0.00,
        ki_yaw: float = 0.00,
        kd_yaw: float = 0.00,
        # XY position hold -> desired attitude
        k_pos_xy: float = 0.055,
        ki_pos_xy: float = 0.001,
        k_vel_xy: float = 0.30,
        max_tilt_cmd: float = 0.055,
        max_tilt_rate: float = 0.08,
        # Lateral authority scheduling for stability under recovery load
        tilt_recovery_alt_err: float = 0.50,
        tilt_recovery_ang_rate: float = 1.20,
        min_lateral_scale: float = 0.60,
        # Deadband-aware lateral actuation floor (servo deadband ~= 0.017 rad)
        min_fin_cmd_xy: float = 0.018,
        xy_active_error: float = 0.20,
        # Throttle bias for gravity compensation
        throttle_hover: float = 0.90,
        max_fin_angle: float = 0.08,
        num_envs: int = 1,
        config: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ):
        super().__init__(config)
        self.kp_alt = kp_alt
        self.ki_alt = ki_alt
        self.kd_alt = kd_alt

        self.kp_att = kp_att
        self.ki_att = ki_att
        self.kd_att = kd_att
        self.kp_yaw = kp_yaw
        self.ki_yaw = ki_yaw
        self.kd_yaw = kd_yaw

        self.k_pos_xy = k_pos_xy
        self.ki_pos_xy = ki_pos_xy
        self.k_vel_xy = k_vel_xy
        self.max_tilt_cmd = max_tilt_cmd
        self.max_tilt_rate = max_tilt_rate
        self.tilt_recovery_alt_err = tilt_recovery_alt_err
        self.tilt_recovery_ang_rate = tilt_recovery_ang_rate
        self.min_lateral_scale = min_lateral_scale
        self.min_fin_cmd_xy = min_fin_cmd_xy
        self.xy_active_error = xy_active_error

        self.throttle_hover = throttle_hover
        self._max_fin_angle = max_fin_angle
        self.num_envs = num_envs
        self.device = device

        self._fin_mixer = PIDFinMixer(max_fin_angle=max_fin_angle)

        self._kp_vec = torch.tensor([kp_att, kp_att, kp_yaw], dtype=torch.float32, device=device)
        self._ki_vec = torch.tensor([ki_att, ki_att, ki_yaw], dtype=torch.float32, device=device)
        self._kd_vec = torch.tensor([kd_att, kd_att, kd_yaw], dtype=torch.float32, device=device)

        # Integrator states
        self._int_alt = torch.zeros(num_envs, device=device)
        self._int_att = torch.zeros(num_envs, 3, device=device)
        self._int_pos_xy = torch.zeros(num_envs, 2, device=device)
        self._desired_tilt_cmd = torch.zeros(num_envs, 2, device=device)
        self._dt = 1.0 / 30.0  # Approximately 30 Hz RL update rate
        self._last_debug: dict[str, Tensor] = {}

    def compute_action(self, obs: Tensor) -> Tensor:
        """Compute PID action from observation.

        Args:
            obs: (num_envs, 24) observation tensor per contract.

        Returns:
            Action (num_envs, 5): [fin0, fin1, fin2, fin3, throttle].
        """
        num_envs = obs.shape[0]
        device = obs.device

        # Extract state from observation
        pos_error_world = obs[:, 0:3]     # (num_envs, 3) target - current in Isaac world frame
        quat_wxyz = obs[:, 3:7]           # (num_envs, 4)
        lin_vel_frd = obs[:, 7:10]        # (num_envs, 3)
        ang_vel_frd = obs[:, 10:13]       # (num_envs, 3)

        # Rotate XY position error into body-FRD to build desired roll/pitch commands.
        q_inv = quat_inverse(quat_normalize(quat_wxyz))
        pos_error_body_isaac = rotate_vector(q_inv, pos_error_world)
        pos_error_body_frd = isaac_velocity_to_frd(pos_error_body_isaac)

        # XY hold uses PI on position + D on velocity to reject steady lateral drift.
        pos_error_xy = pos_error_body_frd[:, 0:2]
        vel_xy = lin_vel_frd[:, 0:2]
        candidate_int_xy = (self._int_pos_xy + pos_error_xy * self._dt).clamp(-1.5, 1.5)
        desired_xy_unsat_candidate = (
            self.k_pos_xy * pos_error_xy
            + self.ki_pos_xy * candidate_int_xy
            - self.k_vel_xy * vel_xy
        )

        integrate_xy = (
            ((desired_xy_unsat_candidate < self.max_tilt_cmd) | (pos_error_xy < 0.0))
            & ((desired_xy_unsat_candidate > -self.max_tilt_cmd) | (pos_error_xy > 0.0))
        )
        self._int_pos_xy = torch.where(integrate_xy, candidate_int_xy, self._int_pos_xy)

        desired_xy_unsat = (
            self.k_pos_xy * pos_error_xy
            + self.ki_pos_xy * self._int_pos_xy
            - self.k_vel_xy * vel_xy
        )
        desired_xy_target = desired_xy_unsat.clamp(-self.max_tilt_cmd, self.max_tilt_cmd)
        pos_error_xy_norm = torch.linalg.vector_norm(pos_error_xy, dim=-1)

        # Stability-first scheduling:
        # reduce lateral authority when altitude or angular-rate recovery is active.
        alt_err_abs = pos_error_world[:, 2].abs()
        ang_rate_norm = torch.linalg.vector_norm(ang_vel_frd, dim=-1)
        alt_load = (alt_err_abs / max(self.tilt_recovery_alt_err, 1e-6)).clamp(0.0, 1.0)
        rate_load = (ang_rate_norm / max(self.tilt_recovery_ang_rate, 1e-6)).clamp(0.0, 1.0)
        recovery_load = torch.maximum(alt_load, rate_load)
        lateral_scale = (1.0 - 0.7 * recovery_load).clamp(self.min_lateral_scale, 1.0)
        desired_xy_target = desired_xy_target * lateral_scale.unsqueeze(-1)

        # Rate-limit desired tilt to keep lateral corrections smooth and avoid snap-over.
        max_tilt_step = self.max_tilt_rate * self._dt
        desired_tilt_delta = (desired_xy_target - self._desired_tilt_cmd).clamp(-max_tilt_step, max_tilt_step)
        self._desired_tilt_cmd = self._desired_tilt_cmd + desired_tilt_delta
        desired_xy = self._desired_tilt_cmd

        # Body-FRD thrust acts along -Z. Positive pitch tilts lift toward +X in
        # the rigid-body dynamics, so X position error must command opposite pitch.
        # Positive roll tilts lift toward body +Y.
        desired_pitch = -desired_xy[:, 0]
        desired_roll = desired_xy[:, 1]
        desired_yaw = torch.zeros(num_envs, device=device, dtype=obs.dtype)

        # --- Altitude control ---
        # z-position error in Isaac world frame (z-up): positive means target is above current position.
        alt_err = pos_error_world[:, 2]
        # Body-FRD z-velocity is positive downward; adding this term increases throttle while descending.
        alt_vel_down = lin_vel_frd[:, 2]

        candidate_int_alt = (self._int_alt + alt_err * self._dt).clamp(-2.5, 2.5)
        throttle_unsat_candidate = (
            self.throttle_hover
            + self.kp_alt * alt_err
            + self.ki_alt * candidate_int_alt
            + self.kd_alt * alt_vel_down
        )

        # Basic anti-windup: only integrate when integrator action helps move away from saturation.
        integrate_alt = (
            ((throttle_unsat_candidate < 1.0) | (alt_err < 0.0))
            & ((throttle_unsat_candidate > 0.0) | (alt_err > 0.0))
        )
        self._int_alt = torch.where(integrate_alt, candidate_int_alt, self._int_alt)

        throttle_correction = (
            self.kp_alt * alt_err
            + self.ki_alt * self._int_alt
            + self.kd_alt * alt_vel_down
        )
        throttle_unsat = self.throttle_hover + throttle_correction
        throttle = throttle_unsat.clamp(0.0, 1.0)

        # --- Attitude control ---
        roll, pitch, yaw = to_euler(quat_wxyz)
        # Isaac and body-FRD share +X, but +Y/+Z are flipped at the control boundary.
        pitch = -pitch
        yaw = -yaw
        att_err = torch.stack(
            [desired_roll - roll, desired_pitch - pitch, desired_yaw - yaw],
            dim=-1,
        )

        self._int_att = (self._int_att + att_err * self._dt).clamp(-0.5, 0.5)

        kp_vec = self._kp_vec.to(device=device, dtype=obs.dtype)
        ki_vec = self._ki_vec.to(device=device, dtype=obs.dtype)
        kd_vec = self._kd_vec.to(device=device, dtype=obs.dtype)

        rate_cmd = (
            kp_vec * att_err
            + ki_vec * self._int_att
            - kd_vec * ang_vel_frd
        )

        # Ensure roll/pitch authority clears servo deadband while lateral error is active.
        # Apply this before mixing so a small cross-axis command is not promoted
        # into an unintended four-fin floor after the mixer.
        lateral_active = pos_error_xy_norm > self.xy_active_error
        rate_cmd_for_mix = rate_cmd.clone()
        roll_pitch_cmd = rate_cmd_for_mix[:, 0:2]
        roll_pitch_mag = roll_pitch_cmd.abs()
        roll_pitch_sign = torch.sign(roll_pitch_cmd)
        needs_floor = (
            lateral_active.unsqueeze(-1)
            & (roll_pitch_mag > 1e-6)
            & (roll_pitch_mag < self.min_fin_cmd_xy)
        )
        rate_cmd_for_mix[:, 0:2] = torch.where(
            needs_floor,
            roll_pitch_sign * self.min_fin_cmd_xy,
            roll_pitch_cmd,
        )

        # Mix to fin angles
        fin_angles = self._fin_mixer.mix(
            rate_cmd_for_mix[:, 0],
            rate_cmd_for_mix[:, 1],
            rate_cmd_for_mix[:, 2],
        )

        action = torch.cat([fin_angles, throttle.unsqueeze(-1)], dim=-1)

        self._last_debug = {
            "pos_error_body_frd": pos_error_body_frd.detach(),
            "pos_error_xy_norm": pos_error_xy_norm.detach(),
            "pos_error_int_xy": self._int_pos_xy.detach(),
            "lateral_scale": lateral_scale.detach(),
            "desired_attitude_rp_unsat": desired_xy_unsat.detach(),
            "desired_attitude_rp_target": desired_xy_target.detach(),
            "desired_attitude_rpy": torch.stack([desired_roll, desired_pitch, desired_yaw], dim=-1).detach(),
            "attitude_error_rpy": att_err.detach(),
            "angular_rate_frd": ang_vel_frd.detach(),
            "rate_cmd_rpy": rate_cmd.detach(),
            "rate_cmd_mixed_rpy": rate_cmd_for_mix.detach(),
            "alt_err": alt_err.detach(),
            "alt_vel_down": alt_vel_down.detach(),
            "throttle_unsat": throttle_unsat.detach(),
            "throttle_cmd": throttle.detach(),
        }

        return self.validate_action(action)

    def get_debug_state(self, env_idx: int = 0) -> dict[str, float | list[float]]:
        """Return last-step controller internals for one environment."""
        if not self._last_debug:
            return {}

        if env_idx < 0 or env_idx >= self.num_envs:
            env_idx = 0

        debug: dict[str, float | list[float]] = {}
        for key, value in self._last_debug.items():
            v = value[env_idx].detach().cpu()
            if v.ndim == 0:
                debug[key] = float(v.item())
            else:
                debug[key] = [float(x) for x in v.tolist()]
        return debug

    def reset(self, env_ids: Tensor | None = None) -> None:
        """Reset integrators for specified environments."""
        if env_ids is None:
            self._int_alt.fill_(0.0)
            self._int_att.fill_(0.0)
            self._int_pos_xy.fill_(0.0)
            self._desired_tilt_cmd.fill_(0.0)
            self._last_debug = {}
        else:
            self._int_alt[env_ids] = 0.0
            self._int_att[env_ids] = 0.0
            self._int_pos_xy[env_ids] = 0.0
            self._desired_tilt_cmd[env_ids] = 0.0
