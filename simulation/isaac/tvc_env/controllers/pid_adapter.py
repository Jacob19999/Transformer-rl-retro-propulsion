"""
PID controller adapter for the TVC environment.

Maps PID output to 5-dim action vector per action_space contract:
  - Altitude loop → throttle [0, 1]
  - Attitude loop → roll/pitch/yaw commands → fin angles via pid_fin_mixer.py

Observation indexing per observation_space contract:
  [0:3]   position_error  (x, y, z) — z component = altitude error
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


class PIDController(BaseController):
    """Simple PID controller for hover stabilization."""

    def __init__(
        self,
        # Altitude PID gains
        kp_alt: float = 2.0,
        ki_alt: float = 0.1,
        kd_alt: float = 1.0,
        # Attitude PID gains (roll, pitch, yaw)
        kp_att: float = 3.0,
        ki_att: float = 0.05,
        kd_att: float = 0.5,
        # Throttle bias for gravity compensation
        throttle_hover: float = 0.72,  # Approximate hover throttle (source: estimate)
        max_fin_angle: float = 0.2,
        num_envs: int = 1,
        config: dict[str, Any] | None = None,
        device: torch.device = None,
    ):
        super().__init__(config)
        self.kp_alt = kp_alt
        self.ki_alt = ki_alt
        self.kd_alt = kd_alt
        self.kp_att = kp_att
        self.ki_att = ki_att
        self.kd_att = kd_att
        self.throttle_hover = throttle_hover
        self._max_fin_angle = max_fin_angle
        self.num_envs = num_envs
        self.device = device

        self._fin_mixer = PIDFinMixer(max_fin_angle=max_fin_angle)

        # Integrator states
        self._int_alt = torch.zeros(num_envs, device=device)
        self._int_att = torch.zeros(num_envs, 3, device=device)
        self._prev_alt_err = torch.zeros(num_envs, device=device)
        self._prev_att_err = torch.zeros(num_envs, 3, device=device)
        self._dt = 1.0 / 30.0  # Approximately 30 Hz RL update rate

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
        pos_error = obs[:, 0:3]           # (num_envs, 3) in world frame
        quat_wxyz = obs[:, 3:7]           # (num_envs, 4)
        ang_vel_frd = obs[:, 10:13]       # (num_envs, 3)

        # --- Altitude control (z-position error in FRD convention) ---
        # Observation pos_error z-component corresponds to altitude
        alt_err = pos_error[:, 2]         # (num_envs,) — positive = below target

        self._int_alt = (self._int_alt + alt_err * self._dt).clamp(-5.0, 5.0)
        d_alt_err = (alt_err - self._prev_alt_err) / self._dt
        self._prev_alt_err = alt_err.clone()

        throttle_correction = (
            self.kp_alt * alt_err +
            self.ki_alt * self._int_alt +
            self.kd_alt * d_alt_err
        )
        throttle = (self.throttle_hover + throttle_correction).clamp(0.0, 1.0)  # (num_envs,)

        # --- Attitude control ---
        roll, pitch, _ = to_euler(quat_wxyz)
        att_err = torch.stack([-roll, -pitch, torch.zeros(num_envs, device=device)], dim=-1)  # (num_envs, 3)

        self._int_att = (self._int_att + att_err * self._dt).clamp(-1.0, 1.0)
        d_att_err = (att_err - self._prev_att_err) / self._dt
        self._prev_att_err = att_err.clone()

        # Add damping from angular rates
        rate_cmd = (
            self.kp_att * att_err +
            self.ki_att * self._int_att +
            self.kd_att * d_att_err -
            ang_vel_frd * 0.2
        )  # (num_envs, 3) roll/pitch/yaw rate commands

        # Mix to fin angles
        fin_angles = self._fin_mixer.mix(rate_cmd[:, 0], rate_cmd[:, 1], rate_cmd[:, 2])  # (num_envs, 4)

        action = torch.cat([fin_angles, throttle.unsqueeze(-1)], dim=-1)  # (num_envs, 5)
        return self.validate_action(action)

    def reset(self, env_ids: Tensor | None = None) -> None:
        """Reset integrators for specified environments."""
        if env_ids is None:
            self._int_alt.fill_(0.0)
            self._int_att.fill_(0.0)
            self._prev_alt_err.fill_(0.0)
            self._prev_att_err.fill_(0.0)
        else:
            self._int_alt[env_ids] = 0.0
            self._int_att[env_ids] = 0.0
            self._prev_alt_err[env_ids] = 0.0
            self._prev_att_err[env_ids] = 0.0
