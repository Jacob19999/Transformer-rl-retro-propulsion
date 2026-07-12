"""
Crash detection criteria for the EDF drone.

Detects crash conditions:
  - Impact speed threshold exceeded at contact
  - Excessive tilt at contact
  - Excessive angular rate at contact
  - Unsafe body contact (non-landing-gear collision)
  - Tip-over after initial contact

All thresholds are configurable from task YAML config.
Evaluation is vectorized for num_envs environments.
"""

from __future__ import annotations
import torch
from torch import Tensor
from tvc_env.common.quaternions import tilt_angle


class CrashDetector:
    """Vectorized crash detection for all environments."""

    def __init__(
        self,
        max_impact_speed: float = 3.0,    # m/s, source: estimate
        max_tilt_at_contact: float = 0.5, # rad (~28°), source: estimate
        max_angular_rate_at_contact: float = 3.0,  # rad/s
        max_tilt: float = 1.57,           # rad (90°), absolute tilt limit
        max_altitude_error: float = 10.0, # m
    ):
        self.max_impact_speed = max_impact_speed
        self.max_tilt_at_contact = max_tilt_at_contact
        self.max_angular_rate_at_contact = max_angular_rate_at_contact
        self.max_tilt = max_tilt
        self.max_altitude_error = max_altitude_error

    @classmethod
    def from_task_config(cls, task_config: dict) -> "CrashDetector":
        """Create CrashDetector from task YAML config."""
        task = task_config.get("task", task_config)
        term = task.get("termination", {})
        return cls(
            max_impact_speed=term.get("max_impact_speed", 3.0),
            max_tilt_at_contact=term.get("max_tilt_at_contact", 0.5),
            max_angular_rate_at_contact=term.get("max_angular_rate_at_contact", 3.0),
            max_tilt=term.get("max_tilt", 1.57),
            max_altitude_error=term.get("max_altitude_error", 10.0),
        )

    def check_impact_speed(
        self,
        impact_speed: Tensor,     # (num_envs,) downward speed at contact (m/s)
        in_contact: Tensor,       # (num_envs,) bool
    ) -> Tensor:
        """Detect crash from excessive impact speed.

        Returns:
            Bool tensor (num_envs,) — True where crash detected.
        """
        return in_contact.bool() & (impact_speed > self.max_impact_speed)

    def check_tilt_at_contact(
        self,
        quaternion_wxyz: Tensor,  # (num_envs, 4)
        in_contact: Tensor,       # (num_envs,) bool
    ) -> Tensor:
        """Detect crash from excessive tilt at the moment of contact."""
        return in_contact.bool() & (tilt_angle(quaternion_wxyz) > self.max_tilt_at_contact)

    def check_angular_rate_at_contact(
        self,
        angular_rate: Tensor,     # (num_envs,) angular rate magnitude (rad/s)
        in_contact: Tensor,       # (num_envs,) bool
    ) -> Tensor:
        """Detect crash from excessive angular rate at contact."""
        return in_contact.bool() & (angular_rate > self.max_angular_rate_at_contact)

    def check_excessive_tilt(self, quaternion_wxyz: Tensor) -> Tensor:
        """Detect crash from exceeding absolute tilt limit (90° = flipped over)."""
        return tilt_angle(quaternion_wxyz) > self.max_tilt

    def check_altitude_error(
        self,
        altitude_error: Tensor,   # (num_envs,) absolute altitude error (m)
    ) -> Tensor:
        """Detect crash from excessive altitude error (too high or too low)."""
        return altitude_error.abs() > self.max_altitude_error

    def evaluate(
        self,
        in_contact: Tensor,
        impact_speed: Tensor,
        quaternion_wxyz: Tensor,
        angular_rate: Tensor,
        altitude_error: Tensor,
    ) -> Tensor:
        """Evaluate all crash criteria and return combined crash signal.

        Returns:
            Bool tensor (num_envs,) — True where ANY crash criterion is met.
        """
        impact_crash = self.check_impact_speed(impact_speed, in_contact)
        tilt_contact_crash = self.check_tilt_at_contact(quaternion_wxyz, in_contact)
        rate_crash = self.check_angular_rate_at_contact(angular_rate, in_contact)
        tilt_crash = self.check_excessive_tilt(quaternion_wxyz)
        altitude_crash = self.check_altitude_error(altitude_error)

        return impact_crash | tilt_contact_crash | rate_crash | tilt_crash | altitude_crash
