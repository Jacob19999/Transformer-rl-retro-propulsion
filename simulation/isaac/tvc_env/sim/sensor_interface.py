"""
Contact and IMU sensor access for the TVC environment.

Reads contact forces from PhysX contact reporter, detects ground contact per
landing-gear contact regions from asset metadata, and provides contact normal
vectors and impact velocities.

Requires Isaac Sim runtime for sensor data access.
"""

from __future__ import annotations
import torch
from torch import Tensor
from typing import Any


class SensorInterface:
    """Reads contact sensor data from Isaac Lab for the EDF drone."""

    def __init__(
        self,
        contact_sensor,     # Isaac Lab ContactSensor object
        metadata: dict[str, Any],
    ):
        self._contact_sensor = contact_sensor
        self._landing_contact_regions = metadata.get("landing_contact_regions", [])

    def get_contact_force_matrix(self) -> Tensor:
        """Get contact force matrix from the sensor.

        Returns:
            Tensor of shape (num_envs, num_bodies, 3) — contact forces in world frame (N).
        """
        if self._contact_sensor is None:
            return torch.zeros(1, 1, 3)
        return self._contact_sensor.data.force_matrix_w.clone()

    def get_net_contact_forces(self) -> Tensor:
        """Get net contact forces summed over all contact bodies.

        Returns:
            Tensor of shape (num_envs, 3) — net contact force per env (N).
        """
        forces = self.get_contact_force_matrix()  # (num_envs, num_bodies, 3)
        return forces.sum(dim=1)  # (num_envs, 3)

    def is_in_contact(self) -> Tensor:
        """Check if any landing contact region is in contact with ground.

        Returns:
            Bool tensor of shape (num_envs,) — True if any landing contact detected.
        """
        forces = self.get_contact_force_matrix()  # (num_envs, num_bodies, 3)
        contact_magnitudes = forces.norm(dim=-1)  # (num_envs, num_bodies)
        return (contact_magnitudes > 0.1).any(dim=-1)  # (num_envs,)

    def get_impact_speed(self, linear_vel_world: Tensor) -> Tensor:
        """Estimate impact speed from downward component of linear velocity at contact.

        Args:
            linear_vel_world: Tensor (num_envs, 3) — linear velocity in Isaac world frame (m/s).

        Returns:
            Tensor (num_envs,) — magnitude of downward velocity component (m/s).
        """
        # In Isaac y-up convention, downward = negative y
        return (-linear_vel_world[:, 1]).clamp(min=0.0)  # Only downward component

    def get_contact_normal_forces(self) -> Tensor:
        """Get contact normal force magnitudes per environment.

        Returns:
            Tensor (num_envs,) — scalar contact normal force magnitude (N).
        """
        return self.get_net_contact_forces().norm(dim=-1)
