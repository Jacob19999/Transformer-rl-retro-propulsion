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
        self._body_link_name = metadata.get(
            "landing_contact_body", metadata.get("body_link_name", "Body")
        )
        self._fin_link_names = set(
            metadata.get("unsafe_contact_links", metadata.get("fin_link_names", []))
        )

        body_names = list(contact_sensor.body_names) if contact_sensor is not None else []
        self._body_index = body_names.index(self._body_link_name) if self._body_link_name in body_names else None
        self._unsafe_body_indices = [
            index for index, name in enumerate(body_names) if name in self._fin_link_names
        ]
        if contact_sensor is not None and self._body_index is None:
            raise RuntimeError(
                f"Contact sensor does not include landing body '{self._body_link_name}'. "
                f"Resolved bodies: {body_names}"
            )

    def get_contact_force_matrix(self) -> Tensor:
        """Get contact force matrix from the sensor.

        Returns:
            Tensor of shape (num_envs, num_bodies, 3) — contact forces in world frame (N).
        """
        if self._contact_sensor is None:
            return torch.zeros(1, 1, 3)
        return self._contact_sensor.data.net_forces_w.clone()

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
        return self.get_landing_contact_force() > 0.1

    def get_landing_contact_force(self) -> Tensor:
        """Return normal-force magnitude on the body/landing structure."""
        forces = self.get_contact_force_matrix()
        if self._body_index is None:
            return torch.zeros(forces.shape[0], device=forces.device, dtype=forces.dtype)
        return forces[:, self._body_index].norm(dim=-1)

    def read_contact_summary(self, force_threshold: float) -> tuple[Tensor, Tensor]:
        """Read landing force and unsafe-link contact from one sensor snapshot."""
        if self._contact_sensor is None:
            forces = self.get_contact_force_matrix()
        else:
            forces = self._contact_sensor.data.net_forces_w

        if self._body_index is None:
            landing_force = torch.zeros(forces.shape[0], device=forces.device, dtype=forces.dtype)
        else:
            landing_force = forces[:, self._body_index].norm(dim=-1)

        if not self._unsafe_body_indices:
            unsafe_contact = torch.zeros(forces.shape[0], dtype=torch.bool, device=forces.device)
        else:
            unsafe_force = forces[:, self._unsafe_body_indices].norm(dim=-1)
            unsafe_contact = (unsafe_force > force_threshold).any(dim=-1)
        return landing_force, unsafe_contact

    def has_unsafe_contact(self, force_threshold: float = 0.1) -> Tensor:
        """Return whether any fin link has contacted the environment."""
        _, unsafe_contact = self.read_contact_summary(force_threshold)
        return unsafe_contact

    def get_impact_speed(self, linear_vel_world: Tensor) -> Tensor:
        """Estimate impact speed from downward component of linear velocity at contact.

        Args:
            linear_vel_world: Tensor (num_envs, 3) — linear velocity in Isaac world frame (m/s).

        Returns:
            Tensor (num_envs,) — magnitude of downward velocity component (m/s).
        """
        # In Isaac Z-up convention, downward = negative z.
        return (-linear_vel_world[:, 2]).clamp(min=0.0)  # Only downward component

    def get_contact_normal_forces(self) -> Tensor:
        """Get contact normal force magnitudes per environment.

        Returns:
            Tensor (num_envs,) — scalar contact normal force magnitude (N).
        """
        return self.get_landing_contact_force()
