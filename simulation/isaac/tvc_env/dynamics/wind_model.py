"""
Wind and drag disturbance model.

Implements:
  - Steady wind vector in world frame
  - Gust event generation (magnitude, duration, random interval)
  - Body drag force: F_drag = 0.5 * ρ * cd * A * |v_rel|² * v_rel_hat
  - Frame transformation: wind to body frame via frames.py boundary

All computations vectorized for (num_envs,) environments.
"""

from __future__ import annotations
import torch
from torch import Tensor
from tvc_env.common.constants import AIR_DENSITY
from tvc_env.common.frames import isaac_velocity_to_frd
from tvc_env.common.quaternions import rotate_vector, inverse as quat_inv, normalize


class WindModel:
    """Wind and atmospheric drag disturbance model."""

    def __init__(
        self,
        steady_vector: list[float] = None,   # m/s, world frame (Isaac convention)
        cd: float = 1.0,                     # body drag coefficient, estimate
        reference_area: float = 0.02,        # m², estimate
        air_density: float = AIR_DENSITY,
        gust_enabled: bool = False,
        gust_magnitude: float = 5.0,         # m/s
        gust_duration: float = 0.5,          # s
        gust_interval_min: float = 5.0,      # s
        gust_interval_max: float = 15.0,     # s
        device: torch.device = None,
    ):
        self.cd = cd
        self.reference_area = reference_area
        self.air_density = air_density
        self.gust_enabled = gust_enabled
        self.gust_magnitude = gust_magnitude
        self.gust_duration = gust_duration
        self.gust_interval_min = gust_interval_min
        self.gust_interval_max = gust_interval_max
        self.device = device

        if steady_vector is None:
            steady_vector = [0.0, 0.0, 0.0]
        self._steady_wind = torch.tensor(steady_vector, dtype=torch.float32, device=device)

        # Gust state
        self._gust_active = False
        self._gust_remaining = 0.0
        self._gust_cooldown = 0.0
        self._gust_direction = torch.zeros(3, device=device)

    @classmethod
    def from_disturbance_config(cls, config: dict, device=None) -> "WindModel":
        """Create WindModel from disturbance config dict."""
        dist = config.get("disturbances", config)
        wind = dist.get("wind", {})
        gust = dist.get("gust", {})
        drag = dist.get("body_drag", {})

        return cls(
            steady_vector=wind.get("steady_vector", [0.0, 0.0, 0.0]),
            cd=drag.get("cd", 1.0),
            reference_area=drag.get("reference_area", 0.02),
            gust_enabled=gust.get("enabled", False),
            gust_magnitude=gust.get("magnitude", 5.0),
            gust_duration=gust.get("duration", 0.5),
            gust_interval_min=gust.get("interval", [5.0, 15.0])[0],
            gust_interval_max=gust.get("interval", [5.0, 15.0])[1],
            device=device,
        )

    def update_gust(self, dt: float) -> None:
        """Update gust state machine (step dt seconds)."""
        if not self.gust_enabled:
            return

        if self._gust_active:
            self._gust_remaining -= dt
            if self._gust_remaining <= 0:
                self._gust_active = False
                self._gust_cooldown = (
                    self.gust_interval_min +
                    torch.rand(1).item() * (self.gust_interval_max - self.gust_interval_min)
                )
        else:
            self._gust_cooldown -= dt
            if self._gust_cooldown <= 0:
                self._gust_active = True
                self._gust_remaining = self.gust_duration
                # Random gust direction in horizontal plane (world frame)
                angle = torch.rand(1).item() * 6.283
                self._gust_direction = torch.tensor(
                    [torch.cos(torch.tensor(angle)).item(), 0.0, torch.sin(torch.tensor(angle)).item()],
                    device=self.device,
                )

    def get_effective_wind_world(self) -> Tensor:
        """Get current total wind vector in Isaac world frame (m/s)."""
        wind = self._steady_wind.clone()
        if self._gust_active:
            wind = wind + self._gust_direction * self.gust_magnitude
        return wind

    def compute_drag_force(
        self,
        linear_vel_world: Tensor,   # (num_envs, 3) in Isaac world frame
        quaternion_wxyz: Tensor,    # (num_envs, 4) body orientation
    ) -> Tensor:
        """Compute aerodynamic drag force in body-FRD frame.

        F_drag = 0.5 * ρ * cd * A * |v_rel|² * v_rel_hat

        where v_rel = v_body - v_wind (in world frame)

        Args:
            linear_vel_world: Body linear velocity in Isaac world frame.
            quaternion_wxyz: Body orientation quaternion.

        Returns:
            Tensor (num_envs, 3) — drag force in body-FRD frame (N).
        """
        wind_world = self.get_effective_wind_world()  # (3,)
        v_rel_world = linear_vel_world - wind_world.unsqueeze(0)  # (num_envs, 3)

        speed_sq = (v_rel_world ** 2).sum(dim=-1, keepdim=True)  # (num_envs, 1)
        speed = speed_sq.sqrt()  # (num_envs, 1)

        # Unit vector opposing relative airflow
        v_rel_hat = v_rel_world / speed.clamp(min=1e-6)  # (num_envs, 3)

        # Drag magnitude
        drag_mag = 0.5 * self.air_density * self.cd * self.reference_area * speed_sq  # (num_envs, 1)

        # Drag force in world frame (opposes relative motion)
        drag_world = -drag_mag * v_rel_hat  # (num_envs, 3)

        # Transform to body-FRD frame
        q_inv = quat_inv(normalize(quaternion_wxyz))
        drag_body_isaac = rotate_vector(q_inv, drag_world)
        drag_body_frd = isaac_velocity_to_frd(drag_body_isaac)

        return drag_body_frd
