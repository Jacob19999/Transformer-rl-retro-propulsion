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
        num_envs: int = 1,
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
        self.num_envs = int(num_envs)

        if steady_vector is None:
            steady_vector = [0.0, 0.0, 0.0]
        self._steady_wind = torch.tensor(steady_vector, dtype=torch.float32, device=device).unsqueeze(0).expand(
            self.num_envs, -1
        ).clone()

        # Gust state — sample an initial cooldown so the first gust is delayed.
        self._gust_active = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self._gust_remaining = torch.zeros(self.num_envs, device=device)
        self._gust_cooldown = self._sample_gust_cooldown(self.num_envs) if gust_enabled else torch.zeros(
            self.num_envs, device=device
        )
        self._gust_direction = torch.zeros(self.num_envs, 3, device=device)

    @classmethod
    def from_disturbance_config(cls, config: dict, num_envs: int = 1, device=None) -> "WindModel":
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
            num_envs=num_envs,
            device=device,
        )

    def _sample_gust_cooldown(self, count: int) -> Tensor:
        """Sample independent wait times until the next gust begins."""
        interval_span = max(self.gust_interval_max - self.gust_interval_min, 0.0)
        rand = torch.rand(count, device=self.device)
        return self.gust_interval_min + rand * interval_span

    def reset(self, env_ids: Tensor | None = None) -> None:
        """Reset gust state for newly reset environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int64)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.int64)
        self._gust_active[env_ids] = False
        self._gust_remaining[env_ids] = 0.0
        self._gust_direction[env_ids] = 0.0
        self._gust_cooldown[env_ids] = (
            self._sample_gust_cooldown(len(env_ids)) if self.gust_enabled else 0.0
        )

    def update_gust(self, dt: float) -> None:
        """Update gust state machine (step dt seconds)."""
        if not self.gust_enabled:
            return

        active = self._gust_active
        self._gust_remaining[active] -= dt
        finished = active & (self._gust_remaining <= 0.0)
        if finished.any():
            self._gust_active[finished] = False
            self._gust_cooldown[finished] = self._sample_gust_cooldown(int(finished.sum().item()))

        inactive = ~self._gust_active
        self._gust_cooldown[inactive] -= dt
        starting = inactive & (self._gust_cooldown <= 0.0)
        if starting.any():
            count = int(starting.sum().item())
            angles = torch.rand(count, device=self.device) * (2.0 * torch.pi)
            self._gust_active[starting] = True
            self._gust_remaining[starting] = self.gust_duration
            self._gust_direction[starting, 0] = torch.cos(angles)
            self._gust_direction[starting, 1] = torch.sin(angles)
            self._gust_direction[starting, 2] = 0.0

    def get_effective_wind_world(self) -> Tensor:
        """Get current total wind vector in Isaac world frame (m/s)."""
        return self._steady_wind + self._gust_direction * (
            self._gust_active.float() * self.gust_magnitude
        ).unsqueeze(-1)

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
        wind_world = self.get_effective_wind_world()  # (num_envs, 3)
        if wind_world.shape[0] != linear_vel_world.shape[0]:
            raise ValueError(
                f"Wind batch ({wind_world.shape[0]}) does not match body batch "
                f"({linear_vel_world.shape[0]})."
            )
        v_rel_world = linear_vel_world - wind_world  # (num_envs, 3)

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
