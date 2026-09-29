"""
Wind and drag disturbance model.

Implements:
  - Steady wind vector in world frame
  - Gust event generation (magnitude, duration, random interval)
  - Body drag force: F_drag = 0.5 * ρ * cd * A * |v_rel|² * v_rel_hat, or with
    a lateral area the body-axis split (independence principle)
      F_axial   = -0.5 ρ cd_axial A_axial |v_z| v_z
      F_lateral = -0.5 ρ cd_lat A_lat |v_xy| v_xy
    A slender body is ~4x larger side-on than end-on (0.35 x 0.12 m: 0.042
    vs 0.011 m²); one isotropic area under- or over-states one of them.
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
        lateral_area: float | None = None,   # m², side-on area; None: isotropic cd * reference_area
        cd_lateral: float | None = None,     # crossflow drag coefficient (default: cd)
    ):
        self.cd = cd
        self.reference_area = reference_area
        self.lateral_area = lateral_area
        self.cd_lateral = cd if cd_lateral is None else cd_lateral
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

    @staticmethod
    def body_drag_from_vehicle(body: dict) -> dict:
        """Axial and side-on drag of the vehicle body (configs/vehicle ``body`` section).

        ``reference_area`` is the end-on (axial) area; the side-on area is
        ``lateral_reference_area``, else length x diameter.
        """
        length, diameter = float(body.get("length", 0.35)), float(body.get("diameter", 0.12))
        cd = float(body.get("cd_body", 1.0))
        return dict(cd=cd, reference_area=float(body.get("reference_area", 0.011)),
                    lateral_area=float(body.get("lateral_reference_area", length * diameter)),
                    cd_lateral=float(body.get("cd_lateral", cd)))

    @classmethod
    def from_disturbance_config(cls, config: dict, num_envs: int = 1, device=None,
                                body: dict | None = None) -> "WindModel":
        """Create WindModel from disturbance config dict.

        With the vehicle ``body`` geometry the body drag comes from it, the
        same in calm air and in wind (a disturbance file used to swap the
        vehicle's 0.011 m² for an isotropic 0.02 m²). Without it the legacy
        isotropic ``body_drag`` section applies.
        """
        dist = config.get("disturbances", config)
        wind = dist.get("wind", {})
        gust = dist.get("gust", {})
        enabled = dist.get("enabled", True)
        if body is not None:
            drag = cls.body_drag_from_vehicle(body)
        else:
            legacy = dist.get("body_drag", {})
            drag = dict(cd=legacy.get("cd", 1.0), reference_area=legacy.get("reference_area", 0.02))

        return cls(
            steady_vector=wind.get("steady_vector", [0.0, 0.0, 0.0])
                if enabled and wind.get("enabled", True) else [0.0, 0.0, 0.0],
            **drag,
            gust_enabled=enabled and gust.get("enabled", False),
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
        if self.lateral_area is not None:
            return self._body_axis_drag(v_rel_world, quaternion_wxyz)

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

    def _body_axis_drag(self, v_rel_world: Tensor, quaternion_wxyz: Tensor) -> Tensor:
        """Axial and crossflow drag, each quadratic in its own body-frame component (body FRD)."""
        v = isaac_velocity_to_frd(rotate_vector(quat_inv(normalize(quaternion_wxyz)), v_rel_world))
        half_rho = 0.5 * self.air_density
        force = torch.empty_like(v)
        lateral = v[:, :2]
        force[:, :2] = -half_rho * self.cd_lateral * self.lateral_area * lateral.norm(dim=-1, keepdim=True) * lateral
        force[:, 2] = -half_rho * self.cd * self.reference_area * v[:, 2].abs() * v[:, 2]
        return force
