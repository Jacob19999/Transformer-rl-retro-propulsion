"""
isaac_wind_model.py — GPU-batched wind disturbance model for Isaac Sim environments.

Implements configurable mean wind + discrete gust events applied as aerodynamic
drag forces. All state tensors are on the same device as the Isaac Sim simulation
(GPU when available). Wind parameters are loaded from ``default_environment.yaml``.

Wind force model (simplified from aero_model.py):
    v_rel = v_wind - v_body          (relative wind in world frame, m/s)
    q_dyn = 0.5 * rho * |v_rel|^2   (dynamic pressure, Pa)
    F_drag = q_dyn * Cd * A * unit(v_rel)  per axis

IsaacWindConfig (data-model.md):
    enabled            bool    — whether wind is active
    mean_vector_range_lo/hi     — per-axis uniform sampling bounds (m/s)
    gust_prob          float   — per-episode gust probability
    gust_magnitude_range        — [min, max] gust speed (m/s)
    drag_coefficient   float   — composite Cd (default 1.0)
    projected_area     list[3] — Cd*A per axis (m²)
"""

from __future__ import annotations

import math
from typing import Any

import torch


class IsaacWindModel:
    """GPU-batched wind model for Isaac Sim EDF landing environments.

    Manages per-environment wind state including mean wind vector, optional
    discrete gust events, and an exponential moving average (EMA) of the
    wind vector for the observation pipeline.

    Usage::
        wind = IsaacWindModel(env_cfg["isaac_wind"], num_envs, device)
        wind.reset(all_env_ids)

        # Inside _apply_action():
        wind_vec = wind.step(dt)
        F_wind = wind.compute_drag_force(wind_vec, body_velocity_world)
        forces += F_wind
    """

    def __init__(self, config: dict[str, Any], num_envs: int, device: torch.device) -> None:
        """Initialize wind model from environment YAML ``isaac_wind`` section.

        Args:
            config: Dict from ``default_environment.yaml`` → ``isaac_wind`` section.
            num_envs: Number of parallel environments.
            device: PyTorch device (typically ``cuda:0`` or ``cpu``).
        """
        self._num_envs = num_envs
        self._device   = device
        self._enabled  = bool(config.get("enabled", False))

        # Wind sampling bounds (m/s per axis)
        lo = config.get("mean_vector_range_lo", [-10.0, -10.0, -2.0])
        hi = config.get("mean_vector_range_hi", [ 10.0,  10.0,  2.0])
        self._lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self._hi = torch.tensor(hi, dtype=torch.float32, device=device)

        # Gust parameters
        self._gust_prob  = float(config.get("gust_prob", 0.1))
        gust_range       = config.get("gust_magnitude_range", [3.0, 8.0])
        self._gust_mag_lo = float(gust_range[0])
        self._gust_mag_hi = float(gust_range[1])

        # Aerodynamic drag parameters
        self._rho   = float(config.get("air_density", 1.225))  # kg/m³ sea-level
        Cd          = float(config.get("drag_coefficient", 1.0))
        proj_area   = config.get("projected_area", [0.01, 0.01, 0.02])
        self._CdA   = torch.tensor(
            [Cd * proj_area[0], Cd * proj_area[1], Cd * proj_area[2]],
            dtype=torch.float32, device=device,
        )

        # EMA time constant (s) for observation smoothing
        self._tau_ema = float(config.get("wind_ema_tau", 0.5))

        # Episode length for gust timing (steps at 1/120 s)
        ep_len_s = float(config.get("episode_duration", 5.0))
        self._ep_len_steps = int(ep_len_s * 120)

        # State tensors  (IsaacWindState from data-model.md)
        self._mean_wind   = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
        self._wind_ema    = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
        self._gust_active = torch.zeros(num_envs, dtype=torch.bool,    device=device)
        self._gust_onset  = torch.zeros(num_envs, dtype=torch.int32,   device=device)
        self._gust_dur    = torch.zeros(num_envs, dtype=torch.int32,   device=device)
        self._gust_vec    = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
        self._step_count  = torch.zeros(num_envs, dtype=torch.int32,   device=device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether wind is active for this environment instance."""
        return self._enabled

    @property
    def wind_ema(self) -> torch.Tensor:
        """Exponential moving average of wind vector for observation vector.

        Shape: (num_envs, 3), world frame (m/s).
        """
        return self._wind_ema

    def reset(self, env_ids: torch.Tensor) -> None:
        """Sample new wind conditions for the specified environments.

        Called at episode reset. Samples fresh mean wind from uniform bounds
        and optionally schedules a gust event.

        Args:
            env_ids: 1-D int tensor of environment indices to reset.
        """
        n = env_ids.numel()
        if n == 0:
            return

        # Sample mean wind: uniform per axis within [lo, hi]
        lo = self._lo.unsqueeze(0).expand(n, -1)  # (n, 3)
        hi = self._hi.unsqueeze(0).expand(n, -1)
        mean_wind = lo + (hi - lo) * torch.rand((n, 3), device=self._device)
        self._mean_wind[env_ids] = mean_wind

        # Reset EMA
        self._wind_ema[env_ids] = torch.zeros((n, 3), device=self._device)
        self._step_count[env_ids] = 0

        # Sample gust event (Bernoulli)
        has_gust = torch.rand(n, device=self._device) < self._gust_prob
        self._gust_active[env_ids] = False

        for i, (eid, gust) in enumerate(zip(env_ids.tolist(), has_gust.tolist())):
            if gust:
                # Random gust magnitude and direction (spherical uniform)
                mag  = self._gust_mag_lo + (self._gust_mag_hi - self._gust_mag_lo) * torch.rand(1).item()
                phi  = 2.0 * math.pi * torch.rand(1).item()
                gust_vec = torch.tensor(
                    [mag * math.cos(phi), mag * math.sin(phi), 0.0],
                    dtype=torch.float32, device=self._device,
                )
                # Gust timing: onset 20–80% through episode, duration 0.5–2 s
                onset_frac = 0.2 + 0.6 * torch.rand(1).item()
                dur_s      = 0.5 + 1.5 * torch.rand(1).item()
                onset_step = int(onset_frac * self._ep_len_steps)
                dur_steps  = max(1, int(dur_s * 120))

                self._gust_vec[eid]   = gust_vec
                self._gust_onset[eid] = onset_step
                self._gust_dur[eid]   = dur_steps

    def step(self, dt: float) -> torch.Tensor:
        """Advance wind state by one simulation step.

        Updates gust activity based on current step counter, computes total
        wind vector, and updates the wind EMA.

        Args:
            dt: Simulation timestep in seconds (typically 1/120 s).

        Returns:
            wind_vector: (num_envs, 3) tensor of current wind velocity in
                world frame (m/s).
        """
        self._step_count += 1

        # Determine which envs have an active gust
        step = self._step_count  # (num_envs,)
        gust_start  = self._gust_onset           # (num_envs,)
        gust_end    = self._gust_onset + self._gust_dur
        gust_active = (step >= gust_start) & (step < gust_end)
        self._gust_active = gust_active

        # Total wind = mean + gust (where active)
        gust_contrib = self._gust_vec * gust_active.unsqueeze(-1).float()
        wind_vec = self._mean_wind + gust_contrib  # (num_envs, 3)

        # Update EMA: α = dt / (dt + τ)
        alpha = dt / (dt + self._tau_ema)
        self._wind_ema = (1.0 - alpha) * self._wind_ema + alpha * wind_vec

        return wind_vec

    def compute_drag_force(
        self,
        wind_vector: torch.Tensor,
        body_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute aerodynamic drag force from relative wind.

        Uses simplified body-drag model:
            v_rel = wind - body_vel
            F = 0.5 * rho * |v_rel|^2 * CdA * unit(v_rel)   (per axis)

        This applies per-axis drag independently, consistent with the
        directional drag model in aero_model.py.

        Args:
            wind_vector:   (num_envs, 3) current wind velocity, world frame (m/s).
            body_velocity: (num_envs, 3) drone body velocity, world frame (m/s).

        Returns:
            F_drag: (num_envs, 3) aerodynamic drag force in world frame (N).
        """
        v_rel = wind_vector - body_velocity  # (num_envs, 3)
        v_sq  = v_rel ** 2                   # element-wise square
        sign  = torch.sign(v_rel)

        # Per-axis: F_i = 0.5 * rho * v_rel_i^2 * CdA_i * sign(v_rel_i)
        F_drag = 0.5 * self._rho * v_sq * self._CdA.unsqueeze(0) * sign  # (num_envs, 3)
        return F_drag

    def set_constant_wind(self, wind_xyz: tuple[float, float, float]) -> None:
        """Override all environments with a fixed constant wind vector.

        Intended for diagnostic scripts (diag_wind.py) that need a specific
        wind value rather than random sampling.

        Args:
            wind_xyz: (wx, wy, wz) wind velocity in world frame (m/s).
        """
        vec = torch.tensor(wind_xyz, dtype=torch.float32, device=self._device)
        self._mean_wind[:] = vec.unsqueeze(0).expand(self._num_envs, -1)
        self._gust_active[:] = False
        self._gust_vec[:] = 0.0
