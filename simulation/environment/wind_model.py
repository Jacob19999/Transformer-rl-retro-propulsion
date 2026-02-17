"""
Wind disturbance model: mean wind + Dryden turbulence + discrete gusts.

Implements env.md §3 and tracker Stage 9.

Wind components:
    v_wind(t, h) = v_mean + v_turb(t, h) + v_gust(t)
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class DrydenFilter:
    """Discrete-time Dryden turbulence filter (3-axis).

    env.md §3.3.4. Generates u_g, v_g, w_g turbulence components
    from white noise via altitude-dependent shaping filters.
    """

    def __init__(self, dt: float, V_ref: float, config: Mapping[str, Any]) -> None:
        """Initialize Dryden filter.

        Args:
            dt: Integration timestep [s].
            V_ref: Reference airspeed for filter design [m/s].
            config: Wind config dict with 'turbulence_intensity'.
        """
        self.dt = float(dt)
        self.V_ref = float(V_ref)
        self.intensity = float(config.get("turbulence_intensity", 0.3))

        self.state_u: float = 0.0
        self.state_v: np.ndarray = np.zeros(2)
        self.state_w: np.ndarray = np.zeros(2)

    def _compute_params(self, h: float) -> tuple[float, float, float, float, float, float]:
        """Compute altitude-dependent Dryden scale lengths and intensities.

        env.md §3.3.3. Clamp h >= 0.5 m to avoid singularities.

        Returns:
            (L_u, L_v, L_w, sigma_u, sigma_v, sigma_w)
        """
        h_clamped = max(float(h), 0.5)
        denom = (0.177 + 0.000823 * h_clamped) ** 1.2
        L_u = L_v = h_clamped / denom
        L_w = h_clamped

        sigma_w = self.intensity
        sigma_u = sigma_v = sigma_w / ((0.177 + 0.000823 * h_clamped) ** 0.4)

        return L_u, L_v, L_w, sigma_u, sigma_v, sigma_w

    def step(self, h: float, white_noise: np.ndarray) -> np.ndarray:
        """Advance filter one step. Returns (u_g, v_g, w_g) turbulence.

        env.md §3.3.4.

        Args:
            h: Altitude above ground [m].
            white_noise: (3,) white noise input for u, v, w axes.

        Returns:
            (3,) array [u_g, v_g, w_g] turbulence in inertial NED.
        """
        L_u, L_v, L_w, sigma_u, sigma_v, sigma_w = self._compute_params(h)

        V = max(self.V_ref, 1.0)  # avoid division by zero

        # Longitudinal (1st-order)
        alpha_u = self.dt * V / L_u
        self.state_u = (1 - alpha_u) * self.state_u + sigma_u * np.sqrt(2 * alpha_u) * white_noise[0]

        # Lateral (2nd-order, simplified discrete approximation)
        alpha_v = self.dt * V / L_v
        self.state_v[0] = (1 - alpha_v) * self.state_v[0] + alpha_v * white_noise[1]
        self.state_v[1] = (1 - alpha_v) * self.state_v[1] + alpha_v * self.state_v[0]
        v_g = sigma_v * (self.state_v[0] + self.state_v[1]) * np.sqrt(L_v / (np.pi * V))

        # Vertical (2nd-order, same structure)
        alpha_w = self.dt * V / L_w
        self.state_w[0] = (1 - alpha_w) * self.state_w[0] + alpha_w * white_noise[2]
        self.state_w[1] = (1 - alpha_w) * self.state_w[1] + alpha_w * self.state_w[0]
        w_g = sigma_w * (self.state_w[0] + self.state_w[1]) * np.sqrt(L_w / (np.pi * V))

        return np.array([self.state_u, v_g, w_g])

    def reset(self) -> None:
        """Zero all filter states."""
        self.state_u = 0.0
        self.state_v.fill(0.0)
        self.state_w.fill(0.0)


class WindModel:
    """Wind disturbance model: mean + Dryden turbulence + discrete gusts.

    env.md §3.5.
    """

    def __init__(self, config: Mapping[str, Any], *, rng: np.random.Generator | None = None) -> None:
        """Initialize WindModel.

        Args:
            config: Wind config with dt, V_ref, mean_vector_range_lo/hi,
                turbulence_intensity, gust_prob, gust_magnitude_range,
                episode_duration.
            rng: Random generator. If None, creates default_rng().
        """
        self.config = dict(config)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.dryden = DrydenFilter(
            dt=self.config.get("dt", 0.005),
            V_ref=self.config.get("V_ref", 10.0),
            config=self.config,
        )

        self.mean_wind: np.ndarray = np.zeros(3)
        self.gust_active: bool = False
        self.gust_onset: float = 0.0
        self.gust_duration: float = 0.0
        self.gust_amplitude: np.ndarray = np.zeros(3)

    def reset(self) -> None:
        """Re-sample mean wind and gust params for new episode. Reset Dryden."""
        self.mean_wind = self._sample_mean_wind()
        self.dryden.reset()
        self._setup_gust()

    def sample(self, t: float, h: float) -> np.ndarray:
        """Return inertial wind vector at time t, altitude h.

        v_wind = v_mean + v_turb + v_gust

        Args:
            t: Time [s].
            h: Altitude above ground [m].

        Returns:
            (3,) wind velocity in NED [m/s].
        """
        noise = self.rng.standard_normal(3)
        turb = self.dryden.step(h, noise)
        gust = self._sample_gust(t)
        return self.mean_wind + turb + gust

    def _sample_mean_wind(self) -> np.ndarray:
        """Sample mean wind uniformly per episode. env.md §3.2."""
        lo = np.array(self.config["mean_vector_range_lo"], dtype=float)
        hi = np.array(self.config["mean_vector_range_hi"], dtype=float)
        return self.rng.uniform(lo, hi)

    def _setup_gust(self) -> None:
        """Bernoulli event for gust; if active, sample amplitude, onset, duration. env.md §3.4."""
        gust_prob = float(self.config.get("gust_prob", 0.1))
        self.gust_active = self.rng.random() < gust_prob

        if self.gust_active:
            mag_lo, mag_hi = self.config.get("gust_magnitude_range", [3.0, 8.0])
            mag = float(self.rng.uniform(mag_lo, mag_hi))

            direction = self.rng.standard_normal(3)
            direction[2] *= 0.3  # horizontal bias
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                direction /= norm
            self.gust_amplitude = mag * direction

            ep_duration = float(self.config.get("episode_duration", 15.0))
            self.gust_onset = float(self.rng.uniform(0.2 * ep_duration, 0.8 * ep_duration))
            self.gust_duration = float(self.rng.uniform(0.5, 2.0))
        else:
            self.gust_amplitude = np.zeros(3)
            self.gust_onset = 0.0
            self.gust_duration = 0.0

    def _sample_gust(self, t: float) -> np.ndarray:
        """Return gust contribution at time t (step-function)."""
        if not self.gust_active:
            return np.zeros(3)
        if self.gust_onset <= t < self.gust_onset + self.gust_duration:
            return self.gust_amplitude.copy()
        return np.zeros(3)
