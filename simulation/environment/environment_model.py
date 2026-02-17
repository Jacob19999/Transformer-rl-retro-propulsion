"""
Top-level environment model: composes AtmosphereModel + WindModel.

Implements env.md §2 and tracker Stage 10.

Coordinate convention:
    - Vehicle position `p` is inertial NED [m].
    - Altitude above ground is computed as h = -p[2].
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from simulation.environment.atmosphere_model import AtmosphereModel, AtmosphereModelConfig
from simulation.environment.wind_model import WindModel


class EnvironmentModel:
    """Environment assembly model (wind + atmosphere)."""

    def __init__(self, config: Mapping[str, Any], *, seed: int | None = None) -> None:
        """Create an EnvironmentModel.

        Args:
            config: Mapping containing `atmosphere` and `wind` subsections. If a top-level
                `environment` key is present, it will be used as the root subsection.
            seed: Optional seed used for episode-level reproducibility via `reset(seed)`.
        """
        cfg_root: Mapping[str, Any] = config.get("environment", config)  # type: ignore[arg-type]

        atmosphere_cfg = cfg_root.get("atmosphere", {})
        wind_cfg = cfg_root.get("wind", {})

        # Initialize with independent RNGs (replaced on reset()).
        self._rng_seed: int | None = None
        self.atmosphere = AtmosphereModel(AtmosphereModelConfig.from_config(atmosphere_cfg))
        self.wind = WindModel(wind_cfg)

        if seed is not None:
            self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        """Reset sub-models for a new episode, ensuring reproducibility.

        Uses independent child RNG streams for atmosphere and wind to avoid coupling
        determinism to call ordering across components.
        """
        if seed is None:
            # Preserve current seed if set, otherwise allow stochastic resets.
            seed = self._rng_seed

        if seed is None:
            # Unseeded: create fresh RNG streams each reset.
            ss = np.random.SeedSequence()
        else:
            self._rng_seed = int(seed)
            ss = np.random.SeedSequence(self._rng_seed)

        ss_atm, ss_wind = ss.spawn(2)
        self.atmosphere.rng = np.random.default_rng(ss_atm)  # type: ignore[assignment]
        self.wind.rng = np.random.default_rng(ss_wind)  # type: ignore[assignment]

        self.atmosphere.reset()
        self.wind.reset()

    def sample_at_state(self, t: float, p: np.ndarray) -> dict[str, Any]:
        """Sample environment at time `t` and inertial position `p` (NED).

        Args:
            t: Time [s].
            p: (3,) inertial position in NED [m].

        Returns:
            Dict with keys:
                - 'wind': (3,) wind vector in NED [m/s]
                - 'T': temperature [K]
                - 'P': pressure [Pa]
                - 'rho': density [kg/m^3]
        """
        p_arr = np.asarray(p, dtype=float).reshape(3)
        h = float(-p_arr[2])
        # Guard against below-ground queries (can happen during contact/termination).
        h = max(h, 0.0)

        T, P, rho = self.atmosphere.get_conditions(h)
        v_wind = self.wind.sample(float(t), h)

        return {
            "wind": np.asarray(v_wind, dtype=float).reshape(3),
            "T": float(T),
            "P": float(P),
            "rho": float(rho),
        }

