"""
ISA-based atmosphere model with per-episode randomization.

Implements env.md §4 and tracker Stage 8.

Computes temperature, pressure, and density at altitude h [m] above ground using:
    T(h)   = T_base + lapse * h
    P(h)   = P_base * (T(h)/T_base) ** (-g/(R*lapse))
    rho(h) = P(h) / (R*T(h))

The main value for this project is per-episode randomization of (T_base, P_base) for
domain randomization; altitude lapse over 0–10 m is negligible but kept for correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class AtmosphereModelConfig:
    """Immutable configuration for AtmosphereModel."""

    T_base: float = 288.15  # K
    T_lapse: float = -0.0065  # K/m (troposphere)
    P_base: float = 101325.0  # Pa
    rho_ref: float = 1.225  # kg/m^3 (reference for thrust scaling)
    randomize_T: float = 10.0  # +/- K uniform per episode
    randomize_P: float = 2000.0  # +/- Pa uniform per episode

    def __post_init__(self) -> None:
        T0 = float(self.T_base)
        P0 = float(self.P_base)
        lapse = float(self.T_lapse)
        rho_ref = float(self.rho_ref)
        dT = float(self.randomize_T)
        dP = float(self.randomize_P)

        if T0 <= 0.0:
            raise ValueError(f"T_base must be > 0 K, got {self.T_base}.")
        if P0 <= 0.0:
            raise ValueError(f"P_base must be > 0 Pa, got {self.P_base}.")
        if lapse == 0.0:
            raise ValueError("T_lapse must be non-zero for the power-law pressure formula.")
        if rho_ref <= 0.0:
            raise ValueError(f"rho_ref must be > 0, got {self.rho_ref}.")
        if dT < 0.0:
            raise ValueError(f"randomize_T must be >= 0, got {self.randomize_T}.")
        if dP < 0.0:
            raise ValueError(f"randomize_P must be >= 0, got {self.randomize_P}.")

        object.__setattr__(self, "T_base", T0)
        object.__setattr__(self, "P_base", P0)
        object.__setattr__(self, "T_lapse", lapse)
        object.__setattr__(self, "rho_ref", rho_ref)
        object.__setattr__(self, "randomize_T", dT)
        object.__setattr__(self, "randomize_P", dP)

    @classmethod
    def from_config(cls, atmosphere: Mapping[str, Any]) -> "AtmosphereModelConfig":
        """Build config from the `environment.atmosphere` YAML section."""
        a = dict(atmosphere)
        return cls(
            T_base=float(a.get("T_base", 288.15)),
            T_lapse=float(a.get("T_lapse", -0.0065)),
            P_base=float(a.get("P_base", 101325.0)),
            rho_ref=float(a.get("rho_ref", 1.225)),
            randomize_T=float(a.get("randomize_T", 10.0)),
            randomize_P=float(a.get("randomize_P", 2000.0)),
        )


class AtmosphereModel:
    """ISA baseline atmosphere with episode-level base condition randomization."""

    # Physical constants (dry air)
    R: float = 287.058  # J/(kg·K)
    g: float = 9.81  # m/s^2

    def __init__(self, config: AtmosphereModelConfig, *, rng: np.random.Generator | None = None) -> None:
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()

        # Episode-varying base conditions; set to nominal until reset() is called.
        self.T_base: float = float(config.T_base)
        self.P_base: float = float(config.P_base)

        # Precompute exponent: -g/(R*lapse) (≈ 5.256 for ISA troposphere lapse).
        self._exponent: float = float(-self.g / (self.R * float(config.T_lapse)))

    @classmethod
    def from_config(cls, atmosphere: Mapping[str, Any], *, rng: np.random.Generator | None = None) -> "AtmosphereModel":
        """Build AtmosphereModel from the `environment.atmosphere` YAML section."""
        return cls(AtmosphereModelConfig.from_config(atmosphere), rng=rng)

    def reset(self) -> None:
        """Randomize base atmosphere for a new episode."""
        dT = float(self.config.randomize_T)
        dP = float(self.config.randomize_P)

        self.T_base = float(self.config.T_base) + float(self.rng.uniform(-dT, dT))
        self.P_base = float(self.config.P_base) + float(self.rng.uniform(-dP, dP))

    def get_conditions(self, h: float) -> tuple[float, float, float]:
        """Compute (T, P, rho) at altitude h [m] above ground."""
        h_f = float(h)

        T = float(self.T_base + float(self.config.T_lapse) * h_f)
        if T <= 0.0:
            raise ValueError(f"Computed temperature must be > 0 K, got T={T} at h={h_f}.")

        # Troposphere power-law pressure with lapse. (Matches env.md §4.3.)
        P = float(self.P_base * (T / float(self.T_base)) ** self._exponent)
        rho = float(P / (self.R * T))
        return T, P, rho

    @property
    def rho_ref(self) -> float:
        """Reference density for thrust normalization."""
        return float(self.config.rho_ref)

