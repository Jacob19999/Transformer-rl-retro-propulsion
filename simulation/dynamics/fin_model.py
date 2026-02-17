"""
Fin forces model for 4 NACA 0012 fins in EDF exhaust.

Implements vehicle.md §6.3 and tracker Stage 5:
- Thin-airfoil lift: C_L = Cl_alpha * alpha_eff with tanh stall soft-clamp at ±15°
- Induced drag: C_D = Cd0 + C_L^2 / (pi * AR)
- Per-fin force: F_k = 0.5 * rho * V_e^2 * A_fin * (C_L * n_L + C_D * n_D)
- Exhaust velocity scaling: V_exhaust = V_exhaust_nominal * (omega_fan / omega_fan_max)
- Mechanical clamp: delta = clip(delta, -delta_max, +delta_max) at ±20°
- Total fin force/torque: sum over 4 fins, tau_fins = Σ (r_fin_k - com) × F_k

All computations are vectorized over 4 fins (no Python loop).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def _as_vec3(x: Sequence[float], *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}.")
    return arr


@dataclass(frozen=True, slots=True)
class FinModelConfig:
    """Configuration for the fin model."""

    Cl_alpha: float  # /rad, thin-airfoil lift slope (2π)
    Cd0: float  # parasitic drag coefficient
    AR: float  # aspect ratio
    stall_angle: float  # rad (15°), onset of C_L plateau
    max_deflection: float  # rad (±20°), mechanical servo travel limit
    planform_area: float  # m² per fin
    V_exhaust_nominal: float  # m/s at full RPM
    omega_fan_max: float  # rad/s, max fan angular velocity
    exhaust_velocity_ratio: bool  # scale V_exhaust with omega_fan/omega_fan_max
    fins_config: tuple[dict[str, Any], ...]  # per-fin: position, lift_direction, drag_direction

    @classmethod
    def from_config(cls, fins_config: Mapping[str, Any], edf_config: Mapping[str, Any]) -> "FinModelConfig":
        """Build config from YAML fins section and EDF section."""
        fins = dict(fins_config)
        edf = dict(edf_config)

        Cl_alpha = float(fins.get("Cl_alpha", 2 * np.pi))
        Cd0 = float(fins.get("Cd0", 0.01))
        AR = float(fins.get("AR", 0.80))
        stall_angle = float(fins.get("stall_angle", np.deg2rad(15)))
        max_deflection = float(fins.get("max_deflection", np.deg2rad(20)))
        planform_area = float(fins.get("planform_area", 0.002))
        V_exhaust_nominal = float(fins.get("V_exhaust_nominal", 70.0))
        exhaust_velocity_ratio = bool(fins.get("exhaust_velocity_ratio", True))
        omega_fan_max = float(edf.get("max_omega", 9948.0))

        fcfg = list(fins.get("fins_config", []))
        for i, fc in enumerate(fcfg):
            if "position" not in fc or "lift_direction" not in fc or "drag_direction" not in fc:
                raise ValueError(f"fin {i}: fins_config must have position, lift_direction, drag_direction")
        fins_config_tuple = tuple(fcfg)

        return cls(
            Cl_alpha=Cl_alpha,
            Cd0=Cd0,
            AR=AR,
            stall_angle=stall_angle,
            max_deflection=max_deflection,
            planform_area=planform_area,
            V_exhaust_nominal=V_exhaust_nominal,
            omega_fan_max=omega_fan_max,
            exhaust_velocity_ratio=exhaust_velocity_ratio,
            fins_config=fins_config_tuple,
        )


class FinModel:
    """
    4 NACA 0012 fins in exhaust, thin-airfoil coefficients, stall soft-clamp.

    Accepts center_of_mass from MassProperties for torque computation.
    """

    def __init__(
        self,
        config: FinModelConfig,
        center_of_mass: np.ndarray,
    ) -> None:
        """
        Initialize the fin model.

        Parameters
        ----------
        config
            Fin model config (Cl_alpha, Cd0, AR, stall_angle, etc.).
        center_of_mass
            CoM (3,) in body frame for torque computation.
        """
        self._config = config
        self._com = _as_vec3(center_of_mass, name="center_of_mass")

        # Pre-extract per-fin arrays for vectorization: (4, 3) each
        n = len(config.fins_config)
        self._positions = np.array([_as_vec3(fc["position"], name=f"fin{i}_position") for i, fc in enumerate(config.fins_config)])
        self._lift_dirs = np.array([_as_vec3(fc["lift_direction"], name=f"fin{i}_lift") for i, fc in enumerate(config.fins_config)])
        self._drag_dirs = np.array([_as_vec3(fc["drag_direction"], name=f"fin{i}_drag") for i, fc in enumerate(config.fins_config)])

        if self._positions.shape[0] != n or self._lift_dirs.shape[0] != n or self._drag_dirs.shape[0] != n:
            raise ValueError(f"Expected {n} fins, got {self._positions.shape[0]}")

    @classmethod
    def from_config(
        cls,
        fins_config: Mapping[str, Any],
        edf_config: Mapping[str, Any],
        center_of_mass: np.ndarray,
    ) -> "FinModel":
        """Build FinModel from vehicle YAML sections."""
        cfg = FinModelConfig.from_config(fins_config, edf_config)
        return cls(cfg, center_of_mass)

    def _exhaust_velocity(self, omega_fan: float) -> float:
        """Exhaust velocity scaling: V_exhaust = V_exhaust_nominal * (omega_fan / omega_fan_max)."""
        if not self._config.exhaust_velocity_ratio:
            return self._config.V_exhaust_nominal
        omega = float(np.clip(omega_fan, 0.0, self._config.omega_fan_max))
        ratio = omega / float(self._config.omega_fan_max)
        return float(self._config.V_exhaust_nominal * ratio)

    def _lift_coefficient(self, alpha_eff: np.ndarray) -> np.ndarray:
        """C_L = Cl_alpha * alpha_eff (thin-airfoil, alpha_eff already stall-soft-clamped)."""
        return self._config.Cl_alpha * alpha_eff

    def _drag_coefficient(self, C_L: np.ndarray) -> np.ndarray:
        """Induced drag: C_D = Cd0 + C_L^2 / (pi * AR)."""
        return self._config.Cd0 + (C_L ** 2) / (np.pi * self._config.AR)

    def compute(
        self,
        delta_actual: np.ndarray,
        omega_fan: float,
        *,
        rho: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute total fin force and torque in body frame.

        Parameters
        ----------
        delta_actual
            Actual fin deflections (4,) in rad (servo-filtered).
        omega_fan
            Fan angular velocity (rad/s).
        rho
            Air density kg/m³ (from EnvironmentModel).

        Returns
        -------
        F_fins
            Total fin force (3,) N in body frame.
        tau_fins
            Total fin torque (3,) N·m in body frame.
        """
        delta = np.asarray(delta_actual, dtype=float).ravel()
        if delta.shape[0] != len(self._config.fins_config):
            raise ValueError(f"delta_actual must have {len(self._config.fins_config)} elements, got {delta.shape[0]}")

        # 5.5 Mechanical clamp: delta = clip(delta, -delta_max, +delta_max)
        delta_clipped = np.clip(delta, -self._config.max_deflection, self._config.max_deflection)

        # 5.1 Stall soft-clamp: alpha_eff = stall_angle * tanh(alpha / stall_angle)
        # For exhaust flow, alpha_k ≈ delta_k (deflection = angle of attack)
        alpha_eff = self._config.stall_angle * np.tanh(delta_clipped / self._config.stall_angle)

        # 5.1 C_L = Cl_alpha * alpha_eff
        C_L = self._lift_coefficient(alpha_eff)

        # 5.2 C_D = Cd0 + C_L^2 / (pi * AR)
        C_D = self._drag_coefficient(C_L)

        # 5.4 Exhaust velocity scaling
        V_e = self._exhaust_velocity(omega_fan)

        # 5.3 Per-fin force: F_k = 0.5 * rho * V_e^2 * A_fin * (C_L * n_L + C_D * n_D)
        q_dyn = 0.5 * rho * (V_e ** 2) * self._config.planform_area
        # (4,) * (4,3) -> (4,3) broadcast; same for C_D
        F_per_fin = q_dyn * (C_L[:, np.newaxis] * self._lift_dirs + C_D[:, np.newaxis] * self._drag_dirs)

        # 5.6 Total force
        F_fins = np.sum(F_per_fin, axis=0)

        # 5.6 Total torque: tau_fins = Σ (r_fin_k - com) × F_k
        r_offsets = self._positions - self._com  # (4, 3)
        tau_per_fin = np.cross(r_offsets, F_per_fin)  # (4, 3)
        tau_fins = np.sum(tau_per_fin, axis=0)

        return F_fins.astype(float), tau_fins.astype(float)
