"""
Aerodynamic drag model for the combined vehicle shape.

Implements vehicle.md §6.2:
- Relative velocity v_rel = v_b - R.T @ v_wind
- Directional drag (per-axis projected areas weighted by velocity direction)
- Drag force F_aero = -0.5 * rho * |v_rel| * v_rel * Cd * A_eff
- Aerodynamic torque tau_aero = (r_cp - com) × F_aero

Projected areas come from MassProperties (aggregated from primitives).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any, Sequence

import numpy as np

from .mass_properties import MassProperties


def _as_vec3(x: Sequence[float], *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}.")
    return arr


@dataclass(frozen=True, slots=True)
class AeroModelConfig:
    """Configuration for the aerodynamic drag model."""

    Cd: float
    r_cp: np.ndarray  # (3,) m, center of pressure in body frame
    A_proj: float  # m², fallback projected frontal area when directional drag disabled
    compute_directional_drag: bool

    @classmethod
    def from_config(cls, aero_config: Mapping[str, Any]) -> "AeroModelConfig":
        """Build config from YAML aero section."""
        Cd = float(aero_config.get("Cd", 0.7))
        r_cp = _as_vec3(aero_config.get("r_cp", [0.0, 0.0, 0.05]), name="r_cp")
        A_proj = float(aero_config.get("A_proj", 0.01) or 0.01)
        compute_directional = bool(aero_config.get("compute_directional_drag", True))
        return cls(Cd=Cd, r_cp=r_cp, A_proj=A_proj, compute_directional_drag=compute_directional)


class AeroModel:
    """
    Combined-shape aerodynamic drag model.

    Accepts MassProperties for projected_area_{x,y,z} when using directional drag.
    """

    def __init__(
        self,
        config: AeroModelConfig,
        mass_properties: MassProperties,
    ) -> None:
        """
        Initialize the aero model.

        Parameters
        ----------
        config
            Aero config (Cd, r_cp, A_proj, compute_directional_drag).
        mass_properties
            Mass properties providing projected_area_x, projected_area_y,
            projected_area_z and center_of_mass.
        """
        self._config = config
        self._mass_props = mass_properties
        self._proj_x = mass_properties.projected_area_x
        self._proj_y = mass_properties.projected_area_y
        self._proj_z = mass_properties.projected_area_z
        self._com = mass_properties.center_of_mass

    def compute(
        self,
        v_b: np.ndarray,
        R: np.ndarray,
        v_wind: np.ndarray,
        *,
        rho: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute aerodynamic force and torque in body frame.

        Parameters
        ----------
        v_b
            Body-frame velocity (3,) m/s.
        R
            Rotation matrix body→inertial (3,3).
        v_wind
            Wind velocity in inertial frame (3,) m/s.
        rho
            Air density kg/m³ (from EnvironmentModel).

        Returns
        -------
        F_aero
            Drag force (3,) N in body frame.
        tau_aero
            Aerodynamic torque (3,) N·m in body frame.
        """
        # 4.1 Relative velocity: v_rel = v_b - R.T @ v_wind
        v_wind_body = R.T @ np.asarray(v_wind, dtype=float).reshape(3)
        v_rel = np.asarray(v_b, dtype=float).reshape(3) - v_wind_body
        speed_rel = float(np.linalg.norm(v_rel))

        if speed_rel < 1e-12:
            return np.zeros(3), np.zeros(3)

        # 4.2 Directional drag: per-axis projected areas weighted by velocity direction
        if self._config.compute_directional_drag:
            v_hat = np.abs(v_rel) / speed_rel
            A_eff = (
                v_hat[0] * self._proj_x
                + v_hat[1] * self._proj_y
                + v_hat[2] * self._proj_z
            )
        else:
            A_eff = self._config.A_proj

        # Clamp A_eff to avoid numerical issues when all projected areas are zero
        A_eff = max(float(A_eff), 1e-20)

        # 4.3 Drag force: F_aero = -0.5 * rho * |v_rel| * v_rel * Cd * A_eff
        F_aero = -0.5 * rho * speed_rel * v_rel * self._config.Cd * A_eff

        # 4.4 Aero torque: tau_aero = (r_cp - com) × F_aero
        r_offset = self._config.r_cp - self._com
        tau_aero = np.cross(r_offset, F_aero)

        return F_aero, tau_aero
