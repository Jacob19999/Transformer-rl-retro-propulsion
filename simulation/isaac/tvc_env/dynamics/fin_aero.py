"""
Semi-empirical jet-vane aerodynamic force model for EDF fins.

Implements subsonic vane aero model per research decision R7:
  F_n = q*S*C_N(α) — normal force with saturation
  F_t = q*S*C_D(α) — tangential (drag) force

Where:
  q = 0.5 * rho * v_exhaust^2  — dynamic pressure from EDF exhaust
  S = fin area
  C_N(α) = C_N_alpha * α * (1 - k_sat * α²)  — lift curve with saturation
  C_D(α) = C_D_0 + C_D_alpha2 * α²            — drag vs angle squared

All computations are vectorized for (num_envs, 4) fin arrays.
Forces are in fin-local frame (normal to fin surface, tangential in fin plane).
"""

from __future__ import annotations
import torch
from torch import Tensor
import math
from tvc_env.common.datatypes import FinForceResult
from tvc_env.common.constants import AIR_DENSITY


class FinAeroModel:
    """Semi-empirical jet-vane aerodynamic force model.

    Parameters follow config_schema convention with source labels.
    """

    def __init__(
        self,
        fin_area: float,                # m², source: measured
        max_deflection: float,          # rad, source: measured
        C_N_alpha: float = 3.5,         # 1/rad, source: estimate (thin airfoil theory ≈ 2π)
        k_sat: float = 2.0,             # saturation coefficient, source: estimate
        C_D_0: float = 0.05,            # zero-deflection drag, source: estimate
        C_D_alpha2: float = 1.5,        # drag coefficient for α², source: estimate
        exhaust_speed: float = 40.0,    # m/s at nominal throttle, source: estimate
        duct_confinement_factor: float = 1.3,  # correction for duct confinement, estimate
        air_density: float = AIR_DENSITY,
    ):
        self.fin_area = fin_area
        self.max_deflection = max_deflection
        self.C_N_alpha = C_N_alpha
        self.k_sat = k_sat
        self.C_D_0 = C_D_0
        self.C_D_alpha2 = C_D_alpha2
        self.exhaust_speed = exhaust_speed
        self.duct_confinement_factor = duct_confinement_factor
        self.air_density = air_density

        # Precompute base dynamic pressure at nominal throttle
        self._q_base = 0.5 * air_density * exhaust_speed ** 2

    @classmethod
    def from_config(cls, vehicle_config: dict, edf_config: dict) -> "FinAeroModel":
        """Create FinAeroModel from vehicle and EDF YAML config dicts."""
        fins = vehicle_config.get("fins", {})
        edf = edf_config.get("edf", edf_config)
        return cls(
            fin_area=fins.get("area", 0.002),
            max_deflection=fins.get("max_deflection", 0.262),
            exhaust_speed=edf.get("exhaust_speed_nominal", 40.0),
        )

    def compute_forces(
        self,
        fin_angles: Tensor,
        throttle_fraction: Tensor,
    ) -> FinForceResult:
        """Compute aerodynamic forces for all fins in all environments.

        Args:
            fin_angles: Tensor of shape (num_envs, 4) — actual fin deflection angles (rad).
            throttle_fraction: Tensor of shape (num_envs,) — current throttle [0, 1].

        Returns:
            FinForceResult with forces in fin-local frame.
        """
        # Dynamic pressure scales with throttle² (thrust ∝ ω², exhaust speed ∝ ω, q ∝ ω²)
        q = self._q_base * throttle_fraction.unsqueeze(-1) ** 2  # (num_envs, 1)
        q = q * self.duct_confinement_factor * self.fin_area  # (num_envs, 1)

        alpha = fin_angles  # (num_envs, 4)

        # Normal force (lift): C_N = C_N_α * α * (1 - k_sat * α²)
        alpha_sq = alpha * alpha
        C_N = self.C_N_alpha * alpha * (1.0 - self.k_sat * alpha_sq)
        F_n = q * C_N  # (num_envs, 4)

        # Tangential force (drag): C_D = C_D_0 + C_D_α² * α²
        C_D = self.C_D_0 + self.C_D_alpha2 * alpha_sq
        F_t = q * C_D  # (num_envs, 4), always positive (opposing flow)

        # Thrust loss from fin blockage: proportional to fin area projected onto exhaust axis
        thrust_loss = F_t.sum(dim=-1, keepdim=False) * 0.1  # (num_envs,), rough estimate

        # Force vector in fin-local frame:
        # normal force is along fin's normal axis (+z in fin-local when positive deflection)
        # tangential drag opposes flow direction (-z in fin-local)
        # We return scalars here; fin_force_dispatch.py combines with geometry
        force_vector = torch.zeros(*fin_angles.shape, 3, device=fin_angles.device, dtype=fin_angles.dtype)
        force_vector[..., 2] = F_n  # normal direction (fin-local z)

        return FinForceResult(
            force_vector=force_vector,
            normal_force=F_n,
            tangential_force=F_t,
            thrust_loss=thrust_loss,
        )
