"""
State derivative utilities and derived quantity helpers.

Helper functions for computing derived quantities used by wind_model and fin_aero:
  - relative airspeed
  - dynamic pressure
  - exhaust velocity at current throttle
"""

from __future__ import annotations
import torch
from torch import Tensor
from tvc_env.common.constants import AIR_DENSITY


def compute_relative_airspeed(
    body_vel_world: Tensor,     # (num_envs, 3) body velocity in world frame (m/s)
    wind_vel_world: Tensor,     # (3,) or (num_envs, 3) wind velocity in world frame (m/s)
) -> Tensor:
    """Compute relative airspeed vector (body velocity relative to air mass).

    v_rel = v_body - v_wind

    Returns:
        Tensor (num_envs, 3) relative airspeed in world frame (m/s).
    """
    if wind_vel_world.dim() == 1:
        wind = wind_vel_world.unsqueeze(0)
    else:
        wind = wind_vel_world
    return body_vel_world - wind


def compute_dynamic_pressure(
    airspeed_magnitude: Tensor,     # (num_envs,) or scalar (m/s)
    air_density: float = AIR_DENSITY,
) -> Tensor:
    """Compute dynamic pressure q = 0.5 * ρ * V².

    Returns:
        Tensor same shape as input — dynamic pressure (Pa).
    """
    return 0.5 * air_density * airspeed_magnitude ** 2


def compute_exhaust_velocity(
    throttle: Tensor,               # (num_envs,) normalized throttle [0, 1]
    max_exhaust_speed: float,       # m/s at full throttle
) -> Tensor:
    """Compute EDF exhaust velocity at current throttle.

    Exhaust speed scales approximately linearly with throttle.
    (RPM ∝ throttle, exhaust speed ∝ RPM for constant geometry)

    Returns:
        Tensor (num_envs,) — exhaust speed in m/s.
    """
    return throttle * max_exhaust_speed


def compute_fin_dynamic_pressure(
    throttle: Tensor,               # (num_envs,)
    max_exhaust_speed: float = 40.0,
    fin_area: float = 0.002,
    duct_correction: float = 1.3,
    air_density: float = AIR_DENSITY,
) -> Tensor:
    """Compute effective dynamic pressure × area for fin aero force computation.

    q_eff = 0.5 * ρ * v_exhaust² * A_fin * k_duct

    Returns:
        Tensor (num_envs, 1) — effective dynamic pressure × area (N per unit C_N or C_D).
    """
    v_exhaust = compute_exhaust_velocity(throttle, max_exhaust_speed)  # (num_envs,)
    q = compute_dynamic_pressure(v_exhaust, air_density)               # (num_envs,)
    q_eff = q * fin_area * duct_correction                             # (num_envs,)
    return q_eff.unsqueeze(-1)  # (num_envs, 1)
