"""
Observation vector assembly per observation_space contract.

24-dim base observation:
  [0:3]   position_error (m)
  [3:7]   attitude_quat_wxyz (w,x,y,z)
  [7:10]  linear_vel_body_frd (m/s)
  [10:13] angular_vel_body_frd (rad/s)
  [13]    height (m)
  [14:18] fin_angles (rad, +X,+Y,-X,-Y)
  [18:22] fin_rates (rad/s)
  [22]    motor_rpm_normalized [0, 1]
  [23]    contact_state (int → float)

Optional 27-dim with wind_estimate at [24:27].
"""

from __future__ import annotations
import torch
from torch import Tensor
from tvc_env.common.datatypes import VehicleState


OBS_DIM_BASE = 24
OBS_DIM_WITH_WIND = 27


def assemble_observation(
    state: VehicleState,
    target_position: Tensor,     # (3,) or (num_envs, 3)
    omega_max: float,
    wind_estimate: Tensor | None = None,  # (num_envs, 3) optional
) -> Tensor:
    """Assemble 24-dim (or 27-dim with wind) observation tensor.

    Args:
        state: Current VehicleState.
        target_position: Target position in world frame (3,) or (num_envs, 3).
        omega_max: Maximum rotor omega (rad/s) for normalization.
        wind_estimate: Optional wind estimate in body-FRD frame (num_envs, 3).

    Returns:
        Tensor of shape (num_envs, 24) or (num_envs, 27) if wind given.
    """
    num_envs = state.position.shape[0]
    device = state.position.device

    # [0:3] Position error: target - current (m)
    if target_position.dim() == 1:
        target = target_position.unsqueeze(0).expand(num_envs, -1).to(device)
    else:
        target = target_position.to(device)
    pos_error = target - state.position  # (num_envs, 3)

    # [3:7] Attitude quaternion (w,x,y,z)
    quat_wxyz = state.quaternion_wxyz  # (num_envs, 4)

    # [7:10] Linear velocity in body-FRD (m/s)
    lin_vel_frd = state.linear_vel_frd  # (num_envs, 3)

    # [10:13] Angular velocity in body-FRD (rad/s)
    ang_vel_frd = state.angular_vel_frd  # (num_envs, 3)

    # [13] Height above ground (m)
    height = state.height.unsqueeze(-1)  # (num_envs, 1)

    # [14:18] Fin angles (rad)
    fin_angles = state.fin_angles  # (num_envs, 4)

    # [18:22] Fin angular rates (rad/s)
    fin_rates = state.fin_rates  # (num_envs, 4)

    # [22] Motor RPM normalized [0, 1]
    rpm_norm = (state.motor_omega / max(omega_max, 1.0)).unsqueeze(-1)  # (num_envs, 1)

    # [23] Contact state (enum int → float)
    contact_float = state.contact_state.float().unsqueeze(-1)  # (num_envs, 1)

    # Concatenate base observation
    obs = torch.cat([
        pos_error,       # 3
        quat_wxyz,       # 4
        lin_vel_frd,     # 3
        ang_vel_frd,     # 3
        height,          # 1
        fin_angles,      # 4
        fin_rates,       # 4
        rpm_norm,        # 1
        contact_float,   # 1
    ], dim=-1)  # (num_envs, 24)

    assert obs.shape[-1] == OBS_DIM_BASE, f"Expected {OBS_DIM_BASE} obs dims, got {obs.shape[-1]}"

    # Optional wind estimate
    if wind_estimate is not None:
        obs = torch.cat([obs, wind_estimate], dim=-1)  # (num_envs, 27)

    return obs


def get_observation_space(include_wind: bool = False):
    """Return the Gymnasium Box observation space definition.

    Returns:
        gym.spaces.Box with appropriate bounds.
    """
    try:
        import gymnasium as gym
        import numpy as np
    except ImportError:
        return None

    dim = OBS_DIM_WITH_WIND if include_wind else OBS_DIM_BASE
    return gym.spaces.Box(
        low=-float("inf"),
        high=float("inf"),
        shape=(dim,),
        dtype=np.float32,
    )
