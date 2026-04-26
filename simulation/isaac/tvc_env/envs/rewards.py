"""
Reward term implementations for the TVC environment.

Each function signature: fn(env_state, config) → Tensor(num_envs,).
All computations are vectorized for (num_envs,) output.

env_state: VehicleState dataclass (or compatible namespace).
config: Task config dict for term-specific parameters.
"""

from __future__ import annotations
import torch
from torch import Tensor
from tvc_env.common.constants import ContactState
from tvc_env.common.quaternions import to_euler


def _target_position(env_state, config: dict, default: list[float]) -> Tensor:
    """Return a target tensor broadcast to the env batch."""
    target = config.get("_target_position_world")
    if target is None:
        target = config.get("task", config).get("target_position", default)
    if not isinstance(target, Tensor):
        target = torch.tensor(target, dtype=env_state.position.dtype, device=env_state.position.device)
    else:
        target = target.to(dtype=env_state.position.dtype, device=env_state.position.device)
    if target.dim() == 1:
        target = target.unsqueeze(0).expand(env_state.position.shape[0], -1)
    return target


# ---- Shared reward terms ----

def compute_alive_bonus(env_state, config: dict) -> Tensor:
    """Constant positive reward for staying alive each step.

    Returns:
        Tensor (num_envs,) — constant 1.0 per environment.
    """
    num_envs = env_state.position.shape[0]
    return torch.ones(num_envs, device=env_state.position.device)


def compute_position_error_reward(env_state, config: dict) -> Tensor:
    """Negative reward proportional to L2 distance from target position.

    Returns:
        Tensor (num_envs,) — position error magnitude (positive, weight applies negative sign).
    """
    target = _target_position(env_state, config, [0, 0, 5])
    error = (env_state.position - target).norm(dim=-1)  # (num_envs,)
    return error


def compute_attitude_error_reward(env_state, config: dict) -> Tensor:
    """Negative reward proportional to tilt (deviation from level flight).

    Returns:
        Tensor (num_envs,) — tilt magnitude in radians.
    """
    roll, pitch, _ = to_euler(env_state.quaternion_wxyz)
    tilt = torch.sqrt(roll ** 2 + pitch ** 2)  # (num_envs,)
    return tilt


def compute_angular_velocity_reward(env_state, config: dict) -> Tensor:
    """Negative reward for body angular rates in FRD frame.

    Returns:
        Tensor (num_envs,) — angular rate norm (rad/s).
    """
    return env_state.angular_vel_frd.norm(dim=-1)


def compute_control_effort_reward(env_state, config: dict) -> Tensor:
    """Negative reward for fin deflection magnitude (control effort).

    Returns:
        Tensor (num_envs,) — mean absolute fin deflection.
    """
    return env_state.fin_angles.abs().mean(dim=-1)


def compute_control_rate_reward(env_state, config: dict) -> Tensor:
    """Negative reward for fin deflection rate (control aggressiveness).

    Returns:
        Tensor (num_envs,) — mean absolute fin deflection rate.
    """
    return env_state.fin_rates.abs().mean(dim=-1)


# ---- Hover-specific terms ----

def compute_hover_stability_reward(env_state, config: dict) -> Tensor:
    """Positive reward for being within hover tolerance (position + attitude).

    Returns:
        Tensor (num_envs,) — 1.0 if within tolerance, else 0.0.
    """
    success_cfg = config.get("task", config).get("success", {})
    max_pos_err = success_cfg.get("max_position_error", 0.5)
    max_tilt = success_cfg.get("max_tilt", 0.26)

    target = _target_position(env_state, config, [0, 0, 5])
    pos_err = (env_state.position - target).norm(dim=-1)
    roll, pitch, _ = to_euler(env_state.quaternion_wxyz)
    tilt = torch.sqrt(roll ** 2 + pitch ** 2)

    within_tolerance = (pos_err < max_pos_err) & (tilt < max_tilt)
    return within_tolerance.float()


def compute_drift_penalty(env_state, config: dict) -> Tensor:
    """Penalty for horizontal drift velocity from hover point.

    Returns:
        Tensor (num_envs,) — horizontal speed magnitude.
    """
    # Horizontal velocity in FRD: x=forward, y=right
    horiz_vel = env_state.linear_vel_frd[:, :2]  # (num_envs, 2)
    return horiz_vel.norm(dim=-1)


def compute_contact_penalty(env_state, config: dict) -> Tensor:
    """One-time penalty for any ground contact (hover task should not touch ground).

    Returns:
        Tensor (num_envs,) — 1.0 on contact, else 0.0.
    """
    is_contact = env_state.contact_state != ContactState.AIRBORNE
    return is_contact.float()


# ---- Landing-specific terms ----

def compute_crash_penalty(env_state, config: dict) -> Tensor:
    """One-time penalty for crashing.

    Returns:
        Tensor (num_envs,) — 1.0 if CRASHED, else 0.0.
    """
    is_crashed = env_state.contact_state == ContactState.CRASHED
    return is_crashed.float()


def compute_touchdown_softness_reward(env_state, config: dict) -> Tensor:
    """Reward for soft touchdown speed (lower impact speed = higher reward).

    Returns:
        Tensor (num_envs,) — softness score [0, 1], higher for softer landing.
    """
    # Reward based on inverse of downward velocity (FRD z = down)
    # Lower downward speed at contact = softer landing
    downward_speed = env_state.linear_vel_frd[:, 2].clamp(min=0.0)  # (num_envs,)
    softness = torch.exp(-downward_speed)  # Exponential decay with speed
    return softness


def compute_landing_success_reward(env_state, config: dict) -> Tensor:
    """One-time reward for successfully landing (LANDED state).

    Returns:
        Tensor (num_envs,) — 1.0 if LANDED, else 0.0.
    """
    is_landed = env_state.contact_state == ContactState.LANDED
    return is_landed.float()


def compute_pad_accuracy_reward(env_state, config: dict) -> Tensor:
    """Reward for landing accuracy (distance from pad center).

    Returns:
        Tensor (num_envs,) — accuracy score, higher for closer to pad.
    """
    is_landed = env_state.contact_state == ContactState.LANDED
    target = _target_position(env_state, config, [0, 0, 0])  # (num_envs, 3)
    # Horizontal distance from pad center (x, y components, per env)
    horiz_dist = (env_state.position[:, :2] - target[:, :2]).norm(dim=-1)
    accuracy = torch.exp(-2.0 * horiz_dist)  # Exp decay with horizontal distance
    return accuracy * is_landed.float()


def compute_vertical_speed_shaping(env_state, config: dict) -> Tensor:
    """Reward for maintaining appropriate descent rate during approach.

    Returns:
        Tensor (num_envs,) — shaped descent rate signal.
    """
    # Target descent: small downward velocity, penalize upward or too-fast descent
    downward_speed = env_state.linear_vel_frd[:, 2]  # z=down in FRD
    # Penalize both upward flight and very fast descent
    target_descent = 0.5  # m/s ideal descent rate
    return (downward_speed - target_descent).abs()
