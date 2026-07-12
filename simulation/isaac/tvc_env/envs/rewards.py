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
from tvc_env.common.quaternions import tilt_angle


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


def compute_horizontal_position_error_reward(env_state, config: dict) -> Tensor:
    """Horizontal distance from the target pad center."""
    target = _target_position(env_state, config, [0, 0, 0])
    return (env_state.position[:, :2] - target[:, :2]).norm(dim=-1)


def compute_attitude_error_reward(env_state, config: dict) -> Tensor:
    """Negative reward proportional to tilt (deviation from level flight).

    Returns:
        Tensor (num_envs,) — tilt magnitude in radians.
    """
    return tilt_angle(env_state.quaternion_wxyz)


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
    tilt = tilt_angle(env_state.quaternion_wxyz)

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
    is_landed = env_state.contact_state == ContactState.LANDED
    # LANDED is emitted after a dwell period, so current velocity is already
    # near zero. Use the first-contact speed captured by the environment.
    touchdown_speed = getattr(env_state, "touchdown_speed", None)
    downward_speed = (
        touchdown_speed
        if touchdown_speed is not None
        else env_state.linear_vel_frd[:, 2].clamp(min=0.0)
    )
    softness = torch.exp(-downward_speed)  # Exponential decay with speed
    return softness * is_landed.float()


def compute_landing_success_reward(env_state, config: dict) -> Tensor:
    """One-time reward for a successful landing.

    Success requires LANDED plus ``success.max_pad_distance`` (lateral
    accuracy) plus, when ``success.max_touchdown_speed`` is configured,
    a soft enough impact. Paying this reward for every LANDED contact
    let PPO learn the easier "touch down softly anywhere" strategy seen
    in fix4/curriculum eval logs; tightening on touchdown speed too
    excludes "land on pad but slam into it" from counting as success
    (Phase-1 follow-on: the ppo_landing_seed0_20260426_192920 5M run
    reported max_touchdown_speed 0.347 m/s, so a 0.25 m/s gate forces
    the policy to learn a genuinely soft terminal flare).
    """
    is_landed = env_state.contact_state == ContactState.LANDED
    task = config.get("task", config)
    success_cfg = task.get("success", {})
    max_pad_distance = float(success_cfg.get("max_pad_distance", 0.5))
    max_touchdown_speed = success_cfg.get("max_touchdown_speed")
    target = _target_position(env_state, config, [0, 0, 0])
    horiz_dist = (env_state.position[:, :2] - target[:, :2]).norm(dim=-1)
    on_pad = horiz_dist <= max_pad_distance
    success = is_landed & on_pad
    if max_touchdown_speed is not None:
        touchdown_speed = getattr(env_state, "touchdown_speed", None)
        downward_speed = (
            touchdown_speed
            if touchdown_speed is not None
            else env_state.linear_vel_frd[:, 2].clamp(min=0.0)
        )
        soft_enough = downward_speed <= float(max_touchdown_speed)
        success = success & soft_enough
    return success.float()


def compute_pad_accuracy_reward(env_state, config: dict) -> Tensor:
    """Reward for landing accuracy (distance from pad center).

    Returns:
        Tensor (num_envs,) — accuracy score, higher for closer to pad.
    """
    is_landed = env_state.contact_state == ContactState.LANDED
    target = _target_position(env_state, config, [0, 0, 0])  # (num_envs, 3)
    # Horizontal distance from pad center (x, y components, per env)
    horiz_dist = (env_state.position[:, :2] - target[:, :2]).norm(dim=-1)
    # Gentler exponent (exp(-d) vs prior exp(-2d)) keeps the gradient on this
    # term meaningful at d > 1 m. With exp(-2d) at d=2 m the term collapses to
    # 0.018 of its peak value, so closing 1 m of lateral offset earns only
    # ~0.4 of weight — too weak to compete with fin-noise penalties early in
    # training, leaving the policy in a "land softly anywhere" local optimum.
    accuracy = torch.exp(-horiz_dist)
    return accuracy * is_landed.float()


def compute_off_pad_landing_penalty(env_state, config: dict) -> Tensor:
    """Terminal indicator for LANDED contacts outside the success pad radius."""
    is_landed = env_state.contact_state == ContactState.LANDED
    task = config.get("task", config)
    max_pad_distance = float(task.get("success", {}).get("max_pad_distance", 0.5))
    target = _target_position(env_state, config, [0, 0, 0])
    horiz_dist = (env_state.position[:, :2] - target[:, :2]).norm(dim=-1)
    off_pad = horiz_dist > max_pad_distance
    return (is_landed & off_pad).float()


def compute_horizontal_closure_reward(env_state, config: dict) -> Tensor:
    """Dense reward for horizontal velocity that closes distance to the pad.

    Positive values mean the vehicle is moving toward the pad center; negative
    values mean it is drifting away. This targets the diagnosed failure mode in
    fix4/curriculum runs: vertical touchdown is learned, but lateral error does
    not reliably decrease before contact.
    """
    target = _target_position(env_state, config, [0, 0, 0])
    to_target_xy = target[:, :2] - env_state.position[:, :2]
    dist = to_target_xy.norm(dim=-1).clamp(min=1e-6)
    direction_to_target = to_target_xy / dist.unsqueeze(-1)
    closing_speed = (env_state.linear_vel_world[:, :2] * direction_to_target).sum(dim=-1)
    return closing_speed.clamp(min=-3.0, max=3.0)


def compute_vertical_speed_shaping(env_state, config: dict) -> Tensor:
    """One-sided fast-descent penalty.

    Returns ``max(v_down - target_descent, 0)``: zero whenever the vehicle is
    descending no faster than the target rate (or hovering / climbing), and
    growing linearly in excess descent speed.

    The previous implementation used ``|v_down - target|`` (two-sided), which
    was the largest single per-step term in the integrated episode budget
    (Phase-1 diagnosis, run ppo_landing_seed0_20260426_192920: -78.8 of
    -129.6 per-step total at 9 m/s descent over ~75 steps). With the
    landing_success/pad_accuracy terminal so easily missed from large
    spawns, that cost dominated the policy's incentive landscape and made
    fast descent (rushing to terminate to cap costs) more attractive than
    correcting laterally first. Switching to a one-sided penalty makes
    "hover and correct laterally, then descend slowly" cheap and only
    penalizes truly unsafe descent rates — closer to the safe-landing
    objective and consistent with CLAUDE.md rule 2 on per-step magnitude
    balance.
    """
    downward_speed = env_state.linear_vel_frd[:, 2]  # z=down in FRD
    target_descent = 0.5  # m/s — fastest descent that incurs no penalty
    return (downward_speed - target_descent).clamp(min=0.0)


def compute_delta_v_cost(env_state, config: dict) -> Tensor:
    """Per-step thrust-fraction proxy for cumulative delta-v consumption.

    Approximates instantaneous thrust as T = T_max * (omega/omega_max)^2 and
    returns the dimensionless ratio T/T_max in [0, 1] per environment. Summed
    over an episode (with a negative weight in the reward), this acts as a
    proxy for delta-v: lower sustained thrust → lower delta-v cost. Multiply
    the weight by the RL step duration (physics_dt * decimation) if you want
    the integrated value to be interpretable in seconds-of-thrust units.

    Returns:
        Tensor (num_envs,) — thrust ratio in [0, 1].
    """
    omega_max = config.get("_omega_max_world", 4300.0)
    if isinstance(omega_max, Tensor):
        omega_max = float(omega_max.item())
    omega_max = max(float(omega_max), 1.0)
    omega = env_state.motor_omega.clamp(min=0.0)
    return (omega / omega_max).clamp(max=1.0).pow(2)
