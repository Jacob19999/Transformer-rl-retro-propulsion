"""
Termination condition checks for the TVC environment.

Checks: max tilt exceeded, max altitude error exceeded, crash state,
episode timeout. All vectorized boolean tensor output.
"""

from __future__ import annotations
import torch
from torch import Tensor
from tvc_env.common.constants import ContactState
from tvc_env.common.quaternions import to_euler


def check_tilt_termination(
    quaternion_wxyz: Tensor,
    max_tilt: float,
) -> Tensor:
    """Check if tilt exceeds max_tilt.

    Args:
        quaternion_wxyz: (num_envs, 4)
        max_tilt: Maximum allowed tilt in radians.

    Returns:
        Bool tensor (num_envs,) — True where tilt exceeded.
    """
    roll, pitch, _ = to_euler(quaternion_wxyz)
    tilt = torch.sqrt(roll ** 2 + pitch ** 2)
    return tilt > max_tilt


def check_altitude_termination(
    position: Tensor,
    target_position: Tensor,
    max_altitude_error: float,
) -> Tensor:
    """Check if altitude error exceeds limit.

    Args:
        position: (num_envs, 3) world frame.
        target_position: (3,) target altitude.
        max_altitude_error: Maximum altitude deviation (m).

    Returns:
        Bool tensor (num_envs,).
    """
    target = target_position.to(position.device)
    # Altitude error: z-component difference (or y in Isaac y-up)
    # Use full 3D distance for simplicity
    altitude_error = (position - target).norm(dim=-1)
    return altitude_error > max_altitude_error


def check_crash_termination(contact_state: Tensor) -> Tensor:
    """Check if any environment is in CRASHED state.

    Returns:
        Bool tensor (num_envs,).
    """
    return contact_state == ContactState.CRASHED


def check_episode_timeout(
    step_count: Tensor,
    max_steps: int,
) -> Tensor:
    """Check if episode has exceeded maximum step count.

    Args:
        step_count: (num_envs,) current step count.
        max_steps: Maximum allowed steps per episode.

    Returns:
        Bool tensor (num_envs,).
    """
    return step_count >= max_steps


def check_all_terminations(
    quaternion_wxyz: Tensor,
    position: Tensor,
    target_position: Tensor,
    contact_state: Tensor,
    step_count: Tensor,
    task_config: dict,
    physics_dt: float,
    decimation: int,
) -> Tensor:
    """Check all termination conditions and return combined done signal.

    Args:
        quaternion_wxyz: (num_envs, 4)
        position: (num_envs, 3)
        target_position: (3,)
        contact_state: (num_envs,) int
        step_count: (num_envs,) int
        task_config: Task config dict.
        physics_dt: Physics timestep (s).
        decimation: RL step = decimation * physics_dt.

    Returns:
        Bool tensor (num_envs,) — True where episode should terminate.
    """
    term = task_config.get("task", task_config).get("termination", {})
    episode_s = task_config.get("task", task_config).get("episode_length_s", 30.0)
    rl_dt = physics_dt * decimation
    max_steps = int(episode_s / rl_dt)

    dones = torch.zeros(position.shape[0], dtype=torch.bool, device=position.device)

    if term.get("crash", True):
        dones = dones | check_crash_termination(contact_state)

    max_tilt = term.get("max_tilt", 1.57)
    dones = dones | check_tilt_termination(quaternion_wxyz, max_tilt)

    max_alt_err = term.get("max_altitude_error", 10.0)
    dones = dones | check_altitude_termination(position, target_position, max_alt_err)

    dones = dones | check_episode_timeout(step_count, max_steps)

    return dones
