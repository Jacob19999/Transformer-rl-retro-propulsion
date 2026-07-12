"""
Termination condition checks for the TVC environment.

Checks: max tilt exceeded, max altitude error exceeded, crash state,
and task success state. Episode timeout is reported separately by the
environment as a truncation. All vectorized boolean tensor output.
"""

from __future__ import annotations
import torch
from torch import Tensor
from tvc_env.common.constants import ContactState
from tvc_env.common.quaternions import tilt_angle


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
    return tilt_angle(quaternion_wxyz) > max_tilt


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
    if target.dim() == 1:
        target_z = target[2]
    else:
        target_z = target[:, 2]
    # Isaac uses Z-up. Lateral pad miss is handled by reward/success criteria;
    # this guard should only fail-stop runaway altitude excursions.
    altitude_error = (position[:, 2] - target_z).abs()
    return altitude_error > max_altitude_error


def check_crash_termination(contact_state: Tensor) -> Tensor:
    """Check if any environment is in CRASHED state.

    Returns:
        Bool tensor (num_envs,).
    """
    return contact_state == ContactState.CRASHED


def check_landed_termination(contact_state: Tensor) -> Tensor:
    """Check if a landing task has reached the terminal LANDED state."""
    return contact_state == ContactState.LANDED


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
    """Check task/physics terminal conditions and return combined signal.

    Episode time limits are intentionally excluded here. The environment
    reports them separately as truncations so RL code can distinguish
    bootstrap-safe timeouts from true terminal states.

    Args:
        quaternion_wxyz: (num_envs, 4)
        position: (num_envs, 3)
        target_position: (3,)
        contact_state: (num_envs,) int
        step_count: (num_envs,) int, accepted for API compatibility.
        task_config: Task config dict.
        physics_dt: Physics timestep (s), accepted for API compatibility.
        decimation: RL step = decimation * physics_dt, accepted for API compatibility.

    Returns:
        Bool tensor (num_envs,) — True where episode should terminate.
    """
    term = task_config.get("task", task_config).get("termination", {})

    dones = torch.zeros(position.shape[0], dtype=torch.bool, device=position.device)

    if term.get("crash", True):
        dones = dones | check_crash_termination(contact_state)

    success = task_config.get("task", task_config).get("success", {})
    if str(success.get("state", "")).upper() == "LANDED":
        dones = dones | check_landed_termination(contact_state)

    max_tilt = term.get("max_tilt", 1.57)
    dones = dones | check_tilt_termination(quaternion_wxyz, max_tilt)

    max_alt_err = term.get("max_altitude_error", 10.0)
    dones = dones | check_altitude_termination(position, target_position, max_alt_err)

    return dones
