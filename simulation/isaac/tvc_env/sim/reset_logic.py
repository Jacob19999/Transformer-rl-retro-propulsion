"""
Episode reset with randomized initial conditions.

Samples position, velocity, and attitude from spawn ranges in task config,
sets root state via body_interface, resets servo/EDF actuator states,
and resets contact state machine. Vectorized per-env reset.
"""

from __future__ import annotations
import torch
from torch import Tensor
from typing import Any


def sample_spawn_state(
    task_config: dict[str, Any],
    env_ids: Tensor,
    device: torch.device = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sample initial position, velocity, and attitude from task spawn ranges.

    Args:
        task_config: Task config dict with spawn.position_range, velocity_range, attitude_range.
        env_ids: Tensor of environment indices to reset.
        device: Target device.

    Returns:
        Tuple (positions, quaternions_wxyz, linear_vels, angular_vels),
        each of shape (len(env_ids), 3 or 4).
    """
    from tvc_env.common.quaternions import from_euler

    task = task_config.get("task", task_config)
    spawn = task.get("spawn", {})
    n = len(env_ids)

    # Sample positions
    pos_range = spawn.get("position_range", [[-1, -1, 4], [1, 1, 6]])
    pos_min = torch.tensor(pos_range[0], dtype=torch.float32, device=device)
    pos_max = torch.tensor(pos_range[1], dtype=torch.float32, device=device)
    positions = pos_min + torch.rand(n, 3, device=device) * (pos_max - pos_min)

    # Sample velocities
    vel_range = spawn.get("velocity_range", [[-0.5]*3, [0.5]*3])
    vel_min = torch.tensor(vel_range[0], dtype=torch.float32, device=device)
    vel_max = torch.tensor(vel_range[1], dtype=torch.float32, device=device)
    linear_vels = vel_min + torch.rand(n, 3, device=device) * (vel_max - vel_min)

    # Sample attitude (Euler angles → quaternion)
    att_range = spawn.get("attitude_range", [[-0.05]*3, [0.05]*3])
    att_min = torch.tensor(att_range[0], dtype=torch.float32, device=device)
    att_max = torch.tensor(att_range[1], dtype=torch.float32, device=device)
    euler_angles = att_min + torch.rand(n, 3, device=device) * (att_max - att_min)
    quaternions = from_euler(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])

    angular_vels = torch.zeros(n, 3, device=device)

    return positions, quaternions, linear_vels, angular_vels


class ResetManager:
    """Orchestrates per-episode resets for all environments."""

    def __init__(
        self,
        body_interface,
        servo_model,
        edf_model,
        contact_state_machine,
        task_config: dict[str, Any],
    ):
        self._body = body_interface
        self._servo = servo_model
        self._edf = edf_model
        self._contacts = contact_state_machine
        self._task_config = task_config

        # Persistent actuator states
        self._servo_state = None
        self._omega_state = None
        self._omega_prev = None

    def initialize(self, num_envs: int, device: torch.device) -> None:
        """Initialize actuator state tensors."""
        self._servo_state = self._servo.reset(num_envs, device)
        self._omega_state = self._edf.reset(num_envs, device)
        self._omega_prev = self._omega_state.clone()

    def reset_envs(self, env_ids: Tensor) -> None:
        """Reset specified environments to randomized initial conditions.

        Args:
            env_ids: Tensor of environment indices to reset.
        """
        if len(env_ids) == 0:
            return

        device = env_ids.device
        positions, quaternions, linear_vels, angular_vels = sample_spawn_state(
            self._task_config, env_ids, device
        )

        # Set root state via body interface
        self._body.set_root_state(positions, quaternions, linear_vels, angular_vels)

        # Reset servo and EDF states for these envs
        if self._servo_state is not None:
            self._servo_state[env_ids] = 0.0
        if self._omega_state is not None:
            self._omega_state[env_ids] = 0.0
            self._omega_prev[env_ids] = 0.0

        # Reset contact state machine
        self._contacts.reset(env_ids)

    @property
    def servo_state(self) -> Tensor:
        return self._servo_state

    @property
    def omega_state(self) -> Tensor:
        return self._omega_state

    @property
    def omega_prev(self) -> Tensor:
        return self._omega_prev
