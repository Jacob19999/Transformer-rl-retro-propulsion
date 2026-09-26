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
    if 'angular_velocity_range' in spawn:
        # User requested adverse-rate recovery (Sept 15). These are body FRD
        # rad/s, rotated into the world frame required by the PhysX writer.
        from tvc_env.common.frames import frd_to_isaac
        from tvc_env.common.quaternions import rotate_vector
        low, high = torch.tensor(spawn['angular_velocity_range'],device=device,dtype=torch.float32)
        body_rates = low + torch.rand(n,3,device=device)*(high-low)
        angular_vels = rotate_vector(quaternions,frd_to_isaac(body_rates))

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
        env_origins: Tensor | None = None,
        com_offset_model=None,
        wind_model=None,
        battery_model=None,
    ):
        self._body = body_interface
        self._servo = servo_model
        self._edf = edf_model
        self._contacts = contact_state_machine
        self._task_config = task_config
        self._env_origins = env_origins
        self._com_offset_model = com_offset_model
        self._wind_model = wind_model
        self._battery_model = battery_model
        self._num_envs = 0

        # Persistent actuator states
        self._servo_state = None
        self._omega_state = None
        self._omega_prev = None
        # Physics-derived level-hover rotor fraction, set by the environment.
        # Used by spawn.initial_motor_omega_fraction: hover.
        self.hover_omega_fraction = None

    def initialize(self, num_envs: int, device: torch.device) -> None:
        """Initialize actuator state tensors."""
        self._servo_state = self._servo.reset(num_envs, device)
        self._omega_state = self._edf.reset(num_envs, device)
        self._omega_prev = self._omega_state.clone()
        self._num_envs = num_envs
        if self._env_origins is None:
            self._env_origins = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        else:
            self._env_origins = self._env_origins.to(device=device, dtype=torch.float32)

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
        positions = positions + self._env_origins[env_ids]

        # Apply the episode's physical COM before writing root state. Offsets
        # are sampled in body-FRD and converted at the Isaac boundary.
        if self._com_offset_model is not None:
            offsets = self._com_offset_model.sample_offsets(self._num_envs, env_ids)
            self._body.set_body_com_offset_frd(offsets[env_ids], env_ids)

        # Set root state via body interface
        self._body.set_root_state(positions, quaternions, linear_vels, angular_vels, env_ids=env_ids)

        # Reset servo and EDF states for these envs
        if self._servo_state is not None:
            self._servo_state[env_ids] = 0.0
            zeros = self._servo_state[env_ids]
            self._body.write_fin_joint_state(zeros, torch.zeros_like(zeros), env_ids=env_ids)
            self._body.set_fin_joint_targets(zeros, env_ids=env_ids)
        task = self._task_config.get("task", self._task_config)
        spawn = task.get("spawn", {})
        if self._omega_state is not None:
            omega_fraction = self._initial_rotor_fraction(spawn, len(env_ids), device)
            initial_omega = omega_fraction * float(self._edf.omega_max)
            self._omega_state[env_ids] = initial_omega
            self._omega_prev[env_ids] = initial_omega

        # Reset contact state machine
        self._contacts.reset(env_ids)
        if self._wind_model is not None:
            self._wind_model.reset(env_ids)
            if "wind_speed_range" in spawn:
                # Per-episode steady horizontal wind, uniform direction. The
                # waypoint task varies it by curriculum stage; gusts remain a
                # disturbance-config option.
                low, high = (float(v) for v in spawn["wind_speed_range"])
                speed = low + torch.rand(len(env_ids), device=device) * (high - low)
                azimuth = torch.rand(len(env_ids), device=device) * (2 * torch.pi)
                wind = torch.stack((speed * azimuth.cos(), speed * azimuth.sin(), torch.zeros_like(speed)), -1)
                self._wind_model._steady_wind[env_ids] = wind
        if self._battery_model is not None:
            self._battery_model.reset(env_ids)
            if "initial_soc_range" in spawn:
                low, high = (float(v) for v in spawn["initial_soc_range"])
                if not 0.0 <= low <= high <= 1.0:
                    raise ValueError("initial_soc_range must lie within [0, 1]")
                battery = self._battery_model
                battery.soc[env_ids] = low + torch.rand(len(env_ids), device=device) * (high - low)
                battery.voltage_v[env_ids] = battery.ocv()[env_ids]
            if self._omega_state is not None:
                self._battery_model.carry_load(env_ids, self._omega_state, self._edf)

        # Isaac Lab requires reset() after state writers so actuator caches and
        # wrench composers cannot carry state across the episode boundary.
        self._body.reset_buffers(env_ids)

    def _initial_rotor_fraction(self, spawn: dict, count: int, device) -> Tensor:
        """Spawn rotor speed as a fraction of omega_max.

        ``hover`` spawns with the rotor already at level-hover speed, i.e. the
        angular momentum of a vehicle that spooled up on the pad (legs hold the
        body) and then took off. Spooling 0 -> hover in the air instead hands
        the body I_rotor*omega/I_zz ~ 40 rad/s of yaw (2e-4*0.85*4650/0.02),
        which the old recovery stages demanded be cancelled after the fact.
        Optional ``initial_motor_omega_jitter`` adds a uniform +/- offset.
        """
        value = spawn.get("initial_motor_omega_fraction", 0.0)
        if isinstance(value, str):
            if value != "hover":
                raise ValueError("initial_motor_omega_fraction must be a number or 'hover'")
            if self.hover_omega_fraction is None:
                raise ValueError("Hover rotor spawn requires the environment's hover fraction")
            value = self.hover_omega_fraction
        jitter = float(spawn.get("initial_motor_omega_jitter", 0.0))
        fraction = float(value) + (torch.rand(count, device=device) * 2 - 1) * jitter
        return fraction.clamp(0.0, 1.0)

    @property
    def servo_state(self) -> Tensor:
        return self._servo_state

    @property
    def omega_state(self) -> Tensor:
        return self._omega_state

    @property
    def omega_prev(self) -> Tensor:
        return self._omega_prev
