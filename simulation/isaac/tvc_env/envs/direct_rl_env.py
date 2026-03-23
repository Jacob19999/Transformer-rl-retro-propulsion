"""
DirectRLEnv implementation for the TVC environment.

Subclasses Isaac Lab's DirectRLEnv and implements:
  _setup_scene()           — builds scene, loads asset, initializes physics systems
  _pre_physics_step()      — clamps and stores actions per action_space contract
  _apply_action()          — servo dynamics → fin aero → force dispatch (called decimation times)
  _get_observations()      — assembles 24-dim observation tensor
  _get_rewards()           — computes weighted reward via reward_registry
  _get_dones()             — evaluates termination conditions
  action_space             — Box(5,) with fin angle × 4 + throttle × 1
  observation_space        — Box(24,) or Box(27,) with wind

Requires Isaac Lab 2.3.2.
"""

from __future__ import annotations
import torch
from torch import Tensor
from pathlib import Path

from tvc_env.envs.base_env import TVCEnvBase, BaseEnvConfig
from tvc_env.common.datatypes import VehicleState
from tvc_env.common.constants import ContactState


class TVCDirectRLEnv(TVCEnvBase):
    """Isaac Lab DirectRLEnv for EDF TVC simulation."""

    def __init__(
        self,
        config: BaseEnvConfig,
        render_mode: str | None = None,
        **kwargs,
    ):
        try:
            from isaaclab.envs import DirectRLEnv
            DirectRLEnvBase = DirectRLEnv
        except ImportError:
            # Fallback for offline testing without Isaac Lab
            DirectRLEnvBase = object

        TVCEnvBase.__init__(self, config)
        if DirectRLEnvBase is not object:
            DirectRLEnvBase.__init__(self, config, render_mode=render_mode, **kwargs)

        self._pending_actions = None
        self._step_count = None
        self._omega_max = config.config.get("edf", {}).get("omega_max") or 3000.0
        self._target_position = torch.tensor(
            config.config.get("task", {}).get("target_position", [0.0, 0.0, 5.0]),
            dtype=torch.float32,
        )

    # ---- Isaac Lab DirectRLEnv interface methods ----

    def _setup_scene(self) -> None:
        """Build scene, load asset, initialize all physics systems."""
        from tvc_env.sim.scene_builder import SceneConfig, build_scene
        from tvc_env.asset.usd_loader import load_asset_metadata
        from tvc_env.asset.mass_properties import load_vehicle_config
        import yaml

        sim_root = Path(__file__).parents[2]
        metadata = load_asset_metadata(sim_root / "assets/metadata/edf_drone_v2.asset.yaml")
        vehicle_config = load_vehicle_config(sim_root / "configs/vehicle/edf_drone_v2.yaml")

        with open(sim_root / "configs/params/edf_90mm.yaml", "r") as f:
            edf_config = yaml.safe_load(f)
        with open(sim_root / "configs/params/servo_mg996r.yaml", "r") as f:
            servo_config = yaml.safe_load(f)

        scene_config = SceneConfig.from_yaml(self._config.config)
        self._scene = build_scene(scene_config)
        self._drone = self._scene["drone"]

        device = self._drone.device
        self._step_count = torch.zeros(self._config.num_envs, dtype=torch.int32, device=device)

        self._initialize_physics_systems(
            self._scene, self._drone, metadata,
            vehicle_config, edf_config, servo_config,
            device=device,
        )

    def _pre_physics_step(self, actions: Tensor) -> None:
        """Clamp and store actions before physics substeps.

        Action layout per action_space contract:
          [0:4] fin target angles (rad)
          [4]   throttle normalized [0, 1]
        """
        max_angle = self._servo_model.max_command_angle
        fin_commands = actions[:, :4].clamp(-max_angle, max_angle)
        throttle = actions[:, 4:5].clamp(0.0, 1.0)
        self._pending_actions = torch.cat([fin_commands, throttle], dim=-1)

    def _apply_action(self) -> None:
        """Apply one substep of actuator dynamics and force dispatch.

        Called `decimation` times per RL step.
        """
        if self._pending_actions is None:
            return

        dt = self._config.physics_dt
        fin_commands = self._pending_actions[:, :4]
        throttle = self._pending_actions[:, 4]

        # Update servo state
        servo_state = self._reset_manager.servo_state
        new_servo_state = self._servo_model.update(servo_state, fin_commands, dt)
        self._reset_manager._servo_state = new_servo_state

        # Update EDF spool state
        omega_prev = self._reset_manager.omega_prev
        omega_state = self._reset_manager.omega_state
        new_omega = self._edf_model.update(omega_state, throttle, dt)
        self._reset_manager._omega_prev = omega_state.clone()
        self._reset_manager._omega_state = new_omega

        # Compute aero forces and dispatch
        forces_body, cops = self._fin_dispatch.compute_body_frame_forces(new_servo_state, throttle)

        # EDF thrust force in body-FRD frame (along thrust axis = +z in FRD)
        thrust = self._edf_model.compute_thrust(new_omega)  # (num_envs,)
        edf_force_body = torch.zeros(thrust.shape[0], 3, device=thrust.device)
        edf_force_body[:, 2] = thrust  # +z direction in FRD = down = thrust direction

        q = self._body_iface.get_root_quaternion_wxyz()
        self._wrench_dispatch.dispatch(forces_body, cops, q, edf_force_body)

    def _get_observations(self) -> dict:
        """Assemble observation dict with 'policy' key."""
        from tvc_env.envs.observations import assemble_observation

        state = self._build_vehicle_state()
        obs = assemble_observation(state, self._target_position, self._omega_max)

        return {"policy": obs}

    def _get_rewards(self) -> Tensor:
        """Compute total reward via reward_registry."""
        from tvc_env.envs.reward_registry import compute_total_reward

        state = self._build_vehicle_state()
        task_cfg = self._config.config
        reward_weights = task_cfg.get("task", task_cfg).get("reward", {})
        return compute_total_reward(reward_weights, state, task_cfg)

    def _get_dones(self) -> tuple[Tensor, Tensor]:
        """Evaluate termination conditions.

        Returns:
            Tuple (terminated, time_out) — both bool tensors (num_envs,).
        """
        from tvc_env.envs.terminations import check_all_terminations

        state = self._build_vehicle_state()
        dones = check_all_terminations(
            state.quaternion_wxyz,
            state.position,
            self._target_position.to(state.position.device),
            state.contact_state,
            self._step_count,
            self._config.config,
            self._config.physics_dt,
            self._config.decimation,
        )
        time_out = self._step_count >= int(
            self._config.config.get("task", {}).get("episode_length_s", 30.0) /
            (self._config.physics_dt * self._config.decimation)
        )
        return dones, time_out

    def _build_vehicle_state(self) -> VehicleState:
        """Collect all state into a VehicleState dataclass."""
        device = self._drone.device
        pos = self._body_iface.get_root_position()
        q = self._body_iface.get_root_quaternion_wxyz()
        lin_vel_w = self._body_iface.get_root_linear_velocity_world()
        ang_vel_w = self._body_iface.get_root_angular_velocity_world()
        lin_vel_frd = self._body_iface.get_linear_velocity_body_frd()
        ang_vel_frd = self._body_iface.get_angular_velocity_body_frd()
        fin_angles = self._reset_manager.servo_state
        fin_rates = self._body_iface.get_fin_joint_velocities()
        omega = self._reset_manager.omega_state
        contact = self._contact_sm.state
        height = self._body_iface.get_altitude()

        return VehicleState(
            position=pos,
            quaternion_wxyz=q,
            linear_vel_world=lin_vel_w,
            angular_vel_world=ang_vel_w,
            linear_vel_frd=lin_vel_frd,
            angular_vel_frd=ang_vel_frd,
            fin_angles=fin_angles,
            fin_rates=fin_rates,
            motor_omega=omega,
            contact_state=contact,
            height=height,
        )

    # ---- Gymnasium spaces ----

    @property
    def action_space(self):
        """5-dim action space: 4 fin angles + 1 throttle."""
        try:
            import gymnasium as gym
            import numpy as np
        except ImportError:
            return None
        max_angle = self._servo_model.max_command_angle if hasattr(self, '_servo_model') else 0.262
        return gym.spaces.Box(
            low=np.array([-max_angle] * 4 + [0.0], dtype=np.float32),
            high=np.array([max_angle] * 4 + [1.0], dtype=np.float32),
            shape=(5,),
            dtype=np.float32,
        )

    @property
    def observation_space(self):
        """24-dim observation space."""
        try:
            import gymnasium as gym
            import numpy as np
        except ImportError:
            return None
        return gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(24,),
            dtype=np.float32,
        )
