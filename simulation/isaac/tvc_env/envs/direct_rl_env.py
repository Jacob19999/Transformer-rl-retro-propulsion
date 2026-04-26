"""
DirectRLEnv implementation for the TVC environment.

Uses Isaac Lab's SimulationContext and InteractiveScene via build_scene(),
and implements a Gymnasium-compatible step/reset/close interface directly.

Methods:
  _setup_scene()           — builds scene, loads asset, initializes physics systems
  _pre_physics_step()      — clamps and stores actions per action_space contract
  _apply_action()          — servo dynamics → fin aero → force dispatch (called decimation times)
  _get_observations()      — assembles 24-dim observation tensor
  _get_rewards()           — computes weighted reward via reward_registry
  _get_dones()             — evaluates termination conditions
  step()                   — full RL step: pre_physics → decimated substeps → obs/reward/done
  reset()                  — reset all envs, return initial observations
  close()                  — release simulation context
  action_space             — Box(5,) with fin angle × 4 + throttle × 1
  observation_space        — Box(24,) or Box(27,) with wind

Requires Isaac Lab 2.3.2.
"""

from __future__ import annotations
import torch
from torch import Tensor
from pathlib import Path
from typing import Any

from tvc_env.envs.base_env import TVCEnvBase, BaseEnvConfig
from tvc_env.common.datatypes import VehicleState
from tvc_env.common.constants import ContactState


class TVCDirectRLEnv(TVCEnvBase):
    """Isaac Lab environment for EDF TVC simulation.

    Uses build_scene() for SimulationContext + InteractiveScene creation,
    and implements step/reset/close directly (no DirectRLEnv inheritance).
    """

    def __init__(
        self,
        config: BaseEnvConfig,
        render_mode: str | None = None,
        **kwargs,
    ):
        TVCEnvBase.__init__(self, config)
        self._pending_actions = None
        self._omega_max = 3000.0
        self._target_position_local = torch.tensor(
            config.config.get("task", {}).get("target_position", [0.0, 0.0, 5.0]),
            dtype=torch.float32,
        )
        self._target_position = self._target_position_local
        self._setup_scene()

    # ---- Scene setup ----

    def _setup_scene(self) -> None:
        """Build scene, load asset, initialize all physics systems."""
        from tvc_env.sim.scene_builder import SceneConfig, build_scene
        from tvc_env.asset.usd_loader import load_asset_metadata
        from tvc_env.asset.mass_properties import load_vehicle_config
        import yaml

        sim_root = Path(__file__).parents[2]
        metadata = load_asset_metadata(sim_root / "assets/metadata/edf_drone_v2.asset.yaml")
        vehicle_config = load_vehicle_config(sim_root / "configs/vehicle/edf_drone_v2.yaml")

        with open(sim_root / "configs/params/edf_90mm.yaml", "r", encoding="utf-8") as f:
            edf_config = yaml.safe_load(f)
        with open(sim_root / "configs/params/servo_mg996r.yaml", "r", encoding="utf-8") as f:
            servo_config = yaml.safe_load(f)

        scene_config = SceneConfig.from_yaml(self._config.config)
        self._sim_scene = build_scene(scene_config)
        self._drone = self._sim_scene["drone"]

        device = self._drone.device
        self._step_count = torch.zeros(self._config.num_envs, dtype=torch.int32, device=device)

        self._initialize_physics_systems(
            self._sim_scene, self._drone, metadata,
            vehicle_config, edf_config, servo_config,
            device=device,
        )

        edf_params = edf_config.get("edf", edf_config)
        self._omega_max = edf_params.get("omega_max") or self._omega_max
        env_origins = getattr(self._sim_scene.scene, "env_origins", None)
        if env_origins is None:
            env_origins = torch.zeros(self._config.num_envs, 3, dtype=torch.float32, device=device)
        else:
            env_origins = env_origins.to(device=device, dtype=torch.float32)
        self._env_origins = env_origins
        self._target_position = env_origins + self._target_position_local.to(device).unsqueeze(0)
        self._config.config["_target_position_world"] = self._target_position
        self._config.config["_omega_max_world"] = float(self._omega_max)

    # ---- Gymnasium interface ----

    @property
    def device(self):
        """The device on which the simulation is running."""
        return self._drone.device

    def step(self, action: Tensor) -> tuple[dict, Tensor, Tensor, Tensor, dict]:
        """Execute one RL step: pre-physics → decimated substeps → obs/reward/done.

        Args:
            action: (num_envs, 5) — 4 fin angles + 1 throttle.

        Returns:
            (obs_dict, reward, terminated, truncated, info)
        """
        action = action.to(self.device)
        self._pre_physics_step(action)

        for _ in range(self._config.decimation):
            self._apply_action()
            self._sim_scene.step()

        self._update_contact_state()
        self._step_count += 1
        terminated, time_out = self._get_dones()
        reward = self._get_rewards()

        # Auto-reset terminated/timed-out envs
        reset_ids = (terminated | time_out).nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self._reset_manager.reset_envs(reset_ids)
            self._step_count[reset_ids] = 0
            self._sim_scene.step()  # propagate reset state

        obs = self._get_observations()
        truncated = time_out & ~terminated
        return obs, reward, terminated, truncated, {}

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset all environments and return initial observations.

        Returns:
            (obs_dict, info)
        """
        indices = torch.arange(self._config.num_envs, device=self.device, dtype=torch.int64)
        self._reset_manager.reset_envs(indices)
        self._step_count.zero_()
        self._pending_actions = None
        self._sim_scene.step()  # propagate reset state
        return self._get_observations(), {}

    def close(self) -> None:
        """Release the simulation context."""
        if hasattr(self, "_sim_scene") and self._sim_scene is not None:
            self._sim_scene.close()
            self._sim_scene = None

    # ---- Physics step hooks ----

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
        self._body_iface.set_fin_joint_targets(new_servo_state)

        # Update EDF spool state
        omega_state = self._reset_manager.omega_state
        new_omega = self._edf_model.update(omega_state, throttle, dt)
        self._reset_manager._omega_prev = omega_state.clone()
        self._reset_manager._omega_state = new_omega

        # Compute aero forces from measured PhysX joint angles, not the target cache.
        measured_fin_angles = self._body_iface.get_fin_joint_positions()
        fin_dispatch = self._fin_dispatch.compute_body_frame_forces(measured_fin_angles, throttle)
        dynamics_cfg = self._config.config.get("dynamics", {})
        enable_fin_forces = dynamics_cfg.get("enable_fin_forces", True)
        enable_thrust_loss = dynamics_cfg.get("enable_thrust_loss", True)
        if not enable_fin_forces:
            fin_dispatch.forces_body.zero_()
            fin_dispatch.normal_force.zero_()
            fin_dispatch.tangential_force.zero_()
            fin_dispatch.thrust_loss.zero_()

        # EDF reaction force and torque on the body in body-FRD.
        # Exhaust exits along +Z_frd (down), so body thrust is along -Z_frd (up).
        body_ang_frd = self._body_iface.get_angular_velocity_body_frd()
        spin_axis = self._edf_model.thrust_axis.to(device=new_omega.device, dtype=new_omega.dtype)
        edf_output = self._edf_model.compute_output(
            new_omega,
            omega_state,
            body_ang_frd,
            dt,
            spin_axis=spin_axis,
        )
        raw_thrust = edf_output.thrust_force
        max_loss = torch.full_like(raw_thrust, 0.3 * self._edf_model.max_thrust)
        thrust_loss = torch.minimum(fin_dispatch.thrust_loss.clamp(min=0.0), max_loss)
        thrust_loss = torch.minimum(thrust_loss, raw_thrust)
        if not enable_thrust_loss:
            thrust_loss = torch.zeros_like(thrust_loss)
        thrust = raw_thrust - thrust_loss
        edf_force_body = torch.zeros(thrust.shape[0], 3, device=thrust.device)
        edf_force_body[:, 2] = -thrust
        static_torque = edf_output.static_reaction_torque
        dynamic_torque = edf_output.dynamic_spool_torque
        gyro_torque = edf_output.gyro_precession_torque
        if not dynamics_cfg.get("enable_edf_static_torque", True):
            static_torque = torch.zeros_like(static_torque)
        if not dynamics_cfg.get("enable_edf_dynamic_torque", True):
            dynamic_torque = torch.zeros_like(dynamic_torque)
        if not dynamics_cfg.get("enable_edf_gyro_torque", True):
            gyro_torque = torch.zeros_like(gyro_torque)
        gyro_torque = gyro_torque * float(dynamics_cfg.get("edf_gyro_torque_scale", 1.0))
        # Body aero/structural damping closes the underdamped roll/pitch mode
        # left by fin-servo lag and EDF gyro coupling during hover recovery.
        body_angular_damping = float(dynamics_cfg.get("body_angular_damping", 0.27))
        body_damping_torque = -body_angular_damping * body_ang_frd
        edf_torque_body = static_torque + dynamic_torque + gyro_torque + body_damping_torque

        q = self._body_iface.get_root_quaternion_wxyz()
        pos = self._body_iface.get_root_position()

        # Wind drag force in body-FRD frame
        wind_force_body = None
        if self._wind_model is not None and dynamics_cfg.get("enable_wind_force", True):
            lin_vel_w = self._body_iface.get_root_linear_velocity_world()
            wind_force_body = self._wind_model.compute_drag_force(lin_vel_w, q)
            self._wind_model.update_gust(dt)

        cops = fin_dispatch.cop_positions.unsqueeze(0).expand_as(fin_dispatch.forces_body)
        fin_torque_body = torch.linalg.cross(cops, fin_dispatch.forces_body).sum(dim=1)
        self._last_dynamics_debug = {
            "fin_force_body_frd_N": fin_dispatch.forces_body.sum(dim=1).detach(),
            "fin_torque_body_frd_Nm": fin_torque_body.detach(),
            "thrust_loss_N": thrust_loss.detach(),
            "edf_raw_thrust_N": raw_thrust.detach(),
            "edf_applied_thrust_N": thrust.detach(),
            "edf_static_torque_body_frd_Nm": static_torque.detach(),
            "edf_dynamic_torque_body_frd_Nm": dynamic_torque.detach(),
            "edf_gyro_torque_body_frd_Nm": gyro_torque.detach(),
            "body_damping_torque_body_frd_Nm": body_damping_torque.detach(),
            "edf_total_torque_body_frd_Nm": edf_torque_body.detach(),
            "wind_force_body_frd_N": (
                wind_force_body.detach()
                if wind_force_body is not None
                else torch.zeros_like(edf_force_body).detach()
            ),
        }

        self._wrench_dispatch.dispatch(
            fin_dispatch.forces_body,
            fin_dispatch.cop_positions,
            q,
            pos,
            edf_force_body,
            edf_torque_body,
            wind_force_body,
        )

    # ---- Observation / Reward / Done ----

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

    def _update_contact_state(self) -> None:
        """Advance the contact state machine using kinematic proxy.

        The original `SensorInterface` requires an Isaac Lab `ContactSensor` that
        is not currently wired into the scene. As a stand-in, we treat a low,
        slow vehicle as `in_contact` and pipe that plus crash heuristics into the
        ContactStateMachine so reward/termination logic actually fire on
        touchdown. Replace with a real contact sensor once available.
        """
        device = self._drone.device
        height = self._body_iface.get_altitude()
        lin_vel_w = self._body_iface.get_root_linear_velocity_world()
        vz = lin_vel_w[:, 2]
        in_contact = (height < 0.40) & (vz.abs() < 0.50)

        impact_speed = (-vz).clamp(min=0.0)
        ang_vel_frd = self._body_iface.get_angular_velocity_body_frd()
        ang_rate = ang_vel_frd.norm(dim=-1)
        q = self._body_iface.get_root_quaternion_wxyz()

        is_crashed = self._crash_detector.check_impact_speed(impact_speed, in_contact)
        is_crashed = is_crashed | self._crash_detector.check_tilt_at_contact(q, in_contact)
        is_crashed = is_crashed | self._crash_detector.check_angular_rate_at_contact(ang_rate, in_contact)
        is_crashed = is_crashed | self._crash_detector.check_excessive_tilt(q)

        # Force surrogate (any positive value above threshold registers as contact).
        contact_force = torch.where(
            in_contact,
            torch.full_like(height, 5.0),
            torch.zeros_like(height),
        ).to(device=device)
        self._contact_sm.update(in_contact, is_crashed, contact_force)

    def _build_vehicle_state(self) -> VehicleState:
        """Collect all state into a VehicleState dataclass."""
        device = self._drone.device
        pos = self._body_iface.get_root_position()
        q = self._body_iface.get_root_quaternion_wxyz()
        lin_vel_w = self._body_iface.get_root_linear_velocity_world()
        ang_vel_w = self._body_iface.get_root_angular_velocity_world()
        lin_vel_frd = self._body_iface.get_linear_velocity_body_frd()
        ang_vel_frd = self._body_iface.get_angular_velocity_body_frd()
        fin_angles = self._body_iface.get_fin_joint_positions()
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
