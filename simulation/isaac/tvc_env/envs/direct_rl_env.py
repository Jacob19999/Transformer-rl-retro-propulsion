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
        self._touchdown_speed = None
        self._max_downward_speed_step = None
        self._landing_contact_force_step = None
        self._unsafe_contact_step = None
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

        # Mission hardware profiles must reach both force models and PhysX
        # actuator limits. Previously only the fixed parameter files were read.
        from tvc_env.envs.task_registry import deep_merge
        edf_config = deep_merge(edf_config, {"edf": self._config.config.get("edf", {})})
        servo_config = deep_merge(servo_config, {"servo": self._config.config.get("servo", {})})
        self._resolved_hardware = dict(vehicle=vehicle_config, edf=edf_config['edf'], servo=servo_config['servo'])

        scene_config = SceneConfig.from_yaml(self._config.config)
        servo_params = servo_config.get("servo", servo_config)
        scene_config.fin_drive_stiffness = float(servo_params.get("drive_stiffness", 80.0))
        scene_config.fin_drive_damping = float(servo_params.get("drive_damping", 2.0))
        scene_config.fin_effort_limit = float(servo_params.get("stall_torque", 1.08))
        scene_config.fin_velocity_limit = float(servo_params.get("max_angular_velocity", 7.54))
        self._sim_scene = build_scene(scene_config)
        self._drone = self._sim_scene["drone"]

        device = self._drone.device
        self._step_count = torch.zeros(self._config.num_envs, dtype=torch.int32, device=device)

        self._initialize_physics_systems(
            self._sim_scene, self._drone, metadata,
            vehicle_config, edf_config, servo_config,
            device=device,
        )
        self._touchdown_speed = torch.zeros(self._config.num_envs, dtype=torch.float32, device=device)
        self._max_downward_speed_step = torch.zeros_like(self._touchdown_speed)
        self._landing_contact_force_step = torch.zeros_like(self._touchdown_speed)
        self._unsafe_contact_step = torch.zeros(
            self._config.num_envs, dtype=torch.bool, device=device
        )
        self._battery_energy_step_wh = torch.zeros_like(self._touchdown_speed)
        self._propulsive_delta_v_step = torch.zeros_like(self._touchdown_speed)
        from tvc_env.envs.rotation_metrics import RotationTracker, DEFAULT_LIMITS_DEG_S
        self._rotation = RotationTracker(self._config.num_envs, device,
            self._config.config.get('task', {}).get('rotation', {}).get('soft_limits_deg_s', DEFAULT_LIMITS_DEG_S))
        self._vehicle_mass = self._drone.root_physx_view.get_masses().sum(dim=-1).to(device)
        if self._config.config.get('env', {}).get('observe_battery', False) and self._battery_model is None:
            raise ValueError('Battery observations require the coupled battery model')

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
        self._navigation = None
        if self._config.config.get('task',{}).get('navigation',{}).get('enabled'):
            from tvc_env.envs.waypoints import WaypointMission
            self._navigation = WaypointMission(self._config.num_envs,device,self._config.config,env_origins,self._target_position)

    # ---- Gymnasium interface ----

    @property
    def device(self):
        """The device on which the simulation is running."""
        return self._drone.device

    def nominal_hover_throttle(self) -> float:
        """Level equilibrium including all link masses and neutral vane drag."""
        import math
        mass = float(self._drone.root_physx_view.get_masses()[0].sum())
        gravity = abs(float(self._config.config.get("physics", {}).get("gravity", [0, 0, -9.81])[2]))
        dynamics = self._config.config.get("dynamics", {})
        drag = 0.0
        if dynamics.get("enable_fin_forces", True) and dynamics.get("enable_thrust_loss", True):
            neutral = self._fin_dispatch.compute_body_frame_forces(
                torch.zeros(1, 4, device=self.device), torch.ones(1, device=self.device)
            )
            drag = min(float(neutral.thrust_loss[0]), self._max_fin_thrust_loss_fraction * self._edf_model.max_thrust)
            if self._coupled_jet is not None:
                jet = self._coupled_jet.compute(torch.zeros(1, 4, device=self.device),
                    torch.ones(1, device=self.device), torch.full((1,), self._edf_model.omega_max, device=self.device),
                    torch.full((1,), self._edf_model.max_thrust, device=self.device),
                    self._fin_dispatch.cop_positions[None], torch.zeros(1, 4, 3, device=self.device),
                    torch.zeros(1, 3, device=self.device))
                drag = float(jet.forces[..., 2].sum())
        net_thrust = float(self._edf_model.compute_thrust(torch.tensor(self._edf_model.omega_max))) - drag
        if net_thrust <= mass * gravity:
            raise ValueError(f"Insufficient level thrust: {net_thrust:.3f} N available for {mass * gravity:.3f} N weight")
        return math.sqrt(mass * gravity / net_thrust)

    def step(self, action: Tensor) -> tuple[dict, Tensor, Tensor, Tensor, dict]:
        """Execute one RL step: pre-physics → decimated substeps → obs/reward/done.

        Args:
            action: (num_envs, 5) — 4 fin angles + 1 throttle.

        Returns:
            (obs_dict, reward, terminated, truncated, info)
        """
        action = action.to(self.device)
        self._pre_physics_step(action)
        self._max_downward_speed_step.zero_()
        self._landing_contact_force_step.zero_()
        self._unsafe_contact_step.zero_()

        self._battery_energy_step_wh.zero_()
        self._propulsive_delta_v_step.zero_()
        self._rotation.begin_step()
        navigation_before = self._body_iface.get_root_position().clone() if self._navigation else None
        navigation_active = self._contact_sm.state < int(ContactState.LANDED)

        for _ in range(self._config.decimation):
            rates_before = self._body_iface.get_angular_velocity_body_frd()
            rotation_active = self._contact_sm.state < int(ContactState.LANDED)
            downward_speed = (-self._body_iface.get_root_linear_velocity_world()[:, 2]).clamp(min=0.0)
            self._max_downward_speed_step = torch.maximum(
                self._max_downward_speed_step, downward_speed
            )
            self._apply_action()
            self._sim_scene.step()
            self._correct_freeflight_orientation()
            self._rotation.update(rates_before, self._body_iface.get_angular_velocity_body_frd(),
                                  self._config.physics_dt, rotation_active)
            landing_force, unsafe_contact = self._sensor_iface.read_contact_summary(
                self._contact_sm.min_contact_force
            )
            self._landing_contact_force_step = torch.maximum(
                self._landing_contact_force_step, landing_force
            )
            self._unsafe_contact_step |= unsafe_contact
            # Contact dwell is defined in physics frames. Updating only once
            # per decimated policy step made dwell_frames=15 mean 0.5 s rather
            # than 0.125 s at 120 Hz, and turned brief within-step bounces into
            # sustained contact. Advance the state machine on each PhysX report.
            self._update_contact_state(landing_force, unsafe_contact, downward_speed)
        self._step_count += 1
        if self._navigation:
            self._navigation.advance(navigation_before, self._body_iface.get_root_position(),
                self._body_iface.get_root_linear_velocity_world(), self._config.physics_dt*self._config.decimation,
                navigation_active)
        state_pre_reset = self._build_vehicle_state()
        terminated, time_out = self._get_dones(state_pre_reset)
        if self._navigation:
            self._navigation.finish_reward(state_pre_reset.position,terminated)
            state_pre_reset.mission_progress_step = self._navigation.step_progress
        reward = self._get_rewards(state_pre_reset)

        # Snapshot pre-reset vehicle state so eval/telemetry can attribute
        # terminal events to LANDED vs CRASHED and record touchdown
        # velocity / pad distance at the moment of termination. The auto-reset
        # below wipes the contact state machine back to AIRBORNE before
        # observations are read, which previously made `landed_fraction` and
        # `crashed_fraction` always zero in eval logs even when terminal
        # events were firing.
        info = {
            "contact_state_pre_reset": state_pre_reset.contact_state.clone(),
            "linear_vel_frd_pre_reset": state_pre_reset.linear_vel_frd.clone(),
            "angular_vel_frd_pre_reset": state_pre_reset.angular_vel_frd.clone(),
            "linear_vel_world_pre_reset": state_pre_reset.linear_vel_world.clone(),
            "position_pre_reset": state_pre_reset.position.clone(),
            "touchdown_speed_pre_reset": self._touchdown_speed.clone(),
            "motor_omega_pre_reset": state_pre_reset.motor_omega.clone(),
            "observation_pre_reset": self._get_observations(state_pre_reset)["policy"],
            "battery_energy_step_wh": self._battery_energy_step_wh.clone(),
            "propulsive_delta_v_step": self._propulsive_delta_v_step.clone(),
            "rotation_pre_reset": self._rotation.snapshot(),
            "mission_ready_to_land_pre_reset": (self._navigation.ready_to_land.clone() if self._navigation
                                                else torch.ones_like(terminated)),
            "waypoints_completed_pre_reset": (self._navigation.index.clone() if self._navigation
                                              else torch.zeros_like(terminated,dtype=torch.long)),
        }
        if self._battery_model is not None:
            info['battery_pre_reset'] = {k: v.clone() for k, v in self._battery_model.telemetry().items()}

        # Auto-reset terminated/timed-out envs
        reset_ids = (terminated | time_out).nonzero(as_tuple=False).squeeze(-1)
        if self._config.auto_reset and len(reset_ids) > 0:
            self._link_force_iface.clear_external_wrenches(reset_ids)
            self._reset_manager.reset_envs(reset_ids)
            self._contact_sensor.reset(reset_ids)
            self._touchdown_speed[reset_ids] = 0.0
            self._step_count[reset_ids] = 0
            self._rotation.reset(reset_ids)
            if self._navigation:
                self._navigation.reset(reset_ids,self._body_iface.get_root_position())

        obs = self._get_observations()
        truncated = time_out & ~terminated
        return obs, reward, terminated, truncated, info

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset all environments and return initial observations.

        Returns:
            (obs_dict, info)
        """
        if seed is not None:
            torch.manual_seed(seed)
        indices = torch.arange(self._config.num_envs, device=self.device, dtype=torch.int64)
        self._link_force_iface.clear_external_wrenches(indices)
        self._reset_manager.reset_envs(indices)
        self._contact_sensor.reset(indices)
        self._touchdown_speed.zero_()
        self._max_downward_speed_step.zero_()
        self._landing_contact_force_step.zero_()
        self._unsafe_contact_step.zero_()
        self._step_count.zero_()
        self._pending_actions = None
        self._battery_energy_step_wh.zero_()
        self._propulsive_delta_v_step.zero_()
        self._rotation.reset()
        if self._navigation:
            self._navigation.reset(indices,self._body_iface.get_root_position())
        return self._get_observations(), {}

    def close(self) -> None:
        """Release the simulation context."""
        if hasattr(self, "_sim_scene") and self._sim_scene is not None:
            self._sim_scene.close()
            self._sim_scene = None

    def render(self) -> None:
        """Pump Kit/render events without advancing an extra physics step."""
        if self._sim_scene is not None:
            self._sim_scene.render()

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
        if self._battery_model is None:
            new_omega = self._edf_model.update(omega_state, throttle, dt)
        else:
            new_omega = self._battery_model.update_motor(self._edf_model, omega_state, throttle, dt)
        self._reset_manager._omega_prev = omega_state.clone()
        self._reset_manager._omega_state = new_omega

        # Compute aero forces from measured PhysX joint angles, not the target cache.
        measured_fin_angles = self._body_iface.get_fin_joint_positions()
        rotor_fraction = (new_omega / max(float(self._edf_model.omega_max), 1.0)).clamp(0.0, 1.0)
        fin_dispatch = self._fin_dispatch.compute_body_frame_forces(measured_fin_angles, rotor_fraction)
        from tvc_env.common.frames import isaac_position_to_frd
        from tvc_env.common.quaternions import inverse, rotate_vector
        q = self._body_iface.get_root_quaternion_wxyz()
        pos = self._body_iface.get_root_position()
        cop_world = self._link_force_iface.get_fin_cop_positions_world(q, pos)
        cops = isaac_position_to_frd(rotate_vector(
            inverse(q)[:, None].expand(-1, 4, -1), cop_world - pos[:, None]))
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
        self._integration_q_before = q.clone()
        self._integration_w_before = body_ang_frd.clone()
        spin_axis = self._edf_model.thrust_axis.to(device=new_omega.device, dtype=new_omega.dtype)
        edf_output = self._edf_model.compute_output(
            new_omega,
            omega_state,
            body_ang_frd,
            dt,
            spin_axis=spin_axis,
            body_inertia=(None if dynamics_cfg.get('gyro_integration') in ('coupled_midpoint','coupled_cayley')
                          else self._locked_body_inertia),
        )
        raw_thrust = edf_output.thrust_force
        jet = None
        if self._coupled_jet is not None:
            jet = self._coupled_jet.compute(measured_fin_angles, rotor_fraction, new_omega,
                raw_thrust, cops, self._link_force_iface.get_cop_velocity_relative_body_frd(cop_world, q), body_ang_frd)
            fin_dispatch.forces_body = jet.forces if enable_fin_forces else torch.zeros_like(jet.forces)
            fin_dispatch.thrust_loss = fin_dispatch.forces_body[..., 2].sum(-1)
            edf_output.static_reaction_torque = jet.reaction_torque
        max_loss = self._max_fin_thrust_loss_fraction * raw_thrust
        thrust_loss = torch.minimum(fin_dispatch.thrust_loss.clamp(min=0.0), max_loss)
        thrust_loss = torch.minimum(thrust_loss, raw_thrust)
        if not enable_thrust_loss:
            thrust_loss = torch.zeros_like(thrust_loss)
        thrust = raw_thrust - thrust_loss
        # Drag acts downstream ON each vane. Applying it only as a scalar
        # subtraction at the EDF removed differential-drag moments and hinge
        # loads. Bound the per-vane contributions together and apply them once.
        drag_scale = thrust_loss / fin_dispatch.thrust_loss.clamp(min=1e-12)
        if jet is None:
            fin_dispatch.forces_body[:, :, 2] = fin_dispatch.tangential_force * drag_scale[:, None]
        else:
            # Finite streamtube model already bounds momentum and dissipates
            # energy. Do not replace its axial component with legacy drag.
            thrust_loss = fin_dispatch.forces_body[..., 2].sum(-1)
            thrust = raw_thrust - thrust_loss
        edf_force_body = torch.zeros(thrust.shape[0], 3, device=thrust.device)
        edf_force_body[:, 2] = -raw_thrust
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
        # Legacy linear damping is retained only for archived task reproduction.
        body_angular_damping = float(
            dynamics_cfg.get("body_angular_damping", self._body_angular_damping)
        )
        body_damping_torque = -body_angular_damping * body_ang_frd
        if jet is not None:
            from tvc_env.dynamics.coupled_jet import cylinder_rotational_drag
            body_damping_torque = cylinder_rotational_drag(body_ang_frd,
                self._body_geometry.get('length', .35), self._body_geometry.get('diameter', .12),
                self._body_geometry.get('cd_body', 1.))
        edf_torque_body = static_torque + dynamic_torque + gyro_torque + body_damping_torque

        q = self._body_iface.get_root_quaternion_wxyz()
        pos = self._body_iface.get_root_position()

        # Wind drag force in body-FRD frame
        wind_force_body = None
        if self._wind_model is not None and dynamics_cfg.get("enable_wind_force", True):
            lin_vel_w = self._body_iface.get_root_linear_velocity_world()
            wind_force_body = self._wind_model.compute_drag_force(lin_vel_w, q)
            self._wind_model.update_gust(dt)

        fin_torque_body = torch.linalg.cross(cops, fin_dispatch.forces_body).sum(dim=1)
        integration_correction = torch.zeros_like(gyro_torque)
        if dynamics_cfg.get('gyro_integration', 'rotor_midpoint') in ('coupled_midpoint','coupled_cayley'):
            from tvc_env.dynamics.rotor_reaction import compute_coupled_midpoint_torques
            # Include known non-gyro moments in the free-flight predictor.
            # PhysX still resolves actual link forces, hinges and contact;
            # this does not overwrite state or synthesize a control action.
            body_com_world = self._drone.data.body_com_pos_w[:, self._art_map.body_index]
            com_frd = isaac_position_to_frd(rotate_vector(inverse(q), body_com_world-pos))
            all_force = edf_force_body + fin_dispatch.forces_body.sum(dim=1)
            if wind_force_body is not None:
                all_force = all_force + wind_force_body
            external_torque = (static_torque + dynamic_torque + body_damping_torque
                               + fin_torque_body - torch.linalg.cross(com_frd, all_force))
            rotor_scale = (self._edf_model.gyro_torque_scale * float(dynamics_cfg.get('edf_gyro_torque_scale', 1.0))
                           if dynamics_cfg.get('enable_edf_gyro_torque', True) else 0.0)
            rotor_h = spin_axis[None] * (.5*(omega_state+new_omega)*self._edf_model.rotor_inertia*rotor_scale)[:,None]
            gyro_torque, integration_correction = compute_coupled_midpoint_torques(
                body_ang_frd, self._locked_body_inertia, rotor_h, external_torque, dt,
                correct_physx_projection=dynamics_cfg.get('gyro_integration')=='coupled_cayley')
            edf_torque_body = static_torque + dynamic_torque + body_damping_torque + gyro_torque + integration_correction
        self._last_dynamics_debug = {
            "fin_force_body_frd_N": fin_dispatch.forces_body.sum(dim=1).detach(),
            "fin_torque_body_frd_Nm": fin_torque_body.detach(),
            "thrust_loss_N": thrust_loss.detach(),
            "edf_raw_thrust_N": raw_thrust.detach(),
            "edf_applied_thrust_N": thrust.detach(),
            "edf_static_torque_body_frd_Nm": static_torque.detach(),
            "edf_dynamic_torque_body_frd_Nm": dynamic_torque.detach(),
            "edf_gyro_torque_body_frd_Nm": gyro_torque.detach(),
            "angular_integration_correction_body_frd_Nm": integration_correction.detach(),
            "body_damping_torque_body_frd_Nm": body_damping_torque.detach(),
            "edf_total_torque_body_frd_Nm": edf_torque_body.detach(),
            "wind_force_body_frd_N": (
                wind_force_body.detach()
                if wind_force_body is not None
                else torch.zeros_like(edf_force_body).detach()
            ),
        }
        if jet is not None:
            self._last_dynamics_debug.update(
                jet_mass_flow_kg_s=jet.mass_flow_per_fin.sum(-1).detach(),
                jet_swirl_power_w=jet.swirl_power.detach(),
                vane_dissipated_power_w=jet.dissipated_power.detach(),
                jet_incoming_velocity_body_frd_m_s=jet.incoming_velocity.detach())

        propulsive_force = edf_force_body + fin_dispatch.forces_body.sum(dim=1)
        self._propulsive_delta_v_step += propulsive_force.norm(dim=-1) / self._vehicle_mass * dt
        if self._battery_model is not None:
            self._battery_energy_step_wh += self._battery_model.power_w * dt / 3600

        self._wrench_dispatch.dispatch(
            fin_dispatch.forces_body,
            cops,
            q,
            pos,
            edf_force_body,
            edf_torque_body,
            wind_force_body,
        )

    def _correct_freeflight_orientation(self):
        """Complete the coupled Lie-midpoint step outside external contacts.

        PhysX retains translation, joint dynamics and all contact impulses.
        Its end-rate angular drift is first order and inconsistent with the
        midpoint virtual rotor; replace that drift with the paired Cayley
        orientation, retaining the actual post-solver body rates. Preserve
        COM position and linear momentum, including off-center COM cases.
        Any measured external contact leaves the entire PhysX pose untouched.
        This is a numerical integration bridge, not an attitude correction
        toward a target. Test18 high-rate conservation and contact regressions
        guard this split; four 1g moving vanes remain a small splitting error.
        """
        if self._config.config.get('dynamics',{}).get('gyro_integration') != 'coupled_cayley':
            return
        from tvc_env.dynamics.rotor_reaction import cayley_body_orientation
        from tvc_env.common.frames import frd_to_isaac
        from tvc_env.common.quaternions import rotate_vector
        contact_force, unsafe = self._sensor_iface.read_contact_summary(1e-8)
        free = (contact_force <= 1e-8) & ~unsafe
        ids = free.nonzero(as_tuple=False).squeeze(-1)
        if not len(ids):
            return
        rates = self._body_iface.get_angular_velocity_body_frd()
        q = cayley_body_orientation(self._integration_q_before,self._integration_w_before,
                                   rates,self._config.physics_dt)
        body_id = self._art_map.body_index
        com_world = self._drone.data.body_com_pos_w[:,body_id].clone()
        com_local = self._drone.data.body_com_pos_b[:,body_id]
        origin = com_world-rotate_vector(q,com_local)
        velocity = self._body_iface.get_root_linear_velocity_world()
        angular = rotate_vector(q,frd_to_isaac(rates))
        self._body_iface.set_root_state(origin[ids],q[ids],velocity[ids],angular[ids],env_ids=ids)

    # ---- Observation / Reward / Done ----

    def _get_observations(self, state: VehicleState | None = None) -> dict:
        """Assemble observation dict with 'policy' key."""
        from tvc_env.envs.observations import apply_sensor_noise, assemble_observation

        if state is None:
            state = self._build_vehicle_state()
        obs = assemble_observation(state, self._navigation.goal if self._navigation else self._target_position, self._omega_max)
        obs = apply_sensor_noise(obs, self._config.config)
        if self._config.config.get('env', {}).get('observe_battery', False):
            obs = torch.cat([obs, self._battery_model.observation()], dim=-1)
        if self._navigation:
            obs = torch.cat([obs,self._navigation.observation(state.quaternion_wxyz)],dim=-1)

        return {"policy": obs}

    def _get_rewards(self, state: VehicleState | None = None) -> Tensor:
        """Compute total reward via reward_registry."""
        from tvc_env.envs.reward_registry import compute_total_reward

        if state is None:
            state = self._build_vehicle_state()
        task_cfg = self._config.config
        reward_weights = task_cfg.get("task", task_cfg).get("reward", {})
        return compute_total_reward(reward_weights, state, task_cfg)

    def _get_dones(self, state: VehicleState | None = None) -> tuple[Tensor, Tensor]:
        """Evaluate termination conditions.

        Returns:
            Tuple (terminated, time_out) — both bool tensors (num_envs,).
        """
        from tvc_env.envs.terminations import check_all_terminations

        if state is None:
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

    def _update_contact_state(
        self,
        contact_force: Tensor,
        unsafe_contact: Tensor,
        downward_speed: Tensor,
    ) -> None:
        """Advance landing/crash state from one PhysX contact frame."""
        in_contact = contact_force >= self._contact_sm.min_contact_force

        previous_state = self._contact_sm.state
        first_contact = in_contact & (previous_state == ContactState.AIRBORNE)
        # Landing sweep v2 reported several zero-speed landings despite a
        # descent. Contact-force flicker/bounces returned the detector to
        # AIRBORNE, overwriting the initial impact with a later settled speed.
        # Preserve the worst arrival across all contacts in this episode so
        # a bounce cannot turn a hard landing into a soft success.
        self._touchdown_speed[first_contact] = torch.maximum(
            self._touchdown_speed[first_contact], downward_speed[first_contact]
        )
        impact_speed = torch.where(
            first_contact,
            downward_speed,
            self._touchdown_speed,
        )
        ang_vel_frd = self._body_iface.get_angular_velocity_body_frd()
        ang_rate = ang_vel_frd.norm(dim=-1)
        q = self._body_iface.get_root_quaternion_wxyz()

        is_crashed = unsafe_contact | self._crash_detector.check_impact_speed(impact_speed, first_contact)
        is_crashed = is_crashed | self._crash_detector.check_tilt_at_contact(q, in_contact)
        is_crashed = is_crashed | self._crash_detector.check_angular_rate_at_contact(ang_rate, in_contact)
        is_crashed = is_crashed | self._crash_detector.check_excessive_tilt(q)

        self._contact_sm.update(in_contact, is_crashed, contact_force)

    def _build_vehicle_state(self) -> VehicleState:
        """Collect all state into a VehicleState dataclass."""
        device = self._drone.device
        pos = self._body_iface.get_root_position()
        q = self._body_iface.get_root_quaternion_wxyz()
        lin_vel_w = self._body_iface.get_root_linear_velocity_world()
        ang_vel_w = self._body_iface.get_root_angular_velocity_world()
        lin_vel_frd = self._body_iface.get_linear_velocity_body_frd(lin_vel_w, q)
        ang_vel_frd = self._body_iface.get_angular_velocity_body_frd(ang_vel_w, q)
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
            touchdown_speed=self._touchdown_speed,
            battery_energy_step_wh=self._battery_energy_step_wh,
            propulsive_delta_v_step=self._propulsive_delta_v_step,
            excess_rotation_cost_step_s=self._rotation.step_cost_s,
            rotation_peak_rate_rad_s=self._rotation.peak_rate_rad_s,
            mission_ready_to_land=self._navigation.ready_to_land if self._navigation else None,
            mission_progress_step=self._navigation.step_progress if self._navigation else None,
            waypoint_completion_step=self._navigation.step_completed if self._navigation else None,
            path_tracking_cost_step=self._navigation.step_path_cost if self._navigation else None,
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
            shape=((28 if self._config.config.get('env', {}).get('observe_battery', False) else 24)
                   + (15 if self._navigation else 0),),
            dtype=np.float32,
        )
