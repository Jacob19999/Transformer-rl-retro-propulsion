"""
Single-environment debug wrapper with gizmos enabled.

Subclasses TVCDirectRLEnv with gizmos initialized, updates gizmos in post-step,
and provides richer HUD output for scripted visual-validation runs.
"""

from __future__ import annotations

from pathlib import Path

import torch

from tvc_env.envs.base_env import BaseEnvConfig
from tvc_env.envs.direct_rl_env import TVCDirectRLEnv
from tvc_env.sim.gizmos import GizmoManager


class SingleEnvDebug(TVCDirectRLEnv):
    """Single-environment debug wrapper with visualization gizmos."""

    def __init__(self, config: BaseEnvConfig, **kwargs):
        assert config.num_envs == 1, "SingleEnvDebug requires num_envs == 1"
        super().__init__(config, **kwargs)
        self._gizmo_manager: GizmoManager | None = None
        self._manual_action = None
        self._latest_action = None
        self._visual_context: dict | None = None

    def _setup_scene(self) -> None:
        """Setup scene and initialize gizmo manager."""
        super()._setup_scene()

        sim_root = Path(__file__).parents[2]
        gizmos_config_path = sim_root / "configs/debug/gizmos.yaml"

        if gizmos_config_path.exists():
            import yaml

            with open(gizmos_config_path, "r") as f:
                gizmos_config = yaml.safe_load(f)
        else:
            gizmos_config = {}

        self._gizmo_manager = GizmoManager(
            config=gizmos_config,
            num_envs=self._config.num_envs,
            enabled=self._config.gizmos_enabled,
        )

    def step(self, actions):
        """Step environment and update gizmos."""
        if self._manual_action is not None:
            actions = self._manual_action

        self._latest_action = actions.clone()
        obs, rewards, terminated, truncated, info = super().step(actions)
        self._update_gizmos(rewards)
        return obs, rewards, terminated, truncated, info

    def set_visual_context(
        self,
        scenario_name: str,
        step_index: int,
        episode_steps: int,
        notes: list[str] | None = None,
        print_terminal: bool = True,
    ) -> None:
        """Set scenario metadata for richer HUD output during visual validation."""
        self._visual_context = {
            "scenario_name": scenario_name,
            "step_index": step_index,
            "episode_steps": episode_steps,
            "notes": notes or [],
            "print_terminal": print_terminal,
        }

    def clear_visual_context(self) -> None:
        """Clear scenario metadata after a visual episode completes."""
        self._visual_context = None

    def _update_gizmos(self, rewards=None) -> None:
        """Update all debug gizmos with current state."""
        if self._gizmo_manager is None or not self._gizmo_manager._enabled:
            return

        try:
            import math
            from tvc_env.common.quaternions import to_euler

            state = self._build_vehicle_state()
            roll, pitch, _ = to_euler(state.quaternion_wxyz)
            tilt_deg = math.degrees(float(torch.sqrt(roll ** 2 + pitch ** 2)[0]))
            pos_error = (
                state.position[0] - self._target_position.to(state.position.device)
            ).norm().item()
            body_rate = state.angular_vel_frd[0].norm().item()
            rpm = state.motor_omega[0].item()
            fin_angles_list = state.fin_angles[0].tolist()
            contact_name = {
                0: "AIRBORNE",
                1: "CANDIDATE",
                2: "LANDED",
                3: "CRASHED",
            }.get(int(state.contact_state[0].item()), "UNKNOWN")
            total_reward = float(rewards[0]) if rewards is not None else 0.0

            action = self._latest_action
            if action is None:
                action = torch.zeros(1, 5, device=state.position.device)
            action = action.to(state.position.device)

            throttle = action[:, 4].clamp(0.0, 1.0)
            fin_forces, cops = self._fin_dispatch.compute_body_frame_forces(
                self._reset_manager.servo_state,
                throttle,
            )
            edf_output = self._edf_model.compute_output(
                state.motor_omega,
                self._reset_manager.omega_prev,
                state.angular_vel_frd,
                self._config.physics_dt,
                self._edf_model.thrust_axis.to(state.position.device),
            )
            total_aero_force = fin_forces[0].sum(dim=0)
            reaction_torque = (
                edf_output.static_reaction_torque[0]
                + edf_output.dynamic_spool_torque[0]
                + edf_output.gyro_precession_torque[0]
            )

            detail_lines = None
            title = None
            print_terminal = True
            if self._visual_context is not None:
                ctx = self._visual_context

                def _fmt_vec(vec: torch.Tensor) -> str:
                    return "[" + ", ".join(f"{float(v):+.3f}" for v in vec.tolist()) + "]"

                detail_lines = [
                    f"step={ctx['step_index'] + 1}/{ctx['episode_steps']}",
                    f"pos_w={_fmt_vec(state.position[0])}",
                    f"quat_wxyz={_fmt_vec(state.quaternion_wxyz[0])}",
                    f"lin_vel_frd={_fmt_vec(state.linear_vel_frd[0])}",
                    f"ang_vel_frd={_fmt_vec(state.angular_vel_frd[0])}",
                    f"action={_fmt_vec(action[0])}",
                    f"fin_aero_sum_frd={_fmt_vec(total_aero_force)}",
                    f"reaction_tau_frd={_fmt_vec(reaction_torque)}",
                ]
                detail_lines.extend(ctx.get("notes", []))
                title = ctx["scenario_name"]
                print_terminal = bool(ctx.get("print_terminal", True))

            self._gizmo_manager.update(
                position=state.position,
                quaternion_wxyz=state.quaternion_wxyz,
                fin_forces=fin_forces,
                thrust=float(self._edf_model.compute_thrust(state.motor_omega)[0].item()),
                cop_positions=cops,
                contact_state=int(state.contact_state[0].item()),
                height=float(state.height[0].item()),
                fin_angles=state.fin_angles,
                motor_rpm=rpm,
                total_reward=total_reward,
                target_position=self._target_position.to(state.position.device),
                task_name=self._config.task_name,
                total_aero_force=total_aero_force,
                reaction_torque=reaction_torque,
            )

            self._gizmo_manager.log_hud(
                pos_error=pos_error,
                tilt_deg=tilt_deg,
                body_rate=body_rate,
                motor_rpm=rpm,
                fin_angles=fin_angles_list,
                total_reward=total_reward,
                contact_state=contact_name,
                task_name=self._config.task_name,
                detail_lines=detail_lines,
                title=title,
                print_terminal=print_terminal,
            )
        except Exception as exc:
            import traceback
            print(f"[SingleEnvDebug] _update_gizmos error: {exc}")
            traceback.print_exc()

    def set_manual_action(self, action) -> None:
        """Set a manual override action (e.g., from keyboard)."""
        self._manual_action = action

    def clear_manual_action(self) -> None:
        """Clear manual action override."""
        self._manual_action = None
