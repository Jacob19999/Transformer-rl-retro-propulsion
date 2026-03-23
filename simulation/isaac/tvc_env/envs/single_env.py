"""
Single-environment debug wrapper with gizmos enabled.

Subclasses TVCDirectRLEnv with gizmos initialized, updates gizmos in post-step,
and provides keyboard/gamepad input handling for manual control.

Only active when num_envs == 1 (gizmos auto-disabled for multi-env).
"""

from __future__ import annotations
from pathlib import Path
import torch

from tvc_env.envs.direct_rl_env import TVCDirectRLEnv
from tvc_env.envs.base_env import BaseEnvConfig
from tvc_env.sim.gizmos import GizmoManager


class SingleEnvDebug(TVCDirectRLEnv):
    """Single-environment debug wrapper with visualization gizmos."""

    def __init__(self, config: BaseEnvConfig, **kwargs):
        assert config.num_envs == 1, "SingleEnvDebug requires num_envs == 1"
        super().__init__(config, **kwargs)
        self._gizmo_manager: GizmoManager | None = None
        self._manual_action = None

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

        obs, rewards, terminated, truncated, info = super().step(actions)
        self._update_gizmos(rewards)
        return obs, rewards, terminated, truncated, info

    def _update_gizmos(self, rewards=None) -> None:
        """Update all debug gizmos with current state."""
        if self._gizmo_manager is None or not self._gizmo_manager._enabled:
            return
        try:
            from tvc_env.common.quaternions import to_euler
            import math

            state = self._build_vehicle_state()
            roll, pitch, _ = to_euler(state.quaternion_wxyz)
            tilt_deg = math.degrees(float(torch.sqrt(roll ** 2 + pitch ** 2)[0]))
            pos_error = (state.position[0] - self._target_position.to(state.position.device)).norm().item()
            body_rate = state.angular_vel_frd[0].norm().item()
            rpm = state.motor_omega[0].item()
            fin_angles_list = state.fin_angles[0].tolist()
            contact_name = {0: "AIRBORNE", 1: "CANDIDATE", 2: "LANDED", 3: "CRASHED"}.get(
                int(state.contact_state[0]), "UNKNOWN"
            )
            total_reward = float(rewards[0]) if rewards is not None else 0.0

            self._gizmo_manager.log_hud(
                pos_error=pos_error,
                tilt_deg=tilt_deg,
                body_rate=body_rate,
                motor_rpm=rpm,
                fin_angles=fin_angles_list,
                total_reward=total_reward,
                contact_state=contact_name,
                task_name=self._config.task_name,
            )
        except Exception:
            pass

    def set_manual_action(self, action) -> None:
        """Set a manual override action (e.g., from keyboard)."""
        self._manual_action = action

    def clear_manual_action(self) -> None:
        """Clear manual action override."""
        self._manual_action = None
