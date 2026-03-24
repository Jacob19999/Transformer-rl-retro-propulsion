"""
Debug visualization manager for the TVC environment.

Uses Isaac Sim debug draw line primitives for force/torque vectors and a small
omni.ui window for HUD/state-vector output during single-environment visual
validation runs.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from tvc_env.common.frames import frd_to_isaac
from tvc_env.common.quaternions import rotate_vector


class GizmoManager:
    """Manages all debug visualizations for single-env mode."""

    def __init__(
        self,
        config: dict[str, Any],
        num_envs: int = 1,
        enabled: bool = True,
    ):
        self._config = config.get("gizmos", config)
        self._enabled = enabled and (num_envs == 1)
        self._draw = None
        self._ui = None
        self._initialized = False
        self._hud_window = None
        self._hud_title_model = None
        self._hud_body_model = None

        if self._enabled:
            self._try_initialize()

    def _try_initialize(self) -> None:
        """Try to initialize Isaac Sim debug draw and HUD APIs."""
        try:
            from isaacsim.util.debug_draw import _debug_draw
            import omni.ui as ui

            self._draw = _debug_draw.acquire_debug_draw_interface()
            self._ui = ui
            self._initialized = True
            print("[GizmoManager] initialized (debug_draw + omni.ui ready)")
        except Exception as exc:
            import traceback
            self._initialized = False
            print(f"[GizmoManager] initialization failed: {exc}")
            traceback.print_exc()

    def update(
        self,
        position: Tensor,           # (1, 3) body position in world frame
        quaternion_wxyz: Tensor,    # (1, 4)
        fin_forces: Tensor,         # (1, 4, 3) fin forces in body-FRD
        thrust: float,              # EDF thrust magnitude (N)
        cop_positions: Tensor,      # (4, 3) COP positions in body-FRD
        contact_state: int,
        height: float,
        fin_angles: Tensor,         # (1, 4)
        motor_rpm: float,
        total_reward: float,
        target_position: Tensor,    # (3,)
        task_name: str = "hover",
        total_aero_force: Tensor | None = None,   # (3,) in body-FRD
        reaction_torque: Tensor | None = None,    # (3,) in body-FRD
    ) -> None:
        """Update all active line-drawn gizmos."""
        del contact_state, height, fin_angles, motor_rpm, total_reward, target_position, task_name

        if not self._enabled:
            return

        # Lazy init: debug_draw may not be ready at scene-build time
        if not self._initialized:
            self._try_initialize()

        if not self._initialized or self._draw is None:
            return

        try:
            self._draw.clear_lines()
        except Exception:
            return

        for fn in (
            lambda: self._update_body_axes(position, quaternion_wxyz),
            lambda: self._update_thrust_vector(position, quaternion_wxyz, thrust),
            lambda: self._update_fin_force_arrows(position, quaternion_wxyz, fin_forces, cop_positions),
            lambda: self._maybe_update_total_aero_force(position, quaternion_wxyz, total_aero_force),
            lambda: self._maybe_update_reaction_torque(position, quaternion_wxyz, reaction_torque),
        ):
            try:
                fn()
            except Exception:
                pass

    def _body_vector_to_world(self, quaternion_wxyz: Tensor, vector_body_frd: Tensor) -> Tensor:
        """Rotate a body-FRD vector into Isaac world coordinates."""
        body_isaac = frd_to_isaac(vector_body_frd)
        return rotate_vector(quaternion_wxyz[0], body_isaac)

    def _body_point_to_world(
        self,
        position: Tensor,
        quaternion_wxyz: Tensor,
        point_body_frd: Tensor,
    ) -> Tensor:
        """Transform a body-FRD point into Isaac world coordinates."""
        return position[0] + self._body_vector_to_world(quaternion_wxyz, point_body_frd)

    def _draw_vector(
        self,
        start: Tensor,
        end: Tensor,
        color: tuple[float, float, float, float],
        thickness: float,
    ) -> None:
        """Draw a single viewport line."""
        self._draw.draw_lines(
            [tuple(float(v) for v in start.tolist())],
            [tuple(float(v) for v in end.tolist())],
            [color],
            [float(thickness)],
        )

    def _update_body_axes(self, position: Tensor, quaternion_wxyz: Tensor) -> None:
        """Render body frame axes at current position."""
        cfg = self._config.get("body_axes", {})
        if not cfg.get("enabled", True):
            return
        origin = position[0]
        scale = float(cfg.get("scale", 0.3))
        axis_specs = [
            (torch.tensor([1.0, 0.0, 0.0], device=origin.device, dtype=origin.dtype), (1.0, 0.2, 0.2, 1.0)),
            (torch.tensor([0.0, 1.0, 0.0], device=origin.device, dtype=origin.dtype), (0.2, 1.0, 0.2, 1.0)),
            (torch.tensor([0.0, 0.0, 1.0], device=origin.device, dtype=origin.dtype), (0.2, 0.6, 1.0, 1.0)),
        ]
        for axis_body, color in axis_specs:
            end = origin + self._body_vector_to_world(quaternion_wxyz, axis_body * scale)
            self._draw_vector(origin, end, color, thickness=2.0)

    def _update_thrust_vector(self, position: Tensor, quaternion_wxyz: Tensor, thrust: float) -> None:
        """Render EDF thrust vector arrow."""
        cfg = self._config.get("thrust_vector", {})
        if not cfg.get("enabled", True):
            return
        origin = position[0]
        scale = float(cfg.get("scale", 0.01))
        color = tuple(cfg.get("color", [1.0, 0.5, 0.0])) + (1.0,)
        thrust_body = torch.tensor([0.0, 0.0, -thrust * scale], device=origin.device, dtype=origin.dtype)
        end = origin + self._body_vector_to_world(quaternion_wxyz, thrust_body)
        self._draw_vector(origin, end, color, thickness=3.0)

    def _update_fin_force_arrows(
        self,
        position: Tensor,
        quaternion_wxyz: Tensor,
        fin_forces: Tensor,
        cop_positions: Tensor,
    ) -> None:
        """Render per-fin force arrows at the fin COPs."""
        cfg = self._config.get("fin_force_arrows", {})
        if not cfg.get("enabled", True):
            return
        scale = float(cfg.get("scale", 0.05))
        color = tuple(cfg.get("color", [0.0, 0.5, 1.0])) + (1.0,)
        for idx in range(fin_forces.shape[1]):
            start = self._body_point_to_world(position, quaternion_wxyz, cop_positions[idx].to(position.device))
            end = start + self._body_vector_to_world(quaternion_wxyz, fin_forces[0, idx].to(position.device) * scale)
            self._draw_vector(start, end, color, thickness=2.0)

    def _maybe_update_total_aero_force(
        self,
        position: Tensor,
        quaternion_wxyz: Tensor,
        total_aero_force: Tensor | None,
    ) -> None:
        """Render the summed aerodynamic force if provided."""
        if total_aero_force is None:
            return
        cfg = self._config.get("total_aero_force", {})
        if not cfg.get("enabled", True):
            return
        origin = position[0]
        scale = float(cfg.get("scale", 0.01))
        color = tuple(cfg.get("color", [0.0, 1.0, 1.0])) + (1.0,)
        end = origin + self._body_vector_to_world(quaternion_wxyz, total_aero_force.to(origin.device) * scale)
        self._draw_vector(origin, end, color, thickness=3.0)

    def _maybe_update_reaction_torque(
        self,
        position: Tensor,
        quaternion_wxyz: Tensor,
        reaction_torque: Tensor | None,
    ) -> None:
        """Render the aggregate reaction torque if provided."""
        if reaction_torque is None:
            return
        cfg = self._config.get("reaction_torque", {})
        if not cfg.get("enabled", True):
            return
        origin = position[0]
        scale = float(cfg.get("scale", 0.1))
        color = tuple(cfg.get("color", [1.0, 0.0, 0.5])) + (1.0,)
        end = origin + self._body_vector_to_world(quaternion_wxyz, reaction_torque.to(origin.device) * scale)
        self._draw_vector(origin, end, color, thickness=3.0)

    def _ensure_hud_window(self) -> None:
        """Create the HUD window lazily when UI is available."""
        if self._ui is None or self._hud_window is not None:
            return

        ui = self._ui
        window = ui.Window(
            "TVC Visual HUD",
            width=520,
            height=360,
            flags=ui.WINDOW_FLAGS_NO_SCROLLBAR | ui.WINDOW_FLAGS_NO_DOCKING,
        )
        with window.frame:
            with ui.VStack(style={"margin": 6}, spacing=6):
                self._hud_title_model = ui.SimpleStringModel("TVC Visual HUD")
                ui.StringField(self._hud_title_model, height=28)
                self._hud_body_model = ui.SimpleStringModel("")
                ui.StringField(self._hud_body_model, multiline=True, height=300)
        self._hud_window = window

    def log_hud(
        self,
        pos_error: float,
        tilt_deg: float,
        body_rate: float,
        motor_rpm: float,
        fin_angles: list[float],
        total_reward: float,
        contact_state: str,
        task_name: str,
        detail_lines: list[str] | None = None,
        title: str | None = None,
        print_terminal: bool = True,
    ) -> None:
        """Write summary telemetry to the HUD window and optionally to the terminal."""
        if not self._enabled:
            return

        # Lazy init in case _try_initialize() was called before omni.ui was ready
        if not self._initialized:
            self._try_initialize()

        summary = (
            f"[HUD] pos_err={pos_error:.3f}m tilt={tilt_deg:.1f}deg "
            f"rate={body_rate:.2f}rad/s rpm={motor_rpm:.0f} "
            f"fins={[f'{a:.2f}' for a in fin_angles]} "
            f"reward={total_reward:.2f} state={contact_state} task={task_name}"
        )
        body_lines = [summary]
        if detail_lines:
            body_lines.extend(detail_lines)

        self._ensure_hud_window()
        if self._hud_title_model is not None:
            self._hud_title_model.set_value(title or f"{task_name} visual validation")
        if self._hud_body_model is not None:
            self._hud_body_model.set_value("\n".join(body_lines))

        if print_terminal:
            print(summary)
            if detail_lines:
                for line in detail_lines:
                    print(f"       {line}")

    def disable(self) -> None:
        """Disable all gizmos."""
        self._enabled = False
