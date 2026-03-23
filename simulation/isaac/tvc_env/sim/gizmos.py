"""
Debug visualization manager for the TVC environment.

Dual API per research decision R6:
  - VisualizationMarkers: 3D shapes (body axes, COM marker, thrust vector arrows,
    fin force arrows, contact normals)
  - debug_draw: line primitives for contact normals and HUD overlay

Auto-disables when num_envs > 1.

Requires Isaac Lab 2.3.2 runtime for actual visualization.
This module stubs gracefully if Isaac Sim is not available.
"""

from __future__ import annotations
import torch
from torch import Tensor
from typing import Any


class GizmoManager:
    """Manages all debug visualizations for single-env mode."""

    def __init__(
        self,
        config: dict[str, Any],
        num_envs: int = 1,
        enabled: bool = True,
    ):
        """
        Args:
            config: Parsed gizmos.yaml config dict.
            num_envs: Number of environments. Gizmos disabled if > 1.
            enabled: Master enable switch.
        """
        self._config = config.get("gizmos", config)
        self._enabled = enabled and (num_envs == 1)
        self._markers = {}  # name → VisualizationMarkers object
        self._draw = None

        if self._enabled:
            self._try_initialize()

    def _try_initialize(self) -> None:
        """Try to initialize Isaac Lab visualization markers. Fails silently."""
        try:
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.utils.math import debug_draw
            self._draw = debug_draw
            self._initialized = True
        except ImportError:
            self._initialized = False

    def update(
        self,
        position: Tensor,           # (1, 3) body position in world frame
        quaternion_wxyz: Tensor,    # (1, 4)
        fin_forces: Tensor,         # (1, 4, 3) fin forces in body frame
        thrust: float,              # EDF thrust magnitude (N)
        cop_positions: Tensor,      # (4, 3) COP positions in body frame
        contact_state: int,
        height: float,
        fin_angles: Tensor,         # (1, 4)
        motor_rpm: float,
        total_reward: float,
        target_position: Tensor,    # (3,)
        task_name: str = "hover",
    ) -> None:
        """Update all active gizmos.

        Args:
            (see parameter names above)
        """
        if not self._enabled or not getattr(self, '_initialized', False):
            return

        # Each gizmo update is wrapped in try/except to avoid crashing the env
        try:
            self._update_body_axes(position, quaternion_wxyz)
        except Exception:
            pass

        try:
            self._update_thrust_vector(position, quaternion_wxyz, thrust)
        except Exception:
            pass

        try:
            self._update_fin_force_arrows(position, quaternion_wxyz, fin_forces, cop_positions)
        except Exception:
            pass

    def _update_body_axes(self, position: Tensor, quaternion_wxyz: Tensor) -> None:
        """Render body frame axes at current position."""
        cfg = self._config.get("body_axes", {})
        if not cfg.get("enabled", True):
            return
        # FrameMarkerCfg visualization via VisualizationMarkers
        # Actual implementation depends on Isaac Lab API

    def _update_thrust_vector(self, position: Tensor, quaternion_wxyz: Tensor, thrust: float) -> None:
        """Render EDF thrust vector arrow."""
        cfg = self._config.get("thrust_vector", {})
        if not cfg.get("enabled", True):
            return
        scale = cfg.get("scale", 0.01)
        # Arrow from body COM in thrust axis direction, scaled by thrust magnitude

    def _update_fin_force_arrows(
        self,
        position: Tensor,
        quaternion_wxyz: Tensor,
        fin_forces: Tensor,
        cop_positions: Tensor,
    ) -> None:
        """Render per-fin force arrows at COP positions."""
        cfg = self._config.get("fin_force_arrows", {})
        if not cfg.get("enabled", True):
            return
        scale = cfg.get("scale", 0.05)
        # Arrow per fin from COP position, direction = force vector, length = magnitude * scale

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
    ) -> None:
        """Log HUD telemetry values (printed to console if debug_draw not available)."""
        if not self._enabled:
            return
        # In full Isaac Sim environment, this would draw text overlay in viewport
        # For offline testing, print to console
        print(
            f"[HUD] pos_err={pos_error:.3f}m tilt={tilt_deg:.1f}° "
            f"rate={body_rate:.2f}rad/s rpm={motor_rpm:.0f} "
            f"fins={[f'{a:.2f}' for a in fin_angles]} "
            f"reward={total_reward:.2f} state={contact_state} task={task_name}"
        )

    def disable(self) -> None:
        """Disable all gizmos."""
        self._enabled = False
