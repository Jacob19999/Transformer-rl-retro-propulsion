"""
debug_draw.py — Force/torque gizmo visualisation using isaacsim.util.debug_draw.

Usage (inside any Isaac Sim script or task)::

    from simulation.isaac.debug_draw import ForceGizmoDrawer

    gizmo = ForceGizmoDrawer(force_scale=0.25, max_envs=1)

    # inside your step loop:
    gizmo.draw(
        fin_origins_w=fin_origins_w,   # (N, 4, 3)  world-frame fin link positions
        fin_forces_w=fin_forces_w,     # (N, 4, 3)  world-frame aero forces per fin
        body_origin_w=body_origin_w,   # (N, 3)     root body world position
        thrust_w=thrust_w,             # (N, 3)     world-frame thrust vector
        body_torque_w=torques_w,       # (N, 3)     optional – body torques (orange)
    )

Arrow colour key
----------------
Fin forces  : Red / Green / Blue / Yellow  (fin order 0-3)
Thrust      : Cyan
Body torque : Orange

The draw interface is acquired lazily; if the extension is unavailable (e.g. in
headless unit tests) all calls are silently no-ops.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Debug-draw interface — lazy acquisition
# ---------------------------------------------------------------------------
_draw_iface = None
_draw_available: bool | None = None


def _get_draw():
    global _draw_iface, _draw_available
    if _draw_available is None:
        for module_name in (
            "isaacsim.util.debug_draw._debug_draw",
            "omni.isaac.debug_draw._debug_draw",
        ):
            try:
                import importlib
                mod = importlib.import_module(module_name)
                _draw_iface = mod.acquire_debug_draw_interface()
                _draw_available = True
                return _draw_iface
            except Exception:
                continue
        _draw_available = False
    return _draw_iface if _draw_available else None


# ---------------------------------------------------------------------------
# Per-entity colour palette
# ---------------------------------------------------------------------------
# Fin colours follow the order in fin_forces_w[:, i, :].  The caller is
# responsible for consistent ordering; colour is purely cosmetic.
FIN_COLORS: list[tuple[float, float, float, float]] = [
    (1.00, 0.25, 0.25, 1.0),   # 0 — red
    (0.25, 1.00, 0.25, 1.0),   # 1 — green
    (0.25, 0.55, 1.00, 1.0),   # 2 — blue
    (1.00, 0.90, 0.10, 1.0),   # 3 — yellow
]
THRUST_COLOR:  tuple[float, float, float, float] = (0.00, 0.90, 1.00, 1.0)  # cyan
TORQUE_COLOR:  tuple[float, float, float, float] = (1.00, 0.50, 0.00, 1.0)  # orange
LABEL_COLOR:   tuple[float, float, float, float] = (1.00, 1.00, 1.00, 0.8)  # white


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _arrow_segments(
    ox: float, oy: float, oz: float,
    vx: float, vy: float, vz: float,
    color: tuple[float, float, float, float],
    *,
    head_fraction: float = 0.22,
    head_spread: float = 0.35,   # radians half-angle for the arrowhead
) -> tuple[list, list, list]:
    """Return (starts, ends, colors) for a 3-segment arrow (shaft + 2 head fins)."""
    tip = (ox + vx, oy + vy, oz + vz)
    starts: list = [(ox, oy, oz)]
    ends:   list = [tip]
    colors: list = [color]

    length = math.sqrt(vx * vx + vy * vy + vz * vz)
    if length < 1e-7:
        return starts, ends, colors

    # Pick a reference vector not parallel to the arrow direction.
    ux, uy, uz = vx / length, vy / length, vz / length
    ref = (0.0, 0.0, 1.0) if abs(uz) < 0.9 else (1.0, 0.0, 0.0)

    # Perpendicular vector (arrow × ref), normalised.
    px = uy * ref[2] - uz * ref[1]
    py = uz * ref[0] - ux * ref[2]
    pz = ux * ref[1] - uy * ref[0]
    pl = math.sqrt(px * px + py * py + pz * pz)
    if pl < 1e-9:
        return starts, ends, colors
    px, py, pz = px / pl, py / pl, pz / pl

    head_len  = length * head_fraction
    side_len  = head_len * math.tan(head_spread)
    back_frac = 1.0 - head_fraction

    bx = ox + vx * back_frac
    by = oy + vy * back_frac
    bz = oz + vz * back_frac

    for sign in (+1.0, -1.0):
        hx = bx + sign * px * side_len
        hy = by + sign * py * side_len
        hz = bz + sign * pz * side_len
        starts.append((hx, hy, hz))
        ends.append(tip)
        colors.append(color)

    return starts, ends, colors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ForceGizmoDrawer:
    """Draws per-fin aero forces, thrust, and optional body torques as debug arrows.

    Parameters
    ----------
    force_scale:
        Visual scale factor: metres of arrow per Newton of force.
        Default 0.25 gives ~1.0 m arrow at ~4 N (≈ hover thrust on each fin).
    torque_scale:
        Visual scale factor: metres per N·m.
    min_mag:
        Minimum force/torque magnitude (SI units) below which the arrow is skipped.
    line_width:
        Line width in pixels.
    max_envs:
        Only draw for the first N environments (default 1) to avoid visual clutter.
    auto_clear:
        If True, clear all previous debug lines at the start of every ``draw()`` call.
    """

    def __init__(
        self,
        *,
        force_scale: float = 0.25,
        torque_scale: float = 0.25,
        min_mag: float = 0.01,
        line_width: float = 3.0,
        max_envs: int = 1,
        auto_clear: bool = True,
    ):
        self.force_scale  = force_scale
        self.torque_scale = torque_scale
        self.min_mag      = min_mag
        self.line_width   = line_width
        self.max_envs     = max_envs
        self.auto_clear   = auto_clear

    # ------------------------------------------------------------------
    def clear(self) -> None:
        draw = _get_draw()
        if draw is not None:
            draw.clear_lines()

    # ------------------------------------------------------------------
    def draw(
        self,
        *,
        fin_origins_w: torch.Tensor,                  # (N, 4, 3)
        fin_forces_w:  torch.Tensor,                  # (N, 4, 3)
        body_origin_w: torch.Tensor,                  # (N, 3)
        thrust_w:      torch.Tensor,                  # (N, 3)
        body_torque_w: Optional[torch.Tensor] = None, # (N, 3)
    ) -> None:
        """Issue debug draw calls for one physics step.

        All tensors must live on any device (CPU or CUDA); they are moved to
        CPU and converted to Python floats internally.
        """
        draw = _get_draw()
        if draw is None:
            return

        if self.auto_clear:
            draw.clear_lines()

        n = min(self.max_envs, fin_origins_w.shape[0])
        if n == 0:
            return

        # Detach and move to CPU once.
        fin_o = fin_origins_w[:n].detach().cpu()   # (n, 4, 3)
        fin_f = fin_forces_w[:n].detach().cpu()    # (n, 4, 3)
        body_o = body_origin_w[:n].detach().cpu()  # (n, 3)
        thr_f  = thrust_w[:n].detach().cpu()       # (n, 3)
        if body_torque_w is not None:
            torq_f = body_torque_w[:n].detach().cpu()  # (n, 3)
        else:
            torq_f = None

        scale_f = self.force_scale
        scale_t = self.torque_scale
        min_m   = self.min_mag

        all_starts: list = []
        all_ends:   list = []
        all_colors: list = []

        for env_i in range(n):
            # --- Per-fin aero forces ---
            for fin_j in range(4):
                ox, oy, oz = fin_o[env_i, fin_j].tolist()
                fx, fy, fz = fin_f[env_i, fin_j].tolist()
                mag = math.sqrt(fx * fx + fy * fy + fz * fz)
                if mag < min_m:
                    continue
                vx, vy, vz = fx * scale_f, fy * scale_f, fz * scale_f
                s, e, c = _arrow_segments(ox, oy, oz, vx, vy, vz, FIN_COLORS[fin_j % 4])
                all_starts.extend(s); all_ends.extend(e); all_colors.extend(c)

            # --- EDF thrust ---
            ox, oy, oz = body_o[env_i].tolist()
            fx, fy, fz = thr_f[env_i].tolist()
            mag = math.sqrt(fx * fx + fy * fy + fz * fz)
            if mag >= min_m:
                vx, vy, vz = fx * scale_f, fy * scale_f, fz * scale_f
                s, e, c = _arrow_segments(ox, oy, oz, vx, vy, vz, THRUST_COLOR)
                all_starts.extend(s); all_ends.extend(e); all_colors.extend(c)

            # --- Body torque (optional) ---
            if torq_f is not None:
                tx, ty, tz = torq_f[env_i].tolist()
                mag = math.sqrt(tx * tx + ty * ty + tz * tz)
                if mag >= min_m:
                    vx, vy, vz = tx * scale_t, ty * scale_t, tz * scale_t
                    # Offset slightly so torque arrow doesn't overlap thrust.
                    s, e, c = _arrow_segments(ox + 0.06, oy, oz, vx, vy, vz, TORQUE_COLOR)
                    all_starts.extend(s); all_ends.extend(e); all_colors.extend(c)

        if not all_starts:
            return

        widths = [self.line_width] * len(all_starts)
        draw.draw_lines(all_starts, all_ends, all_colors, widths)
