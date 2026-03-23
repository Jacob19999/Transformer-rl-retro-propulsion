"""
Diagnostic plot generation for TVC telemetry data.

Generates plots from CSV telemetry files produced by logger.py:
  - Fin force curves vs deflection angle
  - Thrust response over time
  - Torque contribution comparison
  - 3D trajectory
  - State history (position, tilt, angular rate, reward)

Requires matplotlib. Skips gracefully if not installed.
"""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Any


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot generation. "
            "Install with: pip install matplotlib"
        )


def load_telemetry_csv(csv_path: str | Path) -> dict[str, list[float]]:
    """Load a telemetry CSV file into a column-keyed dict of lists.

    Args:
        csv_path: Path to the CSV file produced by TelemetryLogger.

    Returns:
        Dict mapping column name → list of values.
    """
    data: dict[str, list[Any]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                try:
                    data.setdefault(key, []).append(float(val))
                except (ValueError, TypeError):
                    data.setdefault(key, []).append(val)
    return data


def plot_state_history(
    csv_path: str | Path,
    output_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot position error, tilt, angular rate, and reward over time.

    Args:
        csv_path:    Path to telemetry CSV.
        output_path: Where to save the figure (PNG). None = don't save.
        show:        Whether to display interactively.
    """
    plt = _require_matplotlib()
    import math

    data = load_telemetry_csv(csv_path)
    steps = data.get("step", [])
    t = [s / 30.0 for s in steps]  # Convert steps to seconds at 30 Hz

    pos_err = [
        math.sqrt(x**2 + y**2 + z**2)
        for x, y, z in zip(
            data.get("pos_err_x", [0]*len(t)),
            data.get("pos_err_y", [0]*len(t)),
            data.get("pos_err_z", [0]*len(t)),
        )
    ]
    w_vals = data.get("quat_w", [1.0]*len(t))
    tilt_deg = [math.degrees(2.0 * math.acos(min(abs(w), 1.0))) for w in w_vals]

    ang_rate = [
        math.sqrt(x**2 + y**2 + z**2)
        for x, y, z in zip(
            data.get("ang_vel_x", [0]*len(t)),
            data.get("ang_vel_y", [0]*len(t)),
            data.get("ang_vel_z", [0]*len(t)),
        )
    ]
    reward = data.get("reward", [0.0]*len(t))

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(t, pos_err, "b-", linewidth=0.8)
    axes[0].axhline(0.5, color="r", linestyle="--", linewidth=0.6, label="0.5m threshold")
    axes[0].set_ylabel("Pos error (m)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, tilt_deg, "g-", linewidth=0.8)
    axes[1].axhline(15.0, color="r", linestyle="--", linewidth=0.6, label="15° threshold")
    axes[1].set_ylabel("Tilt (deg)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, ang_rate, "m-", linewidth=0.8)
    axes[2].axhline(1.0, color="r", linestyle="--", linewidth=0.6, label="1.0 rad/s threshold")
    axes[2].set_ylabel("Ang rate (rad/s)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, reward, "k-", linewidth=0.8)
    axes[3].set_ylabel("Reward")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(f"State History — {Path(csv_path).stem}", fontsize=12)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_fin_force_curves(
    deflection_angles: list[float],
    normal_forces: list[float],
    drag_forces: list[float],
    output_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot fin normal and drag force vs deflection angle.

    Args:
        deflection_angles: List of fin deflection angles (rad).
        normal_forces:     Corresponding normal forces (N).
        drag_forces:       Corresponding drag forces (N).
        output_path:       Where to save the figure.
        show:              Whether to display interactively.
    """
    plt = _require_matplotlib()
    import math

    angles_deg = [math.degrees(a) for a in deflection_angles]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(angles_deg, normal_forces, "b-o", markersize=3, label="Normal force (N)")
    ax.plot(angles_deg, drag_forces, "r-s", markersize=3, label="Drag force (N)")
    ax.set_xlabel("Fin deflection (deg)")
    ax.set_ylabel("Force (N)")
    ax.set_title("Fin Force vs Deflection Angle")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_torque_contributions(
    csv_path: str | Path,
    output_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot torque contribution magnitudes over time.

    Args:
        csv_path:    Path to telemetry CSV with torque columns.
        output_path: Where to save the figure.
        show:        Whether to display interactively.
    """
    plt = _require_matplotlib()
    import math

    data = load_telemetry_csv(csv_path)
    steps = data.get("step", [])
    t = [s / 30.0 for s in steps]

    def mag(prefix):
        x = data.get(f"{prefix}_x", [0]*len(t))
        y = data.get(f"{prefix}_y", [0]*len(t))
        z = data.get(f"{prefix}_z", [0]*len(t))
        return [math.sqrt(xi**2 + yi**2 + zi**2) for xi, yi, zi in zip(x, y, z)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, mag("torque_fin"),    label="Fin aero",  linewidth=0.8)
    ax.plot(t, mag("torque_static"), label="Static Q",  linewidth=0.8)
    ax.plot(t, mag("torque_gyro"),   label="Gyro",      linewidth=0.8)
    ax.plot(t, mag("torque_spool"),  label="Spool",     linewidth=0.8)
    ax.plot(t, mag("force_wind"),    label="Wind drag", linewidth=0.8, linestyle="--")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Magnitude (N·m or N)")
    ax.set_title("Force/Torque Contributions Over Time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
