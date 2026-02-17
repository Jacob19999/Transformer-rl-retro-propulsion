"""
Visualize PID controller performance: trajectory, body rates, and fin angles.

Creates matplotlib plots showing:
- 3D trajectory and 2D projections
- Body angular rates over time
- Fin deflection angles over time
- Altitude, velocity, and attitude over time
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from simulation.config_loader import load_config
from simulation.training.controllers.pid_controller import PIDController
from simulation.training.edf_landing_env import EDFLandingEnv


def _make_deterministic_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Make environment deterministic for visualization."""
    if cfg is None:
        configs = Path("simulation/configs")
        v = load_config(configs / "default_vehicle.yaml")
        e = load_config(configs / "default_environment.yaml")
        r = load_config(configs / "reward.yaml")
        cfg = {
            "vehicle": v.get("vehicle", v),
            "environment": e.get("environment", e),
            "reward": r.get("reward", r),
        }

    env_cfg = dict(cfg.get("environment", {}))
    atm = dict(env_cfg.get("atmosphere", {}))
    atm["randomize_T"] = 0.0
    atm["randomize_P"] = 0.0
    env_cfg["atmosphere"] = atm

    wind = dict(env_cfg.get("wind", {}))
    wind["mean_vector_range_lo"] = [0.0, 0.0, 0.0]
    wind["mean_vector_range_hi"] = [0.0, 0.0, 0.0]
    wind["turbulence_intensity"] = 0.0
    wind["gust_prob"] = 0.0
    wind["gust_magnitude_range"] = [0.0, 0.0]
    env_cfg["wind"] = wind
    cfg = dict(cfg)
    cfg["environment"] = env_cfg

    cfg["actuator_delay"] = dict(cfg.get("actuator_delay", {}))
    cfg["actuator_delay"]["enabled"] = False
    cfg["obs_latency"] = dict(cfg.get("obs_latency", {}))
    cfg["obs_latency"]["enabled"] = False
    cfg["observation"] = dict(cfg.get("observation", {}))
    cfg["observation"]["noise_std"] = 0.0

    return cfg


def run_episode(env: EDFLandingEnv, ctrl: PIDController, seed: int) -> dict[str, Any]:
    """Run one episode and collect trajectory data."""
    obs, info = env.reset(seed=seed)
    ctrl.reset()

    # Storage for trajectory data
    times = []
    positions = []  # NED frame
    velocities = []  # body frame
    angular_rates = []  # body frame
    quaternions = []
    actions = []  # [thrust, fin1, fin2, fin3, fin4]
    altitudes = []
    velocities_mag = []
    angular_rates_mag = []
    roll_angles = []
    pitch_angles = []
    yaw_angles = []
    # Environment samples (wind, temperature, pressure, density)
    env_wind = []      # NED wind vector [m/s]
    env_T = []         # temperature [K]
    env_P = []         # pressure [Pa]
    env_rho = []       # density [kg/m^3]

    terminated = truncated = False
    ep_info: dict = {}

    while not (terminated or truncated):
        # Get current state
        p, v_b, R, omega, T = env._state_terms()
        
        # Extract quaternion from vehicle state
        state = env.vehicle.state
        q = state[6:10]  # Quaternion from state vector

        # Compute Euler angles (for visualization)
        # Roll: rotation about x-axis
        roll = np.arctan2(R[2, 1], R[2, 2])
        # Pitch: rotation about y-axis
        pitch = np.arcsin(-R[2, 0])
        # Yaw: rotation about z-axis
        yaw = np.arctan2(R[1, 0], R[0, 0])

        # Sample environment at current state (wind, atmosphere)
        env_sample = env.env_model.sample_at_state(float(env.vehicle.time), p)

        # Store data
        times.append(float(env.vehicle.time))
        positions.append(p.copy())
        velocities.append(v_b.copy())
        angular_rates.append(omega.copy())
        quaternions.append(q.copy())
        altitudes.append(float(-p[2]))  # AGL = -z in NED
        velocities_mag.append(float(np.linalg.norm(v_b)))
        angular_rates_mag.append(float(np.linalg.norm(omega)))
        roll_angles.append(float(np.rad2deg(roll)))
        pitch_angles.append(float(np.rad2deg(pitch)))
        yaw_angles.append(float(np.rad2deg(yaw)))
        env_wind.append(env_sample["wind"].copy())
        env_T.append(float(env_sample["T"]))
        env_P.append(float(env_sample["P"]))
        env_rho.append(float(env_sample["rho"]))

        # Get action
        action = ctrl.get_action(obs)
        actions.append(action.copy())

        # Step environment
        obs, _reward, terminated, truncated, ep_info = env.step(action)

    return {
        "times": np.array(times),
        "positions": np.array(positions),
        "velocities": np.array(velocities),
        "angular_rates": np.array(angular_rates),
        "quaternions": np.array(quaternions),
        "actions": np.array(actions),
        "altitudes": np.array(altitudes),
        "velocities_mag": np.array(velocities_mag),
        "angular_rates_mag": np.array(angular_rates_mag),
        "roll_angles": np.array(roll_angles),
        "pitch_angles": np.array(pitch_angles),
        "yaw_angles": np.array(yaw_angles),
        "wind": np.array(env_wind),
        "temperature": np.array(env_T),
        "pressure": np.array(env_P),
        "density": np.array(env_rho),
        "ep_info": ep_info,
        "dt": float(env.dt_policy),
    }


def plot_trajectory(data: dict[str, Any], ax_3d: Any, ax_xy: Any, ax_xz: Any) -> None:
    """Plot 3D trajectory and 2D projections."""
    pos = data["positions"]
    times = data["times"]
    altitudes = data["altitudes"]

    # 3D trajectory
    # Use altitude AGL (positive up) for z-axis so start is above target
    ax_3d.plot(pos[:, 0], pos[:, 1], altitudes, "b-", alpha=0.6, linewidth=1.5)
    ax_3d.scatter(pos[0, 0], pos[0, 1], altitudes[0], c="green", s=100, marker="o", label="Start", zorder=5)
    ax_3d.scatter(
        pos[-1, 0],
        pos[-1, 1],
        altitudes[-1],
        c="red",
        s=100,
        marker="x",
        label="End",
        zorder=5,
    )
    ax_3d.scatter(0, 0, 0.0, c="black", s=50, marker="s", label="Target", zorder=5)

    # Landing pad circle (at z=0)
    theta = np.linspace(0, 2 * np.pi, 100)
    pad_radius = 2.0  # meters
    pad_x = pad_radius * np.cos(theta)
    pad_y = pad_radius * np.sin(theta)
    ax_3d.plot(pad_x, pad_y, np.zeros_like(theta), "k--", alpha=0.3, linewidth=1)

    ax_3d.set_xlabel("North (m)")
    ax_3d.set_ylabel("East (m)")
    ax_3d.set_zlabel("Altitude AGL (m)")
    ax_3d.set_title("3D Trajectory")
    ax_3d.legend()
    ax_3d.grid(True, alpha=0.3)
    ax_3d.set_box_aspect([1, 1, 1])

    # XY projection (top-down)
    ax_xy.plot(pos[:, 0], pos[:, 1], "b-", alpha=0.6, linewidth=1.5)
    ax_xy.scatter(pos[0, 0], pos[0, 1], c="green", s=100, marker="o", label="Start", zorder=5)
    ax_xy.scatter(pos[-1, 0], pos[-1, 1], c="red", s=100, marker="x", label="End", zorder=5)
    circle = Circle((0, 0), pad_radius, fill=False, linestyle="--", color="black", alpha=0.3)
    ax_xy.add_patch(circle)
    ax_xy.scatter(0, 0, c="black", s=50, marker="s", label="Target", zorder=5)
    ax_xy.set_xlabel("North (m)")
    ax_xy.set_ylabel("East (m)")
    ax_xy.set_title("Top-Down View (XY)")
    ax_xy.legend()
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect("equal")

    # XZ projection (side view, North-Down)
    ax_xz.plot(pos[:, 0], altitudes, "b-", alpha=0.6, linewidth=1.5)
    ax_xz.scatter(pos[0, 0], altitudes[0], c="green", s=100, marker="o", label="Start", zorder=5)
    ax_xz.scatter(pos[-1, 0], altitudes[-1], c="red", s=100, marker="x", label="End", zorder=5)
    ax_xz.axhline(y=0.05, color="r", linestyle="--", alpha=0.5, label="Landing threshold")
    ax_xz.set_xlabel("North (m)")
    ax_xz.set_ylabel("Altitude AGL (m)")
    ax_xz.set_title("Side View - North (XZ)")
    ax_xz.legend()
    ax_xz.grid(True, alpha=0.3)


def plot_body_rates(data: dict[str, Any], ax: Any) -> None:
    """Plot body angular rates over time."""
    times = data["times"]
    omega = data["angular_rates"]

    ax.plot(times, omega[:, 0], "r-", label=r"$\omega_x$ (roll rate)", linewidth=1.5)
    ax.plot(times, omega[:, 1], "g-", label=r"$\omega_y$ (pitch rate)", linewidth=1.5)
    ax.plot(times, omega[:, 2], "b-", label=r"$\omega_z$ (yaw rate)", linewidth=1.5)
    ax.plot(times, data["angular_rates_mag"], "k--", label=r"$|\omega|$", linewidth=1, alpha=0.5)

    # Landing thresholds
    ax.axhline(y=0.5, color="r", linestyle=":", alpha=0.3, label="Landing threshold (0.5 rad/s)")
    ax.axhline(y=-0.5, color="r", linestyle=":", alpha=0.3)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular Rate (rad/s)")
    ax.set_title("Body Angular Rates")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_fin_angles(data: dict[str, Any], delta_max: float, ax: Any) -> None:
    """Plot fin deflection angles over time."""
    times = data["times"]
    actions = data["actions"]

    # Convert normalized actions [-1, 1] to actual angles [rad]
    fin1 = actions[:, 1] * delta_max
    fin2 = actions[:, 2] * delta_max
    fin3 = actions[:, 3] * delta_max
    fin4 = actions[:, 4] * delta_max

    ax.plot(times, np.rad2deg(fin1), "r-", label="Fin 1 (right)", linewidth=1.5, alpha=0.7)
    ax.plot(times, np.rad2deg(fin2), "g-", label="Fin 2 (left)", linewidth=1.5, alpha=0.7)
    ax.plot(times, np.rad2deg(fin3), "b-", label="Fin 3 (forward)", linewidth=1.5, alpha=0.7)
    ax.plot(times, np.rad2deg(fin4), "m-", label="Fin 4 (aft)", linewidth=1.5, alpha=0.7)

    # Saturation limits
    max_deg = np.rad2deg(delta_max)
    ax.axhline(y=max_deg, color="k", linestyle="--", alpha=0.3, label=f"Max ({max_deg:.1f}°)")
    ax.axhline(y=-max_deg, color="k", linestyle="--", alpha=0.3)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fin Deflection (deg)")
    ax.set_title("Fin Control Angles")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_altitude_velocity(data: dict[str, Any], ax_alt: Any, ax_vel: Any) -> None:
    """Plot altitude and velocity over time."""
    times = data["times"]
    altitudes = data["altitudes"]
    velocities_mag = data["velocities_mag"]
    v_b = data["velocities"]

    # Altitude plot
    ax_alt.plot(times, altitudes, "b-", linewidth=1.5, label="Altitude AGL")
    ax_alt.axhline(y=0.05, color="r", linestyle="--", alpha=0.5, label="Landing threshold (0.05 m)")
    ax_alt.set_xlabel("Time (s)")
    ax_alt.set_ylabel("Altitude AGL (m)")
    ax_alt.set_title("Altitude Over Time")
    ax_alt.legend()
    ax_alt.grid(True, alpha=0.3)

    # Velocity plot
    ax_vel.plot(times, velocities_mag, "b-", linewidth=1.5, label="Speed |v|")
    ax_vel.plot(times, v_b[:, 2], "g-", linewidth=1.5, label="Vertical (w)", alpha=0.7)
    ax_vel.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="Landing threshold (0.5 m/s)")
    ax_vel.axhline(y=-0.5, color="r", linestyle="--", alpha=0.5)
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_ylabel("Velocity (m/s)")
    ax_vel.set_title("Velocity Over Time")
    ax_vel.legend()
    ax_vel.grid(True, alpha=0.3)


def plot_attitude(data: dict[str, Any], ax: Any) -> None:
    """Plot attitude angles (roll, pitch, yaw) over time."""
    times = data["times"]

    ax.plot(times, data["roll_angles"], "r-", label="Roll", linewidth=1.5)
    ax.plot(times, data["pitch_angles"], "g-", label="Pitch", linewidth=1.5)
    ax.plot(times, data["yaw_angles"], "b-", label="Yaw", linewidth=1.5)

    # Landing threshold (tilt < 15 deg)
    ax.axhline(y=15, color="r", linestyle="--", alpha=0.3, label="Landing threshold (±15°)")
    ax.axhline(y=-15, color="r", linestyle="--", alpha=0.3)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Attitude Angles (Euler)")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_thrust(data: dict[str, Any], ax: Any) -> None:
    """Plot thrust command over time."""
    times = data["times"]
    actions = data["actions"]
    thrust_action = actions[:, 0]  # Normalized [-1, 1]

    ax.plot(times, thrust_action, "b-", linewidth=1.5, label="Thrust command (normalized)")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3, label="Hover (0)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Thrust Command (normalized)")
    ax.set_title("Thrust Command Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)


def create_animated_plots(data: dict[str, Any], delta_max: float) -> tuple[Any, dict[str, Any]]:
    """Create figure and axes for animation, return figure and plot objects dict."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # Row 1: Trajectory plots
    ax_3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xz = fig.add_subplot(gs[0, 2])

    # Row 2: Body rates and fin angles
    ax_omega = fig.add_subplot(gs[1, :2])
    ax_fins = fig.add_subplot(gs[1, 2])

    # Row 3: Altitude and velocity
    ax_alt = fig.add_subplot(gs[2, 0])
    ax_vel = fig.add_subplot(gs[2, 1])
    ax_att = fig.add_subplot(gs[2, 2])

    # Row 4: Thrust
    ax_thrust = fig.add_subplot(gs[3, :])

    # Initialize plots with empty data
    pos = data["positions"]
    times = data["times"]
    altitudes = data["altitudes"]
    omega = data["angular_rates"]
    actions = data["actions"]
    velocities_mag = data["velocities_mag"]
    v_b = data["velocities"]
    roll_angles = data["roll_angles"]
    pitch_angles = data["pitch_angles"]
    yaw_angles = data["yaw_angles"]

    # Set up axes limits
    pad_radius = 2.0
    x_range = [min(pos[:, 0].min(), -pad_radius), max(pos[:, 0].max(), pad_radius)]
    y_range = [min(pos[:, 1].min(), -pad_radius), max(pos[:, 1].max(), pad_radius)]
    z_range = [0, max(altitudes.max(), 5.0)]
    t_range = [0, max(times.max(), 1.0)]

    # 3D trajectory (use altitude AGL on z-axis so start is above target)
    ax_3d.set_xlim(x_range)
    ax_3d.set_ylim(y_range)
    ax_3d.set_zlim(z_range)
    ax_3d.set_xlabel("North (m)")
    ax_3d.set_ylabel("East (m)")
    ax_3d.set_zlabel("Altitude AGL (m)")
    ax_3d.set_title("3D Trajectory")
    ax_3d.scatter(0, 0, 0.0, c="black", s=50, marker="s", label="Target", zorder=5)
    ax_3d.legend()
    ax_3d.grid(True, alpha=0.3)
    ax_3d.set_box_aspect([1, 1, 1])
    traj_3d, = ax_3d.plot([], [], [], "b-", alpha=0.6, linewidth=1.5)
    # Use plot for current position marker (more reliable than scatter in 3D)
    current_3d, = ax_3d.plot([], [], [], "ro", markersize=10, zorder=10)

    # XY projection
    ax_xy.set_xlim(x_range)
    ax_xy.set_ylim(y_range)
    ax_xy.set_xlabel("North (m)")
    ax_xy.set_ylabel("East (m)")
    ax_xy.set_title("Top-Down View (XY)")
    circle = Circle((0, 0), pad_radius, fill=False, linestyle="--", color="black", alpha=0.3)
    ax_xy.add_patch(circle)
    ax_xy.scatter(0, 0, c="black", s=50, marker="s", label="Target", zorder=5)
    ax_xy.legend()
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect("equal")
    traj_xy, = ax_xy.plot([], [], "b-", alpha=0.6, linewidth=1.5)
    current_xy = ax_xy.scatter([], [], c="red", s=150, marker="o", zorder=10)

    # XZ projection
    ax_xz.set_xlim(x_range)
    ax_xz.set_ylim([0, z_range[1]])
    ax_xz.set_xlabel("North (m)")
    ax_xz.set_ylabel("Altitude AGL (m)")
    ax_xz.set_title("Side View - North (XZ)")
    ax_xz.axhline(y=0.05, color="r", linestyle="--", alpha=0.5, label="Landing threshold")
    ax_xz.legend()
    ax_xz.grid(True, alpha=0.3)
    traj_xz, = ax_xz.plot([], [], "b-", alpha=0.6, linewidth=1.5)
    current_xz = ax_xz.scatter([], [], c="red", s=150, marker="o", zorder=10)

    # Body rates
    omega_max = max(np.abs(omega).max(), 1.0)
    ax_omega.set_xlim(t_range)
    ax_omega.set_ylim([-omega_max * 1.1, omega_max * 1.1])
    ax_omega.set_xlabel("Time (s)")
    ax_omega.set_ylabel("Angular Rate (rad/s)")
    ax_omega.set_title("Body Angular Rates")
    ax_omega.axhline(y=0.5, color="r", linestyle=":", alpha=0.3)
    ax_omega.axhline(y=-0.5, color="r", linestyle=":", alpha=0.3)
    ax_omega.grid(True, alpha=0.3)
    omega_x_line, = ax_omega.plot([], [], "r-", label=r"$\omega_x$", linewidth=1.5)
    omega_y_line, = ax_omega.plot([], [], "g-", label=r"$\omega_y$", linewidth=1.5)
    omega_z_line, = ax_omega.plot([], [], "b-", label=r"$\omega_z$", linewidth=1.5)
    omega_mag_line, = ax_omega.plot([], [], "k--", label=r"$|\omega|$", linewidth=1, alpha=0.5)
    ax_omega.legend()

    # Fin angles
    fin_max_deg = np.rad2deg(delta_max) * 1.1
    ax_fins.set_xlim(t_range)
    ax_fins.set_ylim([-fin_max_deg, fin_max_deg])
    ax_fins.set_xlabel("Time (s)")
    ax_fins.set_ylabel("Fin Deflection (deg)")
    ax_fins.set_title("Fin Control Angles")
    ax_fins.axhline(y=np.rad2deg(delta_max), color="k", linestyle="--", alpha=0.3)
    ax_fins.axhline(y=-np.rad2deg(delta_max), color="k", linestyle="--", alpha=0.3)
    ax_fins.grid(True, alpha=0.3)
    fin1_line, = ax_fins.plot([], [], "r-", label="Fin 1", linewidth=1.5, alpha=0.7)
    fin2_line, = ax_fins.plot([], [], "g-", label="Fin 2", linewidth=1.5, alpha=0.7)
    fin3_line, = ax_fins.plot([], [], "b-", label="Fin 3", linewidth=1.5, alpha=0.7)
    fin4_line, = ax_fins.plot([], [], "m-", label="Fin 4", linewidth=1.5, alpha=0.7)
    ax_fins.legend()

    # Altitude
    ax_alt.set_xlim(t_range)
    ax_alt.set_ylim([0, z_range[1]])
    ax_alt.set_xlabel("Time (s)")
    ax_alt.set_ylabel("Altitude AGL (m)")
    ax_alt.set_title("Altitude Over Time")
    ax_alt.axhline(y=0.05, color="r", linestyle="--", alpha=0.5)
    ax_alt.grid(True, alpha=0.3)
    alt_line, = ax_alt.plot([], [], "b-", linewidth=1.5)

    # Velocity
    v_max = max(velocities_mag.max(), 1.0)
    ax_vel.set_xlim(t_range)
    ax_vel.set_ylim([-v_max * 1.1, v_max * 1.1])
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_ylabel("Velocity (m/s)")
    ax_vel.set_title("Velocity Over Time")
    ax_vel.axhline(y=0.5, color="r", linestyle="--", alpha=0.5)
    ax_vel.axhline(y=-0.5, color="r", linestyle="--", alpha=0.5)
    ax_vel.grid(True, alpha=0.3)
    vel_mag_line, = ax_vel.plot([], [], "b-", label="Speed |v|", linewidth=1.5)
    vel_z_line, = ax_vel.plot([], [], "g-", label="Vertical (w)", linewidth=1.5, alpha=0.7)
    ax_vel.legend()

    # Attitude
    att_max = max(np.abs(roll_angles).max(), np.abs(pitch_angles).max(), 20.0)
    ax_att.set_xlim(t_range)
    ax_att.set_ylim([-att_max * 1.1, att_max * 1.1])
    ax_att.set_xlabel("Time (s)")
    ax_att.set_ylabel("Angle (deg)")
    ax_att.set_title("Attitude Angles (Euler)")
    ax_att.axhline(y=15, color="r", linestyle="--", alpha=0.3)
    ax_att.axhline(y=-15, color="r", linestyle="--", alpha=0.3)
    ax_att.grid(True, alpha=0.3)
    roll_line, = ax_att.plot([], [], "r-", label="Roll", linewidth=1.5)
    pitch_line, = ax_att.plot([], [], "g-", label="Pitch", linewidth=1.5)
    yaw_line, = ax_att.plot([], [], "b-", label="Yaw", linewidth=1.5)
    ax_att.legend()

    # Thrust
    thrust_max = max(np.abs(actions[:, 0]).max(), 0.5)
    ax_thrust.set_xlim(t_range)
    ax_thrust.set_ylim([-thrust_max * 1.1, thrust_max * 1.1])
    ax_thrust.set_xlabel("Time (s)")
    ax_thrust.set_ylabel("Thrust Command (normalized)")
    ax_thrust.set_title("Thrust Command Over Time")
    ax_thrust.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax_thrust.grid(True, alpha=0.3)
    thrust_line, = ax_thrust.plot([], [], "b-", linewidth=1.5)

    plot_objects = {
        "traj_3d": traj_3d,
        "current_3d": current_3d,
        "traj_xy": traj_xy,
        "current_xy": current_xy,
        "traj_xz": traj_xz,
        "current_xz": current_xz,
        "omega_x": omega_x_line,
        "omega_y": omega_y_line,
        "omega_z": omega_z_line,
        "omega_mag": omega_mag_line,
        "fin1": fin1_line,
        "fin2": fin2_line,
        "fin3": fin3_line,
        "fin4": fin4_line,
        "alt": alt_line,
        "vel_mag": vel_mag_line,
        "vel_z": vel_z_line,
        "roll": roll_line,
        "pitch": pitch_line,
        "yaw": yaw_line,
        "thrust": thrust_line,
    }

    return fig, plot_objects


def animate_frame(frame: int, data: dict[str, Any], plot_objects: dict[str, Any], delta_max: float) -> list:
    """Update animation for a single frame."""
    times = data["times"]
    pos = data["positions"]
    altitudes = data["altitudes"]
    omega = data["angular_rates"]
    actions = data["actions"]
    velocities_mag = data["velocities_mag"]
    v_b = data["velocities"]
    roll_angles = data["roll_angles"]
    pitch_angles = data["pitch_angles"]
    yaw_angles = data["yaw_angles"]
    angular_rates_mag = data["angular_rates_mag"]

    # Current index (up to frame)
    idx = min(frame, len(times) - 1)
    t_current = times[: idx + 1]
    idx_current = idx

    # Update trajectory plots (3D uses altitude AGL on z-axis)
    plot_objects["traj_3d"].set_data_3d(pos[: idx + 1, 0], pos[: idx + 1, 1], altitudes[: idx + 1])
    plot_objects["current_3d"].set_data_3d(
        [pos[idx_current, 0]], [pos[idx_current, 1]], [altitudes[idx_current]]
    )

    plot_objects["traj_xy"].set_data(pos[: idx + 1, 0], pos[: idx + 1, 1])
    plot_objects["current_xy"].set_offsets([[pos[idx_current, 0], pos[idx_current, 1]]])

    plot_objects["traj_xz"].set_data(pos[: idx + 1, 0], altitudes[: idx + 1])
    plot_objects["current_xz"].set_offsets([[pos[idx_current, 0], altitudes[idx_current]]])

    # Update body rates
    plot_objects["omega_x"].set_data(t_current, omega[: idx + 1, 0])
    plot_objects["omega_y"].set_data(t_current, omega[: idx + 1, 1])
    plot_objects["omega_z"].set_data(t_current, omega[: idx + 1, 2])
    plot_objects["omega_mag"].set_data(t_current, angular_rates_mag[: idx + 1])

    # Update fin angles
    fin1 = np.rad2deg(actions[: idx + 1, 1] * delta_max)
    fin2 = np.rad2deg(actions[: idx + 1, 2] * delta_max)
    fin3 = np.rad2deg(actions[: idx + 1, 3] * delta_max)
    fin4 = np.rad2deg(actions[: idx + 1, 4] * delta_max)
    plot_objects["fin1"].set_data(t_current, fin1)
    plot_objects["fin2"].set_data(t_current, fin2)
    plot_objects["fin3"].set_data(t_current, fin3)
    plot_objects["fin4"].set_data(t_current, fin4)

    # Update altitude
    plot_objects["alt"].set_data(t_current, altitudes[: idx + 1])

    # Update velocity
    plot_objects["vel_mag"].set_data(t_current, velocities_mag[: idx + 1])
    plot_objects["vel_z"].set_data(t_current, v_b[: idx + 1, 2])

    # Update attitude
    plot_objects["roll"].set_data(t_current, roll_angles[: idx + 1])
    plot_objects["pitch"].set_data(t_current, pitch_angles[: idx + 1])
    plot_objects["yaw"].set_data(t_current, yaw_angles[: idx + 1])

    # Update thrust
    plot_objects["thrust"].set_data(t_current, actions[: idx + 1, 0])

    return list(plot_objects.values())


def create_interactive_dashboard(data: dict[str, Any], delta_max: float, seed: int, ep_info: dict[str, Any]) -> None:
    """Create an interactive Plotly dashboard and save as HTML."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for interactive visualization. Install with: pip install plotly")

    times = data["times"]
    pos = data["positions"]
    altitudes = data["altitudes"]
    omega = data["angular_rates"]
    actions = data["actions"]
    velocities_mag = data["velocities_mag"]
    v_b = data["velocities"]
    roll_angles = data["roll_angles"]
    pitch_angles = data["pitch_angles"]
    yaw_angles = data["yaw_angles"]
    angular_rates_mag = data["angular_rates_mag"]
    # Environment history (present for new runs; fall back gracefully if missing)
    wind = data.get("wind")
    temperature = data.get("temperature")
    pressure = data.get("pressure")
    density = data.get("density")

    landed = bool(ep_info.get("landed", False))
    crashed = bool(ep_info.get("crashed", False))
    cep = float(ep_info.get("cep", float("inf")))

    # Create subplots (3D plots need special handling in Plotly)
    # We'll create a 2D grid and add 3D plot separately
    fig = make_subplots(
        rows=4,
        cols=3,
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter", "colspan": 2}, {"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
        ],
        subplot_titles=(
            "Top-Down View (XY)",
            "Side View - North (XZ)",
            "Side View - East (YZ)",
            "Body Angular Rates",
            "Fin Control Angles",
            "Altitude Over Time",
            "Velocity Over Time",
            "Attitude Angles",
            "Thrust Command",
            "Environment (Wind)",
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    # Add 3D trajectory as a separate figure element (will be positioned manually)
    # For now, we'll use the 2D projections and add 3D as an overlay

    # XY projection (top-down) - row 1, col 1
    xy_traj_index = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[pos[0, 0]],
            y=[pos[0, 1]],
            mode="lines",
            name="Trajectory",
            line=dict(color="blue", width=2),
            showlegend=False,
            hovertemplate="North: %{x:.2f}m<br>East: %{y:.2f}m<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[pos[0, 0]],
            y=[pos[0, 1]],
            mode="markers",
            name="Start",
            marker=dict(size=10, color="green"),
            showlegend=False,
            hovertemplate="Start<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[pos[-1, 0]],
            y=[pos[-1, 1]],
            mode="markers",
            name="End",
            marker=dict(size=10, color="red", symbol="x"),
            showlegend=False,
            hovertemplate="End<extra></extra>",
        ),
        row=1,
        col=1,
    )
    # Landing pad circle
    theta = np.linspace(0, 2 * np.pi, 100)
    pad_radius = 2.0
    pad_x = pad_radius * np.cos(theta)
    pad_y = pad_radius * np.sin(theta)
    fig.add_trace(
        go.Scatter(
            x=pad_x,
            y=pad_y,
            mode="lines",
            name="Landing Pad",
            line=dict(color="black", dash="dash", width=1),
            showlegend=False,
            hovertemplate="Landing Pad<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # XZ projection (side view) - row 1, col 2
    xz_traj_index = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[pos[0, 0]],
            y=[altitudes[0]],
            mode="lines",
            name="Trajectory",
            line=dict(color="blue", width=2),
            showlegend=False,
            hovertemplate="North: %{x:.2f}m<br>Altitude: %{y:.3f}m<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=0.05, line_dash="dash", line_color="red", opacity=0.5, row=1, col=2, annotation_text="Landing threshold")

    # YZ projection (side view) - row 1, col 3
    yz_traj_index = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[pos[0, 1]],
            y=[altitudes[0]],
            mode="lines",
            name="Trajectory",
            line=dict(color="blue", width=2),
            showlegend=False,
            hovertemplate="East: %{x:.2f}m<br>Altitude: %{y:.3f}m<extra></extra>",
        ),
        row=1,
        col=3,
    )
    fig.add_hline(y=0.05, line_dash="dash", line_color="red", opacity=0.5, row=1, col=3, annotation_text="Landing threshold")

    # Body angular rates
    fig.add_trace(
        go.Scatter(x=times, y=omega[:, 0], mode="lines", name=r"ωx (roll)", line=dict(color="red"), showlegend=True),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=times, y=omega[:, 1], mode="lines", name=r"ωy (pitch)", line=dict(color="green"), showlegend=True),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=times, y=omega[:, 2], mode="lines", name=r"ωz (yaw)", line=dict(color="blue"), showlegend=True),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=angular_rates_mag,
            mode="lines",
            name=r"|ω|",
            line=dict(color="black", dash="dash", width=1),
            showlegend=True,
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0.5, line_dash="dot", line_color="red", opacity=0.3, row=2, col=1)
    fig.add_hline(y=-0.5, line_dash="dot", line_color="red", opacity=0.3, row=2, col=1)

    # Fin angles - row 2, col 3
    fin1 = np.rad2deg(actions[:, 1] * delta_max)
    fin2 = np.rad2deg(actions[:, 2] * delta_max)
    fin3 = np.rad2deg(actions[:, 3] * delta_max)
    fin4 = np.rad2deg(actions[:, 4] * delta_max)
    fig.add_trace(
        go.Scatter(x=times, y=fin1, mode="lines", name="Fin 1", line=dict(color="red"), showlegend=True),
        row=2,
        col=3,
    )
    fig.add_trace(
        go.Scatter(x=times, y=fin2, mode="lines", name="Fin 2", line=dict(color="green"), showlegend=True),
        row=2,
        col=3,
    )
    fig.add_trace(
        go.Scatter(x=times, y=fin3, mode="lines", name="Fin 3", line=dict(color="blue"), showlegend=True),
        row=2,
        col=3,
    )
    fig.add_trace(
        go.Scatter(x=times, y=fin4, mode="lines", name="Fin 4", line=dict(color="magenta"), showlegend=True),
        row=2,
        col=3,
    )
    max_deg = np.rad2deg(delta_max)
    fig.add_hline(y=max_deg, line_dash="dash", line_color="black", opacity=0.3, row=2, col=3)
    fig.add_hline(y=-max_deg, line_dash="dash", line_color="black", opacity=0.3, row=2, col=3)

    # Altitude
    alt_traj_index = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[times[0]],
            y=[altitudes[0]],
            mode="lines",
            name="Altitude AGL",
            line=dict(color="blue", width=2),
            showlegend=False,
            hovertemplate="Time: %{x:.2f}s<br>Altitude: %{y:.3f}m<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=0.05, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1, annotation_text="Landing threshold")

    # Velocity
    vel_mag_index = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[times[0]],
            y=[velocities_mag[0]],
            mode="lines",
            name="Speed |v|",
            line=dict(color="blue", width=2),
            showlegend=True,
        ),
        row=3,
        col=2,
    )
    vel_z_index = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[times[0]],
            y=[v_b[0, 2]],
            mode="lines",
            name="Vertical (w)",
            line=dict(color="green", width=2),
            showlegend=True,
        ),
        row=3,
        col=2,
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", opacity=0.5, row=3, col=2)
    fig.add_hline(y=-0.5, line_dash="dash", line_color="red", opacity=0.5, row=3, col=2)

    # Attitude
    roll_index = len(fig.data)
    fig.add_trace(
        go.Scatter(x=[times[0]], y=[roll_angles[0]], mode="lines", name="Roll", line=dict(color="red"), showlegend=True),
        row=3,
        col=3,
    )
    pitch_index = len(fig.data)
    fig.add_trace(
        go.Scatter(x=[times[0]], y=[pitch_angles[0]], mode="lines", name="Pitch", line=dict(color="green"), showlegend=True),
        row=3,
        col=3,
    )
    yaw_index = len(fig.data)
    fig.add_trace(
        go.Scatter(x=[times[0]], y=[yaw_angles[0]], mode="lines", name="Yaw", line=dict(color="blue"), showlegend=True),
        row=3,
        col=3,
    )
    fig.add_hline(y=15, line_dash="dash", line_color="red", opacity=0.3, row=3, col=3)
    fig.add_hline(y=-15, line_dash="dash", line_color="red", opacity=0.3, row=3, col=3)

    # Thrust
    thrust_index = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[times[0]],
            y=[actions[0, 0]],
            mode="lines",
            name="Thrust Command",
            line=dict(color="blue", width=2),
            showlegend=False,
            hovertemplate="Time: %{x:.2f}s<br>Thrust: %{y:.3f}<extra></extra>",
        ),
        row=4,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3, row=4, col=1)

    # Create frames for trajectory history and time-series evolution
    frames = []
    for i, t in enumerate(times):
        frame_data = []
        frame_traces = []

        # 2D trajectory projections
        frame_data.append(
            go.Scatter(
                x=pos[: i + 1, 0],
                y=pos[: i + 1, 1],
            )
        )
        frame_traces.append(xy_traj_index)

        frame_data.append(
            go.Scatter(
                x=pos[: i + 1, 0],
                y=altitudes[: i + 1],
            )
        )
        frame_traces.append(xz_traj_index)

        frame_data.append(
            go.Scatter(
                x=pos[: i + 1, 1],
                y=altitudes[: i + 1],
            )
        )
        frame_traces.append(yz_traj_index)

        # Altitude over time
        frame_data.append(
            go.Scatter(
                x=times[: i + 1],
                y=altitudes[: i + 1],
            )
        )
        frame_traces.append(alt_traj_index)

        # Velocity over time
        frame_data.append(
            go.Scatter(
                x=times[: i + 1],
                y=velocities_mag[: i + 1],
            )
        )
        frame_traces.append(vel_mag_index)

        frame_data.append(
            go.Scatter(
                x=times[: i + 1],
                y=v_b[: i + 1, 2],
            )
        )
        frame_traces.append(vel_z_index)

        # Attitude over time
        frame_data.append(
            go.Scatter(
                x=times[: i + 1],
                y=roll_angles[: i + 1],
            )
        )
        frame_traces.append(roll_index)

        frame_data.append(
            go.Scatter(
                x=times[: i + 1],
                y=pitch_angles[: i + 1],
            )
        )
        frame_traces.append(pitch_index)

        frame_data.append(
            go.Scatter(
                x=times[: i + 1],
                y=yaw_angles[: i + 1],
            )
        )
        frame_traces.append(yaw_index)

        # Thrust over time
        frame_data.append(
            go.Scatter(
                x=times[: i + 1],
                y=actions[: i + 1, 0],
            )
        )
        frame_traces.append(thrust_index)

        frames.append(
            go.Frame(
                data=frame_data,
                traces=frame_traces,
                name=f"frame{i}",
            )
        )

    fig.frames = frames

    # Add environment diagnostics (wind) if available
    if wind is not None and wind.size > 0:
        # wind is NED [m/s]; also plot magnitude
        wind_N = wind[:, 0]
        wind_E = wind[:, 1]
        wind_D = wind[:, 2]
        wind_mag = np.linalg.norm(wind, axis=1)

        fig.add_trace(
            go.Scatter(
                x=times,
                y=wind_N,
                mode="lines",
                name="Wind North (N)",
                line=dict(color="red"),
                showlegend=True,
            ),
            row=4,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=wind_E,
                mode="lines",
                name="Wind East (E)",
                line=dict(color="green"),
                showlegend=True,
            ),
            row=4,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=wind_D,
                mode="lines",
                name="Wind Down (D)",
                line=dict(color="blue"),
                showlegend=True,
            ),
            row=4,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=wind_mag,
                mode="lines",
                name="|Wind|",
                line=dict(color="black", dash="dash"),
                showlegend=True,
            ),
            row=4,
            col=2,
        )

    # Update axes labels and layout
    fig.update_xaxes(title_text="North (m)", row=1, col=1)
    fig.update_yaxes(title_text="East (m)", row=1, col=1)
    fig.update_xaxes(title_text="North (m)", row=1, col=2)
    fig.update_yaxes(title_text="Altitude AGL (m)", row=1, col=2)
    fig.update_xaxes(title_text="East (m)", row=1, col=3)
    fig.update_yaxes(title_text="Altitude AGL (m)", row=1, col=3)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Angular Rate (rad/s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=3)
    fig.update_yaxes(title_text="Fin Deflection (deg)", row=2, col=3)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Altitude AGL (m)", row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=2)
    fig.update_yaxes(title_text="Velocity (m/s)", row=3, col=2)
    fig.update_xaxes(title_text="Time (s)", row=3, col=3)
    fig.update_yaxes(title_text="Angle (deg)", row=3, col=3)
    fig.update_xaxes(title_text="Time (s)", row=4, col=1)
    fig.update_yaxes(title_text="Thrust Command (normalized)", row=4, col=1)
    if wind is not None and wind.size > 0:
        fig.update_xaxes(title_text="Time (s)", row=4, col=2)
        fig.update_yaxes(title_text="Wind (m/s)", row=4, col=2)

    # Slider and play/pause controls for trajectory history
    slider_steps = []
    for i, t in enumerate(times):
        slider_steps.append(
            {
                "method": "animate",
                "args": [
                    [f"frame{i}"],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                    },
                ],
                "label": f"{t:.2f}",
            }
        )

    sliders = [
        {
            "active": 0,
            "currentvalue": {
                "prefix": "Time: ",
                "suffix": " s",
            },
            "pad": {"t": 50},
            "steps": slider_steps,
        }
    ]

    # Update main figure layout (including animation controls)
    fig.update_layout(
        height=1400,
        title_text=(
            f"Interactive PID Controller Dashboard - Seed {seed} | "
            f"Landed: {landed} | CEP: {cep:.3f}m | Steps: {len(times)}"
        ),
        title_x=0.5,
        hovermode="closest",
        template="plotly_white",
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "direction": "left",
                "x": 0.1,
                "y": 1.1,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 50, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=sliders,
    )

    # Create a separate 3D figure for embedding
    fig_3d = go.Figure()
    # Use altitude AGL (positive up) for z-axis so start is visually above target
    altitudes = altitudes  # already AGL from data
    fig_3d.add_trace(
        go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=altitudes,
            mode="lines",
            name="Trajectory",
            line=dict(color="blue", width=4),
            hovertemplate="North: %{x:.2f}m<br>East: %{y:.2f}m<br>Altitude: %{z:.2f}m<extra></extra>",
        )
    )
    fig_3d.add_trace(
        go.Scatter3d(
            x=[pos[0, 0]],
            y=[pos[0, 1]],
            z=[altitudes[0]],
            mode="markers",
            name="Start",
            marker=dict(size=8, color="green", symbol="circle"),
            hovertemplate="Start Position<extra></extra>",
        )
    )
    fig_3d.add_trace(
        go.Scatter3d(
            x=[pos[-1, 0]],
            y=[pos[-1, 1]],
            z=[altitudes[-1]],
            mode="markers",
            name="End",
            marker=dict(size=8, color="red", symbol="x"),
            hovertemplate="End Position<extra></extra>",
        )
    )
    fig_3d.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="markers",
            name="Target",
            marker=dict(size=6, color="black", symbol="square"),
            hovertemplate="Target<extra></extra>",
        )
    )
    fig_3d.update_layout(
        scene=dict(
            xaxis_title="North (m)",
            yaxis_title="East (m)",
            zaxis_title="Altitude AGL (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        title="3D Trajectory",
        height=500,
    )

    # Return both figures - we'll combine them in the HTML
    return fig, fig_3d


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize PID controller performance")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for episode")
    parser.add_argument("--save", type=str, default=None, help="Save figure to file (e.g., 'pid_viz.png')")
    parser.add_argument("--show", action="store_true", help="Display interactive plot")
    parser.add_argument("--animate", action="store_true", help="Create animated visualization")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for animation")
    parser.add_argument("--interval", type=int, default=50, help="Animation interval in milliseconds")
    parser.add_argument("--web", action="store_true", help="Create interactive web-based visualization (HTML)")
    args = parser.parse_args(argv)

    # Set default output directory
    output_dir = Path("simulation/Visualization/Output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    configs = Path("simulation/configs")
    pid_yaml = load_config(configs / "pid.yaml")
    delta_max = float(pid_yaml.get("pid", {}).get("delta_max", 0.349))

    # Create environment and controller
    cfg = _make_deterministic_config(None)
    env = EDFLandingEnv(cfg)
    ctrl = PIDController(pid_yaml)

    # Run episode and collect data
    print(f"Running episode with seed={args.seed}...")
    data = run_episode(env, ctrl, seed=int(args.seed))

    ep_info = data["ep_info"]
    landed = bool(ep_info.get("landed", False))
    crashed = bool(ep_info.get("crashed", False))
    oob = bool(ep_info.get("out_of_bounds", False))
    cep = float(ep_info.get("cep", float("inf")))
    reason = str(ep_info.get("termination_reason", "unknown"))

    print(f"Episode result: landed={landed}, crashed={crashed}, oob={oob}, cep={cep:.3f}m, reason={reason}")

    if args.web:
        # Create interactive web-based visualization
        if not PLOTLY_AVAILABLE:
            print("ERROR: Plotly is required for interactive web visualization.")
            print("Install with: pip install plotly")
            return 1

        print("Creating interactive web dashboard...")
        fig, fig_3d = create_interactive_dashboard(data, delta_max, int(args.seed), ep_info)

        if args.save:
            output_file = Path(args.save)
            if not output_file.is_absolute():
                output_file = output_dir / output_file
            if not str(output_file).endswith(".html"):
                output_file = Path(str(output_file).replace(".png", ".html").replace(".gif", ".html"))
                if not str(output_file).endswith(".html"):
                    output_file = output_file.with_suffix(".html")
        else:
            output_file = output_dir / f"pid_viz_seed{args.seed}.html"

        # Write combined HTML file
        # Use Plotly's HTML template system
        from plotly.io import to_html
        
        # Get HTML strings
        html_3d = to_html(fig_3d, include_plotlyjs="cdn", div_id="plot3d")
        html_2d = to_html(fig, include_plotlyjs=False, div_id="plot2d")
        
        # Simple string replacement to combine
        # Remove closing tags from first HTML and combine
        html_3d_body = html_3d.split("</body>")[0].split("<body>")[-1] if "<body>" in html_3d else html_3d
        html_2d_body = html_2d.split("</body>")[0].split("<body>")[-1] if "<body>" in html_2d else html_2d
        html_head = html_3d.split("<body>")[0] if "<body>" in html_3d else "<head></head>"
        
        combined_html = f"""{html_head}
<body style="font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5;">
    <div style="max-width: 1800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h1 style="text-align: center; color: #333;">Interactive PID Controller Dashboard</h1>
        <h2 style="color: #666; margin-top: 20px;">3D Trajectory</h2>
        <div style="margin: 20px 0;">
            {html_3d_body}
        </div>
        <h2 style="color: #666; margin-top: 40px;">2D Analysis Dashboard</h2>
        <div style="margin: 20px 0;">
            {html_2d_body}
        </div>
    </div>
</body>
</html>"""
        
        with open(str(output_file), "w", encoding="utf-8") as f:
            f.write(combined_html)
        
        print(f"Saved interactive dashboard to {output_file}")
        print(f"Open {output_file} in your web browser to view the interactive charts.")

        if args.show:
            import webbrowser
            import os

            file_path = os.path.abspath(str(output_file))
            webbrowser.open(f"file://{file_path}")

        return 0

    if args.animate:
        # Create animated visualization
        print("Creating animated visualization...")
        fig, plot_objects = create_animated_plots(data, delta_max)

        # Add title with episode info
        title = (
            f"PID Controller Animation - Seed {args.seed} | "
            f"Landed: {landed} | CEP: {cep:.3f}m | Steps: {len(data['times'])}"
        )
        fig.suptitle(title, fontsize=14, fontweight="bold")

        # Create animation
        num_frames = len(data["times"])
        anim = FuncAnimation(
            fig,
            lambda frame: animate_frame(frame, data, plot_objects, delta_max),
            frames=num_frames,
            interval=args.interval,
            blit=False,  # blit=False for 3D plots
            repeat=True,
        )

        # Save or show animation
        if args.save:
            save_path = Path(args.save)
            if not save_path.is_absolute():
                save_path = output_dir / save_path
            if str(save_path).endswith(".gif"):
                print(f"Saving animation to {save_path}...")
                anim.save(str(save_path), writer="pillow", fps=args.fps)
                print(f"Saved animation to {save_path}")
            elif str(save_path).endswith(".mp4"):
                print(f"Saving animation to {save_path}...")
                try:
                    anim.save(str(save_path), writer="ffmpeg", fps=args.fps)
                    print(f"Saved animation to {save_path}")
                except Exception as e:
                    print(f"Error saving MP4 (ffmpeg may not be installed): {e}")
                    print("Falling back to GIF format...")
                    gif_name = save_path.with_suffix(".gif")
                    anim.save(str(gif_name), writer="pillow", fps=args.fps)
                    print(f"Saved animation to {gif_name}")
            else:
                # Default to GIF
                gif_name = save_path.with_suffix(".gif")
                print(f"Saving animation to {gif_name}...")
                anim.save(str(gif_name), writer="pillow", fps=args.fps)
                print(f"Saved animation to {gif_name}")
        if args.show:
            plt.show()
        elif not args.save:
            # Default: save to GIF file
            output_file = output_dir / f"pid_viz_seed{args.seed}.gif"
            print(f"Saving animation to {output_file}...")
            anim.save(str(output_file), writer="pillow", fps=args.fps)
            print(f"Saved animation to {output_file}")

        plt.close()
        return 0

    # Static visualization (original code)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # Row 1: Trajectory plots
    ax_3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xz = fig.add_subplot(gs[0, 2])
    plot_trajectory(data, ax_3d, ax_xy, ax_xz)

    # Row 2: Body rates and fin angles
    ax_omega = fig.add_subplot(gs[1, :2])
    plot_body_rates(data, ax_omega)
    ax_fins = fig.add_subplot(gs[1, 2])
    plot_fin_angles(data, delta_max, ax_fins)

    # Row 3: Altitude and velocity
    ax_alt = fig.add_subplot(gs[2, 0])
    ax_vel = fig.add_subplot(gs[2, 1])
    plot_altitude_velocity(data, ax_alt, ax_vel)
    ax_att = fig.add_subplot(gs[2, 2])
    plot_attitude(data, ax_att)

    # Row 4: Thrust
    ax_thrust = fig.add_subplot(gs[3, :])
    plot_thrust(data, ax_thrust)

    # Add title with episode info
    title = (
        f"PID Controller Visualization - Seed {args.seed} | "
        f"Landed: {landed} | CEP: {cep:.3f}m | Steps: {len(data['times'])}"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Save or show
    if args.save:
        save_path = Path(args.save)
        if not save_path.is_absolute():
            save_path = output_dir / save_path
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    if args.show:
        plt.show()
    elif not args.save:
        # Default: save to file
        output_file = output_dir / f"pid_viz_seed{args.seed}.png"
        plt.savefig(str(output_file), dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_file}")

    plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
