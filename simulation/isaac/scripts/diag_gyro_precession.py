"""
diag_gyro_precession.py -- Gyroscopic precession diagnostic for Isaac Sim.

Validates gyroscopic precession from the spinning EDF rotor in two excitation modes:

  external:
    Phase 1 (settle): hold constant thrust, then inject external pitch/roll torque.
    Best for isolated zero-g checks.

  fin_hold:
    Phase 1 (settle): hold constant thrust under gravity with altitude locked.
    Phase 2 (hold): apply constant fin deflection to generate pitch/roll motion
    at approximately constant RPM, then observe the cross-axis precession response.

Physics note: yaw is intentionally not used as the excitation axis. Yaw rate is
parallel to the fan spin axis h_fan = [0,0,L], so yaw alone produces no precession.
Only pitch/roll rates couple with h_fan.

Usage::
    python -m simulation.isaac.scripts.diag_gyro_precession
    python -m simulation.isaac.scripts.diag_gyro_precession --mode external --torque-axis pitch --torque-mag 0.5 --duration 2.0
    python -m simulation.isaac.scripts.diag_gyro_precession --mode fin_hold --torque-axis pitch --thrust 0.68 --fin-deflection 0.5 --duration 3.0
    python -m simulation.isaac.scripts.diag_gyro_precession --disable-precession
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# SimulationApp MUST be created before any isaaclab.sim / carb imports
from isaacsim import SimulationApp  # noqa: E402

_SIM_APP: SimulationApp | None = None

# Physics constants (must match edf_landing_task.py)
_T_MAX    = 45.0      # N
_MASS     = 3.13      # kg
_GRAVITY  = 9.81      # m/s^2
_K_THRUST = 4.55e-7   # N/(rad/s)^2
_I_FAN    = 3.0e-5    # kg·m² (rotating fan blades only)

# Hover thrust: T_hover = m·g  →  norm = T_hover / T_max
_HOVER_NORM = (_MASS * _GRAVITY) / _T_MAX

# Observation indices (must match edf_landing_task.py _get_observations())
_OBS_ALTITUDE = 16   # h_agl (m)
_OBS_OMEGA_X  = 9    # roll rate p  (body frame rad/s)
_OBS_OMEGA_Y  = 10   # pitch rate q (body frame rad/s)
_OBS_OMEGA_Z  = 11   # yaw rate r   (body frame rad/s)


def _make_action(thrust_norm: float, fin_deflections: tuple[float, float, float, float] = (0, 0, 0, 0)) -> np.ndarray:
    """Build 5-dim action array."""
    action = np.zeros(5, dtype=np.float32)
    action[0] = float(np.clip(thrust_norm, -1.0, 1.0))
    action[1:5] = [float(d) for d in fin_deflections]
    return action


def _set_gravity(env, magnitude: float) -> None:
    """Override gravity magnitude on the live physics scene."""
    from pxr import UsdPhysics

    stage = env._task.sim.stage
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            UsdPhysics.Scene(prim).GetGravityMagnitudeAttr().Set(float(magnitude))
            return
    raise RuntimeError("Physics scene not found while setting gravity")


def _lock_position_at_altitude(env, altitude_m: float) -> None:
    """Lock world position at a fixed altitude while leaving rotation free."""
    task = getattr(env, "_task", None)
    if task is None or not hasattr(task, "robot"):
        return

    import torch

    pos_w = task.robot.data.root_pos_w.clone()
    quat_w = task.robot.data.root_quat_w.clone()
    pos_w[:, 2] = altitude_m
    task.robot.write_root_pose_to_sim(torch.cat([pos_w, quat_w], dim=-1))

    vel_w = task.robot.data.root_lin_vel_w.clone()
    ang_w = task.robot.data.root_ang_vel_w.clone()
    vel_w[:] = 0.0
    task.robot.write_root_velocity_to_sim(torch.cat([vel_w, ang_w], dim=-1))


def _fin_hold_action(thrust_norm: float, torque_axis: str, fin_deflection: float) -> np.ndarray:
    """Build a constant fin-hold command that excites pitch or roll."""
    d = float(np.clip(fin_deflection, -1.0, 1.0))
    if torque_axis == "pitch":
        # Positive d1=d2 produces positive pitch rate with current fin convention.
        return _make_action(thrust_norm, (d, d, 0.0, 0.0))
    # Positive roll is produced by negative common-mode on Fin_3+Fin_4.
    return _make_action(thrust_norm, (0.0, 0.0, -d, -d))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isaac Sim gyro precession diagnostic — pitch torque → roll coupling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Physics note:
  --torque-axis 'yaw' is intentionally NOT supported.
  Yaw rate r is parallel to the fan spin axis, so cross(omega_z, h_fan_z) = 0.
  Precession only occurs from pitch/roll rates.
  Use pitch or roll torque to observe the hallmark cross-axis response.
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["external", "fin_hold"],
        default="external",
        help="Excitation mode: direct external torque or gravity-on fin hold (default: external)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to Isaac env YAML config. Defaults to zero-g gyro config for external mode and single-env config for fin_hold mode.",
    )
    parser.add_argument(
        "--torque-axis",
        choices=["pitch", "roll"],
        default="pitch",
        help="Axis of applied external torque: 'pitch' (body Y) or 'roll' (body X). "
             "Default: pitch → expect roll response.",
    )
    parser.add_argument(
        "--torque-mag",
        type=float,
        default=0.5,
        help="External torque magnitude in N·m (default: 0.5)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Total test duration in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--spawn-alt",
        type=float,
        default=5.0,
        help="Spawn altitude in meters (default: 5.0)",
    )
    parser.add_argument(
        "--gravity",
        choices=["on", "off"],
        default=None,
        help="Override gravity mode. Defaults to off for external mode and on for fin_hold mode.",
    )
    parser.add_argument(
        "--disable-precession",
        action="store_true",
        default=False,
        help="Run with gyro_precession.enabled=false for A/B comparison",
    )
    parser.add_argument(
        "--thrust",
        type=float,
        default=_HOVER_NORM,
        help="Normalized constant thrust command in [0,1] (default: hover thrust)",
    )
    parser.add_argument(
        "--fin-deflection",
        type=float,
        default=0.5,
        help="Normalized fin hold command in [-1,1] for fin_hold mode (default: 0.5)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without GUI (default: False -- opens viewer)",
    )
    args = parser.parse_args()

    global _SIM_APP
    _SIM_APP = SimulationApp({"headless": args.headless})

    try:
        _run_diagnostic(args)
    finally:
        _SIM_APP.close()


def _obs_val(obs: np.ndarray, idx: int) -> float:
    if obs.ndim == 2:
        return float(obs[0, idx])
    return float(obs[idx])


def _is_done(done) -> bool:
    return bool(np.any(done))


def _run_diagnostic(args) -> None:
    """Run precession diagnostic; exits with code 0 (pass) or 1 (fail)."""
    from simulation.isaac.envs.edf_isaac_env import EDFIsaacEnv

    gravity_on = (
        args.gravity == "on"
        if args.gravity is not None
        else args.mode == "fin_hold"
    )
    thrust_norm = float(np.clip(args.thrust, 0.0, 1.0))
    omega_fan_cmd = math.sqrt((thrust_norm * _T_MAX) / _K_THRUST)
    L_cmd = _I_FAN * omega_fan_cmd
    default_config = (
        "simulation/isaac/configs/isaac_env_single.yaml"
        if gravity_on
        else "simulation/isaac/configs/isaac_env_gyro_test.yaml"
    )
    config_path = args.config or default_config

    print(f"\n{'='*50}")
    print("Gyro Precession Diagnostic Report")
    print("=" * 50)
    print(f"Isaac Sim v5.1.0 | I_fan = {_I_FAN:.2e} kg·m² | k_thrust = {_K_THRUST:.2e} N/(rad/s)²")
    prec_label = "DISABLED" if args.disable_precession else "ENABLED"
    print(f"Precession: {prec_label} | Gravity: {'ENABLED' if gravity_on else 'DISABLED'} | Spawn alt: {args.spawn_alt:.1f} m")
    print(f"Mode: {args.mode} | Torque axis: {args.torque_axis.upper()} | Duration: {args.duration:.1f} s")
    if args.mode == "external":
        print(f"External torque: {args.torque_mag:.2f} N·m")
    else:
            print(
                f"Fin hold: {args.fin_deflection:+.2f} | Constant thrust: {thrust_norm:.3f} "
                f"({thrust_norm * _T_MAX:.1f} N) | Fixed altitude: {args.spawn_alt:.2f} m"
            )
    print(f"Commanded ω_fan: {omega_fan_cmd:.0f} rad/s | L: {L_cmd:.4e} kg·m²/s")
    print("=" * 50)

    env = EDFIsaacEnv(config_path=config_path)
    task = getattr(env, "_task", None)
    if task is not None:
        task.cfg.spawn_altitude_min = args.spawn_alt
        task.cfg.spawn_altitude_max = args.spawn_alt
        task.cfg.spawn_vel_mag_min = 0.0
        task.cfg.spawn_vel_mag_max = 0.0

    if gravity_on:
        _set_gravity(env, _GRAVITY)
    else:
        _set_gravity(env, 0.0)

    if task is not None:
        if args.disable_precession:
            task._gyro_enabled = False
        task._wind_model = None
        task._anti_torque_enabled = False

    dt         = 1.0 / 120.0
    settle_s = 1.0 if args.mode == "fin_hold" else 0.5
    settle_steps = int(settle_s / dt)
    test_steps = max(1, int((args.duration - settle_s) / dt))
    log_interval = 30

    try:
        obs, _ = env.reset()
        print(f"\nPhase 1: Settle ({settle_s:.1f} s)")

        for step in range(settle_steps):
            action = _make_action(thrust_norm)
            obs, _, done, _, _ = env.step(action)
            if args.mode == "fin_hold":
                _lock_position_at_altitude(env, args.spawn_alt)
            if _is_done(done):
                obs, _ = env.reset()

        alt = _obs_val(obs, _OBS_ALTITUDE)
        print(f"  Thrust: {thrust_norm * _T_MAX:.1f} N | Altitude: {alt:.2f} m")

        torque_axis_label = "Y-axis (body pitch)" if args.torque_axis == "pitch" else "X-axis (body roll)"
        cross_axis = "roll" if args.torque_axis == "pitch" else "pitch"
        print(f"\nPhase 2: {args.mode} excitation ({args.duration - settle_s:.1f} s)")
        if args.mode == "external":
            print(f"  Applied external torque: {args.torque_mag:.2f} N·m about {torque_axis_label}")
        else:
            print(
                f"  Applied fin hold for {args.torque_axis}: {args.fin_deflection:+.2f} "
                f"at constant thrust with altitude lock"
            )
        print(f"  Expected cross-axis response: {cross_axis.upper()} via τ_gyro = −ω×h_fan")
        print()
        print(f"  {'Time(s)':<8} {'Pitch(°/s)':<12} {'Roll(°/s)':<12} {'Yaw(°/s)':<10}")

        pitch_rates = []
        roll_rates  = []
        yaw_rates   = []
        time_stamps = []

        for step in range(test_steps):
            t_sim = step * dt

            if args.mode == "external":
                action = _make_action(thrust_norm)
            else:
                action = _fin_hold_action(thrust_norm, args.torque_axis, args.fin_deflection)
            obs, _, done, _, _ = env.step(action)
            if args.mode == "fin_hold":
                _lock_position_at_altitude(env, args.spawn_alt)

            if args.mode == "external" and task is not None and hasattr(task, "robot"):
                import torch
                num_envs = task.num_envs
                ext_forces  = torch.zeros((num_envs, 1, 3), device=task.device)
                ext_torques = torch.zeros((num_envs, 1, 3), device=task.device)
                if args.torque_axis == "pitch":
                    ext_torques[:, 0, 1] = args.torque_mag
                else:
                    ext_torques[:, 0, 0] = args.torque_mag
                task.robot.set_external_force_and_torque(
                    ext_forces, ext_torques, body_ids=[task._body_id], is_global=False
                )

            if _is_done(done):
                obs, _ = env.reset()

            if step % log_interval == 0:
                p_rate = math.degrees(_obs_val(obs, _OBS_OMEGA_X))
                q_rate = math.degrees(_obs_val(obs, _OBS_OMEGA_Y))
                r_rate = math.degrees(_obs_val(obs, _OBS_OMEGA_Z))
                time_stamps.append(t_sim)
                pitch_rates.append(q_rate)
                roll_rates.append(p_rate)
                yaw_rates.append(r_rate)
                print(f"  {t_sim:<8.2f} {q_rate:<12.2f} {p_rate:<12.2f} {r_rate:<10.2f}")

        print()
        if args.torque_axis == "pitch":
            primary_rates = pitch_rates
            cross_rates = roll_rates
        else:
            primary_rates = roll_rates
            cross_rates = pitch_rates

        active_samples = [
            (t, p, c, y)
            for t, p, c, y in zip(time_stamps, primary_rates, cross_rates, yaw_rates)
            if abs(p) > 0.1 or abs(c) > 0.1
        ]
        primary_peak = max((abs(p) for _, p, _, _ in active_samples), default=0.0)
        cross_peak = max((abs(c) for _, _, c, _ in active_samples), default=0.0)

        if args.mode == "fin_hold":
            # Use the first coherent response before yaw contamination dominates.
            eval_samples = [
                (t, p, c, y)
                for t, p, c, y in active_samples
                if 0.20 <= t <= 0.60 and abs(p) > 20.0 and abs(y) < 100.0
            ]
            if eval_samples:
                ref_time, primary_ref, cross_ref, yaw_ref = eval_samples[0]
            else:
                ref_time, primary_ref, cross_ref, yaw_ref = 0.0, 0.0, 0.0, 0.0
        else:
            if active_samples:
                ref_time, primary_ref, cross_ref, yaw_ref = max(active_samples, key=lambda sample: abs(sample[1]))
            else:
                ref_time, primary_ref, cross_ref, yaw_ref = 0.0, 0.0, 0.0, 0.0

        if abs(primary_ref) > 1e-6:
            primary_sign = 1.0 if primary_ref > 0.0 else -1.0
        else:
            primary_sign = 0.0
        expected_cross_sign = (
            -primary_sign if args.torque_axis == "pitch" else primary_sign
        )
        sign_ok = expected_cross_sign == 0.0 or (cross_ref * expected_cross_sign > 0.0)

        if task is not None:
            omega_fan_actual = math.sqrt(max(float(task.thrust_actual[0].item()), 0.0) / _K_THRUST)
            L_actual = _I_FAN * omega_fan_actual
            inertia = task._body_inertia_default
            if args.torque_axis == "pitch":
                cross_inertia = float(inertia[0, 0].item())
            else:
                cross_inertia = float(inertia[1, 1].item())
        else:
            L_actual = L_cmd
            cross_inertia = 0.005
        predicted_cross_accel = math.degrees(abs(math.radians(primary_ref)) * L_actual / max(cross_inertia, 1e-6))

        print("=" * 50)
        if args.disable_precession:
            threshold = 0.05   # °/s
            passed = cross_peak < threshold
            result_str = "PASS" if passed else "FAIL"
            print(
                f"RESULT: {result_str} — {'No' if passed else 'Unexpected'} {cross_axis} response "
                f"(max {cross_axis}_rate = {cross_peak:.2f} °/s, threshold < {threshold} °/s)"
            )
        else:
            threshold = 0.1   # °/s
            if args.mode == "fin_hold":
                passed = abs(primary_ref) > 20.0 and abs(cross_ref) > 10.0 and sign_ok
            else:
                passed = primary_peak > 0.5 and cross_peak > threshold and sign_ok
            result_str = "PASS" if passed else "FAIL"
            print(
                f"RESULT: {result_str} — {cross_axis.capitalize()} response detected "
                f"(primary_peak={primary_peak:.2f} °/s, cross_peak={cross_peak:.2f} °/s, threshold > {threshold} °/s)"
            )
            if primary_peak > 0:
                ratio = cross_peak / primary_peak
                print(f"  Coupling ratio ({cross_axis}_rate/{args.torque_axis}_rate): {ratio:.3f}")
            print(
                f"  Reference sample: t={ref_time:.2f}s | primary={primary_ref:.2f} °/s | "
                f"cross={cross_ref:.2f} °/s | yaw={yaw_ref:.2f} °/s | "
                f"expected cross sign: {'+' if expected_cross_sign > 0 else '-'}"
            )
            print(f"  Sign check: {'PASS' if sign_ok else 'FAIL'}")
            print(f"  Analytical cross-axis accel estimate: {predicted_cross_accel:.3f} °/s²")

        print("=" * 50)
    finally:
        env.close()

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
