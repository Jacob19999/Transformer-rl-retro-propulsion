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
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from simulation.isaac.conventions import (  # noqa: E402
    OBS_H_AGL,
    OBS_OMEGA_X,
    OBS_OMEGA_Y,
    OBS_OMEGA_Z,
    fin_axis_command,
)
from simulation.isaac.scripts._shared import (  # noqa: E402
    any_done,
    create_sim_app,
    lock_position_at_altitude,
    make_action,
    obs_scalar,
    resolve_repo_path,
    set_gravity,
)

_SIM_APP = None


def _fin_hold_action(thrust_norm: float, torque_axis: str, fin_deflection: float) -> np.ndarray:
    d = float(np.clip(fin_deflection, -1.0, 1.0))
    return make_action(thrust_norm, fin_axis_command(torque_axis, d))


def _quat_wxyz_to_yaw_rad(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in quat_wxyz]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _unwrap_angle(prev: float | None, current: float) -> float:
    if prev is None:
        return current
    delta = current - prev
    delta = math.atan2(math.sin(delta), math.cos(delta))
    return prev + delta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isaac Sim gyro precession diagnostic -- pitch or roll excitation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Physics note:
  --torque-axis 'yaw' is intentionally NOT supported.
  Yaw rate r is parallel to the fan spin axis, so cross(omega_z, h_fan_z) = 0.
  Precession only occurs from pitch/roll rates.
        """,
    )
    parser.add_argument("--mode", choices=["external", "fin_hold"], default="external")
    parser.add_argument("--config", default=None)
    parser.add_argument("--torque-axis", choices=["pitch", "roll"], default="pitch")
    parser.add_argument("--torque-mag", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--spawn-alt", type=float, default=5.0)
    parser.add_argument("--gravity", choices=["on", "off"], default=None)
    parser.add_argument("--disable-precession", action="store_true", default=False)
    parser.add_argument("--thrust", type=float, default=None)
    parser.add_argument("--fin-deflection", type=float, default=0.5)
    parser.add_argument("--headless", action="store_true", default=False)
    args = parser.parse_args()

    global _SIM_APP
    _SIM_APP = create_sim_app(headless=args.headless)
    try:
        _run_diagnostic(args)
    finally:
        _SIM_APP.close()


def _run_diagnostic(args) -> None:
    from simulation.isaac.envs.edf_isaac_env import EDFIsaacEnv

    gravity_on = args.gravity == "on" if args.gravity is not None else args.mode == "fin_hold"
    default_config = (
        "simulation/isaac/configs/isaac_env_single.yaml"
        if gravity_on
        else "simulation/isaac/configs/isaac_env_gyro_test.yaml"
    )
    config_path = resolve_repo_path(args.config or default_config)

    env = EDFIsaacEnv(config_path=config_path)
    task = env._task
    params = task.vehicle_params
    thrust_norm = (
        float(np.clip(args.thrust, 0.0, 1.0))
        if args.thrust is not None
        else float(task.hover_thrust_norm)
    )
    omega_fan_cmd = math.sqrt((thrust_norm * params.t_max) / params.k_thrust)
    commanded_ang_momentum = params.i_fan * omega_fan_cmd

    task.cfg.spawn_altitude_min = args.spawn_alt
    task.cfg.spawn_altitude_max = args.spawn_alt
    task.cfg.spawn_vel_mag_min = 0.0
    task.cfg.spawn_vel_mag_max = 0.0
    set_gravity(
        env,
        params.gravity if gravity_on else 0.0,
        prefix="diag_gyro_precession",
    )
    task.set_runtime_overrides(
        disable_wind=True,
        disable_gyro=args.disable_precession,
        disable_anti_torque=True,
        disable_gravity=False,
    )
    task._wind_model = None

    print(f"\n{'=' * 50}")
    print("Gyro Precession Diagnostic Report")
    print("=" * 50)
    print(
        f"I_fan = {params.i_fan:.2e} kg·m² | "
        f"k_thrust = {params.k_thrust:.2e} N/(rad/s)^2"
    )
    print(
        f"Precession: {'DISABLED' if args.disable_precession else 'ENABLED'} | "
        f"Gravity: {'ENABLED' if gravity_on else 'DISABLED'} | Spawn alt: {args.spawn_alt:.1f} m"
    )
    print(
        f"Mode: {args.mode} | Torque axis: {args.torque_axis.upper()} | "
        f"Duration: {args.duration:.1f} s"
    )
    if args.mode == "external":
        print(f"External torque: {args.torque_mag:.2f} N·m")
    else:
        print(
            f"Fin hold: {args.fin_deflection:+.2f} | Constant thrust: {thrust_norm:.3f} "
            f"({thrust_norm * params.t_max:.1f} N) | Fixed altitude: {args.spawn_alt:.2f} m"
        )
    print(
        f"Commanded ω_fan: {omega_fan_cmd:.0f} rad/s | "
        f"L: {commanded_ang_momentum:.4e} kg·m²/s"
    )
    print("=" * 50)

    dt = float(task._dt)
    settle_s = 1.0 if args.mode == "fin_hold" else 0.5
    settle_steps = int(settle_s / dt)
    test_steps = max(1, int((args.duration - settle_s) / dt))
    log_interval = 30

    try:
        obs, _ = env.reset()
        print(f"\nPhase 1: Settle ({settle_s:.1f} s)")

        for _ in range(settle_steps):
            obs, _, done, _, _ = env.step(make_action(thrust_norm))
            if args.mode == "fin_hold":
                lock_position_at_altitude(env, args.spawn_alt)
            if any_done(done):
                obs, _ = env.reset()

        alt = obs_scalar(obs, OBS_H_AGL)
        print(f"  Thrust: {thrust_norm * params.t_max:.1f} N | Altitude: {alt:.2f} m")

        torque_axis_label = "Y-axis (body pitch)" if args.torque_axis == "pitch" else "X-axis (body roll)"
        cross_axis = "roll" if args.torque_axis == "pitch" else "pitch"
        print(f"\nPhase 2: {args.mode} excitation ({args.duration - settle_s:.1f} s)")
        if args.mode == "external":
            print(f"  Applied external torque: {args.torque_mag:.2f} N·m about {torque_axis_label}")
        else:
            print(
                f"  Applied fin hold for {args.torque_axis}: {args.fin_deflection:+.2f} "
                "at constant thrust with altitude lock"
            )
        print(f"  Expected cross-axis response: {cross_axis.upper()} via τ_gyro = −ω×h_fan")
        print()
        print(f"  {'Time(s)':<8} {'Pitch(°/s)':<12} {'Roll(°/s)':<12} {'Yaw(°/s)':<10}")

        pitch_rates: list[float] = []
        roll_rates: list[float] = []
        yaw_rates: list[float] = []
        time_stamps: list[float] = []
        yaw_unwrapped: float | None = None

        for step in range(test_steps):
            t_sim = step * dt
            action = (
                make_action(thrust_norm)
                if args.mode == "external"
                else _fin_hold_action(thrust_norm, args.torque_axis, args.fin_deflection)
            )
            obs, _, done, _, _ = env.step(action)
            if args.mode == "fin_hold":
                lock_position_at_altitude(env, args.spawn_alt)

            if args.mode == "external":
                import torch

                ext_forces = torch.zeros((task.num_envs, 1, 3), device=task.device)
                ext_torques = torch.zeros((task.num_envs, 1, 3), device=task.device)
                ext_torques[:, 0, 1 if args.torque_axis == "pitch" else 0] = args.torque_mag
                task.robot.set_external_force_and_torque(
                    ext_forces,
                    ext_torques,
                    body_ids=[task._body_id],
                    is_global=False,
                )

            if any_done(done):
                obs, _ = env.reset()

            yaw_raw = _quat_wxyz_to_yaw_rad(task.robot.data.root_quat_w[0].detach().cpu().numpy())
            yaw_unwrapped = _unwrap_angle(yaw_unwrapped, yaw_raw)

            if step % log_interval == 0:
                p_rate = math.degrees(obs_scalar(obs, OBS_OMEGA_X))
                q_rate = math.degrees(obs_scalar(obs, OBS_OMEGA_Y))
                r_rate = math.degrees(obs_scalar(obs, OBS_OMEGA_Z))
                time_stamps.append(t_sim)
                pitch_rates.append(q_rate)
                roll_rates.append(p_rate)
                yaw_rates.append(r_rate)
                print(f"  {t_sim:<8.2f} {q_rate:<12.2f} {p_rate:<12.2f} {r_rate:<10.2f}")

        print()
        primary_rates = pitch_rates if args.torque_axis == "pitch" else roll_rates
        cross_rates = roll_rates if args.torque_axis == "pitch" else pitch_rates
        active_samples = [
            (t, p, c, y)
            for t, p, c, y in zip(time_stamps, primary_rates, cross_rates, yaw_rates)
            if abs(p) > 0.1 or abs(c) > 0.1
        ]
        primary_peak = max((abs(p) for _, p, _, _ in active_samples), default=0.0)
        cross_peak = max((abs(c) for _, _, c, _ in active_samples), default=0.0)

        if args.mode == "fin_hold":
            eval_samples = [
                (t, p, c, y)
                for t, p, c, y in active_samples
                if 0.20 <= t <= 0.60 and abs(p) > 20.0 and abs(y) < 100.0
            ]
            ref_time, primary_ref, cross_ref, yaw_ref = eval_samples[0] if eval_samples else (0.0, 0.0, 0.0, 0.0)
        else:
            ref_time, primary_ref, cross_ref, yaw_ref = (
                max(active_samples, key=lambda sample: abs(sample[1]))
                if active_samples
                else (0.0, 0.0, 0.0, 0.0)
            )

        primary_sign = 0.0 if abs(primary_ref) <= 1e-6 else (1.0 if primary_ref > 0.0 else -1.0)
        expected_cross_sign = -primary_sign if args.torque_axis == "pitch" else primary_sign
        sign_ok = expected_cross_sign == 0.0 or (cross_ref * expected_cross_sign > 0.0)

        omega_fan_actual = math.sqrt(
            max(float(task.thrust_actual[0].item()), 0.0) / params.k_thrust
        )
        angular_momentum_actual = params.i_fan * omega_fan_actual
        inertia = task._body_inertia_default
        cross_inertia = (
            float(inertia[0, 0].item())
            if args.torque_axis == "pitch"
            else float(inertia[1, 1].item())
        )
        predicted_cross_accel = math.degrees(
            abs(math.radians(primary_ref)) * angular_momentum_actual / max(cross_inertia, 1e-6)
        )

        print("=" * 50)
        if args.disable_precession:
            threshold = 0.05
            passed = cross_peak < threshold
            print(
                f"RESULT: {'PASS' if passed else 'FAIL'} — "
                f"{'No' if passed else 'Unexpected'} {cross_axis} response "
                f"(max {cross_axis}_rate = {cross_peak:.2f} °/s, threshold < {threshold} °/s)"
            )
        else:
            threshold = 0.1
            if args.mode == "fin_hold":
                passed = abs(primary_ref) > 20.0 and abs(cross_ref) > 10.0 and sign_ok
            else:
                passed = primary_peak > 0.5 and cross_peak > threshold and sign_ok
            print(
                f"RESULT: {'PASS' if passed else 'FAIL'} — {cross_axis.capitalize()} response detected "
                f"(primary_peak={primary_peak:.2f} °/s, cross_peak={cross_peak:.2f} °/s, threshold > {threshold} °/s)"
            )
            if primary_peak > 0:
                print(f"  Coupling ratio ({cross_axis}_rate/{args.torque_axis}_rate): {cross_peak / primary_peak:.3f}")
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
