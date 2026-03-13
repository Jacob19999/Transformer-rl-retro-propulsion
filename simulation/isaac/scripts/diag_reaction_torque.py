"""
diag_reaction_torque.py -- EDF reaction-torque diagnostic for Isaac Sim.

Validates the two yaw-torque terms from the EDF fan:
  1. steady-state anti-torque: τ_anti = -k_torque * ω_fan^2
  2. RPM-ramp torque:          τ_ramp = -I_fan * dω_fan/dt

Modes:
  constant: zero-g, constant thrust, compare measured yaw-rate build-up to prediction
  ramp:     zero-g, 0 -> full-thrust ramp, verify ramp transient exceeds anti-torque-only reference
  liftoff:  normal gravity, ground start, verify climb + yaw in the full pipeline
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from isaacsim import SimulationApp  # noqa: E402

_SIM_APP: SimulationApp | None = None

_T_MAX = 45.0
_K_THRUST = 4.55e-7
_TAU_MOTOR = 0.10
_DT = 1.0 / 120.0
_ZERO_G_CONFIG = "simulation/isaac/configs/isaac_env_gyro_test.yaml"
_LIFTOFF_CONFIG = "simulation/isaac/configs/isaac_env_single.yaml"

_OBS_ALTITUDE = 16
_OBS_OMEGA_Z = 11


def _resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def _obs_val(obs: np.ndarray, idx: int) -> float:
    if obs.ndim == 2:
        return float(obs[0, idx])
    return float(obs[idx])


def _is_done(done) -> bool:
    return bool(np.any(done))


def _make_action(
    thrust_norm: float,
    fin_deflections: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    action = np.zeros(5, dtype=np.float32)
    action[0] = float(np.clip(thrust_norm, -1.0, 1.0))
    action[1:5] = [float(v) for v in fin_deflections]
    return action


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


def _set_gravity(env, magnitude: float) -> None:
    try:
        from pxr import UsdPhysics

        stage = env._task.sim.stage
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                UsdPhysics.Scene(prim).GetGravityMagnitudeAttr().Set(float(magnitude))
                return
    except Exception as exc:  # pragma: no cover - Isaac-only path
        raise RuntimeError(f"failed to set gravity magnitude to {magnitude}: {exc}") from exc
    raise RuntimeError("physics scene not found while setting gravity")


def _prepare_env_config(args) -> tuple[Path, Path | None]:
    cfg_path = _resolve_path(args.config or (_LIFTOFF_CONFIG if args.mode == "liftoff" else _ZERO_G_CONFIG))
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["vehicle_config_path"] = str(_resolve_path(args.vehicle_config))
    cfg["spawn_velocity_magnitude_range"] = [0.0, 0.0]

    if args.mode == "liftoff":
        cfg["spawn_altitude_range"] = [0.4, 0.4]
    else:
        cfg["spawn_altitude_range"] = [5.0, 5.0]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
        return Path(tmp.name), cfg_path


def _output_csv_path(mode: str, output: str | None) -> Path:
    if output:
        path = _resolve_path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    with tempfile.NamedTemporaryFile(
        prefix=f"diag_reaction_torque_{mode}_",
        suffix=".csv",
        delete=False,
    ) as tmp:
        return Path(tmp.name)


def _log_row(rows: list[list[float]], *, step: int, time_s: float, altitude_m: float, yaw_deg: float, yaw_rate_dps: float, thrust_n: float, tau_anti_nm: float, tau_ramp_nm: float) -> None:
    rows.append([
        step,
        time_s,
        altitude_m,
        yaw_deg,
        yaw_rate_dps,
        thrust_n,
        tau_anti_nm,
        tau_ramp_nm,
    ])


def _write_csv(path: Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "time_s",
            "altitude_m",
            "yaw_deg",
            "yaw_rate_dps",
            "thrust_N",
            "tau_anti_Nm",
            "tau_ramp_Nm",
        ])
        writer.writerows(rows)


def _run_constant_mode(env, args, csv_path: Path) -> bool:
    obs, _ = env.reset()
    task = env._task
    action = _make_action(args.thrust)
    steps = max(1, int(round(args.duration / _DT)))
    yaw_unwrapped: float | None = None
    rows: list[list[float]] = []

    izz = float(task._body_inertia_default[2, 2].item())
    omega_fan_cmd = math.sqrt((max(args.thrust, 0.0) * _T_MAX) / _K_THRUST)
    tau_anti_pred = -task._k_torque * omega_fan_cmd * omega_fan_cmd
    yaw_accel_pred = tau_anti_pred / izz

    pred_rates: list[float] = []
    meas_rates: list[float] = []

    for step in range(steps):
        obs, _, done, _, _ = env.step(action)
        yaw_raw = _quat_wxyz_to_yaw_rad(task.robot.data.root_quat_w[0].detach().cpu().numpy())
        yaw_unwrapped = _unwrap_angle(yaw_unwrapped, yaw_raw)
        time_s = (step + 1) * _DT
        yaw_rate = float(task.robot.data.root_ang_vel_b[0, 2].item())
        altitude = _obs_val(obs, _OBS_ALTITUDE)
        tau_anti = float(task._tau_anti_b[0, 2].item())
        tau_ramp = float(task._tau_ramp_b[0, 2].item())

        _log_row(
            rows,
            step=step,
            time_s=time_s,
            altitude_m=altitude,
            yaw_deg=math.degrees(yaw_unwrapped),
            yaw_rate_dps=math.degrees(yaw_rate),
            thrust_n=float(task.thrust_actual[0].item()),
            tau_anti_nm=tau_anti,
            tau_ramp_nm=tau_ramp,
        )

        if time_s >= 1.0:
            pred_rates.append(math.degrees(yaw_accel_pred * time_s))
            meas_rates.append(math.degrees(yaw_rate))

        if _is_done(done):
            break

    _write_csv(csv_path, rows)

    if not meas_rates:
        print("FAIL constant: no post-1s samples collected")
        return False

    print("Mode: constant")
    print(f"  thrust_cmd={args.thrust:.3f}  izz={izz:.6f} kg*m^2  k_torque={task._k_torque:.3e}")
    print(f"  csv={csv_path}")

    if args.disable_anti_torque:
        peak_rate = max(abs(v) for v in meas_rates)
        passed = peak_rate < 0.5
        print(f"  anti-torque disabled baseline peak_yaw_rate={peak_rate:.3f} deg/s  threshold<0.5 deg/s")
        print(f"RESULT: {'PASS' if passed else 'FAIL'}")
        return passed

    rel_errors = [
        abs(meas - pred) / max(abs(pred), 1e-6)
        for meas, pred in zip(meas_rates, pred_rates)
    ]
    mean_rel_error = float(np.mean(rel_errors))
    final_meas = meas_rates[-1]
    final_pred = pred_rates[-1]
    passed = mean_rel_error <= 0.10

    print(f"  predicted_tau_anti={tau_anti_pred:.6f} N*m  predicted_yaw_accel={math.degrees(yaw_accel_pred):.3f} deg/s^2")
    print(f"  final_yaw_rate_meas={final_meas:.3f} deg/s  final_yaw_rate_pred={final_pred:.3f} deg/s")
    print(f"  mean_relative_error={100.0 * mean_rel_error:.2f}%")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def _run_ramp_mode(env, args, csv_path: Path) -> bool:
    obs, _ = env.reset()
    task = env._task
    total_steps = max(1, int(round(args.duration / _DT)))
    ramp_steps = max(1, int(round(args.ramp_duration / _DT)))
    yaw_unwrapped: float | None = None
    rows: list[list[float]] = []

    izz = float(task._body_inertia_default[2, 2].item())
    peak_ramp_rate = 0.0
    peak_total_yaw_accel_dps2 = 0.0
    peak_anti_yaw_accel_dps2 = 0.0
    end_ramp_total_yaw_accel_dps2 = 0.0
    end_ramp_anti_yaw_accel_dps2 = 0.0

    for step in range(total_steps):
        thrust_cmd = min(1.0, (step + 1) / ramp_steps)
        obs, _, done, _, _ = env.step(_make_action(thrust_cmd))
        yaw_raw = _quat_wxyz_to_yaw_rad(task.robot.data.root_quat_w[0].detach().cpu().numpy())
        yaw_unwrapped = _unwrap_angle(yaw_unwrapped, yaw_raw)
        yaw_rate_dps = math.degrees(float(task.robot.data.root_ang_vel_b[0, 2].item()))
        time_s = (step + 1) * _DT
        tau_anti_nm = float(task._tau_anti_b[0, 2].item())
        tau_ramp_nm = float(task._tau_ramp_b[0, 2].item())
        total_yaw_accel_dps2 = math.degrees((tau_anti_nm + tau_ramp_nm) / izz)
        anti_yaw_accel_dps2 = math.degrees(tau_anti_nm / izz)

        if time_s <= args.ramp_duration:
            peak_ramp_rate = max(peak_ramp_rate, abs(yaw_rate_dps))
            peak_total_yaw_accel_dps2 = max(peak_total_yaw_accel_dps2, abs(total_yaw_accel_dps2))
            peak_anti_yaw_accel_dps2 = max(peak_anti_yaw_accel_dps2, abs(anti_yaw_accel_dps2))
            end_ramp_total_yaw_accel_dps2 = abs(total_yaw_accel_dps2)
            end_ramp_anti_yaw_accel_dps2 = abs(anti_yaw_accel_dps2)

        _log_row(
            rows,
            step=step,
            time_s=time_s,
            altitude_m=_obs_val(obs, _OBS_ALTITUDE),
            yaw_deg=math.degrees(yaw_unwrapped),
            yaw_rate_dps=yaw_rate_dps,
            thrust_n=float(task.thrust_actual[0].item()),
            tau_anti_nm=tau_anti_nm,
            tau_ramp_nm=tau_ramp_nm,
        )

        if _is_done(done):
            break

    _write_csv(csv_path, rows)

    print("Mode: ramp")
    print(f"  ramp_duration={args.ramp_duration:.3f}s  izz={izz:.6f} kg*m^2")
    print(f"  csv={csv_path}")

    if args.disable_anti_torque:
        passed = peak_ramp_rate < 0.5
        print(f"  anti-torque disabled baseline peak_yaw_rate={peak_ramp_rate:.3f} deg/s  threshold<0.5 deg/s")
        print(f"RESULT: {'PASS' if passed else 'FAIL'}")
        return passed

    threshold = 1.10 * end_ramp_anti_yaw_accel_dps2
    passed = end_ramp_total_yaw_accel_dps2 > threshold
    print(f"  measured_peak_ramp_yaw_rate={peak_ramp_rate:.3f} deg/s")
    print(f"  end_ramp_anti_only_yaw_accel={end_ramp_anti_yaw_accel_dps2:.3f} deg/s^2")
    print(f"  end_ramp_total_yaw_accel={end_ramp_total_yaw_accel_dps2:.3f} deg/s^2")
    print(f"  required_threshold={threshold:.3f} deg/s^2")
    print(f"  peak_total_yaw_accel_during_ramp={peak_total_yaw_accel_dps2:.3f} deg/s^2")
    print(f"  peak_anti_only_yaw_accel_during_ramp={peak_anti_yaw_accel_dps2:.3f} deg/s^2")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def _run_liftoff_mode(env, args, csv_path: Path) -> bool:
    obs, _ = env.reset()
    task = env._task
    action = _make_action(1.0)
    steps = max(1, int(round(args.duration / _DT)))
    yaw_unwrapped: float | None = None
    rows: list[list[float]] = []
    final_altitude = 0.0
    final_yaw_deg = 0.0

    for step in range(steps):
        obs, _, done, _, _ = env.step(action)
        yaw_raw = _quat_wxyz_to_yaw_rad(task.robot.data.root_quat_w[0].detach().cpu().numpy())
        yaw_unwrapped = _unwrap_angle(yaw_unwrapped, yaw_raw)
        final_altitude = _obs_val(obs, _OBS_ALTITUDE)
        final_yaw_deg = abs(math.degrees(yaw_unwrapped))

        _log_row(
            rows,
            step=step,
            time_s=(step + 1) * _DT,
            altitude_m=final_altitude,
            yaw_deg=math.degrees(yaw_unwrapped),
            yaw_rate_dps=math.degrees(float(task.robot.data.root_ang_vel_b[0, 2].item())),
            thrust_n=float(task.thrust_actual[0].item()),
            tau_anti_nm=float(task._tau_anti_b[0, 2].item()),
            tau_ramp_nm=float(task._tau_ramp_b[0, 2].item()),
        )

        if _is_done(done):
            break

    _write_csv(csv_path, rows)

    if args.disable_anti_torque:
        passed = final_yaw_deg < 0.5
        print("Mode: liftoff (anti-torque disabled baseline)")
        print(f"  final_altitude={final_altitude:.3f} m  final_abs_yaw={final_yaw_deg:.3f} deg  threshold<0.5 deg")
    else:
        passed = final_altitude > 5.0 and final_yaw_deg > 5.0
        print("Mode: liftoff")
        print(f"  final_altitude={final_altitude:.3f} m  threshold>5.0 m")
        print(f"  final_abs_yaw={final_yaw_deg:.3f} deg  threshold>5.0 deg")

    print(f"  csv={csv_path}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def _run_diagnostic(args) -> bool:
    from simulation.isaac.envs.edf_isaac_env import EDFIsaacEnv

    tmp_cfg_path, cfg_path_used = _prepare_env_config(args)
    env = EDFIsaacEnv(config_path=tmp_cfg_path)
    try:
        if args.mode in {"constant", "ramp"}:
            _set_gravity(env, 0.0)

        task = env._task
        task._gyro_enabled = False
        task._wind_model = None
        if args.disable_anti_torque:
            task._anti_torque_enabled = False

        csv_path = _output_csv_path(args.mode, args.output)
        print(f"Config: {cfg_path_used}")
        print(f"Vehicle config: {_resolve_path(args.vehicle_config)}")
        print(f"Anti-torque: {'DISABLED' if args.disable_anti_torque else 'ENABLED'}")

        if args.mode == "constant":
            return _run_constant_mode(env, args, csv_path)
        if args.mode == "ramp":
            return _run_ramp_mode(env, args, csv_path)
        return _run_liftoff_mode(env, args, csv_path)
    finally:
        env.close()
        tmp_cfg_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isaac Sim diagnostic for EDF steady-state and ramp reaction torque"
    )
    parser.add_argument("--mode", choices=["constant", "ramp", "liftoff"], default="constant")
    parser.add_argument("--thrust", type=float, default=0.68, help="Normalized thrust command for constant mode")
    parser.add_argument("--ramp-duration", type=float, default=1.0, help="0->100% thrust ramp duration in seconds")
    parser.add_argument("--duration", type=float, default=3.0, help="Total diagnostic duration in seconds")
    parser.add_argument(
        "--config",
        default=None,
        help="Isaac env YAML config. Defaults to zero-g config for constant/ramp and single-env config for liftoff.",
    )
    parser.add_argument(
        "--vehicle-config",
        default="simulation/configs/default_vehicle.yaml",
        help="Vehicle YAML config with edf.k_torque and anti_torque settings",
    )
    parser.add_argument(
        "--disable-anti-torque",
        action="store_true",
        default=False,
        help="Disable both steady-state anti-torque and RPM-ramp torque for A/B comparison",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV output path. If omitted, a temporary CSV file is created and printed.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without GUI (default: False)",
    )
    args = parser.parse_args()

    global _SIM_APP
    _SIM_APP = SimulationApp({"headless": args.headless})
    try:
        passed = _run_diagnostic(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    finally:
        if _SIM_APP is not None:
            _SIM_APP.close()

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
