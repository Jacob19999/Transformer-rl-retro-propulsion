"""
tune_pid_isaac.py -- PID tuning entry point for the Isaac hover/landing env.

This file supports two workflows:
- coordinate gain scaling around an existing PID YAML
- Isaac-side Ziegler-Nichols style identification for hover `altitude`, `roll`,
  and `pitch` loops with deterministic perturbations
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import numpy as np
import yaml

from isaacsim import SimulationApp  # pyright: ignore[reportMissingImports]

from simulation.isaac.pid_action_adapter import map_pid_action_to_isaac
from simulation.training.controllers.pid_controller import PIDController
from simulation.training.scripts.tune_pid import (
    _deep_update,
    _load_pid_yaml,
    _pid_cfg_with_gains,
    ziegler_nichols_gains,
)

if TYPE_CHECKING:
    from simulation.isaac.envs.edf_isaac_env import EDFIsaacEnv


_SIM_APP: SimulationApp | None = None
_EPISODE_LOG_COLUMNS = [
    "episode_idx",
    "seed",
    "total_reward",
    "steps",
    "landed",
    "crashed",
    "truncated",
    "out_of_bounds",
    "impact_speed",
    "lateral_dist",
    "final_h_agl",
    "mean_abs_alt_error",
    "mean_speed",
    "mean_ang_speed",
    "mean_lateral",
    "max_abs_roll_deg",
    "max_abs_pitch_deg",
    "max_abs_omega_xy",
]
_SUMMARY_COLUMNS = [
    "candidate",
    "success_rate",
    "crash_rate",
    "trunc_rate",
    "mean_reward",
    "mean_steps",
    "mean_lateral_dist_success",
    "mean_impact_speed_success",
    "mean_abs_alt_error",
    "mean_speed",
    "mean_ang_speed",
    "mean_lateral",
    "mean_max_abs_roll_deg",
    "mean_max_abs_pitch_deg",
    "mean_max_abs_omega_xy",
]
_ZN_TRIAL_COLUMNS = [
    "method",
    "loop",
    "kp_tested",
    "estimated_Ku",
    "oscillation_detected",
    "Tu_s",
    "median_amplitude",
    "max_abs_signal",
    "rms_signal",
    "final_abs_signal",
    "zero_crossings",
    "steps",
    "crashed",
    "truncated",
]
_LOOP_ORDER = ("roll", "pitch", "altitude")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PID tuning for EDF landing task in Isaac Sim."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="simulation/isaac/configs/isaac_env_single.yaml",
        help="Isaac env config YAML path (default: single-env).",
    )
    parser.add_argument(
        "--pid-config",
        type=str,
        default="simulation/configs/pid.yaml",
        help="PID YAML path compatible with PIDController (default: simulation/configs/pid.yaml).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="ziegler-nichols",
        choices=["ziegler-nichols", "relay-autotune"],
        help="Tuning method. `relay-autotune` is supported for roll/pitch loops.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Episodes per candidate for evaluation (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for env / controller (default: 0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/pid_isaac",
        help="Base directory for tuning artifacts (default: runs/pid_isaac).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Optional directory for detailed per-episode logs (default: <output-dir>/logs).",
    )
    parser.add_argument(
        "--disable-wind",
        action="store_true",
        help="Disable Isaac wind model during tuning episodes.",
    )
    parser.add_argument(
        "--disable-gyro",
        action="store_true",
        help="Disable gyro precession torque during tuning episodes.",
    )
    parser.add_argument(
        "--disable-anti-torque",
        action="store_true",
        help="Disable steady-state + ramp EDF anti-torque during tuning episodes.",
    )
    parser.add_argument(
        "--disable-yaw",
        action="store_true",
        help="Disable yaw damping term in PID (forces yaw_Kd=0 for this run).",
    )
    parser.add_argument(
        "--disable-pitch",
        action="store_true",
        help="Disable pitch inner-loop gains in PID for this run.",
    )
    parser.add_argument(
        "--disable-lateral-x",
        action="store_true",
        help="Disable lateral-x outer-loop gains in PID for this run.",
    )
    parser.add_argument(
        "--disable-lateral-y",
        action="store_true",
        help="Disable lateral-y outer-loop gains in PID for this run.",
    )
    parser.add_argument(
        "--disable-roll",
        action="store_true",
        help="Disable roll inner-loop gains in PID for this run.",
    )
    parser.add_argument(
        "--invert-lateral-x",
        action="store_true",
        help="Invert sign of lateral-x outer-loop gains in PID for this run.",
    )
    parser.add_argument(
        "--invert-pitch-damping",
        action="store_true",
        help="Invert sign of pitch-rate damping Kd in PID for this run.",
    )
    parser.add_argument(
        "--invert-altitude-damping",
        action="store_true",
        help="Invert sign of altitude-loop damping Kd in PID for this run.",
    )
    parser.add_argument(
        "--disable-gravity",
        action="store_true",
        help="Disable gravity in the Isaac scene during tuning episodes.",
    )
    parser.add_argument(
        "--isolate-pid-axis",
        action="store_true",
        default=True,
        help="In rotation test, zero fin commands for axes not under test (default: True). Roll=FwdFin+AftFin, Pitch=RightFin+LeftFin, Yaw=all.",
    )
    parser.add_argument(
        "--no-isolate-pid-axis",
        action="store_false",
        dest="isolate_pid_axis",
        help="Disable axis isolation in rotation test; all fins follow full PID output.",
    )
    parser.add_argument(
        "--rotation-zero-yaw-when-isolated",
        action="store_true",
        default=True,
        help="In rotation test, when axis isolation is enabled and axis is roll/pitch, force yaw damping off (yaw_Kd=0, max_yaw_frac=0).",
    )
    parser.add_argument(
        "--no-rotation-zero-yaw-when-isolated",
        action="store_false",
        dest="rotation_zero_yaw_when_isolated",
        help="Keep yaw damping active during isolated roll/pitch rotation tests.",
    )
    parser.add_argument(
        "--rotation-disable-noncommanded-angle-loops",
        action="store_true",
        default=False,
        help="In rotation test, zero non-commanded inner-loop angle gains (roll/pitch) so only the commanded axis loop is active.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Isaac Sim headless for tuning/verification.",
    )
    parser.add_argument(
        "--monitor-window",
        type=int,
        default=0,
        help="Unused compatibility flag; kept for CLI stability.",
    )
    parser.add_argument(
        "--monitor-direction",
        type=str,
        default="decreasing",
        choices=["decreasing", "increasing"],
        help="Unused compatibility flag; kept for CLI stability.",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="landing",
        choices=["landing", "hover", "rotation"],
        help="Evaluation objective (default: landing). Use 'rotation' for zero-g rate-tracking test.",
    )
    parser.add_argument(
        "--hover-altitude",
        type=float,
        default=5.0,
        help="Target hover altitude in meters for --test hover (default: 5.0).",
    )
    parser.add_argument(
        "--hover-alt-tol",
        type=float,
        default=0.5,
        help="Hover success altitude tolerance in meters (default: 0.5).",
    )
    parser.add_argument(
        "--zn-loop",
        type=str,
        default="none",
        choices=["none", "altitude", "roll", "pitch", "all"],
        help="Run Isaac-side ZN identification for the selected hover loop(s).",
    )
    parser.add_argument(
        "--zn-kp-start",
        type=float,
        default=0.05,
        help="Starting Kp for ZN sweep (geomspace).",
    )
    parser.add_argument(
        "--zn-kp-stop",
        type=float,
        default=20.0,
        help="Ending Kp for ZN sweep (geomspace).",
    )
    parser.add_argument(
        "--zn-kp-steps",
        type=int,
        default=24,
        help="Number of geometric sweep points for ZN Kp search.",
    )
    parser.add_argument(
        "--zn-max-seconds",
        type=float,
        default=12.0,
        help="Max simulated seconds per ZN trial.",
    )
    parser.add_argument(
        "--zn-perturb-angle-deg",
        type=float,
        default=3.0,
        help="Initial roll/pitch perturbation angle for attitude ZN runs.",
    )
    parser.add_argument(
        "--zn-perturb-rate",
        type=float,
        default=0.0,
        help="Optional initial body-rate perturbation magnitude (rad/s) for the tested axis.",
    )
    parser.add_argument(
        "--zn-altitude-offset",
        type=float,
        default=0.3,
        help="Initial altitude offset (m) for altitude ZN runs.",
    )
    parser.add_argument(
        "--zn-gain-scale",
        type=float,
        default=0.8,
        help="Scale classical ZN gains by this factor for a more conservative first pass.",
    )
    parser.add_argument(
        "--zn-verify-episodes",
        type=int,
        default=0,
        help="Verification episodes after ZN (0 => reuse --episodes).",
    )
    parser.add_argument(
        "--zn-verify-max-tilt-deg",
        type=float,
        default=20.0,
        help="Combined hover verification limit on |roll| and |pitch|.",
    )
    parser.add_argument(
        "--zn-verify-max-omega-xy",
        type=float,
        default=8.0,
        help="Combined hover verification limit on sqrt(omega_x^2 + omega_y^2).",
    )
    parser.add_argument(
        "--relay-amplitude",
        type=float,
        default=0.15,
        help="Relay output amplitude in normalized fin command for roll/pitch autotune.",
    )
    parser.add_argument(
        "--relay-hysteresis-deg",
        type=float,
        default=0.3,
        help="Relay hysteresis in degrees for roll/pitch autotune.",
    )
    parser.add_argument(
        "--relay-invert",
        action="store_true",
        help="Invert relay actuation sign for roll/pitch autotune A/B tests.",
    )
    parser.add_argument(
        "--relay-trace-steps",
        type=int,
        default=240,
        help="Number of relay trace samples to save for the first autotune run.",
    )
    return parser.parse_args()


def _ensure_sim_app(*, headless: bool) -> None:
    global _SIM_APP
    if _SIM_APP is None:
        _SIM_APP = SimulationApp({"headless": bool(headless)})


def _configure_env_for_test(
    env: "EDFIsaacEnv",
    *,
    test_mode: str,
    hover_altitude: float,
) -> None:
    if test_mode not in ("hover", "rotation"):
        return
    env._task.cfg.spawn_altitude_min = float(hover_altitude)
    env._task.cfg.spawn_altitude_max = float(hover_altitude)
    env._task.cfg.spawn_vel_mag_min = 0.0
    env._task.cfg.spawn_vel_mag_max = 0.0


def _set_env_reset_perturbation(
    env: "EDFIsaacEnv",
    *,
    altitude_offset_m: float = 0.0,
    roll_offset_rad: float = 0.0,
    pitch_offset_rad: float = 0.0,
    ang_vel_frd: tuple[float, float, float] | None = None,
) -> None:
    env.set_reset_perturbation(
        altitude_offset_m=float(altitude_offset_m),
        roll_offset_rad=float(roll_offset_rad),
        pitch_offset_rad=float(pitch_offset_rad),
        ang_vel_frd=ang_vel_frd,
    )


def _roll_pitch_from_obs(obs: np.ndarray) -> tuple[float, float]:
    o = np.asarray(obs, dtype=float).reshape(-1)
    if o.size < 9:
        return 0.0, 0.0
    gx = float(o[6])
    gy = float(o[7])
    gz = float(o[8])
    roll = float(np.arctan2(gy, max(float(np.hypot(gx, gz)), 1e-9)))
    pitch = float(np.arctan2(-gx, max(float(np.hypot(gy, gz)), 1e-9)))
    return roll, pitch


def _format_obs_state(obs: np.ndarray) -> str:
    o = np.asarray(obs, dtype=float).reshape(-1)

    def _vec3(start: int) -> tuple[float, float, float]:
        if o.size < start + 3:
            return (float("nan"), float("nan"), float("nan"))
        return (float(o[start]), float(o[start + 1]), float(o[start + 2]))

    e_px, e_py, e_pz = _vec3(0)
    v_bx, v_by, v_bz = _vec3(3)
    g_bx, g_by, g_bz = _vec3(6)
    omx, omy, omz = _vec3(9)
    wind_x, wind_y, wind_z = _vec3(13)
    roll_est, pitch_est = _roll_pitch_from_obs(o)
    twr = float(o[12]) if o.size >= 13 else float("nan")
    h_agl = float(o[16]) if o.size >= 17 else float("nan")
    speed = float(o[17]) if o.size >= 18 else float("nan")
    ang_speed = float(o[18]) if o.size >= 19 else float("nan")
    time_frac = float(o[19]) if o.size >= 20 else float("nan")
    return (
        f" e_p_body=({e_px:+.3f},{e_py:+.3f},{e_pz:+.3f})"
        f" vel_b=({v_bx:+.3f},{v_by:+.3f},{v_bz:+.3f})"
        f" g_b=({g_bx:+.3f},{g_by:+.3f},{g_bz:+.3f})"
        f" omega=({omx:+.3f},{omy:+.3f},{omz:+.3f})"
        f" roll={roll_est:+.3f} pitch={pitch_est:+.3f}"
        f" twr={twr:+.3f}"
        f" wind_ema=({wind_x:+.3f},{wind_y:+.3f},{wind_z:+.3f})"
        f" h_agl={h_agl:+.3f} speed={speed:+.3f} ang_speed={ang_speed:+.3f}"
        f" time_frac={time_frac:+.3f}"
    )


def _format_relay_run_stats(
    *,
    steps: int,
    flip_count: int,
    signal: Iterable[float],
    max_abs_roll_deg: float,
    max_abs_pitch_deg: float,
    max_lateral_dist: float,
    last_info: Mapping[str, Any],
) -> str:
    sig = np.asarray(list(signal), dtype=float).reshape(-1)
    if sig.size == 0:
        err_min = float("nan")
        err_max = float("nan")
        err_last = float("nan")
    else:
        err_min = float(np.min(sig))
        err_max = float(np.max(sig))
        err_last = float(sig[-1])
    lateral_dist = float(last_info.get("lateral_dist", float("nan")))
    h_agl = float(last_info.get("h_agl", float("nan")))
    speed = float(last_info.get("speed", float("nan")))
    return (
        f" steps={int(steps)} flips={int(flip_count)}"
        f" err_last={err_last:+.6g} err_min={err_min:+.6g} err_max={err_max:+.6g}"
        f" max_roll_deg={max_abs_roll_deg:.3f} max_pitch_deg={max_abs_pitch_deg:.3f}"
        f" lateral_dist={lateral_dist:.3f} max_lateral_dist={max_lateral_dist:.3f}"
        f" h_agl_info={h_agl:.3f} speed_info={speed:.3f}"
    )


def _hover_step_reward(
    obs: np.ndarray,
    action_isaac: np.ndarray,
    *,
    hover_altitude: float,
    hover_thrust_frac: float,
) -> float:
    o = np.asarray(obs, dtype=float).reshape(-1)
    act = np.asarray(action_isaac, dtype=float).reshape(-1)

    h_agl = float(o[16]) if o.size >= 17 else 0.0
    speed = float(o[17]) if o.size >= 18 else 0.0
    omega = float(np.linalg.norm(o[9:12])) if o.size >= 12 else 0.0
    lateral = float(np.linalg.norm(o[0:2])) if o.size >= 2 else 0.0
    fin_effort = float(np.linalg.norm(act[1:5])) if act.size >= 5 else 0.0
    thrust_err = (
        abs(float(act[0]) - float(hover_thrust_frac)) if act.size >= 1 else 0.0
    )

    alt_err = abs(h_agl - float(hover_altitude))
    reward = (
        1.5
        - 2.0 * alt_err
        - 0.4 * speed
        - 0.2 * omega
        - 0.3 * lateral
        - 0.05 * fin_effort
        - 0.05 * thrust_err
    )
    return float(reward)


def _hover_success(
    *,
    truncated: bool,
    crashed: bool,
    final_h_agl: float,
    hover_altitude: float,
    hover_alt_tolerance: float,
    max_abs_roll_deg: float,
    max_abs_pitch_deg: float,
    max_abs_omega_xy: float,
    verification_limits: Mapping[str, float] | None,
) -> bool:
    if not (bool(truncated) and not bool(crashed)):
        return False
    if abs(float(final_h_agl) - float(hover_altitude)) > float(hover_alt_tolerance):
        return False
    if verification_limits is None:
        return True
    max_tilt_deg = float(verification_limits.get("max_tilt_deg", float("inf")))
    max_omega_xy = float(verification_limits.get("max_omega_xy", float("inf")))
    return (
        max_abs_roll_deg <= max_tilt_deg
        and max_abs_pitch_deg <= max_tilt_deg
        and max_abs_omega_xy <= max_omega_xy
    )


def _episode_rollout(
    env: "EDFIsaacEnv",
    pid_yaml: Mapping[str, Any],
    *,
    seed: int,
    test_mode: str,
    hover_altitude: float,
    hover_alt_tolerance: float,
    debug_print: bool,
    verification_limits: Mapping[str, float] | None,
) -> dict[str, Any]:
    obs, _info = env.reset(seed=seed)
    ctrl = PIDController(pid_yaml)
    ctrl.reset()
    hover_thrust_frac = float(env._task.hover_thrust_norm)

    terminated = False
    truncated = False
    total_reward = 0.0
    steps = 0
    last_info: dict[str, Any] = {}
    sum_abs_alt_error = 0.0
    sum_speed = 0.0
    sum_ang_speed = 0.0
    sum_lateral = 0.0
    max_abs_roll_deg = 0.0
    max_abs_pitch_deg = 0.0
    max_abs_omega_xy = 0.0

    while True:
        action_pid, dbg = ctrl.get_action_with_debug(obs)
        action_isaac = map_pid_action_to_isaac(
            action_pid,
            hover_thrust_frac=hover_thrust_frac,
        )
        obs, rew, terminated, truncated, info = env.step(action_isaac)
        terminated = bool(np.asarray(terminated).any())
        truncated = bool(np.asarray(truncated).any())

        rew_arr = np.asarray(rew, dtype=float)
        rew_scalar = (
            float(rew_arr.reshape(())) if rew_arr.size == 1 else float(rew_arr.mean())
        )
        o = np.asarray(obs, dtype=float).reshape(-1)
        h_agl = float(o[16]) if o.size >= 17 else float("nan")
        speed = float(o[17]) if o.size >= 18 else float("nan")
        ang_speed = float(np.linalg.norm(o[9:12])) if o.size >= 12 else float("nan")
        lateral = float(np.linalg.norm(o[0:2])) if o.size >= 2 else float("nan")
        roll_est, pitch_est = _roll_pitch_from_obs(o)
        omega_xy = float(np.linalg.norm(o[9:11])) if o.size >= 11 else float("nan")

        if test_mode == "hover":
            rew_scalar = _hover_step_reward(
                o,
                action_isaac,
                hover_altitude=hover_altitude,
                hover_thrust_frac=hover_thrust_frac,
            )
        total_reward += rew_scalar
        steps += 1
        last_info = dict(info)
        if not math.isnan(h_agl):
            sum_abs_alt_error += abs(h_agl - hover_altitude)
        if not math.isnan(speed):
            sum_speed += speed
        if not math.isnan(ang_speed):
            sum_ang_speed += ang_speed
        if not math.isnan(lateral):
            sum_lateral += lateral
        max_abs_roll_deg = max(max_abs_roll_deg, abs(math.degrees(roll_est)))
        max_abs_pitch_deg = max(max_abs_pitch_deg, abs(math.degrees(pitch_est)))
        if not math.isnan(omega_xy):
            max_abs_omega_xy = max(max_abs_omega_xy, omega_xy)

        if debug_print and (steps <= 5 or steps % 20 == 0 or terminated or truncated):
            e_px = float(o[0]) if o.size >= 1 else float("nan")
            e_py = float(o[1]) if o.size >= 2 else float("nan")
            omx = float(o[9]) if o.size >= 10 else float("nan")
            omy = float(o[10]) if o.size >= 11 else float("nan")
            omz = float(o[11]) if o.size >= 12 else float("nan")
            print(
                "[tune_pid_isaac]"
                f" step={steps:4d}"
                f" h_agl={h_agl:6.2f}m"
                f" speed={speed:6.2f}m/s"
                f" e_p_body=({e_px:+5.2f},{e_py:+5.2f})"
                f" omega=({omx:+6.3f},{omy:+6.3f},{omz:+6.3f})"
                f" thrust_pid={float(action_pid[0]):+5.2f}"
                f" thrust_isaac={float(action_isaac[0]):5.2f}"
                f" alt_rate={dbg['alt_rate']:+6.3f}"
                f" thrust_raw={dbg['thrust_cmd_raw']:+6.3f}"
                f" alt_err={dbg['alt_error']:+6.3f}"
                f" alt_int={dbg['alt_integral']:+6.3f}"
                f" pitch_des={dbg['pitch_des']:+6.3f}"
                f" pitch_est={dbg['pitch_est']:+6.3f}"
                f" pitch_err={dbg['pitch_error']:+6.3f}"
                f" pitch_cmd={dbg['pitch_cmd']:+6.3f}"
                f" [s={dbg.get('gain_scale', float('nan')):+.3f}"
                f" Kp={dbg.get('pitch_Kp', float('nan')):+.3f}"
                f" Kd={dbg.get('pitch_Kd', float('nan')):+.3f}"
                f" ff={dbg.get('gyro_ff', float('nan')):+.3f}"
                f" omy_used={dbg.get('omega_y', float('nan')):+.3f}]"
                f" roll_cmd={dbg['roll_cmd']:+6.3f}"
                f" yaw_cmd={dbg['yaw_total']:+6.3f}"
                f" reward={rew_scalar:+7.3f}"
            )
        if terminated or truncated:
            break

    mean_abs_alt_error = sum_abs_alt_error / max(steps, 1)
    mean_speed = sum_speed / max(steps, 1)
    mean_ang_speed = sum_ang_speed / max(steps, 1)
    mean_lateral = sum_lateral / max(steps, 1)
    final_h_agl = float(last_info.get("h_agl", float("nan")))
    final_speed = float(last_info.get("speed", float("nan")))
    crashed = bool(last_info.get("crashed", False))
    landed = bool(last_info.get("landed", False))
    hover_success = _hover_success(
        truncated=bool(truncated),
        crashed=crashed,
        final_h_agl=final_h_agl,
        hover_altitude=hover_altitude,
        hover_alt_tolerance=hover_alt_tolerance,
        max_abs_roll_deg=max_abs_roll_deg,
        max_abs_pitch_deg=max_abs_pitch_deg,
        max_abs_omega_xy=max_abs_omega_xy,
        verification_limits=verification_limits,
    )
    success = hover_success if test_mode == "hover" else landed

    return {
        "total_reward": total_reward,
        "steps": steps,
        "landed": success,
        "crashed": crashed,
        "truncated": bool(truncated),
        "out_of_bounds": bool(last_info.get("out_of_bounds", False)),
        "impact_speed": (
            final_speed
            if test_mode == "hover"
            else float(last_info.get("impact_speed", float("nan")))
        ),
        "lateral_dist": float(last_info.get("lateral_dist", float("nan"))),
        "final_h_agl": final_h_agl,
        "mean_abs_alt_error": mean_abs_alt_error,
        "mean_speed": mean_speed,
        "mean_ang_speed": mean_ang_speed,
        "mean_lateral": mean_lateral,
        "max_abs_roll_deg": max_abs_roll_deg,
        "max_abs_pitch_deg": max_abs_pitch_deg,
        "max_abs_omega_xy": max_abs_omega_xy,
    }


def _summarize_episode_rows(episode_rows: list[list[Any]]) -> dict[str, float]:
    if not episode_rows:
        return {
            "mean_reward": float("nan"),
            "mean_steps": float("nan"),
            "success_rate": float("nan"),
            "crash_rate": float("nan"),
            "trunc_rate": float("nan"),
            "mean_lateral_dist_success": float("nan"),
            "mean_impact_speed_success": float("nan"),
            "mean_abs_alt_error": float("nan"),
            "mean_speed": float("nan"),
            "mean_ang_speed": float("nan"),
            "mean_lateral": float("nan"),
            "mean_max_abs_roll_deg": float("nan"),
            "mean_max_abs_pitch_deg": float("nan"),
            "mean_max_abs_omega_xy": float("nan"),
        }

    rewards = np.array([float(row[2]) for row in episode_rows], dtype=float)
    steps = np.array([float(row[3]) for row in episode_rows], dtype=float)
    landed = np.array([bool(row[4]) for row in episode_rows], dtype=bool)
    crashed = np.array([bool(row[5]) for row in episode_rows], dtype=bool)
    truncated = np.array([bool(row[6]) for row in episode_rows], dtype=bool)
    impact_speed = np.array([float(row[8]) for row in episode_rows], dtype=float)
    lateral_dist = np.array([float(row[9]) for row in episode_rows], dtype=float)
    abs_alt_err = np.array([float(row[11]) for row in episode_rows], dtype=float)
    mean_speed = np.array([float(row[12]) for row in episode_rows], dtype=float)
    mean_ang_speed = np.array([float(row[13]) for row in episode_rows], dtype=float)
    mean_lateral = np.array([float(row[14]) for row in episode_rows], dtype=float)
    max_roll = np.array([float(row[15]) for row in episode_rows], dtype=float)
    max_pitch = np.array([float(row[16]) for row in episode_rows], dtype=float)
    max_omega_xy = np.array([float(row[17]) for row in episode_rows], dtype=float)

    return {
        "mean_reward": float(rewards.mean()),
        "mean_steps": float(steps.mean()),
        "success_rate": float(landed.mean()),
        "crash_rate": float(crashed.mean()),
        "trunc_rate": float(truncated.mean()),
        "mean_lateral_dist_success": (
            float(lateral_dist[landed].mean()) if landed.any() else float("nan")
        ),
        "mean_impact_speed_success": (
            float(impact_speed[landed].mean()) if landed.any() else float("nan")
        ),
        "mean_abs_alt_error": float(abs_alt_err.mean()),
        "mean_speed": float(mean_speed.mean()),
        "mean_ang_speed": float(mean_ang_speed.mean()),
        "mean_lateral": float(mean_lateral.mean()),
        "mean_max_abs_roll_deg": float(max_roll.mean()),
        "mean_max_abs_pitch_deg": float(max_pitch.mean()),
        "mean_max_abs_omega_xy": float(max_omega_xy.mean()),
    }


def _evaluate_single_env(
    config_path: Path,
    pid_yaml: Mapping[str, Any],
    *,
    episodes: int,
    seed: int,
    test_mode: str,
    hover_altitude: float,
    hover_alt_tolerance: float,
    disable_wind: bool,
    disable_gyro: bool,
    disable_anti_torque: bool,
    disable_gravity: bool,
    headless: bool,
    debug_print: bool,
    episode_log_path: Path | None = None,
    verification_limits: Mapping[str, float] | None = None,
    reset_perturbation: Mapping[str, float] | None = None,
) -> dict[str, float]:
    _ensure_sim_app(headless=headless)
    from simulation.isaac.envs.edf_isaac_env import EDFIsaacEnv

    env = EDFIsaacEnv(
        config_path=str(config_path),
        seed=seed,
        disable_wind=disable_wind,
        disable_gyro=disable_gyro,
        disable_anti_torque=disable_anti_torque,
        disable_gravity=disable_gravity,
    )
    _configure_env_for_test(env, test_mode=test_mode, hover_altitude=hover_altitude)
    _set_env_reset_perturbation(
        env,
        altitude_offset_m=float((reset_perturbation or {}).get("altitude_offset_m", 0.0)),
        roll_offset_rad=float((reset_perturbation or {}).get("roll_offset_rad", 0.0)),
        pitch_offset_rad=float((reset_perturbation or {}).get("pitch_offset_rad", 0.0)),
        ang_vel_frd=(
            float((reset_perturbation or {}).get("ang_vel_x", 0.0)),
            float((reset_perturbation or {}).get("ang_vel_y", 0.0)),
            float((reset_perturbation or {}).get("ang_vel_z", 0.0)),
        ),
    )

    num_envs = env.num_envs
    episode_rows: list[list[Any]] = []

    try:
        if num_envs == 1:
            for i in range(episodes):
                ep_seed = seed + i
                print(f"[tune_pid_isaac] episode {i + 1}/{episodes} seed={ep_seed}")
                stats = _episode_rollout(
                    env,
                    pid_yaml,
                    seed=ep_seed,
                    test_mode=test_mode,
                    hover_altitude=hover_altitude,
                    hover_alt_tolerance=hover_alt_tolerance,
                    debug_print=debug_print,
                    verification_limits=verification_limits,
                )
                episode_rows.append(
                    [
                        i,
                        ep_seed,
                        stats["total_reward"],
                        stats["steps"],
                        stats["landed"],
                        stats["crashed"],
                        stats["truncated"],
                        stats["out_of_bounds"],
                        stats["impact_speed"],
                        stats["lateral_dist"],
                        stats["final_h_agl"],
                        stats["mean_abs_alt_error"],
                        stats["mean_speed"],
                        stats["mean_ang_speed"],
                        stats["mean_lateral"],
                        stats["max_abs_roll_deg"],
                        stats["max_abs_pitch_deg"],
                        stats["max_abs_omega_xy"],
                    ]
                )
        else:
            import torch

            controllers = [PIDController(pid_yaml) for _ in range(num_envs)]
            for controller in controllers:
                controller.reset()

            hover_thrust_frac = float(env._task.hover_thrust_norm)
            global_ep = 0
            cur_rewards = np.zeros(num_envs, dtype=float)
            cur_steps = np.zeros(num_envs, dtype=int)
            cur_abs_alt_err = np.zeros(num_envs, dtype=float)
            cur_speed = np.zeros(num_envs, dtype=float)
            cur_ang_speed = np.zeros(num_envs, dtype=float)
            cur_lateral = np.zeros(num_envs, dtype=float)
            cur_max_roll_deg = np.zeros(num_envs, dtype=float)
            cur_max_pitch_deg = np.zeros(num_envs, dtype=float)
            cur_max_omega_xy = np.zeros(num_envs, dtype=float)
            obs, _info = env.reset(seed=seed)

            while global_ep < episodes:
                obs_arr = np.asarray(obs, dtype=float)
                if obs_arr.ndim == 1:
                    obs_arr = obs_arr.reshape(1, -1)

                actions_pid = np.zeros((num_envs, 5), dtype=np.float32)
                for i in range(num_envs):
                    if global_ep >= episodes:
                        break
                    actions_pid[i] = controllers[i].get_action(obs_arr[i])

                actions_isaac = np.zeros_like(actions_pid)
                for i in range(num_envs):
                    actions_isaac[i] = map_pid_action_to_isaac(
                        actions_pid[i],
                        hover_thrust_frac=hover_thrust_frac,
                    )

                obs, rew, terminated, truncated, info = env.step(
                    actions_isaac.astype(np.float32)
                )
                obs_next = np.asarray(obs, dtype=float)
                if obs_next.ndim == 1:
                    obs_next = obs_next.reshape(1, -1)
                if test_mode == "hover":
                    rew_arr = np.array(
                        [
                            _hover_step_reward(
                                obs_next[i],
                                actions_isaac[i],
                                hover_altitude=hover_altitude,
                                hover_thrust_frac=hover_thrust_frac,
                            )
                            for i in range(num_envs)
                        ],
                        dtype=float,
                    )
                else:
                    rew_arr = np.asarray(rew, dtype=float).reshape(num_envs)
                term_arr = np.asarray(terminated, dtype=bool).reshape(num_envs)
                trunc_arr = np.asarray(truncated, dtype=bool).reshape(num_envs)
                done_arr = term_arr | trunc_arr
                landed_arr = np.asarray(info.get("landed"), dtype=bool).reshape(num_envs)
                crashed_arr = np.asarray(info.get("crashed"), dtype=bool).reshape(num_envs)
                oob_arr = np.asarray(info.get("out_of_bounds"), dtype=bool).reshape(num_envs)
                impact_arr = np.asarray(info.get("impact_speed"), dtype=float).reshape(num_envs)
                lateral_arr = np.asarray(info.get("lateral_dist"), dtype=float).reshape(num_envs)
                h_agl_arr = np.asarray(info.get("h_agl"), dtype=float).reshape(num_envs)
                # EDFIsaacEnv does not currently expose per-env speed in the info dict
                # (it only reports impact_speed / lateral_dist / h_agl).  Derive the
                # instantaneous speed for episode logging directly from the observation
                # channel instead: obs_next[:, 17] is speed in the shared layout.
                speed_arr = obs_next[:, 17]

                cur_rewards += rew_arr
                cur_steps += 1
                cur_abs_alt_err += np.abs(obs_next[:, 16] - float(hover_altitude))
                cur_speed += obs_next[:, 17]
                cur_ang_speed += np.linalg.norm(obs_next[:, 9:12], axis=1)
                cur_lateral += np.linalg.norm(obs_next[:, 0:2], axis=1)
                roll_pitch = np.array(
                    [_roll_pitch_from_obs(obs_next[i]) for i in range(num_envs)],
                    dtype=float,
                )
                cur_max_roll_deg = np.maximum(
                    cur_max_roll_deg, np.abs(np.rad2deg(roll_pitch[:, 0]))
                )
                cur_max_pitch_deg = np.maximum(
                    cur_max_pitch_deg, np.abs(np.rad2deg(roll_pitch[:, 1]))
                )
                cur_max_omega_xy = np.maximum(
                    cur_max_omega_xy,
                    np.linalg.norm(obs_next[:, 9:11], axis=1),
                )

                done_indices = np.nonzero(done_arr)[0]
                if done_indices.size == 0:
                    continue

                for i in done_indices:
                    if global_ep >= episodes:
                        break
                    ep_idx = global_ep
                    ep_seed = seed + ep_idx
                    success = bool(landed_arr[i])
                    if test_mode == "hover":
                        success = _hover_success(
                            truncated=bool(trunc_arr[i]),
                            crashed=bool(crashed_arr[i]),
                            final_h_agl=float(h_agl_arr[i]),
                            hover_altitude=hover_altitude,
                            hover_alt_tolerance=hover_alt_tolerance,
                            max_abs_roll_deg=float(cur_max_roll_deg[i]),
                            max_abs_pitch_deg=float(cur_max_pitch_deg[i]),
                            max_abs_omega_xy=float(cur_max_omega_xy[i]),
                            verification_limits=verification_limits,
                        )
                    episode_rows.append(
                        [
                            ep_idx,
                            ep_seed,
                            float(cur_rewards[i]),
                            int(cur_steps[i]),
                            success,
                            bool(crashed_arr[i]),
                            bool(trunc_arr[i]),
                            bool(oob_arr[i]),
                            float(speed_arr[i])
                            if test_mode == "hover"
                            else float(impact_arr[i]),
                            float(lateral_arr[i]),
                            float(h_agl_arr[i]),
                            float(cur_abs_alt_err[i] / max(cur_steps[i], 1)),
                            float(cur_speed[i] / max(cur_steps[i], 1)),
                            float(cur_ang_speed[i] / max(cur_steps[i], 1)),
                            float(cur_lateral[i] / max(cur_steps[i], 1)),
                            float(cur_max_roll_deg[i]),
                            float(cur_max_pitch_deg[i]),
                            float(cur_max_omega_xy[i]),
                        ]
                    )
                    global_ep += 1

                    if global_ep < episodes:
                        env_ids = torch.tensor(
                            [i], device=env._task.device, dtype=torch.long
                        )
                        env._task._reset_idx(env_ids)
                        obs = env._task._get_observations()["policy"].cpu().numpy()
                        controllers[i].reset()
                        cur_rewards[i] = 0.0
                        cur_steps[i] = 0
                        cur_abs_alt_err[i] = 0.0
                        cur_speed[i] = 0.0
                        cur_ang_speed[i] = 0.0
                        cur_lateral[i] = 0.0
                        cur_max_roll_deg[i] = 0.0
                        cur_max_pitch_deg[i] = 0.0
                        cur_max_omega_xy[i] = 0.0
    finally:
        env.close()

    if episode_log_path is not None and episode_rows:
        episode_log_path.parent.mkdir(parents=True, exist_ok=True)
        with episode_log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(_EPISODE_LOG_COLUMNS)
            writer.writerows(episode_rows)

    return _summarize_episode_rows(episode_rows)


def _scaled_pid_candidate(
    base_pid_yaml: Mapping[str, Any],
    *,
    altitude: float | None = None,
    lateral_x: float | None = None,
    lateral_y: float | None = None,
    roll: float | None = None,
    pitch: float | None = None,
    yaw_kd: float | None = None,
) -> dict[str, Any]:
    root = copy.deepcopy(dict(base_pid_yaml))
    pid = root.get("pid", root)
    outer = pid["outer_loop"]
    inner = pid["inner_loop"]

    def _scale(section: dict[str, Any], factor: float, keys: tuple[str, ...]) -> None:
        for key in keys:
            if key in section:
                section[key] = float(section[key]) * float(factor)

    if altitude is not None:
        _scale(outer["altitude"], altitude, ("Kp", "Ki", "Kd"))
    if lateral_x is not None:
        _scale(outer["lateral_x"], lateral_x, ("Kp", "Kd"))
    if lateral_y is not None:
        _scale(outer["lateral_y"], lateral_y, ("Kp", "Kd"))
    if roll is not None:
        _scale(inner["roll"], roll, ("Kp", "Kd"))
    if pitch is not None:
        _scale(inner["pitch"], pitch, ("Kp", "Kd"))
    if yaw_kd is not None and "yaw_Kd" in inner:
        inner["yaw_Kd"] = float(inner["yaw_Kd"]) * float(yaw_kd)

    root["pid"] = pid
    return root


def _candidate_rank_key(
    stats: Mapping[str, float],
    *,
    test_mode: str,
) -> tuple[float, float, float, float, float]:
    def _nan_to_inf(value: float) -> float:
        return float("inf") if math.isnan(float(value)) else float(value)

    if test_mode == "hover":
        return (
            float(stats["success_rate"]),
            -_nan_to_inf(float(stats["mean_abs_alt_error"])),
            -_nan_to_inf(float(stats["mean_max_abs_roll_deg"])),
            -_nan_to_inf(float(stats["mean_max_abs_omega_xy"])),
            float(stats["mean_reward"]),
        )

    return (
        float(stats["success_rate"]),
        -_nan_to_inf(float(stats["mean_lateral_dist_success"])),
        -_nan_to_inf(float(stats["mean_impact_speed_success"])),
        -float(stats["crash_rate"]),
        float(stats["mean_reward"]),
    )


def _with_hover_target(pid_yaml: Mapping[str, Any], hover_altitude: float) -> dict[str, Any]:
    root = copy.deepcopy(dict(pid_yaml))
    pid = dict(root.get("pid", root))
    outer = dict(pid.get("outer_loop", {}))
    altitude = dict(outer.get("altitude", {}))
    altitude["target_h_agl"] = float(hover_altitude)
    outer["altitude"] = altitude
    pid["outer_loop"] = outer
    root["pid"] = pid
    return root


def _apply_runtime_pid_overrides(
    pid_yaml: Mapping[str, Any],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    root = copy.deepcopy(dict(pid_yaml))
    pid = dict(root.get("pid", root))
    outer = dict(pid.get("outer_loop", {}))
    inner = dict(pid.get("inner_loop", {}))

    if bool(args.disable_yaw):
        inner["yaw_Kd"] = 0.0
        inner["max_yaw_frac"] = 0.0

    if bool(args.disable_pitch):
        pitch = dict(inner.get("pitch", {}))
        pitch["Kp"] = 0.0
        pitch["Kd"] = 0.0
        inner["pitch"] = pitch
        inner["gyro_ff"] = 0.0

    if bool(args.disable_roll):
        roll = dict(inner.get("roll", {}))
        roll["Kp"] = 0.0
        roll["Kd"] = 0.0
        inner["roll"] = roll
        inner["gyro_ff"] = 0.0

    if bool(args.disable_lateral_x):
        lateral_x = dict(outer.get("lateral_x", {}))
        lateral_x["Kp"] = 0.0
        lateral_x["Kd"] = 0.0
        outer["lateral_x"] = lateral_x

    if bool(args.disable_lateral_y):
        lateral_y = dict(outer.get("lateral_y", {}))
        lateral_y["Kp"] = 0.0
        lateral_y["Kd"] = 0.0
        outer["lateral_y"] = lateral_y

    if bool(args.invert_lateral_x):
        lateral_x = dict(outer.get("lateral_x", {}))
        lateral_x["Kp"] = -float(lateral_x.get("Kp", 0.0))
        lateral_x["Kd"] = -float(lateral_x.get("Kd", 0.0))
        outer["lateral_x"] = lateral_x

    if bool(args.invert_pitch_damping):
        pitch = dict(inner.get("pitch", {}))
        pitch["Kd"] = -float(pitch.get("Kd", 0.0))
        inner["pitch"] = pitch

    if bool(args.invert_altitude_damping):
        altitude = dict(outer.get("altitude", {}))
        altitude["Kd"] = -float(altitude.get("Kd", 0.0))
        outer["altitude"] = altitude

    pid["outer_loop"] = outer
    pid["inner_loop"] = inner
    root["pid"] = pid
    return root


def _signal_from_obs(loop_name: str, obs: np.ndarray, *, hover_altitude: float) -> float:
    o = np.asarray(obs, dtype=float).reshape(-1)
    if loop_name == "altitude":
        return float(hover_altitude - float(o[16]))
    roll_est, pitch_est = _roll_pitch_from_obs(o)
    if loop_name == "roll":
        return float(-roll_est)
    if loop_name == "pitch":
        return float(-pitch_est)
    raise ValueError(f"Unknown ZN loop {loop_name!r}")


def _signal_min_amplitude(loop_name: str) -> float:
    if loop_name == "altitude":
        return 0.03
    return math.radians(0.5)


def _signal_stats(signal: Iterable[float]) -> tuple[float, float, float]:
    sig = np.asarray(list(signal), dtype=float).reshape(-1)
    if sig.size == 0:
        return 0.0, 0.0, 0.0
    max_abs = float(np.max(np.abs(sig)))
    rms = float(np.sqrt(np.mean(np.square(sig))))
    final_abs = float(abs(sig[-1]))
    return max_abs, rms, final_abs


def _detect_sustained_oscillation(
    signal: Iterable[float],
    *,
    loop_name: str,
    dt: float,
) -> tuple[float | None, float | None, int]:
    sig = np.asarray(list(signal), dtype=float).reshape(-1)
    if sig.size < 80:
        return None, None, 0
    start = int(0.3 * sig.size)
    tail = sig[start:].copy()
    tail -= float(np.mean(tail))
    cross = np.where(tail[:-1] * tail[1:] <= 0.0)[0]
    if cross.size < 6:
        return None, None, int(cross.size)

    half_periods = np.diff(cross).astype(float) * float(dt)
    if half_periods.size < 4:
        return None, None, int(cross.size)
    Tu = 2.0 * float(np.median(half_periods))

    amps: list[float] = []
    for a, b in zip(cross[:-1], cross[1:], strict=False):
        segment = tail[int(a) : int(b) + 1]
        if segment.size == 0:
            continue
        amps.append(float(np.max(np.abs(segment))))
    if len(amps) < 4:
        return None, None, int(cross.size)
    amp_arr = np.asarray(amps[-6:], dtype=float)
    amp = float(np.median(amp_arr))
    if amp < _signal_min_amplitude(loop_name):
        return None, None, int(cross.size)
    amp_lo = max(float(np.min(amp_arr)), 1e-9)
    amp_ratio = float(np.max(amp_arr) / amp_lo)
    if amp_ratio > 1.75:
        return None, None, int(cross.size)
    return float(Tu), float(amp), int(cross.size)


def _relay_action(
    loop_name: str,
    relay_state: float,
    *,
    hover_thrust_frac: float,
    invert: bool = False,
) -> np.ndarray:
    action = np.zeros(5, dtype=np.float32)
    action[0] = float(hover_thrust_frac)
    u = float(np.clip(relay_state, -1.0, 1.0))
    if bool(invert):
        u = -u
    if loop_name == "roll":
        # Positive roll_cmd maps to negative fin3/fin4 normalized commands.
        action[3] = -u
        action[4] = -u
        return action
    if loop_name == "pitch":
        action[1] = u
        action[2] = u
        return action
    raise ValueError(f"Relay autotune only supports roll/pitch, got {loop_name!r}")


def _estimate_ku_from_relay(
    *,
    relay_amplitude: float,
    oscillation_amplitude: float,
) -> float | None:
    a = float(oscillation_amplitude)
    if a <= 1e-9:
        return None
    d = abs(float(relay_amplitude))
    return float((4.0 * d) / (math.pi * a))


def _write_relay_trace(
    trace_rows: list[list[float]],
    *,
    output_dir: Path,
    loop_name: str,
    inverted: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "inv" if bool(inverted) else "norm"
    path = output_dir / f"relay_trace_{loop_name}_{suffix}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "error", "relay_state"])
        writer.writerows(trace_rows)
    return path


def _zn_reset_perturbation(
    loop_name: str,
    *,
    perturb_angle_deg: float,
    perturb_rate: float,
    altitude_offset_m: float,
) -> dict[str, float]:
    pert: dict[str, float] = {
        "altitude_offset_m": 0.0,
        "roll_offset_rad": 0.0,
        "pitch_offset_rad": 0.0,
        "ang_vel_x": 0.0,
        "ang_vel_y": 0.0,
        "ang_vel_z": 0.0,
    }
    if loop_name == "altitude":
        pert["altitude_offset_m"] = float(altitude_offset_m)
        return pert
    if loop_name == "roll":
        pert["roll_offset_rad"] = math.radians(float(perturb_angle_deg))
        pert["ang_vel_x"] = float(perturb_rate)
        return pert
    if loop_name == "pitch":
        pert["pitch_offset_rad"] = math.radians(float(perturb_angle_deg))
        pert["ang_vel_y"] = float(perturb_rate)
        return pert
    raise ValueError(f"Unknown ZN loop {loop_name!r}")


def _pid_for_zn_loop(
    base_pid_yaml: Mapping[str, Any],
    *,
    loop_name: str,
    kp_value: float,
    hover_altitude: float,
) -> dict[str, Any]:
    pid_yaml = _with_hover_target(base_pid_yaml, hover_altitude)
    pid_yaml = _deep_update(
        pid_yaml,
        {
            "pid": {
                "outer_loop": {
                    "lateral_x": {"Kp": 0.0, "Kd": 0.0},
                    "lateral_y": {"Kp": 0.0, "Kd": 0.0},
                },
                "inner_loop": {
                    "yaw_Kd": 0.0,
                    "max_yaw_frac": 0.0,
                    "gyro_ff": 0.0,
                },
            }
        },
    )
    if loop_name == "altitude":
        return _pid_cfg_with_gains(
            pid_yaml,
            altitude={
                "Kp": float(kp_value),
                "Ki": 0.0,
                "Kd": 0.0,
                "target_h_agl": float(hover_altitude),
            },
        )
    if loop_name == "roll":
        return _pid_cfg_with_gains(
            pid_yaml,
            roll={"Kp": float(kp_value), "Kd": 0.0},
        )
    if loop_name == "pitch":
        return _pid_cfg_with_gains(
            pid_yaml,
            pitch={"Kp": float(kp_value), "Kd": 0.0},
        )
    raise ValueError(f"Unknown ZN loop {loop_name!r}")


def _apply_scaled_zn_gains(
    pid_yaml: Mapping[str, Any],
    *,
    loop_name: str,
    Ku: float,
    Tu: float,
    gain_scale: float,
    hover_altitude: float,
) -> dict[str, Any]:
    Kp, Ki, Kd = ziegler_nichols_gains(Ku, Tu)
    scale = float(gain_scale)
    if loop_name == "altitude":
        return _pid_cfg_with_gains(
            pid_yaml,
            altitude={
                "Kp": scale * Kp,
                "Ki": scale * Ki,
                "Kd": scale * Kd,
                "target_h_agl": float(hover_altitude),
            },
        )
    if loop_name == "roll":
        return _pid_cfg_with_gains(
            pid_yaml,
            roll={"Kp": scale * Kp, "Kd": scale * Kd},
        )
    if loop_name == "pitch":
        return _pid_cfg_with_gains(
            pid_yaml,
            pitch={"Kp": scale * Kp, "Kd": scale * Kd},
        )
    raise ValueError(f"Unknown ZN loop {loop_name!r}")


def _run_zn_sweep(
    config_path: Path,
    base_pid_yaml: Mapping[str, Any],
    *,
    loop_name: str,
    seed: int,
    hover_altitude: float,
    disable_wind: bool,
    disable_gyro: bool,
    disable_anti_torque: bool,
    disable_gravity: bool,
    headless: bool,
    kp_values: Iterable[float],
    max_seconds: float,
    perturb_angle_deg: float,
    perturb_rate: float,
    altitude_offset_m: float,
) -> tuple[float | None, float | None, list[list[Any]]]:
    _ensure_sim_app(headless=headless)
    from simulation.isaac.envs.edf_isaac_env import EDFIsaacEnv

    env = EDFIsaacEnv(
        config_path=str(config_path),
        seed=seed,
        disable_wind=disable_wind,
        disable_gyro=disable_gyro,
        disable_anti_torque=disable_anti_torque,
        disable_gravity=disable_gravity,
    )
    try:
        _configure_env_for_test(env, test_mode="hover", hover_altitude=hover_altitude)
        perturb = _zn_reset_perturbation(
            loop_name,
            perturb_angle_deg=perturb_angle_deg,
            perturb_rate=perturb_rate,
            altitude_offset_m=altitude_offset_m,
        )
        _set_env_reset_perturbation(
            env,
            altitude_offset_m=float(perturb["altitude_offset_m"]),
            roll_offset_rad=float(perturb["roll_offset_rad"]),
            pitch_offset_rad=float(perturb["pitch_offset_rad"]),
            ang_vel_frd=(
                float(perturb["ang_vel_x"]),
                float(perturb["ang_vel_y"]),
                float(perturb["ang_vel_z"]),
            ),
        )
        dt = float(env._task._dt)
        max_steps = max(1, int(math.ceil(float(max_seconds) / dt)))
        hover_thrust_frac = float(env._task.hover_thrust_norm)
        trial_rows: list[list[Any]] = []
        for idx, kp in enumerate(list(kp_values), start=1):
            pid_yaml = _pid_for_zn_loop(
                base_pid_yaml,
                loop_name=loop_name,
                kp_value=float(kp),
                hover_altitude=hover_altitude,
            )
            obs, _ = env.reset(seed=seed + idx - 1)
            ctrl = PIDController(pid_yaml)
            ctrl.reset()
            sig: list[float] = []
            crashed = False
            truncated = False
            steps = 0
            while steps < max_steps:
                o = np.asarray(obs, dtype=float).reshape(-1)
                sig.append(
                    _signal_from_obs(
                        loop_name,
                        o,
                        hover_altitude=hover_altitude,
                    )
                )
                action_pid = ctrl.get_action(obs)
                action_isaac = map_pid_action_to_isaac(
                    action_pid,
                    hover_thrust_frac=hover_thrust_frac,
                )
                obs, _rew, term, trunc, info = env.step(action_isaac)
                steps += 1
                crashed = bool(np.asarray(info.get("crashed", False)).any())
                truncated = bool(np.asarray(trunc).any())
                if bool(np.asarray(term).any()) or truncated:
                    break
            Tu, amp, zero_crossings = _detect_sustained_oscillation(
                sig,
                loop_name=loop_name,
                dt=dt,
            )
            max_abs_signal, rms_signal, final_abs_signal = _signal_stats(sig)
            detected = Tu is not None and amp is not None and not crashed
            trial_rows.append(
                [
                    "ziegler-nichols",
                    loop_name,
                    float(kp),
                    float(kp) if detected else float("nan"),
                    bool(detected),
                    float(Tu) if Tu is not None else float("nan"),
                    float(amp) if amp is not None else float("nan"),
                    float(max_abs_signal),
                    float(rms_signal),
                    float(final_abs_signal),
                    int(zero_crossings),
                    int(steps),
                    bool(crashed),
                    bool(truncated),
                ]
            )
            if detected:
                print(
                    f"[ZN][isaac] loop={loop_name} Ku={float(kp):.6g} "
                    f"Tu={float(Tu):.3f}s amp={float(amp):.6g} "
                    f"max_abs={max_abs_signal:.6g} rms={rms_signal:.6g} "
                    f"zero_crossings={zero_crossings}",
                    flush=True,
                )
                return float(kp), float(Tu), trial_rows
            print(
                f"[ZN][isaac] loop={loop_name} kp={float(kp):.6g} "
                f"trial={idx} detected={detected} crashed={crashed} "
                f"amp={float(amp) if amp is not None else float('nan'):.6g} "
                f"max_abs={max_abs_signal:.6g} rms={rms_signal:.6g} "
                f"final_abs={final_abs_signal:.6g} zero_crossings={zero_crossings}",
                flush=True,
            )
        return None, None, trial_rows
    finally:
        env.close()


def _run_relay_autotune(
    config_path: Path,
    *,
    output_dir: Path,
    loop_name: str,
    seed: int,
    hover_altitude: float,
    disable_wind: bool,
    disable_gyro: bool,
    disable_anti_torque: bool,
    disable_gravity: bool,
    headless: bool,
    max_seconds: float,
    perturb_angle_deg: float,
    perturb_rate: float,
    relay_amplitude: float,
    relay_hysteresis_deg: float,
    relay_invert: bool,
    relay_trace_steps: int,
) -> tuple[float | None, float | None, list[list[Any]]]:
    _ensure_sim_app(headless=headless)
    from simulation.isaac.envs.edf_isaac_env import EDFIsaacEnv

    if loop_name not in ("roll", "pitch"):
        raise ValueError(
            f"relay-autotune only supports roll/pitch loops, got {loop_name!r}"
        )

    env = EDFIsaacEnv(
        config_path=str(config_path),
        seed=seed,
        disable_wind=disable_wind,
        disable_gyro=disable_gyro,
        disable_anti_torque=disable_anti_torque,
        disable_gravity=disable_gravity,
    )
    try:
        _configure_env_for_test(env, test_mode="hover", hover_altitude=hover_altitude)
        perturb = _zn_reset_perturbation(
            loop_name,
            perturb_angle_deg=perturb_angle_deg,
            perturb_rate=perturb_rate,
            altitude_offset_m=0.0,
        )
        _set_env_reset_perturbation(
            env,
            altitude_offset_m=0.0,
            roll_offset_rad=float(perturb["roll_offset_rad"]),
            pitch_offset_rad=float(perturb["pitch_offset_rad"]),
            ang_vel_frd=(
                float(perturb["ang_vel_x"]),
                float(perturb["ang_vel_y"]),
                float(perturb["ang_vel_z"]),
            ),
        )
        obs, _ = env.reset(seed=seed)
        dt = float(env._task._dt)
        max_steps = max(1, int(math.ceil(float(max_seconds) / dt)))
        hover_thrust_frac = float(env._task.hover_thrust_norm)
        hyst = math.radians(float(relay_hysteresis_deg))
        relay_state = float(relay_amplitude)
        sig: list[float] = []
        trace_rows: list[list[float]] = []
        last_info: dict[str, Any] = {}
        flip_count = 0
        prev_relay_state = float(relay_state)
        max_abs_roll_deg = 0.0
        max_abs_pitch_deg = 0.0
        max_lateral_dist = 0.0
        crashed = False
        truncated = False
        steps = 0
        while steps < max_steps:
            o = np.asarray(obs, dtype=float).reshape(-1)
            err = _signal_from_obs(loop_name, o, hover_altitude=hover_altitude)
            sig.append(err)
            roll_est, pitch_est = _roll_pitch_from_obs(o)
            max_abs_roll_deg = max(max_abs_roll_deg, abs(math.degrees(roll_est)))
            max_abs_pitch_deg = max(max_abs_pitch_deg, abs(math.degrees(pitch_est)))
            if err > hyst:
                relay_state = -abs(float(relay_amplitude))
            elif err < -hyst:
                relay_state = abs(float(relay_amplitude))
            if relay_state != prev_relay_state:
                flip_count += 1
            prev_relay_state = float(relay_state)
            if steps < int(relay_trace_steps):
                trace_rows.append([float(steps), float(err), float(relay_state)])
            action = _relay_action(
                loop_name,
                relay_state,
                hover_thrust_frac=hover_thrust_frac,
                invert=bool(relay_invert),
            )
            obs, _rew, term, trunc, info = env.step(action)
            last_info = dict(info)
            max_lateral_dist = max(
                max_lateral_dist,
                float(np.asarray(info.get("lateral_dist", 0.0), dtype=float).reshape(-1)[0]),
            )
            steps += 1
            crashed = bool(np.asarray(info.get("crashed", False)).any())
            truncated = bool(np.asarray(trunc).any())
            if bool(np.asarray(term).any()) or truncated:
                break

        Tu, amp, zero_crossings = _detect_sustained_oscillation(
            sig,
            loop_name=loop_name,
            dt=dt,
        )
        max_abs_signal, rms_signal, final_abs_signal = _signal_stats(sig)
        final_obs = np.asarray(obs, dtype=float).reshape(-1)
        final_state = _format_obs_state(final_obs)
        run_stats = _format_relay_run_stats(
            steps=steps,
            flip_count=flip_count,
            signal=sig,
            max_abs_roll_deg=max_abs_roll_deg,
            max_abs_pitch_deg=max_abs_pitch_deg,
            max_lateral_dist=max_lateral_dist,
            last_info=last_info,
        )
        Ku = (
            _estimate_ku_from_relay(
                relay_amplitude=float(relay_amplitude),
                oscillation_amplitude=float(amp),
            )
            if amp is not None and not crashed
            else None
        )
        detected = Ku is not None and Tu is not None
        trace_path = _write_relay_trace(
            trace_rows,
            output_dir=output_dir,
            loop_name=loop_name,
            inverted=bool(relay_invert),
        )
        trial_rows = [
            [
                "relay-autotune",
                loop_name,
                float(relay_amplitude),
                float(Ku) if Ku is not None else float("nan"),
                bool(detected),
                float(Tu) if Tu is not None else float("nan"),
                float(amp) if amp is not None else float("nan"),
                float(max_abs_signal),
                float(rms_signal),
                float(final_abs_signal),
                int(zero_crossings),
                int(steps),
                bool(crashed),
                bool(truncated),
            ]
        ]
        if detected:
            print(
                f"[relay][isaac] loop={loop_name} Ku={float(Ku):.6g} "
                f"Tu={float(Tu):.3f}s amp={float(amp):.6g} "
                f"max_abs={max_abs_signal:.6g} rms={rms_signal:.6g} "
                f"zero_crossings={zero_crossings} invert={bool(relay_invert)} "
                f"trace={trace_path}"
                f"{run_stats}"
                f"{final_state}",
                flush=True,
            )
            return float(Ku), float(Tu), trial_rows
        print(
            f"[relay][isaac] loop={loop_name} amp={float(amp) if amp is not None else float('nan'):.6g} "
            f"max_abs={max_abs_signal:.6g} rms={rms_signal:.6g} "
            f"final_abs={final_abs_signal:.6g} zero_crossings={zero_crossings} "
            f"crashed={crashed} invert={bool(relay_invert)} trace={trace_path}"
            f"{run_stats}"
            f"{final_state}",
            flush=True,
        )
        return None, None, trial_rows
    finally:
        env.close()


def _write_summary_csv(rows: list[list[Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_SUMMARY_COLUMNS)
        writer.writerows(rows)


def _run_coordinate_search(
    *,
    args: argparse.Namespace,
    config_path: Path,
    pid_path: Path,
    output_dir: Path,
    log_dir: Path,
    base_pid_yaml: Mapping[str, Any],
) -> None:
    scale_grid = (0.6, 0.8, 1.0, 1.2, 1.4)
    search_axes = (
        ("altitude", "altitude"),
        ("lateral_x", "lateral_x"),
        ("lateral_y", "lateral_y"),
        ("roll", "roll"),
        ("pitch", "pitch"),
        ("yaw_kd", "yaw_kd"),
    )
    print(
        f"[tune_pid_isaac] config={config_path} pid_config={pid_path} "
        f"episodes_per_candidate={args.episodes} seed={args.seed} test={args.test}"
    )
    if args.test == "hover":
        print(
            f"[tune_pid_isaac] hover_target: altitude={args.hover_altitude:.2f}m "
            f"alt_tol={args.hover_alt_tol:.2f}m"
        )
    print(
        f"[tune_pid_isaac] force_toggles: "
        f"disable_wind={args.disable_wind} "
        f"disable_gyro={args.disable_gyro} "
        f"disable_anti_torque={args.disable_anti_torque} "
        f"disable_gravity={args.disable_gravity} "
        f"disable_yaw={args.disable_yaw} "
        f"disable_pitch={args.disable_pitch} "
        f"disable_lateral_x={args.disable_lateral_x} "
        f"disable_lateral_y={args.disable_lateral_y} "
        f"disable_roll={args.disable_roll} "
        f"invert_lateral_x={args.invert_lateral_x} "
        f"invert_pitch_damping={args.invert_pitch_damping} "
        f"invert_altitude_damping={args.invert_altitude_damping}"
    )
    print(
        f"[tune_pid_isaac] output_dir={output_dir} log_dir={log_dir} "
        f"search=coordinate scales={scale_grid}"
    )

    rows: list[list[Any]] = []
    current_label = "baseline"
    current_cfg = copy.deepcopy(base_pid_yaml)
    current_stats = _evaluate_single_env(
        config_path=config_path,
        pid_yaml=current_cfg,
        episodes=int(args.episodes),
        seed=int(args.seed),
        test_mode=str(args.test),
        hover_altitude=float(args.hover_altitude),
        hover_alt_tolerance=float(args.hover_alt_tol),
        disable_wind=bool(args.disable_wind),
        disable_gyro=bool(args.disable_gyro),
        disable_anti_torque=bool(args.disable_anti_torque),
        disable_gravity=bool(args.disable_gravity),
        headless=bool(args.headless),
        debug_print=True,
        episode_log_path=log_dir / f"episodes_{current_label}.csv",
    )
    rows.append(
        [
            current_label,
            current_stats["success_rate"],
            current_stats["crash_rate"],
            current_stats["trunc_rate"],
            current_stats["mean_reward"],
            current_stats["mean_steps"],
            current_stats["mean_lateral_dist_success"],
            current_stats["mean_impact_speed_success"],
            current_stats["mean_abs_alt_error"],
            current_stats["mean_speed"],
            current_stats["mean_ang_speed"],
            current_stats["mean_lateral"],
            current_stats["mean_max_abs_roll_deg"],
            current_stats["mean_max_abs_pitch_deg"],
            current_stats["mean_max_abs_omega_xy"],
        ]
    )
    for axis_name, axis_key in search_axes:
        axis_best_label = current_label
        axis_best_cfg = current_cfg
        axis_best_stats = current_stats
        for scale in scale_grid:
            if math.isclose(scale, 1.0):
                continue
            label = f"{axis_name}_x{scale:.2f}"
            cfg = _scaled_pid_candidate(current_cfg, **{axis_key: float(scale)})
            stats = _evaluate_single_env(
                config_path=config_path,
                pid_yaml=cfg,
                episodes=int(args.episodes),
                seed=int(args.seed),
                test_mode=str(args.test),
                hover_altitude=float(args.hover_altitude),
                hover_alt_tolerance=float(args.hover_alt_tol),
                disable_wind=bool(args.disable_wind),
                disable_gyro=bool(args.disable_gyro),
                disable_anti_torque=bool(args.disable_anti_torque),
                disable_gravity=bool(args.disable_gravity),
                headless=bool(args.headless),
                debug_print=False,
                episode_log_path=log_dir / f"episodes_{label}.csv",
            )
            rows.append(
                [
                    label,
                    stats["success_rate"],
                    stats["crash_rate"],
                    stats["trunc_rate"],
                    stats["mean_reward"],
                    stats["mean_steps"],
                    stats["mean_lateral_dist_success"],
                    stats["mean_impact_speed_success"],
                    stats["mean_abs_alt_error"],
                    stats["mean_speed"],
                    stats["mean_ang_speed"],
                    stats["mean_lateral"],
                    stats["mean_max_abs_roll_deg"],
                    stats["mean_max_abs_pitch_deg"],
                    stats["mean_max_abs_omega_xy"],
                ]
            )
            print(
                f"[tune_pid_isaac] candidate={label} "
                f"success_rate={stats['success_rate']:.3f} "
                f"crash_rate={stats['crash_rate']:.3f} "
                f"trunc_rate={stats['trunc_rate']:.3f} "
                f"mean_reward={stats['mean_reward']:.3f} "
                f"mean_abs_alt_error={stats['mean_abs_alt_error']:.3f} "
                f"mean_max_abs_roll_deg={stats['mean_max_abs_roll_deg']:.3f} "
                f"mean_max_abs_omega_xy={stats['mean_max_abs_omega_xy']:.3f}",
                flush=True,
            )
            if _candidate_rank_key(stats, test_mode=str(args.test)) > _candidate_rank_key(
                axis_best_stats,
                test_mode=str(args.test),
            ):
                axis_best_label = label
                axis_best_cfg = cfg
                axis_best_stats = stats

        current_label = axis_best_label
        current_cfg = axis_best_cfg
        current_stats = axis_best_stats

    scores_path = output_dir / "candidate_scores.csv"
    _write_summary_csv(rows, scores_path)

    best_pid_path = output_dir / "best_pid.yaml"
    with best_pid_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(current_cfg, f, sort_keys=False)

    print(
        f"[tune_pid_isaac] BEST candidate={current_label} "
        f"success_rate={current_stats['success_rate']:.3f} "
        f"crash_rate={current_stats['crash_rate']:.3f} "
        f"trunc_rate={current_stats['trunc_rate']:.3f} "
        f"mean_reward={current_stats['mean_reward']:.3f} "
        f"mean_abs_alt_error={current_stats['mean_abs_alt_error']:.3f} "
        f"mean_max_abs_roll_deg={current_stats['mean_max_abs_roll_deg']:.3f} "
        f"mean_max_abs_omega_xy={current_stats['mean_max_abs_omega_xy']:.3f} "
        f"saved_pid={best_pid_path} scores={scores_path} log_dir={log_dir}",
        flush=True,
    )


# Rotation test: zero-g, stationary, command omega on each axis in both directions; pass if PID tracks.
_ROTATION_AXES = ("roll", "pitch", "yaw")  # body omega_x, omega_y, omega_z
_ROTATION_SPEEDS_RAD_S = (2.0, -2.0)  # + and - per axis (roll+, roll-, pitch+, pitch-, yaw+, yaw-)
_ROTATION_DURATION_S = 5.0
_ROTATION_MAX_DURATION_S = 10.0  # cap each omega command run at 10 s (t in logs won't exceed this)
_ROTATION_STEP_HZ = 40
_ROTATION_SETTLE_STEPS = 10
_ROTATION_TOL_ABS_RAD_S = 0.15
_ROTATION_TOL_FRAC = 0.25  # allow up to 25% of |cmd| as mean abs error
_ROTATION_LOG_INTERVAL_STEPS = 40  # log every 1 s at 40 Hz
# Fin order matches task/parts_registry: RightFin, LeftFin, FwdFin, AftFin → action[1:5]
_ROTATION_FIN_LABELS = ("RightFin", "LeftFin", "FwdFin", "AftFin")
# When isolating: zero these 0-based fin indices so only the tested axis has fin authority.
# Roll=FwdFin+AftFin (2,3), Pitch=RightFin+LeftFin (0,1), Yaw=all four.
_ROTATION_FIN_INDICES_TO_ZERO = {"roll": (0, 1), "pitch": (2, 3), "yaw": ()}


def _disable_gravity_rotation(env) -> None:
    """Set physics scene gravity to 0 so rotation test is zero-g."""
    try:
        from pxr import UsdPhysics
        stage = env._task.sim.stage
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                UsdPhysics.Scene(prim).GetGravityMagnitudeAttr().Set(0.0)
                print("[rotation] Gravity disabled (physics scene).", flush=True)
                return
        print("[rotation] WARNING: Physics scene not found; gravity still active.", flush=True)
    except Exception as exc:
        print(f"[rotation] WARNING: Could not disable gravity: {exc}", flush=True)


def _lock_drone_position_xyz(env, x: float, y: float, z: float) -> None:
    """Lock drone root position at (x,y,z), zero linear velocity; leave orientation and angular velocity unchanged."""
    import torch
    task = getattr(env, "_task", None)
    if task is None or not hasattr(task, "robot"):
        return
    robot = task.robot
    pos_w = robot.data.root_pos_w.clone()
    quat_w = robot.data.root_quat_w.clone()
    ang_w = robot.data.root_ang_vel_w.clone()
    pos_w[:, 0] = x
    pos_w[:, 1] = y
    pos_w[:, 2] = z
    robot.write_root_pose_to_sim(torch.cat([pos_w, quat_w], dim=-1))
    vel_w = torch.zeros_like(robot.data.root_lin_vel_w)
    robot.write_root_velocity_to_sim(torch.cat([vel_w, ang_w], dim=-1))


def _log_rotation_reset_pose(env, axis_name: str, speed: float) -> None:
    """Log reset pose (position + orientation) after lock so we can verify one reset per (axis, speed)."""
    task = getattr(env, "_task", None)
    if task is None or not hasattr(task, "robot"):
        return
    robot = task.robot
    pos = robot.data.root_pos_w[0].cpu().numpy()
    quat = robot.data.root_quat_w[0].cpu().numpy()  # wxyz from IsaacLab
    rpy_deg = _quat_wxyz_to_rpy_deg(quat)
    print(
        f"[rotation] reset pose  axis={axis_name} speed={speed:.2f}  "
        f"pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})  "
        f"quat_wxyz=({quat[0]:.4f},{quat[1]:.4f},{quat[2]:.4f},{quat[3]:.4f})  "
        f"rpy_deg=({rpy_deg[0]:.2f},{rpy_deg[1]:.2f},{rpy_deg[2]:.2f})",
        flush=True,
    )


def _quat_wxyz_to_rpy_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to roll, pitch, yaw in degrees."""
    w, x, y, z = quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([math.degrees(roll), math.degrees(pitch), math.degrees(yaw)])


def _rotation_pid_cfg_for_axis(
    base_pid_yaml: Mapping[str, Any],
    *,
    hover_altitude: float,
    axis_name: str,
    isolate_pid_axis: bool,
    zero_yaw_when_isolated: bool,
    disable_noncommanded_angle_loops: bool,
) -> dict[str, Any]:
    """Build a per-axis PID config for rotation debugging."""
    root = _with_hover_target(base_pid_yaml, hover_altitude)
    pid = dict(root.get("pid", root))
    inner = dict(pid.get("inner_loop", {}))

    if (
        isolate_pid_axis
        and axis_name in ("roll", "pitch")
        and bool(zero_yaw_when_isolated)
    ):
        inner["yaw_Kd"] = 0.0
        inner["max_yaw_frac"] = 0.0

    if bool(disable_noncommanded_angle_loops):
        roll_cfg = dict(inner.get("roll", {}))
        pitch_cfg = dict(inner.get("pitch", {}))
        if axis_name == "roll":
            pitch_cfg["Kp"] = 0.0
            pitch_cfg["Kd"] = 0.0
        elif axis_name == "pitch":
            roll_cfg["Kp"] = 0.0
            roll_cfg["Kd"] = 0.0
        elif axis_name == "yaw":
            roll_cfg["Kp"] = 0.0
            roll_cfg["Kd"] = 0.0
            pitch_cfg["Kp"] = 0.0
            pitch_cfg["Kd"] = 0.0
        inner["roll"] = roll_cfg
        inner["pitch"] = pitch_cfg

    pid["inner_loop"] = inner
    root["pid"] = pid
    return root


def _run_rotation_test(
    *,
    config_path: Path,
    base_pid_yaml: Mapping[str, Any],
    args: argparse.Namespace,
) -> bool:
    """Run zero-g rate-tracking test: each axis at 3 speeds; pass iff all track within tolerance."""
    _ensure_sim_app(headless=bool(args.headless))
    from simulation.isaac.envs.edf_isaac_env import EDFIsaacEnv

    env = EDFIsaacEnv(
        config_path=str(config_path),
        seed=int(args.seed),
        disable_wind=True,
        disable_gyro=bool(args.disable_gyro),
        disable_anti_torque=bool(args.disable_anti_torque),
        disable_gravity=True,
    )
    alt = float(args.hover_altitude)
    _configure_env_for_test(env, test_mode="rotation", hover_altitude=alt)
    _set_env_reset_perturbation(env, altitude_offset_m=0.0, roll_offset_rad=0.0, pitch_offset_rad=0.0)

    if env.num_envs != 1:
        print(
            "[tune_pid_isaac][rotation] requires single env; reconfigure and rerun.",
            flush=True,
        )
        env.close()
        return False

    _disable_gravity_rotation(env)
    # Lock position at (0, 0, alt) so test is purely rotational; max thrust so no thrust modulation.
    lock_x, lock_y, lock_z = 0.0, 0.0, alt

    isolate = getattr(args, "isolate_pid_axis", True)
    print(f"[rotation] isolate_pid_axis={isolate} (only tested axis fins active)", flush=True)
    print(
        f"[rotation] zero_yaw_when_isolated={bool(getattr(args, 'rotation_zero_yaw_when_isolated', True))} "
        f"disable_noncommanded_angle_loops={bool(getattr(args, 'rotation_disable_noncommanded_angle_loops', False))}",
        flush=True,
    )

    episodes = max(1, int(args.episodes))
    hover_thrust_frac = float(env._task.hover_thrust_norm)
    steps_per_trial = max(
        int(round(_ROTATION_DURATION_S * _ROTATION_STEP_HZ)), 1
    )
    max_steps_cap = int(_ROTATION_MAX_DURATION_S * _ROTATION_STEP_HZ)
    total_steps_per_run = min(episodes * steps_per_trial, max_steps_cap)
    # Prevent episode truncation mid-run: task default is 5 s (600 steps @ 120 Hz).
    if env._task._max_steps < total_steps_per_run:
        env._task._max_steps = total_steps_per_run
    axis_idx = {"roll": 0, "pitch": 1, "yaw": 2}
    all_passed = True

    try:
        for axis_name in _ROTATION_AXES:
            idx = axis_idx[axis_name]
            pid_yaml_axis = _rotation_pid_cfg_for_axis(
                base_pid_yaml,
                hover_altitude=alt,
                axis_name=axis_name,
                isolate_pid_axis=bool(getattr(args, "isolate_pid_axis", True)),
                zero_yaw_when_isolated=bool(
                    getattr(args, "rotation_zero_yaw_when_isolated", True)
                ),
                disable_noncommanded_angle_loops=bool(
                    getattr(args, "rotation_disable_noncommanded_angle_loops", False)
                ),
            )
            ctrl = PIDController(pid_yaml_axis)
            for speed in _ROTATION_SPEEDS_RAD_S:
                omega_cmd = [0.0, 0.0, 0.0]
                omega_cmd[idx] = speed
                ctrl.clear_omega_cmd()
                ctrl.reset()
                ctrl.set_omega_cmd(
                    roll_rad_s=omega_cmd[0] if omega_cmd[0] != 0 else None,
                    pitch_rad_s=omega_cmd[1] if omega_cmd[1] != 0 else None,
                    yaw_rad_s=omega_cmd[2] if omega_cmd[2] != 0 else None,
                )
                # One reset per (axis, speed), then one continuous run (capped at _ROTATION_MAX_DURATION_S).
                total_steps = total_steps_per_run
                obs, _ = env.reset(seed=int(args.seed))
                _lock_drone_position_xyz(env, lock_x, lock_y, lock_z)
                # Log reset orientation once so we can verify it
                _log_rotation_reset_pose(env, axis_name, speed)
                print(
                    f"\n[rotation] -------- axis={axis_name} speed={speed:.2f} rad/s (1 run, {total_steps} steps = {total_steps / _ROTATION_STEP_HZ:.1f} s) --------",
                    flush=True,
                )
                print(
                    "  (labeled: target_omega/omega_err/omega_act in body FRD rad/s; "
                    "g_body is observed gravity in body frame; pid_fins are pre-isolation, "
                    "applied_fins are after axis isolation; rpy_deg is world-frame Euler for visualization only)",
                    flush=True,
                )
                errors: list[float] = []
                for step in range(total_steps):
                    action_pid, dbg = ctrl.get_action_with_debug(obs)
                    action_isaac = map_pid_action_to_isaac(
                        action_pid, hover_thrust_frac=hover_thrust_frac
                    )
                    action_pre_isolation = action_isaac.copy()
                    action_isaac[0] = 1.0  # max thrust; position locked so purely rotational
                    if getattr(args, "isolate_pid_axis", True):
                        for fi in _ROTATION_FIN_INDICES_TO_ZERO.get(axis_name, ()):
                            action_isaac[1 + fi] = 0.0
                    obs, _rew, term, trunc, _info = env.step(
                        np.asarray(action_isaac, dtype=np.float32)
                    )
                    _lock_drone_position_xyz(env, lock_x, lock_y, lock_z)
                    if step >= _ROTATION_SETTLE_STEPS and not (term or trunc):
                        o = np.asarray(obs, dtype=float).reshape(-1)
                        if o.size >= 12:
                            errors.append(
                                float(np.abs(o[9 + idx] - omega_cmd[idx]))
                            )
                            if (step - _ROTATION_SETTLE_STEPS) % _ROTATION_LOG_INTERVAL_STEPS == 0:
                                t_s = (step + 1) * (1.0 / _ROTATION_STEP_HZ)
                                quat = env._task.robot.data.root_quat_w[0].cpu().numpy()
                                rpy_deg = _quat_wxyz_to_rpy_deg(quat)
                                omega_err = np.asarray(omega_cmd, dtype=float) - o[9:12]
                                print(
                                    f"  t={t_s:.2f} s",
                                    flush=True,
                                )
                                print(
                                    f"    target_omega -> (x={omega_cmd[0]:+.2f}, y={omega_cmd[1]:+.2f}, z={omega_cmd[2]:+.2f})",
                                    flush=True,
                                )
                                print(
                                    f"    omega_err    -> (x={omega_err[0]:+.3f}, y={omega_err[1]:+.3f}, z={omega_err[2]:+.3f})",
                                    flush=True,
                                )
                                print(
                                    f"    g_body       -> (x={o[6]:+.3f}, y={o[7]:+.3f}, z={o[8]:+.3f})",
                                    flush=True,
                                )
                                pid_fin_parts = " ".join(
                                    f"{short}={action_pre_isolation[1 + i]:+.3f}"
                                    for i, short in enumerate(("R", "L", "Fwd", "Aft"))
                                )
                                applied_fin_parts = " ".join(
                                    f"{short}={action_isaac[1 + i]:+.3f}"
                                    for i, short in enumerate(("R", "L", "Fwd", "Aft"))
                                )
                                print(f"    pid_fins     -> ({pid_fin_parts})", flush=True)
                                print(f"    applied_fins -> ({applied_fin_parts})", flush=True)
                                print(
                                    f"    omega_act    -> (x={o[9]:+.3f}, y={o[10]:+.3f}, z={o[11]:+.3f})",
                                    flush=True,
                                )
                                print(
                                    f"    pid_terms    -> (roll_cmd={dbg['roll_cmd']:+.3f}, "
                                    f"pitch_cmd={dbg['pitch_cmd']:+.3f}, "
                                    f"yaw_total={dbg['yaw_total']:+.3f}, "
                                    f"omega_z_filt={dbg['omega_z_filt']:+.3f})",
                                    flush=True,
                                )
                                print(
                                    f"    pid_rate_err -> (x={dbg.get('omega_error_x', float('nan')):+.3f}, "
                                    f"y={dbg.get('omega_error_y', float('nan')):+.3f}, "
                                    f"z={dbg.get('omega_error_z', float('nan')):+.3f})",
                                    flush=True,
                                )
                                print(
                                    f"    rpy_deg      -> (roll={rpy_deg[0]:+.2f}, pitch={rpy_deg[1]:+.2f}, yaw={rpy_deg[2]:+.2f})",
                                    flush=True,
                                )
                ctrl.clear_omega_cmd()
                mean_err = float(np.mean(errors)) if errors else float("inf")
                tol = max(_ROTATION_TOL_ABS_RAD_S, _ROTATION_TOL_FRAC * abs(speed))
                passed = mean_err <= tol
                all_passed = all_passed and passed
                print(
                    f"[rotation] axis={axis_name} speed={speed:.2f} rad/s "
                    f"duration={total_steps / _ROTATION_STEP_HZ:.1f}s mean_|error|={mean_err:.3f} tol={tol:.3f} {'PASS' if passed else 'FAIL'}",
                    flush=True,
                )
    finally:
        env.close()

    print(
        f"[rotation] overall {'PASS' if all_passed else 'FAIL'}",
        flush=True,
    )
    return all_passed


def _run_zn_mode(
    *,
    args: argparse.Namespace,
    config_path: Path,
    output_dir: Path,
    log_dir: Path,
    base_pid_yaml: Mapping[str, Any],
) -> None:
    loops = list(_LOOP_ORDER) if str(args.zn_loop) == "all" else [str(args.zn_loop)]
    kp_values = np.geomspace(
        float(args.zn_kp_start),
        float(args.zn_kp_stop),
        num=max(int(args.zn_kp_steps), 2),
    )
    verify_episodes = (
        int(args.zn_verify_episodes)
        if int(args.zn_verify_episodes) > 0
        else int(args.episodes)
    )
    verification_limits = {
        "max_tilt_deg": float(args.zn_verify_max_tilt_deg),
        "max_omega_xy": float(args.zn_verify_max_omega_xy),
    }

    working_pid = _deep_update(
        _with_hover_target(base_pid_yaml, float(args.hover_altitude)),
        {
            "pid": {
                "outer_loop": {
                    "lateral_x": {"Kp": 0.0, "Kd": 0.0},
                    "lateral_y": {"Kp": 0.0, "Kd": 0.0},
                },
                "inner_loop": {
                    "yaw_Kd": 0.0,
                    "max_yaw_frac": 0.0,
                },
            }
        },
    )
    trial_rows: list[list[Any]] = []
    for loop_name in loops:
        if str(args.method) == "relay-autotune" and loop_name in ("roll", "pitch"):
            Ku, Tu, loop_rows = _run_relay_autotune(
                config_path=config_path,
                output_dir=output_dir,
                loop_name=loop_name,
                seed=int(args.seed) + len(trial_rows),
                hover_altitude=float(args.hover_altitude),
                disable_wind=bool(args.disable_wind),
                disable_gyro=bool(args.disable_gyro),
                disable_anti_torque=bool(args.disable_anti_torque),
                disable_gravity=bool(args.disable_gravity),
                headless=bool(args.headless),
                max_seconds=float(args.zn_max_seconds),
                perturb_angle_deg=float(args.zn_perturb_angle_deg),
                perturb_rate=float(args.zn_perturb_rate),
                relay_amplitude=float(args.relay_amplitude),
                relay_hysteresis_deg=float(args.relay_hysteresis_deg),
                relay_invert=bool(args.relay_invert),
                relay_trace_steps=int(args.relay_trace_steps),
            )
        else:
            if str(args.method) == "relay-autotune" and loop_name == "altitude":
                print(
                    "[relay][isaac] altitude currently falls back to ZN sweep.",
                    flush=True,
                )
            Ku, Tu, loop_rows = _run_zn_sweep(
                config_path=config_path,
                base_pid_yaml=working_pid,
                loop_name=loop_name,
                seed=int(args.seed) + len(trial_rows),
                hover_altitude=float(args.hover_altitude),
                disable_wind=bool(args.disable_wind),
                disable_gyro=bool(args.disable_gyro),
                disable_anti_torque=bool(args.disable_anti_torque),
                disable_gravity=bool(args.disable_gravity),
                headless=bool(args.headless),
                kp_values=kp_values,
                max_seconds=float(args.zn_max_seconds),
                perturb_angle_deg=float(args.zn_perturb_angle_deg),
                perturb_rate=float(args.zn_perturb_rate),
                altitude_offset_m=float(args.zn_altitude_offset),
            )
        trial_rows.extend(loop_rows)
        if Ku is None or Tu is None:
            print(
                f"[{str(args.method)}][isaac] loop={loop_name} failed to find sustained oscillation; "
                "keeping current gains for this loop.",
                flush=True,
            )
            continue
        working_pid = _apply_scaled_zn_gains(
            working_pid,
            loop_name=loop_name,
            Ku=float(Ku),
            Tu=float(Tu),
            gain_scale=float(args.zn_gain_scale),
            hover_altitude=float(args.hover_altitude),
        )

    zn_trials_path = output_dir / "zn_trials.csv"
    zn_trials_path.parent.mkdir(parents=True, exist_ok=True)
    with zn_trials_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_ZN_TRIAL_COLUMNS)
        writer.writerows(trial_rows)

    zn_result_path = output_dir / "zn_result.yaml"
    with zn_result_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(working_pid, f, sort_keys=False)

    baseline_stats = _evaluate_single_env(
        config_path=config_path,
        pid_yaml=_deep_update(
            _with_hover_target(base_pid_yaml, float(args.hover_altitude)),
            {
                "pid": {
                    "outer_loop": {
                        "lateral_x": {"Kp": 0.0, "Kd": 0.0},
                        "lateral_y": {"Kp": 0.0, "Kd": 0.0},
                    },
                    "inner_loop": {"yaw_Kd": 0.0, "max_yaw_frac": 0.0},
                }
            },
        ),
        episodes=verify_episodes,
        seed=int(args.seed),
        test_mode="hover",
        hover_altitude=float(args.hover_altitude),
        hover_alt_tolerance=float(args.hover_alt_tol),
        disable_wind=bool(args.disable_wind),
        disable_gyro=bool(args.disable_gyro),
        disable_anti_torque=bool(args.disable_anti_torque),
        disable_gravity=bool(args.disable_gravity),
        headless=bool(args.headless),
        debug_print=False,
        episode_log_path=log_dir / "episodes_zn_baseline.csv",
        verification_limits=verification_limits,
    )
    tuned_stats = _evaluate_single_env(
        config_path=config_path,
        pid_yaml=working_pid,
        episodes=verify_episodes,
        seed=int(args.seed) + 1000,
        test_mode="hover",
        hover_altitude=float(args.hover_altitude),
        hover_alt_tolerance=float(args.hover_alt_tol),
        disable_wind=bool(args.disable_wind),
        disable_gyro=bool(args.disable_gyro),
        disable_anti_torque=bool(args.disable_anti_torque),
        disable_gravity=bool(args.disable_gravity),
        headless=bool(args.headless),
        debug_print=False,
        episode_log_path=log_dir / "episodes_zn_tuned.csv",
        verification_limits=verification_limits,
    )
    verification_rows = [
        [
            "baseline_isolated",
            baseline_stats["success_rate"],
            baseline_stats["crash_rate"],
            baseline_stats["trunc_rate"],
            baseline_stats["mean_reward"],
            baseline_stats["mean_steps"],
            baseline_stats["mean_lateral_dist_success"],
            baseline_stats["mean_impact_speed_success"],
            baseline_stats["mean_abs_alt_error"],
            baseline_stats["mean_speed"],
            baseline_stats["mean_ang_speed"],
            baseline_stats["mean_lateral"],
            baseline_stats["mean_max_abs_roll_deg"],
            baseline_stats["mean_max_abs_pitch_deg"],
            baseline_stats["mean_max_abs_omega_xy"],
        ],
        [
            "zn_tuned",
            tuned_stats["success_rate"],
            tuned_stats["crash_rate"],
            tuned_stats["trunc_rate"],
            tuned_stats["mean_reward"],
            tuned_stats["mean_steps"],
            tuned_stats["mean_lateral_dist_success"],
            tuned_stats["mean_impact_speed_success"],
            tuned_stats["mean_abs_alt_error"],
            tuned_stats["mean_speed"],
            tuned_stats["mean_ang_speed"],
            tuned_stats["mean_lateral"],
            tuned_stats["mean_max_abs_roll_deg"],
            tuned_stats["mean_max_abs_pitch_deg"],
            tuned_stats["mean_max_abs_omega_xy"],
        ],
    ]
    verification_path = output_dir / "zn_verification.csv"
    _write_summary_csv(verification_rows, verification_path)

    print(
        f"[ZN][isaac] loops={','.join(loops)} "
        f"gain_scale={float(args.zn_gain_scale):.3f} "
        f"result_yaml={zn_result_path} trials={zn_trials_path} "
        f"verification={verification_path}",
        flush=True,
    )
    print(
        f"[ZN][isaac] tuned success_rate={tuned_stats['success_rate']:.3f} "
        f"mean_abs_alt_error={tuned_stats['mean_abs_alt_error']:.3f} "
        f"mean_max_abs_roll_deg={tuned_stats['mean_max_abs_roll_deg']:.3f} "
        f"mean_max_abs_pitch_deg={tuned_stats['mean_max_abs_pitch_deg']:.3f} "
        f"mean_max_abs_omega_xy={tuned_stats['mean_max_abs_omega_xy']:.3f}",
        flush=True,
    )


def main() -> None:
    global _SIM_APP
    args = _parse_args()
    config_path = Path(args.config)
    pid_path = Path(args.pid_config)
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir) if args.log_dir is not None else output_dir / "logs"

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    base_pid_yaml = _load_pid_yaml(pid_path)
    if str(args.test) == "hover":
        base_pid_yaml = _with_hover_target(base_pid_yaml, float(args.hover_altitude))
    base_pid_yaml = _apply_runtime_pid_overrides(base_pid_yaml, args=args)

    if str(args.test) == "rotation":
        passed = _run_rotation_test(
            config_path=config_path,
            base_pid_yaml=base_pid_yaml,
            args=args,
        )
        if _SIM_APP is not None:
            _SIM_APP.close()
            _SIM_APP = None
        raise SystemExit(0 if passed else 1)

    if str(args.zn_loop) != "none":
        _run_zn_mode(
            args=args,
            config_path=config_path,
            output_dir=output_dir,
            log_dir=log_dir,
            base_pid_yaml=base_pid_yaml,
        )
    else:
        _run_coordinate_search(
            args=args,
            config_path=config_path,
            pid_path=pid_path,
            output_dir=output_dir,
            log_dir=log_dir,
            base_pid_yaml=base_pid_yaml,
        )

    if _SIM_APP is not None:
        _SIM_APP.close()
        _SIM_APP = None


if __name__ == "__main__":
    main()

