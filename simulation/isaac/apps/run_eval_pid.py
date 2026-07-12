"""
PID hover evaluation app.

Runs the PID controller in a single environment for a configurable duration and
reports hover metrics. Optional step logging prints full state vectors.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from runner_safety import WallClockWatchdog, force_process_exit


def parse_args():
    parser = argparse.ArgumentParser(description="PID hover evaluation")
    parser.add_argument("--task", default="hover", choices=["hover", "landing"])
    parser.add_argument("--env-config", default="configs/env/single_env_debug.yaml")
    parser.add_argument("--disturbance", default=None, help="Path to disturbance config YAML")
    parser.add_argument("--duration", type=float, default=30.0, help="Evaluation duration in seconds")
    parser.add_argument("--seed", type=int, default=None, help="Seed for repeatable spawn sampling")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")

    parser.add_argument("--kp-alt", type=float, default=0.22)
    parser.add_argument("--ki-alt", type=float, default=0.01)
    parser.add_argument("--kd-alt", type=float, default=0.10)
    parser.add_argument("--kp-att", type=float, default=0.30)
    parser.add_argument("--ki-att", type=float, default=0.00)
    parser.add_argument("--kd-att", type=float, default=0.20)
    parser.add_argument("--kp-yaw", type=float, default=0.20)
    parser.add_argument("--ki-yaw", type=float, default=0.00)
    parser.add_argument("--kd-yaw", type=float, default=0.20)
    parser.add_argument("--k-pos-xy", type=float, default=0.125)
    parser.add_argument("--ki-pos-xy", type=float, default=0.0)
    parser.add_argument("--k-vel-xy", type=float, default=0.21)
    parser.add_argument("--max-tilt-cmd", type=float, default=0.092)
    parser.add_argument("--max-tilt-rate", type=float, default=0.12)
    parser.add_argument("--tilt-recovery-alt-err", type=float, default=1.50)
    parser.add_argument("--tilt-recovery-ang-rate", type=float, default=0.80)
    parser.add_argument("--lateral-recovery-attenuation", type=float, default=0.70)
    parser.add_argument("--min-lateral-scale", type=float, default=0.0)
    parser.add_argument("--max-rate-cmd-rp", type=float, default=None)
    parser.add_argument("--gyro-comp-rp", type=float, default=0.0)
    parser.add_argument("--min-fin-cmd-xy", type=float, default=0.018)
    parser.add_argument("--xy-active-error", type=float, default=0.20)
    parser.add_argument("--throttle-hover", type=float, default=0.90)
    parser.add_argument("--max-fin-angle", type=float, default=0.115)
    parser.add_argument(
        "--summary-decimals",
        type=int,
        default=3,
        help="Decimal places for human-readable summary output",
    )
    parser.add_argument(
        "--log-decimals",
        type=int,
        default=3,
        help="Decimal places for per-step logs",
    )
    parser.add_argument(
        "--log-format",
        choices=["pretty", "json"],
        default="pretty",
        help="Per-step log output format",
    )

    parser.add_argument(
        "--log-state",
        action="store_true",
        help="Print full vehicle state vectors to terminal",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print one state log every N steps when --log-state is enabled",
    )
    parser.add_argument(
        "--log-obs",
        action="store_true",
        help="Include raw observation vector in each per-step state log",
    )
    parser.add_argument(
        "--log-pid",
        action="store_true",
        help="Include PID internal loop signals in each per-step state log",
    )
    parser.add_argument(
        "--max-wall-time",
        type=float,
        default=None,
        help=(
            "Maximum wall-clock seconds before forcing process exit. "
            "Default is max(180, duration * 4 + 180)."
        ),
    )
    parser.add_argument("--disable-fin-forces", action="store_true")
    parser.add_argument("--disable-thrust-loss", action="store_true")
    parser.add_argument("--disable-wind-force", action="store_true")
    parser.add_argument("--disable-edf-static-torque", action="store_true")
    parser.add_argument("--disable-edf-dynamic-torque", action="store_true")
    parser.add_argument("--disable-edf-gyro-torque", action="store_true")
    parser.add_argument(
        "--edf-gyro-torque-scale",
        type=float,
        default=None,
        help="Runtime multiplier for EDF gyroscopic torque after config/model computation.",
    )
    parser.add_argument(
        "--body-angular-damping",
        type=float,
        default=None,
        help="Body-frame angular damping torque coefficient, Nm per rad/s.",
    )
    parser.add_argument(
        "--disable-edf-torques",
        action="store_true",
        help="Disable static, dynamic spool, and gyroscopic EDF body torques.",
    )
    parser.add_argument(
        "--fixed-hover-spawn",
        action="store_true",
        help="Diagnostic reset: spawn at the hover target with zero velocity and level attitude.",
    )
    parser.add_argument(
        "--landing-descent-rate",
        type=float,
        default=1.0,
        help="Descent rate (m/s) commanded by the landing guidance trajectory.",
    )
    parser.add_argument(
        "--landing-flare-alt",
        type=float,
        default=0.5,
        help="Altitude above pad (m) at which the guidance switches to flare descent rate.",
    )
    parser.add_argument(
        "--landing-flare-descent-rate",
        type=float,
        default=0.25,
        help="Descent rate (m/s) used during the final flare phase.",
    )
    parser.add_argument(
        "--landing-touchdown-speed-limit",
        type=float,
        default=1.5,
        help="Pass criterion: max downward speed (m/s) at touchdown.",
    )
    parser.add_argument(
        "--landing-pad-distance-limit",
        type=float,
        default=0.5,
        help="Pass criterion: max horizontal distance (m) from pad center at touchdown.",
    )
    parser.add_argument(
        "--spawn-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Diagnostic reset position in world meters; also zeroes velocity/attitude unless overridden later.",
    )
    return parser.parse_args()


def _to_list(tensor) -> list[float]:
    return [float(x) for x in tensor.detach().cpu().tolist()]


def _to_list_rounded(tensor, decimals: int) -> list[Any]:
    def round_item(value: Any) -> Any:
        if isinstance(value, list):
            return [round_item(item) for item in value]
        return round(float(value), decimals)

    return round_item(tensor.detach().cpu().tolist())


def _round_nested(data: Any, decimals: int) -> Any:
    if isinstance(data, dict):
        return {k: _round_nested(v, decimals) for k, v in data.items()}
    if isinstance(data, list):
        return [_round_nested(v, decimals) for v in data]
    if isinstance(data, float):
        return round(data, decimals)
    return data


def _dynamics_debug(env: Any, decimals: int) -> dict[str, Any]:
    debug = getattr(env, "_last_dynamics_debug", {})
    return {
        key: _to_list_rounded(value, decimals)
        if hasattr(value, "detach")
        else _round_nested(value, decimals)
        for key, value in debug.items()
    }


def _value_to_string(value: Any, decimals: int) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    if isinstance(value, list):
        return "[" + ", ".join(_value_to_string(v, decimals) for v in value) + "]"
    return str(value)


def _print_state_block(payload: dict[str, Any], n_steps: int, dt: float, decimals: int) -> None:
    step = int(payload.get("step", 0))
    sim_time_s = float(payload.get("sim_time_s", 0.0))
    print("")
    print(
        f"--- State Vector | step={step}/{max(n_steps - 1, 0)} | "
        f"dt={dt:.{decimals}f}s | sim_time={sim_time_s:.{decimals}f}s ---"
    )

    base_order = [
        "reward",
        "terminated",
        "truncated",
        "target_position_world_m",
        "position_error_world_m",
        "tilt_rad",
        "roll_pitch_yaw_rad",
        "angular_rate_norm_rad_s",
        "action",
    ]
    for key in base_order:
        if key in payload:
            print(f"{key}: {_value_to_string(payload[key], decimals)}")

    state = payload.get("state")
    if isinstance(state, dict):
        print("state:")
        for key, value in state.items():
            print(f"  {key}: {_value_to_string(value, decimals)}")

    if "obs_vector" in payload:
        print(f"obs_vector: {_value_to_string(payload['obs_vector'], decimals)}")

    pid_debug = payload.get("pid_debug")
    if isinstance(pid_debug, dict):
        print("pid_debug:")
        for key, value in pid_debug.items():
            print(f"  {key}: {_value_to_string(value, decimals)}")
    sys.stdout.flush()


def _print_episode_reset(step: int, sim_time_s: float, n_steps: int, dt: float, decimals: int) -> None:
    print("")
    print(
        f"--- Episode Reset | step={step}/{max(n_steps - 1, 0)} | "
        f"dt={dt:.{decimals}f}s | sim_time={sim_time_s:.{decimals}f}s ---"
    )


def main():
    args = parse_args()
    if args.summary_decimals < 0:
        raise ValueError("--summary-decimals must be >= 0")
    if args.log_decimals < 0:
        raise ValueError("--log-decimals must be >= 0")
    max_wall_time = args.max_wall_time
    if max_wall_time is None:
        max_wall_time = max(180.0, args.duration * 4.0 + 180.0)
    watchdog = WallClockWatchdog(max_wall_time, label="PID hover evaluation")
    watchdog.start()

    sim_root = Path(__file__).parent.parent
    sys.path.insert(0, str(sim_root))

    simulation_app = None
    env = None
    exit_code = 2
    try:
        print(
            f"PID hover eval starting: task={args.task}, duration={args.duration:.1f}s, "
            f"headless={args.headless}, max_wall_time={max_wall_time:.1f}s",
            flush=True,
        )
        print("Bootstrapping Isaac Sim...", flush=True)
        from isaac_launcher import launch_simulation_app

        simulation_app = launch_simulation_app(headless=args.headless)
    except ImportError:
        print("ERROR: Isaac Sim not available.", file=sys.stderr, flush=True)
        watchdog.stop()
        return 1

    try:
        from tvc_env.common.constants import ContactState
        from tvc_env.common.quaternions import tilt_angle, to_euler
        from tvc_env.controllers.landing_guidance import LandingGuidance
        from tvc_env.controllers.pid_adapter import PIDController
        from tvc_env.envs.base_env import BaseEnvConfig
        from tvc_env.envs.direct_rl_env import TVCDirectRLEnv
        import torch

        if args.log_every <= 0:
            raise ValueError("--log-every must be >= 1")
        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)

        disable_all_edf_torques = args.disable_edf_torques
        dynamics_overrides = {
            "enable_fin_forces": not args.disable_fin_forces,
            "enable_thrust_loss": not args.disable_thrust_loss,
            "enable_wind_force": not args.disable_wind_force,
            "enable_edf_static_torque": not (args.disable_edf_static_torque or disable_all_edf_torques),
            "enable_edf_dynamic_torque": not (args.disable_edf_dynamic_torque or disable_all_edf_torques),
            "enable_edf_gyro_torque": not (args.disable_edf_gyro_torque or disable_all_edf_torques),
        }
        if args.edf_gyro_torque_scale is not None:
            dynamics_overrides["edf_gyro_torque_scale"] = args.edf_gyro_torque_scale
        if args.body_angular_damping is not None:
            dynamics_overrides["body_angular_damping"] = args.body_angular_damping
        overrides: dict[str, Any] = {"dynamics": dynamics_overrides}
        if args.fixed_hover_spawn or args.spawn_position is not None:
            position = args.spawn_position if args.spawn_position is not None else [0.0, 0.0, 5.0]
            overrides["task"] = {
                "spawn": {
                    "position_range": [position, position],
                    "velocity_range": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    "attitude_range": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                }
            }
        config = BaseEnvConfig(
            task_name=args.task,
            env_config_path=sim_root / args.env_config,
            disturbance_config_path=sim_root / args.disturbance if args.disturbance else None,
            overrides=overrides,
            sim_root=sim_root,
        )

        print("Isaac Sim ready. Building TVC environment...", flush=True)
        env = TVCDirectRLEnv(config)
        pid = PIDController(
            num_envs=1,
            device=env.device,
            kp_alt=args.kp_alt,
            ki_alt=args.ki_alt,
            kd_alt=args.kd_alt,
            kp_att=args.kp_att,
            ki_att=args.ki_att,
            kd_att=args.kd_att,
            kp_yaw=args.kp_yaw,
            ki_yaw=args.ki_yaw,
            kd_yaw=args.kd_yaw,
            k_pos_xy=args.k_pos_xy,
            ki_pos_xy=args.ki_pos_xy,
            k_vel_xy=args.k_vel_xy,
            max_tilt_cmd=args.max_tilt_cmd,
            max_tilt_rate=args.max_tilt_rate,
            tilt_recovery_alt_err=args.tilt_recovery_alt_err,
            tilt_recovery_ang_rate=args.tilt_recovery_ang_rate,
            lateral_recovery_attenuation=args.lateral_recovery_attenuation,
            min_lateral_scale=args.min_lateral_scale,
            max_rate_cmd_rp=args.max_rate_cmd_rp,
            gyro_comp_rp=args.gyro_comp_rp,
            min_fin_cmd_xy=args.min_fin_cmd_xy,
            xy_active_error=args.xy_active_error,
            throttle_hover=args.throttle_hover,
            max_fin_angle=args.max_fin_angle,
        )

        obs_dict, _ = env.reset(seed=args.seed)
        obs = obs_dict["policy"]
        pid.reset()

        dt = 1.0 / 30.0
        guidance: LandingGuidance | None = None
        if args.task == "landing":
            guidance = LandingGuidance(
                num_envs=1,
                device=env.device,
                target_position=env._target_position,
                descent_rate=args.landing_descent_rate,
                flare_alt=args.landing_flare_alt,
                flare_descent_rate=args.landing_flare_descent_rate,
                dt=dt,
            )
            guidance.reset(obs=obs)
        print("Environment reset complete.", flush=True)
        n_steps = int(args.duration / dt)
        if n_steps <= 0:
            raise ValueError("Duration too short; increase --duration")

        pos_errors: list[float] = []
        tilts: list[float] = []
        ang_rates: list[float] = []
        # Landing-only metrics (collected only when task=landing).
        thrust_ratios: list[float] = []
        landed_step: int | None = None
        touchdown_speed: float | None = None
        touchdown_pad_distance: float | None = None
        crashed: bool = False
        min_altitude: float = float("inf")
        final_pad_distance: float | None = None
        prev_contact: int = int(ContactState.AIRBORNE)

        print(
            f"PID hover eval: task={args.task}, duration={args.duration:.1f}s ({n_steps} steps), "
            f"log_state={args.log_state}",
            flush=True,
        )
        t_start = time.time()

        if args.log_state:
            state = env._build_vehicle_state()
            pos_err_xyz = obs[0, 0:3]
            quat_wxyz = obs[0, 3:7]
            ang_vel_frd = obs[0, 10:13]
            roll, pitch, yaw = to_euler(quat_wxyz.unsqueeze(0))
            roll_v = float(roll[0].item())
            pitch_v = float(pitch[0].item())
            yaw_v = float(yaw[0].item())
            tilt = float(tilt_angle(quat_wxyz.unsqueeze(0))[0].item())
            initial_payload = {
                "type": "state_vector",
                "phase": "initial_reset",
                "step": 0,
                "sim_time_s": 0.0,
                "reward": None,
                "terminated": False,
                "truncated": False,
                "target_position_world_m": _to_list_rounded(
                    env._target_position.to(state.position.device), args.log_decimals
                ),
                "position_error_world_m": _to_list_rounded(pos_err_xyz, args.log_decimals),
                "tilt_rad": round(float(tilt), args.log_decimals),
                "roll_pitch_yaw_rad": [
                    round(roll_v, args.log_decimals),
                    round(pitch_v, args.log_decimals),
                    round(yaw_v, args.log_decimals),
                ],
                "angular_rate_norm_rad_s": round(float(ang_vel_frd.norm().item()), args.log_decimals),
                "action": None,
                "state": {
                    "position_world_m": _to_list_rounded(state.position[0], args.log_decimals),
                    "quaternion_wxyz": _to_list_rounded(state.quaternion_wxyz[0], args.log_decimals),
                    "linear_vel_world_m_s": _to_list_rounded(state.linear_vel_world[0], args.log_decimals),
                    "angular_vel_world_rad_s": _to_list_rounded(state.angular_vel_world[0], args.log_decimals),
                    "linear_vel_frd_m_s": _to_list_rounded(state.linear_vel_frd[0], args.log_decimals),
                    "angular_vel_frd_rad_s": _to_list_rounded(state.angular_vel_frd[0], args.log_decimals),
                    "height_m": round(float(state.height[0].item()), args.log_decimals),
                    "fin_angles_rad": _to_list_rounded(state.fin_angles[0], args.log_decimals),
                    "fin_rates_rad_s": _to_list_rounded(state.fin_rates[0], args.log_decimals),
                    "motor_omega_rad_s": round(float(state.motor_omega[0].item()), args.log_decimals),
                    "contact_state": int(state.contact_state[0].item()),
                },
                "dynamics_debug": {},
            }
            if args.log_obs:
                initial_payload["obs_vector"] = _to_list_rounded(obs[0], args.log_decimals)
            if args.log_format == "json":
                print(json.dumps(initial_payload, separators=(",", ":"), ensure_ascii=True), flush=True)
            else:
                _print_state_block(initial_payload, n_steps=n_steps, dt=dt, decimals=args.log_decimals)

        for step in range(n_steps):
            if guidance is not None:
                pid_obs = guidance.modify_obs(obs)
                action = pid.compute_action(pid_obs)
                action = guidance.post_action(action)
            else:
                action = pid.compute_action(obs)
            obs_dict, rewards, terminated, truncated, _ = env.step(action)
            obs = obs_dict["policy"]

            if guidance is not None:
                # Landing telemetry: detect first AIRBORNE→LANDED transition,
                # log thrust ratio for delta-v proxy, and watch for crashes.
                contact_int = int(obs[0, 23].item())
                state = env._build_vehicle_state()
                pos_world = state.position[0]
                pad_xy = env._target_position[0, :2].to(pos_world.device)
                pad_dist = float((pos_world[:2] - pad_xy).norm().item())
                final_pad_distance = pad_dist
                min_altitude = min(min_altitude, float(pos_world[2].item()))
                omega_max = max(float(env._omega_max), 1.0)
                ratio = float((state.motor_omega[0] / omega_max).clamp(min=0.0, max=1.0).item())
                thrust_ratios.append(ratio * ratio)
                if (
                    landed_step is None
                    and prev_contact != int(ContactState.LANDED)
                    and contact_int == int(ContactState.LANDED)
                ):
                    landed_step = step
                    touchdown_speed = float(state.linear_vel_frd[0, 2].clamp(min=0.0).item())
                    touchdown_pad_distance = pad_dist
                if contact_int == int(ContactState.CRASHED):
                    crashed = True
                prev_contact = contact_int

            pos_err_xyz = obs[0, 0:3]
            quat_wxyz = obs[0, 3:7]
            ang_vel_frd = obs[0, 10:13]

            pos_err_mag = float(pos_err_xyz.norm().item())
            pos_errors.append(pos_err_mag)

            roll, pitch, yaw = to_euler(quat_wxyz.unsqueeze(0))
            roll_v = float(roll[0].item())
            pitch_v = float(pitch[0].item())
            yaw_v = float(yaw[0].item())
            tilt = float(tilt_angle(quat_wxyz.unsqueeze(0))[0].item())
            tilts.append(tilt)

            ang_rate = float(ang_vel_frd.norm().item())
            ang_rates.append(ang_rate)

            if args.log_state and (step % args.log_every == 0):
                state = env._build_vehicle_state()
                payload = {
                    "type": "state_vector",
                    "step": int(step),
                    "sim_time_s": round(float(step * dt), args.log_decimals),
                    "reward": round(float(rewards[0].item()), args.log_decimals),
                    "terminated": bool(terminated[0].item()),
                    "truncated": bool(truncated[0].item()),
                    "target_position_world_m": _to_list_rounded(
                        env._target_position.to(state.position.device), args.log_decimals
                    ),
                    "position_error_world_m": _to_list_rounded(pos_err_xyz, args.log_decimals),
                    "tilt_rad": round(float(tilt), args.log_decimals),
                    "roll_pitch_yaw_rad": [
                        round(roll_v, args.log_decimals),
                        round(pitch_v, args.log_decimals),
                        round(yaw_v, args.log_decimals),
                    ],
                    "angular_rate_norm_rad_s": round(float(ang_rate), args.log_decimals),
                    "action": _to_list_rounded(action[0], args.log_decimals),
                    "state": {
                        "position_world_m": _to_list_rounded(state.position[0], args.log_decimals),
                        "quaternion_wxyz": _to_list_rounded(state.quaternion_wxyz[0], args.log_decimals),
                        "linear_vel_world_m_s": _to_list_rounded(state.linear_vel_world[0], args.log_decimals),
                        "angular_vel_world_rad_s": _to_list_rounded(state.angular_vel_world[0], args.log_decimals),
                        "linear_vel_frd_m_s": _to_list_rounded(state.linear_vel_frd[0], args.log_decimals),
                        "angular_vel_frd_rad_s": _to_list_rounded(state.angular_vel_frd[0], args.log_decimals),
                        "height_m": round(float(state.height[0].item()), args.log_decimals),
                        "fin_angles_rad": _to_list_rounded(state.fin_angles[0], args.log_decimals),
                        "fin_rates_rad_s": _to_list_rounded(state.fin_rates[0], args.log_decimals),
                        "motor_omega_rad_s": round(float(state.motor_omega[0].item()), args.log_decimals),
                        "contact_state": int(state.contact_state[0].item()),
                    },
                    "dynamics_debug": _dynamics_debug(env, args.log_decimals),
                }
                if args.log_obs:
                    payload["obs_vector"] = _to_list_rounded(obs[0], args.log_decimals)
                if args.log_pid:
                    payload["pid_debug"] = _round_nested(pid.get_debug_state(env_idx=0), args.log_decimals)
                if args.log_format == "json":
                    print(json.dumps(payload, separators=(",", ":"), ensure_ascii=True), flush=True)
                else:
                    _print_state_block(payload, n_steps=n_steps, dt=dt, decimals=args.log_decimals)

            done = bool((terminated | truncated)[0].item())
            if guidance is not None:
                # Single-shot landing run: stop after the first crash, or once
                # the vehicle has settled on the pad for a short dwell window.
                if crashed:
                    if args.log_state and args.log_format == "json":
                        print(json.dumps({"type": "episode_end", "reason": "crash", "step": step}, separators=(",", ":")), flush=True)
                    break
                if landed_step is not None and step - landed_step >= 15:
                    if args.log_state and args.log_format == "json":
                        print(json.dumps({"type": "episode_end", "reason": "landed", "step": step}, separators=(",", ":")), flush=True)
                    break
            if done:
                if args.log_state:
                    reset_payload = {
                        "type": "episode_reset",
                        "step": int(step),
                        "sim_time_s": round(float(step * dt), args.log_decimals),
                    }
                    if args.log_format == "json":
                        print(json.dumps(reset_payload, separators=(",", ":"), ensure_ascii=True), flush=True)
                    else:
                        _print_episode_reset(
                            step=reset_payload["step"],
                            sim_time_s=reset_payload["sim_time_s"],
                            n_steps=n_steps,
                            dt=dt,
                            decimals=args.log_decimals,
                        )
                obs_dict, _ = env.reset()
                obs = obs_dict["policy"]
                pid.reset()
                if guidance is not None:
                    guidance.reset(obs=obs)

            simulation_app.update()

        elapsed = time.time() - t_start

        d = args.summary_decimals
        mean_pos = statistics.mean(pos_errors) if pos_errors else 0.0
        max_pos = max(pos_errors) if pos_errors else 0.0
        mean_tilt = statistics.mean(tilts) if tilts else 0.0
        max_tilt = max(tilts) if tilts else 0.0
        mean_rate = statistics.mean(ang_rates) if ang_rates else 0.0
        max_rate = max(ang_rates) if ang_rates else 0.0

        if args.task == "landing":
            delta_v_proxy = sum(thrust_ratios) * dt  # seconds-of-full-thrust
            print("\n=== PID Landing Evaluation Results ===", flush=True)
            print(f"Duration: {args.duration:.0f}s budget ({n_steps} steps), wall time: {elapsed:.1f}s")
            print("")
            print("Tilt (rad):")
            print(
                f"  mean={mean_tilt:.{d}f}  max={max_tilt:.{d}f}  "
                f"({math.degrees(mean_tilt):.1f} deg mean, {math.degrees(max_tilt):.1f} deg max)"
            )
            print("Angular rate (rad/s):")
            print(f"  mean={mean_rate:.{d}f}  max={max_rate:.{d}f}")
            print("")
            print(f"Min altitude reached: {min_altitude:.{d}f} m")
            print(f"Final pad horizontal distance: {final_pad_distance if final_pad_distance is not None else float('nan'):.{d}f} m")
            print(f"Crashed: {crashed}")
            if landed_step is not None:
                print(
                    f"Touchdown: step={landed_step} t={landed_step*dt:.{d}f}s "
                    f"speed={touchdown_speed:.{d}f} m/s pad_dist={touchdown_pad_distance:.{d}f} m"
                )
            else:
                print("Touchdown: did not land within duration budget")
            print(f"Delta-v proxy (sum of (T/T_max)^2 * dt): {delta_v_proxy:.{d}f} s")

            landed_ok = landed_step is not None
            speed_ok = landed_ok and (touchdown_speed is not None) and touchdown_speed < args.landing_touchdown_speed_limit
            pad_ok = landed_ok and (touchdown_pad_distance is not None) and touchdown_pad_distance < args.landing_pad_distance_limit
            crash_ok = not crashed
            tilt_ok = max_tilt < 0.262

            print("\nPass criteria:")
            print(f"  landed within duration:  {'PASS' if landed_ok else 'FAIL'}")
            print(f"  not crashed:  {'PASS' if crash_ok else 'FAIL'}")
            print(
                f"  touchdown speed < {args.landing_touchdown_speed_limit:.2f} m/s:  "
                f"{'PASS' if speed_ok else 'FAIL'}"
            )
            print(
                f"  pad distance < {args.landing_pad_distance_limit:.2f} m:  "
                f"{'PASS' if pad_ok else 'FAIL'}"
            )
            print(f"  max tilt < 15 deg:  {'PASS' if tilt_ok else 'FAIL'}")
            overall = landed_ok and speed_ok and pad_ok and crash_ok and tilt_ok
            print(f"\n{'PASS' if overall else 'FAIL'}: PID landing evaluation")
            exit_code = 0 if overall else 1
            return exit_code

        print("\n=== PID Hover Evaluation Results ===", flush=True)
        print(f"Duration: {args.duration:.0f}s ({n_steps} steps), wall time: {elapsed:.1f}s")
        print("")
        print("Position error (m):")
        print(f"  mean={mean_pos:.{d}f}  max={max_pos:.{d}f}")
        print("Tilt (rad):")
        print(
            f"  mean={mean_tilt:.{d}f}  max={max_tilt:.{d}f}  "
            f"({math.degrees(mean_tilt):.1f} deg mean, {math.degrees(max_tilt):.1f} deg max)"
        )
        print("Angular rate (rad/s):")
        print(f"  mean={mean_rate:.{d}f}  max={max_rate:.{d}f}")

        pos_ok = mean_pos < 0.5
        tilt_ok = max_tilt < 0.262
        rate_ok = mean_rate < 1.0

        print("\nPass criteria:")
        print(f"  mean pos_err < 0.5m:  {'PASS' if pos_ok else 'FAIL'}")
        print(f"  max tilt < 15 deg (0.26 rad):  {'PASS' if tilt_ok else 'FAIL'}")
        print(f"  mean ang_rate < 1.0 rad/s:  {'PASS' if rate_ok else 'FAIL'}")

        overall = pos_ok and tilt_ok and rate_ok
        print(f"\n{'PASS' if overall else 'FAIL'}: PID hover evaluation")
        exit_code = 0 if overall else 1
        return exit_code

    except Exception as exc:
        print(f"\nERROR: PID hover evaluation failed: {exc}", file=sys.stderr, flush=True)
        import traceback

        traceback.print_exc()
        exit_code = 2
        return exit_code

    finally:
        watchdog.reset(30.0, label="PID hover evaluation cleanup")
        if env is not None:
            print("Closing TVC environment...", flush=True)
            env.close()
        if simulation_app is not None:
            print("Closing Isaac Sim...", flush=True)
            from isaac_launcher import close_simulation_app
            closed = close_simulation_app(simulation_app)
            print("Isaac Sim closed." if closed else "Isaac Sim fast shutdown requested.", flush=True)
        watchdog.stop()


if __name__ == "__main__":
    force_process_exit(main())
