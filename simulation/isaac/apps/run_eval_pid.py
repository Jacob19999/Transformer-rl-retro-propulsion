"""
PID hover evaluation app.

Runs the PID controller in a single-environment hover task for a configurable
duration, logging telemetry and reporting position error, tilt, and angular
rate statistics.

Usage:
    python apps/run_eval_pid.py --task hover --duration 30
    python apps/run_eval_pid.py --task hover --env-config configs/env/single_env_debug.yaml
    python apps/run_eval_pid.py --disturbance configs/disturbances/wind.yaml --duration 60
"""

import argparse
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="PID hover evaluation")
    parser.add_argument("--task", default="hover", choices=["hover", "landing"])
    parser.add_argument("--env-config", default="configs/env/single_env_debug.yaml")
    parser.add_argument("--disturbance", default=None, help="Path to disturbance config YAML")
    parser.add_argument("--duration", type=float, default=30.0, help="Evaluation duration (s)")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    sim_root = Path(__file__).parent.parent
    sys.path.insert(0, str(sim_root))

    try:
        from isaacsim import SimulationApp
        simulation_app = SimulationApp({"headless": args.headless})
    except ImportError:
        print("ERROR: Isaac Sim not available.", file=sys.stderr)
        sys.exit(1)

    try:
        import torch
        import math
        from tvc_env.envs.base_env import BaseEnvConfig
        from tvc_env.envs.direct_rl_env import TVCDirectRLEnv
        from tvc_env.controllers.pid_adapter import PIDController

        config = BaseEnvConfig(
            task_name=args.task,
            env_config_path=sim_root / args.env_config,
            disturbance_config_path=sim_root / args.disturbance if args.disturbance else None,
            sim_root=sim_root,
        )

        env = TVCDirectRLEnv(config)
        pid = PIDController(num_envs=1, device=env.device)

        obs_dict, _ = env.reset()
        obs = obs_dict["policy"]
        pid.reset()

        dt = 1.0 / 30.0  # RL step rate
        n_steps = int(args.duration / dt)

        # Telemetry accumulators
        pos_errors = []
        tilts = []
        ang_rates = []

        print(f"PID hover eval: task={args.task}, duration={args.duration}s ({n_steps} steps)")
        t_start = time.time()

        for step in range(n_steps):
            action = pid.compute_action(obs)
            obs_dict, rewards, terminated, truncated, info = env.step(action)
            obs = obs_dict["policy"]

            # Extract state metrics
            pos_err_xyz = obs[0, 0:3]
            quat_wxyz = obs[0, 3:7]
            ang_vel_frd = obs[0, 10:13]

            # Position error magnitude
            pos_err_mag = pos_err_xyz.norm().item()
            pos_errors.append(pos_err_mag)

            # Tilt angle from quaternion (angle from identity)
            # w component gives half-angle: tilt = 2*acos(|w|)
            w = quat_wxyz[0].item()
            tilt = 2.0 * math.acos(min(abs(w), 1.0))
            tilts.append(tilt)

            # Angular rate magnitude
            ang_rate = ang_vel_frd.norm().item()
            ang_rates.append(ang_rate)

            # Auto-reset PID if episode ended
            done = (terminated | truncated)[0].item()
            if done:
                obs_dict, _ = env.reset()
                obs = obs_dict["policy"]
                pid.reset()

            simulation_app.update()

        elapsed = time.time() - t_start

        # Report statistics
        import statistics
        print(f"\n=== PID Hover Evaluation Results ===")
        print(f"Duration: {args.duration:.0f}s ({n_steps} steps), wall time: {elapsed:.1f}s")
        print(f"")
        print(f"Position error (m):")
        print(f"  mean={statistics.mean(pos_errors):.3f}  max={max(pos_errors):.3f}")
        print(f"Tilt (rad):")
        print(f"  mean={statistics.mean(tilts):.4f}  max={max(tilts):.4f}  "
              f"({math.degrees(statistics.mean(tilts)):.1f}° mean, {math.degrees(max(tilts)):.1f}° max)")
        print(f"Angular rate (rad/s):")
        print(f"  mean={statistics.mean(ang_rates):.4f}  max={max(ang_rates):.4f}")

        # Pass/fail thresholds per test_10 spec
        pos_ok = statistics.mean(pos_errors) < 0.5
        tilt_ok = max(tilts) < 0.262  # 15°
        rate_ok = statistics.mean(ang_rates) < 1.0

        print(f"\nPass criteria:")
        print(f"  mean pos_err < 0.5m:  {'✓ PASS' if pos_ok else '✗ FAIL'}")
        print(f"  max tilt < 15° (0.26 rad):  {'✓ PASS' if tilt_ok else '✗ FAIL'}")
        print(f"  mean ang_rate < 1.0 rad/s:  {'✓ PASS' if rate_ok else '✗ FAIL'}")

        overall = pos_ok and tilt_ok and rate_ok
        print(f"\n{'✓ PASS' if overall else '✗ FAIL'}: PID hover evaluation")
        sys.exit(0 if overall else 1)

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
