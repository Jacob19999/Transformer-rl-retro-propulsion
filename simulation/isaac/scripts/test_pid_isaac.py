"""
test_pid_isaac.py -- PID evaluation / trace-logging entry point for Isaac Sim env.

Scaffold for specs/003-pid-controller (T002, T018-T021):
- CLI for config / PID YAML / episodes / seed / log-dir
- Runtime force toggles (wind, gyro, anti-torque, gravity)
- Per-episode evaluation + trace logging to be implemented.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PID controller in Isaac Sim and log episode traces."
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
        required=True,
        help="PID YAML path (typically best_pid.yaml from tuning).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for env / controller (default: 0).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs/pid_isaac_eval",
        help="Directory for episode summaries and per-step traces.",
    )
    parser.add_argument(
        "--disable-wind",
        action="store_true",
        help="Disable Isaac wind model during evaluation episodes.",
    )
    parser.add_argument(
        "--disable-gyro",
        action="store_true",
        help="Disable gyro precession torque during evaluation episodes.",
    )
    parser.add_argument(
        "--disable-anti-torque",
        action="store_true",
        help="Disable steady-state + ramp EDF anti-torque during evaluation episodes.",
    )
    parser.add_argument(
        "--disable-gravity",
        action="store_true",
        help="Disable gravity in the Isaac scene (diagnostic only; not for normal landings).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    pid_path = Path(args.pid_config)
    log_dir = Path(args.log_dir)

    # TODO: implement evaluation + trace logging:
    # - instantiate EDFIsaacEnv
    # - apply runtime force toggles
    # - run episodes with PIDController
    # - write episode_summary.csv, run_metadata.json, and trace_epXXXX_envYY.csv
    print(
        f"[test_pid_isaac] config={config_path} pid={pid_path} "
        f"episodes={args.episodes} seed={args.seed} log_dir={log_dir}"
    )


if __name__ == "__main__":
    main()

