"""
Single-environment debug app with gizmos.

Opens Isaac Sim viewport with drone, all debug gizmos (force arrows, body axes,
HUD telemetry), and keyboard/gamepad interface for manual control.

Usage:
    python apps/run_single_env_debug.py --task hover
    python apps/run_single_env_debug.py --task landing --disturbance configs/disturbances/wind.yaml
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Single-environment debug runner with gizmos")
    parser.add_argument("--task", default="hover", choices=["hover", "landing"])
    parser.add_argument("--env-config", default="configs/env/single_env_debug.yaml")
    parser.add_argument("--disturbance", default="configs/disturbances/nominal.yaml")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides: key=value ...")
    return parser.parse_args()


def parse_overrides(override_list: list[str]) -> dict:
    """Parse CLI override strings into nested dict."""
    overrides = {}
    for item in override_list:
        key, _, value = item.partition("=")
        parts = key.split(".")
        d = overrides
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = _try_cast(value)
    return overrides


def _try_cast(v: str):
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def main():
    args = parse_args()
    sim_root = Path(__file__).parent.parent
    sys.path.insert(0, str(sim_root))

    # Bootstrap Isaac Sim
    try:
        from isaac_launcher import launch_simulation_app
        simulation_app = launch_simulation_app(headless=False)
    except ImportError:
        print("ERROR: Isaac Sim not available. Cannot run single_env_debug.", file=sys.stderr)
        sys.exit(1)

    try:
        import yaml
        from tvc_env.envs.base_env import BaseEnvConfig
        from tvc_env.envs.single_env import SingleEnvDebug

        overrides = parse_overrides(args.override)
        config = BaseEnvConfig(
            task_name=args.task,
            env_config_path=sim_root / args.env_config,
            disturbance_config_path=sim_root / args.disturbance if args.disturbance else None,
            overrides=overrides,
            sim_root=sim_root,
        )

        env = SingleEnvDebug(config)
        obs, _ = env.reset()

        print(f"Running single-env debug: task={args.task}")
        print("Press Ctrl+C to stop")

        import torch
        step = 0
        while simulation_app.is_running():
            # Zero action by default (hover throttle at ~gravity compensation level)
            action = torch.zeros(1, 5)
            action[0, 4] = 0.75  # ~75% throttle to hover
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated.any() or truncated.any():
                obs, _ = env.reset()

            step += 1
            simulation_app.update()

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        from isaac_launcher import close_simulation_app
        close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
