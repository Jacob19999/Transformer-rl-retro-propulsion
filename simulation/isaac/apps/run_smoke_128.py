"""
128-environment vectorized smoke test app.

Runs random actions for N steps with 128 parallel environments,
reports tensor shape validation, NaN check, reset count, and performance metrics.

Usage:
    python apps/run_smoke_128.py --task hover --steps 1000
    python apps/run_smoke_128.py --task hover --steps 1000 --no-headless
    python apps/run_smoke_128.py --override env.num_envs=64
"""

import argparse
import sys
import time
from pathlib import Path

from runner_safety import force_process_exit


def parse_args():
    parser = argparse.ArgumentParser(description="128-env vectorized smoke test")
    parser.add_argument("--task", default="hover", choices=["hover", "landing"])
    parser.add_argument("--env-config", default="configs/env/train_128.yaml")
    parser.add_argument("--disturbance", default="configs/disturbances/nominal.yaml")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    sim_root = Path(__file__).parent.parent
    sys.path.insert(0, str(sim_root))

    try:
        from isaac_launcher import launch_simulation_app
        simulation_app = launch_simulation_app(headless=args.headless)
    except ImportError:
        print("ERROR: Isaac Sim not available.", file=sys.stderr)
        return 1

    try:
        import torch
        from tvc_env.envs.base_env import BaseEnvConfig
        from tvc_env.envs.direct_rl_env import TVCDirectRLEnv

        overrides = {}
        for item in args.override:
            key, _, value = item.partition("=")
            parts = key.split(".")
            d = overrides
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value

        config = BaseEnvConfig(
            task_name=args.task,
            env_config_path=sim_root / args.env_config,
            disturbance_config_path=sim_root / args.disturbance if args.disturbance else None,
            overrides=overrides,
            sim_root=sim_root,
        )
        num_envs = config.num_envs
        print(f"Smoke test: {num_envs} environments, {args.steps} steps, task={args.task}")

        env = TVCDirectRLEnv(config)
        obs_dict, _ = env.reset()
        obs = obs_dict["policy"]

        # Validate initial obs shape
        assert obs.shape == (num_envs, 24), f"Expected obs ({num_envs}, 24), got {obs.shape}"
        assert not torch.isnan(obs).any(), "NaN in initial observations"
        print(f"PASS: Initial obs shape: {obs.shape}")

        # Run steps
        max_angle = 0.1
        t_start = time.time()
        nan_count = 0
        shape_errors = 0
        reset_count = 0

        for step in range(args.steps):
            action = torch.zeros(num_envs, 5)
            action[:, :4] = (torch.rand(num_envs, 4) - 0.5) * 2 * max_angle
            action[:, 4] = torch.rand(num_envs) * 0.3 + 0.6  # [0.6, 0.9]

            obs_dict, rewards, terminated, truncated, info = env.step(action)
            obs = obs_dict["policy"]

            # Shape checks
            if obs.shape != (num_envs, 24):
                shape_errors += 1
            if rewards.shape != (num_envs,):
                shape_errors += 1

            # NaN checks
            if torch.isnan(obs).any() or torch.isnan(rewards).any():
                nan_count += 1

            # Auto-reset counting
            done_mask = terminated | truncated
            if done_mask.any():
                reset_count += done_mask.sum().item()

            env.render()

        elapsed = time.time() - t_start
        steps_per_sec = (args.steps * num_envs) / elapsed

        print(f"\n=== Smoke Test Results ===")
        print(f"Steps: {args.steps}, Envs: {num_envs}")
        print(f"NaN occurrences: {nan_count}")
        print(f"Shape errors: {shape_errors}")
        print(f"Auto-resets: {reset_count}")
        print(f"Steps/sec: {steps_per_sec:.0f}")
        print(f"Time: {elapsed:.1f}s")

        if nan_count == 0 and shape_errors == 0:
            print("\nPASS: Smoke test passed")
            return 0
        else:
            print(f"\nFAIL: {nan_count} NaN occurrences, {shape_errors} shape errors")
            return 1

    finally:
        from isaac_launcher import close_simulation_app
        close_simulation_app(simulation_app)


if __name__ == "__main__":
    force_process_exit(main())
