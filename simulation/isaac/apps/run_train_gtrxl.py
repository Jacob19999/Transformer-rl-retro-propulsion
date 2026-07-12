"""
GTrXL-PPO training entrypoint for the TVC environment.

This repository does not yet contain the sequence-aware PPO optimizer required
for a scientifically valid GTrXL run. The entrypoint therefore refuses to
claim training success and only runs an explicit environment compatibility
smoke when ``--env-smoke-only`` is supplied.

Training outputs are saved to runs/<timestamp>/.

NOTE: Do not use the smoke result as a trained-policy artifact.

Usage:
    python apps/run_train_gtrxl.py --env-smoke-only --task hover --seed 42
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from runner_safety import force_process_exit


def parse_args():
    parser = argparse.ArgumentParser(description="GTrXL-PPO training for TVC environment")
    parser.add_argument("--task", default="hover", choices=["hover", "landing"])
    parser.add_argument("--env-config", default="configs/env/train_128.yaml")
    parser.add_argument("--disturbance", default="configs/disturbances/nominal.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="runs", help="Base output directory")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument(
        "--env-smoke-only",
        action="store_true",
        help="Explicitly run environment compatibility only; no policy is trained.",
    )
    parser.add_argument("--smoke-steps", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.env_smoke_only:
        print(
            "ERROR: run_train_gtrxl.py has no GTrXL-PPO optimizer yet. "
            "Use --env-smoke-only for the explicit compatibility check; "
            "do not treat random rollouts as training.",
            file=sys.stderr,
            flush=True,
        )
        return 2
    sim_root = Path(__file__).parent.parent
    sys.path.insert(0, str(sim_root))

    # Timestamped output directory
    run_name = f"gtrxl_env_smoke_{args.task}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = sim_root / args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

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

        torch.manual_seed(args.seed)

        config = BaseEnvConfig(
            task_name=args.task,
            env_config_path=sim_root / args.env_config,
            disturbance_config_path=sim_root / args.disturbance,
            sim_root=sim_root,
        )
        env = TVCDirectRLEnv(config)
        num_envs = config.num_envs

        print(f"GTrXL environment smoke: task={args.task}, envs={num_envs}, seed={args.seed}")
        print(f"Smoke policy steps: {args.smoke_steps:,}")
        print(f"Output: {output_dir}")

        # --- RL LIBRARY INTEGRATION POINT ---
        # Replace the dummy loop below with your GTrXL-PPO library.
        #
        # GTrXLAdapter wraps the policy with memory management:
        #   def gtrxl_policy(obs, memory):
        #       return raw_action, new_memory  # each (num_envs, ...)
        #
        #   adapter = GTrXLAdapter(gtrxl_policy, max_fin_angle=0.262)
        #   action = adapter.compute_action(obs)
        #   adapter.reset(env_ids)  # called on episode resets
        #
        # Key considerations for GTrXL:
        #   - Observation history length = seq_len (32 by default)
        #   - Memory is reset per-env on episode boundaries
        #   - Use GTrXLAdapter.reset(done_env_ids) after each step
        # -------------------------------------

        obs_dict, _ = env.reset()
        steps_done = 0
        t_start = time.time()

        print("\nEnvironment compatibility only; no policy optimization is performed.")

        for step in range(args.smoke_steps):
            # Dummy random action
            raw_action = torch.zeros(num_envs, 5)
            raw_action[:, :4] = (torch.rand(num_envs, 4) - 0.5) * 2 * 0.262
            raw_action[:, 4] = torch.rand(num_envs)

            obs_dict, rew, done, trunc, info = env.step(raw_action)
            steps_done += num_envs

            env.render()

        elapsed = time.time() - t_start
        steps_per_sec = steps_done / elapsed
        print(f"Env steps/sec: {steps_per_sec:.0f} ({steps_done:,} steps in {elapsed:.1f}s)")
        print(f"\nRun directory: {output_dir}")
        print("PASS: Environment compatibility validated (no GTrXL policy was trained).")
        return 0

    finally:
        from isaac_launcher import close_simulation_app
        close_simulation_app(simulation_app)


if __name__ == "__main__":
    force_process_exit(main())
