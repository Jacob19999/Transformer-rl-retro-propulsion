"""
GTrXL-PPO training entrypoint for the TVC environment.

Instantiates the vectorized environment (128 envs by default) and provides
the scaffolding for a GTrXL-PPO training run. The GTrXL policy handles
sequence context (transformer memory) across steps within an episode.

Training outputs are saved to runs/<timestamp>/.

NOTE: This script sets up the environment scaffolding and GTrXLAdapter.
The GTrXL algorithm should be provided by an external library (e.g. skrl's
GRU/LSTM/Transformer memory wrappers, or a custom GTrXL implementation).

Usage:
    python apps/run_train_gtrxl.py --task hover --seed 42
    python apps/run_train_gtrxl.py --task landing --total-steps 10000000
    python apps/run_train_gtrxl.py --seq-len 32 --memory-dim 64
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="GTrXL-PPO training for TVC environment")
    parser.add_argument("--task", default="hover", choices=["hover", "landing"])
    parser.add_argument("--env-config", default="configs/env/train_128.yaml")
    parser.add_argument("--disturbance", default="configs/disturbances/nominal.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=10_000_000)
    parser.add_argument("--seq-len", type=int, default=32, help="Transformer sequence length")
    parser.add_argument("--memory-dim", type=int, default=64, help="GTrXL memory dimension")
    parser.add_argument("--output-dir", default="runs", help="Base output directory")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    sim_root = Path(__file__).parent.parent
    sys.path.insert(0, str(sim_root))

    # Timestamped output directory
    run_name = f"gtrxl_{args.task}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = sim_root / args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from isaac_launcher import launch_simulation_app
        simulation_app = launch_simulation_app(headless=args.headless)
    except ImportError:
        print("ERROR: Isaac Sim not available.", file=sys.stderr)
        sys.exit(1)

    try:
        import torch
        from tvc_env.envs.base_env import BaseEnvConfig
        from tvc_env.envs.direct_rl_env import TVCDirectRLEnv
        from tvc_env.controllers.gtrxl_adapter import GTrXLAdapter

        torch.manual_seed(args.seed)

        config = BaseEnvConfig(
            task_name=args.task,
            env_config_path=sim_root / args.env_config,
            disturbance_config_path=sim_root / args.disturbance,
            sim_root=sim_root,
        )
        env = TVCDirectRLEnv(config)
        num_envs = config.num_envs

        print(f"GTrXL-PPO training: task={args.task}, envs={num_envs}, seed={args.seed}")
        print(f"Sequence length: {args.seq_len}, memory dim: {args.memory_dim}")
        print(f"Total steps: {args.total_steps:,}")
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

        print("\n[Stub training loop — integrate with GTrXL-PPO library for actual training]")
        print("Running 1000 random-action steps to validate env compatibility...\n")

        for step in range(min(1000, args.total_steps)):
            # Dummy random action
            raw_action = torch.zeros(num_envs, 5)
            raw_action[:, :4] = (torch.rand(num_envs, 4) - 0.5) * 2 * 0.262
            raw_action[:, 4] = torch.rand(num_envs)

            obs_dict, rew, done, trunc, info = env.step(raw_action)
            steps_done += num_envs

            simulation_app.update()

        elapsed = time.time() - t_start
        steps_per_sec = steps_done / elapsed
        print(f"Env steps/sec: {steps_per_sec:.0f} ({steps_done:,} steps in {elapsed:.1f}s)")
        print(f"\nRun directory: {output_dir}")
        print("✓ Environment compatibility validated — integrate GTrXL library to train")

    finally:
        from isaac_launcher import close_simulation_app
        close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
