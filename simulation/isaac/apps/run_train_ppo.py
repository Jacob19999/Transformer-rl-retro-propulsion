"""
PPO training entrypoint for the TVC environment.

Instantiates the vectorized environment (128 envs by default) and runs a
standard PPO training loop with the PPOAdapter action scaling. Training
outputs are saved to runs/<timestamp>/.

NOTE: This script sets up the environment and training scaffolding.  The
PPO algorithm itself should be provided by an external RL library (e.g.
Stable Baselines3, CleanRL, skrl) — see comments in main() for integration
points.

Usage:
    python apps/run_train_ppo.py --task hover --seed 42
    python apps/run_train_ppo.py --task landing --env-config configs/env/train_128.yaml
    python apps/run_train_ppo.py --total-steps 10000000 --seed 0
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="PPO training for TVC environment")
    parser.add_argument("--task", default="hover", choices=["hover", "landing"])
    parser.add_argument("--env-config", default="configs/env/train_128.yaml")
    parser.add_argument("--disturbance", default="configs/disturbances/nominal.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--output-dir", default="runs", help="Base output directory")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    sim_root = Path(__file__).parent.parent
    sys.path.insert(0, str(sim_root))

    # Timestamped output directory
    run_name = f"ppo_{args.task}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = sim_root / args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from isaacsim import SimulationApp
        simulation_app = SimulationApp({"headless": args.headless})
    except ImportError:
        print("ERROR: Isaac Sim not available.", file=sys.stderr)
        sys.exit(1)

    try:
        import torch
        from tvc_env.envs.base_env import BaseEnvConfig
        from tvc_env.envs.direct_rl_env import TVCDirectRLEnv
        from tvc_env.controllers.ppo_adapter import PPOAdapter

        torch.manual_seed(args.seed)

        config = BaseEnvConfig(
            task_name=args.task,
            env_config_path=sim_root / args.env_config,
            disturbance_config_path=sim_root / args.disturbance,
            sim_root=sim_root,
        )
        env = TVCDirectRLEnv(config)
        num_envs = config.num_envs

        print(f"PPO training: task={args.task}, envs={num_envs}, seed={args.seed}")
        print(f"Total steps: {args.total_steps:,}")
        print(f"Output: {output_dir}")

        # --- RL LIBRARY INTEGRATION POINT ---
        # Replace the dummy loop below with your RL library (skrl, SB3, CleanRL, etc.)
        # The environment conforms to:
        #   obs_dict, _ = env.reset()               → obs["policy"]: (num_envs, 24)
        #   obs_dict, rew, done, trunc, info = env.step(action)  → action: (num_envs, 5)
        #
        # PPOAdapter wraps the policy:
        #   adapter = PPOAdapter(policy_fn, max_fin_angle=0.262)
        #   action = adapter.compute_action(obs)
        #
        # action_space: Box(-1, 1, shape=(5,)) — PPOAdapter scales internally
        # observation_space: Box(-inf, inf, shape=(24,))
        # -------------------------------------

        obs_dict, _ = env.reset()
        steps_done = 0
        t_start = time.time()

        print("\n[Stub training loop — integrate with RL library for actual training]")
        print("Running 1000 random-action steps to validate env compatibility...\n")

        for step in range(min(1000, args.total_steps)):
            # Dummy random action (in [-1, 1] range as PPOAdapter expects)
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
        print("✓ Environment compatibility validated — integrate RL library to train")

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
