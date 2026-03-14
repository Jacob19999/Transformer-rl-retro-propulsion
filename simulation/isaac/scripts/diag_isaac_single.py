"""
diag_isaac_single.py — Single-env gravity-fall diagnostic for Isaac Sim.

Implements T022:
- Launch single env
- Step with zero actions for 600 steps
- Print h_agl every 60 steps
- Assert drone contacts ground within episode

Usage::
    python -m simulation.isaac.scripts.diag_isaac_single
    python -m simulation.isaac.scripts.diag_isaac_single --config simulation/isaac/configs/isaac_env_single.yaml

quickstart.md Step 2 validation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from simulation.isaac.conventions import ACTION_DIM, OBS_H_AGL  # noqa: E402
from simulation.isaac.scripts._shared import create_sim_app, resolve_repo_path  # noqa: E402

_SIM_APP = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Isaac Sim single-env gravity-fall diagnostic")
    parser.add_argument(
        "--config",
        default="simulation/isaac/configs/isaac_env_single.yaml",
        help="Path to Isaac env YAML config",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=600,
        help="Number of steps to simulate (default: 600 = 5 s at 1/120 s)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to run (default: 100)",
    )
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config)

    print(f"[diag_isaac_single] Loading env from: {config_path}")

    # Launch Isaac Sim app before importing any isaaclab/carb modules
    global _SIM_APP
    _SIM_APP = create_sim_app(headless=False)

    from simulation.isaac.envs.edf_isaac_env import EDFIsaacEnv

    env = EDFIsaacEnv(config_path=config_path, render_mode="human")

    print(f"[diag_isaac_single] Running {args.episodes} episodes, up to {args.steps} steps each.")

    # Determine number of envs from first reset
    obs_test, _ = env.reset(seed=0)
    multi_env = obs_test.ndim > 1
    num_envs = obs_test.shape[0] if multi_env else 1
    print(f"[diag_isaac_single] Detected {num_envs} env(s) (multi_env={multi_env})")

    # Helper to extract scalar from possibly-batched value (use env 0)
    def _scalar(x):
        x = np.asarray(x)
        return float(x.flat[0])

    zero_action = np.zeros(ACTION_DIM, dtype=np.float32)
    if multi_env:
        zero_action = np.zeros((num_envs, ACTION_DIM), dtype=np.float32)
    min_h_seen = float("inf")
    contacted_ground = False
    episodes_contacted = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        obs0 = obs[0] if multi_env else obs
        print(f"\n[diag_isaac_single] Episode {ep + 1}/{args.episodes}  "
              f"initial h_agl={float(obs0[OBS_H_AGL]):.2f} m")

        for step in range(args.steps):
            obs, rew, terminated, truncated, _ = env.step(zero_action)
            obs0 = obs[0] if multi_env else obs
            h_agl = float(obs0[OBS_H_AGL])
            min_h_seen = min(min_h_seen, h_agl)

            if h_agl < 0.1:
                contacted_ground = True

            if step % 60 == 0 or step == args.steps - 1:
                print(
                    f"  step={step:4d}  h_agl={h_agl:.3f} m  "
                    f"reward={_scalar(rew):.3f}  done={bool(_scalar(terminated)) or bool(_scalar(truncated))}"
                )

            # For multi-env, check env 0 for termination
            if _scalar(terminated) or _scalar(truncated):
                print(f"  Episode ended at step {step} "
                      f"(terminated={bool(_scalar(terminated))}, truncated={bool(_scalar(truncated))})")
                break

        if contacted_ground:
            episodes_contacted += 1
            contacted_ground = False  # reset per-episode flag

    print(f"\n[diag_isaac_single] Completed {args.episodes} episodes.")
    print(f"[diag_isaac_single] Min h_agl observed across all episodes: {min_h_seen:.3f} m")
    print(f"[diag_isaac_single] Episodes with ground contact: {episodes_contacted}/{args.episodes}")

    input("\n[diag_isaac_single] Press Enter to close...")
    env.close()
    if _SIM_APP is not None:
        _SIM_APP.close()

    if episodes_contacted > 0:
        print("[diag_isaac_single] PASS — Drone contacted ground in at least one episode.")
    else:
        print("[diag_isaac_single] WARNING — Drone did not contact ground in any episode. "
              "Check gravity / physics config.")
        sys.exit(1)


if __name__ == "__main__":
    main()
