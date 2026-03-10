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

# SimulationApp MUST be created before any isaaclab.sim / carb imports
from isaacsim import SimulationApp  # noqa: E402

_SIM_APP: SimulationApp | None = None


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

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    print(f"[diag_isaac_single] Loading env from: {config_path}")

    # Launch Isaac Sim app before importing any isaaclab/carb modules
    global _SIM_APP
    _SIM_APP = SimulationApp({"headless": False})

    from simulation.isaac.envs.edf_isaac_env import EDFIsaacEnv

    env = EDFIsaacEnv(config_path=config_path, render_mode="human")

    print(f"[diag_isaac_single] Running {args.episodes} episodes, up to {args.steps} steps each.")

    zero_action = np.zeros(5, dtype=np.float32)
    min_h_seen = float("inf")
    contacted_ground = False
    episodes_contacted = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        print(f"\n[diag_isaac_single] Episode {ep + 1}/{args.episodes}  "
              f"initial h_agl={obs[16]:.2f} m")

        for step in range(args.steps):
            obs, rew, terminated, truncated, _ = env.step(zero_action)
            h_agl = float(obs[16])
            min_h_seen = min(min_h_seen, h_agl)

            if h_agl < 0.1:
                contacted_ground = True

            if step % 60 == 0 or step == args.steps - 1:
                print(
                    f"  step={step:4d}  h_agl={h_agl:.3f} m  "
                    f"reward={rew:.3f}  done={terminated or truncated}"
                )

            if terminated or truncated:
                print(f"  Episode ended at step {step} "
                      f"(terminated={terminated}, truncated={truncated})")
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
