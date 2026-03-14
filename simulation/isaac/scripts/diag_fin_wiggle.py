"""
diag_fin_wiggle.py — Visual fin wiggle diagnostic for Isaac Sim.

Places a single drone just above the ground (gravity disabled so it holds position)
and runs a repeating fin deflection sequence for 100 episodes:

  Per episode:
    1. Each fin individually: smooth sine sweep -1 → +1 → -1  (×N_SWEEPS)
    2. All four fins: hold full-min (-1), then full-max (+1)

Usage::
    python -m simulation.isaac.scripts.diag_fin_wiggle
    python -m simulation.isaac.scripts.diag_fin_wiggle --config simulation/isaac/configs/isaac_env_single.yaml
    python -m simulation.isaac.scripts.diag_fin_wiggle --episodes 50 --sweeps 2
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from simulation.isaac.conventions import ACTION_DIM, FIN_DISPLAY_NAMES  # noqa: E402
from simulation.isaac.scripts._shared import create_sim_app, disable_gravity, resolve_repo_path  # noqa: E402

_SIM_APP = None

FIN_NAMES = list(FIN_DISPLAY_NAMES)

# Timing (steps at 1/120 s each)
_STEPS_PER_SWEEP = 120   # 1 s per sine cycle
_STEPS_HOLD      = 60    # 0.5 s hold for all-fins phases
_STEPS_SETTLE    = 30    # 0.25 s settle / zero-out between phases


def build_episode_sequence(n_sweeps: int) -> tuple[list[np.ndarray], list[str]]:
    """Build (actions, phase_labels) for one episode."""
    actions: list[np.ndarray] = []
    labels:  list[str]        = []

    total_sweep_steps = _STEPS_PER_SWEEP * n_sweeps
    t = np.linspace(0.0, 2.0 * math.pi * n_sweeps, total_sweep_steps, endpoint=False)
    sweep = np.sin(t)  # smooth, continuous sine sweep

    # --- Phase 1-4: wiggle each fin individually ---
    for fin_idx in range(4):
        name = FIN_NAMES[fin_idx]

        for v in sweep:
            action = np.zeros(ACTION_DIM, dtype=np.float32)
            action[fin_idx + 1] = float(v)
            actions.append(action)
            labels.append(f"{name} sweep")

        # Settle to zero
        for _ in range(_STEPS_SETTLE):
            actions.append(np.zeros(ACTION_DIM, dtype=np.float32))
            labels.append("settle")

    # --- Phase 5: axis-aligned pairs (yaw, then pitch, then roll) ---

    # Yaw: all four fins in opposing pairs
    for v in sweep:
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[1] = float(+v)  # Fin_1 (right)
        action[2] = float(-v)  # Fin_2 (left)
        action[3] = float(+v)  # Fin_3 (forward)
        action[4] = float(-v)  # Fin_4 (aft)
        actions.append(action)
        labels.append("Yaw sweep (all 4 fins)")

    for _ in range(_STEPS_SETTLE):
        actions.append(np.zeros(ACTION_DIM, dtype=np.float32))
        labels.append("settle")

    # Pitch: side fins together
    for v in sweep:
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[1] = float(+v)  # Fin_1 (right)
        action[2] = float(+v)  # Fin_2 (left)
        actions.append(action)
        labels.append("Pitch sweep (Fin_1 & Fin_2)")

    for _ in range(_STEPS_SETTLE):
        actions.append(np.zeros(ACTION_DIM, dtype=np.float32))
        labels.append("settle")

    # Roll: forward + aft fins together
    for v in sweep:
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[3] = float(+v)  # Fin_3 (forward)
        action[4] = float(+v)  # Fin_4 (aft)
        actions.append(action)
        labels.append("Roll sweep (Fin_3 & Fin_4)")

    for _ in range(_STEPS_SETTLE):
        actions.append(np.zeros(ACTION_DIM, dtype=np.float32))
        labels.append("settle")

    # --- Phase 6: all fins full-min ---
    all_min = np.zeros(ACTION_DIM, dtype=np.float32)
    all_min[1:] = -1.0
    for _ in range(_STEPS_HOLD):
        actions.append(all_min.copy())
        labels.append("All fins -1.0 (full min)")

    # --- Phase 7: all fins full-max ---
    all_max = np.zeros(ACTION_DIM, dtype=np.float32)
    all_max[1:] = +1.0
    for _ in range(_STEPS_HOLD):
        actions.append(all_max.copy())
        labels.append("All fins +1.0 (full max)")

    # --- Final settle ---
    for _ in range(_STEPS_SETTLE):
        actions.append(np.zeros(ACTION_DIM, dtype=np.float32))
        labels.append("settle")

    return actions, labels

def main() -> None:
    parser = argparse.ArgumentParser(description="Isaac Sim fin wiggle diagnostic")
    parser.add_argument(
        "--config",
        default="simulation/isaac/configs/isaac_env_single.yaml",
        help="Path to Isaac env YAML config",
    )
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="Number of episodes to run (default: 100)",
    )
    parser.add_argument(
        "--sweeps", type=int, default=3,
        help="Sine sweeps per fin per episode (default: 3)",
    )
    parser.add_argument(
        "--spawn-altitude", type=float, default=0.4,
        help="Drone spawn altitude in metres (default: 0.4 — just above ground)",
    )
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config)

    global _SIM_APP
    _SIM_APP = create_sim_app(headless=False)

    from simulation.isaac.envs.edf_isaac_env import EDFIsaacEnv

    env = EDFIsaacEnv(config_path=config_path, render_mode="human")

    # Spawn drone just above the ground with zero initial velocity
    env._task.cfg.spawn_altitude_min = args.spawn_altitude
    env._task.cfg.spawn_altitude_max = args.spawn_altitude
    env._task.cfg.spawn_vel_mag_min  = 0.0
    env._task.cfg.spawn_vel_mag_max  = 0.0

    # Disable gravity so the drone floats in place during the test
    # (the done condition fires at h_agl < 0.05, which would cause constant resets)
    disable_gravity(env, prefix="fin_wiggle")

    # Build the fixed action sequence
    sequence, phase_labels = build_episode_sequence(args.sweeps)
    total_steps = len(sequence)
    episode_s   = total_steps / 120.0

    print(f"\n[fin_wiggle] Config:   {config_path}")
    print(f"[fin_wiggle] Sequence: {total_steps} steps  ({episode_s:.1f} s per episode)")
    print(f"[fin_wiggle] Fins:     {', '.join(FIN_NAMES)}")
    print(f"[fin_wiggle] Sweeps/fin: {args.sweeps}  |  Hold steps: {_STEPS_HOLD}  |  "
          f"Settle steps: {_STEPS_SETTLE}")
    print(f"[fin_wiggle] Running {args.episodes} episodes...\n")

    for ep in range(args.episodes):
        env.reset(seed=ep)
        print(f"[fin_wiggle] === Episode {ep + 1:3d}/{args.episodes} ===")

        prev_label = ""
        for step, (action, label) in enumerate(zip(sequence, phase_labels)):
            env.step(action)

            # Print once at each phase transition
            if label != prev_label:
                fins_str = "  ".join(
                    f"{FIN_NAMES[i].split()[0]}={action[i+1]:+.2f}"
                    for i in range(4)
                )
                print(f"  step {step:4d}  {label:<30s}  [{fins_str}]")
                prev_label = label

        print(f"  [done]  episode {ep + 1} complete\n")

    input("[fin_wiggle] All episodes finished. Press Enter to close...")
    env.close()
    if _SIM_APP is not None:
        _SIM_APP.close()


if __name__ == "__main__":
    main()
