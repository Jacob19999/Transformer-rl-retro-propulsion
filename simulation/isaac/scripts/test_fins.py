"""
test_fins.py — Visual fin sweep test at multiple speeds.

Drone is held stationary (gravity disabled, zero velocity, low altitude).
For each of 4 fins individually, then all 4 together, sweeps from -1 to +1
at 5 different speeds. Repeats for 10 epochs. Simulation runs in real time.

Usage::
    python -m simulation.isaac.scripts.test_fins
    python -m simulation.isaac.scripts.test_fins --config simulation/isaac/configs/isaac_env_single.yaml
    python -m simulation.isaac.scripts.test_fins --epochs 5
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from simulation.isaac.conventions import ACTION_DIM, FIN_DISPLAY_NAMES  # noqa: E402
from simulation.isaac.scripts._shared import create_sim_app, disable_gravity, resolve_repo_path  # noqa: E402

_SIM_APP = None

FIN_NAMES = list(FIN_DISPLAY_NAMES)
_DT = 1.0 / 120.0

# 5 sweep speeds: duration in seconds for a full min-to-max ramp
SWEEP_DURATIONS_S = [2.0, 1.0, 0.5, 0.25, 0.125]
SWEEP_LABELS = ["very slow (2.0s)", "slow (1.0s)", "medium (0.5s)", "fast (0.25s)", "very fast (0.125s)"]

# Settle period between sweeps (seconds)
_SETTLE_S = 0.5

def build_sweep(duration_s: float, fin_indices: list[int]) -> list[np.ndarray]:
    """Build action sequence for a min-to-max sweep on given fin indices.

    Returns list of action arrays. Sweep goes -1 -> +1 linearly over duration_s.
    """
    n_steps = max(int(round(duration_s / _DT)), 1)
    actions = []
    for i in range(n_steps):
        t = i / max(n_steps - 1, 1)  # 0..1
        cmd = -1.0 + 2.0 * t          # -1..+1
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        for fi in fin_indices:
            action[fi + 1] = cmd
        actions.append(action)
    return actions


def build_settle() -> list[np.ndarray]:
    """Build a settle-to-zero sequence."""
    n_steps = max(int(round(_SETTLE_S / _DT)), 1)
    return [np.zeros(ACTION_DIM, dtype=np.float32) for _ in range(n_steps)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Isaac Sim fin sweep test")
    parser.add_argument(
        "--config",
        default="simulation/isaac/configs/isaac_env_single.yaml",
        help="Path to Isaac env YAML config",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of epochs to repeat (default: 10)",
    )
    parser.add_argument(
        "--spawn-altitude", type=float, default=0.4,
        help="Drone spawn altitude in metres (default: 0.4)",
    )
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config)

    global _SIM_APP
    _SIM_APP = create_sim_app(headless=False)

    from simulation.isaac.envs.edf_isaac_env import EDFIsaacEnv

    env = EDFIsaacEnv(config_path=config_path, render_mode="human", seed=0)

    # Pin drone in place: low altitude, zero velocity
    env._task.cfg.spawn_altitude_min = args.spawn_altitude
    env._task.cfg.spawn_altitude_max = args.spawn_altitude
    env._task.cfg.spawn_vel_mag_min = 0.0
    env._task.cfg.spawn_vel_mag_max = 0.0

    disable_gravity(env, prefix="test_fins")

    # Pre-build action sequences for one epoch
    # Structure: for each fin individually (0-3), then all 4 together,
    #            sweep min->max at each of the 5 speeds
    epoch_actions: list[np.ndarray] = []
    epoch_labels: list[str] = []

    fin_groups = [(i, FIN_NAMES[i]) for i in range(4)]
    fin_groups.append(("all", "All 4 fins"))

    for group_id, group_name in fin_groups:
        if group_id == "all":
            fin_indices = [0, 1, 2, 3]
        else:
            fin_indices = [group_id]

        for speed_idx, (dur, speed_label) in enumerate(zip(SWEEP_DURATIONS_S, SWEEP_LABELS)):
            label = f"{group_name} — {speed_label}"

            # Sweep -1 -> +1
            sweep = build_sweep(dur, fin_indices)
            epoch_actions.extend(sweep)
            epoch_labels.extend([label] * len(sweep))

            # Settle back to zero
            settle = build_settle()
            epoch_actions.extend(settle)
            epoch_labels.extend(["settle"] * len(settle))

    total_steps = len(epoch_actions)
    epoch_duration_s = total_steps * _DT

    print(f"\n[test_fins] Config:       {config_path}")
    print(f"[test_fins] Epochs:       {args.epochs}")
    print(f"[test_fins] Steps/epoch:  {total_steps}  ({epoch_duration_s:.1f}s)")
    print(f"[test_fins] Sweep speeds: {', '.join(SWEEP_LABELS)}")
    print(f"[test_fins] Fin groups:   4 individual + all-4-together")
    print(f"[test_fins] Running in real time...\n")

    for epoch in range(args.epochs):
        env.reset(seed=epoch)
        print(f"[test_fins] === Epoch {epoch + 1:2d}/{args.epochs} ===")

        prev_label = ""
        for step_idx, (action, label) in enumerate(zip(epoch_actions, epoch_labels)):
            wall_start = time.perf_counter()

            env.step(action)

            # Print phase transitions
            if label != prev_label:
                fins_str = "  ".join(
                    f"{FIN_NAMES[i].split()[0]}={action[i+1]:+.2f}"
                    for i in range(4)
                )
                print(f"  step {step_idx:5d}  {label:<40s}  [{fins_str}]")
                prev_label = label

            # Real-time pacing: sleep for remainder of timestep
            elapsed = time.perf_counter() - wall_start
            sleep_time = _DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print(f"  [done] epoch {epoch + 1} complete\n")

    input("[test_fins] All epochs finished. Press Enter to close...")
    env.close()
    if _SIM_APP is not None:
        _SIM_APP.close()


if __name__ == "__main__":
    main()
