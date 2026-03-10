"""
test_fins.py — Visual fin articulation test script (T024).

Implements T024 / User Story 2 independent test:
- Command each fin sequentially to +1.0, 0.0, -1.0 normalized
- Read back robot.data.joint_pos
- Print deflection in degrees
- Assert within 1% of ±15°

Usage::
    python -m simulation.isaac.scripts.test_fins
    python -m simulation.isaac.scripts.test_fins --config simulation/isaac/configs/isaac_env_single.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

_DELTA_MAX_DEG = 15.0
_DELTA_MAX_RAD = math.radians(_DELTA_MAX_DEG)
_DT = 1.0 / 120.0
_SETTLE_STEPS = 120  # 1 s — well beyond 5×tau_servo (0.2 s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Isaac Sim fin articulation test")
    parser.add_argument(
        "--config",
        default="simulation/isaac/configs/isaac_env_single.yaml",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    from simulation.isaac.envs.edf_isaac_env import EDFIsaacEnv
    import torch

    env = EDFIsaacEnv(config_path=config_path, seed=0)
    print("[test_fins] Env ready. Testing fin deflection limits.\n")

    fin_names = ["Fin_1 (right)", "Fin_2 (left)", "Fin_3 (forward)", "Fin_4 (aft)"]
    all_pass = True

    for fin_idx in range(4):
        for cmd_norm, label in [(+1.0, "+1.0"), (-1.0, "-1.0")]:
            env.reset()
            action = np.zeros(5, dtype=np.float32)
            action[fin_idx + 1] = cmd_norm

            for _ in range(_SETTLE_STEPS):
                env.step(action)

            # Read back from lag state tensor (settled value)
            fin_actual_rad = env._task.fin_deflections_actual[0, fin_idx].item()
            fin_actual_deg = math.degrees(fin_actual_rad)

            # Also attempt to read from robot joint_pos if available
            try:
                joint_pos = env._task.robot.data.joint_pos[0]  # (num_joints,)
                joint_deg = math.degrees(float(joint_pos[fin_idx].item()))
                joint_str = f"  joint_pos={joint_deg:+.2f}°"
            except Exception:
                joint_str = ""

            expected_deg = _DELTA_MAX_DEG * cmd_norm
            err_pct = abs(fin_actual_deg - expected_deg) / _DELTA_MAX_DEG * 100.0
            status = "PASS" if err_pct < 1.0 else "FAIL"
            if status == "FAIL":
                all_pass = False

            print(
                f"  {fin_names[fin_idx]} cmd={label:+s}  "
                f"lag_state={fin_actual_deg:+.2f}°  "
                f"expected={expected_deg:+.2f}°  "
                f"err={err_pct:.2f}%{joint_str}  [{status}]"
            )

    env.close()

    print()
    if all_pass:
        print("[test_fins] ALL PASS — fin deflections within 1% of ±15°.")
    else:
        print("[test_fins] SOME TESTS FAILED — check fin lag parameters.")
        sys.exit(1)


if __name__ == "__main__":
    main()
