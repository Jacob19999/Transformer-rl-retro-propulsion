"""
diag_yaw_isaac.py — Yaw torque diagnostic for Isaac Sim env (T040).

Equivalent of diag_yaw.py for the Isaac Sim environment.
Steps with a pure yaw torque input and verifies angular response.

Usage::
    python -m simulation.isaac.scripts.diag_yaw_isaac
    python -m simulation.isaac.scripts.diag_yaw_isaac --config simulation/isaac/configs/isaac_env_single.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from simulation.isaac.conventions import OBS_H_AGL, OBS_OMEGA_Z, yaw_fin_command  # noqa: E402
from simulation.isaac.scripts._shared import (  # noqa: E402
    create_sim_app,
    make_action,
    resolve_repo_path,
)

_SIM_APP = None

_YAW_FIN_ACTION = make_action(0.3, yaw_fin_command(0.5))
# Differential yaw fin pattern with partial thrust to stay airborne.


def main() -> None:
    parser = argparse.ArgumentParser(description="Isaac Sim yaw torque diagnostic")
    parser.add_argument(
        "--config",
        default="simulation/isaac/configs/isaac_env_single.yaml",
    )
    parser.add_argument("--steps", type=int, default=300, help="Steps to simulate")
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config)

    global _SIM_APP
    _SIM_APP = create_sim_app(headless=False)

    from simulation.isaac.envs.edf_isaac_env import EDFIsaacEnv

    env = EDFIsaacEnv(config_path=config_path, render_mode="human")
    obs, _ = env.reset(seed=0)

    print(f"[diag_yaw_isaac] Initial h_agl = {obs[OBS_H_AGL]:.2f} m")
    print(f"[diag_yaw_isaac] Applying yaw action: {_YAW_FIN_ACTION}")
    print(f"[diag_yaw_isaac] Monitoring omega_z (body yaw rate)...\n")

    yaw_rates = []
    for step in range(args.steps):
        obs, rew, term, trunc, info = env.step(_YAW_FIN_ACTION)
        omega_z = float(obs[OBS_OMEGA_Z])
        yaw_rates.append(omega_z)

        if step % 30 == 0:
            print(
                f"  step={step:3d}  h_agl={obs[OBS_H_AGL]:.3f} m  "
                f"omega_z={omega_z:+.3f} rad/s  reward={rew:.3f}"
            )

        if term or trunc:
            print(f"[diag_yaw_isaac] Episode ended at step {step}.")
            break

    input("\n[diag_yaw_isaac] Press Enter to close...")
    env.close()
    if _SIM_APP is not None:
        _SIM_APP.close()

    max_yaw_rate = max(abs(r) for r in yaw_rates)
    mean_yaw_rate = np.mean([abs(r) for r in yaw_rates])
    print(f"\n[diag_yaw_isaac] Max |omega_z|  : {max_yaw_rate:.4f} rad/s")
    print(f"[diag_yaw_isaac] Mean |omega_z| : {mean_yaw_rate:.4f} rad/s")

    if max_yaw_rate > 0.01:
        print("[diag_yaw_isaac] PASS — Yaw angular response detected.")
    else:
        print("[diag_yaw_isaac] WARNING — No significant yaw response. "
              "Check fin force model and joint configuration.")
        sys.exit(1)


if __name__ == "__main__":
    main()
