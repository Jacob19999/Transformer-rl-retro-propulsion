"""
diag_fin_forces.py — Per-fin force vector & magnitude gizmo diagnostic.

Spawns a single drone (gravity disabled, position locked at 1 m) and applies
configurable thrust + fin deflections.  A ``ForceGizmoDrawer`` renders
colour-coded arrows at each fin's COM in the viewport while per-fin force
magnitudes and world-frame vectors are printed to the console every
``--print-interval`` steps.

Arrow colour key (matches ``debug_draw.py``):
    Fin_1 (right)   — Red
    Fin_2 (left)    — Green
    Fin_3 (forward) — Blue
    Fin_4 (aft)     — Yellow
    Thrust          — Cyan
    Body torque     — Orange
    Dot at tip      — same colour, size ∝ force magnitude

Usage::
    python -m simulation.isaac.scripts.diag_fin_forces
    python -m simulation.isaac.scripts.diag_fin_forces --thrust 0.7 --fins 0.5 0.0 -0.5 0.0
    python -m simulation.isaac.scripts.diag_fin_forces --sweep --sweep-hz 0.5 --duration 10.0
    python -m simulation.isaac.scripts.diag_fin_forces --thrust 0.7 --fins 0.3 0.3 0.3 0.3 --print-interval 12
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from simulation.isaac.conventions import (  # noqa: E402
    ACTION_DIM,
    FIN_DISPLAY_NAMES,
    FRD_BODY_FRAME_TEXT,
)
from simulation.isaac.scripts._shared import (  # noqa: E402
    create_sim_app,
    disable_gravity,
    lock_position_at_altitude,
    make_action,
    resolve_repo_path,
)

_SIM_APP = None

FIN_NAMES = list(FIN_DISPLAY_NAMES)
_DT = 1.0 / 120.0  # Isaac Sim physics step


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-fin force gizmo diagnostic (Isaac Sim viewport)",
    )
    parser.add_argument(
        "--config",
        default="simulation/isaac/configs/isaac_env_single.yaml",
        help="Path to Isaac env YAML config",
    )
    parser.add_argument(
        "--thrust",
        type=float,
        default=0.7,
        help="Normalised thrust command [0, 1] (default: 0.7 ≈ hover)",
    )
    parser.add_argument(
        "--fins",
        type=float,
        nargs=4,
        default=[0.0, 0.0, 0.0, 0.0],
        metavar=("F1", "F2", "F3", "F4"),
        help="Static fin deflection commands in [-1, 1] (default: all zero)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        default=False,
        help="Sine-sweep all 4 fins together instead of holding static",
    )
    parser.add_argument(
        "--sweep-hz",
        type=float,
        default=0.5,
        help="Sweep frequency in Hz (default: 0.5)",
    )
    parser.add_argument(
        "--sweep-amplitude",
        type=float,
        default=1.0,
        help="Sweep amplitude in [-1, 1] (default: 1.0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Run duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--print-interval",
        type=int,
        default=24,
        help="Print magnitudes every N steps (default: 24 = 5× per second)",
    )
    parser.add_argument(
        "--force-scale",
        type=float,
        default=0.25,
        help="Arrow scale: metres per Newton (default: 0.25)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to run (default: 100)",
    )
    parser.add_argument(
        "--disable-gyro",
        action="store_true",
        default=False,
        help="Disable gyro-precession torque",
    )
    parser.add_argument(
        "--disable-anti-torque",
        action="store_true",
        default=False,
        help="Disable EDF anti-torque",
    )
    parser.add_argument(
        "--disable-wind",
        action="store_true",
        default=False,
        help="Disable wind disturbance model",
    )
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config)
    total_steps = max(1, round(args.duration / _DT))

    global _SIM_APP
    _SIM_APP = create_sim_app(headless=False)

    from simulation.isaac.envs.edf_isaac_env import EDFIsaacEnv  # noqa: E402

    env = EDFIsaacEnv(
        config_path=config_path,
        render_mode="human",
        debug_draw_forces=True,
    )

    # Lock drone in space so we can observe pure fin forces
    env._task.cfg.spawn_altitude_min = 1.0
    env._task.cfg.spawn_altitude_max = 1.0
    env._task.cfg.spawn_vel_mag_min = 0.0
    env._task.cfg.spawn_vel_mag_max = 0.0
    disable_gravity(env, prefix="fin_forces")

    # Runtime overrides
    if hasattr(env, "_task"):
        try:
            env._task.set_runtime_overrides(
                disable_wind=args.disable_wind,
                disable_gyro=args.disable_gyro,
                disable_anti_torque=args.disable_anti_torque,
                disable_gravity=False,
            )
        except Exception as exc:
            print(f"[fin_forces] WARNING: could not apply runtime overrides: {exc}")

    # Get the gizmo reference for console readout
    gizmo = env._task._force_gizmo

    # Print header
    print(f"\n[fin_forces] Body frame: {FRD_BODY_FRAME_TEXT}")
    print(f"[fin_forces] Config:     {config_path}")
    print(f"[fin_forces] Thrust:     {args.thrust:.2f}")
    if args.sweep:
        print(f"[fin_forces] Mode:       SINE SWEEP  {args.sweep_hz:.2f} Hz  amp={args.sweep_amplitude:.2f}")
    else:
        fins_str = "  ".join(f"{FIN_NAMES[i]}={args.fins[i]:+.3f}" for i in range(4))
        print(f"[fin_forces] Fins:       {fins_str}")
    print(f"[fin_forces] Duration:   {args.duration:.1f} s  ({total_steps} steps)")
    print(f"[fin_forces] Gizmo:      arrows + magnitude dots (scale={args.force_scale} m/N)")
    if args.disable_gyro:
        print("[fin_forces] Gyro:       DISABLED")
    if args.disable_anti_torque:
        print("[fin_forces] Anti-torque: DISABLED")
    if args.disable_wind:
        print("[fin_forces] Wind:       DISABLED")

    print(f"\n[fin_forces] Colour key:")
    print(f"  Fin_1 (right)   — Red")
    print(f"  Fin_2 (left)    — Green")
    print(f"  Fin_3 (forward) — Blue")
    print(f"  Fin_4 (aft)     — Yellow")
    print(f"  Thrust          — Cyan")
    print(f"  Body torque     — Orange")
    print(f"\n[fin_forces] Running {args.episodes} episode(s)...\n")

    for ep in range(args.episodes):
        env.reset(seed=ep)
        lock_position_at_altitude(env, altitude_m=1.0)

        print(f"{'='*78}")
        print(f"  EPISODE {ep + 1}/{args.episodes}")
        print(f"{'='*78}")
        print(
            f"  {'step':>5s}  {'t(s)':>5s}  "
            f"{'Fin1_R(N)':>9s}  {'Fin2_L(N)':>9s}  "
            f"{'Fin3_F(N)':>9s}  {'Fin4_A(N)':>9s}  "
            f"{'Thrust(N)':>9s}  {'Torque':>9s}"
        )
        print(f"  {'-'*72}")

        for step in range(total_steps):
            # Build action
            if args.sweep:
                t = step * _DT
                v = args.sweep_amplitude * math.sin(2.0 * math.pi * args.sweep_hz * t)
                fin_cmds = (v, v, v, v)
            else:
                fin_cmds = tuple(args.fins)

            action = make_action(args.thrust, fin_cmds)
            env.step(action)
            lock_position_at_altitude(env, altitude_m=1.0)

            # Print magnitudes
            if gizmo is not None and step % args.print_interval == 0:
                t_s = step * _DT
                mags = gizmo.fin_magnitudes
                if mags and len(mags) > 0:
                    fm = mags[0]  # env 0
                    thrust_m = gizmo.thrust_magnitudes[0] if gizmo.thrust_magnitudes else 0.0
                    torque_m = gizmo.torque_magnitudes[0] if gizmo.torque_magnitudes else 0.0
                    print(
                        f"  {step:5d}  {t_s:5.2f}  "
                        f"{fm[0]:9.4f}  {fm[1]:9.4f}  "
                        f"{fm[2]:9.4f}  {fm[3]:9.4f}  "
                        f"{thrust_m:9.4f}  {torque_m:9.5f}"
                    )

        # Print final detailed vectors
        if gizmo is not None and gizmo.fin_vectors:
            vecs = gizmo.fin_vectors[0]
            mags = gizmo.fin_magnitudes[0]
            print(f"\n  Final per-fin force vectors (world frame):")
            for j in range(4):
                fx, fy, fz = vecs[j]
                print(
                    f"    {FIN_NAMES[j]:20s}  "
                    f"|F| = {mags[j]:.4f} N  "
                    f"F = ({fx:+.4f}, {fy:+.4f}, {fz:+.4f}) N"
                )

        print(f"\n  [done] episode {ep + 1} complete\n")

    input("[fin_forces] All episodes finished. Press Enter to close...")
    env.close()
    if _SIM_APP is not None:
        _SIM_APP.close()


if __name__ == "__main__":
    main()
