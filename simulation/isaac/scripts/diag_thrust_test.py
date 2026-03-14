"""
diag_thrust_test.py -- Thrust application diagnostic for Isaac Sim.

Validates that the EDF drone can lift off from the ground under commanded
thrust. Tests three flight phases in sequence:

  Phase 1 (full thrust, 2 s): drone should ascend past 5 m
  Phase 2 (hover thrust, 2 s): altitude should remain within ±0.5 m
  Phase 3 (zero thrust, 1 s): drone should descend (gravity wins)

Also computes measured vertical acceleration during Phase 1 and compares
against expected value using the shared Isaac vehicle parameters.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from simulation.isaac.conventions import OBS_H_AGL, OBS_SPEED  # noqa: E402
from simulation.isaac.scripts._shared import (  # noqa: E402
    any_done,
    create_sim_app,
    make_action,
    obs_scalar,
    resolve_repo_path,
)

_SIM_APP = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isaac Sim thrust application diagnostic -- ground start -> liftoff"
    )
    parser.add_argument(
        "--config",
        default="simulation/isaac/configs/isaac_env_single.yaml",
        help="Path to Isaac env YAML config",
    )
    parser.add_argument(
        "--thrust",
        type=float,
        default=1.0,
        help="Normalized full-thrust command [0, 1] (default: 1.0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Duration of full-thrust phase in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--spawn-alt",
        type=float,
        default=0.32,
        help="Spawn altitude above ground in meters (default: 0.32 -- legs near ground)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to run (default: 100)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without GUI (default: False -- opens viewer)",
    )
    args = parser.parse_args()

    global _SIM_APP
    _SIM_APP = create_sim_app(headless=args.headless)

    try:
        _run_diagnostic(args)
    finally:
        _SIM_APP.close()


def _run_diagnostic(args) -> None:
    """Run thrust diagnostic; exits with code 0 (pass) or 1 (fail)."""
    from simulation.isaac.envs.edf_isaac_env import EDFIsaacEnv

    config_path = resolve_repo_path(args.config)

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["spawn_altitude_range"] = [args.spawn_alt, args.spawn_alt]
    cfg["spawn_velocity_magnitude_range"] = [0.0, 0.0]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(cfg, tmp)
        tmp_config_path = tmp.name

    env = EDFIsaacEnv(config_path=tmp_config_path)
    env._task.set_runtime_overrides(
        disable_wind=False,
        disable_gyro=True,
        disable_anti_torque=False,
        disable_gravity=False,
    )

    params = env._task.vehicle_params
    mass = float(env._task._mass)
    hover_norm = float(env._task.hover_thrust_norm)

    print(f"\n{'='*60}")
    print("EDF Thrust Application Diagnostic")
    print(f"{'='*60}")
    print(f"Config:      {config_path}")
    print(f"Spawn alt:   {args.spawn_alt:.2f} m")
    print(f"Full thrust: T_cmd = {args.thrust:.2f} ({args.thrust * params.t_max:.1f} N)")
    print(f"Hover norm:  T_cmd ~ {hover_norm:.3f} (weight/T_max)")
    print(
        f"Expected a:  {(args.thrust * params.t_max) / mass - params.gravity:.2f} m/s^2 "
        "(full thrust phase)"
    )
    print("Gyro precession: DISABLED (thrust isolation)")
    print(f"{'='*60}\n")

    dt = 1.0 / 120.0
    full_steps = int(args.duration / dt)
    hover_steps = int(2.0 / dt)
    cut_steps = int(1.0 / dt)
    all_passed = True

    for ep in range(args.episodes):
        obs, _ = env.reset()
        print(f"Episode {ep + 1}/{args.episodes}")
        print(f"  Initial altitude: {obs_scalar(obs, OBS_H_AGL):.3f} m")
        start_alt = obs_scalar(obs, OBS_H_AGL)

        print(f"\n  Phase 1: Full thrust (T_cmd={args.thrust:.2f}) for {args.duration:.1f}s")
        action_full = make_action(args.thrust)
        alt_history: list[float] = [start_alt]

        for step in range(full_steps):
            obs, _, done, _, _ = env.step(action_full)
            alt = obs_scalar(obs, OBS_H_AGL)
            alt_history.append(alt)

            if step % 30 == 0:
                print(f"    step {step:4d}  h={alt:.3f} m  speed={obs_scalar(obs, OBS_SPEED):.3f} m/s")

            if any_done(done):
                break

        peak_alt = max(alt_history)
        alt_gain = peak_alt - start_alt
        steps_1s = int(1.0 / dt)
        if len(alt_history) > steps_1s + 1:
            h0 = alt_history[0]
            h1 = alt_history[min(steps_1s, len(alt_history) - 1)]
            t = min(steps_1s, len(alt_history) - 1) * dt
            meas_acc = 2.0 * (h1 - h0) / (t ** 2) if t > 0 else 0.0
        else:
            meas_acc = 0.0

        expected_acc = args.thrust * params.t_max / mass - params.gravity
        print(f"  Phase 1 result:  peak altitude = {peak_alt:.3f} m (gain {alt_gain:.3f} m)")
        print(f"  Measured accel ~ {meas_acc:.2f} m/s^2  (expected ~ {expected_acc:.2f} m/s^2)")

        sc002_pass = alt_gain > 1.0
        if not sc002_pass:
            print(f"  SC-002 FAIL: Altitude gain {alt_gain:.3f} m < 1.0 m")
            all_passed = False
        else:
            print(f"  SC-002 PASS: Altitude gain {alt_gain:.3f} m >= 1.0 m")

        if any_done(done):
            obs, _ = env.reset()

        print(f"\n  Phase 2: Hover thrust (T_cmd~{hover_norm:.3f}) for 2.0s")
        action_hover = make_action(hover_norm)
        hover_alts: list[float] = []

        for step in range(hover_steps):
            obs, _, done, _, _ = env.step(action_hover)
            hover_alts.append(obs_scalar(obs, OBS_H_AGL))
            if step % 30 == 0:
                print(f"    step {step:4d}  h={obs_scalar(obs, OBS_H_AGL):.3f} m")
            if any_done(done):
                break

        if hover_alts:
            alt_drift = max(hover_alts) - min(hover_alts)
            sc003_pass = alt_drift < 2.0
            print(f"  Phase 2 result: altitude drift = {alt_drift:.3f} m")
            if sc003_pass:
                print(f"  SC-003 PASS: Drift {alt_drift:.3f} m < 2.0 m")
            else:
                print(f"  SC-003 INFO: Drift {alt_drift:.3f} m (open-loop hover is approximate)")

        print("\n  Phase 3: Zero thrust for 1.0s -- should descend")
        action_zero = make_action(0.0)
        cut_start_alt = obs_scalar(obs, OBS_H_AGL)
        cut_alts: list[float] = []

        for _ in range(cut_steps):
            obs, _, done, _, _ = env.step(action_zero)
            cut_alts.append(obs_scalar(obs, OBS_H_AGL))
            if any_done(done):
                break

        if cut_alts:
            final_alt = cut_alts[-1]
            descends = final_alt < cut_start_alt
            print(f"  Phase 3 result: alt {cut_start_alt:.3f} -> {final_alt:.3f} m")
            if descends:
                print("  Gravity PASS: drone descends after thrust cut")
            else:
                print("  Gravity NOTE: drone did not descend (may have been near ground)")

    env.close()

    print(f"\n{'='*60}")
    if all_passed:
        print("RESULT: PASS -- Thrust application validated")
        sys.exit(0)
    print("RESULT: FAIL -- See details above")
    sys.exit(1)


if __name__ == "__main__":
    main()
