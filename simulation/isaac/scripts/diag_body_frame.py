"""
diag_body_frame.py — Direct angular-velocity injection to validate FRD body-frame convention.

WHAT THIS DOES
--------------
Gravity disabled.  Drone spawned at fixed altitude, orientation locked to identity.
For each body axis (X → Y → Z) and each sign (positive → negative):

  1. Reset drone to identity orientation, zero velocity.
  2. Print a clear banner: which axis, which sign, what visual motion to expect.
  3. Each simulation step: compute world-frame angular velocity = R(q) · omega_body,
     write it directly into the simulation via write_root_velocity_to_sim.
     No fin deflections, no aerodynamics, no forces — purely a velocity override.
  4. Also print the OBSERVED omega from the observation vector every _PRINT_EVERY steps
     so you can confirm the obs indices match the commanded axis.

WHAT TO LOOK FOR
----------------
Body frame (FRD): +X = forward/nose,  +Y = right,  +Z = down

  ωx (ROLL+)  → drone should roll clockwise when viewed from behind the nose
  ωy (PITCH+) → drone nose should pitch DOWN (FRD +Z is down, so +τ_y pitches nose toward +Z)
  ωz (YAW+)   → drone should yaw clockwise when viewed from above (nose sweeps right)

If ROLL+ visually looks like pitching → X/Y axes are swapped in the body frame.
If the sign is inverted   → the axis sign convention differs from FRD.

Usage::
    python -m simulation.isaac.scripts.diag_body_frame
    python -m simulation.isaac.scripts.diag_body_frame --omega 1.0 --hold-secs 3.0
    python -m simulation.isaac.scripts.diag_body_frame --headless   # no viewer (unit test)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from simulation.isaac.conventions import ACTION_DIM, OBS_H_AGL, OBS_OMEGA_X, OBS_OMEGA_Y, OBS_OMEGA_Z  # noqa: E402
from simulation.isaac.quaternion_isaac import (  # noqa: E402
    identity_quat_wxyz,
    rotate_body_to_world_wxyz,
)
from simulation.isaac.scripts._shared import create_sim_app, disable_gravity, obs_scalar, resolve_repo_path  # noqa: E402

_SIM_APP = None

_PRINT_EVERY  = 30    # print obs omega every 0.25 s during hold

# ──────────────────────────────────────────────────────────────────────────────
# Axis descriptors: what the drone should look like visually for each command
# ──────────────────────────────────────────────────────────────────────────────
_AXIS_INFO = {
    #  axis_idx  sign   banner label         obs_idx      visual description
    "X+": (0, +1.0, "ωx = +omega  [ROLL+]",  _OBS_OMEGA_X,
           "drone rolls CW viewed from BEHIND the nose  (right wing drops)"),
    "X-": (0, -1.0, "ωx = -omega  [ROLL-]",  _OBS_OMEGA_X,
           "drone rolls CCW viewed from BEHIND the nose (left wing drops)"),
    "Y+": (1, +1.0, "ωy = +omega  [PITCH+]", _OBS_OMEGA_Y,
           "drone nose pitches DOWN  (FRD +Z=down, so +ωy tilts nose toward ground)"),
    "Y-": (1, -1.0, "ωy = -omega  [PITCH-]", _OBS_OMEGA_Y,
           "drone nose pitches UP"),
    "Z+": (2, +1.0, "ωz = +omega  [YAW+]",   _OBS_OMEGA_Z,
           "drone yaws CW viewed from ABOVE  (nose sweeps right)"),
    "Z-": (2, -1.0, "ωz = -omega  [YAW-]",   _OBS_OMEGA_Z,
           "drone yaws CCW viewed from ABOVE (nose sweeps left)"),
}

_SEQUENCE = ["X+", "X-", "Y+", "Y-", "Z+", "Z-"]

def _reset_to_identity(task, altitude_m: float = 3.0) -> None:
    """Reset drone to identity orientation, zero velocity, at given altitude."""
    import torch
    pos_w  = task.robot.data.root_pos_w.clone()
    pos_w[:, 2] = altitude_m
    quat_w = identity_quat_wxyz(pos_w.shape[0], device=pos_w.device)
    task.robot.write_root_pose_to_sim(torch.cat([pos_w, quat_w], dim=-1))

    vel_w = torch.zeros(pos_w.shape[0], 3, device=pos_w.device)
    ang_w = torch.zeros(pos_w.shape[0], 3, device=pos_w.device)
    task.robot.write_root_velocity_to_sim(torch.cat([vel_w, ang_w], dim=-1))


def _set_body_omega(task, omega_body: list[float]) -> None:
    """Write a constant body-frame angular velocity into the simulation each step.

    Transforms omega_body → world frame using the current drone quaternion, then
    calls write_root_velocity_to_sim so the drone actually rotates.
    Position is kept at current world position; linear velocity is zeroed.
    """
    import torch

    pos_w  = task.robot.data.root_pos_w.clone()         # (N, 3)
    quat_w = task.robot.data.root_quat_w.clone()        # (N, 4) IsaacLab wxyz [qw,qx,qy,qz]

    device = pos_w.device
    N = pos_w.shape[0]

    omega_b = torch.tensor(omega_body, dtype=torch.float32, device=device)  # (3,)

    # omega_world = R(q) · omega_body  (body → world)
    omega_w = rotate_body_to_world_wxyz(quat_w, omega_b.unsqueeze(0).expand(N, -1))

    lin_vel = torch.zeros_like(omega_w)
    task.robot.write_root_velocity_to_sim(torch.cat([lin_vel, omega_w], dim=-1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject angular velocity per body axis — visual body-frame validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Body frame (FRD): +X=forward/nose, +Y=right, +Z=down
  ωx+ = ROLL+  → right wing drops (CW from behind nose)
  ωy+ = PITCH+ → nose pitches down toward ground
  ωz+ = YAW+   → nose sweeps right (CW from above)
        """,
    )
    parser.add_argument(
        "--config",
        default="simulation/isaac/configs/isaac_env_single.yaml",
        help="Isaac env YAML config (default: isaac_env_single.yaml)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of times to repeat the full X/Y/Z sequence (default: 1)",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=1.0,
        help="Angular velocity magnitude in rad/s (default: 1.0)",
    )
    parser.add_argument(
        "--hold-secs",
        type=float,
        default=3.0,
        help="Hold duration per axis/sign in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--altitude",
        type=float,
        default=3.0,
        help="Spawn/hold altitude in metres (default: 3.0)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without GUI",
    )
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config)

    global _SIM_APP
    _SIM_APP = create_sim_app(headless=args.headless)

    from simulation.isaac.envs.edf_isaac_env import EDFIsaacEnv
    import torch

    env = EDFIsaacEnv(config_path=config_path, render_mode="human")
    task = env._task

    # Disable gravity and all aerodynamic / gyro effects for a clean test
    disable_gravity(env, prefix="body_frame")
    try:
        task._gyro_enabled = False
        print("[body_frame] Gyro precession: DISABLED")
    except Exception:
        pass
    try:
        task._wind_model = None
        print("[body_frame] Wind model:      DISABLED")
    except Exception:
        pass

    # Spawn at fixed altitude
    task.cfg.spawn_altitude_min = args.altitude
    task.cfg.spawn_altitude_max = args.altitude
    task.cfg.spawn_vel_mag_min = 0.0
    task.cfg.spawn_vel_mag_max = 0.0

    hold_steps = max(1, round(args.hold_secs * 120))
    settle_steps = 30  # 0.25 s settle between phases

    print(f"\n[body_frame] Config:      {config_path.name}")
    print(f"[body_frame] episodes:    {args.episodes}")
    print(f"[body_frame] omega:       ±{args.omega:.2f} rad/s per axis")
    print(f"[body_frame] hold:        {args.hold_secs:.1f} s ({hold_steps} steps) per direction")
    print(f"[body_frame] altitude:    {args.altitude:.1f} m (fixed)")
    print(f"\n[body_frame] Body frame (FRD): +X=nose/fwd  +Y=right  +Z=down")
    print(f"[body_frame] Expected:  ωx+=ROLL+  ωy+=PITCH+  ωz+=YAW+")
    print(f"\n[body_frame] Starting sequence: {' → '.join(_SEQUENCE)}\n")

    obs, _ = env.reset(seed=0)

    bar_wide  = "═" * 64
    bar_thin  = "─" * 64

    for ep in range(args.episodes):
        if args.episodes > 1:
            print(f"\n{'#'*64}")
            print(f"  EPISODE {ep + 1} / {args.episodes}")
            print(f"{'#'*64}")

        for key in _SEQUENCE:
            axis_idx, sign, label, obs_idx, visual_desc = _AXIS_INFO[key]

            omega_body = [0.0, 0.0, 0.0]
            omega_body[axis_idx] = sign * args.omega

            # ── Banner ──────────────────────────────────────────────────────
            print(f"\n  {bar_wide}")
            print(f"  ══  PHASE: {label}")
            print(f"      commanded omega_body = {omega_body}")
            print(f"      EXPECTED VISUAL: {visual_desc}")
            print(f"      obs check: obs[{obs_idx}] should read ≈ {sign * args.omega:+.2f} rad/s")
            print(f"  {bar_wide}")

            # Reset orientation to identity, zero velocity
            _reset_to_identity(task, altitude_m=args.altitude)
            env.step(np.zeros(ACTION_DIM, dtype=np.float32))  # flush reset into sim

            # Short settle with zero velocity
            for _ in range(settle_steps):
                _reset_to_identity(task, altitude_m=args.altitude)
                obs, _, done, _, _ = env.step(np.zeros(ACTION_DIM, dtype=np.float32))

            print(f"\n  {bar_thin}")
            print(f"  {'step':>6}  {'t(s)':>6}  {'obs ωx(roll)':>14}  "
                  f"{'obs ωy(pitch)':>14}  {'obs ωz(yaw)':>12}  {'h(m)':>7}")
            print(f"  {bar_thin}")

            # Hold: inject body-frame omega each step
            for step in range(hold_steps):
                _set_body_omega(task, omega_body)
                obs, _, done, _, _ = env.step(np.zeros(ACTION_DIM, dtype=np.float32))

                if step % _PRINT_EVERY == 0:
                    t_s   = step / 120.0
                    ox    = obs_scalar(obs, OBS_OMEGA_X)
                    oy    = obs_scalar(obs, OBS_OMEGA_Y)
                    oz    = obs_scalar(obs, OBS_OMEGA_Z)
                    h     = obs_scalar(obs, OBS_H_AGL)
                    ox_s = f">>> {ox:+.4f} <<<" if axis_idx == 0 else f"    {ox:+.4f}    "
                    oy_s = f">>> {oy:+.4f} <<<" if axis_idx == 1 else f"    {oy:+.4f}    "
                    oz_s = f">>> {oz:+.4f} <<<" if axis_idx == 2 else f"    {oz:+.4f}    "
                    print(f"  {step:>6}  {t_s:>6.2f}  {ox_s}  {oy_s}  {oz_s}  {h:>7.3f}")

            # End-of-phase summary
            ox = obs_scalar(obs, OBS_OMEGA_X)
            oy = obs_scalar(obs, OBS_OMEGA_Y)
            oz = obs_scalar(obs, OBS_OMEGA_Z)
            obs_commanded = [ox, oy, oz][axis_idx]
            match_str = "✓ MATCH" if (obs_commanded * sign) > (args.omega * 0.5) else "✗ MISMATCH"
            print(f"  {bar_thin}")
            print(f"  END: obs ωx={ox:+.4f}  ωy={oy:+.4f}  ωz={oz:+.4f} rad/s")
            print(f"  commanded axis obs[{obs_idx}] = {obs_commanded:+.4f} rad/s  →  {match_str}")

    print(f"\n{'='*64}")
    print("  ALL EPISODES COMPLETE")
    print(f"{'='*64}\n")

    env.close()
    if _SIM_APP is not None:
        _SIM_APP.close()


if __name__ == "__main__":
    main()
