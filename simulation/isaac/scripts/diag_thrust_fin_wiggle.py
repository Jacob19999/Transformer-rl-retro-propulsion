"""
diag_thrust_fin_wiggle.py — Step-hold fin/torque axis diagnostic for Isaac Sim.

Places a single drone at fixed altitude (gravity disabled, position locked at 1 m,
rotation free) and sequences through yaw / roll / pitch step-holds to validate that
each fin pair produces torque on the expected body axis.

Per episode, for each axis (Yaw → Roll → Pitch):
  1. Settle + orientation reset.
  2. Hold fins at −max_deflection for hold_secs (negative-torque direction).
  3. Settle + orientation reset.
  4. Hold fins at +max_deflection for hold_secs (positive-torque direction).
  5. Settle + orientation reset.

Body frame: FRD (+X=fwd/nose, +Y=right, +Z=down)
  ωx = ROLL  (rotation about +X = forward/nose axis)
  ωy = PITCH (rotation about +Y = right/lateral axis)
  ωz = YAW   (rotation about +Z = down axis)

Fin physical layout (confirmed from Isaac Sim 3D model top-down view):
  Fin_1(fwd)    at FRD +X  →  hinge X, lift +X  →  τ_y → PITCH  (Fin_1+Fin_2)
  Fin_2(aft)    at FRD −X  →  hinge X, lift +X  →  τ_y → PITCH  (Fin_1+Fin_2)
  Fin_3(left)   at FRD −Y  →  hinge Y, lift +Y  →  τ_x → ROLL   (Fin_3+Fin_4)
  Fin_4(right)  at FRD +Y  →  hinge Y, lift +Y  →  τ_x → ROLL   (Fin_3+Fin_4)
Note: YAML fins_config names (fin_1_right, fin_3_forward) are INCORRECT vs the 3D model.

Expected axis responses:
  Yaw  (all 4 fins): ωz dominant  →  d1=∓v  d2=±v  d3=±v  d4=∓v
  Roll (Fin_3+Fin_4): ωx dominant →  d1=0   d2=0   d3=±v  d4=±v
  Pitch(Fin_1+Fin_2): ωy dominant →  d1=±v  d2=±v  d3=0   d4=0

Usage::
    python -m simulation.isaac.scripts.diag_thrust_fin_wiggle --fixed-altitude
    python -m simulation.isaac.scripts.diag_thrust_fin_wiggle --thrust 0.75 --max-deflection 0.5 --fixed-altitude
    python -m simulation.isaac.scripts.diag_thrust_fin_wiggle --hold-secs 2.0 --fixed-altitude
    python -m simulation.isaac.scripts.diag_thrust_fin_wiggle --fixed-altitude --disable-wind --disable-gyro --disable-anti-torque
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

# Physical drone body frame: FRD (+X=fwd/nose, +Y=right, +Z=down)
# Fin physical positions (confirmed from Isaac Sim 3D model top-down view):
#   Fin_1(fwd)  at FRD +X, Fin_2(aft)   at FRD -X  — hinge X, lift +X → τ_y → PITCH
#   Fin_3(left) at FRD -Y, Fin_4(right) at FRD +Y  — hinge Y, lift +Y → τ_x → ROLL
# NOTE: YAML fins_config names (fin_1_right, fin_3_forward) are INCORRECT;
#       the actual USD geometry is: Fin_1=nose, Fin_2=aft, Fin_3=left, Fin_4=right.
FIN_NAMES = ["Fin_1(fwd)", "Fin_2(aft)", "Fin_3(left)", "Fin_4(right)"]

# Timing (steps at 1/120 s each)
_STEPS_HOLD   = 120   # 1.0 s hold per direction (overridden by --hold-secs)
_STEPS_SETTLE = 30    # 0.25 s settle + orientation reset between phases
_PRINT_EVERY  = 12    # print omega every 0.1 s during hold phases

# Observation indices (match edf_landing_task.py observation layout)
# Body frame: FRD (+X=fwd/nose, +Y=right, +Z=down)
_OBS_OMEGA_X = 9     # body angular rate about X → ROLL  (forward/nose spin axis)
_OBS_OMEGA_Y = 10    # body angular rate about Y → PITCH (right/lateral tilt axis)
_OBS_OMEGA_Z = 11    # body angular rate about Z → YAW   (vertical spin axis)
_OBS_ALTITUDE = 16   # h_agl (m)
_OBS_SPEED = 17      # |v_body| (m/s)

# Hold phase labels and their expected dominant axis + sign
HOLD_LABELS = {"Yaw-", "Yaw+", "Roll-", "Roll+", "Pitch-", "Pitch+"}
_EXPECTED = {
    "Yaw-":   ("ωz(yaw)",    "ωz < 0"),
    "Yaw+":   ("ωz(yaw)",    "ωz > 0"),
    "Roll-":  ("ωx(roll)",   "ωx < 0"),
    "Roll+":  ("ωx(roll)",   "ωx > 0"),
    "Pitch-": ("ωy(pitch)",  "ωy < 0"),
    "Pitch+": ("ωy(pitch)",  "ωy > 0"),
}


def _disable_gravity(env) -> None:
    """Zero gravity so thrust-only hover is easier to interpret."""
    try:
        from pxr import UsdPhysics
        stage = env._task.sim.stage
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                UsdPhysics.Scene(prim).GetGravityMagnitudeAttr().Set(0.0)
                print("[thrust_fin_wiggle] Gravity disabled — drone will not fall under weight.")
                return
        print("[thrust_fin_wiggle] WARNING: Physics scene not found; gravity still active.")
    except Exception as exc:
        print(f"[thrust_fin_wiggle] WARNING: Could not disable gravity: {exc}")


def _lock_position_at_altitude(env, altitude_m: float = 1.0) -> None:
    """Lock drone world position at given altitude, leaving rotation free."""
    task = getattr(env, "_task", None)
    if task is None or not hasattr(task, "robot"):
        return

    import torch

    pos_w  = task.robot.data.root_pos_w.clone()
    quat_w = task.robot.data.root_quat_w.clone()
    pos_w[:, 2] = altitude_m
    task.robot.write_root_pose_to_sim(torch.cat([pos_w, quat_w], dim=-1))

    vel_w = task.robot.data.root_lin_vel_w.clone()
    ang_w = task.robot.data.root_ang_vel_w.clone()
    vel_w[:] = 0.0
    task.robot.write_root_velocity_to_sim(torch.cat([vel_w, ang_w], dim=-1))


def _reset_orientation(env) -> None:
    """Reset drone orientation to identity and zero angular velocity, keep position."""
    task = getattr(env, "_task", None)
    if task is None or not hasattr(task, "robot"):
        return

    import torch
    from simulation.isaac.quaternion_isaac import identity_quat_wxyz

    pos_w  = task.robot.data.root_pos_w.clone()
    quat_w = identity_quat_wxyz(pos_w.shape[0], device=pos_w.device)
    task.robot.write_root_pose_to_sim(torch.cat([pos_w, quat_w], dim=-1))

    vel_w = task.robot.data.root_lin_vel_w.clone()
    ang_w = task.robot.data.root_ang_vel_w.clone()
    vel_w[:] = 0.0
    ang_w[:] = 0.0
    task.robot.write_root_velocity_to_sim(torch.cat([vel_w, ang_w], dim=-1))


def _override_inertia(env, ixx: float, iyy: float, izz: float) -> None:
    """Override the drone body inertia diagonal (kg·m²) via UsdPhysics.MassAPI.

    Sets /Drone/Body diagonalInertia to (ixx, iyy, izz) with identity principal
    axes so the diagonal maps directly to body-frame axes.  Changes take effect
    on the next env.reset() (IsaacLab re-reads USD mass properties during sim.reset()).

    Note: UsdPhysics.MassAPI(prim) always returns a non-None wrapper regardless of
    whether the API is applied; use prim.HasAPI() to check application state.
    """
    try:
        from pxr import Gf, UsdPhysics
        stage = env._task.sim.stage
        body = stage.GetPrimAtPath("/Drone/Body")
        if not body or not body.IsValid():
            print("[thrust_fin_wiggle] WARNING: /Drone/Body not found; inertia not overridden.")
            return

        # Apply the schema if not yet present; otherwise get a view of the existing one.
        if not body.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI.Apply(body)
            print("[thrust_fin_wiggle] UsdPhysics.MassAPI applied to /Drone/Body")
        else:
            mass_api = UsdPhysics.MassAPI(body)

        # CreateXxxAttr authors the attribute (safe whether or not it already exists).
        mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(ixx, iyy, izz))
        # Identity quaternion → principal axes == prim frame axes (no rotation needed)
        mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        print(
            f"[thrust_fin_wiggle] Inertia overridden on /Drone/Body: "
            f"Ixx={ixx:.4f}  Iyy={iyy:.4f}  Izz={izz:.4f} kg·m²"
        )
        print("[thrust_fin_wiggle] (takes effect on next env.reset() / sim.reset())")
    except Exception as exc:
        print(f"[thrust_fin_wiggle] WARNING: Could not override inertia: {exc}")


def _obs_val(obs: np.ndarray, idx: int) -> float:
    """Extract scalar obs[idx] from env 0, works for 1-D or 2-D arrays."""
    if obs.ndim == 2:
        return float(obs[0, idx])
    return float(obs[idx])


def build_episode_sequence(
    thrust_norm: float,
    max_deflection: float,
    hold_steps: int = 120,
) -> tuple[list[np.ndarray], list[str]]:
    """Build step-hold episode sequence (no sine wiggle).

    Uses the same routine for every axis:
      settle | <axis>- hold | settle | <axis>+ hold
    and appends one final settle at the end.

    Fin commands per axis (FRD body frame):
      Yaw  τ_z = 0.055·k·(−d1+d2+d3−d4)
        Yaw-: d1=+v  d2=−v  d3=−v  d4=+v  →  τ_z < 0
        Yaw+: d1=−v  d2=+v  d3=+v  d4=−v  →  τ_z > 0
      Roll  τ_x = −Z·k·(d3+d4)   [Fin_3(left)+Fin_4(right) at ±Y, lift +Y]
        Roll-: d3=+v  d4=+v  →  τ_x < 0
        Roll+: d3=−v  d4=−v  →  τ_x > 0
      Pitch τ_y = +Z·k·(d1+d2)   [Fin_1(fwd)+Fin_2(aft) at ±X, lift +X]
        Pitch-: d1=−v  d2=−v  →  τ_y < 0
        Pitch+: d1=+v  d2=+v  →  τ_y > 0
    """
    actions: list[np.ndarray] = []
    labels: list[str] = []

    thrust_norm = float(np.clip(thrust_norm, 0.0, 1.0))
    v = float(np.clip(max_deflection, 0.0, 1.0))

    def _hold(fin_cmds: list[float], label: str) -> None:
        for _ in range(hold_steps):
            a = np.zeros(5, dtype=np.float32)
            a[0] = thrust_norm
            for i, cmd in enumerate(fin_cmds):
                a[i + 1] = cmd
            actions.append(a)
            labels.append(label)

    def _settle() -> None:
        for _ in range(_STEPS_SETTLE):
            a = np.zeros(5, dtype=np.float32)
            a[0] = thrust_norm
            actions.append(a)
            labels.append("settle")

    axis_routine: list[tuple[str, list[float], list[float]]] = [
        # τ_z = 0.055·k·(−d1+d2+d3−d4)
        ("Yaw",   [+v, -v, -v, +v], [-v, +v, +v, -v]),
        # τ_x = −Z·k·(d3+d4) [Fin_3(left)+Fin_4(right)]
        ("Roll",  [0.0, 0.0, +v, +v], [0.0, 0.0, -v, -v]),
        # τ_y = +Z·k·(d1+d2) [Fin_1(fwd)+Fin_2(aft)]
        ("Pitch", [-v, -v, 0.0, 0.0], [+v, +v, 0.0, 0.0]),
    ]

    for axis_name, neg_cmd, pos_cmd in axis_routine:
        _settle()
        _hold(neg_cmd, f"{axis_name}-")
        _settle()
        _hold(pos_cmd, f"{axis_name}+")
    _settle()

    return actions, labels


def _print_phase_header(label: str, action: np.ndarray, hold_secs: float) -> None:
    """Print the ══ PHASE HOLD header with fin commands and expected response."""
    _, ax_expect = _EXPECTED[label]
    fins_str = "  ".join(f"{FIN_NAMES[i]}={action[i+1]:+.3f}" for i in range(4))
    bar = "═" * 60
    print(f"\n  {bar}")
    print(f"  ══  {label} HOLD  ({hold_secs:.1f} s)  —  expect {ax_expect}")
    print(f"  fins: {fins_str}")
    print(f"  {bar}")


def _print_hold_summary(
    label: str,
    peak_x: float,
    peak_y: float,
    peak_z: float,
) -> None:
    """Print peak omega summary at the end of a hold phase."""
    ax_name, ax_expect = _EXPECTED[label]
    dominant = max(peak_x, peak_y, peak_z)
    contamination_pct = ""
    if dominant > 1e-6:
        other = [("ωx(roll)", peak_x), ("ωy(pitch)", peak_y), ("ωz(yaw)", peak_z)]
        cross = [(n, p) for n, p in other if n != ax_name and dominant > 0]
        contamination_pct = "  cross: " + "  ".join(
            f"{n}={100*p/dominant:.1f}%" for n, p in cross
        )
    print(
        f"\n  [{label} END] peak  ωx(roll)={peak_x:.4f}  "
        f"ωy(pitch)={peak_y:.4f}  ωz(yaw)={peak_z:.4f} rad/s"
        f"{contamination_pct}"
    )
    print(f"  [{label} END] expected dominant: {ax_name}  ({ax_expect})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isaac Sim step-hold fin/torque axis diagnostic"
    )
    parser.add_argument(
        "--config",
        default="simulation/isaac/configs/isaac_env_single.yaml",
        help="Path to Isaac env YAML config",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to run (default: 100)",
    )
    parser.add_argument(
        "--hold-secs",
        type=float,
        default=1.0,
        help="Hold duration per direction in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--thrust",
        type=float,
        default=0.7,
        help="Normalized thrust command [0, 1] applied throughout (default: 0.7)",
    )
    parser.add_argument(
        "--spawn-altitude",
        type=float,
        default=0.4,
        help="Drone spawn altitude in metres (default: 0.4 — just above ground)",
    )
    parser.add_argument(
        "--max-deflection",
        type=float,
        default=1.0,
        help="Max absolute fin command in [-1, 1] (default: 1.0)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without GUI (default: False -- opens viewer)",
    )
    parser.add_argument(
        "--fixed-altitude",
        action="store_true",
        default=False,
        help="Disable gravity and lock drone position at 1.0 m altitude (rotation free)",
    )
    parser.add_argument(
        "--disable-wind",
        action="store_true",
        default=False,
        help="Disable Isaac wind disturbance model for this run (default: False)",
    )
    parser.add_argument(
        "--disable-gyro",
        action="store_true",
        default=False,
        help="Disable gyro-precession torque injection for this run (default: False)",
    )
    parser.add_argument(
        "--disable-anti-torque",
        action="store_true",
        default=False,
        help="Disable EDF reaction/anti-torque injection for this run (default: False)",
    )
    parser.add_argument(
        "--override-inertia",
        type=float,
        nargs=3,
        metavar=("Ixx", "Iyy", "Izz"),
        default=None,
        help=(
            "Override body inertia diagonal (kg·m²) to decouple axes. "
            "e.g. --override-inertia 0.1 0.1 0.1"
        ),
    )
    args = parser.parse_args()

    hold_steps = max(1, round(args.hold_secs * 120))

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    global _SIM_APP
    _SIM_APP = SimulationApp({"headless": args.headless})

    from simulation.isaac.envs.edf_isaac_env import EDFIsaacEnv

    env = EDFIsaacEnv(config_path=config_path, render_mode="human")

    # ------------------------------------------------------------------
    # Print fin geometry / hinge axes and body-frame convention
    # ------------------------------------------------------------------
    try:
        print("\n[thrust_fin_wiggle] Body frame: Y-forward (+X=right, +Y=fwd/nose, +Z=up/EDF)")
        print("[thrust_fin_wiggle] Axis semantics: ωx=ROLL  ωy=PITCH  ωz=YAW")
        print("[thrust_fin_wiggle] Fin hinge configuration from Isaac Sim asset:")
        for i, pos in enumerate(env._task._fin_anchor_pos_frd.tolist(), start=1):
            print(
                f"  Fin_{i}: hinge_pos_frd={tuple(f'{x:.4f}' for x in pos)}  "
                f"lift_dir_frd={tuple(float(x) for x in env._task._fin_lift[i - 1].tolist())}"
            )
        print(
            "[thrust_fin_wiggle] Body CoM from Isaac Sim:"
            f" {tuple(f'{x:.4f}' for x in env._task._body_com_default_frd.tolist())}"
        )
        print(
            "[thrust_fin_wiggle] Joint visual sign correction:"
            f" {tuple(int(x) for x in env._task._fin_joint_visual_sign.tolist())}"
        )
        print()
    except Exception as exc:
        print(f"[thrust_fin_wiggle] WARNING: could not print fin hinge configuration: {exc}")

    # Optional runtime effect toggles (layered on top of YAML config)
    if hasattr(env, "_task"):
        try:
            env._task.set_runtime_overrides(
                disable_wind=args.disable_wind,
                disable_gyro=args.disable_gyro,
                disable_anti_torque=args.disable_anti_torque,
                disable_gravity=False,
            )
        except Exception as exc:
            print(f"[thrust_fin_wiggle] WARNING: could not apply runtime overrides: {exc}")

        if args.disable_gyro:
            print("[thrust_fin_wiggle] Gyro precession: DISABLED (runtime override)")
        if args.disable_wind:
            print("[thrust_fin_wiggle] Wind model:      DISABLED (runtime override)")
        if args.disable_anti_torque:
            print("[thrust_fin_wiggle] Reaction torque: DISABLED (runtime override)")

    if args.fixed_altitude:
        _disable_gravity(env)

    if args.override_inertia is not None:
        _override_inertia(env, *args.override_inertia)

    env._task.cfg.spawn_altitude_min = args.spawn_altitude
    env._task.cfg.spawn_altitude_max = args.spawn_altitude
    env._task.cfg.spawn_vel_mag_min = 0.0
    env._task.cfg.spawn_vel_mag_max = 0.0

    sequence, phase_labels = build_episode_sequence(
        thrust_norm=args.thrust,
        max_deflection=args.max_deflection,
        hold_steps=hold_steps,
    )
    total_steps = len(sequence)
    episode_s   = total_steps / 120.0

    print(f"\n[thrust_fin_wiggle] Config:       {config_path}")
    print(f"[thrust_fin_wiggle] Thrust:       T_cmd = {args.thrust:.2f}")
    print(f"[thrust_fin_wiggle] Max deflect:  {args.max_deflection:.2f}")
    print(f"[thrust_fin_wiggle] Hold:         {args.hold_secs:.1f} s ({hold_steps} steps) per direction")
    print(f"[thrust_fin_wiggle] Settle:       {_STEPS_SETTLE/120:.2f} s ({_STEPS_SETTLE} steps)")
    print(f"[thrust_fin_wiggle] Episode:      {total_steps} steps  ({episode_s:.1f} s)")
    if args.fixed_altitude:
        print(f"[thrust_fin_wiggle] Mode:         FIXED-ALTITUDE (gravity off, pos locked at 1.0 m)")
    if args.override_inertia is not None:
        ixx, iyy, izz = args.override_inertia
        print(f"[thrust_fin_wiggle] Inertia:      Ixx={ixx:.4f}  Iyy={iyy:.4f}  Izz={izz:.4f} kg·m² (OVERRIDDEN)")
    print(f"[thrust_fin_wiggle] Running {args.episodes} episode(s)...\n")

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        print(f"\n{'='*70}")
        print(f"  EPISODE {ep + 1}/{args.episodes}")
        print(f"{'='*70}")

        start_alt   = _obs_val(obs, _OBS_ALTITUDE)
        target_alt  = start_alt + 0.2
        thrust_cmd  = float(np.clip(args.thrust, 0.0, 1.0))

        print(f"  initial h_agl={start_alt:.3f} m  speed={_obs_val(obs, _OBS_SPEED):.3f} m/s")
        print(f"  pre-phase: thrust only until h_agl >= {target_alt:.3f} m")

        # Pre-phase: climb to target altitude, fins zeroed
        pre_steps = 0
        done = np.array(False)
        while _obs_val(obs, _OBS_ALTITUDE) < target_alt:
            action = np.zeros(5, dtype=np.float32)
            action[0] = thrust_cmd
            obs, _, done, _, _ = env.step(action)
            if args.fixed_altitude:
                _lock_position_at_altitude(env, altitude_m=1.0)
            pre_steps += 1
            if pre_steps % 30 == 0 or bool(np.any(done)):
                print(
                    f"    pre step {pre_steps:4d}  "
                    f"h={_obs_val(obs, _OBS_ALTITUDE):.3f} m  "
                    f"speed={_obs_val(obs, _OBS_SPEED):.3f} m/s"
                )
            if bool(np.any(done)):
                print("  [done] terminated before fin sequence.\n")
                break

        if bool(np.any(done)):
            continue

        print(
            f"  reached h_agl={_obs_val(obs, _OBS_ALTITUDE):.3f} m "
            "— starting step-hold sequence\n"
        )

        prev_label    = ""
        step_in_phase = 0
        phase_x       = 0.0   # peak |ωx| for current hold phase
        phase_y       = 0.0
        phase_z       = 0.0

        for _, (action, label) in enumerate(zip(sequence, phase_labels)):
            obs, _, done, _, _ = env.step(action)
            if args.fixed_altitude:
                _lock_position_at_altitude(env, altitude_m=1.0)

            omega_x = _obs_val(obs, _OBS_OMEGA_X)
            omega_y = _obs_val(obs, _OBS_OMEGA_Y)
            omega_z = _obs_val(obs, _OBS_OMEGA_Z)
            h       = _obs_val(obs, _OBS_ALTITUDE)
            speed   = _obs_val(obs, _OBS_SPEED)

            # ── Phase transition ────────────────────────────────────────
            if label != prev_label:
                # Summarise completed hold phase
                if prev_label in HOLD_LABELS:
                    _print_hold_summary(prev_label, phase_x, phase_y, phase_z)

                # Reset peaks for new phase
                phase_x = phase_y = phase_z = 0.0
                step_in_phase = 0

                if label == "settle":
                    # Reset orientation at the start of every settle
                    _reset_orientation(env)
                    print(
                        f"\n  ── settle ({_STEPS_SETTLE/120:.2f} s, orientation reset) ──"
                    )
                elif label in HOLD_LABELS:
                    # Enforce deterministic "zero angular velocity -> hold" sequence.
                    _reset_orientation(env)
                    if args.fixed_altitude:
                        _lock_position_at_altitude(env, altitude_m=1.0)
                    _print_phase_header(label, action, args.hold_secs)

                prev_label = label

            # ── Track peaks and print detail during holds ───────────────
            if label in HOLD_LABELS:
                phase_x = max(phase_x, abs(omega_x))
                phase_y = max(phase_y, abs(omega_y))
                phase_z = max(phase_z, abs(omega_z))

                if step_in_phase % _PRINT_EVERY == 0:
                    t_hold = step_in_phase / 120.0
                    fins_str = "  ".join(
                        f"{FIN_NAMES[i]}={action[i+1]:+.3f}" for i in range(4)
                    )
                    print(
                        f"  t={t_hold:.2f}s  "
                        f"ωx(roll)={omega_x:+.4f}  "
                        f"ωy(pitch)={omega_y:+.4f}  "
                        f"ωz(yaw)={omega_z:+.4f} rad/s  "
                        f"h={h:.3f}m  spd={speed:.3f}m/s  [{fins_str}]"
                    )

            step_in_phase += 1

            if bool(np.any(done)):
                if label in HOLD_LABELS:
                    _print_hold_summary(label, phase_x, phase_y, phase_z)
                print("  [done] episode terminated early.\n")
                break

        else:
            if prev_label in HOLD_LABELS:
                _print_hold_summary(prev_label, phase_x, phase_y, phase_z)
            print(f"\n  [done] episode {ep + 1} complete")

    env.close()
    if _SIM_APP is not None:
        _SIM_APP.close()


if __name__ == "__main__":
    main()
