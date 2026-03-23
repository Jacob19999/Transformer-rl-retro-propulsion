"""Auto-calibrate fin-axis mapping from measured Isaac angular-rate response.

For each fin, runs three axis-isolation experiments:
  - free roll axis, lock pitch+yaw
  - free pitch axis, lock roll+yaw
  - free yaw axis, lock roll+pitch
The largest |delta omega| axis per fin becomes that fin's dominant mapping.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from simulation.isaac.fin_mapping import derive_mapping_from_axis_response  # noqa: E402
from simulation.isaac.scripts._shared import (  # noqa: E402
    create_sim_app,
    disable_gravity,
    lock_position_at_altitude,
    resolve_repo_path,
)


def _axis_idx(axis: str) -> int:
    return {"roll": 0, "pitch": 1, "yaw": 2}[axis]


def _enforce_axis_lock(env, free_axis: str) -> None:
    """Zero locked angular-rate channels each step."""
    task = env._task
    idx = _axis_idx(free_axis)
    vel_w = task.robot.data.root_lin_vel_w.clone()
    ang_w = task.robot.data.root_ang_vel_w.clone()
    for i in range(3):
        if i != idx:
            ang_w[:, i] = 0.0
    task.robot.write_root_velocity_to_sim(torch.cat([vel_w, ang_w], dim=-1))


def _run_trial(
    env,
    *,
    fin_idx: int,
    fin_cmd: float,
    thrust_cmd: float,
    free_axis: str,
    settle_steps: int,
    hold_steps: int,
    lock_altitude_m: float,
) -> tuple[float, dict[str, Any] | None]:
    """Run a single axis-isolation trial and return (signed_delta_omega, force_snapshot).

    *force_snapshot* is a dict with per-fin magnitudes and vectors from the
    last hold step (or ``None`` if the force gizmo is not active).
    """
    obs, _ = env.reset()
    _ = obs
    lock_position_at_altitude(env, altitude_m=lock_altitude_m)
    zero_action = np.zeros(5, dtype=np.float32)
    zero_action[0] = thrust_cmd
    for _ in range(settle_steps):
        env.step(zero_action)
        _enforce_axis_lock(env, free_axis)
        lock_position_at_altitude(env, altitude_m=lock_altitude_m)

    task = env._task
    axis_i = _axis_idx(free_axis)
    omega0 = float(task.robot.data.root_ang_vel_b[0, axis_i].item())
    peak_delta = 0.0

    hold_action = np.zeros(5, dtype=np.float32)
    hold_action[0] = thrust_cmd
    hold_action[fin_idx + 1] = fin_cmd
    for _ in range(hold_steps):
        env.step(hold_action)
        _enforce_axis_lock(env, free_axis)
        lock_position_at_altitude(env, altitude_m=lock_altitude_m)
        omega = float(task.robot.data.root_ang_vel_b[0, axis_i].item())
        peak_delta = max(peak_delta, abs(omega - omega0))
    omega_end = float(task.robot.data.root_ang_vel_b[0, axis_i].item())
    signed_delta = omega_end - omega0
    delta = signed_delta if abs(signed_delta) >= peak_delta * 0.6 else np.sign(signed_delta) * peak_delta

    # Capture force gizmo snapshot from the last hold step.
    force_snap: dict[str, Any] | None = None
    gizmo = getattr(task, "_force_gizmo", None)
    if gizmo is not None and gizmo.fin_magnitudes:
        force_snap = {
            "fin_magnitudes": list(gizmo.fin_magnitudes[0]),
            "fin_vectors": [list(v) for v in gizmo.fin_vectors[0]],
            "thrust_mag": gizmo.thrust_magnitudes[0] if gizmo.thrust_magnitudes else 0.0,
            "torque_mag": gizmo.torque_magnitudes[0] if gizmo.torque_magnitudes else 0.0,
        }

    return delta, force_snap


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate fin mapping from Isaac delta-omega response")
    parser.add_argument(
        "--config",
        default="simulation/isaac/configs/isaac_env_single.yaml",
        help="Isaac env config path",
    )
    parser.add_argument(
        "--output",
        default="simulation/configs/fin_mapping.yaml",
        help="Output mapping YAML path",
    )
    parser.add_argument("--thrust", type=float, default=0.70, help="Normalized thrust during calibration")
    parser.add_argument("--deflection", type=float, default=0.65, help="Normalized fin command magnitude")
    parser.add_argument("--settle-steps", type=int, default=45)
    parser.add_argument("--hold-steps", type=int, default=90)
    parser.add_argument("--lock-altitude", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--draw-forces",
        action="store_true",
        default=False,
        help=(
            "Overlay thrust / fin-aero / body-torque gizmo arrows in the viewport "
            "(requires GUI; ignored with --headless)."
        ),
    )
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config)
    output_path = resolve_repo_path(args.output)

    sim_app = create_sim_app(headless=bool(args.headless))
    try:
        from simulation.isaac.envs.edf_isaac_env import EDFIsaacEnv

        env = EDFIsaacEnv(
            config_path=config_path,
            seed=int(args.seed),
            disable_wind=True,
            disable_gyro=True,
            disable_anti_torque=True,
            debug_draw_forces=bool(args.draw_forces) and not bool(args.headless),
        )
        try:
            env._task.cfg.spawn_altitude_min = float(args.lock_altitude)
            env._task.cfg.spawn_altitude_max = float(args.lock_altitude)
            env._task.cfg.spawn_vel_mag_min = 0.0
            env._task.cfg.spawn_vel_mag_max = 0.0
            disable_gravity(env, prefix="calibrate_fin_mapping")

            per_fin: list[dict[str, Any]] = []
            dominant_axis: list[str] = []
            dominant_sign: list[float] = []
            yaw_torque_signs: list[float] = []
            axis_order = ("roll", "pitch", "yaw")

            # Read body inertia diagonal for torque normalization.
            # The calibration measures angular rate (ω), but the controller cares
            # about torque authority (τ = I·α ∝ I·ω).  With Izz ≪ Ixx=Iyy the
            # small yaw inertia amplifies yaw rates, fooling a raw-ω classifier
            # into labelling every fin as yaw-dominant.  Normalizing ω by I gives
            # a torque proxy that correctly identifies pitch/roll as dominant.
            inertia_diag = env._task._body_inertia_default.diagonal().cpu()
            inertia_scale = {
                "roll": float(inertia_diag[0]),
                "pitch": float(inertia_diag[1]),
                "yaw": float(inertia_diag[2]),
            }
            print(
                f"[calibrate_fin_mapping] Inertia diag (Z-up): "
                f"Ixx={inertia_scale['roll']:.6f}  "
                f"Iyy={inertia_scale['pitch']:.6f}  "
                f"Izz={inertia_scale['yaw']:.6f}"
            )

            for fin_idx in range(4):
                axis_deltas: dict[str, float] = {}
                fin_force_snaps: dict[str, dict[str, Any] | None] = {}
                for axis in axis_order:
                    pos, pos_snap = _run_trial(
                        env,
                        fin_idx=fin_idx,
                        fin_cmd=abs(float(args.deflection)),
                        thrust_cmd=float(args.thrust),
                        free_axis=axis,
                        settle_steps=int(args.settle_steps),
                        hold_steps=int(args.hold_steps),
                        lock_altitude_m=float(args.lock_altitude),
                    )
                    neg, neg_snap = _run_trial(
                        env,
                        fin_idx=fin_idx,
                        fin_cmd=-abs(float(args.deflection)),
                        thrust_cmd=float(args.thrust),
                        free_axis=axis,
                        settle_steps=int(args.settle_steps),
                        hold_steps=int(args.hold_steps),
                        lock_altitude_m=float(args.lock_altitude),
                    )
                    chosen = pos if abs(pos) >= abs(neg) else neg
                    axis_deltas[axis] = float(chosen)
                    fin_force_snaps[axis] = pos_snap if abs(pos) >= abs(neg) else neg_snap

                # Normalize by inertia to get torque authority (τ ∝ I·ω).
                torque_deltas = {a: axis_deltas[a] * inertia_scale[a] for a in axis_order}
                dom = max(axis_order, key=lambda a: abs(torque_deltas[a]))
                dom_sign = 1.0 if torque_deltas[dom] >= 0.0 else -1.0
                dominant_axis.append(dom)
                dominant_sign.append(dom_sign)
                yaw_torque_signs.append(1.0 if torque_deltas["yaw"] >= 0.0 else -1.0)
                fin_entry: dict[str, Any] = {
                    "fin_index": fin_idx,
                    "delta_omega_roll": axis_deltas["roll"],
                    "delta_omega_pitch": axis_deltas["pitch"],
                    "delta_omega_yaw": axis_deltas["yaw"],
                    "torque_roll": torque_deltas["roll"],
                    "torque_pitch": torque_deltas["pitch"],
                    "torque_yaw": torque_deltas["yaw"],
                    "dominant_axis": dom,
                    "dominant_sign": dom_sign,
                }

                # Attach force gizmo snapshot from the dominant-axis trial.
                dom_snap = fin_force_snaps.get(dom)
                if dom_snap is not None:
                    fin_entry["force_magnitude"] = dom_snap["fin_magnitudes"][fin_idx]
                    fin_entry["force_vector"] = dom_snap["fin_vectors"][fin_idx]
                    fin_entry["thrust_magnitude"] = dom_snap["thrust_mag"]

                per_fin.append(fin_entry)

                print(
                    f"[calibrate_fin_mapping] fin={fin_idx+1} "
                    f"dω_roll={axis_deltas['roll']:+.5f} "
                    f"dω_pitch={axis_deltas['pitch']:+.5f} "
                    f"dω_yaw={axis_deltas['yaw']:+.5f} "
                    f"| τ_roll={torque_deltas['roll']:+.6f} "
                    f"τ_pitch={torque_deltas['pitch']:+.6f} "
                    f"τ_yaw={torque_deltas['yaw']:+.6f} "
                    f"-> {dom} ({dom_sign:+.0f})"
                )
                if dom_snap is not None:
                    fmag = dom_snap["fin_magnitudes"][fin_idx]
                    fvec = dom_snap["fin_vectors"][fin_idx]
                    print(
                        f"                        "
                        f"|F| = {fmag:.4f} N  "
                        f"F = ({fvec[0]:+.4f}, {fvec[1]:+.4f}, {fvec[2]:+.4f})"
                    )

            mapping = derive_mapping_from_axis_response(
                dominant_axis, dominant_sign, yaw_torque_signs=yaw_torque_signs,
            )
            payload = {
                "fin_mapping": {
                    "joint_source_indices": list(mapping.joint_source_indices),
                    "joint_signs": list(mapping.joint_signs),
                    "pitch_weights": list(mapping.pitch_weights),
                    "roll_weights": list(mapping.roll_weights),
                    "yaw_weights": list(mapping.yaw_weights),
                    "generated_by": "calibrate_fin_mapping.py",
                    "measurement": {
                        "thrust": float(args.thrust),
                        "deflection": float(args.deflection),
                        "settle_steps": int(args.settle_steps),
                        "hold_steps": int(args.hold_steps),
                        "inertia_diag_zup": {
                            "Ixx": inertia_scale["roll"],
                            "Iyy": inertia_scale["pitch"],
                            "Izz": inertia_scale["yaw"],
                        },
                        "per_fin": per_fin,
                    },
                }
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(payload, f, sort_keys=False)
            print(f"[calibrate_fin_mapping] wrote: {output_path}")
        finally:
            env.close()
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
