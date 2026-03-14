"""Shared helpers for Isaac Sim diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from isaacsim import SimulationApp

from simulation.isaac.conventions import ACTION_DIM, ACTION_FIN_SLICE, ACTION_THRUST_IDX

REPO_ROOT = Path(__file__).resolve().parents[3]


def create_sim_app(*, headless: bool) -> SimulationApp:
    """Create the SimulationApp before importing IsaacLab modules."""
    return SimulationApp({"headless": headless})


def resolve_repo_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def obs_scalar(obs: np.ndarray, idx: int) -> float:
    """Extract obs[idx] from env 0 for 1-D or batched observations."""
    if obs.ndim == 2:
        return float(obs[0, idx])
    return float(obs[idx])


def any_done(done: bool | np.ndarray | Iterable[bool]) -> bool:
    """Reduce scalar or batched done flags to a single bool."""
    return bool(np.any(done))


def make_action(
    thrust_norm: float,
    fin_deflections: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    """Build the canonical 5D Isaac action array."""
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[ACTION_THRUST_IDX] = float(np.clip(thrust_norm, -1.0, 1.0))
    action[ACTION_FIN_SLICE] = [float(v) for v in fin_deflections]
    return action


def set_gravity(env, magnitude: float, *, prefix: str = "isaac_shared") -> None:
    """Override gravity magnitude on the live physics scene."""
    try:
        from pxr import UsdPhysics

        stage = env._task.sim.stage
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                UsdPhysics.Scene(prim).GetGravityMagnitudeAttr().Set(float(magnitude))
                return
        raise RuntimeError("Physics scene not found")
    except Exception as exc:
        raise RuntimeError(f"[{prefix}] Could not set gravity: {exc}") from exc


def disable_gravity(env, *, prefix: str) -> None:
    """Zero gravity and print a diagnostic-friendly status line."""
    try:
        set_gravity(env, 0.0, prefix=prefix)
        print(f"[{prefix}] Gravity disabled.")
    except Exception as exc:
        print(f"[{prefix}] WARNING: {exc}")


def lock_position_at_altitude(env, altitude_m: float = 1.0) -> None:
    """Lock drone world position at a fixed altitude, leaving rotation free."""
    task = getattr(env, "_task", None)
    if task is None or not hasattr(task, "robot"):
        return

    import torch

    pos_w = task.robot.data.root_pos_w.clone()
    quat_w = task.robot.data.root_quat_w.clone()
    pos_w[:, 2] = altitude_m
    task.robot.write_root_pose_to_sim(torch.cat([pos_w, quat_w], dim=-1))

    vel_w = task.robot.data.root_lin_vel_w.clone()
    ang_w = task.robot.data.root_ang_vel_w.clone()
    vel_w[:] = 0.0
    task.robot.write_root_velocity_to_sim(torch.cat([vel_w, ang_w], dim=-1))


def reset_orientation(env) -> None:
    """Reset orientation to identity and zero angular velocity, keep position."""
    task = getattr(env, "_task", None)
    if task is None or not hasattr(task, "robot"):
        return

    import torch

    from simulation.isaac.quaternion_isaac import identity_quat_wxyz

    pos_w = task.robot.data.root_pos_w.clone()
    quat_w = identity_quat_wxyz(pos_w.shape[0], device=pos_w.device)
    task.robot.write_root_pose_to_sim(torch.cat([pos_w, quat_w], dim=-1))

    vel_w = task.robot.data.root_lin_vel_w.clone()
    ang_w = task.robot.data.root_ang_vel_w.clone()
    vel_w[:] = 0.0
    ang_w[:] = 0.0
    task.robot.write_root_velocity_to_sim(torch.cat([vel_w, ang_w], dim=-1))

