"""
Shared helpers for scripted visual-validation scenarios.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import torch


def sim_root_from_here() -> Path:
    """Return the simulation/isaac root from the visual tests directory."""
    return Path(__file__).parents[2]


def build_debug_env(
    sim_root: Path,
    task_name: str,
    disturbance: str | None = None,
    overrides: dict | None = None,
):
    """Build a single-env debug environment with gizmos enabled."""
    from tvc_env.envs.base_env import BaseEnvConfig
    from tvc_env.envs.single_env import SingleEnvDebug

    # Render every physics substep so the viewport runs at physics rate (120 Hz)
    # instead of the control rate (30 Hz). TVCSimScene.step() checks this env var.
    os.environ["ISAAC_VIZ_SLOW"] = "1"

    config = BaseEnvConfig(
        task_name=task_name,
        env_config_path=sim_root / "configs/env/single_env_debug.yaml",
        disturbance_config_path=sim_root / disturbance if disturbance else None,
        overrides=overrides,
        sim_root=sim_root,
    )
    env = SingleEnvDebug(config)
    env.reset()
    return env


def reset_to_state(
    env,
    position: list[float] | None = None,
    quaternion_wxyz: list[float] | None = None,
    linear_vel: list[float] | None = None,
    angular_vel: list[float] | None = None,
) -> None:
    """Reset the single environment to a deterministic root state."""
    device = env._drone.device
    pos = torch.tensor([position or [0.0, 0.0, 5.0]], dtype=torch.float32, device=device)
    quat = torch.tensor([quaternion_wxyz or [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    lin = torch.tensor([linear_vel or [0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    ang = torch.tensor([angular_vel or [0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    env._body_iface.set_root_state(pos, quat, lin, ang)
    env._reset_manager._servo_state.zero_()
    env._reset_manager._omega_state.zero_()
    env._reset_manager._omega_prev.zero_()
    env._target_position = pos[0].detach().clone()


def action_tensor(values: list[float], device: torch.device) -> torch.Tensor:
    """Create a single-env action tensor on the target device."""
    return torch.tensor([values], dtype=torch.float32, device=device)


def _fmt_vec(vec: torch.Tensor) -> str:
    return "[" + ", ".join(f"{float(v):+.3f}" for v in vec.tolist()) + "]"


def _print_step_detail(step: int, episode_steps: int, env, action: torch.Tensor,
                        obs, reward, notes: list[str]) -> None:
    """Print detailed per-step telemetry to the terminal."""
    state = env._build_vehicle_state()
    pos = state.position[0]
    quat = state.quaternion_wxyz[0]
    lin_vel = state.linear_vel_frd[0]
    ang_vel = state.angular_vel_frd[0]
    omega = state.motor_omega[0].item()
    height = state.height[0].item()

    # Compute forces for readout
    throttle = action[:, 4].clamp(0.0, 1.0)
    fin_forces, _ = env._fin_dispatch.compute_body_frame_forces(
        env._reset_manager.servo_state, throttle,
    )
    thrust_N = float(env._edf_model.compute_thrust(state.motor_omega)[0].item())
    aero_sum = fin_forces[0].sum(dim=0)

    reward_val = float(reward[0]) if reward is not None else 0.0

    print(
        f"[step {step + 1:>4d}/{episode_steps}]  "
        f"pos_w={_fmt_vec(pos)}  h={height:+.3f}m  "
        f"vel_frd={_fmt_vec(lin_vel)}  "
        f"ang_frd={_fmt_vec(ang_vel)}"
    )
    print(
        f"    action={_fmt_vec(action[0])}  "
        f"thrust={thrust_N:.2f}N  omega={omega:.0f}rad/s  "
        f"aero_sum_frd={_fmt_vec(aero_sum)}  reward={reward_val:.3f}"
    )
    if notes:
        print(f"    {' | '.join(notes)}")


def play_scripted_episode(
    simulation_app,
    env,
    scenario_name: str,
    description: str,
    episode_steps: int,
    action_fn: Callable[[int, object, dict], torch.Tensor],
    setup_fn: Callable[[object], None] | None = None,
    note_fn: Callable[[int, object, dict], list[str]] | None = None,
    print_every: int = 1,
    num_episodes: int = 0,
) -> bool:
    """Run scripted visual-validation episodes with real-time pacing.

    Args:
        num_episodes: Number of times to repeat the episode. 0 = loop forever.
    """
    # Sim time per env.step() = decimation * physics_dt
    sim_dt = env._config.decimation * env._config.physics_dt

    # Kit event-loop pump for real-time gap filling (non-blocking)
    try:
        import omni.kit.app
        _kit_app = omni.kit.app.get_app()
    except Exception:
        _kit_app = None

    loop_label = "∞" if num_episodes == 0 else str(num_episodes)
    print(f"=== Visual Scenario: {scenario_name} (episodes={loop_label}) ===")
    print(f"    {description}")
    print(f"    sim_dt={sim_dt:.4f}s  decimation={env._config.decimation}  physics_dt={env._config.physics_dt:.5f}s")
    print()

    ep = 0
    while num_episodes == 0 or ep < num_episodes:
        ep += 1
        obs, _ = env.reset()
        if setup_fn is not None:
            setup_fn(env)

        if num_episodes != 1:
            ep_label = f"ep={ep}" if num_episodes == 0 else f"ep={ep}/{num_episodes}"
            print(f"--- {ep_label} ---")

        wall_start = time.perf_counter()

        for step in range(episode_steps):
            action = action_fn(step, env, obs)
            notes = note_fn(step, env, obs) if note_fn is not None else []
            env.set_visual_context(
                scenario_name=scenario_name,
                step_index=step,
                episode_steps=episode_steps,
                notes=notes,
                print_terminal=False,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            simulation_app.update()

            if step % max(print_every, 1) == 0:
                _print_step_detail(step, episode_steps, env, action, obs, reward, notes)

            # Real-time pacing via Kit event loop (non-blocking)
            target_wall = wall_start + (step + 1) * sim_dt
            while time.perf_counter() < target_wall:
                if _kit_app is not None:
                    _kit_app.update()
                else:
                    remaining = target_wall - time.perf_counter()
                    if remaining > 0.001:
                        time.sleep(0.001)
                    break

            if terminated.any() or truncated.any():
                print(f"  [{scenario_name}] early termination at step {step + 1}")
                break

        elapsed = time.perf_counter() - wall_start
        sim_time = (step + 1) * sim_dt
        print(f"  ep done | {step + 1} steps | sim={sim_time:.2f}s wall={elapsed:.2f}s")

    print(f"\n=== Completed: {scenario_name} ===")
    env.clear_visual_context()
    return True
