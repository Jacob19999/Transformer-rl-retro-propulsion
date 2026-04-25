"""
Shared helpers for scripted visual-validation scenarios.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import torch
from tvc_env.common.frames import frd_force_to_isaac
from tvc_env.common.quaternions import rotate_vector


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
    edf_force_body_frd = torch.tensor(
        [0.0, 0.0, -thrust_N], dtype=torch.float32, device=state.position.device
    )
    edf_force_body_isaac = frd_force_to_isaac(edf_force_body_frd).unsqueeze(0)
    edf_force_world_expected = rotate_vector(
        quat.unsqueeze(0), edf_force_body_isaac
    )[0]
    aero_sum = fin_forces[0].sum(dim=0)
    composed_msg = "unavailable"
    body_link_idx = None
    if hasattr(env, "_drone") and hasattr(env, "_art_map"):
        try:
            body_link_idx = int(env._art_map.body_index)
            composer = env._drone.permanent_wrench_composer
            composed = composer.composed_force_as_torch
            if composed is None:
                composed_msg = "no tensor"
            elif composed.numel() == 0:
                composed_msg = f"empty tensor shape={list(composed.shape)}"
            elif 0 <= body_link_idx < composed.shape[1]:
                composed_msg = f"body_id={body_link_idx} composed_force={_fmt_vec(composed[0, body_link_idx].detach())}"
            else:
                composed_msg = f"body_id={body_link_idx} out_of_range shape={list(composed.shape)}"
        except Exception as exc:
            composed_msg = f"error={type(exc).__name__}: {exc}"

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
    print(f"    expected_body_force_world={_fmt_vec(edf_force_world_expected)}")
    print(f"    wrench_composer {composed_msg}")
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

    # Safe render-only pump: use sim.render() which disables physics
    # auto-stepping before calling app.update(), preventing extra unforced
    # physics steps that would dilute external wrenches.
    _sim_ctx = env._sim_scene.sim if hasattr(env, "_sim_scene") else None

    loop_label = "∞" if num_episodes == 0 else str(num_episodes)
    print(f"=== Visual Scenario: {scenario_name} (episodes={loop_label}) ===")
    print(f"    {description}")
    print(f"    sim_dt={sim_dt:.4f}s  decimation={env._config.decimation}  physics_dt={env._config.physics_dt:.5f}s")
    if hasattr(env, "_drone") and hasattr(env, "_art_map"):
        try:
            body_idx = int(env._art_map.body_index)
            body_name = env._drone.body_names[body_idx]
            print(f"    body_link_index={body_idx} body_link_name={body_name}")
        except Exception:
            pass
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
            if _sim_ctx is not None:
                _sim_ctx.render()
            else:
                simulation_app.update()

            if step % max(print_every, 1) == 0:
                _print_step_detail(step, episode_steps, env, action, obs, reward, notes)

            # Real-time pacing — use sim.render() to keep the viewport alive
            # without triggering extra physics steps.
            target_wall = wall_start + (step + 1) * sim_dt
            while time.perf_counter() < target_wall:
                if _sim_ctx is not None:
                    _sim_ctx.render()
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
