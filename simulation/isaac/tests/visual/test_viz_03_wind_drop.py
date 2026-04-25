"""
Visual validation scenario for wind drift / drop displacement.
"""

from __future__ import annotations

from pathlib import Path

import torch

from visual_common import build_debug_env, play_scripted_episode, reset_to_state, sim_root_from_here


def run(
    simulation_app,
    episode_steps: int = 100,
    print_every: int = 1,
    sim_root: str | Path | None = None,
    num_episodes: int = 0,
) -> bool:
    """Run the wind-drop visual validation."""
    sim_root = Path(sim_root) if sim_root is not None else sim_root_from_here()
    env = build_debug_env(
        sim_root=sim_root,
        task_name="landing",
        disturbance="configs/disturbances/wind.yaml",
        overrides={
            "task": {
                "episode_length_s": 10.0,
                "termination": {"crash": False, "max_tilt": 3.14, "max_altitude": 100.0},
            }
        },
    )
    wind_origin = {"position": None}

    def setup_fn(debug_env) -> None:
        reset_to_state(debug_env, position=[0.0, 0.0, 10.0])
        wind_origin["position"] = debug_env._body_iface.get_root_position()[0].detach().clone()
        debug_env._target_position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=debug_env._drone.device)

    def action_fn(step: int, debug_env, obs) -> torch.Tensor:
        del step, obs
        return torch.zeros(1, 5, dtype=torch.float32, device=debug_env._drone.device)

    def note_fn(step: int, debug_env, obs) -> list[str]:
        del step, obs
        current = debug_env._body_iface.get_root_position()[0]
        drift = current - wind_origin["position"]
        drift_str = "[" + ", ".join(f"{float(v):+.3f}" for v in drift.tolist()) + "]"
        return [
            "mode=wind drop / zero-throttle release",
            f"displacement_w={drift_str}",
            "goal=visually confirm wind-driven lateral drift while the vehicle falls",
        ]

    return play_scripted_episode(
        simulation_app=simulation_app,
        env=env,
        scenario_name="Wind Drop Drift",
        description="Drops the vehicle from height in wind with zero throttle and prints the world-frame displacement vector throughout the 100-step episode.",
        episode_steps=episode_steps,
        action_fn=action_fn,
        setup_fn=setup_fn,
        note_fn=note_fn,
        print_every=print_every,
        num_episodes=num_episodes,
    )
