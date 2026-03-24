"""
Visual validation scenario for fin sweep / wiggle confirmation.
"""

from __future__ import annotations

from pathlib import Path

import torch

from visual_common import action_tensor, build_debug_env, play_scripted_episode, reset_to_state, sim_root_from_here

MAX_DEFLECTION = 0.262


def run(
    simulation_app,
    episode_steps: int = 100,
    print_every: int = 1,
    sim_root: str | Path | None = None,
    num_episodes: int = 0,
) -> bool:
    """Run the fin-sweep visual validation."""
    sim_root = Path(sim_root) if sim_root is not None else sim_root_from_here()
    env = build_debug_env(
        sim_root=sim_root,
        task_name="hover",
        disturbance="configs/disturbances/nominal.yaml",
        overrides={
            "task": {
                "episode_length_s": 10.0,
                "termination": {"crash": False, "max_tilt": 3.14, "max_altitude_error": 100.0},
            }
        },
    )

    segment_steps = max(episode_steps // 4, 1)

    def setup_fn(debug_env) -> None:
        reset_to_state(debug_env, position=[0.0, 0.0, 1.5])

    def action_fn(step: int, debug_env, obs) -> torch.Tensor:
        del obs
        device = debug_env._drone.device
        action = torch.zeros(1, 5, dtype=torch.float32, device=device)
        action[0, 4] = 0.95
        active_fin = min(step // segment_steps, 3)
        local_step = min(step % segment_steps, segment_steps - 1)
        fraction = 0.0 if segment_steps == 1 else local_step / (segment_steps - 1)
        angle = -MAX_DEFLECTION + (2.0 * MAX_DEFLECTION * fraction)
        action[0, active_fin] = angle
        return action

    def note_fn(step: int, debug_env, obs) -> list[str]:
        del debug_env, obs
        active_fin = min(step // segment_steps, 3)
        fin_name = ["+X", "+Y", "-X", "-Y"][active_fin]
        return [
            "mode=pre-launch fin wiggle / sweep",
            f"active_fin={fin_name}",
            "goal=visually confirm each fin sweeps both directions and the force arrows follow",
        ]

    return play_scripted_episode(
        simulation_app=simulation_app,
        env=env,
        scenario_name="Fin Sweep Wiggle",
        description="Sweeps each fin across +/- max deflection while holding throttle to make the aero-force arrows visible.",
        episode_steps=episode_steps,
        action_fn=action_fn,
        setup_fn=setup_fn,
        note_fn=note_fn,
        print_every=print_every,
        num_episodes=num_episodes,
    )
