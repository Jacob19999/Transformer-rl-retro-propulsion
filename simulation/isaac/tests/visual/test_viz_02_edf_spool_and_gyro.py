"""
Visual validation scenario for EDF spool, reaction torque, and gyro coupling.
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
    """Run the EDF spool / gyro visual validation."""
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

    def setup_fn(debug_env) -> None:
        # Start in free-flight with a modest downward speed, then verify the EDF
        # spool-up arrests descent and transitions into climb.
        reset_to_state(
            debug_env,
            position=[0.0, 0.0, 8.0],
            linear_vel=[0.0, 0.0, -1.0],  # world frame: -Z is downward
        )
        # Pre-spool near hover thrust so the episode demonstrates descent arrest
        # and climb rather than a ground impact from a cold-start free-fall.
        debug_env._reset_manager._omega_state.fill_(debug_env._edf_model.omega_max * 0.95)
        debug_env._reset_manager._omega_prev.copy_(debug_env._reset_manager._omega_state)

    def action_fn(step: int, debug_env, obs) -> torch.Tensor:
        del obs
        device = debug_env._drone.device
        action = torch.zeros(1, 5, dtype=torch.float32, device=device)

        # Hold full throttle immediately; pre-spooled omega still reveals motor lag.
        action[0, 4] = 1.0

        if 50 <= step < 70:
            action[0, 1] = 0.18
            action[0, 3] = -0.18
        elif step >= 70:
            action[0, 1] = -0.18
            action[0, 3] = 0.18

        return action

    def note_fn(step: int, debug_env, obs) -> list[str]:
        del debug_env, obs
        if step < 50:
            phase = "phase=hold full throttle / watch spool and reaction torque arrows"
        elif step < 70:
            phase = "phase=deflect +Y/-Y fin pair"
        else:
            phase = "phase=reverse fin-pair deflection"
        return [
            phase,
            "goal=visually confirm spool-up, body response, and gyro/reaction coupling",
        ]

    return play_scripted_episode(
        simulation_app=simulation_app,
        env=env,
        scenario_name="EDF Spool + Gyro",
        description="Ramps throttle to full, then uses a fin pair to induce body response while showing thrust, aero-force, and reaction-torque arrows.",
        episode_steps=episode_steps,
        action_fn=action_fn,
        setup_fn=setup_fn,
        note_fn=note_fn,
        print_every=print_every,
        num_episodes=num_episodes,
    )
