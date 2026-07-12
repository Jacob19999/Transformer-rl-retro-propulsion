"""Isaac integration regression for the RL contact/impact pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

try:
    import omni.usd  # noqa: F401
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False

pytestmark = pytest.mark.skipif(not ISAAC_AVAILABLE, reason="Isaac Sim runtime not available")


def test_upright_drop_uses_physx_contact_and_captures_impact_speed():
    from tvc_env.common.constants import ContactState
    from tvc_env.envs.base_env import BaseEnvConfig
    from tvc_env.envs.direct_rl_env import TVCDirectRLEnv

    sim_root = Path(__file__).parents[2]
    config = BaseEnvConfig(
        task_name="landing",
        env_config_path=sim_root / "configs/env/single_env_debug.yaml",
        disturbance_config_path=sim_root / "configs/disturbances/nominal.yaml",
        overrides={
            "task": {
                "spawn": {
                    "position_range": [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                    "velocity_range": [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
                    "attitude_range": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    "curriculum": {"enabled": False},
                },
                "contact": {"dwell_frames": 2, "min_contact_force": 1.0},
                "termination": {
                    "crash": True,
                    "max_impact_speed": 0.5,
                    "max_tilt": 1.57,
                    "max_tilt_at_contact": 0.5,
                    "max_angular_rate_at_contact": 3.0,
                    "max_altitude_error": 10.0,
                },
            }
        },
        sim_root=sim_root,
    )
    env = TVCDirectRLEnv(config)
    try:
        env.reset(seed=7)
        action = torch.zeros(1, 5, device=env.device)
        for _ in range(120):
            _, _, terminated, truncated, info = env.step(action)
            if bool((terminated | truncated)[0]):
                break
        else:
            pytest.fail("Drop did not produce a terminal PhysX contact within four seconds")

        assert int(info["contact_state_pre_reset"][0]) == int(ContactState.CRASHED)
        assert float(info["touchdown_speed_pre_reset"][0]) > 0.5
    finally:
        env.close()
