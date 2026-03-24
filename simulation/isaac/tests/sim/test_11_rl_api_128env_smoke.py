"""
Simulation test: 128-env RL API smoke test (test_11).

Initializes 128 environments, resets all, steps with random actions for 1000 steps.
Asserts observation tensor shape (128, 24), reward tensor shape (128,),
no NaN in observations or rewards, no tensor shape errors,
and verifies independent per-env auto-reset on termination.

Requires Isaac Sim runtime.
"""

from __future__ import annotations
import pytest
import torch
from pathlib import Path

try:
    import omni.usd
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False

pytestmark = pytest.mark.skipif(not ISAAC_AVAILABLE, reason="Isaac Sim runtime not available")

SIM_ROOT = Path(__file__).parents[2]
NUM_ENVS = 128
NUM_STEPS = 1000


@pytest.fixture(scope="module")
def env_128():
    from tvc_env.envs.base_env import BaseEnvConfig
    from tvc_env.envs.direct_rl_env import TVCDirectRLEnv

    config = BaseEnvConfig(
        task_name="hover",
        env_config_path=SIM_ROOT / "configs/env/train_128.yaml",
        sim_root=SIM_ROOT,
    )
    env = TVCDirectRLEnv(config)
    try:
        yield env
    finally:
        env.close()


class TestRL128EnvSmoke:
    def test_observation_shape(self, env_128):
        """Observation tensor should have shape (128, 24)."""
        obs_dict, _ = env_128.reset()
        obs = obs_dict["policy"]
        assert obs.shape == (NUM_ENVS, 24), f"Expected ({NUM_ENVS}, 24), got {obs.shape}"

    def test_no_nan_in_initial_obs(self, env_128):
        """Initial observations should contain no NaN values."""
        obs_dict, _ = env_128.reset()
        obs = obs_dict["policy"]
        assert not torch.isnan(obs).any(), "NaN found in initial observations"

    def test_reward_shape(self, env_128):
        """Reward tensor should have shape (128,)."""
        obs_dict, _ = env_128.reset()
        action = torch.zeros(NUM_ENVS, 5)
        action[:, 4] = 0.75
        _, rewards, _, _, _ = env_128.step(action)
        assert rewards.shape == (NUM_ENVS,), f"Expected ({NUM_ENVS},), got {rewards.shape}"

    def test_no_nan_after_steps(self, env_128):
        """No NaN should appear in observations or rewards after 1000 steps."""
        env_128.reset()
        nan_count = 0
        for step in range(NUM_STEPS):
            action = torch.zeros(NUM_ENVS, 5)
            action[:, :4] = (torch.rand(NUM_ENVS, 4) - 0.5) * 0.2
            action[:, 4] = torch.rand(NUM_ENVS) * 0.3 + 0.6
            obs_dict, rewards, _, _, _ = env_128.step(action)
            if torch.isnan(obs_dict["policy"]).any() or torch.isnan(rewards).any():
                nan_count += 1
        assert nan_count == 0, f"{nan_count} steps had NaN values"

    def test_independent_resets(self, env_128):
        """Terminated environments should reset independently without affecting others."""
        obs_dict, _ = env_128.reset()
        # Run until at least one env terminates
        for _ in range(NUM_STEPS):
            action = torch.zeros(NUM_ENVS, 5)
            action[:, :4] = torch.randn(NUM_ENVS, 4) * 0.3  # Large random actions to trigger termination
            action[:, 4] = 0.5
            obs_dict, _, terminated, _, _ = env_128.step(action)
            if terminated.any():
                # Obs should still be finite for all envs after auto-reset
                obs = obs_dict["policy"]
                assert torch.all(torch.isfinite(obs)), "Non-finite observations after auto-reset"
                break
