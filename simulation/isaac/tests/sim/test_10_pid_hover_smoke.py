"""
Simulation test: PID hover smoke test (test_10).

Tests:
  - PID controller hovers for 10+ seconds with full physics enabled
  - Position error stays < 0.5 m throughout
  - Tilt stays < 15 deg (0.262 rad) throughout
  - Angular rate stays < 1.0 rad/s on average
  - No NaN in any state variable
  - No ground contact during hover

Requires Isaac Sim runtime.
"""

from __future__ import annotations
import math
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


@pytest.fixture
def pid_env():
    from tvc_env.envs.base_env import BaseEnvConfig
    from tvc_env.envs.direct_rl_env import TVCDirectRLEnv

    config = BaseEnvConfig(
        task_name="hover",
        env_config_path=SIM_ROOT / "configs/env/single_env_debug.yaml",
        sim_root=SIM_ROOT,
    )
    env = TVCDirectRLEnv(config)
    env.reset()
    try:
        yield env
    finally:
        env.close()


class TestPIDHoverSmoke:
    def test_position_error_stable(self, pid_env):
        """PID hover position error should stay below 0.5 m for 10+ seconds."""
        from tvc_env.controllers.pid_adapter import PIDController

        pid = PIDController(num_envs=1, device=pid_env.device)
        obs_dict, _ = pid_env.reset()
        obs = obs_dict["policy"]
        pid.reset()

        dt = 1.0 / 30.0
        n_steps = int(10.0 / dt)  # 10 seconds
        max_pos_err = 0.0

        for _ in range(n_steps):
            action = pid.compute_action(obs)
            obs_dict, _, terminated, truncated, _ = pid_env.step(action)
            obs = obs_dict["policy"]

            pos_err = obs[0, 0:3].norm().item()
            max_pos_err = max(max_pos_err, pos_err)

            assert not torch.isnan(obs).any(), "NaN in observation"

            if (terminated | truncated)[0].item():
                obs_dict, _ = pid_env.reset()
                obs = obs_dict["policy"]
                pid.reset()

        assert max_pos_err < 0.5, f"Max position error {max_pos_err:.3f}m exceeds 0.5m threshold"

    def test_tilt_within_bounds(self, pid_env):
        """Tilt angle should stay below 15 deg (0.262 rad) during PID hover."""
        from tvc_env.controllers.pid_adapter import PIDController
        from tvc_env.common.quaternions import to_euler

        pid = PIDController(num_envs=1, device=pid_env.device)
        obs_dict, _ = pid_env.reset()
        obs = obs_dict["policy"]
        pid.reset()

        dt = 1.0 / 30.0
        n_steps = int(10.0 / dt)
        max_tilt = 0.0

        for _ in range(n_steps):
            action = pid.compute_action(obs)
            obs_dict, _, terminated, truncated, _ = pid_env.step(action)
            obs = obs_dict["policy"]

            quat_wxyz = obs[0, 3:7].unsqueeze(0)
            roll, pitch, _ = to_euler(quat_wxyz)
            tilt = torch.sqrt(roll * roll + pitch * pitch)[0].item()
            max_tilt = max(max_tilt, tilt)

            if (terminated | truncated)[0].item():
                obs_dict, _ = pid_env.reset()
                obs = obs_dict["policy"]
                pid.reset()

        assert max_tilt < 0.262, (
            f"Max tilt {math.degrees(max_tilt):.1f} deg exceeds 15 deg (0.262 rad) threshold"
        )

    def test_angular_rate_bounded(self, pid_env):
        """Mean angular rate should stay below 1.0 rad/s during PID hover."""
        from tvc_env.controllers.pid_adapter import PIDController

        pid = PIDController(num_envs=1, device=pid_env.device)
        obs_dict, _ = pid_env.reset()
        obs = obs_dict["policy"]
        pid.reset()

        dt = 1.0 / 30.0
        n_steps = int(10.0 / dt)
        ang_rates = []

        for _ in range(n_steps):
            action = pid.compute_action(obs)
            obs_dict, _, terminated, truncated, _ = pid_env.step(action)
            obs = obs_dict["policy"]

            ang_rate = obs[0, 10:13].norm().item()
            ang_rates.append(ang_rate)

            if (terminated | truncated)[0].item():
                obs_dict, _ = pid_env.reset()
                obs = obs_dict["policy"]
                pid.reset()

        mean_rate = sum(ang_rates) / len(ang_rates)
        assert mean_rate < 1.0, f"Mean angular rate {mean_rate:.3f} rad/s exceeds 1.0 rad/s threshold"

    def test_no_nan_in_state(self, pid_env):
        """No NaN should appear in obs, action, or reward during PID hover."""
        from tvc_env.controllers.pid_adapter import PIDController

        pid = PIDController(num_envs=1, device=pid_env.device)
        obs_dict, _ = pid_env.reset()
        obs = obs_dict["policy"]
        pid.reset()

        dt = 1.0 / 30.0
        n_steps = int(10.0 / dt)

        for step in range(n_steps):
            action = pid.compute_action(obs)
            assert not torch.isnan(action).any(), f"NaN in action at step {step}"

            obs_dict, rewards, terminated, truncated, _ = pid_env.step(action)
            obs = obs_dict["policy"]

            assert not torch.isnan(obs).any(), f"NaN in obs at step {step}"
            assert not torch.isnan(rewards).any(), f"NaN in rewards at step {step}"

            if (terminated | truncated)[0].item():
                obs_dict, _ = pid_env.reset()
                obs = obs_dict["policy"]
                pid.reset()

    def test_no_ground_contact_while_hovering(self, pid_env):
        """Vehicle should not touch the ground during hover task."""
        from tvc_env.controllers.pid_adapter import PIDController

        pid = PIDController(num_envs=1, device=pid_env.device)
        obs_dict, _ = pid_env.reset()
        obs = obs_dict["policy"]
        pid.reset()

        dt = 1.0 / 30.0
        n_steps = int(10.0 / dt)
        contact_state_idx = 23  # last element of 24-dim obs

        for step in range(n_steps):
            action = pid.compute_action(obs)
            obs_dict, _, terminated, truncated, _ = pid_env.step(action)
            obs = obs_dict["policy"]

            contact_state = obs[0, contact_state_idx].item()
            assert contact_state == 0, (
                f"Ground contact detected at step {step} (contact_state={contact_state:.0f})"
            )

            if (terminated | truncated)[0].item():
                obs_dict, _ = pid_env.reset()
                obs = obs_dict["policy"]
                pid.reset()
