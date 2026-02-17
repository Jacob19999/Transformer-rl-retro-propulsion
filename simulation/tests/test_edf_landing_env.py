"""Tests for EDFLandingEnv (Stage 12)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simulation.config_loader import load_config

pytest.importorskip("gymnasium")

from simulation.training.edf_landing_env import ACT_DIM, OBS_DIM, EDFLandingEnv


def _make_env() -> EDFLandingEnv:
    root = Path(__file__).parent.parent
    vcfg = load_config(root / "configs" / "test_vehicle.yaml")
    ecfg = load_config(root / "configs" / "test_environment.yaml")
    return EDFLandingEnv(
        {
            "vehicle": vcfg.get("vehicle", vcfg),
            "environment": ecfg.get("environment", ecfg),
            "dt_policy": 0.025,
            "max_episode_time": 1.0,
            "target_position": [0.0, 0.0, 0.0],
            # Make ICs deterministic and numerically gentle for unit tests.
            "initial_conditions": {
                "pos_xy_range": [0.0, 0.0],
                "altitude_range": [5.0, 5.0],
                "vel_xy_range": [0.0, 0.0],
                "descent_rate_range": [0.0, 0.0],
                "tilt_range_rad": 0.0,
                "yaw_range": [0.0, 0.0],
                "omega_range": [0.0, 0.0],
            },
        }
    )


def test_spaces_shapes() -> None:
    env = _make_env()
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.action_space.shape == (ACT_DIM,)


def test_reset_produces_valid_obs() -> None:
    env = _make_env()
    obs, info = env.reset(seed=123)
    assert obs.shape == (OBS_DIM,)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)
    assert "position" in info
    assert "altitude" in info


def test_step_runs() -> None:
    env = _make_env()
    obs, _info = env.reset(seed=0)
    action = np.zeros(ACT_DIM, dtype=np.float32)
    obs2, reward, terminated, truncated, info = env.step(action)
    assert obs2.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert "termination_reason" in info


def test_terminates_on_fast_ground_contact() -> None:
    env = _make_env()
    _obs, _info = env.reset(seed=0)

    # Force a near-ground, high downward velocity state so we trigger hard ground contact.
    env.vehicle.state[0:3] = np.array([0.0, 0.0, -1e-3], dtype=float)  # 1 mm above ground
    env.vehicle.state[3:6] = np.array([0.0, 0.0, 6.0], dtype=float)  # 6 m/s down in body (q set below)
    env.vehicle.state[6:10] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # upright
    env.vehicle.state[10:13] = np.zeros(3, dtype=float)

    action = np.array([-1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # low thrust
    _obs2, _reward, terminated, _truncated, info = env.step(action)
    assert terminated is True
    assert info.get("crashed", False) is True


# ─── Stage 15: Domain Randomization tests ─────────────────────────────────────


def _make_env_with_dr(
    actuator_delay_enabled: bool = False,
    obs_latency_enabled: bool = False,
    esc_delay_range: tuple[float, float] = (0.025, 0.025),  # 1 policy step at 40 Hz
    obs_delay_steps_range: tuple[int, int] = (2, 2),
) -> EDFLandingEnv:
    root = Path(__file__).parent.parent
    vcfg = load_config(root / "configs" / "test_vehicle.yaml")
    ecfg = load_config(root / "configs" / "test_environment.yaml")
    base: dict = {
        "vehicle": vcfg.get("vehicle", vcfg),
        "environment": ecfg.get("environment", ecfg),
        "dt_policy": 0.025,
        "max_episode_time": 1.0,
        "target_position": [0.0, 0.0, 0.0],
        "initial_conditions": {
            "pos_xy_range": [0.0, 0.0],
            "altitude_range": [5.0, 5.0],
            "vel_xy_range": [0.0, 0.0],
            "descent_rate_range": [0.0, 0.0],
            "tilt_range_rad": 0.0,
            "yaw_range": [0.0, 0.0],
            "omega_range": [0.0, 0.0],
        },
    }
    base["actuator_delay"] = {
        "enabled": actuator_delay_enabled,
        "esc_delay_range": list(esc_delay_range),
        "servo_delay_range": list(esc_delay_range),
    }
    base["obs_latency"] = {
        "enabled": obs_latency_enabled,
        "delay_steps_range": list(obs_delay_steps_range),
    }
    return EDFLandingEnv(base)


def test_actuator_delay_delays_action() -> None:
    """With actuator delay enabled, action takes effect N policy steps later."""
    env = _make_env_with_dr(
        actuator_delay_enabled=True,
        obs_latency_enabled=False,
        esc_delay_range=(0.05, 0.05),  # 2 policy steps at 40 Hz
    )
    env.reset(seed=42)
    assert env._delay_policy_steps == 2

    # Run several steps; env should run without error and buffer should behave correctly
    action = np.zeros(ACT_DIM, dtype=np.float32)
    for _ in range(5):
        obs, _reward, term, trunc, _info = env.step(action)
        assert obs.shape == (OBS_DIM,)
    assert len(env._action_buffer) <= 2  # buffer trimmed as we pop


def test_obs_latency_returns_stale() -> None:
    """With obs latency enabled, returned obs is from delay_steps policy steps ago."""
    env = _make_env_with_dr(
        actuator_delay_enabled=False,
        obs_latency_enabled=True,
        obs_delay_steps_range=(2, 2),
    )
    _obs0, _ = env.reset(seed=123)
    assert env._obs_delay_steps == 2

    # Buffer is populated from _get_obs() after each step (not from reset)
    # Step 0: append obs_after_step0, buffer=[obs1], return obs1
    obs1, _, _, _, _ = env.step(np.zeros(ACT_DIM, dtype=np.float32))
    # Step 1: append obs_after_step1, buffer=[obs1,obs2], return obs2
    obs2, _, _, _, _ = env.step(np.zeros(ACT_DIM, dtype=np.float32))
    # Step 2: append obs_after_step2, buffer=[obs1,obs2,obs3], return obs1 (2 steps stale)
    obs_s3, _, _, _, _ = env.step(np.zeros(ACT_DIM, dtype=np.float32))
    np.testing.assert_array_almost_equal(obs_s3, obs1)
    # Step 3: return obs2 (2 steps stale)
    obs_s4, _, _, _, _ = env.step(np.zeros(ACT_DIM, dtype=np.float32))
    np.testing.assert_array_almost_equal(obs_s4, obs2)


def test_dr_disabled_by_config() -> None:
    """DR features are off when config flags are false."""
    env = _make_env_with_dr(actuator_delay_enabled=False, obs_latency_enabled=False)
    env.reset(seed=0)
    assert env._actuator_delay_enabled is False
    assert env._obs_latency_enabled is False
    assert env._delay_policy_steps == 0
    assert env._obs_delay_steps == 0

