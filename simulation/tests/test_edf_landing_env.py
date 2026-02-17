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

