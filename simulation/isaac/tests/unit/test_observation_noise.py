"""Unit tests for disturbance-backed observation noise."""

import torch

from tvc_env.envs.observations import apply_sensor_noise, measured_state


def test_disabled_sensor_noise_returns_same_tensor():
    obs = torch.zeros(2, 24)
    assert apply_sensor_noise(obs, {"disturbances": {"sensor_noise": {"enabled": False}}}) is obs


def test_position_noise_keeps_position_error_and_height_consistent():
    obs = torch.zeros(4, 24)
    config = {
        "disturbances": {
            "sensor_noise": {
                "enabled": True,
                "position_std": 0.1,
                "velocity_std": 0.0,
                "attitude_std": 0.0,
                "angular_velocity_std": 0.0,
            }
        }
    }
    torch.manual_seed(7)
    noisy = apply_sensor_noise(obs, config)
    assert not torch.equal(noisy, obs)
    # target - measured_position changes by the negative of measurement noise,
    # while measured height changes by its positive z component.
    assert torch.allclose(noisy[:, 2], -noisy[:, 13])


def test_attitude_noise_preserves_unit_quaternion():
    obs = torch.zeros(8, 24)
    obs[:, 3] = 1.0
    config = {
        "disturbances": {
            "sensor_noise": {
                "enabled": True,
                "position_std": 0.0,
                "velocity_std": 0.0,
                "attitude_std": 0.01,
                "angular_velocity_std": 0.0,
            }
        }
    }
    noisy = apply_sensor_noise(obs, config)
    assert torch.allclose(noisy[:, 3:7].norm(dim=-1), torch.ones(8), atol=1e-6)


def test_measured_state_inverts_the_noisy_observation():
    obs = torch.zeros(3, 24)
    obs[:, 3] = 1.0
    target = torch.tensor([1.0, 2.0, 3.0])
    config = {"disturbances": {"sensor_noise": {"enabled": True, "position_std": 0.1, "velocity_std": 0.1,
                                                "attitude_std": 0.01, "angular_velocity_std": 0.1}}}
    torch.manual_seed(11)
    noisy = apply_sensor_noise(obs, config)
    measured = measured_state(noisy, target)
    # obs holds target - position, so the believed position is target - obs[0:3];
    # its z matches the believed height the observation reports.
    assert torch.allclose(measured["position"], target - noisy[:, 0:3])
    assert torch.allclose(measured["position"][:, 2] - target[2], noisy[:, 13])
    assert torch.equal(measured["quaternion_wxyz"], noisy[:, 3:7])
    assert torch.equal(measured["angular_vel_frd"], noisy[:, 10:13])
