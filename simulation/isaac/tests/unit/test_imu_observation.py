"""The simulated IMU replaces white attitude/rate noise in both observation builders."""

import math
from pathlib import Path

import pytest
import torch
import yaml

from tvc_env.common.quaternions import from_euler
from tvc_env.dynamics.imu_model import ImuModel, imu_model_from_config
from tvc_env.envs import waypoint_flight as wf
from tvc_env.envs.observations import apply_sensor_noise
from tvc_env.envs.task_registry import load_merged_config

ROOT = Path(__file__).resolve().parents[2]
BIASED = {"sample_rate_hz": 1000.0, "gyro": {"turn_on_bias_dps": 30.0}}


def biased_imu(n):
    model = ImuModel(n, "cpu", 0.002, BIASED, torch.Generator().manual_seed(4))
    q = torch.zeros(n, 4)
    q[:, 0] = 1.0
    model.reset(torch.arange(n), q, torch.zeros(n, 3), torch.zeros(n, 3))
    model.step(q, torch.zeros(n, 3), torch.zeros(n, 3))
    return model


def test_legacy_observation_takes_attitude_and_rates_from_the_imu_not_white_noise():
    n = 3
    obs = torch.zeros(n, 24)
    obs[:, 3] = 1.0
    config = {"disturbances": {"sensor_noise": {"enabled": True, "position_std": 0.0, "velocity_std": 0.0,
                                                "attitude_std": 0.5, "angular_velocity_std": 0.5}}}
    imu = biased_imu(n)
    noisy = apply_sensor_noise(obs, config, imu=imu)
    assert torch.equal(noisy[:, 3:7], imu.quaternion_wxyz)
    assert torch.equal(noisy[:, 10:13], imu.gyro_frd)
    assert noisy[:, 10:13].abs().max() > 0.1            # the sensor's bias, not the 0.5 rad/s white noise
    assert torch.equal(noisy[:, 7:10], obs[:, 7:10])   # velocity_std = 0 still leaves velocity alone


def test_legacy_observation_still_adds_position_and_velocity_noise_with_an_imu():
    obs = torch.zeros(64, 24)
    obs[:, 3] = 1.0
    config = {"disturbances": {"sensor_noise": {"enabled": True, "position_std": 0.1, "velocity_std": 0.1}}}
    noisy = apply_sensor_noise(obs, config, imu=biased_imu(64))
    assert noisy[:, 0:3].abs().max() > 0 and noisy[:, 7:10].abs().max() > 0


def test_waypoint_observation_uses_the_imu_and_records_it_as_the_measurement():
    env = yaml.safe_load((ROOT / "configs/env/train_waypoint_flight.yaml").read_text())
    cfg = load_merged_config("waypoint_flight", env_config=env, sim_root=ROOT)
    flight = wf.WaypointFlightTask(2, "cpu", cfg, torch.zeros(2, 3), 1 / 30)
    flight.set_explicit_missions([[dict(position=[10, 0, 5], type="flypass", radius_m=1.0),
                                   dict(position=[10, 0, 1], type="land")]] * 2)
    start = torch.tensor([[0.0, 0.0, 5.0]] * 2)
    flight.reset(torch.arange(2), start)
    imu = biased_imu(2)
    q = from_euler(torch.zeros(2), torch.zeros(2), torch.zeros(2))
    args = (start, q, torch.zeros(2, 3), torch.zeros(2, 3), start[:, 2], torch.zeros(2, 4), torch.zeros(2, 4),
            torch.full((2,), 0.83), torch.zeros(2, dtype=torch.long), torch.zeros(2, 4), torch.zeros(2, 5),
            0.262, 6.98)
    noise = dict(enabled=True, position_std=0.0, velocity_std=0.0, attitude_std=0.3, angular_velocity_std=0.3)
    obs = flight.observation(*args, noise, imu=imu)
    m = flight.measurement
    assert torch.equal(m["angular_vel_frd"], imu.gyro_frd) and torch.equal(m["quaternion_wxyz"], imu.quaternion_wxyz)
    assert obs[0, 28:31].tolist() == pytest.approx((imu.gyro_frd[0] / math.pi).tolist(), abs=1e-6)
    # Noise disabled: the IMU is ignored and truth passes through.
    flight.observation(*args, None, imu=imu)
    assert torch.equal(flight.measurement["quaternion_wxyz"], q)


def test_wtgahrs1_disturbance_file_builds_the_model_and_expands_its_profile_in_place():
    disturbance = yaml.safe_load((ROOT / "configs/disturbances/sensor_imu_wtgahrs1.yaml").read_text())
    sensor_noise = disturbance["disturbances"]["sensor_noise"]
    assert sensor_noise["imu"]["profile"] == "wtgahrs1"
    model = imu_model_from_config(2, "cpu", 1 / 480, sensor_noise)
    assert isinstance(model, ImuModel)
    assert "profile" not in sensor_noise["imu"] and sensor_noise["imu"]["gyro"]["range_dps"] == 2000
    assert imu_model_from_config(2, "cpu", 1 / 480, sensor_noise) is not None     # idempotent


def test_env_config_expands_and_validates_the_imu_profile_before_isaac_starts(tmp_path):
    from tvc_env.envs.base_env import BaseEnvConfig
    kwargs = dict(task_name="waypoint_flight", env_config_path=ROOT / "configs/env/train_waypoint_flight.yaml",
                  sim_root=ROOT)
    config = BaseEnvConfig(disturbance_config_path=ROOT / "configs/disturbances/sensor_imu_wtgahrs1.yaml", **kwargs)
    imu = config.config["disturbances"]["sensor_noise"]["imu"]
    assert "profile" not in imu and imu["bandwidth_hz"] == 20.0 and imu["gyro"]["range_dps"] == 2000
    bad = tmp_path / "bad.yaml"
    bad.write_text("disturbances:\n  enabled: true\n  sensor_noise:\n    enabled: true\n    imu:\n      profile: wtgahrs1\n"
                   "      gyro:\n        noise_densty: 1\n")
    with pytest.raises(ValueError, match="noise_densty"):
        BaseEnvConfig(disturbance_config_path=bad, **kwargs)
    nominal = BaseEnvConfig(disturbance_config_path=ROOT / "configs/disturbances/nominal.yaml", **kwargs)
    assert "imu" not in nominal.config["disturbances"]["sensor_noise"]
