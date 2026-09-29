"""The three IMU hardware profiles (WTGAHRS1, BNO085, VN-110E) and the temperature-drift term.

These check that each profile loads, carries its datasheet figures, and runs a stable chain. They are
NOT a static accuracy validation: that needs bench data or a scripted-manoeuvre fit, and is deferred.
"""

import math

import pytest
import torch

from tvc_env.dynamics.imu_model import STANDARD_GRAVITY as G, ImuModel, ImuParams

DT = 1.0 / 480.0   # the environment's physics step
PROFILES = ("wtgahrs1", "bno085", "vn110e")


def make(cfg, n=32, seed=0):
    return ImuModel(n, "cpu", DT, cfg, torch.Generator().manual_seed(seed))


def upright(n):
    q = torch.zeros(n, 4)
    q[:, 0] = 1.0
    return q


def hold(model, seconds, w=None):
    """Keep the vehicle motionless and upright (optionally spinning body rate ``w`` without attitude change)."""
    n = model.n
    q, v = upright(n), torch.zeros(n, 3)
    w = torch.zeros(n, 3) if w is None else w
    model.reset(torch.arange(n), q, v, w)
    for _ in range(int(seconds / DT)):
        model.step(q, v, w)
    return q


def tilt_error(model, q):
    return 2.0 * torch.acos((model.quaternion_wxyz * q).sum(-1).abs().clamp(max=1.0))


@pytest.mark.parametrize("name", PROFILES)
def test_each_profile_loads_and_survives_a_static_run(name):
    model = make({"profile": name})
    q = hold(model, 5.0)
    for tensor in (model.gyro_frd, model.quaternion_wxyz, model.accel_frd):
        assert torch.isfinite(tensor).all()
    assert float(model.gyro_frd.abs().max()) < math.radians(2.0)     # at rest: bias plus noise only
    assert float(tilt_error(model, q).max()) < math.radians(5.0)


def test_wtgahrs1_carries_the_manual_figures():
    p = ImuParams.from_config({"profile": "wtgahrs1"})
    assert (p.sample_rate_hz, p.bandwidth_hz) == (200.0, 20.0)
    assert p.gyro.saturation == pytest.approx(math.radians(2000.0))
    assert p.accel.saturation == pytest.approx(16 * G)
    assert p.accel.turn_on == pytest.approx(2e-3 * G)                # manual p.16, after accelerometer calibration


def test_bno085_carries_the_bmi085_datasheet_figures():
    p = ImuParams.from_config({"profile": "bno085"})
    assert p.gyro.noise_density == pytest.approx(math.radians(0.014))
    assert p.gyro.saturation == pytest.approx(math.radians(2000.0))
    assert p.gyro.g_sensitivity == pytest.approx(math.radians(0.1) / G)
    assert p.gyro.tco == pytest.approx(math.radians(0.015))
    assert p.accel.turn_on == pytest.approx(20e-3 * G)
    assert p.accel.tco == pytest.approx(0.2e-3 * G)
    assert p.accel.noise_density == pytest.approx(135e-6 * G)
    assert p.attitude.yaw_mode == "gyro"                             # game rotation vector has no magnetometer


def test_vn110e_carries_the_vn110_datasheet_figures():
    p = ImuParams.from_config({"profile": "vn110e"})
    assert p.sample_rate_hz == 400.0 and p.bandwidth_hz == 240.0
    assert p.gyro.saturation == pytest.approx(math.radians(490.0))
    assert p.accel.saturation == pytest.approx(15 * G)
    assert p.gyro.noise_density == pytest.approx(math.radians(5.0 / 3600.0), rel=0.01)
    assert p.accel.noise_density == pytest.approx(40e-6 * G)
    assert p.gyro.misalignment == pytest.approx(math.radians(0.05), rel=0.01)   # cross-axis < 0.05 deg
    # Accelerometer turn-on half-width chosen so its rms reproduces the 0.05 deg RMS static pitch/roll.
    static_tilt_deg = math.degrees(math.asin(p.accel.turn_on / math.sqrt(3.0) / G))
    assert static_tilt_deg == pytest.approx(0.05, rel=0.1)


def test_vn110e_gyro_saturates_at_its_490_dps_range():
    cfg = {"profile": "vn110e", "rate_truth": "reported", "gyro": {"noise_density_dps_rthz": 0.0, "bias_instability_dps": 0.0,
                                         "turn_on_bias_dps": 0.0, "scale_factor_pct": 0.0,
                                         "misalignment_mrad": 0.0, "g_sensitivity_dps_per_g": 0.0}}
    model = make(cfg, n=4)
    hold(model, 0.5, w=torch.tensor([[10.0, 0.0, 0.0]]).repeat(4, 1))         # 573 dps
    assert torch.allclose(model.gyro_frd[:, 0], torch.full((4,), math.radians(490.0)), rtol=1e-3)


def test_tactical_vn110e_is_much_quieter_at_rest_than_the_two_hobby_parts():
    # Only the tactical part is ranked: the BNO085's 116 Hz gyro bandwidth passes more noise than
    # the WTGAHRS1's 20 Hz, so hobby-versus-hobby ordering is not a grade ordering.
    rate_rms = {}
    for name in PROFILES:
        model = make({"profile": name}, n=64, seed=3)
        hold(model, 3.0)
        rate_rms[name] = float(model.gyro_frd.pow(2).mean().sqrt())
    assert rate_rms["vn110e"] * 3.0 < min(rate_rms["bno085"], rate_rms["wtgahrs1"])


def test_profile_sample_rates_and_latency_fit_the_physics_step():
    for name in PROFILES:
        model = make({"profile": name}, n=2)
        assert model._rate <= 1.0 / DT + 1e-9
        assert model._delay >= 1                                     # every profile carries at least one substep


# --------------------------------------------------------------------------- temperature drift

THERMAL = {"sample_rate_hz": 480.0, "thermal": {"start_spread_k": 0.0, "self_heating_k": 10.0, "warmup_tau_s": 1.0}}


def test_gyro_tco_turns_a_warm_up_into_a_bias_ramp():
    model = make(dict(THERMAL, gyro={"tco_dps_per_k": 0.1}), n=64)
    q = upright(64)
    v, w = torch.zeros(64, 3), torch.zeros(64, 3)
    model.reset(torch.arange(64), q, v, w)
    assert float(model.gyro_frd.abs().max()) == 0.0                  # power-up at the calibration point
    for _ in range(int(5.0 / DT)):                                   # five time constants
        model.step(q, v, w)
    assert float(model.temperature_k.mean()) == pytest.approx(10.0 * (1.0 - math.exp(-5.0)), rel=1e-2)
    limit = math.radians(0.1) * float(model.temperature_k.max())
    assert float(model.gyro_frd.abs().max()) <= limit + 1e-9         # bounded by the coefficient's half-width
    assert float(model.gyro_frd.abs().max()) > 0.5 * limit           # and it does reach that order


def test_accel_tco_moves_the_accelerometer_register():
    model = make(dict(THERMAL, accel={"tco_mg_per_k": 1.0}), n=64)
    hold(model, 5.0)
    expected_max = 1e-3 * G * float(model.temperature_k.max())
    down = model.accel_frd - torch.tensor([0.0, 0.0, -G])            # at rest FRD z reads -g
    assert float(down.abs().max()) <= expected_max + 1e-6
    assert float(down.abs().max()) > 0.5 * expected_max


def test_no_temperature_coefficient_means_no_thermal_state_change():
    model = make(dict(THERMAL, gyro={"tco_dps_per_k": 0.0}), n=4)
    hold(model, 2.0)
    assert float(model.temperature_k.abs().max()) == 0.0             # never advanced: no coefficient to drive


def test_power_up_temperature_is_redrawn_per_reset_within_the_spread():
    model = make({"sample_rate_hz": 480.0, "gyro": {"tco_dps_per_k": 0.01},
                  "thermal": {"start_spread_k": 5.0, "self_heating_k": 0.0, "warmup_tau_s": 10.0}}, n=256)
    q, v, w = upright(256), torch.zeros(256, 3), torch.zeros(256, 3)
    model.reset(torch.arange(256), q, v, w)
    first = model.temperature_k.clone()
    assert float(first.abs().max()) <= 5.0 and float(first.std()) > 2.0
    model.reset(torch.arange(128), q, v, w)                          # only the first half is power-cycled
    assert not torch.equal(model.temperature_k[:128], first[:128])
    assert torch.equal(model.temperature_k[128:], first[128:])


def test_thermal_configuration_is_validated():
    for bad in ({"thermal": {"typo": 1.0}}, {"thermal": {"warmup_tau_s": 0.0}},
                {"thermal": {"self_heating_k": -1.0}}, {"gyro": {"tco_dps_per_k": -0.1}}):
        with pytest.raises(ValueError):
            ImuParams.from_config(bad)


@pytest.mark.parametrize("name", PROFILES)
def test_disturbance_config_for_each_profile_builds_a_model(name):
    import yaml
    from pathlib import Path
    from tvc_env.dynamics.imu_model import imu_model_from_config
    path = Path(__file__).resolve().parents[2] / "configs" / "disturbances" / f"sensor_imu_{name}.yaml"
    noise = yaml.safe_load(path.read_text())["disturbances"]["sensor_noise"]
    assert noise["imu"]["profile"] == name
    assert isinstance(imu_model_from_config(2, "cpu", DT, noise), ImuModel)
