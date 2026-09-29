"""Unit tests for the batched IMU error chain, sensor timing and attitude filter."""

import math

import pytest
import torch

from tvc_env.common.frames import frd_to_isaac, isaac_to_frd
from tvc_env.common.quaternions import from_euler, inverse, multiply, normalize, rotate_vector, to_euler
from tvc_env.dynamics.imu_model import (
    GM_PEAK_ALLAN_RATIO, STANDARD_GRAVITY as G, ImuModel, ImuParams, imu_model_from_config, resolve_imu_config)

DT = 0.002


def make(cfg, n=8, dt=DT, seed=0):
    return ImuModel(n, "cpu", dt, cfg, torch.Generator().manual_seed(seed))


def upright(n):
    q = torch.zeros(n, 4)
    q[:, 0] = 1.0
    return q


def start(model, q=None, v=None, w=None):
    n = model.n
    q = upright(n) if q is None else q
    v = torch.zeros(n, 3) if v is None else v
    w = torch.zeros(n, 3) if w is None else w
    model.reset(torch.arange(n), q, v, w)
    return q, v, w


def propagate(q, w_frd, dt):
    """Isaac convention: q rotates body local axes into the world."""
    rate = torch.cat((torch.zeros_like(w_frd[:, :1]), frd_to_isaac(w_frd)), dim=-1)
    return normalize(q + 0.5 * dt * multiply(q, rate))


def angle_between(a, b):
    return 2.0 * torch.acos((a * b).sum(-1).abs().clamp(max=1.0))


def rms(x):
    return float(x.pow(2).mean().sqrt())


IDEAL = {"sample_rate_hz": 1000.0}


# --------------------------------------------------------------------------- error chain

def test_ideal_configuration_passes_body_rate_through_exactly():
    model = make(IDEAL)
    w = torch.tensor([[0.1, -0.2, 0.3]]).repeat(model.n, 1)
    q, v, _ = start(model, w=w)
    for _ in range(3):
        model.step(q, v, w)
    assert torch.allclose(model.gyro_frd, w, atol=1e-7)


def test_white_noise_sigma_follows_a_one_sided_density():
    density = 0.01  # deg/s/sqrt(Hz)
    model = make({**IDEAL, "gyro": {"noise_density_dps_rthz": density}}, n=20000)
    q, v, w = start(model)
    model.step(q, v, w)
    expected = math.radians(density) / math.sqrt(2.0 * DT)   # variance = density^2 / (2 dt)
    assert rms(model.gyro_frd) == pytest.approx(expected, rel=0.03)


def test_lowpass_shrinks_white_noise_by_the_discrete_first_order_gain():
    density, fc = 0.01, 20.0
    model = make({**IDEAL, "bandwidth_hz": fc, "gyro": {"noise_density_dps_rthz": density}}, n=20000)
    q, v, w = start(model)
    for _ in range(200):
        model.step(q, v, w)
    alpha = 1.0 - math.exp(-2.0 * math.pi * fc * DT)
    expected = math.radians(density) / math.sqrt(2.0 * DT) * math.sqrt(alpha / (2.0 - alpha))
    assert rms(model.gyro_frd) == pytest.approx(expected, rel=0.04)
    # Continuous-time check: noise-equivalent bandwidth of a first-order low-pass is (pi/2) fc.
    assert expected == pytest.approx(math.radians(density) * math.sqrt(math.pi / 2 * fc), rel=0.03)


def test_lowpass_attenuates_a_tone_at_the_cutoff_to_about_minus_three_db():
    dt, fc = 0.0005, 20.0
    model = make({"sample_rate_hz": 1 / dt, "bandwidth_hz": fc}, n=1, dt=dt)
    q, v, _ = start(model)
    out = []
    for k in range(1600):
        w = torch.tensor([[math.sin(2 * math.pi * fc * k * dt), 0.0, 0.0]])
        model.step(q, v, w)
        out.append(float(model.gyro_frd[0, 0]))
    assert max(out[800:]) == pytest.approx(0.707, abs=0.03)


def test_gauss_markov_bias_has_target_sigma_and_correlation_time():
    floor, tau, dt = 0.05, 1.0, 0.01
    model = make({"sample_rate_hz": 100.0, "gyro": {"bias_instability_dps": floor,
                                                   "bias_correlation_time_s": tau}}, n=20000, dt=dt)
    start(model)
    sigma = math.radians(floor) / GM_PEAK_ALLAN_RATIO
    x0 = model._gyro.gm.clone()
    assert rms(x0) == pytest.approx(sigma, rel=0.03)
    q, v, w = upright(model.n), torch.zeros(model.n, 3), torch.zeros(model.n, 3)
    for _ in range(int(tau / dt)):
        model.step(q, v, w)
    xt = model._gyro.gm
    assert rms(xt) == pytest.approx(sigma, rel=0.03)               # stays stationary
    corr = float((x0 * xt).mean() / (rms(x0) * rms(xt)))
    assert corr == pytest.approx(math.exp(-1.0), abs=0.03)


def test_gauss_markov_peak_allan_ratio_constant():
    # Allan variance of a first-order Gauss-Markov process with std 1 and tau 1.
    u = torch.linspace(0.05, 20.0, 40000, dtype=torch.float64)
    variance = 2.0 / u ** 2 * (u - 2.0 * (1.0 - torch.exp(-u)) + 0.5 * (1.0 - torch.exp(-2.0 * u)))
    assert float(variance.sqrt().max()) == pytest.approx(GM_PEAK_ALLAN_RATIO, abs=1e-3)
    assert int(variance.argmax()) * (20.0 - 0.05) / 39999 + 0.05 == pytest.approx(1.89, abs=0.02)


def test_random_walk_bias_grows_as_k_root_t():
    k_dps, dt, steps = 0.01, 0.01, 400
    model = make({"sample_rate_hz": 100.0, "gyro": {"rate_random_walk_dps_rt_s": k_dps}}, n=20000, dt=dt)
    q, v, w = start(model)
    for _ in range(steps):
        model.step(q, v, w)
    expected = math.radians(k_dps) * math.sqrt(steps * dt)
    assert rms(model._gyro.walk) == pytest.approx(expected, rel=0.03)


def test_turn_on_bias_is_constant_in_an_episode_and_redrawn_only_for_reset_envs():
    model = make({**IDEAL, "gyro": {"turn_on_bias_dps": 0.5}}, n=6)
    q, v, w = start(model)
    model.step(q, v, w)
    first = model.gyro_frd.clone()
    assert first.abs().max() > 0
    for _ in range(5):
        model.step(q, v, w)
    assert torch.allclose(model.gyro_frd, first)
    assert first.abs().max() <= math.radians(0.5) + 1e-6
    model.reset(torch.tensor([0, 1]), q, v, w)
    assert not torch.allclose(model.gyro_frd[:2], first[:2])
    assert torch.allclose(model.gyro_frd[2:], first[2:])


def test_saturation_and_quantization():
    cfg = {**IDEAL, "gyro": {"range_dps": 100.0, "resolution_dps": 0.5}}
    model = make(cfg, n=2)
    q, v, _ = start(model)
    w = torch.tensor([[math.radians(300.0), 0.0, 0.0], [math.radians(0.26), 0.0, 0.0]])
    model.step(q[:2], v[:2], w)
    assert math.degrees(float(model.gyro_frd[0, 0])) == pytest.approx(100.0, abs=1e-4)
    assert math.degrees(float(model.gyro_frd[1, 0])) == pytest.approx(0.5, abs=1e-4)
    steps = torch.round(torch.rad2deg(model.gyro_frd) / 0.5)
    assert torch.allclose(torch.rad2deg(model.gyro_frd), steps * 0.5, atol=1e-4)


def test_scale_and_misalignment_stay_within_their_configured_bounds():
    model = make({**IDEAL, "gyro": {"scale_factor_pct": 2.0, "misalignment_mrad": 10.0}}, n=2000)
    start(model)
    m = model._gyro.matrix
    diag = torch.diagonal(m, dim1=1, dim2=2)
    off = m - torch.diag_embed(diag)
    assert (diag - 1.0).abs().max() <= 0.02 + 1e-6 and (diag - 1.0).abs().max() > 0.015
    assert off.abs().max() <= 0.010 + 1e-6 and off.abs().max() > 0.008


def test_gyro_g_sensitivity_couples_specific_force_into_the_rate():
    model = make({**IDEAL, "gyro": {"g_sensitivity_dps_per_g": 0.5}}, n=4)
    q, v, w = start(model)
    for _ in range(2):
        model.step(q, v, w)          # rest: specific force is (0, 0, -g) in FRD
    coefficient = model._gyro.g_sens
    expected = coefficient * torch.tensor([0.0, 0.0, -G])
    assert torch.allclose(model.gyro_frd, expected, atol=1e-7)
    assert model.gyro_frd.abs().max() > 0


# --------------------------------------------------------------------------- timing

def test_output_rate_holds_the_last_sample_between_ticks():
    dt = 0.01
    model = make({"sample_rate_hz": 10.0}, n=1, dt=dt)
    q, v, _ = start(model)
    seen = []
    for k in range(1, 51):
        model.step(q, v, torch.tensor([[0.001 * k, 0.0, 0.0]]))
        seen.append(float(model.gyro_frd[0, 0]))
    changes = [i for i in range(1, len(seen)) if seen[i] != seen[i - 1]]
    assert changes == [9, 19, 29, 39, 49]        # a new sample every 10 substeps


def test_latency_delays_the_measurement_by_whole_substeps():
    dt = 0.01
    model = make({"sample_rate_hz": 100.0, "latency_s": 0.03}, n=1, dt=dt)
    q, v, _ = start(model)
    out = []
    for k in range(10):
        w = torch.tensor([[1.0 if k >= 5 else 0.0, 0.0, 0.0]])
        model.step(q, v, w)
        out.append(float(model.gyro_frd[0, 0]))
    assert out == [0.0] * 8 + [1.0] * 2          # true step at k=5 shows up at k=8


def test_reset_flushes_the_delay_line_of_the_previous_episode():
    dt = 0.01
    model = make({"sample_rate_hz": 100.0, "latency_s": 0.05}, n=2, dt=dt)
    q, v, _ = start(model)
    hot = torch.full((2, 3), 2.0)
    for _ in range(8):
        model.step(q, v, hot)
    model.reset(torch.tensor([0]), q, v, torch.zeros(2, 3))
    assert float(model.gyro_frd[0, 0]) == 0.0 and float(model.gyro_frd[1, 0]) > 1.0
    model.step(q, v, hot)
    assert float(model.gyro_frd[0, 0]) == 0.0    # stale samples did not leak into env 0


def test_outputs_are_not_aliased_to_internal_buffers():
    model = make({"sample_rate_hz": 500.0}, n=2)
    q, v, _ = start(model)
    model.step(q, v, torch.full((2, 3), 0.5))
    held, snapshot = model.gyro_frd, model.gyro_frd.clone()
    for _ in range(5):
        model.step(q, v, torch.full((2, 3), -0.7))
    assert torch.equal(held, snapshot)


# --------------------------------------------------------------------------- specific force

def test_accelerometer_reads_minus_g_on_the_down_axis_at_rest_and_zero_in_free_fall():
    model = make(IDEAL, n=2)
    q, v, w = start(model)
    for _ in range(3):
        model.step(q, v, w)
    assert torch.allclose(model.accel_frd, torch.tensor([0.0, 0.0, -G]).repeat(2, 1), atol=1e-4)
    for k in range(1, 4):
        v = torch.tensor([0.0, 0.0, -G * DT * k]).repeat(2, 1)
        model.step(q, v, w)
    assert model.accel_frd.abs().max() < 1e-3    # free fall: no specific force


def test_accelerometer_direction_follows_attitude():
    q = normalize(from_euler(torch.tensor([0.3]), torch.tensor([-0.2]), torch.tensor([0.9])))
    model = make(IDEAL, n=1)
    start(model, q=q)
    model.step(q, torch.zeros(1, 3), torch.zeros(1, 3))
    expected = isaac_to_frd(rotate_vector(inverse(q), torch.tensor([[0.0, 0.0, G]])))
    assert torch.allclose(model.accel_frd, expected, atol=1e-4)


# --------------------------------------------------------------------------- attitude filter

def run_rotation(model, q, w, steps, v=None):
    v = torch.zeros(model.n, 3) if v is None else v
    for _ in range(steps):
        q = propagate(q, w, model.dt)
        model.step(q, v, w)
    return q


def test_attitude_filter_converges_to_truth_from_a_tilt_error():
    dt = 0.005
    model = make({"sample_rate_hz": 200.0, "attitude": {"kp": 2.0}}, n=4, dt=dt)
    q = normalize(from_euler(torch.full((4,), 0.3), torch.full((4,), -0.2), torch.zeros(4)))
    start(model, q=q)
    model._q_est = normalize(multiply(q, from_euler(torch.full((4,), 0.1), torch.full((4,), -0.08), torch.zeros(4))))
    run_rotation(model, q, torch.zeros(4, 3), 1500)
    assert float(torch.rad2deg(angle_between(model.quaternion_wxyz, q)).max()) < 0.2


def test_gyro_only_yaw_is_not_corrected_but_tilt_is():
    dt = 0.005
    model = make({"sample_rate_hz": 200.0, "attitude": {"kp": 2.0}}, n=2, dt=dt)
    q = upright(2)
    start(model, q=q)
    yaw_error = math.radians(5.0)
    model._q_est = normalize(multiply(q, from_euler(torch.full((2,), 0.05), torch.zeros(2),
                                                     torch.full((2,), yaw_error))))
    run_rotation(model, q, torch.zeros(2, 3), 1500)
    roll, pitch, yaw = to_euler(model.quaternion_wxyz)
    assert float(roll.abs().max()) < math.radians(0.2) and float(pitch.abs().max()) < math.radians(0.2)
    assert float(yaw.mean()) == pytest.approx(yaw_error, abs=math.radians(0.3))


def test_magnetometer_heading_pulls_yaw_to_the_measured_heading():
    dt = 0.005
    cfg = {"sample_rate_hz": 200.0, "attitude": {"kp": 2.0, "yaw": {"mode": "magnetometer", "gain": 2.0}}}
    model = make(cfg, n=2, dt=dt)
    q = upright(2)
    start(model, q=q)
    model._q_est = normalize(multiply(q, from_euler(torch.zeros(2), torch.zeros(2), torch.full((2,), 0.2))))
    run_rotation(model, q, torch.zeros(2, 3), 2000)
    yaw = to_euler(model.quaternion_wxyz)[2]
    assert float(yaw.abs().max()) < math.radians(0.3)


def test_gyro_bias_becomes_linear_yaw_drift_while_tilt_stays_bounded():
    dt, seconds = 0.01, 60.0
    model = make({"sample_rate_hz": 100.0, "gyro": {"turn_on_bias_dps": 0.2}}, n=6, dt=dt)
    q = upright(6)
    start(model, q=q)
    bias_frd = model._gyro.turn_on.clone()
    run_rotation(model, q, torch.zeros(6, 3), int(seconds / dt))
    roll, pitch, yaw = to_euler(model.quaternion_wxyz)
    # FRD z is down, Isaac z is up: the yaw error carries the opposite sign.
    assert torch.allclose(yaw, -bias_frd[:, 2] * seconds, atol=math.radians(0.05), rtol=0.03)
    assert float(roll.abs().max()) < math.radians(0.3) and float(pitch.abs().max()) < math.radians(0.3)


def test_accelerometer_gate_rejects_gravity_reference_under_hard_acceleration():
    dt = 0.005
    q = upright(1)
    gentle, hard = 0.15 * G, 1.0 * G                 # sustained horizontal acceleration
    errors = {}
    for name, a in (("gentle", gentle), ("hard", hard)):
        model = make({"sample_rate_hz": 200.0, "attitude": {"kp": 2.0, "accel_gate": 0.3}}, n=1, dt=dt)
        start(model, q=q)
        v = torch.zeros(1, 3)
        for _ in range(800):
            v = v + torch.tensor([[a * dt, 0.0, 0.0]])
            model.step(q, v, torch.zeros(1, 3))
        errors[name] = float(torch.rad2deg(angle_between(model.quaternion_wxyz, q)))
    assert errors["gentle"] > 3.0        # thrust acceleration is mistaken for a tilt
    assert errors["hard"] < 0.5          # outside the gate: gyro-only, no bias


# --------------------------------------------------------------------------- configuration

def test_profile_loads_and_inline_keys_override_it():
    params = ImuParams.from_config({"profile": "wtgahrs1", "bandwidth_hz": 40.0, "latency_s": 0.0})
    assert params.sample_rate_hz == 200.0 and params.bandwidth_hz == 40.0 and params.latency_s == 0.0
    assert params.gyro.saturation == pytest.approx(math.radians(2000.0))
    assert params.gyro.resolution == pytest.approx(math.radians(0.061))
    assert params.attitude.yaw_mode == "gyro"


def test_bad_configuration_is_rejected():
    for bad in ({"gyro": {"noise_densty": 1}}, {"typo": 1}, {"attitude": {"yaw": {"mode": "compass"}}},
                {"gyro": {"range_dps": -1}}, {"sample_rate_hz": float("nan")}, {"profile": "../etc"},
                {"profile": "nonexistent"}):
        with pytest.raises(ValueError):
            ImuParams.from_config(bad)


def test_model_is_built_only_when_sensor_noise_and_imu_are_enabled():
    kwargs = dict(num_envs=2, device="cpu", physics_dt=DT)
    assert imu_model_from_config(sensor_noise=None, **kwargs) is None
    assert imu_model_from_config(sensor_noise={"enabled": False, "imu": {"profile": "wtgahrs1"}}, **kwargs) is None
    assert imu_model_from_config(sensor_noise={"enabled": True}, **kwargs) is None
    assert imu_model_from_config(sensor_noise={"enabled": True, "imu": {"enabled": False}}, **kwargs) is None
    assert isinstance(imu_model_from_config(sensor_noise={"enabled": True, "imu": {"profile": "wtgahrs1"}},
                                            **kwargs), ImuModel)


def test_resolve_profile_keeps_inline_nested_overrides():
    cfg = resolve_imu_config({"profile": "wtgahrs1", "gyro": {"turn_on_bias_dps": 0.0}})
    assert cfg["gyro"]["turn_on_bias_dps"] == 0.0 and cfg["gyro"]["range_dps"] == 2000
