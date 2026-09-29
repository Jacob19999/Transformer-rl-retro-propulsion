"""Unaided strapdown navigation: position and velocity integrated from the IMU's own registers."""

import math

import pytest
import torch

from tvc_env.common.frames import isaac_velocity_to_frd
from tvc_env.common.quaternions import from_euler, inverse, normalize, rotate_vector
from tvc_env.dynamics.imu_model import STANDARD_GRAVITY as G, ImuModel, ImuParams
from tvc_env.envs.observations import apply_sensor_noise

DT = 1.0 / 480.0
IDEAL_NAV = {"sample_rate_hz": 480.0, "attitude": {"kp": 0.0, "ki": 0.0}, "nav": {"enabled": True}}


def make(cfg, n=8, seed=0):
    return ImuModel(n, "cpu", DT, cfg, torch.Generator().manual_seed(seed))


def upright(n):
    q = torch.zeros(n, 4)
    q[:, 0] = 1.0
    return q


def fly(model, seconds, accel_world, q=None, start_position=None):
    """Truth: constant world acceleration from rest at ``start_position``; returns truth position, velocity."""
    n = model.n
    q = upright(n) if q is None else q
    p0 = torch.zeros(n, 3) if start_position is None else start_position
    w = torch.zeros(n, 3)
    model.reset(torch.arange(n), q, torch.zeros(n, 3), w, position_world=p0)
    a = torch.tensor(accel_world).repeat(n, 1)
    steps = int(seconds / DT)
    for k in range(1, steps + 1):
        model.step(q, a * (k * DT), w)
    t = steps * DT
    return p0 + 0.5 * a * t * t, a * t


def test_ideal_sensor_integrates_to_the_true_position_and_velocity():
    model = make(IDEAL_NAV)
    p, v = fly(model, 5.0, [1.0, -0.5, 2.0], start_position=torch.tensor([3.0, 4.0, 5.0]).repeat(8, 1))
    assert torch.allclose(model.nav_velocity, v, atol=2e-3)
    assert torch.allclose(model.nav_position, p, atol=2e-3)


def test_free_fall_reads_zero_specific_force_and_still_tracks():
    model = make(IDEAL_NAV)
    p, v = fly(model, 2.0, [0.0, 0.0, -G])
    assert torch.allclose(model.nav_position, p, atol=2e-3)


def test_tilted_vehicle_resolves_specific_force_with_its_attitude():
    q = from_euler(torch.full((8,), 0.4), torch.full((8,), -0.3), torch.full((8,), 1.1))
    model = make(IDEAL_NAV)
    p, v = fly(model, 3.0, [0.7, 0.2, 1.0], q=q)
    assert torch.allclose(model.nav_position, p, atol=3e-3)


def test_navigation_is_off_by_default_and_reset_needs_no_position_then():
    params = ImuParams.from_config({"sample_rate_hz": 480.0})
    assert params.nav.enabled is False
    model = make({"sample_rate_hz": 480.0})
    model.reset(torch.arange(8), upright(8), torch.zeros(8, 3), torch.zeros(8, 3))
    assert not model.nav_enabled and float(model.nav_position.abs().max()) == 0.0


def test_reset_requires_a_position_when_navigation_is_on_and_only_reseeds_the_reset_envs():
    model = make(IDEAL_NAV)
    with pytest.raises(ValueError, match="position_world"):
        model.reset(torch.arange(8), upright(8), torch.zeros(8, 3), torch.zeros(8, 3))
    fly(model, 1.0, [0.0, 0.0, 1.0])
    before = model.nav_position.clone()
    position = torch.full((8, 3), 9.0)
    model.reset(torch.tensor([0, 1]), upright(8), torch.zeros(8, 3), torch.zeros(8, 3), position_world=position)
    assert torch.equal(model.nav_position[:2], position[:2]) and torch.equal(model.nav_position[2:], before[2:])


def test_initial_position_and_velocity_errors_are_drawn_per_reset():
    cfg = dict(IDEAL_NAV, nav={"enabled": True, "initial_position_std_m": 2.0, "initial_velocity_std_m_s": 0.5})
    model = make(cfg, n=512)
    model.reset(torch.arange(512), upright(512), torch.zeros(512, 3), torch.zeros(512, 3),
                position_world=torch.zeros(512, 3))
    assert float(model.nav_position.std()) == pytest.approx(2.0, rel=0.1)
    assert float(model.nav_velocity.std()) == pytest.approx(0.5, rel=0.1)


def test_accelerometer_bias_drifts_position_as_half_b_t_squared():
    cfg = dict(IDEAL_NAV, accel={"turn_on_bias_mg": 5.0})
    model = make(cfg, n=256)
    p, _ = fly(model, 10.0, [0.0, 0.0, 0.0])
    error = (model.nav_position - p).abs().max(dim=-1).values
    limit = 0.5 * 5e-3 * G * 10.0 ** 2                    # 2.45 m at the bias half-width
    assert float(error.max()) <= limit * 1.02
    assert float(error.max()) > 0.7 * limit


def test_tilt_error_leaks_gravity_into_horizontal_position():
    """1 deg (1-sigma per axis) of frozen tilt error => horizontal error rms ~ 0.5*g*sigma*sqrt(2)*t^2."""
    cfg = dict(IDEAL_NAV, attitude={"kp": 0.0, "ki": 0.0, "initial_tilt_error_deg": 1.0})
    model = make(cfg, n=512, seed=2)
    p, _ = fly(model, 10.0, [0.0, 0.0, 0.0])
    horizontal = (model.nav_position - p)[:, :2]
    expected = 0.5 * G * math.radians(1.0) * math.sqrt(2.0) * 10.0 ** 2
    assert float(horizontal.pow(2).sum(-1).mean().sqrt()) == pytest.approx(expected, rel=0.15)
    vertical = (model.nav_position - p)[:, 2]
    assert float(vertical.pow(2).mean().sqrt()) < 0.1 * expected       # only g*(1-cos eps): second order in tilt


def test_latency_delays_the_integrated_solution_as_a_whole():
    lagged = make(dict(IDEAL_NAV, latency_s=0.05))
    p, v = fly(lagged, 2.0, [0.0, 0.0, 1.0])
    # Constant acceleration a delayed by d loses a*d of velocity and a*d*t of position.
    assert torch.allclose(lagged.nav_velocity[:, 2], v[:, 2] - 1.0 * 0.05, atol=0.01)


def test_nav_configuration_is_validated():
    for bad in ({"nav": {"enabled": "yes"}}, {"nav": {"typo": 1}}, {"nav": {"initial_position_std_m": -1.0}}):
        with pytest.raises(ValueError):
            ImuParams.from_config(bad)


# --------------------------------------------------------------------------- observation

def observation_for(model, true_position, target):
    n = model.n
    obs = torch.zeros(n, 24)
    obs[:, 0:3] = target - true_position
    obs[:, 3] = 1.0
    obs[:, 13] = true_position[:, 2]
    cfg = {"disturbances": {"sensor_noise": {"enabled": True, "position_std": 0.5, "velocity_std": 0.5}}}
    return obs, cfg


def test_inertial_observation_uses_the_integrated_position_velocity_and_height():
    model = make(IDEAL_NAV)
    fly(model, 1.0, [0.0, 0.0, 0.0])
    model._nav_p = torch.tensor([1.0, 2.0, 3.0]).repeat(8, 1)                 # the solution has drifted
    model._nav_v = torch.tensor([0.4, -0.2, 0.1]).repeat(8, 1)
    true_position = torch.zeros(8, 3)
    target = torch.tensor([5.0, 5.0, 10.0]).repeat(8, 1)
    obs, cfg = observation_for(model, true_position, target)
    noisy = apply_sensor_noise(obs, cfg, imu=model, true_position=true_position)
    assert torch.allclose(target - noisy[:, 0:3], model.nav_position, atol=1e-6)
    assert torch.allclose(noisy[:, 13], model.nav_position[:, 2])
    q = normalize(model.quaternion_wxyz)
    assert torch.allclose(noisy[:, 7:10], isaac_velocity_to_frd(rotate_vector(inverse(q), model.nav_velocity)), atol=1e-6)
    assert torch.equal(noisy[:, 3:7], model.quaternion_wxyz)                   # attitude still from the chain


def test_inertial_observation_ignores_the_white_position_and_velocity_noise():
    model = make(IDEAL_NAV)
    fly(model, 0.5, [0.0, 0.0, 0.0])
    true_position = model.nav_position.clone()
    obs, cfg = observation_for(model, true_position, torch.zeros(8, 3))
    first = apply_sensor_noise(obs, cfg, imu=model, true_position=true_position)
    second = apply_sensor_noise(obs, cfg, imu=model, true_position=true_position)
    assert torch.equal(first, second)                                          # no random draw
    assert float((first[:, 0:3] + model.nav_position).abs().max()) < 1e-6


def test_inertial_observation_needs_the_true_position_and_external_mode_still_adds_white_noise():
    model = make(IDEAL_NAV)
    fly(model, 0.1, [0.0, 0.0, 0.0])
    obs, cfg = observation_for(model, torch.zeros(8, 3), torch.zeros(8, 3))
    with pytest.raises(ValueError, match="true_position"):
        apply_sensor_noise(obs, cfg, imu=model)
    external = make({"sample_rate_hz": 480.0})
    external.reset(torch.arange(8), upright(8), torch.zeros(8, 3), torch.zeros(8, 3))
    external.step(upright(8), torch.zeros(8, 3), torch.zeros(8, 3))
    noisy = apply_sensor_noise(obs, cfg, imu=external, true_position=torch.zeros(8, 3))
    assert float(noisy[:, 0:3].abs().max()) > 0.0 and float(noisy[:, 7:10].abs().max()) > 0.0


# --------------------------------------------------------------------------- pose-derived truth rate

def rotate_about_body_x(q, rate, dt):
    from tvc_env.common.quaternions import multiply
    omega = torch.zeros(q.shape[0], 4)
    omega[:, 1] = rate
    return normalize(q + 0.5 * dt * multiply(q, omega))


def drive(model, q0, steps, rate, reported_rate=0.0):
    n = model.n
    q = q0.clone()
    v, w = torch.zeros(n, 3), torch.zeros(n, 3)
    w[:, 0] = reported_rate
    model.reset(torch.arange(n), q, v, w, position_world=torch.zeros(n, 3))
    for _ in range(steps):
        q = rotate_about_body_x(q, rate, DT)
        model.step(q, v, w)
    return q


def test_pose_mode_takes_the_rate_from_the_attitude_change_not_the_reported_velocity():
    model = make(dict(IDEAL_NAV, rate_truth="pose"))
    drive(model, upright(8), 200, rate=0.5, reported_rate=0.0)     # the engine reports 0 rad/s while the pose rotates
    assert torch.allclose(model.gyro_frd[:, 0], torch.full((8,), 0.5), atol=1e-3)
    reported = make(dict(IDEAL_NAV, rate_truth="reported"))
    drive(reported, upright(8), 200, rate=0.5, reported_rate=0.0)
    assert float(reported.gyro_frd.abs().max()) == 0.0


def test_pose_mode_keeps_the_estimate_on_the_true_attitude_through_an_unreported_pose_jump():
    def final_error(mode):
        model = make(dict(IDEAL_NAV, rate_truth=mode))
        q = upright(8)
        v, w = torch.zeros(8, 3), torch.zeros(8, 3)
        model.reset(torch.arange(8), q, v, w, position_world=torch.zeros(8, 3))
        for k in range(100):
            if k == 50:                                              # a 1 deg contact-style pose correction, no velocity
                q = normalize(rotate_about_body_x(q, math.radians(1.0) / DT, DT))
            model.step(q, v, w)
        return float(2.0 * torch.acos((model.quaternion_wxyz * q).sum(-1).abs().clamp(max=1.0)).max())
    assert final_error("pose") < math.radians(0.01)
    assert final_error("reported") == pytest.approx(math.radians(1.0), rel=0.05)


def test_pose_rate_is_insensitive_to_quaternion_sign_and_reseeded_by_reset():
    model = make(dict(IDEAL_NAV, rate_truth="pose"))
    q = upright(8)
    v, w = torch.zeros(8, 3), torch.zeros(8, 3)
    model.reset(torch.arange(8), q, v, w, position_world=torch.zeros(8, 3))
    for k in range(20):
        q = rotate_about_body_x(q, 0.3, DT)
        model.step(-q if k % 2 else q, v, w)                        # q and -q are the same orientation
    assert torch.allclose(model.gyro_frd[:, 0], torch.full((8,), 0.3), atol=2e-3)
    jumped = rotate_about_body_x(q, 30.0, DT)                        # an override of the vehicle state...
    model.reset(torch.arange(8), jumped, v, w, position_world=torch.zeros(8, 3))
    model.step(jumped, v, w)                                         # ...must not read as a rotation
    assert float(model.gyro_frd.abs().max()) < 1e-3


def test_rate_truth_configuration_is_validated_and_defaults_to_reported():
    assert ImuParams.from_config({}).rate_truth == "reported"
    assert ImuParams.from_config({"profile": "vn110e"}).rate_truth == "pose"
    with pytest.raises(ValueError, match="rate_truth"):
        ImuParams.from_config({"rate_truth": "magic"})
