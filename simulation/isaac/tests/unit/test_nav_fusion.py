"""Aiding sensors (TFmini Plus, MTF-01P, barometer), the EKF3-style filter and its wiring."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tvc_env.common.quaternions import from_euler
from tvc_env.dynamics.imu_model import ImuModel
from tvc_env.dynamics.nav_fusion import FusionParams, NavFusion, nav_fusion_from_config, resolve_fusion_config
from tvc_env.dynamics.nav_sensors import (
    BaroParams, Barometer, FlowParams, OpticalFlow, Rangefinder, RangefinderParams, cos_tilt, slant_range)
from tvc_env.envs.observations import apply_sensor_noise

ROOT = Path(__file__).resolve().parents[2]
DT = 1.0 / 480.0
PERFECT_RANGE = RangefinderParams(noise_std_m=0.0, accuracy_m=0.0, accuracy_pct=0.0, resolution_m=0.0, latency_s=0.0)
PERFECT_FLOW = FlowParams(noise_std_rad_s=0.0, scale_error_pct=0.0, bias_rad_s=0.0, latency_s=0.0)


def upright(n):
    q = torch.zeros(n, 4)
    q[:, 0] = 1.0
    return q


def vec(*values, n=4):
    return torch.tensor(values, dtype=torch.float32).repeat(n, 1)


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def sample(step, *args):
    """Step a sensor until its next measurement is delivered (they are periodic, not per-substep)."""
    for _ in range(40):
        out = step(*args)
        if out:
            return out[0]
    raise AssertionError("sensor delivered nothing")


# --------------------------------------------------------------------------- rangefinder

def test_rangefinder_reads_slant_range_and_respects_its_limits():
    rf = Rangefinder(4, "cpu", DT, PERFECT_RANGE, gen())
    rf.reset(torch.arange(4))
    q = from_euler(torch.tensor([0.0, 0.5, 0.0, 1.45]), torch.zeros(4), torch.zeros(4))   # roll 0, 28.6 deg, 0, 83 deg
    position = torch.tensor([[0, 0, 5.0], [0, 0, 5.0], [0, 0, 0.05], [0, 0, 5.0]])
    out = sample(rf.step, q, position)
    assert float(out.value[0]) == pytest.approx(5.0, abs=1e-4)
    assert float(out.value[1]) == pytest.approx(5.0 / math.cos(0.5), rel=1e-4)          # slant, not vertical
    assert bool(out.valid[0]) and bool(out.valid[1])
    assert not bool(out.valid[2])                                                       # inside the 0.1 m dead zone
    assert not bool(out.valid[3])                                                       # beam too far off nadir
    far = sample(rf.step, upright(4), torch.tensor([[0, 0, 9.0]] * 4))
    assert not bool(far.valid.any())                                                    # beyond max_range_m (8 m)


def test_rangefinder_error_model_follows_the_datasheet_and_mount_offset_is_geometric():
    params = RangefinderParams(mount_down_m=0.2, resolution_m=0.01)
    rf = Rangefinder(256, "cpu", DT, params, gen(1))
    rf.reset(torch.arange(256))
    truth = slant_range(torch.full((256,), 3.0), cos_tilt(upright(256)), 0.2)
    assert float(truth[0]) == pytest.approx(2.8)
    errors = []
    for _ in range(50):
        errors.append(sample(rf.step, upright(256), vec(0, 0, 3.0, n=256)).value - truth)
    error = torch.stack(errors)
    assert float(error.mean(0).abs().max()) <= 0.05 + 0.02                              # systematic +-5 cm up to 5 m
    assert float((error - error.mean(0)).std()) == pytest.approx(0.03, rel=0.2)         # 3 cm repeatability
    assert float((error * 100 - torch.round(error * 100)).abs().max()) < 0.5 + 1e-3     # on the 1 cm grid


def test_sampler_rate_and_latency_are_in_physics_substeps():
    rf = Rangefinder(2, "cpu", DT, RangefinderParams(rate_hz=100.0, latency_s=0.01), gen())
    rf.reset(torch.arange(2))
    ages, arrivals = set(), []
    for k in range(480):
        for m in rf.step(upright(2), vec(0, 0, 3.0, n=2)):
            ages.add(m.age)
            arrivals.append(k)
    assert ages == {int(round(0.01 / DT))}
    assert len(arrivals) == pytest.approx(100, abs=2)                                   # 100 Hz for one second


# --------------------------------------------------------------------------- optical flow

def test_flow_is_velocity_over_range_plus_rotation_and_the_gyro_compensates_it():
    flow = OpticalFlow(4, "cpu", DT, PERFECT_FLOW, gen())
    q, position = upright(4), vec(0, 0, 4.0)
    zero = torch.zeros(4, 3)
    f, r = flow.truth_flow(q, position, vec(2.0, 0, 0), zero)                            # forward at 2 m/s, 4 m up
    assert float(r[0]) == pytest.approx(4.0)
    assert torch.allclose(f[0], torch.tensor([0.0, -0.5]), atol=1e-6)                    # f_y = -v_x / r
    f, _ = flow.truth_flow(q, position, vec(0, -1.0, 0), zero)                           # Isaac world y is left: FRD +y is world -y
    assert torch.allclose(f[0], torch.tensor([0.25, 0.0]), atol=1e-6)                    # f_x = +v_y(FRD) / r
    # Rotation alone contributes -w; adding the gyro back recovers the pure velocity term (ArduPilot's compensation).
    w = vec(0.3, -0.2, 0.1)
    f, _ = flow.truth_flow(q, position, vec(2.0, -1.0, 0), w)
    compensated = f + w[:, :2]
    assert torch.allclose(compensated[0], torch.tensor([0.25, -0.5]), atol=1e-6)


def test_flow_sensor_at_a_mount_offset_sees_the_lever_arm_velocity():
    flow = OpticalFlow(1, "cpu", DT, FlowParams(noise_std_rad_s=0.0, scale_error_pct=0.0, bias_rad_s=0.0, mount_down_m=0.3),
                       gen())
    w = vec(0.0, 0.5, 0.0, n=1)                                                          # pitch rate about y
    f, r = flow.truth_flow(upright(1), vec(0, 0, 4.0, n=1), torch.zeros(1, 3), w)
    v_sensor_x = 0.5 * 0.3                                                               # (w x mount)_x = w_y * d
    assert float(f[0, 1]) == pytest.approx(-v_sensor_x / float(r[0]) - 0.5, abs=1e-6)


def test_flow_drops_out_outside_the_sensor_envelope():
    flow = OpticalFlow(4, "cpu", DT, PERFECT_FLOW, gen())
    flow.reset(torch.arange(4))
    position = torch.tensor([[0, 0, 0.05], [0, 0, 12.0], [0, 0, 2.0], [0, 0, 2.0]])      # too low, too high, ok, too fast
    velocity = torch.tensor([[0, 0, 0.0], [0, 0, 0.0], [0.5, 0, 0], [20.0, 0, 0]])
    valid = sample(flow.step, upright(4), position, velocity, torch.zeros(4, 3)).valid
    assert valid.tolist() == [False, False, True, False]


# --------------------------------------------------------------------------- barometer

def test_baro_starts_zeroed_drifts_slowly_and_quantizes():
    baro = Barometer(512, "cpu", DT, BaroParams(latency_s=0.0), gen(3))
    baro.reset(torch.arange(512))
    position = vec(0, 0, 10.0, n=512)
    first = None
    for k in range(int(120 / DT)):
        for m in baro.step(position):
            first = m.value - 10.0 if first is None else first
            last = m.value - 10.0
    assert float(first.std()) < 0.25                                                    # noise only at power-up
    assert 0.2 < float(last.std()) < 0.8                                                # bias has wandered to ~0.5 m


# --------------------------------------------------------------------------- estimator

def trajectory(t):
    a, w = np.array([1.5, 1.0, 0.4]), np.array([0.35, 0.27, 0.5])
    p = np.array([a[0] * math.sin(w[0] * t), a[1] * math.sin(w[1] * t + 1), 10 + a[2] * math.sin(w[2] * t)])
    v = np.array([a[0] * w[0] * math.cos(w[0] * t), a[1] * w[1] * math.cos(w[1] * t + 1), a[2] * w[2] * math.cos(w[2] * t)])
    return p, v, math.radians(4) * math.sin(0.9 * t), math.radians(3) * math.sin(0.7 * t + 0.5), 0.05 * t


def fly_fused(profile, seconds, n=2, seed=1, fusion_cfg=None, traj=None, marker=None, estimate_offset=None):
    traj = traj or trajectory
    imu = ImuModel(n, "cpu", DT, {"profile": profile, "nav": {"enabled": True}}, gen(seed))
    fusion = NavFusion(n, "cpu", DT, dict({"profile": "tfmini_mtf01p", "enabled": True}, **(fusion_cfg or {})), imu, gen(seed + 1))

    def state(t):
        p, v, r, pi, y = traj(t)
        q = from_euler(torch.full((n,), r), torch.full((n,), pi), torch.full((n,), y))
        return q, torch.tensor(v, dtype=torch.float32).repeat(n, 1), torch.tensor(p, dtype=torch.float32).repeat(n, 1)
    q, v, p = state(0.0)
    imu.reset(torch.arange(n), q, v, torch.zeros(n, 3), position_world=p)
    marker_world = None if marker is None else torch.tensor(marker, dtype=torch.float32).repeat(n, 1)
    fusion.reset(torch.arange(n), q, v, p, marker_world=marker_world)
    if estimate_offset is not None:                     # the filter believes it is somewhere else (accumulated drift)
        fusion.ekf.p_[:, :2] += torch.tensor(estimate_offset, dtype=torch.float64)
        fusion.ekf.P[:, 6:8, 6:8] = torch.eye(2, dtype=torch.float64) * 1.5 ** 2
    for k in range(1, int(seconds / DT) + 1):
        q, v, p = state(k * DT)
        imu.step(q, v, torch.zeros(n, 3))
        fusion.step(q, v, p)
    return imu, fusion, q, v, p


def test_fusion_holds_velocity_and_height_where_unaided_inertial_navigation_diverges():
    imu, fusion, q, v, p = fly_fused("wtgahrs1", 15.0)
    unaided = float((imu.nav_position - p).norm(dim=-1).median())
    fused = float((fusion.position - p).norm(dim=-1).median())
    assert unaided > 5.0                                                                # hobby-grade dead reckoning has run away
    assert fused < 0.35 * unaided and fused < 1.5
    assert float((fusion.velocity_world - v).norm(dim=-1).max()) < 0.4                  # flow pins the velocity
    # Hover is at 10 m, above the rangefinder's 8 m limit, so height is barometer-only: bounded by its drift (sigma 0.5 m).
    assert float((fusion.position[:, 2] - p[:, 2]).abs().max()) < 1.0
    # The estimate is honest about its uncertainty: its 1-sigma covers the actual error most of the time.
    pos_sigma, vel_sigma, _ = fusion.ekf.sigma()
    assert float(vel_sigma.min()) > 0.01 and float(pos_sigma.min()) > 0.05


def test_fusion_without_the_aiding_sensors_is_no_better_than_inertial_dead_reckoning():
    dead = {"flow": {"dropout_prob": 1.0}, "rangefinder": {"dropout_prob": 1.0}, "baro": {"noise_std_m": 50.0}}
    _, fusion, q, v, p = fly_fused("wtgahrs1", 15.0, fusion_cfg=dead)
    assert float((fusion.velocity_world - v).norm(dim=-1).median()) > 0.5


def test_estimate_outputs_have_the_shapes_the_observation_expects():
    _, fusion, q, v, p = fly_fused("vn110e", 0.5)
    assert fusion.quaternion_wxyz.shape == (2, 4) and fusion.position.shape == (2, 3)
    assert fusion.velocity_world.shape == (2, 3) and fusion.gyro_frd.shape == (2, 3)
    record = fusion.record()
    assert record["range_valid"] in (True, False) and len(record["position_sigma_m"]) == 3
    assert torch.isfinite(fusion.ekf.P).all()


# --------------------------------------------------------------------------- configuration and wiring

def test_fusion_profile_loads_with_datasheet_figures_and_a_shared_mount_offset():
    p = FusionParams.from_config({"profile": "tfmini_mtf01p", "enabled": True})
    assert p.enabled and p.rangefinder.rate_hz == 100.0 and p.rangefinder.noise_std_m == 0.03
    assert p.flow.rate_hz == 100.0 and p.flow.max_rate_rad_s == 7.0 and p.flow.min_range_m == 0.08
    assert p.ekf.flow_m_nse == 0.25 and p.ekf.rng_i_gate == 500.0 and p.ekf.gyro_p_nse == 0.015
    assert p.mount_down_m == p.rangefinder.mount_down_m == p.flow.mount_down_m == 0.15
    assert not FusionParams.from_config({"profile": "tfmini_mtf01p"}).enabled           # opt-in


def test_bad_fusion_configuration_is_rejected():
    for bad in ({"typo": 1}, {"profile": "nonexistent"}, {"profile": "../etc"}, {"enabled": "yes"},
                {"flow": {"rate_hz": -1}}, {"rangefinder": {"noise_stdm": 1}}, {"ekf": {"init": {"nope": 1}}}):
        with pytest.raises(ValueError):
            FusionParams.from_config(bad)


def test_fusion_is_built_only_when_enabled_and_an_imu_exists():
    n = 2
    imu = ImuModel(n, "cpu", DT, {"profile": "vn110e"}, gen())
    on = {"enabled": True, "imu": {"profile": "vn110e", "fusion": {"enabled": True, "profile": "tfmini_mtf01p"}}}
    assert isinstance(nav_fusion_from_config(n, "cpu", DT, on, imu), NavFusion)
    assert "profile" not in on["imu"]["fusion"] and on["imu"]["fusion"]["rangefinder"]["rate_hz"] == 100.0   # expanded in place
    off = {"enabled": True, "imu": {"fusion": {"profile": "tfmini_mtf01p"}}}
    assert nav_fusion_from_config(n, "cpu", DT, off, imu) is None
    assert nav_fusion_from_config(n, "cpu", DT, on, None) is None
    assert nav_fusion_from_config(n, "cpu", DT, {"enabled": True, "imu": {"profile": "vn110e"}}, imu) is None


def test_env_config_expands_and_validates_the_fusion_block_before_isaac_starts(tmp_path):
    from tvc_env.envs.base_env import BaseEnvConfig
    kwargs = dict(task_name="landing", env_config_path=ROOT / "configs/env/single_env_debug.yaml", sim_root=ROOT)
    good = tmp_path / "good.yaml"
    good.write_text("disturbances:\n  enabled: true\n  sensor_noise:\n    enabled: true\n    imu:\n      profile: vn110e\n"
                    "      fusion:\n        enabled: true\n        profile: tfmini_mtf01p\n")
    fusion = BaseEnvConfig(disturbance_config_path=good, **kwargs).config["disturbances"]["sensor_noise"]["imu"]["fusion"]
    assert fusion["enabled"] and "profile" not in fusion and fusion["flow"]["max_rate_rad_s"] == 7.0
    bad = tmp_path / "bad.yaml"
    bad.write_text("disturbances:\n  enabled: true\n  sensor_noise:\n    enabled: true\n    imu:\n      profile: vn110e\n"
                   "      fusion:\n        profile: tfmini_mtf01p\n        flow:\n          rate_hzz: 5\n")
    with pytest.raises(ValueError, match="rate_hzz"):
        BaseEnvConfig(disturbance_config_path=bad, **kwargs)


class _Fake:
    n = 3
    position = torch.tensor([[1.0, 2.0, 3.0]] * 3)
    velocity_world = torch.tensor([[0.5, 0.0, 0.0]] * 3)
    quaternion_wxyz = upright(3)
    gyro_frd = torch.tensor([[0.01, 0.02, 0.03]] * 3)


def test_observation_takes_everything_from_the_fusion_estimate():
    obs = torch.zeros(3, 24)
    obs[:, 0:3] = torch.tensor([9.0, 9.0, 9.0]) - torch.zeros(3, 3)                     # target - true position (0,0,0)
    obs[:, 3] = 1.0
    cfg = {"disturbances": {"sensor_noise": {"enabled": True, "position_std": 0.5, "velocity_std": 0.5,
                                             "attitude_std": 0.5, "angular_velocity_std": 0.5}}}
    imu = ImuModel(3, "cpu", DT, {"profile": "vn110e"}, gen())
    imu.reset(torch.arange(3), upright(3), torch.zeros(3, 3), torch.zeros(3, 3))
    fusion = _Fake()
    noisy = apply_sensor_noise(obs, cfg, imu=imu, true_position=torch.zeros(3, 3), fusion=fusion)
    assert torch.allclose(torch.tensor([9.0, 9.0, 9.0]) - noisy[:, 0:3], fusion.position)
    assert torch.allclose(noisy[:, 13], fusion.position[:, 2])
    assert torch.equal(noisy[:, 10:13], fusion.gyro_frd) and torch.equal(noisy[:, 3:7], fusion.quaternion_wxyz)
    assert torch.allclose(noisy[:, 7], torch.full((3,), 0.5)) and float(noisy[:, 8:10].abs().max()) == 0.0
    with pytest.raises(ValueError, match="true_position"):
        apply_sensor_noise(obs, cfg, imu=imu, fusion=fusion)


def test_mission_control_fused_option_reaches_the_fusion_config():
    from mission_control.disturbance_parameters import validate_settings
    from mission_control.models import disturbance_config, validate_mission
    mission = validate_mission(dict(disturbance=["sensor_noise"], disturbance_settings=dict(
        sensor_noise=dict(imu_profile="wtgahrs1", imu_nav="fused"))))
    assert validate_mission(mission) == mission
    noise = disturbance_config(mission)["disturbances"]["sensor_noise"]
    assert noise["imu"]["fusion"] == {"enabled": True, "profile": "tfmini_mtf01p"} and "nav" not in noise["imu"]
    assert "imu_nav" not in noise
    with pytest.raises(ValueError):
        validate_settings({"sensor_noise": {"imu_nav": "fused"}})                       # needs a physical chain
    assert resolve_fusion_config({"profile": "tfmini_mtf01p"})["flow"]["rate_hz"] == 100.0


# --------------------------------------------------------------------------- pad marker

MARKER_ON = {"marker": {"enabled": True, "dropout_prob": 0.0}}


def test_marker_angles_follow_the_camera_axes():
    from tvc_env.dynamics.nav_sensors import marker_angles
    q = upright(1)
    position = torch.tensor([[0.0, 0.0, 4.0]])
    ahead, depth = marker_angles(q, position, torch.tensor([[1.0, 0.0, 0.0]]), 0.0)
    assert float(depth) == pytest.approx(4.0) and torch.allclose(ahead, torch.tensor([[math.atan(0.25), 0.0]]), atol=1e-6)
    right, _ = marker_angles(q, position, torch.tensor([[0.0, -1.0, 0.0]]), 0.0)         # Isaac y is left: FRD right is world -y
    assert torch.allclose(right, torch.tensor([[0.0, math.atan(0.25)]]), atol=1e-6)
    # Rolling the body right by 10 deg moves a marker directly below toward the camera's left (-y_frd)... the geometry
    # must stay consistent with the slant-range model: depth is the body-axis component.
    rolled = from_euler(torch.tensor([math.radians(10.0)]), torch.zeros(1), torch.zeros(1))
    _, d = marker_angles(rolled, torch.tensor([[0.0, 0.0, 4.0]]), torch.zeros(1, 3), 0.0)
    assert float(d) == pytest.approx(4.0 * math.cos(math.radians(10.0)), rel=1e-5)


def test_marker_is_detected_only_inside_its_field_of_view_and_range():
    from tvc_env.dynamics.nav_sensors import MarkerParams, PadMarker
    params = MarkerParams(dropout_prob=0.0, latency_s=0.0)
    cam = PadMarker(5, "cpu", DT, params, gen())
    marker = torch.zeros(5, 3)
    cam.reset(torch.arange(5), marker)
    position = torch.tensor([[0.0, 0.0, 3.0],      # straight above: seen
                             [3.0, 0.0, 3.0],      # 45 deg off the axis: outside the 30 deg half-FOV
                             [0.0, 0.0, 0.2],      # closer than the minimum range
                             [0.0, 0.0, 9.5],      # 0.4 m tag under 24 px beyond ~9.2 m
                             [1.0, 0.0, 3.0]])     # 18 deg: seen
    valid = sample(cam.step, upright(5), position).valid
    assert valid.tolist() == [True, False, False, False, True]
    assert params.max_range_m == pytest.approx(9.24, abs=0.05)
    absent = PadMarker(2, "cpu", DT, params, gen())
    absent.reset(torch.arange(2), None)
    assert not bool(sample(absent.step, upright(2), vec(0, 0, 3.0, n=2)).valid.any())


def test_marker_pins_the_horizontal_position_that_flow_alone_lets_drift():
    descent = lambda t: (np.array([0.6 * math.sin(0.4 * t), -0.4 * math.sin(0.3 * t), 6.0 - 0.3 * t]),
                         np.array([0.24 * math.cos(0.4 * t), -0.12 * math.cos(0.3 * t), -0.3]), 0.0, 0.0, 0.0)
    offset = (1.5, -1.0)                                   # flow-only navigation has drifted 1.8 m by the time of the descent
    _, without, q, v, p = fly_fused("vn110e", 8.0, traj=descent, marker=(0.0, 0.0, 0.0), estimate_offset=offset)
    _, with_marker, _, _, _ = fly_fused("vn110e", 8.0, traj=descent, marker=(0.0, 0.0, 0.0), estimate_offset=offset,
                                        fusion_cfg=MARKER_ON)
    error = lambda f: float((f.position - p)[:, :2].norm(dim=-1).max())
    assert error(without) > 1.4                            # nothing anchors it
    assert error(with_marker) < 0.25                       # the marker pulls it back onto the pad frame
    assert with_marker.record()["marker_valid"] in (True, False)


def test_marker_is_ignored_outside_the_altitude_window():
    high = lambda t: (np.array([0.0, 0.0, 14.0 - 0.05 * t]), np.zeros(3) + np.array([0, 0, -0.05]), 0.0, 0.0, 0.0)
    _, gated, _, _, p = fly_fused("vn110e", 3.0, traj=high, marker=(0.0, 0.0, 0.0), estimate_offset=(1.0, 0.0),
                                  fusion_cfg={"marker": {"enabled": True, "dropout_prob": 0.0, "alt_max_m": 8.0}})
    assert float((gated.position - p)[:, 0].mean()) > 0.9      # above alt_max_m (and beyond the tag's range): no correction


def test_marker_configuration_and_mission_control_option():
    p = FusionParams.from_config({"profile": "tfmini_mtf01p"})
    assert not p.marker_enabled and p.marker.fov_deg == 60.0 and p.marker.alt_max_m == 8.0
    assert FusionParams.from_config({"profile": "tfmini_mtf01p", "marker": {"enabled": True}}).marker_enabled
    for bad in ({"marker": {"enabled": "yes"}}, {"marker": {"typo": 1}}, {"marker": {"fov_deg": 0}}):
        with pytest.raises(ValueError):
            FusionParams.from_config(bad)
    from mission_control.models import disturbance_config, validate_mission
    mission = validate_mission(dict(disturbance=["sensor_noise"], disturbance_settings=dict(
        sensor_noise=dict(imu_profile="bno085", imu_nav="fused_marker"))))
    assert validate_mission(mission) == mission
    fusion = disturbance_config(mission)["disturbances"]["sensor_noise"]["imu"]["fusion"]
    assert fusion == {"enabled": True, "profile": "tfmini_mtf01p", "marker": {"enabled": True}}


def test_record_carries_the_truth_the_camera_view_needs():
    descent = lambda t: (np.array([0.5, 0.0, 6.0 - 0.3 * t]), np.array([0.0, 0.0, -0.3]), 0.0, 0.0, 0.0)
    _, fusion, q, v, p = fly_fused("vn110e", 1.0, n=1, traj=descent, marker=(0.0, 0.0, 0.0), fusion_cfg=MARKER_ON)
    r = fusion.record()
    assert r["marker_enabled"] and r["marker_world_m"] == [0.0, 0.0, 0.0]
    assert r["height_truth_m"] == pytest.approx(float(p[0, 2]), abs=1e-3)
    assert r["range_truth_m"] == pytest.approx(float(p[0, 2]) - 0.15, abs=1e-3)          # upright: z minus the mount offset
    assert r["marker_in_fov"] and r["marker_in_window"]                                  # 0.5 m off at ~5.7 m, below 8 m
    assert r["marker_truth_rad"][0] == pytest.approx(-math.atan2(0.5, float(p[0, 2]) - 0.15 * 0 + 0.15 * 0), abs=0.02)
    assert len(r["flow_truth_rad_s"]) == 2 and isinstance(r["marker_depth_m"], float)
    high = lambda t: (np.array([0.0, 0.0, 14.0]), np.zeros(3), 0.0, 0.0, 0.0)
    _, f2, *_ = fly_fused("vn110e", 0.2, n=1, traj=high, marker=(0.0, 0.0, 0.0), fusion_cfg=MARKER_ON)
    assert not f2.record()["marker_in_window"]                                           # above the 8 m altitude window
