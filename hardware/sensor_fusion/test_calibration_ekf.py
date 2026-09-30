"""Tests for the calibration procedures, mount detection, the EKF3-style filter and the setup API (no hardware).

    python -m pytest hardware/sensor_fusion -q
"""
import math
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest

import calibration as cal
import sim
import tfmini
import witmotion
from calibration import CalibSession, Calibration
from ekf3 import NavEkf
from frames import (DEFAULT_MOUNT, MOUNTS, euler_to_matrix, matrix_to_euler, matrix_to_quat, quat_to_matrix,
                    sensor_attitude_enu)
from fusion import FusionParams
from hub import ROW_FIELDS as ROW, FusionHub

G = witmotion.G


@pytest.fixture(autouse=True)
def _sim_pose_reset():
    sim.CONTROL["pose"] = "moving"
    yield
    sim.CONTROL["pose"] = "moving"


def _hub(tmp=None, **kw):
    """A hub whose calibration lives in a temp dir (never the user's real calibration.json)."""
    path = (tmp or Path(tempfile.mkdtemp())) / "c.json"
    return FusionHub(FusionParams(), history=10 ** 6, calib=kw.pop("calib", None) or Calibration(), calib_path=path, **kw)


def _f_sensor(mount_key, f_vehicle):
    """Specific force as the sensor reads it: vehicle = R_vs @ sensor, so sensor = R_vs^T @ vehicle."""
    return MOUNTS[mount_key].T @ np.asarray(f_vehicle, float)


# ---- frames and mounts -------------------------------------------------------------------------------------
def test_there_are_24_proper_mounts_and_default_is_datasheet_orientation():
    assert len(MOUNTS) == 24 and all(round(np.linalg.det(R)) == 1 for R in MOUNTS.values())
    # WTGAHRS1 flat, Y arrow forward: vehicle forward = sensor Y, right = sensor X, down = -sensor Z
    assert np.allclose(MOUNTS[DEFAULT_MOUNT] @ [0, 1, 0], [1, 0, 0]) and np.allclose(MOUNTS[DEFAULT_MOUNT] @ [0, 0, 1], [0, 0, -1])


def test_frame_roundtrips():
    rng = random.Random(1)
    for _ in range(30):
        rpy = (rng.uniform(-3, 3), rng.uniform(-1.4, 1.4), rng.uniform(-3, 3))
        R = euler_to_matrix(*rpy)
        assert np.allclose(matrix_to_euler(R), rpy, atol=1e-9)
        assert np.allclose(quat_to_matrix(matrix_to_quat(R)), R, atol=1e-9)
    # a level, north-facing vehicle with the default mount reads zero attitude on the sensor's own ENU output
    assert np.allclose(sensor_attitude_enu(np.eye(3), MOUNTS[DEFAULT_MOUNT]), np.eye(3))


@pytest.mark.parametrize("key", sorted(MOUNTS))
def test_mount_detection_recovers_every_mount(key):
    theta = math.radians(60)
    f_level = _f_sensor(key, [0, 0, -G])
    f_nose = _f_sensor(key, [G * math.sin(theta), 0, -G * math.cos(theta)])         # nose pitched up
    got, resid, nose = cal.detect_mount(f_level, f_nose)
    assert got == key and resid < 1e-6 and nose == pytest.approx(60, abs=1e-6)


def test_mount_detection_tolerates_a_tilted_hold_and_finds_the_trim():
    key = "+X+Y+Z"
    R0 = euler_to_matrix(math.radians(3.0), math.radians(-2.0), 0.0)                # the vehicle is really 3 / -2 deg off level
    f_level = _f_sensor(key, R0.T @ [0, 0, -G])
    f_nose = _f_sensor(key, euler_to_matrix(math.radians(3.0), math.radians(45.0), 0.0).T @ [0, 0, -G])
    got, resid, _ = cal.detect_mount(f_level, f_nose)
    assert got == key and resid < 6.0
    roll, pitch = cal.level_trim_deg(MOUNTS[key] @ f_level)
    assert (roll, pitch) == pytest.approx((3.0, -2.0), abs=1e-6)
    c = Calibration(mount=key, trim_deg=[roll, pitch, 0.0])
    assert np.allclose(c.accel_vehicle(f_level), [0, 0, -G], atol=1e-9)             # the trim makes the level pose read level


def test_detect_mount_rejects_a_small_nose_up():
    with pytest.raises(ValueError):
        cal.detect_mount([0, 0, G], [G * math.sin(math.radians(5)), 0, G * math.cos(math.radians(5))])


# ---- fits ----------------------------------------------------------------------------------------------------
def test_gyro_bias_and_still_check():
    rng = np.random.default_rng(0)
    gyro = np.array([0.4, -0.3, 0.2]) + rng.normal(0, 0.05, (500, 3))
    bias, std = cal.fit_gyro_bias(gyro)
    assert bias == pytest.approx([0.4, -0.3, 0.2], abs=0.01) and std == pytest.approx([0.05] * 3, abs=0.01)
    acc = np.array([0.1, 0.2, 9.8]) + rng.normal(0, 0.02, (500, 3))
    assert cal.is_still(acc, gyro) is None
    assert "rotation" in cal.is_still(acc, gyro + np.linspace(0, 30, 500)[:, None])
    assert "accelerometer" in cal.is_still(acc + np.linspace(0, 3, 500)[:, None], gyro)


def test_accel_six_position_recovers_offset_and_scale():
    off, sc = np.array([0.15, -0.10, 0.20]), np.array([1.005, 0.995, 1.003])
    caps = {}
    for i, ax in enumerate("XYZ"):
        for sign in "+-":
            true = np.zeros(3)
            true[i] = G if sign == "+" else -G
            caps[sign + ax] = true / sc + off                                        # raw = truth / scale + offset
            assert cal.classify_face(caps[sign + ax]) == sign + ax
    o, s, done = cal.fit_accel_six(caps)
    assert done == ["X", "Y", "Z"] and o == pytest.approx(off, abs=1e-9) and s == pytest.approx(sc, abs=1e-9)
    assert cal.classify_face([5.0, 5.0, 5.0]) is None and cal.classify_face([0, 0, 3.0]) is None
    o, s, done = cal.fit_accel_six({"+Z": caps["+Z"], "-Z": caps["-Z"]})
    assert done == ["Z"] and s[0] == 1.0                                              # only axes with both faces change


def test_mag_sphere_fit_recovers_hard_and_soft_iron():
    rng = np.random.default_rng(1)
    u = rng.normal(size=(2000, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    off, gain = np.array([30.0, -20.0, 10.0]), np.array([1.10, 0.90, 1.0])
    raw = u * 300.0 * gain + off + rng.normal(0, 2.0, u.shape)
    c, scale, resid, _ = cal.fit_mag_sphere(raw)
    assert c == pytest.approx(off, abs=3.0) and resid < 0.03
    corrected = (raw - c) * scale
    assert corrected.std(axis=0) == pytest.approx([corrected.std(axis=0).mean()] * 3, rel=0.05)     # spherical again
    assert cal.mag_coverage(raw) > 0.9 and cal.mag_coverage(raw[np.abs(u[:, 2]) < 0.2]) < 0.6          # equator only


def test_calibration_json_roundtrip_and_bad_files(tmp_path):
    c = Calibration(gyro_bias_dps=[0.1, 0.2, 0.3], mount="+X+Y+Z", trim_deg=[1.0, 2.0, 0.0])
    c.stamp("gyro", samples=5)
    c.save(tmp_path / "c.json")
    d = Calibration.load(tmp_path / "c.json")
    assert d.gyro_bias_dps == [0.1, 0.2, 0.3] and d.mount == "+X+Y+Z" and "gyro" in d.meta
    (tmp_path / "bad.json").write_text("{not json")
    assert Calibration.load(tmp_path / "bad.json") == Calibration()
    (tmp_path / "worse.json").write_text('{"mount": "nope"}')
    assert Calibration.load(tmp_path / "worse.json").mount == DEFAULT_MOUNT
    with pytest.raises(ValueError):
        Calibration(accel_scale=[9.0, 1, 1]).validate()


# ---- guided procedures ------------------------------------------------------------------------------------------
def _hold(session, acc, gyro=(0, 0, 0), mag=(0, 0, 0), seconds=None, hz=100, noise=0.02):
    rng = np.random.default_rng(3)
    dur = seconds or session.DURATIONS.get(session.task, 3.0)
    for i in range(int(dur * hz) + 5):
        session.feed(i / hz, np.array(acc) + rng.normal(0, noise, 3), np.array(gyro) + rng.normal(0, 0.05, 3),
                     np.array(mag, float))


def _session(tmp_path):
    box = {"c": Calibration()}

    def setter(x):
        x.save(tmp_path / "c.json")
        box["c"] = x
    return CalibSession(lambda: box["c"], setter), box


def test_session_gyro_accel_and_failures(tmp_path):
    s, box = _session(tmp_path)
    s.start("gyro")
    _hold(s, [0, 0, G], gyro=[0.4, -0.3, 0.2])
    assert s.error is None and box["c"].gyro_bias_dps == pytest.approx([0.4, -0.3, 0.2], abs=0.02) and "gyro" in box["c"].meta
    s.start("gyro")
    _hold(s, [0, 0, G], gyro=[40.0, 0, 0])                                            # rotating: rejected
    assert s.task is None and "rotation" in s.error
    s.start("accel", "+Z")
    _hold(s, [G, 0, 0.5])                                                             # +X is up, not +Z
    assert "expected +Z" in s.error
    off, sc = np.array([0.15, -0.10, 0.20]), np.array([1.005, 0.995, 1.003])
    for i, ax in enumerate("XYZ"):
        for sign in "+-":
            true = np.zeros(3)
            true[i] = G if sign == "+" else -G
            s.start("accel", sign + ax)
            _hold(s, true / sc + off)
            assert s.error is None, s.error
    c = box["c"]
    assert c.accel_offset == pytest.approx(off, abs=3e-3) and c.accel_scale == pytest.approx(sc, abs=3e-4)   # mean of noisy samples
    assert s.status()["accel_faces"] == sorted(cal.FACES)


def test_session_mount_level_and_mag(tmp_path):
    key, tilt = "+X-Y-Z", euler_to_matrix(math.radians(2.0), math.radians(-1.5), 0.0)
    s, box = _session(tmp_path)
    with pytest.raises(ValueError):
        s.start("mount_nose")                                                         # level first
    s.start("mount_level")
    _hold(s, _f_sensor(key, tilt.T @ [0, 0, -G]))
    assert s.status()["mount_level"]
    s.start("mount_nose")
    _hold(s, _f_sensor(key, euler_to_matrix(0, math.radians(50), 0).T @ [0, 0, -G]))
    c = box["c"]
    assert s.error is None and c.mount == key and c.trim_deg[:2] == pytest.approx([2.0, -1.5], abs=0.1)
    assert np.allclose(c.accel_vehicle(_f_sensor(key, tilt.T @ [0, 0, -G])), [0, 0, -G], atol=0.05)
    s.start("level")                                                                  # level trim alone, mount already known
    _hold(s, _f_sensor(key, tilt.T @ [0, 0, -G]))
    assert box["c"].trim_deg[:2] == pytest.approx([2.0, -1.5], abs=0.1)
    rng = np.random.default_rng(2)
    u = rng.normal(size=(3000, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    s.start("mag")
    for i, m in enumerate(u * 300 + [30, -20, 10]):
        s.feed(i / 100, None, None, m)
    s.finish_mag()
    assert s.error is None and box["c"].mag_offset == pytest.approx([30, -20, 10], abs=2) and box["c"].has_mag
    s.start("mag")
    for i in range(500):
        s.feed(i / 100, None, None, np.array([300.0, 0, 0]) + rng.normal(0, 1, 3))    # one orientation only
    s.finish_mag()
    assert s.error and box["c"].mag_offset == pytest.approx([30, -20, 10], abs=2)        # rejected, calibration untouched


# ---- EKF3 -------------------------------------------------------------------------------------------------------
def _run_static(e, seconds, roll=0.0, pitch=0.0, bias=(0, 0, 0), hz=100, range_m=None):
    f = euler_to_matrix(math.radians(roll), math.radians(pitch), 0).T @ [0, 0, -G]
    rng = np.random.default_rng(5)
    for i in range(int(seconds * hz)):
        e.predict(1 / hz, f + rng.normal(0, 0.02, 3), np.array(bias, float) + rng.normal(0, 0.001, 3))
        if range_m:
            e.update_range(i / hz, range_m / max(e.R[2, 2], 0.2), 3000, True)


def test_ekf_aligns_from_gravity_at_any_tilt():
    e = NavEkf()
    _run_static(e, 1.5, roll=20.0, pitch=-15.0)
    assert e.aligned
    assert e.rpy_deg[:2] == pytest.approx((20.0, -15.0), abs=0.2)


def test_ekf_static_hold_keeps_height_and_learns_gyro_bias():
    e = NavEkf()
    _run_static(e, 40.0, bias=(0.01, -0.006, 0.004), range_m=1.5)                     # 0.57 / -0.34 / 0.23 dps
    assert e.h == pytest.approx(1.5, abs=0.01) and abs(e.vd) < 0.02
    assert e.bg == pytest.approx([0.01, -0.006, 0.004], abs=0.002) and e.still and e.aid["stationary"] > 100
    assert e.range_rejects == 0


def test_ekf_height_uses_tilt_and_lever_arm():
    e = NavEkf()
    e.tf_offset = np.array([0.1, 0.0, 0.05])                                          # TFmini 10 cm ahead, 5 cm below the IMU
    _run_static(e, 1.5, pitch=30.0)                                                   # nose up 30 deg
    e.update_range(2.0, 2.0, 3000, True)
    R = e.R
    assert e.h == pytest.approx(2.0 * R[2, 2] + (R @ e.tf_offset)[2], abs=1e-6)
    assert e.h == pytest.approx(2.0 * math.cos(math.radians(30)) + (R @ e.tf_offset)[2], abs=1e-3)


def test_gravity_aiding_pulls_a_wrong_tilt_back_and_can_be_switched_off():
    on, off = NavEkf(), NavEkf()
    off.bench_aiding = False
    for e in (on, off):
        _run_static(e, 1.5)
        e.q = matrix_to_quat(euler_to_matrix(math.radians(4.0), 0.0, 0.0))             # inject a 4 deg roll error (4 sigma)
        _run_static(e, 12.0)
    assert abs(on.rpy_deg[0]) < 0.6 and abs(off.rpy_deg[0] - 4.0) < 0.6


def test_gravity_aiding_is_skipped_under_high_acceleration():
    e = NavEkf()
    _run_static(e, 1.5)
    n = e.aid["gravity"]
    e.predict(0.01, np.array([0.0, 0.0, -2.0 * G]), np.zeros(3))                      # 2 g: not gravity
    assert e.aid["gravity"] == n


def test_range_gate_reset_and_reasons():
    e = NavEkf()
    _run_static(e, 1.5, range_m=1.0)
    assert e.range_state == "ok" and e.h == pytest.approx(1.0, abs=0.01)
    e.update_range(2.0, 3.5, 3000, True)
    assert "gate" in e.range_state and e.h == pytest.approx(1.0, abs=0.02)
    for text, args in (("blind", (0.05, 3000, True)), ("beyond", (9.0, 3000, True)), ("no return", (0.0, 30, False))):
        e.update_range(3.0, *args)
        assert text in e.range_state
    t = 3.0
    for _ in range(80):
        t += 0.01
        e.predict(0.01, np.array([0, 0, -G]), np.zeros(3))
        e.update_range(t, 3.5, 3000, True)
    assert e.resets == 1 and e.h == pytest.approx(3.5, abs=0.05)


def test_mag_yaw_corrects_heading_drift_and_is_off_by_default():
    field = np.array([133.0, 0.0, 267.0])
    for enabled in (True, False):
        e = NavEkf()
        e.mag_yaw, e.bench_aiding = enabled, False            # no stationary aiding: it would learn the gyro bias itself
        rng = np.random.default_rng(7)
        for i in range(6000):                                  # static vehicle, gyro z bias 0.17 deg/s -> 10 deg drift in 60 s
            e.predict(0.01, np.array([0, 0, -G]) + rng.normal(0, 0.02, 3), np.array([0, 0, 0.003]) + rng.normal(0, 0.001, 3))
            e.update_mag(i / 100.0, field)
        err = abs(e.rpy_deg[2])
        assert (err < 2.0) if enabled else (err > 5.0)


def test_mag_heading_is_absolute_whatever_the_ekf_yaw_origin_and_tilt():
    field = np.array([133.0, 0.0, 267.0])                                             # NED: north + down
    for roll, pitch in ((0.0, 0.0), (20.0, -10.0), (-15.0, 25.0)):
        e = NavEkf()
        _run_static(e, 1.5, roll=roll, pitch=pitch)                                  # aligned from gravity, EKF yaw = 0
        for heading in (0.0, 30.0, 90.0, 200.0, 359.0):
            R_true = euler_to_matrix(math.radians(roll), math.radians(pitch), math.radians(heading))
            got = e.mag_heading_deg(R_true.T @ field)
            assert abs((got - heading + 180) % 360 - 180) < 1.0, (roll, pitch, heading, got)
    assert NavEkf().mag_heading_deg(field) is None                                   # not aligned yet


# ---- end to end: a bolted-on, biased simulated sensor -----------------------------------------------------------------
def _true_calibration(errs):
    return Calibration(gyro_bias_dps=list(errs.gyro_bias_dps), accel_offset=list(errs.accel_offset),
                       accel_scale=list(errs.accel_scale), mount=errs.mount)


def _run_attitude(calib, errs, seconds=25.0):
    """RMS of the larger of the roll / pitch errors after settling, and the height RMS, against the simulator's truth."""
    hub = _hub(calib=calib)
    tfp, wp, tf_s, imu_s = tfmini.TfParser(), witmotion.WitParser(), sim.SimTfmini(100), sim.SimWitmotion(100, errs)
    att, height = [], []
    for i in range(int(seconds * 100)):
        t = i / 100
        for fr in tfp.feed(tf_s.emit(t), 1000 + t):
            hub.on_tf(fr)
        for smp in wp.feed(imu_s.emit(t), 1000 + t):
            hub.on_imu(smp)
        if t > 8 and hub.ekf.aligned:
            d = np.array(hub.ekf.rpy_deg[:2]) - np.degrees(matrix_to_euler(sim.truth(t)["R"])[:2])
            att.append(np.abs(d).max())
            if hub.ekf.have_height:
                height.append(hub.ekf.h - sim.truth(t)["h"])
    # a wrong mount can leave the filter without any valid range (it believes the beam points sideways / up)
    return np.sqrt(np.mean(np.square(att))), (np.sqrt(np.mean(np.square(height))) if height else float("inf"))


@pytest.mark.parametrize("mount", [DEFAULT_MOUNT, "+X+Y+Z", "-Z+Y+X"])
def test_ekf_with_the_right_mount_tracks_attitude_for_any_mounting(mount):
    errs = sim.SimErrors(mount=mount)
    att, height = _run_attitude(_true_calibration(errs), errs)
    assert att < 0.8 and height < 0.03


def test_wrong_mount_is_visibly_worse_than_the_right_one():
    errs = sim.SimErrors(mount="+X+Y+Z")
    right, _ = _run_attitude(_true_calibration(errs), errs)
    wrong, _ = _run_attitude(_true_calibration(sim.SimErrors(mount=DEFAULT_MOUNT)), errs)
    assert right < 1.0 and wrong > 3 * right


def test_guided_gyro_mount_and_accel_procedures_in_the_simulator(tmp_path):
    """Drives the same procedures the page uses, freezing the simulated vehicle in the poses the wizard asks for."""
    errs = sim.SimErrors(mount="-Z+Y+X")
    hub = _hub(tmp_path)
    wp, imu_s = witmotion.WitParser(), sim.SimWitmotion(100, errs)
    clock = {"t": 0.0}

    def run(seconds):
        for _ in range(int(seconds * 100) + 10):
            for smp in wp.feed(imu_s.emit(clock["t"]), 1000 + clock["t"]):
                hub.on_imu(smp)
            clock["t"] += 0.01

    sim.CONTROL["pose"] = "level"
    hub.session.start("gyro")
    run(11)
    assert hub.session.error is None
    assert hub.calib.gyro_bias_dps == pytest.approx(list(errs.gyro_bias_dps), abs=0.06)
    hub.session.start("mount_level")
    run(4)
    sim.CONTROL["pose"] = "nose_up"
    hub.session.start("mount_nose")
    run(4)
    assert hub.session.error is None, hub.session.error
    assert hub.calib.mount == errs.mount and abs(hub.calib.trim_deg[0]) < 2.0 and abs(hub.calib.trim_deg[1]) < 2.0   # includes the not-yet-calibrated accel offsets (~1 deg)
    assert (tmp_path / "c.json").exists()
    for face in ("+X", "-X"):
        sim.CONTROL["pose"] = face
        hub.session.start("accel", face)
        run(4)
        assert hub.session.error is None, hub.session.error
    assert hub.calib.accel_offset[0] == pytest.approx(errs.accel_offset[0], abs=0.03)
    assert hub.calib.accel_scale[0] == pytest.approx(errs.accel_scale[0], abs=0.003)


# ---- setup API ---------------------------------------------------------------------------------------------------
def _api(tmp_path, simulate=False):
    import portal
    hub = _hub(tmp_path)

    class Reader:
        port, baud, hz, count, bytes_rx, age_s, live = "COMx", 115200, 100.0, 5, 50, 0.1, True
        parser = type("P", (), dict(crc_errors=0, junk_bytes=0, invalid=0, packets={0x51: 3}))()

        def __init__(self):
            self.sent = []

        def diagnosis(self):
            return ""

        def send(self, *packets):
            self.sent.extend(packets)
    imu = Reader()
    return hub, imu, portal.create_app(hub, Reader(), imu, simulate).test_client()


def test_api_config_roundtrip_validation_and_persistence(tmp_path):
    hub, _, c = _api(tmp_path)
    j = c.get("/api/config").get_json()
    assert len(j["mounts"]) == 24 and j["calibration"]["mount"] == DEFAULT_MOUNT and j["settings"]["kind"] == "ekf3"
    assert c.post("/api/config", json=dict(mount="+X+Y+Z", trim_deg=[1, 2, 3], tf_offset_m=[0.1, 0, 0.05])).get_json()["ok"]
    assert hub.calib.mount == "+X+Y+Z" and hub.calib.trim_deg == [1, 2, 3] and not hub.ekf.aligned      # filter restarted
    assert np.allclose(hub.ekf.tf_offset, [0.1, 0, 0.05])
    assert Calibration.load(tmp_path / "c.json").mount == "+X+Y+Z"                                     # persisted
    assert c.post("/api/config", json=dict(mount="bogus")).status_code == 400 and hub.calib.mount == "+X+Y+Z"
    assert c.post("/api/config", json=dict(trim_deg=[1, 2])).status_code == 400
    assert c.post("/api/config", json=dict(kind="simple", bench_aiding=False, mag_yaw=True)).get_json()["ok"]
    assert hub.settings() == dict(kind="simple", bench_aiding=False, mag_yaw=True)
    assert c.post("/api/config", json=dict(kind="nope")).status_code == 400
    assert c.post("/api/calib/reset").get_json()["ok"] and hub.calib == Calibration()


def test_api_calibration_and_sensor_commands(tmp_path):
    hub, imu, c = _api(tmp_path)
    assert c.post("/api/calib/start", json=dict(task="gyro")).get_json()["ok"]
    assert c.get("/status").get_json()["calib"]["task"] == "gyro"
    assert c.post("/api/calib/cancel").get_json()["ok"] and hub.session.task is None
    assert c.post("/api/calib/start", json=dict(task="bogus")).status_code == 400
    assert c.post("/api/calib/start", json=dict(task="accel", face="+Q")).status_code == 400
    assert c.post("/api/calib/start", json=dict(task="mount_nose")).status_code == 400
    assert c.post("/api/imu/gyro_autozero", json=dict(enabled=False)).get_json()["ok"]
    assert imu.sent == [witmotion.UNLOCK, bytes.fromhex("ffaa630100"), witmotion.SAVE]                # datasheet 5.2.7: 1 = removed
    assert c.post("/api/imu/gyro_autozero", json=dict(enabled="yes")).status_code == 400
    assert c.post("/api/sim/pose", json=dict(pose="level")).status_code == 404                        # hardware mode
    _, _, cs = _api(tmp_path, simulate=True)
    assert cs.post("/api/sim/pose", json=dict(pose="nose_up")).get_json()["ok"] and sim.CONTROL["pose"] == "nose_up"
    assert cs.post("/api/sim/pose", json=dict(pose="sideways")).status_code == 400
    assert cs.get("/status").get_json()["sim"] is True


def test_mag_heading_row_field_tracks_the_simulated_heading_only_when_calibrated(tmp_path):
    errs = sim.SimErrors()
    with_mag = _true_calibration(errs)
    with_mag.mag_offset = list(errs.mag_offset)
    with_mag.stamp("mag")
    out = {}
    for name, calib in (("with", with_mag), ("without", _true_calibration(errs))):
        hub = _hub(tmp_path, calib=calib)
        tfp, wp, tf_s, imu_s = tfmini.TfParser(), witmotion.WitParser(), sim.SimTfmini(100), sim.SimWitmotion(100, errs)
        errors, seen = [], 0
        for i in range(2500):
            t = i / 100
            for fr in tfp.feed(tf_s.emit(t), 1000 + t):
                hub.on_tf(fr)
            for smp in wp.feed(imu_s.emit(t), 1000 + t):
                hub.on_imu(smp)
            row = hub.rows[-1] if hub.rows else None
            mh = None if row is None else row[ROW.index("mag_hdg")]
            if mh is not None and t > 8:
                truth_heading = math.degrees(sim.truth(t)["rpy"][2]) % 360
                errors.append(abs((mh - truth_heading + 180) % 360 - 180))
                seen += 1
        out[name] = (seen, np.mean(errors) if errors else None)
    assert out["without"][0] == 0                                                     # no magnetometer calibration -> no heading
    assert out["with"][0] > 1000 and out["with"][1] < 4.0                             # tracks a turning, rolling vehicle
