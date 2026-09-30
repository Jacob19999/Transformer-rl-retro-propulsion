"""Response of the fused height to rapid and slow vertical motion (regression tests for the stationary-aid freeze).

    python -m pytest hardware/sensor_fusion -q

Background: with bench aiding on, "no rotation and no acceleration" was read as "stationary" and a zero-velocity update held
the vertical speed at 0 during a steady 0.3-0.5 m/s descent. The prediction stalled, the range was rejected by the
innovation gate for 0.5 s (25-36 cm of error), then the filter reset.
"""
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sim
import tfmini
import witmotion
from calibration import Calibration
from ekf3 import NavEkf
from fusion import FusionParams
from hub import FusionHub

G = witmotion.G


def _descent(vmax, hold=3.0, h0=2.0, h1=0.4, ramp=0.4):
    """Level vehicle: hover, raised-cosine speed up to ``vmax`` downward, cruise, brake to rest at ``h1``."""
    dt = 1e-3
    ts = np.arange(0, 14, dt)
    total = max((h0 - h1 - vmax * ramp) / vmax, 0.0) + 2 * ramp
    v = np.zeros_like(ts)
    for i, t in enumerate(ts):
        s = t - hold
        if 0 < s < total:
            k = min(s, total - s, ramp)
            v[i] = -vmax * 0.5 * (1 - math.cos(math.pi * k / ramp)) if k < ramp else -vmax
    h = h0 + np.cumsum(v) * dt
    a = np.gradient(v, dt)

    def truth(t, mount=None):
        i = min(max(int(t / dt), 0), len(ts) - 1)
        return dict(h=float(h[i]), a_up=float(a[i]), R=np.eye(3), rpy=(0.0, 0.0, 0.0), static=False)
    return truth


def _run(monkeypatch, truth, seconds=9.0, bench=True, bandwidth=None, tf_latency=0.0):
    monkeypatch.setattr(sim, "truth", truth)
    errs = sim.SimErrors(bandwidth_hz=bandwidth)
    calib = Calibration(gyro_bias_dps=list(errs.gyro_bias_dps), accel_offset=list(errs.accel_offset),
                        accel_scale=list(errs.accel_scale), mount=errs.mount)
    hub = FusionHub(FusionParams(), history=10 ** 6, calib=calib, calib_path=Path(tempfile.mkdtemp()) / "c.json", bench_aiding=bench)
    tfp, wp, tf_s, imu_s = tfmini.TfParser(), witmotion.WitParser(), sim.SimTfmini(100, latency_s=tf_latency), sim.SimWitmotion(100, errs)
    t_log, err = [], []
    for i in range(int(seconds * 100)):
        t = i / 100
        for fr in tfp.feed(tf_s.emit(t), 1000 + t):
            hub.on_tf(fr)
        for s in wp.feed(imu_s.emit(t), 1000 + t):
            hub.on_imu(s)
        if t > 2.5 and hub.ekf.have_height:
            t_log.append(t)
            err.append(hub.ekf.h - truth(t)["h"])
    return hub, np.array(t_log), np.array(err)


@pytest.fixture(autouse=True)
def _restore_pose():
    sim.CONTROL["pose"] = "moving"
    yield


@pytest.mark.parametrize("vmax", [0.2, 0.3, 0.5, 1.0, 3.0])
def test_descents_are_tracked_with_bench_aiding_on(monkeypatch, vmax):
    """Slow steady descents used to freeze the height for ~0.3 s; every speed must now track within a few cm."""
    hub, t, err = _run(monkeypatch, _descent(vmax))
    assert np.abs(err).max() < 0.06
    assert hub.ekf.range_rejects == 0 and hub.ekf.resets == 0


def test_descent_lag_is_small_with_sensor_filtering_and_range_latency(monkeypatch):
    """WTGAHRS1 output low-pass (~20 Hz) plus 20 ms of TFmini latency: the estimate is no worse than the raw range."""
    truth = _descent(3.0, h0=6.0, h1=0.5)
    hub, t, err = _run(monkeypatch, truth, bandwidth=20.0, tf_latency=0.02)
    lag_s = np.max(np.abs(err)) / 3.0
    assert np.abs(err).max() < 0.08 and lag_s < 0.03
    assert hub.ekf.range_rejects == 0


def test_bench_aiding_makes_no_difference_to_the_descent(monkeypatch):
    _, _, on = _run(monkeypatch, _descent(0.4), bench=True)
    _, _, off = _run(monkeypatch, _descent(0.4), bench=False)
    assert abs(np.abs(on).max() - np.abs(off).max()) < 0.01


# ---- the zero-velocity guard and the fast recovery, on the filter alone ---------------------------------------------
def _still(e, seconds, range_of_t=None, t0=0.0, hz=100):
    rng = np.random.default_rng(3)
    for i in range(int(seconds * hz)):
        t = t0 + i / hz
        e.predict(1 / hz, np.array([0, 0, -G]) + rng.normal(0, 0.01, 3), rng.normal(0, 0.0005, 3))
        if range_of_t is not None:
            e.update_range(t, range_of_t(t) + rng.normal(0, 0.004), 3000, True)
    return t0 + seconds


def test_zero_velocity_update_is_withheld_while_the_range_is_changing():
    e = NavEkf()
    t = _still(e, 2.0, lambda t: 1.5)
    assert e.aid["zupt"] > 0 and e.zupt_blocked == ""                       # genuinely still: applied
    n, start = e.aid["zupt"], t
    _still(e, 2.0, lambda t: 1.5 - 0.3 * (t - start), t0=start)              # steady 0.3 m/s descent, the IMU sees nothing
    # a handful slip through in the first ~70 ms, before there is any evidence of motion; then it is withheld, with a reason
    assert e.aid["zupt"] - n <= 12 and e.zupt_blocked and e.range_rejects == 0
    assert e.still                                                           # the gyro-bias aid still applies
    assert e.h == pytest.approx(1.5 - 0.3 * 2.0, abs=0.04)


def test_zero_velocity_update_is_withheld_when_the_filter_already_sees_vertical_speed():
    e = NavEkf()
    _still(e, 1.5, lambda t: 1.5)
    e.v[2] = 0.5                                                             # 0.5 m/s downward
    n = e.aid["zupt"]
    e.predict(0.01, np.array([0, 0, -G]), np.zeros(3))
    assert e.aid["zupt"] == n and e.zupt_blocked == "vertical speed"


def test_consistent_rejections_reset_within_a_tenth_of_a_second_keeping_the_velocity():
    e = NavEkf()
    e.bench_aiding = False                                                   # isolate the gate + reset logic
    _still(e, 2.0, lambda t: 1.5)
    t0 = 2.0
    resets = e.resets
    for i in range(30):                                                      # the range now falls at 0.6 m/s from 0.9 m away
        t = t0 + i / 100
        e.predict(0.01, np.array([0, 0, -G]), np.zeros(3))
        e.update_range(t, 0.9 - 0.6 * (i / 100), 3000, True)
        if e.resets > resets:
            break
    assert e.resets == resets + 1 and i <= 12                                # a dozen samples, not the 50 of a 0.5 s timeout
    assert e.vd == pytest.approx(-0.6, abs=0.1)                              # restarted with the measured descent rate


def test_isolated_outliers_are_still_rejected_without_a_reset():
    e = NavEkf()
    _still(e, 2.0, lambda t: 1.5)
    for i in range(60):
        t = 2.0 + i / 100
        e.predict(0.01, np.array([0, 0, -G]), np.zeros(3))
        e.update_range(t, 3.5 if i % 6 == 3 else 1.5, 3000, True)           # a bad reading every sixth sample
    assert e.resets == 0 and e.range_rejects >= 8 and e.h == pytest.approx(1.5, abs=0.02)


# ---- simulator options -------------------------------------------------------------------------------------------------
def test_tfmini_latency_and_imu_bandwidth_options(monkeypatch):
    monkeypatch.setattr(sim, "truth", lambda t, mount=None: dict(h=1.0 + t, a_up=0.0, R=np.eye(3), rpy=(0, 0, 0), static=False))
    p = tfmini.TfParser()
    (fr,) = p.feed(sim.SimTfmini(100, latency_s=0.5).emit(2.0), 0.0)
    assert fr.dist_m == pytest.approx(2.5, abs=0.03)                         # the height it had 0.5 s earlier

    def truth(t, mount=None):
        return dict(h=1.0, a_up=(9.0 if t > 1.0 else 0.0), R=np.eye(3), rpy=(0, 0, 0), static=False)
    monkeypatch.setattr(sim, "truth", truth)
    out = {}
    for bw in (None, 5.0):
        w, imu = witmotion.WitParser(), sim.SimWitmotion(100, sim.SimErrors(bandwidth_hz=bw, accel_offset=(0, 0, 0), accel_scale=(1, 1, 1)))
        z = []
        for i in range(200):
            for s in w.feed(imu.emit(i / 100), 0.0):
                z.append(s.acc[2])
        out[bw] = z
    i = 102                                                                  # just after the 9 m/s^2 step at t = 1 s
    assert out[None][i] - out[None][95] > 8.0                                # ideal sensor: instant
    assert out[5.0][i] - out[5.0][95] < 6.0                                  # 5 Hz low-pass: only about half way there
