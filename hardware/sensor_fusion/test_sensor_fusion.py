"""Tests for the TFmini / WTGAHRS1 codecs, the height filter and the portal API (no hardware needed).

    python -m pytest hardware/sensor_fusion -q
"""
import math
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sim
import tfmini
import witmotion
from calibration import Calibration
from fusion import FusionParams, HeightFusion, quat_from_rpy, tf_repeatability_m, up_component
from hub import ROW_FIELDS, FusionHub
from witmotion import G


# ---- TFmini ---------------------------------------------------------------------------------------
def test_tfmini_frame_roundtrip_and_units():
    f = tfmini.encode_frame(164, 3370, 31.0)
    (fr,) = tfmini.TfParser().feed(f, 1.0)
    assert fr.dist_m == pytest.approx(1.64) and fr.strength == 3370 and fr.temp_c == pytest.approx(31.0) and fr.valid


def test_tfmini_flags_unreliable_readings():
    p = tfmini.TfParser()
    frames = p.feed(tfmini.encode_frame(0, 30) + tfmini.encode_frame(500, 65535) + tfmini.encode_frame(500, 99), 0.0)
    assert [x.valid for x in frames] == [False, False, False] and p.invalid == 3


def test_tfmini_resyncs_after_garbage_split_chunks_and_bad_checksum():
    good = tfmini.encode_frame(120, 2000)
    bad = bytearray(good)
    bad[8] ^= 0xFF
    stream = b"\x01\x02\x58" + good[:4], good[4:] + bytes(bad) + good  # 0x58: not a header byte, unlike a stray 0x59
    p = tfmini.TfParser()
    out = []
    for chunk in stream:
        out += p.feed(chunk, 0.0)
    assert len(out) == 2 and p.crc_errors == 1 and all(o.dist_m == pytest.approx(1.2) for o in out)


def test_tfmini_command_bytes_match_manual():
    assert tfmini.CMD_VERSION.hex() == "5a04015f"
    assert tfmini.CMD_SAVE.hex() == "5a04116f"
    assert tfmini.cmd_frame_rate(100).hex() == "5a0603640000"[:10] + "c7"  # 5A 06 03 64 00 + sum
    assert tfmini.find_reply(b"\x59\x59" + bytes.fromhex("5a0701050102") + bytes([sum(bytes.fromhex("5a0701050102")) & 255]), 1)


def test_tfmini_repeatability_matches_manual_table():
    # table 2 (100 Hz): 0.9 cm at 120 and 500 a.u., 0.6 at 1000, 0.5 at 5000. The manual's formula is an average fit.
    assert tf_repeatability_m(500, 100) * 100 == pytest.approx(0.9, abs=0.2)
    assert tf_repeatability_m(120, 100) * 100 == pytest.approx(0.9, abs=0.5)
    assert tf_repeatability_m(3370, 100) * 100 < 0.6 and tf_repeatability_m(120, 100) > tf_repeatability_m(3370, 100)


# ---- WitMotion ------------------------------------------------------------------------------------
def _burst(**kw):
    a = dict(acc_g=(0.1, -0.2, 1.0), gyro_dps=(1.5, -2.5, 10.0), rpy_deg=(10.0, -20.0, 170.0), mag=(100, -200, 300),
             pressure_pa=97110, baro_cm=35688, quat=(0.9, 0.1, -0.2, 0.3))
    a.update(kw)
    return witmotion.encode_burst(**a)


def test_wit_burst_decodes_with_datasheet_scaling():
    p = witmotion.WitParser()
    out = p.feed(_burst() + _burst(), 5.0)          # the second burst flushes the first
    assert len(out) == 1
    s = out[0]
    assert s.acc == pytest.approx((0.1 * G, -0.2 * G, G), abs=0.01)
    assert s.gyro == pytest.approx((1.5, -2.5, 10.0), abs=0.07)
    assert s.rpy == pytest.approx((10.0, -20.0, 170.0), abs=0.01)
    assert s.quat == pytest.approx((0.9, 0.1, -0.2, 0.3), abs=1e-4)
    assert s.pressure_pa == 97110 and s.baro_h_m == pytest.approx(356.88) and s.mag == (100, -200, 300)
    assert sorted(p.packets) == list(witmotion.DATA_TYPES)


def test_wit_parser_resyncs_and_counts_bad_checksums():
    b = bytearray(_burst())
    b[10] ^= 0xFF                                    # corrupt the first packet's checksum
    p = witmotion.WitParser()
    out = p.feed(b"\x00\x11\x66" + bytes(b) + _burst() + _burst(), 0.0)
    assert p.crc_errors == 1 and len(out) >= 1 and out[-1].quat is not None


def test_wit_config_commands_match_datasheet():
    assert witmotion.UNLOCK.hex() == "ffaa6988b5"
    assert witmotion.SAVE.hex() == "ffaa000000"
    assert witmotion.cmd_rate(100).hex() == "ffaa030900"          # 5.2.9: 0x09 = 100 Hz
    assert witmotion.cmd_rate(20).hex() == "ffaa030700"           # the datasheet's own 20 Hz example
    assert witmotion.cmd_baud(115200).hex() == "ffaa040600"       # 5.2.10: 0x06 = 115200
    # 0x51 0x52 0x53 0x54 0x56 -> RSWL bits 1,2,3,4,6 = 0x5E; 0x59 -> RSWH bit 1
    assert witmotion.cmd_content(witmotion.DATA_TYPES).hex() == "ffaa02" + "5e" + "02"


# ---- attitude helpers -----------------------------------------------------------------------------
def test_up_component_matches_rotation_matrix():
    rng = random.Random(3)
    for _ in range(20):
        rpy = [rng.uniform(-60, 60), rng.uniform(-60, 60), rng.uniform(-180, 180)]
        q = quat_from_rpy(*rpy)
        r, p, y = (math.radians(a) for a in rpy)
        Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
        Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
        v = np.array([rng.uniform(-3, 3) for _ in range(3)])
        assert up_component(q, v) == pytest.approx((Rz @ Ry @ Rx @ v)[2], abs=1e-9)


def test_sim_truth_is_self_consistent():
    t = 3.7
    tr = sim.truth(t)
    assert tr["a_up"] == pytest.approx((sim.truth(t + 1e-3)["h"] - 2 * tr["h"] + sim.truth(t - 1e-3)["h"]) / 1e-6, abs=1e-4)


# ---- filter ---------------------------------------------------------------------------------------
def _params(**kw):
    return FusionParams(**kw)


def test_static_height_converges_and_ignores_gravity_bias():
    f = HeightFusion(_params())
    for i in range(400):
        t = i * 0.01
        f.predict(t, 0.01, 0.15)                     # a 0.15 m/s^2 accelerometer bias
        f.update_range(t, 1.50, 3000, 1.0, True)
    assert f.h == pytest.approx(1.50, abs=0.01) and abs(f.v) < 0.05
    assert f.x[2] == pytest.approx(0.15, abs=0.08)   # bias learned (observable through the range)


def test_range_gates_and_reasons():
    f = HeightFusion(_params())
    f.update_range(0, 1.0, 3000, 1.0, True)
    assert f.range_state == "ok"
    for args, text in (((0.05, 3000, 1.0, True), "blind"), ((9.0, 3000, 1.0, True), "beyond"),
                       ((1.0, 3000, 0.3, True), "tilted"), ((0.0, 30, 1.0, False), "no return")):
        f.update_range(1, *args)
        assert text in f.range_state
    assert f.range_updates == 1


def test_innovation_gate_rejects_spike_then_resets_when_persistent():
    f = HeightFusion(_params())
    for i in range(200):
        f.predict(i * 0.01, 0.01, 0.0)
        f.update_range(i * 0.01, 1.0, 3000, 1.0, True)
    f.update_range(2.0, 3.0, 3000, 1.0, True)        # single spike
    assert "gate" in f.range_state and f.h == pytest.approx(1.0, abs=0.02) and f.range_rejects == 1
    t = 2.0
    for _ in range(80):                              # the floor really moved: 3 m for > 0.5 s
        t += 0.01
        f.predict(t, 0.01, 0.0)
        f.update_range(t, 3.0, 3000, 1.0, True)
    assert f.resets == 1 and f.h == pytest.approx(3.0, abs=0.05)


def test_tilt_compensation_uses_vertical_component():
    f = HeightFusion(_params())
    ct = math.cos(math.radians(30))
    f.update_range(0, 2.0, 3000, ct, True)
    assert f.h == pytest.approx(2.0 * ct)


def test_baro_carries_height_when_range_is_lost():
    f = HeightFusion(_params())

    def h(t):                                        # hover at 1 m, then a smooth 0.5 m climb and return over 3-7 s
        return 1.0 + (0.25 * (1 - math.cos(2 * math.pi * (t - 3) / 4)) if 3 <= t <= 7 else 0.0)

    def a(t):
        return 0.25 * (2 * math.pi / 4) ** 2 * math.cos(2 * math.pi * (t - 3) / 4) if 3 <= t <= 7 else 0.0

    err, mid = [], None
    for i in range(1000):
        t = i * 0.01
        f.predict(t, 0.01, a(t))
        f.update_baro(t, 356.88 + h(t))
        if t < 3 or t > 7:
            f.update_range(t, h(t), 3000, 1.0, True)
        elif t > 3.5:
            err.append(abs(f.h - h(t)))              # range lost: IMU + baro only
        if abs(t - 5.0) < 1e-6:
            mid = f.h
            assert f.mode(t) == 2
    assert f.x[3] == pytest.approx(356.88, abs=0.15)
    assert mid == pytest.approx(1.5, abs=0.1) and max(err) < 0.12


def test_baro_before_first_range_does_not_pull_height():
    f = HeightFusion(_params())
    for i in range(50):
        f.update_baro(i * 0.05, 300.0 + 0.4)
    f.update_range(2.6, 1.2, 3000, 1.0, True)
    assert f.x[3] == pytest.approx(300.4 - 1.2)
    for i in range(60, 120):
        f.update_baro(i * 0.05, 300.4)
    assert f.h == pytest.approx(1.2, abs=0.02)


# ---- end to end: simulator bytes -> parsers -> hub ----------------------------------------------------
def _hub(tmp=None, **kw):
    """A hub with default calibration stored in a temp dir (never the user's real calibration.json)."""
    path = (tmp or Path(tempfile.mkdtemp())) / "c.json"
    return FusionHub(FusionParams(), history=10 ** 6, calib=kw.pop("calib", None) or Calibration(), calib_path=path, **kw)


def _run_sim(seconds=30.0, hz=100.0, tmp=None, **hub_kw):
    hub, tfp, wp = _hub(tmp, **hub_kw), tfmini.TfParser(), witmotion.WitParser()
    tf_s, imu_s = sim.SimTfmini(hz), sim.SimWitmotion(hz)
    t0, err, cover_err = 1000.0, [], []
    for i in range(int(seconds * hz)):
        t = i / hz
        for fr in tfp.feed(tf_s.emit(t), t0 + t + 0.002):
            hub.on_tf(fr)
        for s in wp.feed(imu_s.emit(t), t0 + t + 0.004):
            hub.on_imu(s)
        f = hub.filter
        if t > 3.0 and f.have_height:
            e = f.h - sim.truth(t)["h"]
            (cover_err if 12.5 < t % 20 < 15.0 else err).append(e)
    return hub, np.array(err), np.array(cover_err)


def test_end_to_end_sim_tracks_truth_including_range_dropout():
    hub, err, cover_err = _run_sim(45.0)  # EKF3 path on raw IMU with gyro bias / accel offset / scale errors, no calibration
    assert np.sqrt(np.mean(err ** 2)) < 0.06
    assert np.max(np.abs(cover_err)) < 0.35          # coasting on IMU + baro while the TFmini is covered
    assert hub.filter.range_updates > 3000 and hub.filter.resets == 0


def test_hub_rows_match_field_list_and_snapshot():
    hub, *_ = _run_sim(3.0)
    row = hub.rows[-1]
    assert len(row) == len(ROW_FIELDS)
    snap = hub.snapshot()
    assert snap["fusion"]["mode_name"] and snap["tf"]["valid"] is True


def test_baro_fields_in_rows_and_snapshot():
    hub, *_ = _run_sim(12.0)
    i = ROW_FIELDS.index
    rows = [r for r in hub.rows if r[i("baro_res")] is not None]
    assert rows, "no rows with a barometer residual"
    r = rows[-1]
    assert 900.0 < r[i("pressure")] < 1100.0                                   # hPa, sea-level-ish
    # simulated altitude = 356.88 + height (0.7-2.1 m) + up to 0.3 m of weather drift + noise
    assert sim.BASE_ALT_M + 0.7 - 0.5 < r[i("baro_alt")] < sim.BASE_ALT_M + 2.1 + 0.5
    assert r[i("baro_res")] == pytest.approx(r[i("baro_h")] - r[i("h_fused")], abs=2e-4)
    res = np.array([x[i("baro_res")] for x in rows])
    assert abs(res.mean()) < 0.5 and res.std() < 0.5                           # the baro tracks the fused height loosely
    b = hub.snapshot()["baro"]
    assert 900.0 < b["pressure_hpa"] < 1100.0 and b["height_m"] is not None and b["offset_m"] is not None
    assert b["res_std_m"] is not None and b["pressure_std_pa"] is not None and b["noise_model_m"] > 0


# ---- portal API ------------------------------------------------------------------------------------
def test_portal_endpoints(monkeypatch):
    import portal
    hub, *_ = _run_sim(2.0)

    class R:                                          # stand-in readers with the attributes the API reads
        port, baud, hz, count, bytes_rx, age_s, live = "COMx", 115200, 100.0, 5, 50, 0.1, True
        parser = type("P", (), dict(crc_errors=0, junk_bytes=0, invalid=0, packets={0x51: 3}))()
        def diagnosis(self): return ""
    c = portal.create_app(hub, R(), R()).test_client()
    assert b"Sensor fusion" in c.get("/").data
    s = c.get("/status").get_json()
    assert s["tf_link"]["name"] == "TFmini" and s["imu_link"]["packets"] == {"0x51": 3} and "fusion" in s
    j = c.get("/samples?after=0").get_json()
    assert j["fields"] == list(ROW_FIELDS) and j["samples"] and j["latest"] == hub.count
    assert c.get("/samples.csv").data.decode().splitlines()[0].startswith("n,t,range_m")
    assert c.post("/api/reset").get_json()["ok"] and not hub.filter.have_height
