"""Joins the two sensor streams to a filter and keeps a rolling record for the web layer.

Two filters can run behind the same inputs: ``ekf3`` (default; raw calibrated IMU in the vehicle frame, ekf3.py) and
``simple`` (the 4-state vertical Kalman filter of fusion.py that trusts the sensor's onboard attitude).
"""
from __future__ import annotations

import collections
import copy
import csv
import threading
import time

import numpy as np

from calibration import PATH as CALIB_PATH, CalibSession, Calibration
from ekf3 import EkfParams, NavEkf
from frames import T_EN, matrix_to_euler, quat_to_matrix
from fusion import MODE_NAMES, FusionParams, HeightFusion, quat_from_rpy, up_component
from tfmini import Frame
from witmotion import G, ImuSample

ROW_FIELDS = ("n", "t", "range_m", "strength", "range_ok", "h_meas", "h_fused", "v_fused", "h_sigma", "a_bias", "baro_h",
              "pressure", "baro_alt", "baro_res",
              "roll", "pitch", "yaw", "s_roll", "s_pitch", "s_yaw", "mag_hdg", "gx", "gy", "gz", "bgx", "bgy", "bgz",
              "ax", "ay", "az", "a_up", "pn", "pe", "vn", "ve", "nis", "mode")
IMU_STALE_S = 0.3
TF_STALE_S = 0.3
KINDS = ("ekf3", "simple")


def _r(v, n=4):
    return None if v is None else round(float(v), n)


def _vec(v, n=3):
    return (None,) * 3 if v is None else tuple(_r(x, n) for x in v)


class FusionHub:
    def __init__(self, params: FusionParams | None = None, history: int = 6000, log_path: str | None = None,
                 calib: Calibration | None = None, calib_path=CALIB_PATH, kind: str = "ekf3",
                 bench_aiding: bool = True, mag_yaw: bool = False, ekf_overrides: dict | None = None):
        self.params = params or FusionParams.from_profile()
        self.simple = HeightFusion(self.params)
        self.ekf = NavEkf(EkfParams.from_fusion(self.params, **(ekf_overrides or {})))
        self.lock = threading.RLock()
        self.calib_path = calib_path
        self.calib = calib if calib is not None else Calibration.load(calib_path)
        self.session = CalibSession(lambda: self.calib, self._set_calib)
        self.kind = kind if kind in KINDS else "ekf3"
        self.ekf.bench_aiding, self.ekf.mag_yaw = bench_aiding, mag_yaw
        self.ekf.tf_offset = np.array(self.calib.tf_offset_m, float)
        self.rows: collections.deque = collections.deque(maxlen=history)
        self.count = 0
        self.t0 = time.time()
        self.imu: ImuSample | None = None
        self.tf: Frame | None = None
        self.quat = (1.0, 0.0, 0.0, 0.0)          # the sensor's onboard attitude (simple filter, comparison)
        self.cos_tilt = 1.0
        self.a_up = 0.0
        self.h_meas: float | None = None
        self.mag_hdg: float | None = None         # magnetic heading (needs the magnetometer calibration)
        self.f_v: np.ndarray | None = None        # calibrated vehicle-frame specific force
        self.w_v: np.ndarray | None = None        # calibrated vehicle-frame rate, rad/s
        self._imu_prev_t: float | None = None
        self._imu_dt = 0.01
        self._csv_file = open(log_path, "w", newline="") if log_path else None
        self._csv = csv.writer(self._csv_file) if self._csv_file else None
        if self._csv:
            self._csv.writerow(("pc_time_s",) + ROW_FIELDS)

    # ---- configuration ----------------------------------------------------------------------------------------
    @property
    def filter(self):
        return self.ekf if self.kind == "ekf3" else self.simple

    def _set_calib(self, c: Calibration) -> None:
        """Validate, persist and apply a calibration; the EKF restarts because its inputs just changed meaning."""
        c.validate()
        c.save(self.calib_path)
        with self.lock:
            self.calib = c
            self.ekf.tf_offset = np.array(c.tf_offset_m, float)
            self.ekf.reset()

    def set_calibration_fields(self, mount=None, trim_deg=None, tf_offset_m=None) -> None:
        with self.lock:
            c = copy.deepcopy(self.calib)
            if mount is not None:
                c.mount = mount
            if trim_deg is not None:
                c.trim_deg = [float(x) for x in trim_deg]
            if tf_offset_m is not None:
                c.tf_offset_m = [float(x) for x in tf_offset_m]
            self._set_calib(c)

    def reset_calibration(self) -> None:
        with self.lock:
            self.session.reset_captures()
            self._set_calib(Calibration())

    def set_settings(self, kind=None, bench_aiding=None, mag_yaw=None) -> None:
        with self.lock:
            if kind is not None:
                if kind not in KINDS:
                    raise ValueError(f"filter must be one of {KINDS}")
                self.kind = kind
            if bench_aiding is not None:
                self.ekf.bench_aiding = bool(bench_aiding)
            if mag_yaw is not None:
                self.ekf.mag_yaw = bool(mag_yaw)

    def settings(self) -> dict:
        return dict(kind=self.kind, bench_aiding=self.ekf.bench_aiding, mag_yaw=self.ekf.mag_yaw)

    def reset(self) -> None:
        with self.lock:
            self.simple.reset()
            self.ekf.reset()
            self.h_meas = None

    # ---- sensor callbacks (reader threads) -------------------------------------------------------------------
    def on_imu(self, s: ImuSample) -> None:
        with self.lock:
            self.imu = s
            self.session.feed(s.t, s.acc, s.gyro, s.mag)
            if s.quat is not None:
                self.quat = s.quat
            elif s.rpy is not None:
                self.quat = quat_from_rpy(*s.rpy)
            gap = None if self._imu_prev_t is None else s.t - self._imu_prev_t
            have_motion = s.acc is not None and s.gyro is not None
            if have_motion:
                self.f_v = self.calib.accel_vehicle(s.acc)
                self.w_v = self.calib.gyro_vehicle_rad(s.gyro)
                if gap is not None and gap < 0.2:
                    self._imu_dt += 0.05 * (gap - self._imu_dt)   # arrival times are bursty: smooth to the true period
                if self.kind == "ekf3":
                    self.ekf.predict(self._imu_dt if gap is not None and gap < 0.2 else 0.0, self.f_v, self.w_v)
                    self.mag_hdg = None
                    if s.mag is not None and self.calib.has_mag:
                        m_v = self.calib.mag_vehicle(s.mag)
                        self.ekf.update_mag(s.t, m_v)
                        self.mag_hdg = self.ekf.mag_heading_deg(m_v)
                    self.a_up = -float((self.ekf.R @ (self.f_v - self.ekf.ba))[2] + G) if self.ekf.aligned else 0.0
                    self.cos_tilt = max(self.ekf.cos_tilt, 0.0)
                elif s.quat is not None or s.rpy is not None:
                    self.a_up = up_component(self.quat, s.acc) - G
                    if gap is not None and gap < 0.2:
                        self.simple.predict(s.t, self._imu_dt, self.a_up)
                    self.cos_tilt = max(up_component(self.quat, (0.0, 0.0, 1.0)), 0.0)
                self._imu_prev_t = s.t
            if s.baro_h_m is not None:
                (self.ekf if self.kind == "ekf3" else self.simple).update_baro(s.t, s.baro_h_m)
            self._record(s.t)

    def on_tf(self, fr: Frame) -> None:
        with self.lock:
            self.tf = fr
            imu_live = self._imu_live(fr.t)
            if self.kind == "ekf3":
                self.ekf.update_range(fr.t, fr.dist_m, fr.strength, fr.valid)
                cos_t = max(self.ekf.cos_tilt, 0.0) if self.ekf.aligned else 1.0
            else:
                cos_t = self.cos_tilt if imu_live else 1.0
                self.simple.update_range(fr.t, fr.dist_m, fr.strength, cos_t, fr.valid)
            f = self.filter
            ok = f.range_state == "ok"
            self.h_meas = (fr.dist_m + float(self.calib.tf_offset_m[2])) * cos_t if ok else None
            if not imu_live:
                self._record(fr.t)

    def _imu_live(self, t: float) -> bool:
        return self.imu is not None and t - self._imu_prev_t < IMU_STALE_S if self._imu_prev_t else False

    # ---- recording ---------------------------------------------------------------------------------------------
    def _record(self, t: float) -> None:
        f, s, tf = self.filter, self.imu, self.tf
        ekf = self.kind == "ekf3"
        imu_live = self._imu_live(t)
        tf_live = tf is not None and t - tf.t < TF_STALE_S
        mode = f.mode(t, imu_live)
        has = f.have_height
        onboard = s.rpy if s and s.rpy else (None,) * 3
        if ekf and self.ekf.aligned:
            att = tuple(_r(a, 2) for a in self.ekf.rpy_deg)
            bg = _vec(np.degrees(self.ekf.bg), 3)
            ne = (_r(self.ekf.pos[0]), _r(self.ekf.pos[1]), _r(self.ekf.v[0]), _r(self.ekf.v[1]))
            a_bias = _r(-(self.ekf.R @ self.ekf.ba)[2]) if has else None
            v_up, h_val = (_r(self.ekf.vd), _r(self.ekf.h)) if has else (None, None)
        elif ekf:
            att, bg, ne, a_bias, v_up, h_val = (None,) * 3, (None,) * 3, (None,) * 4, None, None, None
        else:
            att = (None,) * 3
            if s and s.quat is not None:                      # onboard sensor attitude, rotated to the vehicle frame
                R_nb = T_EN @ quat_to_matrix(self.quat) @ self.calib.R.T
                att = tuple(_r(np.degrees(a), 2) for a in matrix_to_euler(R_nb))
            bg, ne = (None,) * 3, (None,) * 4
            a_bias = _r(self.simple.x[2]) if has else None
            v_up, h_val = (_r(self.simple.v), _r(self.simple.h)) if has else (None, None)
        gyro_dps = None if self.w_v is None or not imu_live else np.degrees(self.w_v)
        acc_v = None if self.f_v is None or not imu_live else self.f_v
        baro_h = f.baro_height(s.baro_h_m) if s and imu_live else None
        baro_res = None if baro_h is None or h_val is None else baro_h - h_val     # barometer minus the fused height
        self.count += 1
        row = (self.count, round(t - self.t0, 3),
               _r(tf.dist_m if tf_live and tf.valid else None), tf.strength if tf_live else None,
               int(tf_live and f.range_state == "ok"), _r(self.h_meas if tf_live else None),
               h_val, v_up, _r(f.h_sigma if has else None), a_bias,
               _r(baro_h), _r(s.pressure_pa / 100.0, 2) if s and imu_live and s.pressure_pa is not None else None,
               _r(s.baro_h_m, 2) if s and imu_live else None, _r(baro_res),
               *att, *(_r(a, 2) for a in onboard), _r(self.mag_hdg if imu_live else None, 1), *_vec(gyro_dps, 2), *bg, *_vec(acc_v, 3),
               _r(self.a_up if imu_live else None, 3), *ne, _r(f.nis, 2), mode)
        self.rows.append(row)
        if self._csv:
            self._csv.writerow((f"{time.time():.4f}",) + row)
            if self.count % 100 == 0:
                self._csv_file.flush()

    def since(self, after: int, limit: int = 600) -> list:
        with self.lock:
            return [r for r in self.rows if r[0] > after][-limit:]

    def snapshot(self) -> dict:
        with self.lock:
            f, s, tf, now = self.filter, self.imu, self.tf, time.time()
            has = f.have_height
            mode = f.mode(now, self._imu_live(now))
            ekf = self.ekf
            out = dict(
                fusion=dict(mode=mode, mode_name=MODE_NAMES[mode], kind=self.kind,
                            height=_r(ekf.h if self.kind == "ekf3" else f.h) if has else None,
                            vspeed=_r(ekf.vd if self.kind == "ekf3" else f.v) if has else None,
                            sigma=_r(f.h_sigma) if has else None, range_state=f.range_state, nis=_r(f.nis, 2),
                            range_updates=f.range_updates, range_rejects=f.range_rejects, resets=f.resets,
                            a_up=_r(self.a_up, 3), cos_tilt=_r(self.cos_tilt, 4),
                            baro_offset=_r(f.bb if self.kind == "ekf3" else f.x[3]) if f.have_baro else None,
                            accel_bias=(_r(np.linalg.norm(ekf.ba), 3) if self.kind == "ekf3" else _r(f.x[2], 3)) if has else None),
                ekf=dict(aligned=ekf.aligned, still=ekf.still, zupt_blocked=ekf.zupt_blocked, aid=dict(ekf.aid), bg_dps=_vec(np.degrees(ekf.bg), 3),
                         ba=_vec(ekf.ba, 3), rpy=_vec(ekf.rpy_deg, 1) if ekf.aligned else None,
                         pos_ne=_vec(ekf.pos[:2], 2), vel_ne=_vec(ekf.v[:2], 2), mag_hdg=_r(self.mag_hdg, 1),
                         has_mag_cal=self.calib.has_mag),
                tf=dict(dist=_r(tf.dist_m) if tf else None, strength=tf.strength if tf else None,
                        temp=_r(tf.temp_c, 1) if tf else None, valid=bool(tf and tf.valid)),
                imu=dict(temp=_r(s.temp_c, 1) if s else None, pressure=_r(s.pressure_pa, 0) if s else None,
                         baro_alt=_r(s.baro_h_m, 2) if s else None, mag=list(s.mag) if s and s.mag else None,
                         acc_raw=_vec(s.acc, 2) if s and s.acc else None),
                baro=self._baro_stats(),
                calib=self.session.status())
            return out

    def _baro_stats(self) -> dict:
        """Barometer summary for the page: latest readings, the learned offset, and how it compares with the fused height."""
        f, s = self.filter, self.imu
        i = ROW_FIELDS.index
        recent = list(self.rows)[-1000:]                     # ~10 s
        res = np.array([r[i("baro_res")] for r in recent if r[i("baro_res")] is not None], float)
        pr = np.array([r[i("pressure")] for r in recent[-300:] if r[i("pressure")] is not None], float)
        return dict(
            pressure_hpa=_r(s.pressure_pa / 100.0, 2) if s and s.pressure_pa is not None else None,
            altitude_m=_r(s.baro_h_m, 2) if s and s.baro_h_m is not None else None,
            height_m=_r(f.baro_height(s.baro_h_m)) if s and s.baro_h_m is not None else None,
            offset_m=_r(f.bb if self.kind == "ekf3" else f.x[3]) if f.have_baro else None,
            res_mean_m=_r(res.mean()) if len(res) > 20 else None, res_std_m=_r(res.std()) if len(res) > 20 else None,
            pressure_std_pa=_r(pr.std() * 100.0, 1) if len(pr) > 20 else None,
            noise_model_m=self.params.baro_std_m, rate_hz=self.params.baro_rate_hz)
