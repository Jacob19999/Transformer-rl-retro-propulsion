"""EKF3-style navigation filter on the raw calibrated IMU (numpy port of the design in nav_ekf.py, for the bench).

Not ArduPilot's code: it follows AP_NavEKF3's structure and default tuning, as the simulator's filter does. 16-state
error-state EKF, strapdown prediction from gyro + accelerometer, corrected by the TFmini range and the barometer:

    error state  d_theta (world-frame rotation vector), d_vel, d_pos, d_gyro_bias, d_accel_bias, d_baro_offset
    navigation frame NED (yaw origin arbitrary, no compass), body frame FRD; pos_d is negative above the ground

Differences from nav_ekf.py: the terrain state is replaced by a baro-offset state (flat ground at pos_d = 0), there is no
optical flow (none connected), measurement noise comes from the sensors' own datasheet numbers rather than EKF3's
conservative defaults, and two optional *bench* aids are added because with only range and baro the tilt is otherwise
observable only through the height (that is how EKF3 behaves without GPS or flow):

  gravity aiding      accelerometer direction -> roll/pitch, noise inflated with |a| - g (level-seeking in thrust-vectored
                      flight, so switch it off for flight-like tests)
  stationary aiding   when the sensor is still: zero velocity, and the gyro reading is a bias measurement
  magnetic yaw        optional, needs the magnetometer calibration; heading of the tilt-compensated field vs its value
                      at alignment (so yaw is relative to the power-on heading)
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from frames import (euler_to_matrix, matrix_to_euler, matrix_to_quat, quat_mul, quat_to_matrix, rotvec_to_quat, skew,
                    wrap_pi)
from fusion import MODE_COAST, MODE_FUSED, MODE_INIT, MODE_RANGE_ONLY, FusionParams, tf_repeatability_m
from witmotion import G

N = 16
TH, VEL, POS, BG, BA, BB = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15), 15
PD = 8
E_Z = np.array([0.0, 0.0, 1.0])


@dataclass
class EkfParams:
    # EKF3 process noise defaults (same values as nav_ekf.EkfParams)
    gyro_p_nse: float = 0.015
    acc_p_nse: float = 0.35
    gbias_p_nse: float = 1.0e-3
    abias_p_nse: float = 2.0e-2
    cov_interval_s: float = 0.01
    baro_bias_walk: float = 0.05
    # initial uncertainties
    tilt_std_deg: float = 1.0
    yaw_std_deg: float = 5.0
    vel_std: float = 0.1
    gyro_bias_std_dps: float = 0.3
    accel_bias_std: float = 0.05       # m/s^2, 5 mg (EkfInit default): assumes the accelerometer is calibrated
    # range + baro (from FusionParams / the sim's fusion profile)
    min_range_m: float = 0.10
    max_range_m: float = 8.0
    max_tilt_deg: float = 60.0
    sigma_floor_m: float = 0.02
    range_gate: float = 5.0
    reset_after_s: float = 0.5
    baro_std_m: float = 0.15
    baro_rate_hz: float = 20.0
    baro_gate: float = 5.0
    # bench aids
    grav_sigma: float = 0.05
    grav_max_dev: float = 0.3
    zupt_sigma: float = 0.03
    still_gyro_sigma: float = 0.003
    mag_sigma_rad: float = 0.2
    mag_rate_hz: float = 10.0
    # response guards (see NavEkf._zupt_blocker / the consistent-rejection reset); False reproduces the earlier behaviour for A/B tests
    zupt_guard: bool = True
    fast_reset: bool = True

    @classmethod
    def from_fusion(cls, fp: FusionParams, **overrides) -> "EkfParams":
        p = cls(min_range_m=fp.min_range_m, max_range_m=fp.max_range_m, max_tilt_deg=fp.max_tilt_deg,
                sigma_floor_m=fp.sigma_floor_m, baro_std_m=fp.baro_std_m, baro_rate_hz=fp.baro_rate_hz,
                baro_bias_walk=fp.baro_bias_walk, reset_after_s=fp.reset_after_s, range_gate=fp.range_gate)
        for k, v in overrides.items():
            setattr(p, k, v)
        return p


class NavEkf:
    def __init__(self, params: EkfParams | None = None):
        self.p = params or EkfParams()
        self.tf_offset = np.zeros(3)       # TFmini position from the IMU, body FRD
        self.bench_aiding = True
        self.mag_yaw = False
        self.reset()

    def reset(self) -> None:
        self.aligned = False
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.v = np.zeros(3)
        self.pos = np.zeros(3)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.bb = 0.0
        self.P = np.eye(N) * 1e-2
        self.have_height = self.have_baro = False
        self.baro_raw, self.last_baro_t = 0.0, -1e9
        self.last_range_use_t = -1e9
        self._rej_since: float | None = None
        self._align: list = []
        self._dt_acc = 0.0
        self._win: deque = deque(maxlen=50)
        self.still = False
        self.zupt_blocked = ""          # why the zero-velocity update is being withheld (empty when allowed / not still)
        self._steps = 0                  # IMU steps so far: the clock the range window is indexed by
        self._dt_ema = 0.01
        self._nis_ema = 0.0              # running mean of |range innovation| in sigma: ~0.8 when the model fits
        self._rng_win: deque = deque(maxlen=60)   # (imu step, tilt-compensated height) of every plausible range reading
        self._rej_z: list = []           # consecutive gate-rejected readings (t, height): a smooth run means real motion
        self._mag_ref: float | None = None
        self._mag_norms: deque = deque(maxlen=200)
        self._last_mag_t = -1e9
        self.f_last = np.array([0.0, 0.0, -G])
        self.nis = 0.0
        self.range_state = "waiting"
        self.range_updates = self.range_rejects = self.resets = 0
        self.aid = dict(gravity=0, stationary=0, zupt=0, mag=0)

    # ---- outputs ------------------------------------------------------------------------------------------
    @property
    def R(self) -> np.ndarray:
        return quat_to_matrix(self.q)

    @property
    def rpy_deg(self) -> tuple[float, float, float]:
        return tuple(math.degrees(a) for a in matrix_to_euler(self.R))

    @property
    def h(self) -> float:
        return float(-self.pos[2])

    @property
    def vd(self) -> float:
        return float(-self.v[2])

    @property
    def h_sigma(self) -> float:
        return math.sqrt(max(self.P[PD, PD], 0.0))

    @property
    def cos_tilt(self) -> float:
        return float(self.R[2, 2])

    # ---- alignment -----------------------------------------------------------------------------------------
    def _try_align(self, f: np.ndarray, w: np.ndarray) -> None:
        self._align.append((f, w))
        if len(self._align) < 50:
            return
        fs = np.array([a[0] for a in self._align])
        ws = np.array([a[1] for a in self._align])
        fm = fs[-50:].mean(axis=0)
        n = np.linalg.norm(fm)
        if n < 0.5 * G or (np.abs(ws[-50:].mean(axis=0)).max() > 0.5 and len(self._align) < 150):
            self._align = self._align[-100:]
            return                                   # moving: keep waiting (up to 1.5 s), then align anyway
        u = fm / n
        roll, pitch = math.atan2(-u[1], -u[2]), math.asin(max(-1.0, min(1.0, u[0])))
        self.q = matrix_to_quat(euler_to_matrix(roll, pitch, 0.0))
        p = self.p
        std = np.zeros(N)
        std[0:2], std[2] = math.radians(p.tilt_std_deg), math.radians(p.yaw_std_deg)
        std[VEL], std[POS] = p.vel_std, 10.0
        std[BG], std[BA], std[BB] = math.radians(p.gyro_bias_std_dps), p.accel_bias_std, 5.0
        self.P = np.diag(std ** 2)
        self.aligned = True
        self._align = []

    # ---- prediction -----------------------------------------------------------------------------------------
    def predict(self, dt: float, f_v: np.ndarray, w_v: np.ndarray) -> None:
        """One IMU sample: ``f_v`` specific force and ``w_v`` angular rate, calibrated, in the vehicle FRD frame."""
        self.f_last = f_v
        if not self.aligned:
            self._try_align(f_v, w_v)
            return
        dt = min(max(dt, 1e-4), 0.05)
        self._steps += 1
        self._dt_ema += 0.02 * (dt - self._dt_ema)
        w = w_v - self.bg
        f = f_v - self.ba
        self.q = quat_mul(self.q, rotvec_to_quat(w * dt))
        self.q /= np.linalg.norm(self.q)
        R = quat_to_matrix(self.q)
        a = R @ f + np.array([0.0, 0.0, G])
        self.pos = self.pos + self.v * dt + 0.5 * a * dt * dt
        self.v = self.v + a * dt
        self._dt_acc += dt
        if self._dt_acc >= self.p.cov_interval_s - 1e-9:
            self._predict_cov(self._dt_acc, R, f)
            self._dt_acc = 0.0
        if self.bench_aiding:
            self._win.append((w_v.copy(), float(np.linalg.norm(f_v))))
            self._aid_stationary(w_v, f_v)
            self._aid_gravity(f_v, w)

    def _predict_cov(self, dt: float, R: np.ndarray, f: np.ndarray) -> None:
        F = np.zeros((N, N))
        F[TH, BG] = -R
        F[VEL, TH] = -skew(R @ f)
        F[VEL, BA] = -R
        F[POS, VEL] = np.eye(3)
        Phi = np.eye(N) + F * dt + 0.5 * (F @ F) * dt * dt
        p = self.p
        var = np.zeros(N)
        var[TH], var[VEL] = (p.gyro_p_nse * dt) ** 2, (p.acc_p_nse * dt) ** 2
        var[BG], var[BA] = (p.gbias_p_nse * dt) ** 2, (p.abias_p_nse * dt) ** 2
        var[BB] = (p.baro_bias_walk ** 2) * dt
        self.P = Phi @ self.P @ Phi.T + np.diag(var)
        self._condition()

    def _condition(self) -> None:
        self.P = 0.5 * (self.P + self.P.T)
        d = np.diag(self.P)
        if (d < 0).any():
            self.P += np.diag(np.maximum(-d, 0) + 1e-12)

    # ---- update plumbing -------------------------------------------------------------------------------------
    def _apply(self, dx: np.ndarray) -> None:
        self.q = quat_mul(rotvec_to_quat(dx[TH]), self.q)
        self.q /= np.linalg.norm(self.q)
        self.v = self.v + dx[VEL]
        self.pos = self.pos + dx[POS]
        self.bg = self.bg + dx[BG]
        self.ba = self.ba + dx[BA]
        self.bb += dx[BB]

    def _update(self, y: np.ndarray, H: np.ndarray, Rm: np.ndarray, gate: float) -> tuple[bool, float]:
        """Kalman update; ``(accepted, normalised innovation)``. Gate is on sqrt(y' S^-1 y / m)."""
        m = len(y)
        S = H @ self.P @ H.T + Rm
        try:
            Sinv_y = np.linalg.solve(S, y)
        except np.linalg.LinAlgError:
            return False, 0.0
        nis = math.sqrt(max(float(y @ Sinv_y), 0.0) / m)
        if nis > gate:
            return False, nis
        K = self.P @ H.T @ np.linalg.inv(S)
        self._apply(K @ y)
        IKH = np.eye(N) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ Rm @ K.T
        self._condition()
        return True, nis

    def _perturbed(self, i: int, eps: float):
        q, pos = self.q, self.pos
        if i < 3:
            e = np.zeros(3)
            e[i] = eps
            q = quat_mul(rotvec_to_quat(e), q)
        elif 6 <= i < 9:
            pos = pos.copy()
            pos[i - 6] += eps
        return q, pos

    # ---- range ---------------------------------------------------------------------------------------------
    def _range_pred(self, q, pos) -> float:
        R = quat_to_matrix(q)
        return float(-(pos[2] + (R @ self.tf_offset)[2]) / max(R[2, 2], 1e-3))

    def update_range(self, t: float, dist_m: float, strength: int, valid: bool, freq_hz: float = 100.0) -> None:
        p = self.p
        if not self.aligned:
            self.range_state = "waiting for IMU alignment"
            return
        R = self.R
        if not valid:
            self.range_state = "no return (weak signal / out of range)"
            return
        if dist_m < p.min_range_m:
            self.range_state = "blind zone (<10 cm)"
            return
        if dist_m > p.max_range_m:
            self.range_state = f"beyond {p.max_range_m:g} m"
            return
        if R[2, 2] < math.cos(math.radians(p.max_tilt_deg)):
            self.range_state = f"tilted > {p.max_tilt_deg:g} deg"
            return
        sigma = math.sqrt(tf_repeatability_m(strength, freq_hz) ** 2 + p.sigma_floor_m ** 2)
        z_h = dist_m * R[2, 2] + float((R @ self.tf_offset)[2])
        self._rng_win.append((self._steps, z_h))
        if not self.have_height:
            self._init_height(t, dist_m, sigma)
            self.range_updates += 1
            self.range_state = "ok"
            return
        pred = self._range_pred(self.q, self.pos)
        H = np.zeros((1, N))
        eps = 1e-6
        for i in (0, 1, 2, 8):
            qi, pi = self._perturbed(i, eps)
            H[0, i] = (self._range_pred(qi, pi) - pred) / eps
        ok, nis = self._update(np.array([dist_m - pred]), H, np.array([[sigma ** 2]]), p.range_gate)
        self.nis = float(math.copysign(nis, dist_m - pred))
        self._nis_ema += 0.2 * (min(nis, 10.0) - self._nis_ema)
        if ok:
            self._rej_since = None
            self._rej_z.clear()
            self.range_updates += 1
            self.last_range_use_t = t
            self.range_state = "ok"
            return
        self.range_rejects += 1
        self.range_state = f"rejected by innovation gate ({self.nis:+.1f} sigma)"
        if self._rej_since is None:
            self._rej_since = t
        self._rej_z.append((t, z_h))
        del self._rej_z[:-12]
        if self.p.fast_reset and len(self._rej_z) >= 8:
            # eight rejections in a row that lie on a smooth line are not outliers: the vehicle really moved and the
            # prediction fell behind, so restart from the measurements and carry their slope as the vertical velocity
            tt = np.array([r[0] for r in self._rej_z[-8:]])
            zz = np.array([r[1] for r in self._rej_z[-8:]])
            slope, icpt = np.polyfit(tt - tt[-1], zz, 1)
            if np.std(zz - (slope * (tt - tt[-1]) + icpt)) < 0.02 and abs(slope) < 20.0:
                self._init_height(t, dist_m, sigma, vd=-float(slope))
                self.resets += 1
                return
        if t - self._rej_since > p.reset_after_s:
            self._init_height(t, dist_m, sigma)
            self.resets += 1

    def _init_height(self, t: float, dist_m: float, sigma: float, vd: float = 0.0) -> None:
        R = self.R
        self.pos[2] = -dist_m * R[2, 2] - (R @ self.tf_offset)[2]
        self.v[2] = vd
        for idx in (PD, 5):
            self.P[idx, :] = self.P[:, idx] = 0.0
        self.P[PD, PD] = (sigma * R[2, 2]) ** 2 + 1e-4
        self.P[5, 5] = 0.2 ** 2 if vd == 0.0 else 0.4 ** 2
        self._rej_z.clear()
        if self.have_baro:
            self.bb = self.baro_raw - self.h
            self.P[BB, :] = self.P[:, BB] = 0.0
            self.P[BB, BB] = self.p.baro_std_m ** 2 + self.P[PD, PD]
        self.have_height = True
        self.last_range_use_t = t
        self._rej_since = None

    # ---- baro ----------------------------------------------------------------------------------------------
    def update_baro(self, t: float, baro_h_m: float) -> None:
        self.baro_raw = baro_h_m
        if not self.have_baro:
            self.have_baro, self.last_baro_t = True, t
            if self.have_height:
                self.bb = baro_h_m - self.h
                self.P[BB, BB] = self.p.baro_std_m ** 2 + self.P[PD, PD]
            return
        if not (self.aligned and self.have_height) or t - self.last_baro_t < 1.0 / self.p.baro_rate_hz:
            return
        self.last_baro_t = t
        H = np.zeros((1, N))
        H[0, PD], H[0, BB] = -1.0, 1.0
        self._update(np.array([baro_h_m - (-self.pos[2] + self.bb)]), H, np.array([[self.p.baro_std_m ** 2]]),
                     self.p.baro_gate)

    # ---- bench aids -------------------------------------------------------------------------------------------
    def _aid_gravity(self, f_v: np.ndarray, w: np.ndarray) -> None:
        # The direction comes from the calibrated reading itself, not f - ba: while the sensor is still a tilt error and
        # an accelerometer bias are indistinguishable, and subtracting a wrongly-learned bias here made them fight
        # (a 4 deg roll error stalled at 1 deg). Fixing the offsets is what the 6-position calibration is for.
        f = f_v
        n = float(np.linalg.norm(f))
        dev = abs(n - G) / G
        if dev > self.p.grav_max_dev or float(np.linalg.norm(w)) > 3.0:
            return
        R = self.R
        H = np.zeros((3, N))
        H[:, TH] = -R.T @ skew(E_Z)
        y = f / n - (-R[2, :])
        sig2 = self.p.grav_sigma ** 2 + (1.0 * dev) ** 2
        ok, _ = self._update(y, H, np.eye(3) * sig2, 5.0)
        self.aid["gravity"] += ok

    def _aid_stationary(self, w_v: np.ndarray, f_v: np.ndarray) -> None:
        if len(self._win) < 50:
            self.still = False
            return
        ws = np.array([a[0] for a in self._win])
        ns = np.array([a[1] for a in self._win])
        self.still = bool(ws.std(axis=0).max() < 0.0075 and np.abs(ws.mean(axis=0) - self.bg).max() < 0.03
                          and ns.std() < 0.08 and abs(ns.mean() - G) < 0.1 * G)
        self.zupt_blocked = ""
        if not self.still:
            return
        # No rotation and no acceleration is also a steady descent or climb. Zero-velocity is only right when the range
        # agrees, otherwise it holds the velocity at zero while the vehicle moves and the range gets rejected as outliers.
        self.zupt_blocked = self._zupt_blocker()
        if not self.zupt_blocked:
            H = np.zeros((3, N))
            H[:, VEL] = np.eye(3)
            ok, _ = self._update(-self.v, H, np.eye(3) * self.p.zupt_sigma ** 2, 10.0)
            self.aid["zupt"] += ok
        H = np.zeros((3, N))
        H[:, BG] = np.eye(3)
        ok, _ = self._update(w_v - self.bg, H, np.eye(3) * self.p.still_gyro_sigma ** 2, 10.0)
        self.aid["stationary"] += ok

    def _zupt_blocker(self) -> str:
        """Empty when a zero-velocity update is safe, else the reason it is not."""
        if not self.p.zupt_guard:
            return ""
        if abs(self.v[2]) > 0.15:
            return "vertical speed"
        if self._nis_ema > 1.6:                            # the range keeps disagreeing with the prediction
            return "range innovations"
        recent = [(s, z) for s, z in self._rng_win if s >= self._steps - 30]
        if len(recent) >= 15:                              # the range has been reporting: is the height changing?
            x = np.array([r[0] for r in recent], float) * self._dt_ema
            z = np.array([r[1] for r in recent])
            if abs(np.polyfit(x, z, 1)[0]) > 0.03 or np.ptp(z) > 0.04:
                return "range is changing"
        return ""

    def update_mag(self, t: float, m_v: np.ndarray) -> None:
        """Yaw from the tilt-compensated calibrated field (vehicle frame), relative to the heading at first use."""
        if not (self.mag_yaw and self.aligned) or t - self._last_mag_t < 1.0 / self.p.mag_rate_hz:
            return
        self._last_mag_t = t
        n = float(np.linalg.norm(m_v))
        self._mag_norms.append(n)
        med = float(np.median(self._mag_norms))
        if n < 1e-6 or abs(n - med) > 0.3 * med:
            return                                     # field magnitude off: interference
        m_n = self.R @ m_v
        psi = math.atan2(m_n[1], m_n[0])
        if self._mag_ref is None:
            self._mag_ref = psi
            return
        H = np.zeros((1, N))
        H[0, 2] = 1.0
        ok, _ = self._update(np.array([wrap_pi(self._mag_ref - psi)]), H, np.array([[self.p.mag_sigma_rad ** 2]]), 5.0)
        self.aid["mag"] += ok

    def mag_heading_deg(self, m_v: np.ndarray) -> float | None:
        """Magnetic heading of the vehicle nose (0-360, no declination) from the calibrated field and the EKF tilt.
        Independent of the EKF's own yaw origin: the field direction seen in the EKF's NED frame gives the rotation
        between that frame and magnetic north."""
        if not self.aligned:
            return None
        m_n = self.R @ m_v
        if math.hypot(m_n[0], m_n[1]) < 1e-6:
            return None
        psi_field = math.atan2(m_n[1], m_n[0])
        yaw = matrix_to_euler(self.R)[2]
        return math.degrees(wrap_pi(yaw - psi_field)) % 360.0

    # ---- status ---------------------------------------------------------------------------------------------------
    def mode(self, t: float, imu_live: bool = True) -> int:
        if not (self.aligned and self.have_height):
            return MODE_INIT
        if not imu_live:
            return MODE_RANGE_ONLY
        return MODE_FUSED if t - self.last_range_use_t < 0.3 else MODE_COAST

    def baro_height(self, baro_h_m: float | None) -> float | None:
        return None if baro_h_m is None or not self.have_baro else baro_h_m - self.bb
