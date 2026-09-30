"""1-D vertical fusion: TFmini Plus range + WTGAHRS1 acceleration/attitude + WTGAHRS1 barometer.

State x = [h, v, b_a, b_b]: height above the ground under the vehicle (m, up), vertical speed (m/s, up), world-z
accelerometer bias (m/s^2), and the barometer offset (m) so that baro_height = h + b_b.

  predict   IMU: a_up = (R_row_z . a_body) - g, R from the WTGAHRS1's own quaternion; h, v integrate (a_up - b_a).
  range     TFmini: h = (r + d) cos(tilt), noise from the manual's repeatability fit plus a tilt-error term;
            gated on strength / blind zone / max range / tilt / an innovation test.
  baro      WTGAHRS1 pressure altitude, 20 Hz, learns b_b while the range is good so the baro can carry the height
            (with the IMU) when the range is lost - beyond max_range, tilted, or covered.

This is a small linear Kalman filter, not the 16-state EKF3-style filter used in the simulator (nav_ekf.py): it has no
horizontal channel because no optical-flow sensor is connected. Rangefinder limits and baro noise are read from
simulation/isaac/configs/sensors/fusion_tfmini_mtf01p.yaml when it is available so both stacks agree.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROFILE = Path(__file__).resolve().parents[2] / "simulation/isaac/configs/sensors/fusion_tfmini_mtf01p.yaml"

MODE_INIT, MODE_FUSED, MODE_COAST, MODE_RANGE_ONLY = 0, 1, 2, 3
MODE_NAMES = ("no height yet", "fused", "coasting (range lost)", "range only (no IMU)")


@dataclass
class FusionParams:
    min_range_m: float = 0.10       # TFmini blind zone 0-10 cm
    max_range_m: float = 8.0        # 12 m @ 90 % reflectivity, 4 m @ 10 %; 8 m assumed for a mid-grey floor
    max_tilt_deg: float = 60.0
    mount_down_m: float = 0.0       # TFmini distance below the IMU along the body z axis
    sigma_floor_m: float = 0.02     # accuracy +-5 cm read as ~2 sigma, on top of the repeatability
    tilt_err_deg: float = 1.0       # attitude error assumed when projecting the slant range
    baro_std_m: float = 0.15
    baro_rate_hz: float = 20.0
    acc_noise: float = 0.35         # ArduPilot EK3_ACC_P_NSE default
    acc_bias_walk: float = 0.02     # EK3_ABIAS_P_NSE
    baro_bias_walk: float = 0.05    # m/sqrt(s): weather drift
    range_gate: float = 5.0         # innovation gate, sigma
    reset_after_s: float = 0.5      # rejected this long -> trust the range and reinitialise

    @classmethod
    def from_profile(cls, path: Path = PROFILE, **overrides) -> "FusionParams":
        p = cls()
        try:
            import yaml
            cfg = yaml.safe_load(path.read_text())["fusion"]
            r, b = cfg["rangefinder"], cfg["baro"]
            p.min_range_m, p.max_range_m, p.max_tilt_deg = r["min_range_m"], r["max_range_m"], r["max_tilt_deg"]
            p.sigma_floor_m = max(p.sigma_floor_m, r.get("accuracy_m", 0.05) / 2)
            p.baro_std_m, p.baro_rate_hz = b["noise_std_m"], b["rate_hz"]
        except (OSError, KeyError, TypeError, ImportError):
            pass  # defaults above are the same numbers
        for k, v in overrides.items():
            setattr(p, k, v)
        return p


def tf_repeatability_m(strength: float, freq_hz: float = 100.0) -> float:
    """Distance standard deviation from the TFmini Plus manual, section 2.4 (x = log10 strength, y = log10 rate)."""
    x, y = math.log10(max(strength, 1.0)), math.log10(freq_hz)
    cm = 0.9758 - 0.6072 * x + 1.175 * y + 0.09501 * x * x - 0.2904 * x * y
    return max(cm, 0.3) / 100.0


def up_component(quat, acc) -> float:
    """World-z (up) component of a body-frame vector for quaternion (w, x, y, z): third row of R times the vector."""
    w, x, y, z = quat
    return 2 * (x * z - w * y) * acc[0] + 2 * (y * z + w * x) * acc[1] + (1 - 2 * (x * x + y * y)) * acc[2]


def quat_from_rpy(roll_deg: float, pitch_deg: float, yaw_deg: float):
    r, p, y = (math.radians(a) / 2 for a in (roll_deg, pitch_deg, yaw_deg))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy)


class HeightFusion:
    def __init__(self, params: FusionParams | None = None):
        self.p = params or FusionParams.from_profile()
        self.reset()

    def reset(self) -> None:
        self.x = np.zeros(4)
        self.P = np.diag([1.0, 1.0, 0.3 ** 2, 1.0])
        self.have_height = False
        self.have_baro = False
        self.baro_raw = 0.0
        self.last_range_use_t = -1e9
        self.last_predict_t: float | None = None
        self.last_baro_t = -1e9
        self._rej = 0
        self._rej_since: float | None = None
        self.nis = 0.0            # normalised range innovation of the last range update, sigma
        self.range_state = "waiting"  # ok / gated / reason the last TFmini frame was not used
        self.range_updates = self.range_rejects = self.resets = 0

    # ---- helpers -----------------------------------------------------------------------------------
    @property
    def h(self) -> float:
        return float(self.x[0])

    @property
    def v(self) -> float:
        return float(self.x[1])

    @property
    def h_sigma(self) -> float:
        return float(math.sqrt(max(self.P[0, 0], 0.0)))

    def mode(self, t: float, imu_live: bool = True) -> int:
        if not self.have_height:
            return MODE_INIT
        if not imu_live:
            return MODE_RANGE_ONLY
        return MODE_FUSED if t - self.last_range_use_t < 0.3 else MODE_COAST

    def _update(self, z: float, H: np.ndarray, R: float) -> tuple[float, float]:
        S = float(H @ self.P @ H) + R
        K = (self.P @ H) / S
        y = z - float(H @ self.x)
        self.x = self.x + K * y
        IKH = np.eye(4) - np.outer(K, H)
        self.P = IKH @ self.P @ IKH.T + R * np.outer(K, K)  # Joseph form
        return y, S

    # ---- inputs ------------------------------------------------------------------------------------
    def predict(self, t: float, dt: float, a_up: float) -> None:
        """One IMU step of ``dt`` s with ``a_up`` = world-z acceleration with gravity removed (m/s^2)."""
        self.last_predict_t = t
        if not self.have_height:
            return
        dt = min(dt, 0.05)
        a = a_up - self.x[2]
        h, v = self.x[0], self.x[1]
        self.x[0] = h + v * dt + 0.5 * a * dt * dt
        self.x[1] = v + a * dt
        F = np.array([[1, dt, -0.5 * dt * dt, 0], [0, 1, -dt, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
        g = np.array([0.5 * dt * dt, dt, 0.0, 0.0])
        Q = np.outer(g, g) * self.p.acc_noise ** 2
        Q[2, 2] += (self.p.acc_bias_walk ** 2) * dt
        Q[3, 3] += (self.p.baro_bias_walk ** 2) * dt
        self.P = F @ self.P @ F.T + Q

    def update_range(self, t: float, dist_m: float, strength: int, cos_tilt: float, valid: bool,
                     freq_hz: float = 100.0) -> None:
        p = self.p
        if not valid:
            self.range_state = "no return (weak signal / out of range)"
            return
        if dist_m < p.min_range_m:
            self.range_state = "blind zone (<10 cm)"
            return
        if dist_m > p.max_range_m:
            self.range_state = f"beyond {p.max_range_m:g} m"
            return
        if cos_tilt < math.cos(math.radians(p.max_tilt_deg)):
            self.range_state = f"tilted > {p.max_tilt_deg:g} deg"
            return
        z = (dist_m + p.mount_down_m) * cos_tilt
        tan_tilt = math.sqrt(max(1.0 - cos_tilt ** 2, 0.0)) / max(cos_tilt, 1e-3)
        sigma = math.sqrt((tf_repeatability_m(strength, freq_hz) * cos_tilt) ** 2 + p.sigma_floor_m ** 2
                          + (z * tan_tilt * math.radians(p.tilt_err_deg)) ** 2)
        if not self.have_height:
            self._init_height(t, z, sigma)
            self.range_updates += 1
            self.range_state = "ok"
            return
        H = np.array([1.0, 0.0, 0.0, 0.0])
        S = float(H @ self.P @ H) + sigma ** 2
        y = z - self.x[0]
        self.nis = y / math.sqrt(S)
        if abs(self.nis) > p.range_gate:
            self.range_rejects += 1
            self.range_state = f"rejected by innovation gate ({self.nis:+.1f} sigma)"
            if self._rej_since is None:
                self._rej_since = t
            if t - self._rej_since > p.reset_after_s:  # range consistently disagrees: trust it, restart the state
                self._init_height(t, z, sigma)
                self.resets += 1
            return
        self._rej_since = None
        self._update(z, H, sigma ** 2)
        self.range_updates += 1
        self.last_range_use_t = t
        self.range_state = "ok"

    def _init_height(self, t: float, z: float, sigma: float) -> None:
        """Start (or restart) the height from a range reading; the baro offset is re-derived from the latest baro."""
        self.x[0], self.x[1] = z, 0.0
        self.P[0, :] = self.P[:, 0] = 0.0
        self.P[1, :] = self.P[:, 1] = 0.0
        self.P[0, 0], self.P[1, 1] = sigma ** 2, 0.2 ** 2
        if self.have_baro:
            self.x[3] = self.baro_raw - z
            self.P[3, :] = self.P[:, 3] = 0.0
            self.P[3, 3] = self.p.baro_std_m ** 2 + sigma ** 2
        self.have_height = True
        self.last_range_use_t = t
        self._rej_since = None

    def update_baro(self, t: float, baro_h_m: float) -> None:
        """``baro_h_m``: barometric altitude (any constant offset is absorbed into b_b)."""
        self.baro_raw = baro_h_m
        if not self.have_baro:
            self.have_baro = True
            self.last_baro_t = t
            if self.have_height:
                self.x[3] = baro_h_m - self.x[0]
                self.P[3, 3] = self.p.baro_std_m ** 2 + self.P[0, 0]
            return
        if t - self.last_baro_t < 1.0 / self.p.baro_rate_hz or not self.have_height:
            return
        self.last_baro_t = t
        self._update(baro_h_m, np.array([1.0, 0.0, 0.0, 1.0]), self.p.baro_std_m ** 2)

    def baro_height(self, baro_h_m: float | None) -> float | None:
        """The barometer's implied height (baro reading minus the learned offset)."""
        return None if baro_h_m is None or not self.have_baro else baro_h_m - float(self.x[3])
