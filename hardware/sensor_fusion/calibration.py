"""IMU calibration and mount orientation, applied on the host in front of the EKF (nothing is written to the sensor).

    corrected_sensor = (raw - offset) * scale        accelerometer and magnetometer
    corrected_sensor = raw - bias                    gyroscope
    vehicle          = R_trim @ R_vs @ corrected_sensor        (R_vs from the mount key, R_trim from the trim angles)

Procedures (``CalibSession``): gyro bias (still), 6-position accelerometer (offset + scale per axis), magnetometer sphere
fit (rotate through all attitudes), level trim, and a two-step mount detection (level, then nose-up) that finds which
sensor axis is the vehicle's forward, right and down axis. Fits are pure functions so they can be tested without hardware.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from frames import DEFAULT_MOUNT, MOUNTS, euler_to_matrix, nearest_mount
from witmotion import G

PATH = Path(__file__).with_name("calibration.json")
FACES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")   # the sensor axis that points up


@dataclass
class Calibration:
    gyro_bias_dps: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    accel_offset: list = field(default_factory=lambda: [0.0, 0.0, 0.0])   # m/s^2
    accel_scale: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    mag_offset: list = field(default_factory=lambda: [0.0, 0.0, 0.0])     # raw counts
    mag_scale: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    mount: str = DEFAULT_MOUNT
    trim_deg: list = field(default_factory=lambda: [0.0, 0.0, 0.0])       # roll, pitch, yaw
    tf_offset_m: list = field(default_factory=lambda: [0.0, 0.0, 0.0])    # TFmini position from the IMU, vehicle FRD
    meta: dict = field(default_factory=dict)                               # when / how each item was calibrated

    # ---- derived ------------------------------------------------------------------------------------------
    @property
    def R(self) -> np.ndarray:
        """Total sensor -> vehicle rotation."""
        r, p, y = (math.radians(a) for a in self.trim_deg)
        return euler_to_matrix(r, p, y) @ MOUNTS[self.mount]

    def accel_sensor(self, raw) -> np.ndarray:
        return (np.asarray(raw, float) - self.accel_offset) * self.accel_scale

    def accel_vehicle(self, raw) -> np.ndarray:
        return self.R @ self.accel_sensor(raw)

    def gyro_vehicle_rad(self, raw_dps) -> np.ndarray:
        return self.R @ (np.radians(np.asarray(raw_dps, float) - self.gyro_bias_dps))

    def mag_vehicle(self, raw) -> np.ndarray:
        return self.R @ ((np.asarray(raw, float) - self.mag_offset) * self.mag_scale)

    @property
    def has_mag(self) -> bool:
        return "mag" in self.meta

    # ---- persistence ----------------------------------------------------------------------------------------
    def validate(self) -> None:
        if self.mount not in MOUNTS:
            raise ValueError(f"unknown mount {self.mount!r}")
        for name in ("gyro_bias_dps", "accel_offset", "accel_scale", "mag_offset", "mag_scale", "trim_deg", "tf_offset_m"):
            v = getattr(self, name)
            if len(v) != 3 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in v):
                raise ValueError(f"{name} must be three finite numbers")
        if min(self.accel_scale) < 0.5 or max(self.accel_scale) > 1.5 or min(self.mag_scale) <= 0:
            raise ValueError("scale factors out of range")

    def save(self, path: Path = PATH) -> None:
        self.validate()
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path = PATH) -> "Calibration":
        try:
            data = json.loads(path.read_text())
            c = cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})
            c.validate()
            return c
        except (OSError, ValueError, TypeError, KeyError):
            return cls()

    def stamp(self, item: str, **info) -> None:
        self.meta[item] = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), **info)


# ---- fits (pure) -------------------------------------------------------------------------------------------
def is_still(acc: np.ndarray, gyro_dps: np.ndarray, acc_std_max: float = 0.25, gyro_std_max: float = 0.6,
             gyro_mean_max: float = 10.0) -> str | None:
    """None when the captured samples look stationary, else the reason they do not."""
    if acc.std(axis=0).max() > acc_std_max:
        return f"accelerometer moved (std {acc.std(axis=0).max():.2f} m/s^2)"
    if gyro_dps.std(axis=0).max() > gyro_std_max or np.abs(gyro_dps.mean(axis=0)).max() > gyro_mean_max:
        return f"rotation detected (gyro std {gyro_dps.std(axis=0).max():.2f} dps)"
    return None


def classify_face(mean_acc) -> str | None:
    """Which sensor axis points up (specific force is along +up), or None when not clearly on a face."""
    a = np.asarray(mean_acc, float)
    n = float(np.linalg.norm(a))
    if not 0.85 * G < n < 1.15 * G:
        return None
    i = int(np.argmax(np.abs(a)))
    if abs(a[i]) < 0.94 * n:  # within ~20 deg of an axis
        return None
    return ("+" if a[i] > 0 else "-") + "XYZ"[i]


def fit_gyro_bias(gyro_dps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(bias, per-axis noise std) in dps from a still capture."""
    return gyro_dps.mean(axis=0), gyro_dps.std(axis=0)


def fit_accel_six(caps: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Offset and scale per axis from the +/- faces of each axis: offset = (a+ + a-)/2, scale = 2g / (a+ - a-).
    Returns (offset, scale, axes solved); axes lacking one of their faces stay at 0 / 1."""
    off, scale, done = np.zeros(3), np.ones(3), []
    for i, ax in enumerate("XYZ"):
        if "+" + ax in caps and "-" + ax in caps:
            hi, lo = caps["+" + ax][i], caps["-" + ax][i]
            if hi - lo < 1.0:
                raise ValueError(f"{ax} faces do not differ ({hi:.2f} vs {lo:.2f} m/s^2)")
            off[i], scale[i] = (hi + lo) / 2, 2 * G / (hi - lo)
            done.append(ax)
    return off, scale, done


def fit_mag_sphere(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Hard-iron offset, per-axis soft-iron scale (unit mean), fit residual (fraction of the radius) and the radius."""
    A = np.hstack([2 * m, np.ones((len(m), 1))])
    sol, *_ = np.linalg.lstsq(A, (m ** 2).sum(axis=1), rcond=None)
    c = sol[:3]
    r = math.sqrt(max(sol[3] + c @ c, 1e-12))
    d = m - c
    half = (np.percentile(d, 98, axis=0) - np.percentile(d, 2, axis=0)) / 2
    scale = half.mean() / np.maximum(half, 1e-9)
    resid = float(np.std(np.linalg.norm(d * scale, axis=1)) / max(np.linalg.norm(d * scale, axis=1).mean(), 1e-9))
    return c, scale, resid, r


def mag_coverage(m: np.ndarray, center: np.ndarray | None = None) -> float:
    """Fraction of 8 azimuth x 4 elevation direction bins visited (1.0 = the sensor saw all orientations)."""
    d = m - (m.mean(axis=0) if center is None else center)
    n = np.linalg.norm(d, axis=1)
    d = d[n > 1e-9] / n[n > 1e-9, None]
    if not len(d):
        return 0.0
    az = ((np.arctan2(d[:, 1], d[:, 0]) + math.pi) / (2 * math.pi) * 8).astype(int).clip(0, 7)
    el = ((np.arcsin(d[:, 2].clip(-1, 1)) + math.pi / 2) / math.pi * 4).astype(int).clip(0, 3)
    return len({(a, e) for a, e in zip(az, el)}) / 32.0


def level_trim_deg(f_vehicle) -> tuple[float, float]:
    """(roll, pitch) in degrees of a level-at-rest capture in the vehicle frame; the trim that removes them."""
    f = np.asarray(f_vehicle, float)
    f = f / np.linalg.norm(f)
    return math.degrees(math.atan2(-f[1], -f[2])), math.degrees(math.asin(max(-1.0, min(1.0, f[0]))))


def detect_mount(f_level, f_nose_up) -> tuple[str, float, float]:
    """Mount from two still captures of the sensor's specific force (sensor axes): the vehicle level, then pitched nose
    up by at least ~20 deg. Returns (mount key, residual angle of the snap in degrees, nose-up angle in degrees)."""
    down = -np.asarray(f_level, float)
    down = down / np.linalg.norm(down)
    u = np.asarray(f_nose_up, float)
    u = u / np.linalg.norm(u)
    fwd = u - (u @ down) * down                     # nose-up: forward tilts toward "up", the rest of u is along down
    s = float(np.linalg.norm(fwd))
    if s < 0.3:
        raise ValueError("nose-up tilt too small (need more than ~20 degrees)")
    fwd = fwd / s
    right = np.cross(down, fwd)
    key, resid = nearest_mount(np.vstack([fwd, right, down]))
    return key, resid, math.degrees(math.asin(min(s, 1.0)))


# ---- guided procedures -----------------------------------------------------------------------------------------
class CalibSession:
    """One procedure at a time; the hub feeds it every raw IMU sample. ``status()`` is what the page polls."""

    DURATIONS = {"gyro": 10.0, "accel": 3.0, "level": 3.0, "mount_level": 3.0, "mount_nose": 3.0}
    MAG_MAX_S = 90.0
    SETTLE_S = 0.5      # samples right after the button press (hand still moving) are discarded

    def __init__(self, calib_getter, calib_setter):
        self._get, self._set = calib_getter, calib_setter
        self.task: str | None = None
        self.face: str | None = None
        self.t0: float | None = None       # timestamp of the first sample after start()
        self.t_last = 0.0
        self.buf: list = []
        self.accel_caps: dict[str, np.ndarray] = {}
        self.mount_level: np.ndarray | None = None
        self.message = "idle"
        self.error: str | None = None
        self.result: dict = {}

    # ---- control ------------------------------------------------------------------------------------------------
    def start(self, task: str, face: str | None = None) -> None:
        if task not in ("gyro", "accel", "level", "mount_level", "mount_nose", "mag"):
            raise ValueError(f"unknown calibration task {task!r}")
        if task == "accel" and face not in FACES:
            raise ValueError(f"accel needs a face from {FACES}")
        if task == "mount_nose" and self.mount_level is None:
            raise ValueError("capture the level pose first")
        self.task, self.face, self.t0, self.t_last, self.buf, self.error = task, face, None, 0.0, [], None
        self.message = {"gyro": "keep the sensor still...", "accel": f"keep still with {face} pointing up...",
                        "level": "keep the vehicle level and still...", "mount_level": "keep the vehicle level and still...",
                        "mount_nose": "keep the nose pitched up and still...",
                        "mag": "rotate the sensor slowly through every orientation..."}[task]

    def cancel(self) -> None:
        self.task, self.message = None, "cancelled"

    def reset_captures(self) -> None:
        self.accel_caps.clear()
        self.mount_level = None

    # ---- data ---------------------------------------------------------------------------------------------------
    def feed(self, t: float, raw_acc, raw_gyro_dps, raw_mag) -> None:
        if self.task is None:
            return
        if self.t0 is None:
            self.t0 = t
        self.t_last = t
        elapsed = t - self.t0
        if self.task == "mag" or elapsed >= self.SETTLE_S:
            self.buf.append((raw_acc, raw_gyro_dps, raw_mag))
        if self.task == "mag":
            if elapsed > self.MAG_MAX_S:
                self.finish_mag()
            return
        if elapsed >= self.DURATIONS[self.task]:
            self._finish()

    def _arrays(self):
        acc = np.array([b[0] for b in self.buf if b[0] is not None], float)
        gyro = np.array([b[1] for b in self.buf if b[1] is not None], float)
        return acc, gyro

    def _fail(self, msg: str) -> None:
        self.task, self.error, self.message = None, msg, "failed"

    def _finish(self) -> None:
        task, c = self.task, self._get()
        acc, gyro = self._arrays()
        if len(acc) < 20 or len(gyro) < 20:
            return self._fail("not enough samples (is the IMU streaming acc + gyro?)")
        reason = is_still(acc, gyro)
        if reason:
            return self._fail(reason + ": hold still and retry")
        self.task = None
        try:
            if task == "gyro":
                bias, std = fit_gyro_bias(gyro)
                c.gyro_bias_dps = [round(float(b), 4) for b in bias]
                c.stamp("gyro", noise_std_dps=[round(float(s), 4) for s in std], samples=len(gyro))
                self.result = dict(bias_dps=c.gyro_bias_dps, noise_std_dps=[round(float(s), 3) for s in std])
                self.message = "gyro bias saved"
            elif task == "accel":
                mean = acc.mean(axis=0)
                got = classify_face(mean)
                if got != self.face:
                    return self._fail(f"expected {self.face} up but the sensor reads {got or 'no clear face'} "
                                      f"(a = {np.round(mean, 2).tolist()})")
                self.accel_caps[self.face] = mean
                off, scale, done = fit_accel_six(self.accel_caps)
                if done:
                    for i, ax in enumerate("XYZ"):
                        if ax in done:
                            c.accel_offset[i], c.accel_scale[i] = round(float(off[i]), 4), round(float(scale[i]), 5)
                    c.stamp("accel", axes=done, faces=sorted(self.accel_caps))
                self.result = dict(faces=sorted(self.accel_caps), axes_solved=done, offset=c.accel_offset, scale=c.accel_scale)
                self.message = f"{self.face} captured" + (f"; solved {''.join(done)}" if done else "")
            elif task == "level":
                roll, pitch = level_trim_deg(MOUNTS[c.mount] @ c.accel_sensor(acc.mean(axis=0)))
                c.trim_deg = [round(roll, 3), round(pitch, 3), c.trim_deg[2]]
                c.stamp("level", roll_deg=round(roll, 3), pitch_deg=round(pitch, 3))
                self.result = dict(trim_deg=c.trim_deg)
                self.message = f"level trim roll {roll:+.2f} deg, pitch {pitch:+.2f} deg saved"
            elif task == "mount_level":
                self.mount_level = c.accel_sensor(acc.mean(axis=0))
                self.message = "level pose captured: now pitch the nose up (>20 deg, ideally 45-90) and capture"
                self.result = dict(level=self.mount_level.round(3).tolist())
            elif task == "mount_nose":
                key, resid, nose = detect_mount(self.mount_level, c.accel_sensor(acc.mean(axis=0)))
                c.mount = key
                c.trim_deg = [0.0, 0.0, 0.0]
                roll, pitch = level_trim_deg(MOUNTS[c.mount] @ self.mount_level)
                c.trim_deg = [round(roll, 3), round(pitch, 3), 0.0]
                c.stamp("mount", key=key, snap_residual_deg=round(resid, 1), nose_up_deg=round(nose, 1))
                self.result = dict(mount=key, snap_residual_deg=round(resid, 1), nose_up_deg=round(nose, 1),
                                   trim_deg=c.trim_deg)
                self.message = f"mount {key} (snap residual {resid:.1f} deg); level trim {roll:+.2f}/{pitch:+.2f} deg"
                self.mount_level = None
        except ValueError as e:
            return self._fail(str(e))
        self._set(c)

    def finish_mag(self) -> None:
        if self.task != "mag":
            return
        c = self._get()
        m = np.array([b[2] for b in self.buf if b[2] is not None], float)
        self.task = None
        if len(m) < 200:
            return self._fail("not enough magnetometer samples")
        off, scale, resid, radius = fit_mag_sphere(m)
        if radius < 30.0:
            return self._fail(f"field only {radius:.0f} counts (expected hundreds): rotate the sensor through all orientations")
        cov = mag_coverage(m, off)            # seen from the fitted centre: a cluster of noise around one point covers little
        if cov < 0.5:
            return self._fail(f"orientation coverage only {cov * 100:.0f} %: rotate through more attitudes and retry")
        if resid > 0.15:
            return self._fail(f"sphere fit residual {resid * 100:.0f} % of the radius: interference nearby?")
        c.mag_offset, c.mag_scale = [round(float(x), 2) for x in off], [round(float(x), 4) for x in scale]
        c.stamp("mag", coverage=round(cov, 2), residual=round(resid, 3), radius_counts=round(radius, 1), samples=len(m))
        self.result = dict(offset=c.mag_offset, scale=c.mag_scale, coverage=round(cov, 2), residual=round(resid, 3))
        self.message = f"magnetometer saved (coverage {cov * 100:.0f} %, residual {resid * 100:.1f} %)"
        self._set(c)

    # ---- reporting ----------------------------------------------------------------------------------------------
    def status(self) -> dict:
        elapsed = 0.0 if self.t0 is None else self.t_last - self.t0
        frac, cov = 0.0, None
        if self.task == "mag":
            m = np.array([b[2] for b in self.buf if b[2] is not None], float)
            cov = mag_coverage(m) if len(m) > 20 else 0.0
            frac = min(elapsed / self.MAG_MAX_S, 1.0)
        elif self.task:
            frac = min(elapsed / self.DURATIONS[self.task], 1.0)
        return dict(task=self.task, face=self.face, progress=round(frac, 3), message=self.message, error=self.error,
                    result=self.result, accel_faces=sorted(self.accel_caps), mount_level=self.mount_level is not None,
                    mag_coverage=None if cov is None else round(cov, 2))

