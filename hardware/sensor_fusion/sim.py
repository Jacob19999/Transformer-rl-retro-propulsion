"""Synthetic TFmini + WTGAHRS1 byte streams from one shared truth motion, for running the portal without hardware.

The vehicle (FRD body, NED world) bobs 0.7-2.1 m above the floor while rolling, pitching and yawing; every 20 s the TFmini
is "covered" for 3 s so the fused height must coast on the IMU and barometer. The WTGAHRS1 is bolted on with a
configurable mount and has raw gyro bias, accelerometer offset / scale error and magnetometer hard-iron, so the calibration
and mount procedures have something to find. ``CONTROL["pose"]`` freezes the vehicle in a still pose ("level", "nose_up",
or a sensor face such as "+X" pointing up) for the guided procedures. The streams are real wire-format bytes.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

import tfmini
import witmotion
from frames import MOUNTS, DEFAULT_MOUNT, euler_to_matrix, matrix_to_euler, matrix_to_quat, sensor_attitude_enu, vee
from witmotion import G

T0 = time.perf_counter()
BASE_ALT_M = 356.88
FIELD_NED = np.array([133.0, 0.0, 267.0])          # magnetic field in raw counts (0.15 uT / count)
CONTROL = {"pose": "moving"}


@dataclass
class SimErrors:
    mount: str = DEFAULT_MOUNT
    gyro_bias_dps: tuple = (0.40, -0.30, 0.20)
    accel_offset: tuple = (0.15, -0.10, 0.20)      # m/s^2
    accel_scale: tuple = (1.005, 0.995, 1.003)
    mag_offset: tuple = (30.0, -20.0, 10.0)        # counts
    noise: dict = field(default_factory=lambda: dict(gyro_dps=0.05, acc=0.02, mag=3.0, baro=0.05))
    bandwidth_hz: float | None = None          # the sensor's own output low-pass on acc + gyro (WTGAHRS1: ~20 Hz); None = ideal


def _static_R(pose: str, mount: np.ndarray) -> np.ndarray | None:
    if pose == "level":
        return np.eye(3)
    if pose == "nose_up":
        return euler_to_matrix(0.0, math.radians(60.0), 0.0)
    if len(pose) == 2 and pose[0] in "+-" and pose[1] in "XYZ":     # this sensor axis points up
        f_s = np.zeros(3)
        f_s["XYZ".index(pose[1])] = G if pose[0] == "+" else -G
        u = -(mount @ f_s) / G                                      # third row of R_nb
        theta, phi = -math.asin(max(-1.0, min(1.0, u[0]))), (math.atan2(u[1], u[2]) if abs(u[0]) < 0.999 else 0.0)
        return euler_to_matrix(phi, theta, 0.0)
    return None


def truth(t: float, mount: np.ndarray | None = None) -> dict:
    """Height, vertical acceleration and attitude (R_nb, body FRD -> NED) at time ``t``."""
    static = _static_R(CONTROL["pose"], MOUNTS[DEFAULT_MOUNT] if mount is None else mount)
    if static is not None:
        return dict(h=1.4, a_up=0.0, R=static, rpy=matrix_to_euler(static), static=True)
    h = 1.4 + 0.7 * math.sin(0.5 * t)
    a_up = -0.7 * 0.25 * math.sin(0.5 * t)
    rpy = (math.radians(15) * math.sin(0.8 * t + 1.0), math.radians(10) * math.sin(0.55 * t), math.radians(25.0) * t)
    return dict(h=h, a_up=a_up, R=euler_to_matrix(*rpy), rpy=rpy, static=False)


def body_rates(t: float, eps: float = 1e-3) -> np.ndarray:
    """Body angular rate in rad/s (FRD): vee(R^T dR/dt) by central difference."""
    Ra, Rb, R = truth(t - eps)["R"], truth(t + eps)["R"], truth(t)["R"]
    return vee(R.T @ (Rb - Ra) / (2 * eps))


class _Paced:
    def __init__(self, hz: float):
        self.dt = 1.0 / hz
        self._next = time.perf_counter()
        self.rng = random.Random(7)

    def read(self) -> bytes:
        time.sleep(max(0.0, self._next - time.perf_counter()) + 0.004)
        out, now = bytearray(), time.perf_counter()
        while self._next <= now:
            out += self.emit(self._next - T0)
            self._next += self.dt
        return bytes(out)

    def write(self, data: bytes) -> None:  # commands are ignored by the simulator
        pass

    def close(self) -> None:
        pass


class SimTfmini(_Paced):
    def __init__(self, hz: float = tfmini.NOMINAL_HZ, latency_s: float = 0.0):
        super().__init__(hz)
        self.latency_s = latency_s                    # the reading reflects the scene this long ago

    def emit(self, t: float) -> bytes:
        tr = truth(max(t - self.latency_s, 0.0))
        r = tr["h"] / max(tr["R"][2, 2], 0.05)                       # beam along body +z (down)
        covered = (t % 20.0) > 12.0 and (t % 20.0) < 15.0 and not tr["static"]
        if r > 8.0 or covered:
            return tfmini.encode_frame(0, 30)
        dist_cm = int(round(r * 100 + self.rng.gauss(0, 0.4)))
        return tfmini.encode_frame(dist_cm, int(min(5400 / r, 60000) + self.rng.gauss(0, 15)), 31.0)


class SimWitmotion(_Paced):
    def __init__(self, hz: float = witmotion.DEFAULT_HZ, errors: SimErrors | None = None):
        super().__init__(hz)
        self.err = errors or SimErrors()
        self.R_vs = MOUNTS[self.err.mount]
        self._lp = None                               # low-pass state (acc, gyro)

    def emit(self, t: float) -> bytes:
        e, n = self.err, self.rng.gauss
        tr = truth(t, self.R_vs)
        R_nb, Rsv = tr["R"], self.R_vs.T                              # sensor = R_vs^T vehicle
        f_b = -(tr["a_up"] + G) * R_nb[2, :]                          # specific force, body FRD (up = -z)
        w_b = np.zeros(3) if tr["static"] else body_rates(t)
        acc = Rsv @ f_b
        acc = acc / np.array(e.accel_scale) + np.array(e.accel_offset) + np.array([n(0, e.noise["acc"]) for _ in range(3)])
        gyro = np.degrees(Rsv @ w_b) + np.array(e.gyro_bias_dps) + np.array([n(0, e.noise["gyro_dps"]) for _ in range(3)])
        mag = Rsv @ (R_nb.T @ FIELD_NED) + np.array(e.mag_offset) + np.array([n(0, e.noise["mag"]) for _ in range(3)])
        if e.bandwidth_hz:
            a = 1.0 - math.exp(-self.dt * 2 * math.pi * e.bandwidth_hz)
            cur = np.concatenate([acc, gyro])
            self._lp = cur if self._lp is None else self._lp + a * (cur - self._lp)
            acc, gyro = self._lp[:3], self._lp[3:]
        R_es = sensor_attitude_enu(R_nb, self.R_vs)
        q = matrix_to_quat(R_es)
        rpy = tuple(math.degrees(a) for a in matrix_to_euler(R_es))
        alt = BASE_ALT_M + tr["h"] + n(0, e.noise["baro"]) + 0.3 * math.sin(0.02 * t)
        pressure = 101325.0 * (1 - alt / 44330.0) ** 5.255
        return witmotion.encode_burst(acc / G, gyro, rpy, tuple(int(v) for v in mag), pressure, alt * 100, tuple(q))
