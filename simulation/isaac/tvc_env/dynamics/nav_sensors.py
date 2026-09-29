"""
Aiding-sensor models for the navigation filter (batched torch): a downward single-point lidar
(Benewake TFmini Plus), a downward optical-flow sensor (MicoAir MTF-01P) and a barometer (the
WTGAHRS1's).

Each model samples the *physics truth* (pose, velocity, body rate) at its own rate, corrupts it, holds it
for its latency, and hands it over as a ``Measurement`` that says how many physics substeps ago it was
taken, so the estimator can compare it with the state it had then.

Frames: quaternions are Isaac wxyz rotating body local axes into the Z-up world; body vectors are FRD
(x forward, y right, z down); conversion only through tvc_env.common.frames. Flat ground at z = 0.

Sensor geometry
---------------
The downward sensors look along body +z (FRD down). With cos_tilt = R_zz (the world-up component of the
body-up axis) a sensor mounted ``mount_down_m`` below the root along body z reads the slant range

    r = z_root / cos_tilt - mount_down_m

The optical-flow output is the apparent angular rate of the ground in the sensor's axes,

    f = ( v_s,y / r - w_x ,  -v_s,x / r - w_y ),        v_s = v_body + w x mount

(v_body the root velocity in body FRD, w the true body rate). It contains the vehicle's own rotation; a
flight computer adds its gyro back, ``f + (w_x, w_y) = (v_s,y / r, -v_s,x / r)``, as ArduPilot does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from tvc_env.common.frames import isaac_velocity_to_frd
from tvc_env.common.quaternions import inverse, normalize, rotate_vector

MIN_COS_TILT = 0.05


@dataclass
class Measurement:
    """One delivered sample. ``age`` is how many physics substeps ago it was taken."""
    kind: str
    value: Tensor          # (n,) or (n, 2)
    valid: Tensor          # (n,) bool
    age: int


def cos_tilt(q: Tensor) -> Tensor:
    """World-up component of the body-up axis for Isaac wxyz quaternions."""
    return 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)


def slant_range(position_z: Tensor, cos_t: Tensor, mount_down_m: float) -> Tensor:
    return position_z / cos_t.clamp(min=MIN_COS_TILT) - mount_down_m


class _Sampler:
    """Fixed-rate sampling on the physics grid plus a fixed transport latency."""

    def __init__(self, dt: float, rate_hz: float, latency_s: float):
        self.dt = float(dt)
        self.rate = min(float(rate_hz), 1.0 / self.dt)
        self.age = int(round(latency_s / self.dt))
        self._phase = 0.0
        self._step = 0
        self._pending: list[tuple[int, Measurement]] = []

    def reset(self) -> None:
        self._pending.clear()

    def due(self) -> bool:
        self._step += 1
        self._phase += self.dt * self.rate
        if self._phase >= 1.0 - 1e-9:
            self._phase = min(self._phase - 1.0, 1.0)
            return True
        return False

    def push(self, measurement: Measurement) -> None:
        self._pending.append((self._step + self.age, measurement))

    def deliver(self) -> list[Measurement]:
        ready = [m for due, m in self._pending if due <= self._step]
        self._pending = [(due, m) for due, m in self._pending if due > self._step]
        return ready


def _uniform(gen, device, *shape) -> Tensor:
    return torch.rand(*shape, device=device, generator=gen) * 2.0 - 1.0


# --------------------------------------------------------------------------- rangefinder

@dataclass(frozen=True)
class RangefinderParams:
    rate_hz: float = 100.0
    latency_s: float = 0.01
    min_range_m: float = 0.1
    max_range_m: float = 8.0
    noise_std_m: float = 0.03
    accuracy_m: float = 0.05          # per-episode bias half-width up to 5 m
    accuracy_pct: float = 1.0         # beyond 5 m, percent of range
    resolution_m: float = 0.01
    max_tilt_deg: float = 60.0
    dropout_prob: float = 0.0
    mount_down_m: float = 0.0


class Rangefinder:
    """Benewake TFmini Plus (or any single-point downward ranger)."""

    def __init__(self, n: int, device, dt: float, params: RangefinderParams, generator=None):
        self.n, self.device, self.p, self._gen = n, torch.device(device), params, generator
        self.sampler = _Sampler(dt, params.rate_hz, params.latency_s)
        self._u = torch.zeros(n, device=self.device)         # per-episode systematic error, in [-1, 1]
        self._cos_limit = math.cos(math.radians(params.max_tilt_deg))
        self.last_value = torch.zeros(n, device=self.device)
        self.last_valid = torch.zeros(n, dtype=torch.bool, device=self.device)

    def reset(self, ids: Tensor) -> None:
        self._u[ids] = _uniform(self._gen, self.device, len(ids))
        self.sampler.reset()

    def true_range(self, q: Tensor, position: Tensor) -> Tensor:
        return slant_range(position[:, 2], cos_tilt(q), self.p.mount_down_m)

    def step(self, q: Tensor, position: Tensor) -> list[Measurement]:
        if self.sampler.due():
            p = self.p
            r = self.true_range(q, position)
            bound = torch.where(r < 5.0, torch.full_like(r, p.accuracy_m), r * (p.accuracy_pct / 100.0))
            noise = torch.randn(self.n, device=self.device, generator=self._gen) * p.noise_std_m
            measured = r + self._u * bound + noise
            if p.resolution_m > 0.0:
                measured = torch.round(measured / p.resolution_m) * p.resolution_m
            valid = (r >= p.min_range_m) & (r <= p.max_range_m) & (cos_tilt(q) >= self._cos_limit)
            if p.dropout_prob > 0.0:
                valid = valid & (torch.rand(self.n, device=self.device, generator=self._gen) >= p.dropout_prob)
            self.sampler.push(Measurement("range", measured, valid, self.sampler.age))
        out = self.sampler.deliver()
        for m in out:
            self.last_value, self.last_valid = m.value, m.valid
        return out


# --------------------------------------------------------------------------- optical flow

@dataclass(frozen=True)
class FlowParams:
    rate_hz: float = 100.0
    latency_s: float = 0.015
    noise_std_rad_s: float = 0.05
    scale_error_pct: float = 5.0      # per-axis, per-episode half-width
    bias_rad_s: float = 0.01          # per-axis, per-episode half-width
    min_range_m: float = 0.08
    max_height_m: float = 10.0
    max_rate_rad_s: float = 7.0
    max_tilt_deg: float = 45.0
    dropout_prob: float = 0.0
    mount_down_m: float = 0.0


class OpticalFlow:
    """MicoAir MTF-01P optical-flow sensor (downward camera)."""

    def __init__(self, n: int, device, dt: float, params: FlowParams, generator=None):
        self.n, self.device, self.p, self._gen = n, torch.device(device), params, generator
        self.sampler = _Sampler(dt, params.rate_hz, params.latency_s)
        self._scale = torch.zeros(n, 2, device=self.device)
        self._bias = torch.zeros(n, 2, device=self.device)
        self._cos_limit = math.cos(math.radians(params.max_tilt_deg))
        self.last_value = torch.zeros(n, 2, device=self.device)
        self.last_valid = torch.zeros(n, dtype=torch.bool, device=self.device)

    def reset(self, ids: Tensor) -> None:
        k = len(ids)
        self._scale[ids] = _uniform(self._gen, self.device, k, 2) * (self.p.scale_error_pct / 100.0)
        self._bias[ids] = _uniform(self._gen, self.device, k, 2) * self.p.bias_rad_s
        self.sampler.reset()

    def truth_flow(self, q: Tensor, position: Tensor, velocity_world: Tensor, rate_frd: Tensor):
        """Noise-free flow (n,2) and the slant range (n,) it is measured through."""
        r = slant_range(position[:, 2], cos_tilt(q), self.p.mount_down_m)
        v_body = isaac_velocity_to_frd(rotate_vector(inverse(normalize(q)), velocity_world))
        mount = torch.zeros_like(v_body)
        mount[:, 2] = self.p.mount_down_m
        v_sensor = v_body + torch.linalg.cross(rate_frd, mount)
        rr = r.clamp(min=self.p.min_range_m)
        flow = torch.stack((v_sensor[:, 1] / rr - rate_frd[:, 0], -v_sensor[:, 0] / rr - rate_frd[:, 1]), dim=-1)
        return flow, r

    def step(self, q: Tensor, position: Tensor, velocity_world: Tensor, rate_frd: Tensor) -> list[Measurement]:
        if self.sampler.due():
            p = self.p
            flow, r = self.truth_flow(q, position, velocity_world, rate_frd)
            noise = torch.randn(self.n, 2, device=self.device, generator=self._gen) * p.noise_std_rad_s
            measured = flow * (1.0 + self._scale) + self._bias + noise
            height = position[:, 2]
            valid = ((r >= p.min_range_m) & (height <= p.max_height_m) & (cos_tilt(q) >= self._cos_limit)
                     & (flow.abs().max(dim=-1).values <= p.max_rate_rad_s))
            if p.dropout_prob > 0.0:
                valid = valid & (torch.rand(self.n, device=self.device, generator=self._gen) >= p.dropout_prob)
            self.sampler.push(Measurement("flow", measured, valid, self.sampler.age))
        out = self.sampler.deliver()
        for m in out:
            self.last_value, self.last_valid = m.value, m.valid
        return out


# --------------------------------------------------------------------------- barometer

@dataclass(frozen=True)
class BaroParams:
    rate_hz: float = 20.0
    latency_s: float = 0.05
    noise_std_m: float = 0.15
    drift_sigma_m: float = 0.5        # Gauss-Markov bias, zero at power-up (the height is zeroed at boot)
    drift_tau_s: float = 300.0
    resolution_m: float = 0.01


class Barometer:
    """Barometric height (world z), referenced to the start height at boot."""

    def __init__(self, n: int, device, dt: float, params: BaroParams, generator=None):
        self.n, self.device, self.p, self._gen, self.dt = n, torch.device(device), params, generator, float(dt)
        self.sampler = _Sampler(dt, params.rate_hz, params.latency_s)
        self._bias = torch.zeros(n, device=self.device)
        self._phi = math.exp(-self.dt / params.drift_tau_s)
        self.last_value = torch.zeros(n, device=self.device)

    def reset(self, ids: Tensor) -> None:
        self._bias[ids] = 0.0
        self.sampler.reset()

    def step(self, position: Tensor) -> list[Measurement]:
        p = self.p
        if p.drift_sigma_m > 0.0:
            innovation = math.sqrt(max(1.0 - self._phi ** 2, 0.0)) * p.drift_sigma_m
            self._bias = self._phi * self._bias + innovation * torch.randn(self.n, device=self.device,
                                                                          generator=self._gen)
        if self.sampler.due():
            measured = position[:, 2] + self._bias + torch.randn(self.n, device=self.device, generator=self._gen) * p.noise_std_m
            if p.resolution_m > 0.0:
                measured = torch.round(measured / p.resolution_m) * p.resolution_m
            self.sampler.push(Measurement("baro", measured, torch.ones(self.n, dtype=torch.bool, device=self.device),
                                          self.sampler.age))
        out = self.sampler.deliver()
        for m in out:
            self.last_value = m.value
        return out


# --------------------------------------------------------------------------- pad marker camera

@dataclass(frozen=True)
class MarkerParams:
    """Downward camera watching a fiducial (AprilTag-style) marker at the landing pad."""
    rate_hz: float = 30.0
    latency_s: float = 0.06           # image capture + detection
    fov_deg: float = 60.0             # full field of view, used per axis
    resolution_px: float = 640.0      # pixels across the field of view
    marker_size_m: float = 0.4
    min_pixels: float = 24.0          # marker must span this many pixels to be detected
    min_range_m: float = 0.3
    noise_px: float = 0.5             # centre detection noise, 1-sigma
    boresight_error_rad: float = 0.004  # per-axis, per-episode camera alignment error (half-width)
    dropout_prob: float = 0.02
    alt_max_m: float = 8.0            # the flight computer uses the marker between these heights (PLND_ALT_MAX/MIN)
    alt_min_m: float = 0.3
    mount_down_m: float = 0.0

    @property
    def focal_px(self) -> float:
        return 0.5 * self.resolution_px / math.tan(math.radians(self.fov_deg) / 2.0)

    @property
    def max_range_m(self) -> float:
        return self.marker_size_m * self.focal_px / self.min_pixels


def marker_angles(q: Tensor, position: Tensor, marker_world: Tensor, mount_down_m: float):
    """Camera-frame (FRD) angular offsets of the marker, atan2(x, z) and atan2(y, z), and the depth z."""
    from tvc_env.common.frames import isaac_to_frd
    mount = torch.zeros_like(position)
    mount[:, 2] = -mount_down_m                      # Isaac body axes: the camera sits below the root
    camera = position + rotate_vector(normalize(q), mount)
    r = isaac_to_frd(rotate_vector(inverse(normalize(q)), marker_world - camera))
    depth = r[:, 2]
    safe = depth.clamp(min=1e-3)
    return torch.stack((torch.atan2(r[:, 0], safe), torch.atan2(r[:, 1], safe)), dim=-1), depth


class PadMarker:
    """Detector output as ArduPilot's LANDING_TARGET carries it: the marker's angular offsets in the camera."""

    def __init__(self, n: int, device, dt: float, params: MarkerParams, generator=None):
        self.n, self.device, self.p, self._gen = n, torch.device(device), params, generator
        self.sampler = _Sampler(dt, params.rate_hz, params.latency_s)
        self.marker_world = torch.zeros(n, 3, device=self.device)
        self.present = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._boresight = torch.zeros(n, 2, device=self.device)
        self.last_value = torch.zeros(n, 2, device=self.device)
        self.last_valid = torch.zeros(n, dtype=torch.bool, device=self.device)

    def reset(self, ids: Tensor, marker_world: Tensor | None) -> None:
        self._boresight[ids] = _uniform(self._gen, self.device, len(ids), 2) * self.p.boresight_error_rad
        if marker_world is None:
            self.present[ids] = False
        else:
            self.marker_world[ids] = marker_world[ids].to(self.device)
            self.present[ids] = True
        self.sampler.reset()

    def step(self, q: Tensor, position: Tensor) -> list[Measurement]:
        if self.sampler.due():
            p = self.p
            angles, depth = marker_angles(q, position, self.marker_world, p.mount_down_m)
            sigma = math.atan(p.noise_px / p.focal_px)
            measured = angles + self._boresight + torch.randn(self.n, 2, device=self.device, generator=self._gen) * sigma
            half = math.radians(p.fov_deg) / 2.0
            valid = (self.present & (depth >= p.min_range_m) & (depth <= p.max_range_m)
                     & (angles.abs().max(dim=-1).values <= half))
            if p.dropout_prob > 0.0:
                valid = valid & (torch.rand(self.n, device=self.device, generator=self._gen) >= p.dropout_prob)
            self.sampler.push(Measurement("marker", measured, valid, self.sampler.age))
        out = self.sampler.deliver()
        for m in out:
            self.last_value, self.last_valid = m.value, m.valid
        return out
