"""
Sensor fusion for the simulated flight computer: IMU + downward rangefinder + optical flow + barometer,
fused by an EKF3-style filter (tvc_env.dynamics.nav_ekf).

Configuration lives under ``disturbances.sensor_noise.imu.fusion`` and is normally a profile,
``configs/sensors/fusion_<name>.yaml`` (e.g. ``fusion_tfmini_mtf01p``), with inline keys overriding it:

    imu:
      profile: wtgahrs1
      fusion: {enabled: true, profile: tfmini_mtf01p}

The filter consumes the IMU chain's *raw* registers (gyro and accelerometer, with their errors, delay
and quantization). The IMU's own onboard attitude is not used: like ArduPilot, the flight computer
estimates attitude itself. What reaches the controller (attitude, body rate, position, velocity,
height) is the filter's estimate; physics truth is only used to generate sensor readings and to score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import yaml
from torch import Tensor

from tvc_env.dynamics.imu_model import PROFILE_DIR, _PROFILE_NAME, _merge, _num, _section
from tvc_env.dynamics.nav_ekf import EkfInit, EkfParams, NavEkf
from tvc_env.dynamics.nav_sensors import (
    BaroParams, Barometer, FlowParams, MarkerParams, OpticalFlow, PadMarker, RangefinderParams, Rangefinder,
    cos_tilt, marker_angles, slant_range)

_FUSION_KEYS = {"enabled", "profile", "mount_down_m", "ekf", "rangefinder", "flow", "baro", "marker"}


def _keys(cls) -> set[str]:
    return {f.name for f in fields(cls)}


def resolve_fusion_config(cfg: dict) -> dict:
    """Expand ``profile: <name>`` from configs/sensors/fusion_<name>.yaml; inline keys override it."""
    cfg = dict(cfg or {})
    profile = cfg.pop("profile", None)
    if profile is None:
        return cfg
    if not isinstance(profile, str) or not _PROFILE_NAME.match(profile):
        raise ValueError(f"fusion.profile must match [a-z0-9_]+, got {profile!r}")
    path = PROFILE_DIR / f"fusion_{profile}.yaml"
    if not path.is_file():
        raise ValueError(f"Unknown fusion profile {profile!r}: {path} does not exist")
    base = (yaml.safe_load(path.read_text()) or {}).get("fusion")
    if not isinstance(base, dict):
        raise ValueError(f"{path} must contain a top-level 'fusion' mapping")
    return _merge(base, cfg)


def _build(cls, section: dict, label: str, bounds: dict | None = None):
    """Instantiate a parameter dataclass from a mapping, defaulting to the dataclass default."""
    default = cls()
    kwargs = {}
    for f in fields(cls):
        if f.name == "init":
            continue
        low, high = (bounds or {}).get(f.name, (0.0, math.inf))
        kwargs[f.name] = _num(section, f.name, getattr(default, f.name), label, low, high)
    return cls(**kwargs)


@dataclass(frozen=True)
class FusionParams:
    enabled: bool
    mount_down_m: float
    ekf: EkfParams
    rangefinder: RangefinderParams
    flow: FlowParams
    baro: BaroParams
    marker: MarkerParams
    marker_enabled: bool

    @classmethod
    def from_config(cls, cfg: dict | None) -> "FusionParams":
        cfg = resolve_fusion_config(cfg)
        unknown = set(cfg) - _FUSION_KEYS
        if unknown:
            raise ValueError(f"imu.fusion: unknown keys {sorted(unknown)}")
        enabled = cfg.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError(f"imu.fusion.enabled must be true or false, got {enabled!r}")
        mount = _num(cfg, "mount_down_m", 0.0, "fusion", 0.0, 5.0)

        e = _section(cfg, "ekf", _keys(EkfParams))
        i = _section(e, "init", _keys(EkfInit))
        ekf_init = _build(EkfInit, i, "fusion.ekf.init", {"yaw_error_deg": (0.0, 180.0), "tilt_std_deg": (0.0, 90.0),
                                                          "yaw_std_deg": (0.0, 180.0)})
        ekf = _build(EkfParams, e, "fusion.ekf", {"cov_interval_s": (1e-3, 0.1), "terr_grad": (0.0, 10.0)})
        ekf = EkfParams(**{**{f.name: getattr(ekf, f.name) for f in fields(ekf) if f.name != "init"}, "init": ekf_init})

        rf = _section(cfg, "rangefinder", _keys(RangefinderParams))
        fl = _section(cfg, "flow", _keys(FlowParams))
        ba = _section(cfg, "baro", _keys(BaroParams))
        mk = _section(cfg, "marker", _keys(MarkerParams) | {"enabled"})
        marker_enabled = mk.get("enabled", False)
        if not isinstance(marker_enabled, bool):
            raise ValueError(f"imu.fusion.marker.enabled must be true or false, got {marker_enabled!r}")
        rf_p = _build(RangefinderParams, rf, "fusion.rangefinder",
                      {"rate_hz": (0.1, 10000.0), "latency_s": (0.0, 0.5), "max_tilt_deg": (0.0, 89.0),
                       "dropout_prob": (0.0, 1.0)})
        fl_p = _build(FlowParams, fl, "fusion.flow", {"rate_hz": (0.1, 10000.0), "latency_s": (0.0, 0.5),
                                                      "max_tilt_deg": (0.0, 89.0), "dropout_prob": (0.0, 1.0),
                                                      "scale_error_pct": (0.0, 50.0)})
        ba_p = _build(BaroParams, ba, "fusion.baro", {"rate_hz": (0.1, 10000.0), "latency_s": (0.0, 0.5),
                                                      "drift_tau_s": (1e-3, 1e7)})
        mk_p = _build(MarkerParams, {k: v for k, v in mk.items() if k != "enabled"}, "fusion.marker",
                      {"rate_hz": (0.1, 10000.0), "latency_s": (0.0, 0.5), "fov_deg": (1.0, 170.0),
                       "resolution_px": (16.0, 1e5), "marker_size_m": (0.01, 10.0), "min_pixels": (1.0, 1e4),
                       "dropout_prob": (0.0, 1.0)})
        # The mount offset is one physical fact shared by both downward sensors and the filter.
        rf_p = RangefinderParams(**{**{f.name: getattr(rf_p, f.name) for f in fields(rf_p)}, "mount_down_m": mount})
        fl_p = FlowParams(**{**{f.name: getattr(fl_p, f.name) for f in fields(fl_p)}, "mount_down_m": mount})
        mk_p = MarkerParams(**{**{f.name: getattr(mk_p, f.name) for f in fields(mk_p)}, "mount_down_m": mount})
        return cls(enabled, mount, ekf, rf_p, fl_p, ba_p, mk_p, marker_enabled)

    @property
    def max_age_s(self) -> float:
        return max(self.rangefinder.latency_s, self.flow.latency_s, self.baro.latency_s,
                   self.marker.latency_s if self.marker_enabled else 0.0)


class NavFusion:
    """Sensors + filter for ``num_envs`` vehicles. Call ``reset`` per power cycle and ``step`` per physics
    substep, after the IMU model has stepped."""

    def __init__(self, num_envs: int, device, physics_dt: float, params: FusionParams | dict, imu,
                 generator: torch.Generator | None = None):
        if not isinstance(params, FusionParams):
            params = FusionParams.from_config(params)
        self.n, self.device, self.dt, self.params, self.imu = num_envs, torch.device(device), float(physics_dt), params, imu
        self._gen = generator
        self.rangefinder = Rangefinder(num_envs, device, physics_dt, params.rangefinder, generator)
        self.flow = OpticalFlow(num_envs, device, physics_dt, params.flow, generator)
        self.baro = Barometer(num_envs, device, physics_dt, params.baro, generator)
        self.marker = PadMarker(num_envs, device, physics_dt, params.marker, generator)
        max_age = int(round(params.max_age_s / physics_dt)) + 1
        self.ekf = NavEkf(num_envs, device, physics_dt, params.ekf, params.mount_down_m, max_age)
        self._initialised = False
        self._truth = None          # latest true (q, position, velocity, body rate), for the telemetry only

    # ---- estimate ---------------------------------------------------------------

    @property
    def quaternion_wxyz(self) -> Tensor:
        return self.ekf.q.float()

    @property
    def position(self) -> Tensor:
        return self.ekf.p_.float()

    @property
    def velocity_world(self) -> Tensor:
        return self.ekf.v.float()

    @property
    def gyro_frd(self) -> Tensor:
        """Body rate the controller holds: the gyro register with the filter's bias estimate removed."""
        return self.ekf.gyro_corrected.float()

    @property
    def initialised(self) -> bool:
        return self._initialised

    # ---- lifecycle ----------------------------------------------------------------

    def reset(self, env_ids: Tensor, quaternion_wxyz: Tensor, linear_vel_world: Tensor, position_world: Tensor,
              marker_world: Tensor | None = None) -> None:
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if ids.numel() == 0:
            return
        self.rangefinder.reset(ids)
        self.flow.reset(ids)
        self.baro.reset(ids)
        self.marker.reset(ids, marker_world if self.params.marker_enabled else None)
        if marker_world is not None:
            self.ekf.marker[ids] = marker_world[ids].to(self.ekf.marker.dtype)
        yaw_err = torch.randn(len(ids), device=self.device, generator=self._gen) * math.radians(
            self.params.ekf.init.yaw_error_deg)
        self.ekf.reset(ids, quaternion_wxyz, linear_vel_world, position_world, self.imu.accel_frd, yaw_err)
        self._truth = (quaternion_wxyz.clone(), position_world.clone(), linear_vel_world.clone(),
                       torch.zeros(self.n, 3, device=self.device))
        self._initialised = True

    def step(self, quaternion_wxyz: Tensor, linear_vel_world: Tensor, position_world: Tensor) -> None:
        """Advance one physics substep from the true state (used only to generate the sensor readings)."""
        if not self._initialised:
            raise RuntimeError("NavFusion.step before reset")
        self.ekf.predict(self.imu.gyro_frd, self.imu.accel_out_frd)
        rate = self.imu.true_rate_frd
        self._truth = (quaternion_wxyz, position_world, linear_vel_world, rate)
        for m in self.rangefinder.step(quaternion_wxyz, position_world):
            self.ekf.fuse_range(m)
        for m in self.flow.step(quaternion_wxyz, position_world, linear_vel_world, rate):
            self.ekf.fuse_flow(m)
        for m in self.baro.step(position_world):
            self.ekf.fuse_baro(m)
        for m in self.marker.step(quaternion_wxyz, position_world):
            if self.params.marker_enabled:
                # The flight computer only trusts the marker inside its altitude window, judged from its own
                # height estimate (ArduPilot PLND_ALT_MAX / PLND_ALT_MIN).
                height = (self.ekf.p_[:, 2] - self.ekf.t).float()
                window = (height <= self.params.marker.alt_max_m) & (height >= self.params.marker.alt_min_m)
                m.valid = m.valid & window
                self.ekf.fuse_marker(m)

    # ---- telemetry -----------------------------------------------------------------

    def record(self) -> dict:
        """Environment 0's sensors and filter health, for the mission telemetry."""
        pos_sigma, vel_sigma, att_sigma = self.ekf.sigma()
        f = lambda t: [round(float(x), 5) for x in t[0].reshape(-1)]
        q, position, velocity, rate = self._truth
        mount = self.params.mount_down_m
        mk = self.params.marker
        # What the downward sensors actually face (physics truth), so the UI can draw the camera view.
        flow_truth, _ = self.flow.truth_flow(q, position, velocity, rate)
        angles, depth = marker_angles(q, position, self.marker.marker_world, mount)
        height_est = (self.ekf.p_[:, 2] - self.ekf.t).float()
        in_fov = bool(self.marker.present[0] & (depth[0] > 0) & (angles[0].abs().max() <= math.radians(mk.fov_deg) / 2.0))
        truth = dict(
            height_truth_m=round(float(position[0, 2]), 4), height_est_m=round(float(height_est[0]), 4),
            range_truth_m=round(float(slant_range(position[:, 2], cos_tilt(q), mount)[0]), 4),
            flow_truth_rad_s=f(flow_truth),
            marker_enabled=bool(self.params.marker_enabled), marker_world_m=f(self.marker.marker_world),
            marker_truth_rad=f(angles), marker_depth_m=round(float(depth[0]), 4), marker_in_fov=in_fov,
            marker_in_window=bool(mk.alt_min_m <= float(height_est[0]) <= mk.alt_max_m),
        )
        return dict(truth, 
            range_m=f(self.rangefinder.last_value), range_valid=bool(self.rangefinder.last_valid[0]),
            flow_rad_s=f(self.flow.last_value), flow_valid=bool(self.flow.last_valid[0]),
            baro_m=f(self.baro.last_value),
            marker_rad=f(self.marker.last_value), marker_valid=bool(self.marker.last_valid[0]),
            position_sigma_m=f(pos_sigma), velocity_sigma_m_s=f(vel_sigma),
            attitude_sigma_deg=[round(math.degrees(x), 4) for x in att_sigma[0].tolist()],
            gyro_bias_dps=[round(math.degrees(x), 5) for x in self.ekf.bg[0].tolist()],
            accel_bias_mg=[round(x / 9.80665 * 1000.0, 4) for x in self.ekf.ba[0].tolist()],
            terrain_m=round(float(self.ekf.t[0]), 4),
            accepted=dict(range=bool(self.ekf.accepted["range"][0]), flow=bool(self.ekf.accepted["flow"][0]),
                          baro=bool(self.ekf.accepted["baro"][0]), marker=bool(self.ekf.accepted["marker"][0])),
        )


def nav_fusion_from_config(num_envs: int, device, physics_dt: float, sensor_noise: dict | None, imu,
                           generator: torch.Generator | None = None) -> NavFusion | None:
    """Build the fusion stack when ``sensor_noise.imu.fusion`` enables it, else None.

    The profile is expanded into ``sensor_noise['imu']['fusion']`` in place so a run record shows the
    parameters actually used. Requires the IMU model (its raw registers are the filter's inertial input).
    """
    if imu is None or not sensor_noise:
        return None
    fusion = (sensor_noise.get("imu") or {}).get("fusion")
    if not fusion:
        return None
    resolved = resolve_fusion_config(fusion)
    sensor_noise["imu"]["fusion"] = resolved
    params = FusionParams.from_config(resolved)
    if not params.enabled:
        return None
    return NavFusion(num_envs, device, physics_dt, params, imu, generator)
