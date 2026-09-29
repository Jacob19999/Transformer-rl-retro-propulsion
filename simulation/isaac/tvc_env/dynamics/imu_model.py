"""
Strapdown IMU + onboard attitude-filter measurement model (batched torch).

The environment used to corrupt attitude and body rates with independent
white noise redrawn at every control step. A real fused-attitude IMU (the
WITMOTION WTGAHRS1 on the drone) instead delivers:

    truth -> sensor errors -> digital low-pass -> sample&hold -> onboard
    attitude filter -> transport delay -> flight computer

This module simulates that chain per environment:

* Gyro and accelerometer error chain (Nitsch's IMU-Simulator, BSD-3, was the
  reference for the structure; the code here is an independent torch
  implementation): turn-on bias, scale-factor / misalignment matrix, white
  noise (angle/velocity random walk), Gauss-Markov bias instability, random-walk
  bias, gyro g-sensitivity, saturation, quantization. Errors are redrawn per
  episode, so an episode is one power cycle of one randomly drawn unit.
* Temperature drift: a per-axis temperature coefficient of offset (TCO) times a
  warm-up profile (start-temperature offset from the calibration point plus a
  first-order self-heating rise), drawn per episode.
* Truth angular rate (``rate_truth``): "reported" feeds the physics engine's angular velocity to the
  gyro; "pose" derives it from the change of the true attitude over each substep. They agree in free
  flight, but PhysX changes the pose during ground contact (depenetration) without a matching
  angular velocity, so a gyro integrating the reported rate never sees that rotation and the
  attitude estimate carries a fixed error of a fraction of a degree from the first pad contact on.
* Optional unaided strapdown navigation (``nav.enabled``): the flight computer integrates the
  delayed accelerometer register, rotated by the delayed onboard attitude estimate, minus gravity,
  into velocity and position. Sensor bias, attitude error (tilt leaks g*eps into horizontal
  acceleration), noise and latency therefore drift the solution exactly as they would on the bench;
  nothing aids it (no baro, optical flow or GNSS yet).
* On-sensor digital low-pass (WTGAHRS1 manual 2.4.9: 20 Hz default), output
  rate (2.4.3: 10 Hz default, 200 Hz max) with hold-last-value, and a
  transport latency.
* A Mahony complementary attitude filter fed by the *measured* gyro and
  accelerometer, so attitude error is slow and correlated (gyro bias leaks into
  yaw, thrust acceleration tilts the gravity reference) instead of white.
  Yaw is either gyro-only (6-axis algorithm) or corrected by a magnetometer
  heading with its own error (9-axis algorithm).

Frames follow tvc_env.common.frames: quaternions are Isaac (w,x,y,z) rotating
body local axes into the Z-up world; sensor vectors are body FRD. Frame
conversion goes through tvc_env.common.frames only.

Not modelled: lever-arm (centripetal/tangential) acceleration at the mount
point, EDF vibration, sensitivity-versus-temperature, magnetometer disturbance
correlated with motor current. Parameters in configs/sensors/imu_<name>.yaml
(wtgahrs1, bno085, vn110e) carry a provenance comment; values marked ESTIMATE
or PLACEHOLDER must be replaced with a bench Allan-variance fit
(tools/imu_allan_variance.py).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from torch import Tensor

from tvc_env.common.frames import frd_to_isaac, isaac_to_frd
from tvc_env.common.quaternions import inverse, multiply, normalize, rotate_vector, to_euler

STANDARD_GRAVITY = 9.80665  # m/s^2

# Peak of the Allan deviation of a first-order Gauss-Markov process as a fraction of
# its standard deviation: sqrt(0.3812) = 0.6174, reached at T ~ 1.89 tau (analytic, and
# confirmed by simulating an AR(1) process). Spec sheets quote the Allan floor as "bias
# instability", so sigma_GM = floor / 0.6174. (The reference IMU-Simulator quotes the IEEE
# flicker coefficient B = floor / 0.664 and 0.4365 inside its own formula; that is a
# different parameterisation and must not be mixed with this one.)
GM_PEAK_ALLAN_RATIO = 0.6174

PROFILE_DIR = Path(__file__).resolve().parents[2] / "configs" / "sensors"
_PROFILE_NAME = re.compile(r"^[a-z0-9_]+$")

_IMU_KEYS = {"enabled", "profile", "sample_rate_hz", "bandwidth_hz", "latency_s", "gyro", "accel", "attitude",
             "thermal", "nav", "rate_truth", "fusion"}
_NAV_KEYS = {"enabled", "initial_position_std_m", "initial_velocity_std_m_s"}
_THERMAL_KEYS = {"start_spread_k", "self_heating_k", "warmup_tau_s"}
_GYRO_KEYS = {"range_dps", "resolution_dps", "noise_density_dps_rthz", "bias_instability_dps",
              "bias_correlation_time_s", "rate_random_walk_dps_rt_s", "turn_on_bias_dps",
              "scale_factor_pct", "misalignment_mrad", "g_sensitivity_dps_per_g", "tco_dps_per_k"}
_ACCEL_KEYS = {"range_g", "resolution_g", "noise_density_ug_rthz", "bias_instability_mg",
               "bias_correlation_time_s", "random_walk_ug_rt_s", "turn_on_bias_mg",
               "scale_factor_pct", "misalignment_mrad", "tco_mg_per_k"}
_ATTITUDE_KEYS = {"kp", "ki", "integral_limit_dps", "accel_gate", "initial_tilt_error_deg", "yaw"}
_YAW_KEYS = {"mode", "gain", "initial_error_deg", "mag_turn_on_deg", "mag_sigma_deg",
             "mag_correlation_time_s"}


# --------------------------------------------------------------------------- config

def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def resolve_imu_config(cfg: dict) -> dict:
    """Expand ``profile: <name>`` from configs/sensors/imu_<name>.yaml; inline keys override it."""
    cfg = dict(cfg or {})
    profile = cfg.pop("profile", None)
    if profile is None:
        return cfg
    if not isinstance(profile, str) or not _PROFILE_NAME.match(profile):
        raise ValueError(f"imu.profile must match [a-z0-9_]+, got {profile!r}")
    path = PROFILE_DIR / f"imu_{profile}.yaml"
    if not path.is_file():
        raise ValueError(f"Unknown IMU profile {profile!r}: {path} does not exist")
    base = (yaml.safe_load(path.read_text()) or {}).get("imu")
    if not isinstance(base, dict):
        raise ValueError(f"{path} must contain a top-level 'imu' mapping")
    return _merge(base, cfg)


def _section(cfg: dict, name: str, allowed: set[str]) -> dict:
    value = cfg.get(name) or {}
    if not isinstance(value, dict):
        raise ValueError(f"imu.{name} must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"imu.{name}: unknown keys {sorted(unknown)}")
    return value


def _num(section: dict, key: str, default: float, label: str, low: float = 0.0, high: float = math.inf) -> float:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or not low <= value <= high:
        path = f"imu.{label}.{key}" if label else f"imu.{key}"
        raise ValueError(f"{path} must be a finite number in [{low}, {high}], got {value!r}")
    return float(value)


@dataclass(frozen=True)
class ChannelParams:
    """One three-axis sensor error model, all quantities in SI (rad/s or m/s^2)."""
    saturation: float      # symmetric measurement range; inf disables
    resolution: float      # quantization step; 0 disables
    noise_density: float   # white noise, one-sided datasheet ASD, unit/sqrt(Hz)
    gm_sigma: float        # std of the Gauss-Markov bias
    gm_tau: float          # its correlation time, s
    walk: float            # random-walk bias, unit/sqrt(s)
    turn_on: float         # uniform half-width of the per-episode constant bias
    scale: float           # uniform half-width of per-axis scale error (fraction)
    misalignment: float    # uniform half-width of cross-axis coupling (rad)
    g_sensitivity: float   # uniform half-width of unit per (m/s^2) of specific force
    tco: float = 0.0       # uniform half-width of the offset temperature coefficient, unit per kelvin


@dataclass(frozen=True)
class AttitudeParams:
    kp: float
    ki: float
    integral_limit: float       # rad/s
    accel_gate: float           # accept accelerometer when | |f| - g | <= gate * g
    initial_tilt_error: float   # rad, 1-sigma
    yaw_mode: str               # "gyro" or "magnetometer"
    yaw_gain: float             # 1/s, heading correction (magnetometer mode)
    initial_yaw_error: float    # rad, 1-sigma (gyro mode)
    mag_turn_on: float          # rad, uniform half-width
    mag_sigma: float            # rad, Gauss-Markov std
    mag_tau: float              # s


@dataclass(frozen=True)
class ThermalParams:
    """Temperature relative to the factory calibration point, per episode (one power cycle)."""
    start_spread: float   # K, uniform half-width of the power-up offset from the calibration point
    self_heating: float   # K, first-order rise after power-up
    tau: float            # s, self-heating time constant


@dataclass(frozen=True)
class NavParams:
    """Unaided strapdown navigation driven by the delayed accelerometer and attitude registers."""
    enabled: bool
    initial_position_std: float   # m, 1-sigma error of the position the solution starts from
    initial_velocity_std: float   # m/s, 1-sigma error of the velocity it starts from


@dataclass(frozen=True)
class ImuParams:
    sample_rate_hz: float
    bandwidth_hz: float
    latency_s: float
    gyro: ChannelParams
    accel: ChannelParams
    attitude: AttitudeParams
    thermal: ThermalParams = ThermalParams(0.0, 0.0, 300.0)
    nav: NavParams = NavParams(False, 0.0, 0.0)
    rate_truth: str = "reported"     # "reported" physics angular velocity or "pose" (from the attitude change)

    @classmethod
    def from_config(cls, cfg: dict) -> "ImuParams":
        cfg = resolve_imu_config(cfg)
        unknown = set(cfg) - _IMU_KEYS
        if unknown:
            raise ValueError(f"imu: unknown keys {sorted(unknown)}")
        rate = _num(cfg, "sample_rate_hz", 200.0, "", 0.1, 10000.0)
        bandwidth = _num(cfg, "bandwidth_hz", 0.0, "", 0.0, 5000.0)
        latency = _num(cfg, "latency_s", 0.0, "", 0.0, 0.5)

        g = _section(cfg, "gyro", _GYRO_KEYS)
        rad = math.radians
        gyro = ChannelParams(
            saturation=rad(_num(g, "range_dps", 0.0, "gyro")) or math.inf,
            resolution=rad(_num(g, "resolution_dps", 0.0, "gyro")),
            noise_density=rad(_num(g, "noise_density_dps_rthz", 0.0, "gyro")),
            gm_sigma=rad(_num(g, "bias_instability_dps", 0.0, "gyro")) / GM_PEAK_ALLAN_RATIO,
            gm_tau=_num(g, "bias_correlation_time_s", 100.0, "gyro", 1e-3, 1e7),
            walk=rad(_num(g, "rate_random_walk_dps_rt_s", 0.0, "gyro")),
            turn_on=rad(_num(g, "turn_on_bias_dps", 0.0, "gyro")),
            scale=_num(g, "scale_factor_pct", 0.0, "gyro", 0.0, 50.0) / 100.0,
            misalignment=_num(g, "misalignment_mrad", 0.0, "gyro", 0.0, 500.0) * 1e-3,
            g_sensitivity=rad(_num(g, "g_sensitivity_dps_per_g", 0.0, "gyro")) / STANDARD_GRAVITY,
            tco=rad(_num(g, "tco_dps_per_k", 0.0, "gyro")),
        )
        a = _section(cfg, "accel", _ACCEL_KEYS)
        mg = STANDARD_GRAVITY * 1e-3
        ug = STANDARD_GRAVITY * 1e-6
        accel = ChannelParams(
            saturation=_num(a, "range_g", 0.0, "accel") * STANDARD_GRAVITY or math.inf,
            resolution=_num(a, "resolution_g", 0.0, "accel") * STANDARD_GRAVITY,
            noise_density=_num(a, "noise_density_ug_rthz", 0.0, "accel") * ug,
            gm_sigma=_num(a, "bias_instability_mg", 0.0, "accel") * mg / GM_PEAK_ALLAN_RATIO,
            gm_tau=_num(a, "bias_correlation_time_s", 100.0, "accel", 1e-3, 1e7),
            walk=_num(a, "random_walk_ug_rt_s", 0.0, "accel") * ug,
            turn_on=_num(a, "turn_on_bias_mg", 0.0, "accel") * mg,
            scale=_num(a, "scale_factor_pct", 0.0, "accel", 0.0, 50.0) / 100.0,
            misalignment=_num(a, "misalignment_mrad", 0.0, "accel", 0.0, 500.0) * 1e-3,
            g_sensitivity=0.0,
            tco=_num(a, "tco_mg_per_k", 0.0, "accel") * mg,
        )
        t = _section(cfg, "attitude", _ATTITUDE_KEYS)
        y = _section(t, "yaw", _YAW_KEYS)
        mode = y.get("mode", "gyro")
        if mode not in ("gyro", "magnetometer"):
            raise ValueError(f"imu.attitude.yaw.mode must be 'gyro' or 'magnetometer', got {mode!r}")
        attitude = AttitudeParams(
            kp=_num(t, "kp", 1.0, "attitude", 0.0, 100.0),
            ki=_num(t, "ki", 0.02, "attitude", 0.0, 100.0),
            integral_limit=rad(_num(t, "integral_limit_dps", 5.0, "attitude", 0.0, 1000.0)),
            accel_gate=_num(t, "accel_gate", 0.3, "attitude", 0.0, 10.0),
            initial_tilt_error=rad(_num(t, "initial_tilt_error_deg", 0.0, "attitude", 0.0, 30.0)),
            yaw_mode=mode,
            yaw_gain=_num(y, "gain", 0.5, "attitude.yaw", 0.0, 100.0),
            initial_yaw_error=rad(_num(y, "initial_error_deg", 0.0, "attitude.yaw", 0.0, 180.0)),
            mag_turn_on=rad(_num(y, "mag_turn_on_deg", 0.0, "attitude.yaw", 0.0, 180.0)),
            mag_sigma=rad(_num(y, "mag_sigma_deg", 0.0, "attitude.yaw", 0.0, 180.0)),
            mag_tau=_num(y, "mag_correlation_time_s", 60.0, "attitude.yaw", 1e-3, 1e7),
        )
        th = _section(cfg, "thermal", _THERMAL_KEYS)
        thermal = ThermalParams(
            start_spread=_num(th, "start_spread_k", 0.0, "thermal", 0.0, 100.0),
            self_heating=_num(th, "self_heating_k", 0.0, "thermal", 0.0, 100.0),
            tau=_num(th, "warmup_tau_s", 300.0, "thermal", 1e-3, 1e7),
        )
        nv = _section(cfg, "nav", _NAV_KEYS)
        enabled = nv.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError(f"imu.nav.enabled must be true or false, got {enabled!r}")
        nav = NavParams(
            enabled=enabled,
            initial_position_std=_num(nv, "initial_position_std_m", 0.0, "nav", 0.0, 1000.0),
            initial_velocity_std=_num(nv, "initial_velocity_std_m_s", 0.0, "nav", 0.0, 100.0),
        )
        rate_truth = cfg.get("rate_truth", "reported")
        if rate_truth not in ("reported", "pose"):
            raise ValueError(f"imu.rate_truth must be 'reported' or 'pose', got {rate_truth!r}")
        return cls(rate, bandwidth, latency, gyro, accel, attitude, thermal, nav, rate_truth)


# --------------------------------------------------------------------------- sensor channel

class _Channel:
    """Error chain of one three-axis sensor for ``n`` environments."""

    def __init__(self, n: int, device, dt: float, params: ChannelParams, generator):
        self.n, self.device, self.dt, self.p, self._gen = n, device, dt, params, generator
        z = lambda *s: torch.zeros(*s, device=device)
        self.turn_on = z(n, 3)
        self.matrix = torch.eye(3, device=device).repeat(n, 1, 1)
        self.g_sens = z(n, 3)
        self.tco = z(n, 3)
        self.gm = z(n, 3)
        self.walk = z(n, 3)
        self.lp = z(n, 3)
        self.lp_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self._phi = math.exp(-dt / params.gm_tau)
        self._offdiag = 1.0 - torch.eye(3, device=device)

    def _randn(self, *shape):
        return torch.randn(*shape, device=self.device, generator=self._gen)

    def _uniform(self, *shape, half: float):
        return (torch.rand(*shape, device=self.device, generator=self._gen) * 2.0 - 1.0) * half

    def reset(self, ids: Tensor) -> None:
        p, k = self.p, len(ids)
        self.turn_on[ids] = self._uniform(k, 3, half=p.turn_on)
        matrix = torch.eye(3, device=self.device).repeat(k, 1, 1)
        matrix = matrix + self._uniform(k, 3, 3, half=p.misalignment) * self._offdiag
        matrix = matrix + torch.diag_embed(self._uniform(k, 3, half=p.scale))
        self.matrix[ids] = matrix
        self.g_sens[ids] = self._uniform(k, 3, half=p.g_sensitivity)
        self.tco[ids] = self._uniform(k, 3, half=p.tco)
        self.gm[ids] = self._randn(k, 3) * p.gm_sigma   # stationary draw: an in-run bias already exists
        self.walk[ids] = 0.0
        self.lp_valid[ids] = False

    def sample(self, truth: Tensor, specific_force: Tensor | None, ids: Tensor | None = None,
               temperature: Tensor | None = None) -> Tensor:
        """Corrupt ``truth`` (n,3) and advance the stochastic states; saturated, unfiltered.

        ``temperature`` (n,) is the offset from the calibration point in kelvin; it drives the TCO term."""
        p = self.p
        sel = slice(None) if ids is None else ids
        x = truth[sel]
        count = x.shape[0]
        y = torch.einsum("nij,nj->ni", self.matrix[sel], x) + self.turn_on[sel]
        if p.g_sensitivity > 0.0 and specific_force is not None:
            y = y + self.g_sens[sel] * specific_force[sel]
        if p.tco > 0.0 and temperature is not None:
            y = y + self.tco[sel] * temperature[sel].unsqueeze(-1)
        if p.gm_sigma > 0.0:
            innovation = math.sqrt(max(1.0 - self._phi ** 2, 0.0)) * p.gm_sigma
            self.gm[sel] = self._phi * self.gm[sel] + innovation * self._randn(count, 3)
            y = y + self.gm[sel]
        if p.walk > 0.0:
            self.walk[sel] = self.walk[sel] + p.walk * math.sqrt(self.dt) * self._randn(count, 3)
            y = y + self.walk[sel]
        if p.noise_density > 0.0:
            # noise_density is a datasheet one-sided amplitude spectral density (unit/sqrt(Hz)), so a
            # sample at period dt has variance density^2 / (2 dt). (Allan's white-noise coefficient N,
            # as in the reference repo, equals density / sqrt(2); do not mix the two conventions.)
            y = y + (p.noise_density / math.sqrt(2.0 * self.dt)) * self._randn(count, 3)
        if math.isfinite(p.saturation):
            y = y.clamp(-p.saturation, p.saturation)
        return y

    def lowpass(self, y: Tensor, alpha: float, ids: Tensor | None = None) -> Tensor:
        """First-order digital low-pass; a fresh env starts from its first sample."""
        sel = slice(None) if ids is None else ids
        valid = self.lp_valid[sel].unsqueeze(-1)
        blended = self.lp[sel] + alpha * (y - self.lp[sel])
        self.lp[sel] = torch.where(valid, blended, y)
        self.lp_valid[sel] = True
        return self.lp[sel].clone()   # a copy: callers keep it as a held register

    def quantize(self, y: Tensor) -> Tensor:
        if self.p.resolution > 0.0:
            y = torch.round(y / self.p.resolution) * self.p.resolution
        return y


def _rotvec_to_quat(v: Tensor) -> Tensor:
    theta = v.norm(dim=-1, keepdim=True)
    scale = torch.where(theta > 1e-9, torch.sin(theta * 0.5) / theta.clamp(min=1e-9), torch.full_like(theta, 0.5))
    return normalize(torch.cat((torch.cos(theta * 0.5), v * scale), dim=-1))


def _wrap(angle: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


# --------------------------------------------------------------------------- model

class ImuModel:
    """Per-environment IMU measurement chain. Call ``reset`` for new episodes and
    ``step`` once per physics substep; read ``gyro_frd`` / ``quaternion_wxyz``."""

    def __init__(self, num_envs: int, device, physics_dt: float, params: ImuParams | dict,
                 generator: torch.Generator | None = None):
        if not isinstance(params, ImuParams):
            params = ImuParams.from_config(params)
        if physics_dt <= 0.0:
            raise ValueError("physics_dt must be positive")
        self.n, self.device, self.dt, self.params = num_envs, torch.device(device), float(physics_dt), params
        self._gen = generator
        self._gyro = _Channel(num_envs, self.device, self.dt, params.gyro, generator)
        self._acc = _Channel(num_envs, self.device, self.dt, params.accel, generator)
        # A sensor cannot report faster than the physics ticks it is fed.
        self._rate = min(params.sample_rate_hz, 1.0 / self.dt)
        self._alpha = 1.0 if params.bandwidth_hz <= 0.0 else 1.0 - math.exp(-2.0 * math.pi * params.bandwidth_hz * self.dt)
        self._delay = int(round(params.latency_s / self.dt))
        self._thermal_active = params.gyro.tco > 0.0 or params.accel.tco > 0.0
        self._temp = torch.zeros(num_envs, device=self.device)         # K above the calibration point
        self._temp_target = torch.zeros(num_envs, device=self.device)
        self._temp_alpha = 1.0 - math.exp(-self.dt / params.thermal.tau)
        self._phase = 0.0
        self._since_tick = 0.0
        self._up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        z = lambda *s: torch.zeros(*s, device=self.device)
        self._v_prev = z(num_envs, 3)
        self._q_prev = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_envs, 1)
        self._gyro_reg = z(num_envs, 3)
        self._acc_reg = z(num_envs, 3)
        self._q_est = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_envs, 1)
        self._integral = z(num_envs, 3)
        self._mag_err = z(num_envs)
        self._buffer = z(self._delay + 1, num_envs, 10)     # gyro (3), attitude (4), accelerometer (3)
        self._ptr = 0
        self._out_gyro = z(num_envs, 3)
        self._out_q = self._q_est.clone()
        self._out_acc = z(num_envs, 3)
        self._true_rate = z(num_envs, 3)   # body rate the sensor actually experienced, FRD
        self._nav_p = z(num_envs, 3)      # world XYZ, m
        self._nav_v = z(num_envs, 3)      # world XYZ, m/s
        self._initialised = False

    # ---- public state -------------------------------------------------------

    @property
    def gyro_frd(self) -> Tensor:
        """Body rate the flight computer holds (rad/s, FRD): delayed, held, quantized."""
        return self._out_gyro

    @property
    def quaternion_wxyz(self) -> Tensor:
        """Attitude the flight computer holds (Isaac wxyz): onboard-filter output, delayed."""
        return self._out_q

    @property
    def accel_frd(self) -> Tensor:
        """Latest accelerometer register (m/s^2, FRD, undelayed) that feeds the attitude filter."""
        return self._acc_reg

    @property
    def accel_out_frd(self) -> Tensor:
        """Accelerometer register the flight computer holds (m/s^2, FRD): delayed, held, quantized."""
        return self._out_acc

    @property
    def true_rate_frd(self) -> Tensor:
        """Body rate the sensor physically experienced this substep (rad/s, FRD), before any sensor error."""
        return self._true_rate

    @property
    def nav_enabled(self) -> bool:
        return self.params.nav.enabled

    @property
    def nav_position(self) -> Tensor:
        """Integrated position (world XYZ, m). Only meaningful when ``nav.enabled``."""
        return self._nav_p

    @property
    def nav_velocity(self) -> Tensor:
        """Integrated velocity (world XYZ, m/s). Only meaningful when ``nav.enabled``."""
        return self._nav_v

    @property
    def temperature_k(self) -> Tensor:
        """Sensor temperature offset from its calibration point (K); zero unless a TCO is configured."""
        return self._temp

    @property
    def initialised(self) -> bool:
        return self._initialised

    # ---- helpers ------------------------------------------------------------

    def _randn(self, *shape):
        return torch.randn(*shape, device=self.device, generator=self._gen)

    def _specific_force_frd(self, q_true: Tensor, v_world: Tensor, ids: Tensor | None = None) -> Tensor:
        """Accelerometer input f = R^T (a - g) from the velocity difference over one substep."""
        if ids is None:
            accel_w = (v_world - self._v_prev) / self.dt
            self._v_prev = v_world.clone()
            q = q_true
        else:
            accel_w = torch.zeros_like(v_world[ids])   # unknown at reset: assume unaccelerated
            self._v_prev[ids] = v_world[ids]
            q = q_true[ids]
        f_world = accel_w + self._up * STANDARD_GRAVITY
        return isaac_to_frd(rotate_vector(inverse(q), f_world))

    def _pose_rate_frd(self, q: Tensor) -> Tensor:
        """Body rate (FRD) that reproduces this substep's change of the true attitude."""
        dq = multiply(inverse(self._q_prev), q)                     # body-frame rotation over the substep
        dq = torch.where(dq[:, :1] < 0.0, -dq, dq)
        vec = dq[:, 1:]
        s = vec.norm(dim=-1, keepdim=True)
        scale = torch.where(s > 1e-9, 2.0 * torch.atan2(s, dq[:, :1]) / s.clamp(min=1e-9),
                            2.0 / dq[:, :1].clamp(min=1e-9))
        self._q_prev = q.clone()
        return isaac_to_frd(vec * scale / self.dt)

    # ---- lifecycle ----------------------------------------------------------

    def reset(self, env_ids: Tensor, quaternion_wxyz: Tensor, linear_vel_world: Tensor,
              angular_vel_frd: Tensor, position_world: Tensor | None = None) -> None:
        """Power-cycle the sensors of ``env_ids``: redraw errors, restart the filter.

        The tensors are full-batch (num_envs, ...) state after the reset was applied.
        ``position_world`` is required when ``nav.enabled``: the navigation solution starts from it.
        """
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if ids.numel() == 0:
            return
        att = self.params.attitude
        k = ids.numel()
        self._gyro.reset(ids)
        self._acc.reset(ids)
        self._q_prev[ids] = quaternion_wxyz[ids]
        self._true_rate[ids] = angular_vel_frd[ids]
        th = self.params.thermal
        start = (torch.rand(k, device=self.device, generator=self._gen) * 2.0 - 1.0) * th.start_spread
        self._temp[ids] = start
        self._temp_target[ids] = start + th.self_heating
        f_frd = self._specific_force_frd(quaternion_wxyz, linear_vel_world, ids)
        full_f = torch.zeros(self.n, 3, device=self.device)
        full_f[ids] = f_frd
        gyro_y = self._gyro.sample(angular_vel_frd, full_f, ids, self._temp)
        acc_y = self._acc.sample(full_f, None, ids, self._temp)
        self._gyro_reg[ids] = self._gyro.quantize(self._gyro.lowpass(gyro_y, self._alpha, ids))
        self._acc_reg[ids] = self._acc.quantize(self._acc.lowpass(acc_y, self._alpha, ids))
        self._integral[ids] = 0.0
        # Magnetometer heading error: constant hard-iron-like offset + slow drift.
        self._mag_err[ids] = ((torch.rand(k, device=self.device, generator=self._gen) * 2 - 1) * att.mag_turn_on
                              + self._randn(k) * att.mag_sigma)
        yaw_err = self._mag_err[ids] if att.yaw_mode == "magnetometer" else self._randn(k) * att.initial_yaw_error
        tilt = self._randn(k, 2) * att.initial_tilt_error
        rotvec = torch.cat((tilt, yaw_err.unsqueeze(-1)), dim=-1)
        self._q_est[ids] = normalize(multiply(quaternion_wxyz[ids], _rotvec_to_quat(rotvec)))
        # Fill the delay line so the previous episode's samples never leak in.
        register = torch.cat((self._gyro_reg[ids], self._q_est[ids], self._acc_reg[ids]), dim=-1)
        self._buffer[:, ids] = register.unsqueeze(0)
        self._out_gyro[ids] = self._gyro_reg[ids]
        self._out_q[ids] = self._q_est[ids]
        self._out_acc[ids] = self._acc_reg[ids]
        nav = self.params.nav
        if nav.enabled:
            if position_world is None:
                raise ValueError("ImuModel.reset needs position_world when nav.enabled")
            self._nav_p[ids] = position_world[ids] + self._randn(k, 3) * nav.initial_position_std
            self._nav_v[ids] = linear_vel_world[ids] + self._randn(k, 3) * nav.initial_velocity_std
        self._initialised = True

    def step(self, quaternion_wxyz: Tensor, linear_vel_world: Tensor, angular_vel_frd: Tensor) -> None:
        """Advance the whole chain by one physics substep from the true state."""
        if not self._initialised:
            self.reset(torch.arange(self.n, device=self.device), quaternion_wxyz, linear_vel_world,
                       angular_vel_frd)
        if self.params.rate_truth == "pose":
            angular_vel_frd = self._pose_rate_frd(quaternion_wxyz)
        self._true_rate = angular_vel_frd
        f_frd = self._specific_force_frd(quaternion_wxyz, linear_vel_world)
        if self._thermal_active:
            self._temp = self._temp + self._temp_alpha * (self._temp_target - self._temp)
        gyro_lp = self._gyro.lowpass(self._gyro.sample(angular_vel_frd, f_frd, None, self._temp), self._alpha)
        acc_lp = self._acc.lowpass(self._acc.sample(f_frd, None, None, self._temp), self._alpha)
        self._since_tick += self.dt
        self._phase += self.dt * self._rate
        if self._phase >= 1.0 - 1e-9:
            self._phase = min(self._phase - 1.0, 1.0)
            self._gyro_reg = self._gyro.quantize(gyro_lp)
            self._acc_reg = self._acc.quantize(acc_lp)
            self._update_attitude(self._since_tick, quaternion_wxyz)
            self._since_tick = 0.0
        self._buffer[self._ptr] = torch.cat((self._gyro_reg, self._q_est, self._acc_reg), dim=-1)
        self._ptr = (self._ptr + 1) % (self._delay + 1)
        delayed = self._buffer[self._ptr]
        self._out_gyro = delayed[:, :3].clone()
        self._out_q = delayed[:, 3:7].clone()
        self._out_acc = delayed[:, 7:10].clone()
        if self.params.nav.enabled:
            self._integrate_navigation()

    # ---- strapdown navigation -----------------------------------------------

    def _integrate_navigation(self) -> None:
        """One substep of the flight computer's dead reckoning.

        The accelerometer and attitude registers arrive together (same latency), so the specific force
        is resolved with the attitude estimate that travelled with it, gravity is removed with the
        nominal constant, and the held values are integrated at the physics rate (zero-order hold, so
        it equals integrating at the sensor rate). Velocity is trapezoidal into position.
        """
        f_world = rotate_vector(self._out_q, frd_to_isaac(self._out_acc))
        accel = f_world - self._up * STANDARD_GRAVITY
        v_next = self._nav_v + accel * self.dt
        self._nav_p = self._nav_p + 0.5 * (self._nav_v + v_next) * self.dt
        self._nav_v = v_next

    # ---- onboard attitude filter --------------------------------------------

    def _update_attitude(self, dt: float, q_true: Tensor) -> None:
        """Mahony explicit complementary filter, run at the sensor output rate."""
        att = self.params.attitude
        gyro = frd_to_isaac(self._gyro_reg)
        acc = frd_to_isaac(self._acc_reg)
        q = self._q_est
        up_body = rotate_vector(inverse(q), self._up.expand_as(gyro))       # estimated "up" in body axes
        magnitude = acc.norm(dim=-1, keepdim=True)
        accept = ((magnitude - STANDARD_GRAVITY).abs() <= att.accel_gate * STANDARD_GRAVITY).to(gyro.dtype)
        error = torch.linalg.cross(acc / magnitude.clamp(min=1e-6), up_body) * accept
        correction = att.kp * error
        integral_error = error
        if att.yaw_mode == "magnetometer":
            # Heading measured by the magnetometer = true heading + its error state.
            decay = math.exp(-dt / att.mag_tau)
            drift = math.sqrt(max(1.0 - decay ** 2, 0.0)) * att.mag_sigma
            self._mag_err = decay * self._mag_err + drift * self._randn(self.n)
            heading_error = _wrap(to_euler(q_true)[2] + self._mag_err - to_euler(q)[2])
            weight = up_body[:, 2].clamp(min=0.0)    # heading is ill-defined as the body leaves upright
            heading_vec = (heading_error * weight).unsqueeze(-1) * up_body
            correction = correction + att.yaw_gain * heading_vec
            integral_error = integral_error + heading_vec
        self._integral = (self._integral + att.ki * integral_error * dt).clamp(-att.integral_limit,
                                                                              att.integral_limit)
        omega = gyro + correction + self._integral
        rate_quat = torch.cat((torch.zeros_like(omega[:, :1]), omega), dim=-1)
        self._q_est = normalize(q + 0.5 * dt * multiply(q, rate_quat))


def imu_model_from_config(num_envs: int, device, physics_dt: float, sensor_noise: dict | None,
                          generator: torch.Generator | None = None) -> ImuModel | None:
    """Build the IMU model when ``disturbances.sensor_noise`` enables it, else None.

    A ``profile`` reference is expanded into ``sensor_noise['imu']`` in place, so a
    run record that dumps the config shows the parameters actually used.
    """
    if not sensor_noise or not sensor_noise.get("enabled", False):
        return None
    imu = sensor_noise.get("imu")
    if not imu or not imu.get("enabled", True):
        return None
    resolved = resolve_imu_config(imu)
    sensor_noise["imu"] = resolved
    return ImuModel(num_envs, device, physics_dt, ImuParams.from_config(resolved), generator)
