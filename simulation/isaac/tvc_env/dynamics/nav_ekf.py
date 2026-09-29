"""
EKF3-style navigation filter (batched torch, float64), modelled on ArduPilot's AP_NavEKF3 for a
vehicle with no GNSS and no compass: IMU strapdown prediction corrected by a downward rangefinder,
an optical-flow sensor and a barometer.

This is NOT the ArduPilot code. It follows its structure and default tuning:

* Error-state EKF, 16 states: attitude error (world-frame rotation vector), velocity, position,
  gyro bias, accelerometer bias, terrain height. The accelerometer does not aid tilt directly (unlike
  a complementary filter); tilt is observed only through the velocity and height measurements, which
  is why unaided drift and level-seeking are handled differently from the onboard-filter path.
* Covariance is predicted once ~10 ms of IMU data has accumulated, with EKF3's process-noise defaults
  (EK3_GYRO_P_NSE, EK3_ACC_P_NSE, EK3_GBIAS_P_NSE, EK3_ABIAS_P_NSE) and a terrain state that grows with
  horizontal speed (EK3_TERR_GRAD).
* Measurements arrive late. Each carries its age in substeps; the innovation is formed against the
  nominal state the filter had then (a stored history) and the Kalman correction is applied to the
  current state and covariance. EKF3 runs its filter behind a delayed "fusion horizon" and re-propagates
  instead; this is the usual cheaper approximation.
* Optical flow: EKF3's line-of-sight model. Predicted flow (v_s,y / r, -v_s,x / r) from the estimated
  velocity, attitude and slant range; measurement is the sensor flow plus the gyro rate (gyro bias
  removed); noise EK3_FLOW_M_NSE, gate EK3_FLOW_I_GATE. Slant range comes from the estimated height
  above the estimated terrain.
* Rangefinder: fuses the slant range against height minus terrain (EK3_RNG_M_NSE, EK3_RNG_I_GATE), so it
  constrains height and terrain together. With the small terrain prior this makes it act as the height
  source at low altitude, the effect EKF3's EK3_RNG_USE_HGT selects.
* Barometer: height (EK3_ALT_M_NSE, EK3_HGT_I_GATE).

Not modelled: magnetometer/GNSS/yaw aiding (yaw is gyro-only and unobservable), EKF3's multi-core lane
switching, flow-quality weighting, on-ground constant-position fusion, sensor lever-arm compensation
beyond the down-axis mount offset.

Frames: quaternions Isaac wxyz (body local axes into the Z-up world); IMU registers arrive in body FRD
and are converted at the boundary; the bias states are in Isaac body axes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from tvc_env.common.frames import frd_to_isaac, isaac_to_frd, isaac_velocity_to_frd
from tvc_env.common.quaternions import inverse, multiply, normalize, rotate_vector
from tvc_env.dynamics.nav_sensors import Measurement, MIN_COS_TILT, marker_angles

GRAVITY = 9.80665
N_STATES = 16
TH, V, P_, BG, BA, T = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15), 15
MIN_FLOW_RANGE = 0.1


@dataclass(frozen=True)
class EkfInit:
    tilt_std_deg: float = 0.5
    yaw_std_deg: float = 5.0
    yaw_error_deg: float = 1.0        # 1-sigma of the heading the filter starts from (no compass)
    vel_std: float = 0.1
    pos_std: float = 0.5
    gyro_bias_std_dps: float = 0.3
    accel_bias_std_mg: float = 5.0
    terrain_std: float = 0.3


@dataclass(frozen=True)
class EkfParams:
    # ArduPilot EKF3 defaults (Copter build) unless noted.
    gyro_p_nse: float = 0.015         # rad/s
    acc_p_nse: float = 0.35           # m/s^2
    gbias_p_nse: float = 1.0e-3       # rad/s per step-second
    abias_p_nse: float = 2.0e-2       # m/s^2 per step-second
    terr_grad: float = 0.1            # terrain gradient, m per m
    alt_m_nse: float = 3.0            # baro, m
    hgt_i_gate: float = 500.0         # percent of 1 sigma
    rng_m_nse: float = 0.5            # rangefinder, m
    rng_i_gate: float = 500.0
    flow_m_nse: float = 0.25          # rad/s
    flow_i_gate: float = 300.0
    cov_interval_s: float = 0.01
    marker_m_nse: float = 0.01        # pad-marker angle noise, rad (not an EK3_ parameter; ArduPilot's precision
                                      # landing runs its own target filter outside EKF3)
    marker_i_gate: float = 500.0
    init: EkfInit = field(default_factory=EkfInit)


def skew(v: Tensor) -> Tensor:
    z = torch.zeros_like(v[..., 0])
    return torch.stack((torch.stack((z, -v[..., 2], v[..., 1]), -1),
                        torch.stack((v[..., 2], z, -v[..., 0]), -1),
                        torch.stack((-v[..., 1], v[..., 0], z), -1)), -2)


def quat_to_matrix(q: Tensor) -> Tensor:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack((
        torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
        torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
        torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1)), -2)


def rotvec_to_quat(v: Tensor) -> Tensor:
    theta = v.norm(dim=-1, keepdim=True)
    scale = torch.where(theta > 1e-12, torch.sin(theta * 0.5) / theta.clamp(min=1e-12), torch.full_like(theta, 0.5))
    return normalize(torch.cat((torch.cos(theta * 0.5), v * scale), dim=-1))


class NavEkf:
    """One filter per environment, all advanced together."""

    def __init__(self, n: int, device, dt: float, params: EkfParams, mount_down_m: float, max_age_steps: int):
        self.n, self.device, self.dt, self.p = n, torch.device(device), float(dt), params
        self.mount = float(mount_down_m)
        d = dict(device=self.device, dtype=torch.float64)
        eye = torch.eye(4, **d)[0]
        self.q = eye.repeat(n, 1)
        self.v = torch.zeros(n, 3, **d)
        self.p_ = torch.zeros(n, 3, **d)
        self.bg = torch.zeros(n, 3, **d)         # Isaac body axes, rad/s
        self.ba = torch.zeros(n, 3, **d)         # Isaac body axes, m/s^2
        self.t = torch.zeros(n, **d)             # terrain height, world z
        self.P = torch.eye(N_STATES, **d).repeat(n, 1, 1) * 1e-2
        self._eye = torch.eye(N_STATES, **d)
        self._up = torch.tensor([0.0, 0.0, GRAVITY], **d)
        self._size = max_age_steps + 2
        self._hq = eye.repeat(self._size, n, 1)
        self._hv = torch.zeros(self._size, n, 3, **d)
        self._hp = torch.zeros(self._size, n, 3, **d)
        self._hw = torch.zeros(self._size, n, 3, **d)     # bias-corrected body rate, FRD
        self._ptr = 0
        self._dt_acc = 0.0
        self._f_last = torch.zeros(n, 3, **d)
        self.gyro_corrected = torch.zeros(n, 3, **d)      # FRD
        self.marker = torch.zeros(n, 3, **d)                # pad marker, world XYZ (known)
        kinds = ("range", "flow", "baro", "marker")
        self.accepted = {k: torch.zeros(n, dtype=torch.bool, device=self.device) for k in kinds}
        self.innovation = {k: torch.zeros(n, device=self.device, dtype=torch.float64) for k in kinds}

    # ---- lifecycle ----------------------------------------------------------

    def reset(self, ids: Tensor, q_true: Tensor, v_true: Tensor, p_true: Tensor, acc_frd: Tensor,
              yaw_error: Tensor) -> None:
        """Initial alignment. Tilt from the accelerometer at rest (so its bias and scale error are in
        it), heading from ``yaw_error`` about the true heading, velocity and position from the pad."""
        ini = self.p.init
        d = dict(device=self.device, dtype=torch.float64)
        k = len(ids)
        q_t = normalize(q_true[ids].to(**d))
        up_true_body = rotate_vector(inverse(q_t), self._up.expand(k, 3) / GRAVITY)
        f_i = frd_to_isaac(acc_frd[ids].to(**d))
        up_meas = f_i / f_i.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        axis = torch.linalg.cross(up_meas, up_true_body)
        s = axis.norm(dim=-1, keepdim=True)
        angle = torch.atan2(s, (up_meas * up_true_body).sum(-1, keepdim=True))
        rotvec = torch.where(s > 1e-12, axis / s.clamp(min=1e-12) * angle, torch.zeros_like(axis))
        q_est = normalize(multiply(q_t, rotvec_to_quat(rotvec)))
        zero = torch.zeros_like(yaw_error.to(**d))
        q_est = normalize(multiply(rotvec_to_quat(torch.stack((zero, zero, yaw_error.to(**d)), -1)), q_est))
        self.q[ids] = q_est
        self.v[ids] = v_true[ids].to(**d)
        self.p_[ids] = p_true[ids].to(**d)
        self.bg[ids] = 0.0
        self.ba[ids] = 0.0
        self.t[ids] = 0.0
        std = torch.zeros(k, N_STATES, **d)
        std[:, 0:2] = math.radians(ini.tilt_std_deg)
        std[:, 2] = math.radians(ini.yaw_std_deg)
        std[:, V] = ini.vel_std
        std[:, P_] = ini.pos_std
        std[:, BG] = math.radians(ini.gyro_bias_std_dps)
        std[:, BA] = ini.accel_bias_std_mg * 1e-3 * GRAVITY
        std[:, T] = ini.terrain_std
        self.P[ids] = torch.diag_embed(std ** 2)
        for buf, value in ((self._hq, self.q), (self._hv, self.v), (self._hp, self.p_)):
            buf[:, ids] = value[ids].unsqueeze(0)
        self._hw[:, ids] = 0.0
        self.gyro_corrected[ids] = 0.0

    # ---- prediction ---------------------------------------------------------

    def predict(self, gyro_frd: Tensor, acc_frd: Tensor) -> None:
        """One physics substep of strapdown propagation from the (delayed) IMU registers."""
        d = dict(device=self.device, dtype=torch.float64)
        dt = self.dt
        w_i = frd_to_isaac(gyro_frd.to(**d)) - self.bg
        f_i = frd_to_isaac(acc_frd.to(**d)) - self.ba
        self.q = normalize(multiply(self.q, rotvec_to_quat(w_i * dt)))
        R = quat_to_matrix(self.q)
        accel = torch.einsum("nij,nj->ni", R, f_i) - self._up
        self.p_ = self.p_ + self.v * dt + 0.5 * accel * dt * dt
        self.v = self.v + accel * dt
        self._f_last = f_i
        self.gyro_corrected = isaac_to_frd(w_i)
        self._ptr = (self._ptr + 1) % self._size
        self._hq[self._ptr], self._hv[self._ptr], self._hp[self._ptr] = self.q, self.v, self.p_
        self._hw[self._ptr] = self.gyro_corrected
        self._dt_acc += dt
        if self._dt_acc >= self.p.cov_interval_s - 1e-9:
            self._predict_covariance(self._dt_acc, R)
            self._dt_acc = 0.0

    def _predict_covariance(self, dt: float, R: Tensor) -> None:
        n = self.n
        F = torch.zeros(n, N_STATES, N_STATES, device=self.device, dtype=torch.float64)
        F[:, TH, BG] = -R
        F[:, V, TH] = -skew(torch.einsum("nij,nj->ni", R, self._f_last))
        F[:, V, BA] = -R
        F[:, P_, V] = torch.eye(3, device=self.device, dtype=torch.float64)
        Phi = self._eye + F * dt + 0.5 * (F @ F) * dt * dt
        q = self.p
        var = torch.zeros(n, N_STATES, device=self.device, dtype=torch.float64)
        var[:, TH] = (q.gyro_p_nse * dt) ** 2
        var[:, V] = (q.acc_p_nse * dt) ** 2
        var[:, BG] = (q.gbias_p_nse * dt) ** 2
        var[:, BA] = (q.abias_p_nse * dt) ** 2
        speed = self.v[:, :2].norm(dim=-1)
        var[:, T] = (q.terr_grad * speed * dt) ** 2 + (1e-3 * dt) ** 2
        self.P = Phi @ self.P @ Phi.transpose(-1, -2) + torch.diag_embed(var)
        self._condition()

    def _condition(self) -> None:
        self.P = 0.5 * (self.P + self.P.transpose(-1, -2))
        diag = torch.diagonal(self.P, dim1=-2, dim2=-1)
        bad = diag < 0
        if bool(bad.any()):
            self.P = self.P + torch.diag_embed((-diag).clamp(min=0.0) + 1e-12)

    # ---- measurement plumbing -----------------------------------------------

    def _at_age(self, age: int):
        i = (self._ptr - age) % self._size
        return self._hq[i], self._hv[i], self._hp[i], self._hw[i]

    def _terrain_at_age(self) -> Tensor:
        return self.t          # the terrain state moves slowly; the current value stands in for the delayed one

    def _perturb(self, i: int, eps: float, q, v, p, t):
        """State perturbed by eps along error-state component i (world-left attitude convention)."""
        if i < 3:
            e = torch.zeros(self.n, 3, device=self.device, dtype=torch.float64)
            e[:, i] = eps
            return normalize(multiply(rotvec_to_quat(e), q)), v, p, t
        if i < 6:
            v = v.clone()
            v[:, i - 3] += eps
            return q, v, p, t
        if i < 9:
            p = p.clone()
            p[:, i - 6] += eps
            return q, v, p, t
        if i == 15:
            return q, v, p, t + eps
        return q, v, p, t

    def _jacobian(self, h, q, v, p, t, eps: float = 1e-6):
        base = h(q, v, p, t)
        cols = []
        for i in range(N_STATES):
            if 9 <= i < 15:
                cols.append(torch.zeros_like(base))
                continue
            cols.append((h(*self._perturb(i, eps, q, v, p, t)) - base) / eps)
        return base, torch.stack(cols, dim=-1)            # (n, [m,] 16)

    def _apply(self, dx: Tensor) -> None:
        self.q = normalize(multiply(rotvec_to_quat(dx[:, TH]), self.q))
        self.v = self.v + dx[:, V]
        self.p_ = self.p_ + dx[:, P_]
        self.bg = self.bg + dx[:, BG]
        self.ba = self.ba + dx[:, BA]
        self.t = self.t + dx[:, T]

    def _scalar_update(self, y: Tensor, H: Tensor, r_var: float, gate: float, valid: Tensor) -> Tensor:
        """Sequential scalar Kalman update with EKF3-style innovation gating; returns the accepted mask."""
        PHt = torch.einsum("nij,nj->ni", self.P, H)
        S = (H * PHt).sum(-1) + r_var
        ratio = y * y / (max(0.01 * gate, 1.0) ** 2 * S.clamp(min=1e-12))
        ok = valid & (ratio < 1.0) & (S > 0)
        K = PHt / S.clamp(min=1e-12).unsqueeze(-1) * ok.unsqueeze(-1)
        self._apply(K * y.unsqueeze(-1))
        IKH = self._eye - K.unsqueeze(-1) * H.unsqueeze(-2)
        self.P = IKH @ self.P @ IKH.transpose(-1, -2) + r_var * K.unsqueeze(-1) * K.unsqueeze(-2)
        self._condition()
        return ok

    # ---- measurement models ---------------------------------------------------

    def _range_pred(self, q, p, t):
        cos_t = (1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)).clamp(min=MIN_COS_TILT)
        return (p[:, 2] - t) / cos_t - self.mount

    def fuse_range(self, m: Measurement) -> None:
        q, v, p, _ = self._at_age(m.age)
        z = m.value.to(torch.float64)
        pred, H = self._jacobian(lambda q_, v_, p_, t_: self._range_pred(q_, p_, t_), q, v, p, self._terrain_at_age())
        ok = self._scalar_update(z - pred, H, self.p.rng_m_nse ** 2, self.p.rng_i_gate, m.valid)
        self.accepted["range"], self.innovation["range"] = ok, z - pred

    def fuse_baro(self, m: Measurement) -> None:
        q, v, p, _ = self._at_age(m.age)
        z = m.value.to(torch.float64)
        pred, H = self._jacobian(lambda q_, v_, p_, t_: p_[:, 2], q, v, p, self._terrain_at_age())
        ok = self._scalar_update(z - pred, H, self.p.alt_m_nse ** 2, self.p.hgt_i_gate, m.valid)
        self.accepted["baro"], self.innovation["baro"] = ok, z - pred

    def _flow_pred(self, q, v, p, t, w_frd):
        v_body = isaac_velocity_to_frd(rotate_vector(inverse(normalize(q)), v))
        mount = torch.zeros_like(v_body)
        mount[:, 2] = self.mount
        v_s = v_body + torch.linalg.cross(w_frd, mount)
        r = self._range_pred(q, p, t).clamp(min=MIN_FLOW_RANGE)
        return torch.stack((v_s[:, 1] / r, -v_s[:, 0] / r), dim=-1)

    def fuse_flow(self, m: Measurement) -> None:
        q, v, p, w = self._at_age(m.age)
        z = m.value.to(torch.float64) + w[:, :2]              # add the gyro rate back: sensor flow contains rotation
        pred, H = self._jacobian(lambda q_, v_, p_, t_: self._flow_pred(q_, v_, p_, t_, w), q, v, p, self._terrain_at_age())
        y = z - pred
        r_var = self.p.flow_m_nse ** 2
        ok0 = self._scalar_update(y[:, 0], H[:, 0], r_var, self.p.flow_i_gate, m.valid)
        ok1 = self._scalar_update(y[:, 1], H[:, 1], r_var, self.p.flow_i_gate, m.valid)
        self.accepted["flow"], self.innovation["flow"] = ok0 & ok1, y.norm(dim=-1)

    def fuse_marker(self, m: Measurement) -> None:
        """Pad-marker angular offsets (ArduPilot LANDING_TARGET style): a position fix relative to the known pad."""
        q, v, p, _ = self._at_age(m.age)
        z = m.value.to(torch.float64)
        marker = self.marker
        pred, H = self._jacobian(lambda q_, v_, p_, t_: marker_angles(q_, p_, marker, self.mount)[0],
                                 q, v, p, self._terrain_at_age())
        y = z - pred
        r_var = self.p.marker_m_nse ** 2
        ok0 = self._scalar_update(y[:, 0], H[:, 0], r_var, self.p.marker_i_gate, m.valid)
        ok1 = self._scalar_update(y[:, 1], H[:, 1], r_var, self.p.marker_i_gate, m.valid)
        self.accepted["marker"], self.innovation["marker"] = ok0 & ok1, y.norm(dim=-1)

    # ---- outputs --------------------------------------------------------------

    def sigma(self):
        diag = torch.diagonal(self.P, dim1=-2, dim2=-1).clamp(min=0.0).sqrt()
        return diag[:, P_], diag[:, V], diag[:, TH]
