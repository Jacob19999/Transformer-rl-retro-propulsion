"""
Observation pipeline for EDFLandingEnv (Stage 13).

Implements:
- Stage 13.1: 20-dim observation construction from true state
- Stage 13.2: per-component Gaussian sensor noise (configurable std)
- Stage 13.3: wind estimate via exponential moving average (EMA)

Observation layout (OBS_DIM = 20):
    0..2   e_p_body   : target position error in body frame (3,)
    3..5   v_b        : body-frame velocity (3,)
    6..8   g_body     : gravity direction expressed in body frame (3,)
    9..11  omega      : body angular velocity (3,)
    12     twr        : thrust-to-weight ratio (scalar)
    13..15 wind_ema   : EMA estimate of wind in body frame (3,)
    16     h_agl      : altitude above ground (scalar, m)
    17     speed      : |v_b| (scalar, m/s)
    18     ang_speed  : |omega| (scalar, rad/s)
    19     time_frac  : t / max_time clamped to [0,1] (scalar)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


OBS_DIM = 20


def _as_vec(x: object, n: int, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size != n:
        raise ValueError(f"{name} must have shape ({n},), got {arr.shape}.")
    return arr


def _expand_group(value: object, n: int, *, name: str) -> np.ndarray:
    """Expand a scalar or length-n sequence into a (n,) float array."""
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.full((n,), float(arr[0]), dtype=float)
    if arr.size == n:
        return arr.astype(float)
    raise ValueError(f"{name} must be a scalar or length {n}, got {arr!r}.")


@dataclass(frozen=True, slots=True)
class ObservationConfig:
    """Configuration for observation generation."""

    wind_ema_alpha: float = 0.05
    noise_std: np.ndarray = field(default_factory=lambda: np.zeros((OBS_DIM,), dtype=float))

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "ObservationConfig":
        cfg = dict(config or {})
        wind_ema_alpha = float(cfg.get("wind_ema_alpha", 0.05))
        if not (0.0 <= wind_ema_alpha <= 1.0):
            raise ValueError("wind_ema_alpha must be in [0, 1].")

        noise = cfg.get("noise_std", 0.0)
        noise_std = _parse_noise_std(noise)
        return cls(wind_ema_alpha=wind_ema_alpha, noise_std=noise_std)


def _parse_noise_std(noise: object) -> np.ndarray:
    """Parse noise std configuration into a (20,) vector.

    Supported formats:
    - scalar: broadcast to all 20 dims
    - list/array length 20
    - dict of groups:
        {
          "e_p_body": scalar|len3,
          "v_b": scalar|len3,
          "g_body": scalar|len3,
          "omega": scalar|len3,
          "twr": scalar,
          "wind_ema": scalar|len3,
          "scalars": scalar|len4  # [h_agl, speed, ang_speed, time_frac]
        }
    """
    if isinstance(noise, Mapping):
        e_p = _expand_group(noise.get("e_p_body", 0.0), 3, name="noise_std.e_p_body")
        v_b = _expand_group(noise.get("v_b", 0.0), 3, name="noise_std.v_b")
        g_b = _expand_group(noise.get("g_body", 0.0), 3, name="noise_std.g_body")
        omg = _expand_group(noise.get("omega", 0.0), 3, name="noise_std.omega")
        twr = _expand_group(noise.get("twr", 0.0), 1, name="noise_std.twr")
        wema = _expand_group(noise.get("wind_ema", 0.0), 3, name="noise_std.wind_ema")
        sca = _expand_group(noise.get("scalars", 0.0), 4, name="noise_std.scalars")
        out = np.concatenate([e_p, v_b, g_b, omg, twr, wema, sca]).astype(float)
        if out.size != OBS_DIM:
            raise RuntimeError(f"noise_std parse bug: expected {OBS_DIM}, got {out.size}")
        return out

    arr = np.asarray(noise, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.full((OBS_DIM,), float(arr[0]), dtype=float)
    if arr.size == OBS_DIM:
        return arr.astype(float)
    raise ValueError(f"noise_std must be scalar, length {OBS_DIM}, or dict; got {arr!r}.")


def compute_true_observation(
    *,
    p: np.ndarray,
    v_b: np.ndarray,
    R_body_to_inertial: np.ndarray,
    omega: np.ndarray,
    T: float,
    mass: float,
    g: float,
    p_target: np.ndarray,
    v_wind_ned: np.ndarray,
    wind_ema_body: np.ndarray,
    t: float,
    max_time: float,
) -> np.ndarray:
    """Compute noise-free 20-dim observation."""
    p = _as_vec(p, 3, name="p")
    v_b = _as_vec(v_b, 3, name="v_b")
    omega = _as_vec(omega, 3, name="omega")
    p_target = _as_vec(p_target, 3, name="p_target")
    v_wind_ned = _as_vec(v_wind_ned, 3, name="v_wind_ned")
    wind_ema_body = _as_vec(wind_ema_body, 3, name="wind_ema_body")

    R = np.asarray(R_body_to_inertial, dtype=float).reshape(3, 3)

    # Target offset in body frame
    e_p_body = R.T @ (p_target - p)

    # Gravity direction in body frame (NED gravity direction = +z).
    g_body = R.T @ np.array([0.0, 0.0, 1.0], dtype=float)

    # Scalars
    h_agl = max(float(-p[2]), 0.0)
    speed = float(np.linalg.norm(v_b))
    ang_speed = float(np.linalg.norm(omega))
    twr = float(T / (mass * g + 1e-9))
    time_frac = float(np.clip(float(t) / max(float(max_time), 1e-9), 0.0, 1.0))

    obs = np.concatenate(
        [
            e_p_body.astype(float),
            v_b.astype(float),
            g_body.astype(float),
            omega.astype(float),
            np.array([twr], dtype=float),
            wind_ema_body.astype(float),
            np.array([h_agl, speed, ang_speed, time_frac], dtype=float),
        ]
    ).astype(float)
    if obs.size != OBS_DIM:
        raise RuntimeError(f"Expected obs dim {OBS_DIM}, got {obs.size}.")
    return obs


class ObservationPipeline:
    """Stateful observation pipeline (wind EMA + noise injection)."""

    def __init__(self, config: ObservationConfig) -> None:
        self.config = config
        self.wind_ema_body = np.zeros(3, dtype=float)
        self._rng: np.random.Generator = np.random.default_rng()

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self.wind_ema_body[...] = 0.0

    def get_obs(
        self,
        *,
        p: np.ndarray,
        v_b: np.ndarray,
        R_body_to_inertial: np.ndarray,
        omega: np.ndarray,
        T: float,
        mass: float,
        g: float,
        p_target: np.ndarray,
        v_wind_ned: np.ndarray,
        t: float,
        max_time: float,
    ) -> np.ndarray:
        R = np.asarray(R_body_to_inertial, dtype=float).reshape(3, 3)
        v_wind_body = R.T @ _as_vec(v_wind_ned, 3, name="v_wind_ned")

        a = float(self.config.wind_ema_alpha)
        self.wind_ema_body = (1.0 - a) * self.wind_ema_body + a * v_wind_body

        obs = compute_true_observation(
            p=p,
            v_b=v_b,
            R_body_to_inertial=R,
            omega=omega,
            T=float(T),
            mass=float(mass),
            g=float(g),
            p_target=p_target,
            v_wind_ned=v_wind_ned,
            wind_ema_body=self.wind_ema_body,
            t=float(t),
            max_time=float(max_time),
        )

        std = np.asarray(self.config.noise_std, dtype=float).reshape(OBS_DIM)
        if np.any(std > 0.0):
            obs = obs + self._rng.normal(loc=0.0, scale=std, size=(OBS_DIM,))

        return obs.astype(np.float32)


__all__ = ["OBS_DIM", "ObservationConfig", "ObservationPipeline", "compute_true_observation"]

