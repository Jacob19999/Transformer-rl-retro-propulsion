"""
Allan-variance characterisation of a static IMU log, in the parameterisation of
tvc_env.dynamics.imu_model.

Model fitted to the overlapping Allan variance of one axis:

    sigma_A^2(tau) = N^2 / tau                 white noise (angle random walk)
                   + s^2 * g(tau / tau_c)      first-order Gauss-Markov bias, std s
                   + K^2 * tau / 3             random-walk bias (rate random walk)

    g(u) = (2 / u^2) * (u - 3/2 + 2 e^-u - e^-2u / 2)

The model is linear in (N^2, s^2, K^2) for a fixed tau_c, so tau_c is scanned on a log
grid and the amplitudes are solved as a non-negative least squares problem in relative
error. Outputs are converted to the profile conventions:

    noise_density      = sqrt(2) * N          one-sided datasheet ASD (unit/sqrt(Hz))
    bias_instability   = 0.6174 * s           peak Allan deviation of the bias (the model's "floor")
    bias_correlation_time_s = tau_c
    rate_random_walk   = K                    (unit/sqrt(s))

Requires a static, temperature-settled log of at least a few hours; the fit is only
as good as the longest averaging time it covers.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

GM_PEAK_ALLAN_RATIO = 0.6174


def overlapping_allan_deviation(x: np.ndarray, rate_hz: float, taus: np.ndarray | None = None):
    """Overlapping Allan deviation of a rate signal ``x`` sampled at ``rate_hz``.

    Returns (taus, adev) with tau restricted to cluster sizes m <= (N - 1) / 2.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    dt = 1.0 / rate_hz
    theta = np.concatenate(([0.0], np.cumsum(x) * dt))
    if taus is None:
        top = max(int((n - 1) // 9), 2)
        clusters = np.unique(np.round(np.logspace(0, math.log10(top), 60)).astype(int))
    else:
        clusters = np.unique(np.maximum(np.round(np.asarray(taus) * rate_hz).astype(int), 1))
    out_tau, out_adev = [], []
    for m in clusters:
        if 2 * m >= theta.size:
            continue
        d = theta[2 * m:] - 2.0 * theta[m:-m] + theta[:-2 * m]
        variance = np.mean(d * d) / (2.0 * (m * dt) ** 2)
        out_tau.append(m * dt)
        out_adev.append(math.sqrt(variance))
    return np.array(out_tau), np.array(out_adev)


def gauss_markov_shape(u: np.ndarray) -> np.ndarray:
    """g(u): Allan variance of a unit-variance Gauss-Markov process versus u = tau / tau_c."""
    u = np.asarray(u, dtype=np.float64)
    series = 2.0 * (u / 3.0 - u ** 2 / 4.0 + 7.0 * u ** 3 / 60.0)
    # u - 3/2 + 2 e^-u - e^-2u / 2, rewritten with expm1 so the constants cancel exactly.
    exact = (2.0 / np.maximum(u, 1e-300) ** 2) * (u + 2.0 * np.expm1(-u) - 0.5 * np.expm1(-2.0 * u))
    return np.where(u < 1e-2, series, exact)


def _nnls3(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Non-negative least squares for a handful of columns: try every active set."""
    best, best_cost = np.zeros(design.shape[1]), float(np.sum(target ** 2))
    for size in range(1, design.shape[1] + 1):
        for cols in itertools.combinations(range(design.shape[1]), size):
            sol, *_ = np.linalg.lstsq(design[:, cols], target, rcond=None)
            if np.any(sol < 0):
                continue
            cost = float(np.sum((design[:, cols] @ sol - target) ** 2))
            if cost < best_cost:
                best, best_cost = np.zeros(design.shape[1]), cost
                best[list(cols)] = sol
    return best


def fit_allan(taus: np.ndarray, adev: np.ndarray, tau_c_grid: np.ndarray | None = None) -> dict:
    """Fit (N, s, tau_c, K) to an Allan deviation curve; see the module docstring."""
    taus = np.asarray(taus, dtype=np.float64)
    var = np.asarray(adev, dtype=np.float64) ** 2
    if tau_c_grid is None:
        tau_c_grid = np.logspace(math.log10(taus[0]) - 0.5, math.log10(taus[-1]) + 1.0, 80)
    weight = 1.0 / var                      # relative error: every decade counts equally
    best = None
    for tau_c in tau_c_grid:
        design = np.stack((1.0 / taus, gauss_markov_shape(taus / tau_c), taus / 3.0), axis=1)
        amp = _nnls3(design * weight[:, None], var * weight)
        cost = float(np.sum(((design @ amp - var) * weight) ** 2))
        if best is None or cost < best[0]:
            best = (cost, tau_c, amp)
    _, tau_c, (n2, s2, k2) = best
    sigma_bias = math.sqrt(s2)
    return dict(noise_coefficient=math.sqrt(n2), noise_density=math.sqrt(2.0 * n2),
                bias_sigma=sigma_bias, bias_instability=GM_PEAK_ALLAN_RATIO * sigma_bias,
                bias_correlation_time_s=float(tau_c), rate_random_walk=math.sqrt(k2),
                relative_rms_error=math.sqrt(best[0] / len(taus)))
