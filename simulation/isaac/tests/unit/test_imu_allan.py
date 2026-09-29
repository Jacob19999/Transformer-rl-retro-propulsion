"""Allan-variance fit recovers the parameters of the imu_model generative model."""

import math

import numpy as np
import pytest

from tvc_env.dynamics.imu_allan import (
    GM_PEAK_ALLAN_RATIO, fit_allan, gauss_markov_shape, overlapping_allan_deviation)


def synthetic(n, rate, asd=0.0, gm_sigma=0.0, tau_c=100.0, walk=0.0, seed=0):
    """White (one-sided ASD) + Gauss-Markov bias + random-walk bias, as imu_model draws them."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / rate
    x = rng.standard_normal(n) * asd / math.sqrt(2.0 * dt)
    if gm_sigma > 0:
        phi = math.exp(-dt / tau_c)
        w = rng.standard_normal(n) * gm_sigma * math.sqrt(1 - phi ** 2)
        bias = np.empty(n)
        state = rng.standard_normal() * gm_sigma
        for i in range(n):
            state = phi * state + w[i]
            bias[i] = state
        x = x + bias
    if walk > 0:
        x = x + np.cumsum(rng.standard_normal(n) * walk * math.sqrt(dt))
    return x


def test_gauss_markov_shape_is_continuous_across_the_series_switch_and_peaks_at_the_known_ratio():
    u = np.array([0.0099, 0.0101])
    g = gauss_markov_shape(u)
    assert g[1] / g[0] == pytest.approx(0.0101 / 0.0099, rel=2e-3)          # ~ linear in u for small u
    direct = lambda v: 2.0 / v ** 2 * (v - 1.5 + 2.0 * math.exp(-v) - 0.5 * math.exp(-2.0 * v))
    assert gauss_markov_shape(np.array([1.0]))[0] == pytest.approx(direct(1.0), rel=1e-12)
    grid = np.linspace(0.05, 20.0, 40000)
    assert math.sqrt(gauss_markov_shape(grid).max()) == pytest.approx(GM_PEAK_ALLAN_RATIO, abs=1e-3)


def test_white_noise_allan_deviation_is_the_random_walk_coefficient_over_root_tau():
    rate, sigma = 100.0, 0.3
    taus, adev = overlapping_allan_deviation(synthetic(1_000_000, rate, asd=sigma * math.sqrt(2.0 / rate)), rate)
    n_coeff = sigma / math.sqrt(rate)                  # N = sigma_w * sqrt(dt)
    assert (adev * np.sqrt(taus))[:20].mean() == pytest.approx(n_coeff, rel=0.05)


def test_fit_recovers_white_noise_bias_and_random_walk():
    rate = 10.0
    truth = dict(asd=0.02, gm_sigma=0.01, tau_c=200.0, walk=2e-4)
    x = synthetic(1_500_000, rate, **truth, seed=3)
    taus, adev = overlapping_allan_deviation(x, rate)
    fit = fit_allan(taus, adev)
    assert fit["noise_density"] == pytest.approx(truth["asd"], rel=0.10)
    assert fit["bias_sigma"] == pytest.approx(truth["gm_sigma"], rel=0.30)
    assert fit["bias_correlation_time_s"] == pytest.approx(truth["tau_c"], rel=0.6)
    assert fit["rate_random_walk"] == pytest.approx(truth["walk"], rel=0.40)
    assert fit["bias_instability"] == pytest.approx(GM_PEAK_ALLAN_RATIO * fit["bias_sigma"])


def test_pure_gauss_markov_floor_matches_the_peak_of_its_own_allan_curve():
    rate = 10.0
    x = synthetic(1_500_000, rate, gm_sigma=0.05, tau_c=100.0, seed=8)
    taus, adev = overlapping_allan_deviation(x, rate)
    fit = fit_allan(taus, adev)
    assert adev.max() == pytest.approx(fit["bias_instability"], rel=0.15)
    assert fit["noise_density"] < 0.01


def test_characterise_reads_three_axes_from_a_csv(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "imu_allan_variance", Path(__file__).resolve().parents[2] / "tools" / "imu_allan_variance.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    rate = 50.0
    cols = np.stack([synthetic(200_000, rate, asd=0.05, seed=s) for s in (1, 2, 3)], axis=1)
    path = tmp_path / "log.csv"
    np.savetxt(path, cols, delimiter=",", header="gx,gy,gz", comments="")
    data = tool.load_columns(path, ["gx", "gy", "gz"])
    assert data.shape == (200_000, 3)
    fits = tool.characterise(data, rate, 1.0)
    assert all(f["noise_density"] == pytest.approx(0.05, rel=0.15) for f in fits)
