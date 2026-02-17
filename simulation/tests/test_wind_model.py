"""Unit tests for wind model: DrydenFilter + WindModel. Stage 9."""

import numpy as np

from simulation.environment.wind_model import DrydenFilter, WindModel


def test_seeded_reproducibility() -> None:
    """Same seed produces identical wind sequences."""
    cfg = {
        "dt": 0.005,
        "V_ref": 10.0,
        "mean_vector_range_lo": [-2.0, -2.0, -0.5],
        "mean_vector_range_hi": [2.0, 2.0, 0.5],
        "turbulence_intensity": 0.3,
        "gust_prob": 0.0,
        "episode_duration": 15.0,
    }
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    w1 = WindModel(cfg, rng=rng1)
    w2 = WindModel(cfg, rng=rng2)
    w1.reset()
    w2.reset()

    for _ in range(10):
        t = np.random.uniform(0, 5)
        h = np.random.uniform(1, 10)
        v1 = w1.sample(t, h)
        v2 = w2.sample(t, h)
        np.testing.assert_array_almost_equal(v1, v2, err_msg="Seeded reproducibility")


def test_zero_config_yields_zero_wind() -> None:
    """Zero mean, zero turbulence, zero gust → zero wind."""
    cfg = {
        "dt": 0.005,
        "V_ref": 10.0,
        "mean_vector_range_lo": [0.0, 0.0, 0.0],
        "mean_vector_range_hi": [0.0, 0.0, 0.0],
        "turbulence_intensity": 0.0,
        "gust_prob": 0.0,
        "gust_magnitude_range": [0.0, 0.0],
        "episode_duration": 15.0,
    }
    rng = np.random.default_rng(0)
    w = WindModel(cfg, rng=rng)
    w.reset()

    for t in [0.0, 1.0, 5.0, 10.0]:
        for h in [0.5, 5.0, 10.0]:
            v = w.sample(t, h)
            np.testing.assert_array_almost_equal(v, [0.0, 0.0, 0.0], err_msg=f"t={t}, h={h}")


def test_gust_timing_bounds() -> None:
    """Gust onset within [0.2*T_ep, 0.8*T_ep]; duration in [0.5, 2.0] s."""
    ep_duration = 15.0
    cfg = {
        "dt": 0.005,
        "V_ref": 10.0,
        "mean_vector_range_lo": [0.0, 0.0, 0.0],
        "mean_vector_range_hi": [0.0, 0.0, 0.0],
        "turbulence_intensity": 0.0,
        "gust_prob": 1.0,  # always gust for this test
        "gust_magnitude_range": [5.0, 5.0],
        "episode_duration": ep_duration,
    }
    onsets = []
    durations = []
    for seed in range(100):
        rng = np.random.default_rng(seed)
        w = WindModel(cfg, rng=rng)
        w.reset()
        if w.gust_active:
            onsets.append(w.gust_onset)
            durations.append(w.gust_duration)

    if onsets:
        assert all(0.2 * ep_duration <= o <= 0.8 * ep_duration for o in onsets)
    if durations:
        assert all(0.5 <= d <= 2.0 for d in durations)


def test_dryden_variance_approximately_intensity_squared() -> None:
    """Dryden turbulence variance scales with intensity²; non-zero when intensity > 0."""
    # Test that variance is positive and scales with intensity²
    def collect_var(sigma: float, n: int = 5000) -> np.ndarray:
        cfg = {"dt": 0.005, "V_ref": 10.0, "turbulence_intensity": sigma}
        df = DrydenFilter(dt=0.005, V_ref=10.0, config=cfg)
        rng = np.random.default_rng(42)
        samples = [df.step(5.0, rng.standard_normal(3)) for _ in range(n)]
        return np.var(samples, axis=0)

    var_03 = collect_var(0.3)
    var_06 = collect_var(0.6)

    # All components have positive variance when intensity > 0
    assert np.all(var_03 > 0) and np.all(var_06 > 0)

    # Doubling intensity should roughly quadruple variance (var ∝ sigma²)
    ratio = var_06 / (var_03 + 1e-12)
    assert np.all(ratio >= 2.0) and np.all(ratio <= 8.0), (
        f"Variance ratio {ratio} expected ~4 when doubling intensity"
    )


def test_dryden_reset_zeros_states() -> None:
    """DrydenFilter.reset() zeros all filter states."""
    cfg = {"turbulence_intensity": 0.5}
    df = DrydenFilter(dt=0.005, V_ref=10.0, config=cfg)

    # Advance filter
    for _ in range(5):
        df.step(5.0, np.random.standard_normal(3))

    df.reset()
    assert df.state_u == 0.0
    np.testing.assert_array_equal(df.state_v, [0.0, 0.0])
    np.testing.assert_array_equal(df.state_w, [0.0, 0.0])


def test_mean_wind_in_range() -> None:
    """Mean wind sampled within configured range."""
    lo = np.array([-5.0, -3.0, -1.0])
    hi = np.array([5.0, 3.0, 1.0])
    cfg = {
        "dt": 0.005,
        "V_ref": 10.0,
        "mean_vector_range_lo": lo.tolist(),
        "mean_vector_range_hi": hi.tolist(),
        "turbulence_intensity": 0.0,
        "gust_prob": 0.0,
        "episode_duration": 15.0,
    }
    for seed in range(50):
        rng = np.random.default_rng(seed)
        w = WindModel(cfg, rng=rng)
        w.reset()
        assert np.all(w.mean_wind >= lo - 1e-10) and np.all(w.mean_wind <= hi + 1e-10)


def test_sample_returns_shape_3() -> None:
    """WindModel.sample returns (3,) array."""
    cfg = {
        "dt": 0.005,
        "V_ref": 10.0,
        "mean_vector_range_lo": [0.0, 0.0, 0.0],
        "mean_vector_range_hi": [1.0, 1.0, 1.0],
        "turbulence_intensity": 0.1,
        "gust_prob": 0.0,
        "episode_duration": 15.0,
    }
    w = WindModel(cfg, rng=np.random.default_rng(0))
    w.reset()
    v = w.sample(1.0, 5.0)
    assert v.shape == (3,)
