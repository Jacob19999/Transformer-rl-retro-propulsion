"""Unit tests for EnvironmentModel (assembly). Stage 10."""

from __future__ import annotations

import numpy as np

from simulation.environment.environment_model import EnvironmentModel


def test_sample_at_state_returns_expected_keys_and_shapes() -> None:
    cfg = {
        "environment": {
            "atmosphere": {
                "T_base": 288.15,
                "T_lapse": -0.0065,
                "P_base": 101325.0,
                "rho_ref": 1.225,
                "randomize_T": 0.0,
                "randomize_P": 0.0,
            },
            "wind": {
                "dt": 0.005,
                "V_ref": 10.0,
                "mean_vector_range_lo": [0.0, 0.0, 0.0],
                "mean_vector_range_hi": [0.0, 0.0, 0.0],
                "turbulence_intensity": 0.0,
                "gust_prob": 0.0,
                "gust_magnitude_range": [0.0, 0.0],
                "episode_duration": 15.0,
            },
        }
    }

    env = EnvironmentModel(cfg)
    env.reset(123)
    out = env.sample_at_state(t=0.5, p=np.array([1.0, 2.0, -3.0]))

    assert set(out.keys()) == {"wind", "rho", "T", "P"}
    assert isinstance(out["T"], float)
    assert isinstance(out["P"], float)
    assert isinstance(out["rho"], float)
    assert np.asarray(out["wind"]).shape == (3,)


def test_ned_altitude_conversion_matches_atmosphere_query() -> None:
    """Altitude above ground computed as h = -p[2] (NED)."""
    cfg = {
        "environment": {
            "atmosphere": {
                "T_base": 288.15,
                "T_lapse": -0.0065,
                "P_base": 101325.0,
                "rho_ref": 1.225,
                "randomize_T": 0.0,
                "randomize_P": 0.0,
            },
            "wind": {
                "dt": 0.005,
                "V_ref": 10.0,
                "mean_vector_range_lo": [0.0, 0.0, 0.0],
                "mean_vector_range_hi": [0.0, 0.0, 0.0],
                "turbulence_intensity": 0.0,
                "gust_prob": 0.0,
                "episode_duration": 15.0,
            },
        }
    }

    env = EnvironmentModel(cfg)
    env.reset(0)

    # p[2] = -10 m => h = 10 m
    p = np.array([0.0, 0.0, -10.0])
    out = env.sample_at_state(t=0.0, p=p)

    T_exp, P_exp, rho_exp = env.atmosphere.get_conditions(10.0)
    np.testing.assert_allclose(out["T"], T_exp)
    np.testing.assert_allclose(out["P"], P_exp)
    np.testing.assert_allclose(out["rho"], rho_exp)


def test_seeded_determinism_over_sequence() -> None:
    cfg = {
        "environment": {
            "atmosphere": {
                "randomize_T": 0.0,
                "randomize_P": 0.0,
            },
            "wind": {
                "dt": 0.005,
                "V_ref": 10.0,
                "mean_vector_range_lo": [-2.0, -2.0, -0.5],
                "mean_vector_range_hi": [2.0, 2.0, 0.5],
                "turbulence_intensity": 0.3,
                "gust_prob": 0.0,
                "episode_duration": 15.0,
            },
        }
    }

    e1 = EnvironmentModel(cfg)
    e2 = EnvironmentModel(cfg)
    e1.reset(42)
    e2.reset(42)

    times = np.linspace(0.0, 1.0, 25)
    positions = [np.array([0.0, 0.0, -h]) for h in np.linspace(1.0, 10.0, 25)]

    for t, p in zip(times, positions, strict=True):
        o1 = e1.sample_at_state(float(t), p)
        o2 = e2.sample_at_state(float(t), p)
        np.testing.assert_allclose(o1["wind"], o2["wind"])
        np.testing.assert_allclose(o1["T"], o2["T"])
        np.testing.assert_allclose(o1["P"], o2["P"])
        np.testing.assert_allclose(o1["rho"], o2["rho"])

