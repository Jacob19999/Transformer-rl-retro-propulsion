"""Unit tests for observation pipeline (Stage 13)."""

from __future__ import annotations

import numpy as np

from simulation.training.observation import (
    OBS_DIM,
    ObservationConfig,
    ObservationPipeline,
    compute_true_observation,
)


def test_compute_true_observation_layout_and_values() -> None:
    p = np.array([1.0, 2.0, -3.0], dtype=float)  # NED (z down)
    p_target = np.zeros(3, dtype=float)
    R = np.eye(3, dtype=float)  # body == inertial
    v_b = np.array([0.1, -0.2, 0.3], dtype=float)
    omega = np.array([0.4, 0.0, -0.3], dtype=float)
    mass = 2.0
    g = 9.81
    T = mass * g
    v_wind_ned = np.zeros(3, dtype=float)
    wind_ema = np.zeros(3, dtype=float)
    t = 0.5
    max_time = 1.0

    obs = compute_true_observation(
        p=p,
        v_b=v_b,
        R_body_to_inertial=R,
        omega=omega,
        T=T,
        mass=mass,
        g=g,
        p_target=p_target,
        v_wind_ned=v_wind_ned,
        wind_ema_body=wind_ema,
        t=t,
        max_time=max_time,
    )
    assert obs.shape == (OBS_DIM,)

    # Layout spot-checks
    np.testing.assert_allclose(obs[0:3], np.array([-1.0, -2.0, 3.0], dtype=float))  # e_p_body
    np.testing.assert_allclose(obs[3:6], v_b)  # v_b
    np.testing.assert_allclose(obs[6:9], np.array([0.0, 0.0, 1.0], dtype=float))  # g_body
    np.testing.assert_allclose(obs[9:12], omega)  # omega
    np.testing.assert_allclose(obs[12], 1.0, rtol=1e-6, atol=1e-6)  # twr
    np.testing.assert_allclose(obs[13:16], np.zeros(3, dtype=float))  # wind_ema
    np.testing.assert_allclose(obs[16], 3.0)  # h_agl
    np.testing.assert_allclose(obs[17], float(np.linalg.norm(v_b)))  # speed
    np.testing.assert_allclose(obs[18], float(np.linalg.norm(omega)))  # ang_speed
    np.testing.assert_allclose(obs[19], 0.5)  # time_frac


def test_wind_ema_update() -> None:
    cfg = ObservationConfig.from_config({"wind_ema_alpha": 0.05, "noise_std": 0.0})
    pipe = ObservationPipeline(cfg)
    pipe.reset(np.random.default_rng(0))

    p = np.zeros(3, dtype=float)
    v_b = np.zeros(3, dtype=float)
    R = np.eye(3, dtype=float)
    omega = np.zeros(3, dtype=float)
    T = 1.0
    mass = 1.0
    g = 9.81
    p_target = np.zeros(3, dtype=float)
    v_wind = np.array([10.0, 0.0, 0.0], dtype=float)

    obs1 = pipe.get_obs(
        p=p,
        v_b=v_b,
        R_body_to_inertial=R,
        omega=omega,
        T=T,
        mass=mass,
        g=g,
        p_target=p_target,
        v_wind_ned=v_wind,
        t=0.0,
        max_time=10.0,
    ).astype(float)
    np.testing.assert_allclose(obs1[13], 0.5, atol=1e-12)  # 0.05 * 10

    obs2 = pipe.get_obs(
        p=p,
        v_b=v_b,
        R_body_to_inertial=R,
        omega=omega,
        T=T,
        mass=mass,
        g=g,
        p_target=p_target,
        v_wind_ned=v_wind,
        t=0.0,
        max_time=10.0,
    ).astype(float)
    np.testing.assert_allclose(obs2[13], 0.975, atol=1e-12)  # 0.95*0.5 + 0.5


def test_noise_std_approximately_matches_configuration() -> None:
    # Use a single nonzero std on one component to make the test robust.
    sigma = 0.25
    noise_std = np.zeros(OBS_DIM, dtype=float)
    noise_std[0] = sigma

    cfg = ObservationConfig.from_config({"wind_ema_alpha": 0.0, "noise_std": noise_std})
    pipe = ObservationPipeline(cfg)
    pipe.reset(np.random.default_rng(123))

    p = np.zeros(3, dtype=float)
    v_b = np.zeros(3, dtype=float)
    R = np.eye(3, dtype=float)
    omega = np.zeros(3, dtype=float)
    T = 1.0
    mass = 1.0
    g = 9.81
    p_target = np.zeros(3, dtype=float)
    v_wind = np.zeros(3, dtype=float)

    N = 5000
    samples = np.empty((N,), dtype=float)
    for i in range(N):
        obs = pipe.get_obs(
            p=p,
            v_b=v_b,
            R_body_to_inertial=R,
            omega=omega,
            T=T,
            mass=mass,
            g=g,
            p_target=p_target,
            v_wind_ned=v_wind,
            t=0.0,
            max_time=10.0,
        )
        samples[i] = float(obs[0])

    empirical = float(np.std(samples, ddof=1))
    assert empirical == empirical  # not NaN
    assert abs(empirical - sigma) < 0.03

