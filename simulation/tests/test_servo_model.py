"""Unit tests for the servo model (Stage 6)."""

import numpy as np

from simulation.dynamics.servo_model import ServoModel, ServoModelConfig


def _make_servo_config(
    *,
    n_fins: int = 4,
    tau_servo: float = 0.04,
    tau_servo_range: tuple[float, float] = (0.04, 0.04),
    rate_max: float = 10.5,
    aero_load_derating: float = 0.0,
) -> ServoModelConfig:
    return ServoModelConfig(
        n_fins=n_fins,
        tau_servo=tau_servo,
        tau_servo_range=tau_servo_range,
        rate_max=rate_max,
        aero_load_derating=aero_load_derating,
    )


def test_small_step_first_order_response_is_63_percent_at_tau() -> None:
    """Small step should behave like a first-order lag when not rate-limited."""
    cfg = _make_servo_config(tau_servo=0.04, tau_servo_range=(0.04, 0.04), rate_max=10.5, aero_load_derating=0.0)
    m = ServoModel(cfg)
    m.reset()

    delta_cmd = np.array([0.1, 0.0, 0.0, 0.0])  # small enough to avoid rate saturation
    dt = 1e-4
    steps = int(round(cfg.tau_servo / dt))
    for _ in range(steps):
        m.step(delta_cmd, dt)

    # Analytic continuous-time solution: delta(t) = cmd * (1 - exp(-t/tau))
    expected = float(delta_cmd[0] * (1.0 - np.exp(-1.0)))
    assert np.isclose(m.delta_actual[0], expected, atol=2e-3, rtol=0.0)
    assert np.allclose(m.delta_actual[1:], 0.0, atol=1e-14)


def test_large_step_is_rate_limited() -> None:
    """Large step should clip delta_dot to rate_max_eff."""
    cfg = _make_servo_config(tau_servo=0.04, rate_max=10.5, aero_load_derating=0.0)
    m = ServoModel(cfg)
    m.reset()

    delta_cmd = np.ones(4) * 1.0
    delta_actual = np.zeros(4)
    delta_dot = m.compute_rate(delta_cmd, delta_actual)

    # Desired rate = 1/0.04 = 25 rad/s > 10.5 -> clipped
    assert np.allclose(delta_dot, np.ones(4) * 10.5, atol=0.0, rtol=0.0)


def test_derating_reduces_rate_limit() -> None:
    """Derating should reduce max slew rate: rate_max_eff = rate_max * (1 - derating)."""
    cfg = _make_servo_config(tau_servo=0.04, rate_max=10.0, aero_load_derating=0.5)
    m = ServoModel(cfg)
    m.reset()

    # With 50% derating: rate_max_eff = 5 rad/s
    assert np.isclose(m.rate_max_eff(), 5.0, atol=0.0, rtol=0.0)

    delta_cmd = np.array([1.0, 1.0, 1.0, 1.0])
    delta_dot = m.compute_rate(delta_cmd, np.zeros(4))
    assert np.allclose(delta_dot, 5.0, atol=0.0, rtol=0.0)


def test_step_euler_updates_positions() -> None:
    """ServoModel.step should Euler-integrate delta_actual."""
    cfg = _make_servo_config(tau_servo=0.04, rate_max=10.5, aero_load_derating=0.0)
    m = ServoModel(cfg)
    m.reset()

    delta_cmd = np.array([1.0, 0.0, 0.0, 0.0])
    dt = 0.01
    m.step(delta_cmd, dt)

    # Rate is clipped to 10.5 rad/s -> delta should advance by 0.105 rad.
    assert np.isclose(m.delta_actual[0], 10.5 * dt, atol=1e-14, rtol=0.0)
    assert np.allclose(m.delta_actual[1:], 0.0, atol=1e-14)


def test_reset_zeros_positions_and_randomizes_tau_and_derating_when_seeded() -> None:
    cfg = _make_servo_config(tau_servo=0.04, tau_servo_range=(0.03, 0.05), rate_max=10.5, aero_load_derating=0.3)
    m = ServoModel(cfg)

    # Put it in a non-zero state first.
    m.delta_actual = np.ones(4) * 0.123
    m.tau = 999.0
    m.derating = 0.999

    m.reset(seed=123)
    assert np.allclose(m.delta_actual, np.zeros(4))

    assert 0.03 <= m.tau <= 0.05
    assert 0.2 <= m.derating <= 0.5

    # Determinism for same seed
    tau1, d1 = m.tau, m.derating
    m.reset(seed=123)
    assert np.isclose(m.tau, tau1)
    assert np.isclose(m.derating, d1)

