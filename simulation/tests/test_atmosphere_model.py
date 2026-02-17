import numpy as np

from simulation.environment.atmosphere_model import AtmosphereModel, AtmosphereModelConfig


def test_isa_sea_level_density_is_approximately_1p225() -> None:
    cfg = AtmosphereModelConfig(
        T_base=288.15,
        T_lapse=-0.0065,
        P_base=101325.0,
        rho_ref=1.225,
        randomize_T=0.0,
        randomize_P=0.0,
    )
    m = AtmosphereModel(cfg, rng=np.random.default_rng(0))
    m.reset()

    T, P, rho = m.get_conditions(0.0)
    assert np.isclose(T, 288.15, atol=0.0, rtol=0.0)
    assert np.isclose(P, 101325.0, atol=0.0, rtol=0.0)

    # Using the ideal gas constant R=287.058 gives ~1.2250 at sea level.
    assert np.isclose(rho, 1.225, atol=2e-3, rtol=0.0)


def test_reset_randomization_bounds_are_respected() -> None:
    cfg = AtmosphereModelConfig(
        T_base=288.15,
        T_lapse=-0.0065,
        P_base=101325.0,
        rho_ref=1.225,
        randomize_T=10.0,
        randomize_P=2000.0,
    )
    rng = np.random.default_rng(123)
    m = AtmosphereModel(cfg, rng=rng)

    for _ in range(100):
        m.reset()
        assert (cfg.T_base - cfg.randomize_T) <= m.T_base <= (cfg.T_base + cfg.randomize_T)
        assert (cfg.P_base - cfg.randomize_P) <= m.P_base <= (cfg.P_base + cfg.randomize_P)


def test_ideal_gas_consistency_holds() -> None:
    cfg = AtmosphereModelConfig(
        T_base=288.15,
        T_lapse=-0.0065,
        P_base=101325.0,
        rho_ref=1.225,
        randomize_T=5.0,
        randomize_P=500.0,
    )
    m = AtmosphereModel(cfg, rng=np.random.default_rng(7))
    m.reset()

    for h in [0.0, 5.0, 10.0]:
        T, P, rho = m.get_conditions(h)
        assert np.isclose(rho, P / (m.R * T), rtol=1e-12, atol=0.0)

