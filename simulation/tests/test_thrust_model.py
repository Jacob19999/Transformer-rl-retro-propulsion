import numpy as np

from simulation.dynamics.thrust_model import ThrustModel, ThrustModelConfig


def test_thrust_curve_and_inverse_round_trip() -> None:
    cfg = ThrustModelConfig(
        k_thrust=4.55e-7,
        tau_motor=0.1,
        r_thrust=np.array([0.0, 0.0, 0.08]),
        r_duct=0.045,
        I_fan=0.001,
    )
    m = ThrustModel(cfg)

    omega = 5000.0
    T = m.thrust_from_omega(omega)
    omega2 = m.omega_from_thrust(T)
    assert np.isclose(omega2, omega, rtol=1e-12, atol=0.0)


def test_motor_lag_step_response_is_63_percent_at_tau() -> None:
    cfg = ThrustModelConfig(
        k_thrust=1.0,
        tau_motor=0.1,
        r_thrust=np.array([0.0, 0.0, 0.0]),
        r_duct=0.1,
        I_fan=0.0,
    )
    m = ThrustModel(cfg)

    T_cmd = 10.0
    T = 0.0
    dt = 1e-4
    steps = int(round(cfg.tau_motor / dt))
    for _ in range(steps):
        T += m.thrust_dot(T=T, T_cmd=T_cmd) * dt

    # Continuous-time analytic solution: T(t) = T_cmd * (1 - exp(-t/tau))
    expected = T_cmd * (1.0 - np.exp(-1.0))
    assert np.isclose(T, expected, atol=2e-3, rtol=0.0)


def test_ground_effect_at_h_equals_r_duct_is_1p5x() -> None:
    cfg = ThrustModelConfig(
        k_thrust=1.0,
        tau_motor=0.1,
        r_thrust=np.array([0.0, 0.0, 0.0]),
        r_duct=0.045,
        I_fan=0.0,
    )
    m = ThrustModel(cfg)

    T = 10.0
    rho = cfg.rho_ref
    T_eff = m.effective_thrust(T=T, h=cfg.r_duct, rho=rho)
    assert np.isclose(T_eff, 15.0, atol=0.0, rtol=0.0)


def test_density_correction_scales_linearly() -> None:
    cfg = ThrustModelConfig(
        k_thrust=1.0,
        tau_motor=0.1,
        r_thrust=np.array([0.0, 0.0, 0.0]),
        r_duct=0.1,
        I_fan=0.0,
    )
    m = ThrustModel(cfg)

    T = 10.0
    h = 100.0
    rho = 0.5 * cfg.rho_ref
    T_eff = m.effective_thrust(T=T, h=h, rho=rho)
    expected = T * m.ground_effect_factor(h=h) * (rho / cfg.rho_ref)
    assert np.isclose(T_eff, expected, atol=0.0, rtol=0.0)


def test_zero_thrust_produces_zero_force_and_torque() -> None:
    cfg = ThrustModelConfig(
        k_thrust=1.0,
        tau_motor=0.1,
        r_thrust=np.array([1.0, 2.0, 3.0]),
        r_duct=0.1,
        I_fan=0.1,
    )
    m = ThrustModel(cfg)

    F, tau, T_dot = m.outputs(T=0.0, T_cmd=0.0, h=1.0, rho=cfg.rho_ref)
    assert np.allclose(F, np.zeros(3))
    assert np.allclose(tau, np.zeros(3))
    assert T_dot == 0.0


def test_thrust_force_vector_and_offset_torque() -> None:
    cfg = ThrustModelConfig(
        k_thrust=1.0,
        tau_motor=0.1,
        r_thrust=np.array([1.0, 0.0, 0.0]),
        r_duct=0.1,
        I_fan=0.0,
    )
    m = ThrustModel(cfg)

    T_eff = 10.0
    F = m.thrust_force(T_eff=T_eff)
    assert np.allclose(F, np.array([0.0, 0.0, 10.0]))

    tau = m.thrust_torque(r_offset=cfg.r_thrust, F_thrust=F)
    assert np.allclose(tau, np.array([0.0, -10.0, 0.0]))


def test_motor_reaction_torque_uses_omega_dot_from_T_dot() -> None:
    cfg = ThrustModelConfig(
        k_thrust=1.0,
        tau_motor=0.1,
        r_thrust=np.array([0.0, 0.0, 0.0]),
        r_duct=0.1,
        I_fan=0.1,
    )
    m = ThrustModel(cfg)

    # T = k*omega^2, choose omega=2 -> T=4
    T = 4.0
    T_dot = 2.0
    tau = m.motor_reaction_torque(T=T, T_dot=T_dot)

    # omega=2, omega_dot = T_dot / (2*k*omega) = 2/(2*1*2)=0.5
    # tau_z = -I_fan * omega_dot = -0.1*0.5 = -0.05
    assert np.allclose(tau, np.array([0.0, 0.0, -0.05]))

