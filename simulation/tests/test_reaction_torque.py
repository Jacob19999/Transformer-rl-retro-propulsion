import numpy as np

from simulation.dynamics.thrust_model import ThrustModel, ThrustModelConfig


def _make_model(*, k_torque: float = 1.0e-8, anti_torque_enabled: bool = True) -> ThrustModel:
    cfg = ThrustModelConfig(
        k_thrust=4.55e-7,
        tau_motor=0.1,
        r_thrust=np.array([0.0, 0.0, 0.08]),
        r_duct=0.045,
        I_fan=3.0e-5,
        k_torque=k_torque,
        anti_torque_enabled=anti_torque_enabled,
    )
    return ThrustModel(cfg)


def test_anti_torque_at_hover() -> None:
    model = _make_model()
    tau = model.steady_state_anti_torque(T=30.5)
    assert np.allclose(tau, np.array([0.0, 0.0, -0.67032967]), rtol=1e-2, atol=0.0)


def test_anti_torque_at_zero() -> None:
    model = _make_model()
    tau = model.steady_state_anti_torque(T=0.0)
    assert np.allclose(tau, np.zeros(3))


def test_anti_torque_proportional() -> None:
    model = _make_model()
    thrust_levels = np.array([13.5, 30.5, 45.0], dtype=float)
    tau_z = np.array([model.steady_state_anti_torque(T=T)[2] for T in thrust_levels])
    assert np.allclose(tau_z / tau_z[0], thrust_levels / thrust_levels[0], rtol=1e-6, atol=0.0)


def test_anti_torque_disabled() -> None:
    model = _make_model(anti_torque_enabled=False)
    _, tau, _ = model.outputs(T=30.5, T_cmd=30.5, h=1.0, rho=model.config.rho_ref)
    assert np.allclose(tau, np.zeros(3))


def test_ramp_torque_sign() -> None:
    model = _make_model()
    tau = model.motor_reaction_torque(T=30.5, T_dot=100.0)
    assert tau[2] < 0.0


def test_ramp_torque_at_constant_thrust() -> None:
    model = _make_model()
    tau = model.motor_reaction_torque(T=30.5, T_dot=0.0)
    assert np.allclose(tau, np.zeros(3))


def test_combined_torque() -> None:
    model = _make_model()
    tau_anti = model.steady_state_anti_torque(T=30.5)
    tau_ramp = model.motor_reaction_torque(T=30.5, T_dot=100.0)
    assert tau_anti[2] != 0.0
    assert tau_ramp[2] != 0.0
    assert (tau_anti + tau_ramp)[2] < tau_anti[2]
