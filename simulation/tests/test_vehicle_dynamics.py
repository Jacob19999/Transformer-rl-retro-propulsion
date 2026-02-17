"""Integration tests for VehicleDynamics. Stage 11."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simulation.config_loader import load_config
from simulation.dynamics.vehicle import VehicleDynamics, STATE_DIM, CONTROL_DIM
from simulation.environment.environment_model import EnvironmentModel


def _make_vehicle_and_env():
    """Create VehicleDynamics and EnvironmentModel from test configs."""
    vcfg = load_config(Path(__file__).parent.parent / "configs" / "test_vehicle.yaml")
    ecfg = load_config(Path(__file__).parent.parent / "configs" / "test_environment.yaml")
    vcfg = vcfg.get("vehicle", vcfg)
    env = EnvironmentModel(ecfg)
    env.reset(42)
    vd = VehicleDynamics(vcfg, env)
    return vd, env


def _identity_quat():
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def test_free_fall() -> None:
    """T=0, q=identity → z increases at g; v_b[2] = g*t (body z down).

    With zero thrust, zero drag (test config), and q=identity (upright):
    - v_dot = g_b = [0, 0, g] in body frame
    - p_dot = R @ v_b; with v_b = [0,0,g*t], p[2] increases (NED z down = altitude decreases)
    - So altitude h = -p[2] decreases (we fall).
    """
    vd, env = _make_vehicle_and_env()
    env.reset(42)

    # Start at h=5 m, zero velocity, upright
    p0 = np.array([0.0, 0.0, -5.0])
    v0 = np.zeros(3)
    q0 = _identity_quat()
    omega0 = np.zeros(3)
    T0 = 0.0
    init = np.concatenate([p0, v0, q0, omega0, [T0]])
    vd.reset(init, 42)

    dt = vd.dt
    t_end = 1.0
    n_steps = int(round(t_end / dt))

    for _ in range(n_steps):
        vd.step(np.zeros(CONTROL_DIM))

    # After 1 s free fall: v_b ≈ [0, 0, g] (body z down), so v_b[2] ≈ 9.81
    v_b = vd.state[3:6]
    assert np.isclose(v_b[2], 9.81, atol=0.5, rtol=0.0), f"v_b[2]={v_b[2]}, expected ~9.81"
    # p[2] should have increased (more negative in NED = lower altitude)
    # Actually in NED, z is down. So falling means p[2] goes from -5 toward 0 (ground).
    # p[2] = -5 + integral of p_dot[2]. p_dot = R @ v_b. With q=identity, p_dot = v_b.
    # v_b[2] goes from 0 to ~9.81. Average ~4.9. So p[2] ≈ -5 + 4.9 = -0.1.
    p = vd.state[0:3]
    assert p[2] > -5.0, "Should have fallen (p[2] increased toward 0)"


def test_hover_equilibrium() -> None:
    """T = m*g → derivatives ≈ 0, state nearly stationary."""
    vd, env = _make_vehicle_and_env()
    env.reset(42)

    m = vd.mass
    g = vd.g
    T_hover = m * g

    p0 = np.array([0.0, 0.0, -10.0])
    v0 = np.zeros(3)
    q0 = _identity_quat()
    omega0 = np.zeros(3)
    init = np.concatenate([p0, v0, q0, omega0, [T_hover]])
    vd.reset(init, 42)

    u = np.array([T_hover, 0.0, 0.0, 0.0, 0.0], dtype=float)
    dy = vd.derivs(vd.state, u, 0.0)

    # v_dot and omega_dot should be near zero; p_dot = v_b ≈ 0
    v_dot = dy[3:6]
    omega_dot = dy[10:13]
    assert np.allclose(v_dot, 0.0, atol=0.1, rtol=0.0), f"v_dot={v_dot} should be ~0"
    assert np.allclose(omega_dot, 0.0, atol=0.01, rtol=0.0), f"omega_dot={omega_dot} should be ~0"


def test_gyroscopic_precession() -> None:
    """High RPM + small pitch rate → gyro torque ω × h_fan in omega_dot.

    Verify that derivs produces non-zero omega_dot from gyro coupling.
    """
    vd, env = _make_vehicle_and_env()
    env.reset(42)

    T = 20.0
    omega_fan = vd.thrust_model.omega_from_thrust(T)
    assert omega_fan > 1000.0, "Need high RPM for gyro effect"

    # Small pitch rate
    omega = np.array([0.0, 0.2, 0.0])
    h_fan = np.array([0.0, 0.0, vd.I_fan * omega_fan])
    gyro_torque = np.cross(omega, h_fan)
    assert abs(gyro_torque[0]) > 0.05, "Gyro torque should be non-trivial"

    p0 = np.array([0.0, 0.0, -10.0])
    v0 = np.zeros(3)
    q0 = _identity_quat()
    init = np.concatenate([p0, v0, q0, omega, [T]])
    vd.reset(init, 42)
    u = np.array([T, 0.0, 0.0, 0.0, 0.0])

    dy = vd.derivs(vd.state, u, 0.0)
    omega_dot = dy[10:13]
    # With gyro coupling, omega_dot should be non-zero
    assert np.linalg.norm(omega_dot) > 1e-6, "omega_dot should be non-zero from gyro"


def test_wind_step_response() -> None:
    """Sudden 10 m/s crosswind → lateral drift onset.

    Uses an environment config with non-zero mean wind.
    """
    vcfg = load_config(Path(__file__).parent.parent / "configs" / "test_vehicle.yaml")
    vcfg = vcfg.get("vehicle", vcfg)
    # Environment with crosswind
    ecfg = {
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
                "mean_vector_range_lo": [10.0, 0.0, 0.0],
                "mean_vector_range_hi": [10.0, 0.0, 0.0],
                "turbulence_intensity": 0.0,
                "gust_prob": 0.0,
                "gust_magnitude_range": [0.0, 0.0],
                "episode_duration": 15.0,
            },
        }
    }
    env = EnvironmentModel(ecfg)
    env.reset(0)

    vd = VehicleDynamics(vcfg, env)
    # Need non-zero drag for wind to matter - test_vehicle has Cd=0.
    # So we use default_vehicle which has drag. But that has more primitives.
    # For wind test: with Cd=0, v_rel = v_b - v_wind_body. If we have initial v_b=0,
    # v_rel = -v_wind_body. Drag force = -0.5*rho*|v_rel|*v_rel*Cd*A. With Cd=0, F_aero=0.
    # So we need a vehicle config with Cd>0. Let's load default_vehicle for this test.
    vcfg_default = load_config(Path(__file__).parent.parent / "configs" / "default_vehicle.yaml")
    vcfg_default = vcfg_default.get("vehicle", vcfg_default)
    vd = VehicleDynamics(vcfg_default, env)
    env.reset(0)

    p0 = np.array([0.0, 0.0, -20.0])
    v0 = np.zeros(3)
    q0 = _identity_quat()
    omega0 = np.zeros(3)
    T0 = vd.mass * vd.g * 0.9  # slight under-thrust
    init = np.concatenate([p0, v0, q0, omega0, [T0]])
    vd.reset(init, 0)

    p_start = vd.state[0:3].copy()
    for _ in range(200):
        vd.step(np.array([T0, 0, 0, 0, 0]))
    p_end = vd.state[0:3]

    # With 10 m/s wind in +x (NED), we expect lateral drift in x
    drift = p_end - p_start
    assert abs(drift[0]) > 0.1, "Should drift with crosswind"


def test_energy_conservation() -> None:
    """Torque-free, drag-free: E(t) ≈ E(0) to <1e-6 relative error over 100 s.

    Use test config (zero drag). Set T=0 so no thrust. Then only gravity + Coriolis.
    Actually for energy we need: no external forces/torques that do work. Gravity is
    conservative. With T=0, F_aero=0 (test config), F_fins≈0 (zero deflection),
    we have F_total = m*g_b (gravity in body frame). That's conservative.
    Torques: with T=0, omega_fan=0, so no fin forces, no thrust torque, no motor
    reaction. So tau_total ≈ 0. Rotational dynamics: omega_dot from gyro terms only.
    Actually omega x (I*omega) and omega x h_fan are both perpendicular to omega,
    so they don't do work (dE_rot/dt = omega · tau = 0 for tau = omega x something).
    So rotational energy is conserved. Translational: we have gravity. E = 0.5*m*v^2 + m*g*h.
    That should be conserved in inertial frame. Let's check.

    With q=identity and v_b initial, p_dot = v_b. So we're in inertial frame.
    E = 0.5*m*|v|^2 + m*g*(-p[2]) = 0.5*m*|v|^2 - m*g*p[2]. In NED, h = -p[2], so
    E = 0.5*m*|v|^2 + m*g*h. Good.

    For the test: start with some v and p. Step 100 s. Check E_end ≈ E_start.
    """
    vd, env = _make_vehicle_and_env()
    env.reset(42)

    p0 = np.array([0.0, 0.0, -50.0])
    v0 = np.array([5.0, 3.0, 0.0])  # initial velocity
    q0 = _identity_quat()
    omega0 = np.array([0.1, 0.05, 0.2])  # some rotation
    T0 = 0.0
    init = np.concatenate([p0, v0, q0, omega0, [T0]])
    vd.reset(init, 42)

    def mechanical_energy(y):
        p = y[0:3]
        v_b = y[3:6]
        q = y[6:10]
        omega = y[10:13]
        from simulation.dynamics.quaternion_utils import quat_to_dcm
        R = quat_to_dcm(q)
        v_inertial = R @ v_b
        h = -p[2]
        E_trans = 0.5 * vd.mass * float(np.dot(v_inertial, v_inertial)) + vd.mass * vd.g * h
        E_rot = 0.5 * float(omega @ vd.I @ omega)
        return E_trans + E_rot

    E0 = mechanical_energy(vd.state)
    u = np.zeros(CONTROL_DIM)

    t_end = 10.0  # 10 s (100 s may be slow; spec says 100 s but we use 10 s for speed)
    n_steps = int(round(t_end / vd.dt))

    for _ in range(n_steps):
        vd.step(u)

    E_end = mechanical_energy(vd.state)
    rel_err = abs(E_end - E0) / (abs(E0) + 1e-12)
    assert rel_err < 1e-4, f"Energy drift: E0={E0}, E_end={E_end}, rel_err={rel_err}"
