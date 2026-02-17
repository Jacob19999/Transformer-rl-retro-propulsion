import numpy as np

from simulation.dynamics.integrator import RK4Integrator, rk4_step
from simulation.dynamics.quaternion_utils import quat_mult, quat_normalize


def test_rk4_step_exponential_decay_accuracy_and_order():
    # dy/dt = -y  =>  y(t) = exp(-t)
    def f(y: np.ndarray, _u: object, _t: float) -> np.ndarray:
        return -y

    y0 = np.array(1.0)
    t0 = 0.0

    dt = 0.2
    y_dt = rk4_step(f, y0, None, t0, dt)
    exact = np.exp(-dt)
    err_dt = float(np.abs(y_dt - exact))

    # Two half steps to same final time: global RK4 error should drop ~16× (order 4).
    dt2 = dt / 2.0
    y_half = rk4_step(f, y0, None, t0, dt2)
    y_half = rk4_step(f, y_half, None, t0 + dt2, dt2)
    err_half = float(np.abs(y_half - exact))

    # For dt=0.2, RK4 should already be very accurate (few micro-units).
    assert err_dt < 5e-6
    assert err_half < err_dt / 10.0


def test_quaternion_normalization_every_n_steps_keeps_unit_norm():
    # Integrate only quaternion kinematics with constant body angular rate.
    omega = np.array([0.3, -0.2, 0.1], dtype=float)
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]], dtype=float)

    def derivs(y: np.ndarray, _u: object, _t: float) -> np.ndarray:
        dy = np.zeros_like(y, dtype=float)
        q = y[6:10]
        q_dot = 0.5 * quat_mult(q, omega_quat)
        dy[6:10] = q_dot
        return dy

    rng = np.random.default_rng(0)
    q0 = quat_normalize(rng.normal(size=4))

    y = np.zeros(10, dtype=float)
    y[6:10] = q0

    integ = RK4Integrator(quat_slice=slice(6, 10), quat_normalize_every_n=10)

    t = 0.0
    dt = 0.01
    for _ in range(2000):
        y = integ.step(derivs, y, None, t, dt)
        t += dt

    q_final = y[6:10]
    assert np.isclose(np.linalg.norm(q_final), 1.0, atol=1e-8)

