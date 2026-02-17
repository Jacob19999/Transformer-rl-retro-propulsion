import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from simulation.dynamics.quaternion_utils import (
    euler_to_quat,
    quat_mult,
    quat_normalize,
    quat_to_dcm,
    quat_to_euler,
)


def test_quat_to_dcm_matches_scipy_for_random_quaternions():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q = q / np.linalg.norm(q)

        dcm = quat_to_dcm(q)

        # SciPy expects [x, y, z, w] ordering.
        r = R.from_quat([q[1], q[2], q[3], q[0]])
        dcm_scipy = r.as_matrix()

        assert np.allclose(dcm, dcm_scipy, atol=1e-8)


def test_quat_to_dcm_identity_and_90deg_rotations():
    # Identity
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])
    R_id = quat_to_dcm(q_identity)
    assert np.allclose(R_id, np.eye(3), atol=1e-8)

    # 90° roll about x-axis
    q_roll_90 = euler_to_quat(np.pi / 2.0, 0.0, 0.0)
    R_roll_90 = quat_to_dcm(q_roll_90)
    x_body = np.array([1.0, 0.0, 0.0])
    z_body = np.array([0.0, 0.0, 1.0])

    x_inertial = R_roll_90 @ x_body
    z_inertial = R_roll_90 @ z_body

    assert np.allclose(x_inertial, [1.0, 0.0, 0.0], atol=1e-8)
    # With the chosen convention, a +90° roll rotates body z into -body y.
    assert np.allclose(z_inertial, [0.0, -1.0, 0.0], atol=1e-8)


def test_quat_mult_identity_and_associativity():
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    q_a = euler_to_quat(0.3, -0.2, 0.1)
    q_b = euler_to_quat(-0.1, 0.5, -0.4)

    # Identity element
    np.testing.assert_allclose(quat_mult(q_id, q_a), q_a, atol=1e-8)
    np.testing.assert_allclose(quat_mult(q_a, q_id), q_a, atol=1e-8)

    # Associativity: (a ⊗ b) ⊗ id == a ⊗ (b ⊗ id)
    left = quat_mult(quat_mult(q_a, q_b), q_id)
    right = quat_mult(q_a, quat_mult(q_b, q_id))
    np.testing.assert_allclose(left, right, atol=1e-8)


def test_quat_normalize_unit_norm_and_error_on_zero():
    q = np.array([2.0, 0.0, 0.0, 0.0])
    q_norm = quat_normalize(q)
    assert np.isclose(np.linalg.norm(q_norm), 1.0, atol=1e-12)
    assert np.allclose(q_norm, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    with pytest.raises(ValueError):
        quat_normalize([0.0, 0.0, 0.0, 0.0])


def test_euler_quat_roundtrip_small_angles():
    rng = np.random.default_rng(1)
    for _ in range(20):
        # Avoid singularities by keeping angles within [-pi/2, pi/2]
        roll, pitch, yaw = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size=3)

        q = euler_to_quat(roll, pitch, yaw)
        roll2, pitch2, yaw2 = quat_to_euler(q)

        assert np.allclose([roll2, pitch2, yaw2], [roll, pitch, yaw], atol=1e-6)


def test_quat_to_dcm_double_cover():
    q = euler_to_quat(0.3, -0.4, 0.7)
    q_neg = -q

    R_q = quat_to_dcm(q)
    R_q_neg = quat_to_dcm(q_neg)

    assert np.allclose(R_q, R_q_neg, atol=1e-8)

