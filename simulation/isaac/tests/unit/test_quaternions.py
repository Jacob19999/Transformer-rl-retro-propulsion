"""
Unit tests for quaternion operations.

Tests multiply, rotate_vector, convention conversion round-trip, and edge cases.
All quaternions in (w,x,y,z) convention per Isaac Lab 2.3.2.
"""

import pytest
import torch
import math
from tvc_env.common.quaternions import (
    identity,
    normalize,
    multiply,
    inverse,
    rotate_vector,
    to_rotation_matrix,
    from_euler,
    tilt_angle,
    to_euler,
    isaac_wxyz_to_xyzw,
    xyzw_to_isaac_wxyz,
)


class TestIdentity:
    def test_identity_single(self):
        """Identity quaternion should be (1, 0, 0, 0)."""
        q = identity()
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assert torch.allclose(q, expected)

    def test_identity_batch(self):
        """Batch identity should have correct shape."""
        q = identity(num=10)
        assert q.shape == (10, 4)
        assert torch.all(q[:, 0] == 1.0)
        assert torch.all(q[:, 1:] == 0.0)


class TestNormalize:
    def test_unit_quaternion_unchanged(self):
        """Normalizing a unit quaternion should not change it."""
        q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assert torch.allclose(normalize(q), q)

    def test_normalizes_to_unit_length(self):
        """Any non-zero quaternion should normalize to unit length."""
        q = torch.tensor([2.0, 1.0, 1.0, 1.0])
        qn = normalize(q)
        assert abs(qn.norm().item() - 1.0) < 1e-6


class TestMultiply:
    def test_identity_product(self):
        """q * identity = q."""
        q = normalize(torch.tensor([0.707, 0.707, 0.0, 0.0]))
        qi = identity()
        result = multiply(q, qi)
        assert torch.allclose(result, q, atol=1e-5)

    def test_inverse_product(self):
        """q * q^-1 = identity."""
        q = normalize(torch.tensor([0.707, 0.0, 0.707, 0.0]))
        result = multiply(q, inverse(q))
        assert torch.allclose(result, identity(), atol=1e-5)

    def test_180_degree_rotations(self):
        """Two 90° rotations around same axis = 180° rotation."""
        # 90° around z-axis in wxyz: w=cos(45°), z=sin(45°)
        half = math.sqrt(2) / 2
        q90 = torch.tensor([half, 0.0, 0.0, half])
        q180 = multiply(q90, q90)
        # Should be (0, 0, 0, 1) for 180° around z (or its negative)
        expected_w = torch.cos(torch.tensor(math.pi / 2))
        assert abs(abs(q180[3].item()) - 1.0) < 1e-5  # z component dominant


class TestRotateVector:
    def test_identity_rotation(self):
        """Identity quaternion should not rotate vector."""
        q = identity()
        v = torch.tensor([1.0, 2.0, 3.0])
        result = rotate_vector(q, v)
        assert torch.allclose(result, v, atol=1e-6)

    def test_90_degree_z_rotation(self):
        """90° rotation around z-axis: x → y, y → -x."""
        half = math.sqrt(2) / 2
        q = torch.tensor([half, 0.0, 0.0, half])  # 90° around z
        v_x = torch.tensor([1.0, 0.0, 0.0])
        rotated = rotate_vector(q, v_x)
        expected = torch.tensor([0.0, 1.0, 0.0])
        assert torch.allclose(rotated, expected, atol=1e-6)

    def test_180_degree_rotation(self):
        """180° rotation around z-axis: x → -x, y → -y."""
        q = torch.tensor([0.0, 0.0, 0.0, 1.0])  # 180° around z
        v = torch.tensor([1.0, 0.0, 0.0])
        rotated = rotate_vector(q, v)
        expected = torch.tensor([-1.0, 0.0, 0.0])
        assert torch.allclose(rotated, expected, atol=1e-6)

    def test_preserves_vector_length(self):
        """Rotation should preserve vector magnitude."""
        q = normalize(torch.tensor([0.5, 0.3, 0.7, 0.2]))
        v = torch.tensor([1.0, 2.0, 3.0])
        rotated = rotate_vector(q, v)
        assert abs(rotated.norm().item() - v.norm().item()) < 1e-5

    def test_batch_rotation(self):
        """Batch rotation should work for (N, 4) quaternions and (N, 3) vectors."""
        N = 32
        q = normalize(torch.randn(N, 4))
        v = torch.randn(N, 3)
        rotated = rotate_vector(q, v)
        assert rotated.shape == (N, 3)
        # All lengths preserved
        orig_norms = v.norm(dim=-1)
        new_norms = rotated.norm(dim=-1)
        assert torch.allclose(orig_norms, new_norms, atol=1e-4)


class TestConventionConversion:
    def test_wxyz_to_xyzw_round_trip(self):
        """wxyz → xyzw → wxyz should be identity."""
        q_wxyz = normalize(torch.tensor([0.707, 0.0, 0.707, 0.0]))
        assert torch.allclose(xyzw_to_isaac_wxyz(isaac_wxyz_to_xyzw(q_wxyz)), q_wxyz)

    def test_xyzw_to_wxyz_round_trip(self):
        """xyzw → wxyz → xyzw should be identity."""
        q_xyzw = torch.tensor([0.0, 0.707, 0.0, 0.707])  # xyzw ordering
        assert torch.allclose(isaac_wxyz_to_xyzw(xyzw_to_isaac_wxyz(q_xyzw)), q_xyzw)

    def test_convention_conversion_correctness(self):
        """wxyz (1,0,0,0) → xyzw should be (0,0,0,1)."""
        q_wxyz = torch.tensor([1.0, 0.0, 0.0, 0.0])
        q_xyzw = isaac_wxyz_to_xyzw(q_wxyz)
        assert torch.allclose(q_xyzw, torch.tensor([0.0, 0.0, 0.0, 1.0]))


class TestEulerConversions:
    def test_zero_euler_is_identity(self):
        """Zero roll/pitch/yaw should give identity quaternion."""
        z = torch.tensor(0.0)
        q = from_euler(z, z, z)
        assert torch.allclose(q, identity(), atol=1e-6)

    def test_euler_round_trip(self):
        """from_euler(to_euler(q)) should recover original quaternion."""
        roll = torch.tensor(0.3)
        pitch = torch.tensor(0.1)
        yaw = torch.tensor(0.5)
        q = from_euler(roll, pitch, yaw)
        r, p, y = to_euler(q)
        assert abs(r.item() - roll.item()) < 1e-5
        assert abs(p.item() - pitch.item()) < 1e-5
        assert abs(y.item() - yaw.item()) < 1e-5


class TestTiltAngle:
    def test_yaw_does_not_change_tilt(self):
        zero = torch.tensor(0.0)
        q = from_euler(zero, zero, torch.tensor(1.7))
        assert torch.allclose(tilt_angle(q), zero, atol=1e-6)

    def test_combined_rotation_matches_body_vertical(self):
        q = from_euler(torch.tensor(0.6), torch.tensor(0.5), torch.tensor(1.2))
        body_up_world = rotate_vector(q, torch.tensor([0.0, 0.0, 1.0]))
        expected = torch.acos(body_up_world[2].clamp(-1.0, 1.0))
        assert torch.allclose(tilt_angle(q), expected, atol=1e-6)
