"""
Unit tests for frame conversion correctness.

Tests round-trip FRD↔Isaac identity, known vector transforms, and batch ops.
"""

import pytest
import torch
import math
from tvc_env.common.frames import (
    frd_to_isaac,
    isaac_to_frd,
    get_frd_to_isaac_matrix,
    get_isaac_to_frd_matrix,
)


class TestFrdToIsaac:
    def test_forward_in_frd_is_backward_in_isaac(self):
        """x-forward in FRD should map to -z in Isaac (back axis flipped)."""
        v_frd = torch.tensor([1.0, 0.0, 0.0])
        v_isaac = frd_to_isaac(v_frd)
        expected = torch.tensor([0.0, 0.0, -1.0])
        assert torch.allclose(v_isaac, expected, atol=1e-6), f"Got {v_isaac}"

    def test_right_in_frd_is_right_in_isaac(self):
        """y-right in FRD should map to +x in Isaac."""
        v_frd = torch.tensor([0.0, 1.0, 0.0])
        v_isaac = frd_to_isaac(v_frd)
        expected = torch.tensor([1.0, 0.0, 0.0])
        assert torch.allclose(v_isaac, expected, atol=1e-6), f"Got {v_isaac}"

    def test_down_in_frd_is_down_in_isaac(self):
        """z-down in FRD should map to -y in Isaac (y-up convention)."""
        v_frd = torch.tensor([0.0, 0.0, 1.0])
        v_isaac = frd_to_isaac(v_frd)
        expected = torch.tensor([0.0, -1.0, 0.0])
        assert torch.allclose(v_isaac, expected, atol=1e-6), f"Got {v_isaac}"


class TestIsaacToFrd:
    def test_right_in_isaac_is_right_in_frd(self):
        """x-right in Isaac should map to y-right in FRD."""
        v_isaac = torch.tensor([1.0, 0.0, 0.0])
        v_frd = isaac_to_frd(v_isaac)
        expected = torch.tensor([0.0, 1.0, 0.0])
        assert torch.allclose(v_frd, expected, atol=1e-6), f"Got {v_frd}"

    def test_up_in_isaac_is_up_in_frd(self):
        """y-up in Isaac should map to -z in FRD (z=down)."""
        v_isaac = torch.tensor([0.0, 1.0, 0.0])
        v_frd = isaac_to_frd(v_isaac)
        expected = torch.tensor([0.0, 0.0, -1.0])
        assert torch.allclose(v_frd, expected, atol=1e-6), f"Got {v_frd}"


class TestRoundTrip:
    def test_frd_to_isaac_round_trip(self):
        """FRD → Isaac → FRD should recover original vector."""
        v = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(isaac_to_frd(frd_to_isaac(v)), v, atol=1e-6)

    def test_isaac_to_frd_round_trip(self):
        """Isaac → FRD → Isaac should recover original vector."""
        v = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(frd_to_isaac(isaac_to_frd(v)), v, atol=1e-6)

    def test_round_trip_batch(self):
        """Round-trip on batched inputs."""
        v = torch.randn(128, 3)
        recovered = isaac_to_frd(frd_to_isaac(v))
        assert torch.allclose(recovered, v, atol=1e-5)


class TestMatrixProperties:
    def test_rotation_matrix_orthogonal(self):
        """FRD→Isaac rotation matrix should be orthogonal (R^T R = I)."""
        R = get_frd_to_isaac_matrix(dtype=torch.float64)
        I = torch.eye(3, dtype=torch.float64)
        assert torch.allclose(R.T @ R, I, atol=1e-10)

    def test_rotation_matrix_determinant_one(self):
        """FRD→Isaac matrix should have det=+1 (proper rotation)."""
        R = get_frd_to_isaac_matrix(dtype=torch.float64)
        assert abs(torch.det(R).item() - 1.0) < 1e-10

    def test_inverse_is_transpose(self):
        """Isaac→FRD should be the transpose of FRD→Isaac."""
        R_fwd = get_frd_to_isaac_matrix(dtype=torch.float64)
        R_inv = get_isaac_to_frd_matrix(dtype=torch.float64)
        assert torch.allclose(R_fwd.T, R_inv, atol=1e-10)
