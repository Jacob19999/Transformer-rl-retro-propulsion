"""
Unit tests for the semi-empirical fin aerodynamic model.

Tests: near-zero normal force at zero deflection, approximately linear response at
small angles, saturation at large angles, drag increases with deflection magnitude,
and vectorized computation across 4 fins.
"""

import pytest
import torch
import math
from tvc_env.dynamics.fin_aero import FinAeroModel


@pytest.fixture
def aero_model():
    return FinAeroModel(
        fin_area=0.002,
        max_deflection=0.262,
        C_N_alpha=3.5,
        k_sat=2.0,
        C_D_0=0.05,
        C_D_alpha2=1.5,
        exhaust_speed=40.0,
    )


class TestNormalForce:
    def test_near_zero_at_zero_deflection(self, aero_model):
        """Normal force should be near zero at zero deflection."""
        angles = torch.zeros(1, 4)
        throttle = torch.ones(1)
        result = aero_model.compute_forces(angles, throttle)
        assert result.normal_force.abs().max().item() < 1e-6

    def test_positive_angle_positive_normal_force(self, aero_model):
        """Positive deflection should give positive normal force."""
        angles = torch.full((1, 4), 0.1)
        throttle = torch.ones(1)
        result = aero_model.compute_forces(angles, throttle)
        assert torch.all(result.normal_force > 0)

    def test_negative_angle_negative_normal_force(self, aero_model):
        """Negative deflection should give negative normal force."""
        angles = torch.full((1, 4), -0.1)
        throttle = torch.ones(1)
        result = aero_model.compute_forces(angles, throttle)
        assert torch.all(result.normal_force < 0)

    def test_approximately_linear_at_small_angles(self, aero_model):
        """Normal force should scale linearly with deflection at small angles."""
        throttle = torch.ones(1)
        a1 = torch.full((1, 4), 0.05)
        a2 = torch.full((1, 4), 0.10)
        r1 = aero_model.compute_forces(a1, throttle)
        r2 = aero_model.compute_forces(a2, throttle)
        ratio = (r2.normal_force / r1.normal_force).mean().item()
        # Should be approximately 2.0 (linear scaling)
        assert abs(ratio - 2.0) < 0.2, f"Expected ~2.0, got {ratio:.3f}"

    def test_saturation_at_large_angles(self, aero_model):
        """Normal force should saturate and not grow indefinitely at large angles."""
        throttle = torch.ones(1)
        angles_small = torch.full((1, 4), 0.1)
        angles_large = torch.full((1, 4), 0.25)
        r_small = aero_model.compute_forces(angles_small, throttle)
        r_large = aero_model.compute_forces(angles_large, throttle)
        # Rate of increase should be less than linear at large angles
        # 0.25/0.1 = 2.5 linear ratio, but saturation means actual ratio < 2.5
        ratio = r_large.normal_force.mean().item() / r_small.normal_force.mean().item()
        assert ratio < 2.5, f"No saturation detected: ratio={ratio:.3f}"


class TestTangentialForce:
    def test_drag_always_positive(self, aero_model):
        """Tangential drag should always be positive regardless of deflection sign."""
        throttle = torch.ones(1)
        for angle in [-0.2, -0.1, 0.0, 0.1, 0.2]:
            angles = torch.full((1, 4), angle)
            result = aero_model.compute_forces(angles, throttle)
            assert torch.all(result.tangential_force >= 0), \
                f"Negative drag at angle={angle}"

    def test_drag_increases_with_deflection_magnitude(self, aero_model):
        """Drag should increase as deflection magnitude increases."""
        throttle = torch.ones(1)
        a_small = torch.full((1, 4), 0.05)
        a_large = torch.full((1, 4), 0.20)
        r_small = aero_model.compute_forces(a_small, throttle)
        r_large = aero_model.compute_forces(a_large, throttle)
        assert r_large.tangential_force.mean() > r_small.tangential_force.mean()

    def test_symmetry_at_equal_magnitude(self, aero_model):
        """Equal positive and negative deflections should give equal drag."""
        throttle = torch.ones(1)
        a_pos = torch.full((1, 4), 0.15)
        a_neg = torch.full((1, 4), -0.15)
        r_pos = aero_model.compute_forces(a_pos, throttle)
        r_neg = aero_model.compute_forces(a_neg, throttle)
        assert torch.allclose(r_pos.tangential_force, r_neg.tangential_force, atol=1e-6)


class TestVectorization:
    def test_batch_dimension(self, aero_model):
        """Should handle batch of 128 environments with 4 fins each."""
        num_envs = 128
        angles = torch.randn(num_envs, 4) * 0.1
        throttle = torch.rand(num_envs)
        result = aero_model.compute_forces(angles, throttle)
        assert result.normal_force.shape == (num_envs, 4)
        assert result.tangential_force.shape == (num_envs, 4)

    def test_throttle_scaling(self, aero_model):
        """Force at 50% throttle should be ~25% of force at 100% throttle (q ∝ throttle²)."""
        angles = torch.full((1, 4), 0.1)
        r_100 = aero_model.compute_forces(angles, torch.ones(1))
        r_50 = aero_model.compute_forces(angles, torch.full((1,), 0.5))
        ratio = r_50.normal_force.mean().item() / r_100.normal_force.mean().item()
        assert abs(ratio - 0.25) < 0.01, f"Expected 0.25, got {ratio:.4f}"

    def test_output_is_finite(self, aero_model):
        """All outputs should be finite for valid inputs."""
        angles = torch.randn(32, 4) * 0.15
        throttle = torch.rand(32)
        result = aero_model.compute_forces(angles, throttle)
        assert torch.all(torch.isfinite(result.normal_force))
        assert torch.all(torch.isfinite(result.tangential_force))
        assert torch.all(torch.isfinite(result.force_vector))
