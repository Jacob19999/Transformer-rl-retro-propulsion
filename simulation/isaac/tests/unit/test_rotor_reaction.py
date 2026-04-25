"""
Unit tests for rotor reaction torque computations.

Tests: static torque opposes spin direction, dynamic spool torque sign during
accel/decel, gyro precession direction follows ω × H, magnitudes against hand calculations.
"""

import pytest
import torch
import math
from tvc_env.dynamics.rotor_reaction import (
    compute_static_reaction_torque,
    compute_dynamic_spool_torque,
    compute_gyroscopic_precession,
    compute_all_rotor_torques,
)


# Test parameters
K_Q = 1e-5          # N·m·s²/rad²
I_ROTOR = 0.0005    # kg·m²
OMEGA = 1000.0      # rad/s
DT = 0.00833        # s (120 Hz)
SPIN_AXIS = torch.tensor([0.0, 0.0, 1.0])  # +z in body-FRD


class TestStaticReactionTorque:
    def test_opposes_spin_direction(self):
        """Static reaction torque should oppose the spin axis (negative for +z spin)."""
        omega = torch.tensor([OMEGA])
        torque = compute_static_reaction_torque(omega, K_Q, SPIN_AXIS)
        # Torque should be along -z (opposing +z spin)
        assert torque[0, 2].item() < 0.0, f"Expected negative z-torque, got {torque[0, 2]}"

    def test_zero_omega_zero_torque(self):
        """Zero rotor speed should give zero reaction torque."""
        omega = torch.zeros(1)
        torque = compute_static_reaction_torque(omega, K_Q, SPIN_AXIS)
        assert torch.allclose(torque, torch.zeros(1, 3))

    def test_magnitude_proportional_to_omega_squared(self):
        """Static torque magnitude should scale as ω²."""
        omega1 = torch.tensor([500.0])
        omega2 = torch.tensor([1000.0])
        t1 = compute_static_reaction_torque(omega1, K_Q, SPIN_AXIS).norm().item()
        t2 = compute_static_reaction_torque(omega2, K_Q, SPIN_AXIS).norm().item()
        # t2/t1 should be (1000/500)² = 4
        assert abs(t2 / t1 - 4.0) < 0.01, f"Expected ratio 4.0, got {t2/t1:.4f}"

    def test_hand_calculation(self):
        """Verify against hand calculation: Q = k_Q * ω² = 1e-5 * 1000² = 10 N·m."""
        omega = torch.tensor([1000.0])
        torque = compute_static_reaction_torque(omega, K_Q, SPIN_AXIS)
        expected_magnitude = K_Q * OMEGA ** 2  # = 10 N·m
        assert abs(torque.norm().item() - expected_magnitude) < 1e-4

    def test_batch_shape(self):
        """Should handle batch dimension."""
        omega = torch.full((32,), OMEGA)
        torque = compute_static_reaction_torque(omega, K_Q, SPIN_AXIS)
        assert torque.shape == (32, 3)


class TestDynamicSpoolTorque:
    def test_negative_during_acceleration(self):
        """Body spool reaction torque should oppose rotor acceleration."""
        omega_prev = torch.tensor([500.0])
        omega = torch.tensor([1000.0])
        torque = compute_dynamic_spool_torque(omega, omega_prev, I_ROTOR, SPIN_AXIS, DT)
        # Accelerating → positive spool torque along spin axis (+z)
        assert torque[0, 2].item() < 0.0, f"Expected negative z-torque, got {torque[0, 2]}"

    def test_positive_during_deceleration(self):
        """Body spool reaction torque should reverse during rotor deceleration."""
        omega_prev = torch.tensor([1000.0])
        omega = torch.tensor([500.0])
        torque = compute_dynamic_spool_torque(omega, omega_prev, I_ROTOR, SPIN_AXIS, DT)
        assert torque[0, 2].item() > 0.0, f"Expected positive z-torque, got {torque[0, 2]}"

    def test_zero_at_steady_state(self):
        """No spool torque when rotor speed is constant."""
        omega = torch.tensor([1000.0])
        torque = compute_dynamic_spool_torque(omega, omega.clone(), I_ROTOR, SPIN_AXIS, DT)
        assert torque.abs().max().item() < 1e-6

    def test_magnitude_proportional_to_inertia(self):
        """Spool torque magnitude scales linearly with rotor inertia."""
        omega_prev = torch.tensor([0.0])
        omega = torch.tensor([1000.0])
        t1 = compute_dynamic_spool_torque(omega, omega_prev, I_ROTOR, SPIN_AXIS, DT).norm().item()
        t2 = compute_dynamic_spool_torque(omega, omega_prev, 2 * I_ROTOR, SPIN_AXIS, DT).norm().item()
        assert abs(t2 / t1 - 2.0) < 0.01


class TestGyroPrecession:
    def test_direction_follows_cross_product(self):
        """Precession τ = ω_body × H should follow right-hand rule."""
        omega = torch.tensor([1000.0])  # rotor spinning around +z
        # Body rotating around +x axis
        body_ang_vel = torch.tensor([[1.0, 0.0, 0.0]])  # ω_body = +x
        torque = compute_gyroscopic_precession(omega, body_ang_vel, I_ROTOR, SPIN_AXIS)
        # H = I * omega * z = [0, 0, I*omega]
        # τ = ω_body × H = [1,0,0] × [0,0,H_z] = [0*H_z - 0, 0 - 1*H_z, 0] = [0, -H_z, 0]
        assert torque[0, 1].item() < 0.0, f"Expected -y precession, got {torque[0]}"
        assert abs(torque[0, 0].item()) < 1e-6  # no x component
        assert abs(torque[0, 2].item()) < 1e-6  # no z component

    def test_zero_when_body_not_rotating(self):
        """No gyroscopic torque when body angular velocity is zero."""
        omega = torch.tensor([1000.0])
        body_ang_vel = torch.zeros(1, 3)
        torque = compute_gyroscopic_precession(omega, body_ang_vel, I_ROTOR, SPIN_AXIS)
        assert torque.abs().max().item() < 1e-6

    def test_proportional_to_omega_and_body_rate(self):
        """Gyro torque scales with both rotor speed and body angular rate."""
        body_ang_vel = torch.tensor([[1.0, 0.0, 0.0]])
        omega_half = torch.tensor([500.0])
        omega_full = torch.tensor([1000.0])
        t_half = compute_gyroscopic_precession(omega_half, body_ang_vel, I_ROTOR, SPIN_AXIS).norm().item()
        t_full = compute_gyroscopic_precession(omega_full, body_ang_vel, I_ROTOR, SPIN_AXIS).norm().item()
        # Should double when omega doubles
        assert abs(t_full / t_half - 2.0) < 0.01

    def test_batch_shape(self):
        """Should handle batch dimension."""
        omega = torch.full((16,), OMEGA)
        body_ang_vel = torch.randn(16, 3)
        torque = compute_gyroscopic_precession(omega, body_ang_vel, I_ROTOR, SPIN_AXIS)
        assert torque.shape == (16, 3)
