"""
Rotor reaction torque computation.

Computes:
  - Static reaction torque: Q = k_Q * ω² (opposes spin direction)
  - Dynamic spool torque: I_rotor * dω/dt
  - Gyroscopic precession: τ = ω_body × H_rotor (H = I_rotor * ω * spin_axis)

All outputs are separate vec3 tensors for independent logging per FR-018.
"""

from __future__ import annotations
import torch
from torch import Tensor


def compute_static_reaction_torque(
    omega: Tensor,          # (num_envs,) rotor speed (rad/s)
    k_Q: float,             # torque coefficient (N·m·s²/rad²)
    spin_axis: Tensor,      # (3,) unit vector in body-FRD frame
) -> Tensor:
    """Compute static reaction torque opposing rotor spin.

    τ_static = -k_Q * ω² * spin_axis

    Args:
        omega: Current rotor angular velocity (rad/s).
        k_Q: Torque coefficient (N·m·s²/rad²).
        spin_axis: Rotor spin axis unit vector in body-FRD frame.

    Returns:
        Tensor (num_envs, 3) — reaction torque in body-FRD (N·m).
    """
    Q_magnitude = k_Q * omega ** 2  # (num_envs,)
    return -spin_axis.unsqueeze(0) * Q_magnitude.unsqueeze(-1)


def compute_dynamic_spool_torque(
    omega: Tensor,          # (num_envs,) current rotor speed
    omega_prev: Tensor,     # (num_envs,) previous rotor speed
    rotor_inertia: float,   # kg·m²
    spin_axis: Tensor,      # (3,) unit vector
    dt: float,
) -> Tensor:
    """Compute dynamic torque from rotor angular acceleration.

    τ_dynamic = I_rotor * (dω/dt) * spin_axis

    Args:
        omega: Current rotor angular velocity (rad/s).
        omega_prev: Previous rotor angular velocity (rad/s).
        rotor_inertia: Rotor moment of inertia (kg·m²).
        spin_axis: Rotor spin axis unit vector in body-FRD frame.
        dt: Timestep (s).

    Returns:
        Tensor (num_envs, 3) — spool torque in body-FRD (N·m).
    """
    d_omega = (omega - omega_prev) / max(dt, 1e-8)  # (num_envs,)
    torque_magnitude = rotor_inertia * d_omega
    return spin_axis.unsqueeze(0) * torque_magnitude.unsqueeze(-1)


def compute_gyroscopic_precession(
    omega: Tensor,              # (num_envs,) rotor speed
    body_angular_vel: Tensor,   # (num_envs, 3) body angular velocity in body-FRD
    rotor_inertia: float,
    spin_axis: Tensor,          # (3,) unit vector
) -> Tensor:
    """Compute gyroscopic precession torque.

    τ_gyro = ω_body × H_rotor
    where H_rotor = I_rotor * ω_rotor * spin_axis (angular momentum vector)

    Args:
        omega: Rotor angular velocity (rad/s), shape (num_envs,).
        body_angular_vel: Body angular velocity in body-FRD (rad/s), shape (num_envs, 3).
        rotor_inertia: Rotor moment of inertia (kg·m²).
        spin_axis: Rotor spin axis unit vector in body-FRD frame.

    Returns:
        Tensor (num_envs, 3) — gyroscopic precession torque in body-FRD (N·m).
    """
    H_rotor = spin_axis.unsqueeze(0) * (rotor_inertia * omega).unsqueeze(-1)  # (num_envs, 3)
    return torch.linalg.cross(body_angular_vel, H_rotor)


def compute_all_rotor_torques(
    omega: Tensor,
    omega_prev: Tensor,
    body_angular_vel: Tensor,
    k_Q: float,
    rotor_inertia: float,
    spin_axis: Tensor,
    dt: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute all three rotor torque contributions.

    Returns:
        Tuple (static_reaction, dynamic_spool, gyro_precession),
        each of shape (num_envs, 3) in body-FRD frame (N·m).
    """
    static = compute_static_reaction_torque(omega, k_Q, spin_axis)
    dynamic = compute_dynamic_spool_torque(omega, omega_prev, rotor_inertia, spin_axis, dt)
    gyro = compute_gyroscopic_precession(omega, body_angular_vel, rotor_inertia, spin_axis)
    return static, dynamic, gyro
