"""
Rotor reaction torque computation.

All torques returned by this module act ON THE BODY (Newton's third law applied
where appropriate), so callers can sum them directly into the body wrench.

Computes:
  - Static reaction torque: -k_Q * omega^2 * spin_axis (opposes rotor spin)
  - Dynamic spool reaction torque: -I_rotor * d_omega/dt * spin_axis
  - Gyroscopic precession on body: -(omega_body x H_rotor)
    (PhysX integrates I_body * alpha = tau_applied; the rotor's angular
    momentum is not part of the articulation, so we apply the
    reaction -ω x H as an external torque on the body.)

All outputs are separate vec3 tensors for independent logging per FR-018.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _coerce_spin_axis(spin_axis: Tensor, reference: Tensor) -> Tensor:
    """Place the spin axis on the same device and dtype as the reference tensor."""
    return spin_axis.to(device=reference.device, dtype=reference.dtype)


def compute_static_reaction_torque(
    omega: Tensor,          # (num_envs,) rotor speed (rad/s)
    k_Q: float,             # torque coefficient (N*m*s^2/rad^2)
    spin_axis: Tensor,      # (3,) unit vector in body-FRD frame
) -> Tensor:
    """Compute static reaction torque opposing rotor spin."""
    spin_axis = _coerce_spin_axis(spin_axis, omega)
    q_magnitude = k_Q * omega ** 2  # (num_envs,)
    return -spin_axis.unsqueeze(0) * q_magnitude.unsqueeze(-1)


def compute_dynamic_spool_torque(
    omega: Tensor,          # (num_envs,) current rotor speed
    omega_prev: Tensor,     # (num_envs,) previous rotor speed
    rotor_inertia: float,   # kg*m^2
    spin_axis: Tensor,      # (3,) unit vector
    dt: float,
) -> Tensor:
    """Compute body reaction torque from rotor angular acceleration."""
    spin_axis = _coerce_spin_axis(spin_axis, omega)
    d_omega = (omega - omega_prev) / max(dt, 1e-8)  # (num_envs,)
    torque_magnitude = rotor_inertia * d_omega
    return -spin_axis.unsqueeze(0) * torque_magnitude.unsqueeze(-1)


def compute_gyroscopic_precession(
    omega: Tensor,              # (num_envs,) rotor speed
    body_angular_vel: Tensor,   # (num_envs, 3) body angular velocity in body-FRD
    rotor_inertia: float,
    spin_axis: Tensor,          # (3,) unit vector
) -> Tensor:
    """Compute gyroscopic precession torque ON THE BODY.

    The torque required to precess the rotor is ``omega_body x H_rotor``
    (applied by the body to the rotor). By Newton's third law, the rotor
    exerts the negative of that on the body. PhysX does not see the virtual
    rotor's angular momentum, so we apply ``-(omega_body x H_rotor)`` as an
    external body torque to close the books.
    """
    omega = omega.to(device=body_angular_vel.device, dtype=body_angular_vel.dtype)
    spin_axis = _coerce_spin_axis(spin_axis, body_angular_vel)
    h_rotor = spin_axis.unsqueeze(0) * (rotor_inertia * omega).unsqueeze(-1)  # (num_envs, 3)
    return -torch.linalg.cross(body_angular_vel, h_rotor)


def compute_all_rotor_torques(
    omega: Tensor,
    omega_prev: Tensor,
    body_angular_vel: Tensor,
    k_Q: float,
    rotor_inertia: float,
    spin_axis: Tensor,
    dt: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute all three rotor torque contributions."""
    static = compute_static_reaction_torque(omega, k_Q, spin_axis)
    dynamic = compute_dynamic_spool_torque(omega, omega_prev, rotor_inertia, spin_axis, dt)
    gyro = compute_gyroscopic_precession(omega, body_angular_vel, rotor_inertia, spin_axis)
    return static, dynamic, gyro
