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


def compute_midpoint_gyroscopic_torque(omega: Tensor, body_angular_vel: Tensor,
        rotor_inertia: float, spin_axis: Tensor, body_inertia: Tensor, dt: float) -> Tensor:
    """Cayley/midpoint integration of the rotor's skew gyroscopic operator.

    Continuous torque remains H x w, at full physical magnitude. Forward
    Euler adds energy because its torque is perpendicular to OLD velocity,
    not midpoint velocity. Real Isaac test 18 (2026-09-15) measured 10.20x
    rotational energy after two unforced seconds at 240 Hz. Do not mask that
    numerical instability with damping or a reduced gyro scale.

    Solve (I - dt*[H]x/2) w_mid = I*w_old, then apply [H]x*w_mid. This
    conserves .5*w'I*w for the isolated gyro substep, even at high rotor RPM.
    PhysX separately integrates the locked-body rigid Euler term. Splitting
    error with other torques must still be checked by timestep refinement.
    """
    h = _coerce_spin_axis(spin_axis, body_angular_vel)[None] * (rotor_inertia*omega)[:,None]
    cross = body_angular_vel.new_zeros((omega.shape[0],3,3))
    cross[:,0,1]=-h[:,2]; cross[:,0,2]=h[:,1]
    cross[:,1,0]=h[:,2]; cross[:,1,2]=-h[:,0]
    cross[:,2,0]=-h[:,1]; cross[:,2,1]=h[:,0]
    inertia=body_inertia.to(body_angular_vel)
    momentum=torch.matmul(inertia,body_angular_vel.unsqueeze(-1))
    midpoint=torch.linalg.solve(inertia-.5*dt*cross,momentum).squeeze(-1)
    return torch.linalg.cross(h,midpoint)


def compute_coupled_midpoint_torques(body_angular_vel: Tensor, body_inertia: Tensor,
        rotor_momentum: Tensor, external_torque: Tensor, dt: float,
        correct_physx_projection: bool = False) -> tuple[Tensor, Tensor]:
    """Midpoint body-Euler + virtual-rotor operator for one external impulse.

    Solve I(w_mid-w_old) = dt/2 * (tau_ext - w_mid x (I w_mid + H)).
    PhysX already supplies -w_old x Iw_old. Return rotor gyro and the
    difference between midpoint and explicit locked-body Euler terms.
    This difference is an integration correction, NOT physical damping.

    Requires TGS enable_external_forces_every_iteration=False: repeating the
    old-world-frame impulse while rotating the body breaks this discrete
    update. Sept 15 Isaac test 18: native explicit body dynamics amplified
    transverse rate squared 114x at |r|=40 rad/s with the rotor stopped;
    rotor-only midpoint amplified it 1301x with a counter-rotating rotor.
    A combined midpoint with one external impulse reduced amplification to
    1.006x at 480Hz. Full articulation/timestep conservation tests remain
    required because the preconditioner omits the four small moving vanes.
    NVIDIA documents explicit articulation Coriolis integration:
    https://nvidia-omniverse.github.io/PhysX/physx/5.5.0/docs/Articulations.html
    """
    def cross_matrix(v):
        matrix = v.new_zeros((*v.shape[:-1], 3, 3))
        matrix[...,0,1] = -v[...,2]; matrix[...,0,2] = v[...,1]
        matrix[...,1,0] = v[...,2]; matrix[...,1,2] = -v[...,0]
        matrix[...,2,0] = -v[...,1]; matrix[...,2,1] = v[...,0]
        return matrix

    inertia = body_inertia.to(body_angular_vel)
    old = body_angular_vel
    midpoint = old.clone()
    for _ in range(4):
        iw = (inertia @ midpoint.unsqueeze(-1)).squeeze(-1)
        f = external_torque - torch.linalg.cross(midpoint, iw+rotor_momentum)
        residual = (inertia @ (midpoint-old).unsqueeze(-1)).squeeze(-1) - .5*dt*f
        derivative = cross_matrix(iw+rotor_momentum) - cross_matrix(midpoint) @ inertia
        midpoint = midpoint - torch.linalg.solve(inertia-.5*dt*derivative, residual.unsqueeze(-1)).squeeze(-1)
    iw = (inertia @ midpoint.unsqueeze(-1)).squeeze(-1)
    old_iw = (inertia @ old.unsqueeze(-1)).squeeze(-1)
    gyro = torch.linalg.cross(rotor_momentum, midpoint)
    correction = -torch.linalg.cross(midpoint, iw) + torch.linalg.cross(old, old_iw)
    if correct_physx_projection:
        # PhysX applies external acceleration, then its explicit body Euler
        # update, then preserves the PRE-INTERNAL angular momentum magnitude.
        # Inverting that scalar projection is necessary when H and body spin
        # interact. Otherwise p=q=6.28,r=40 at 480Hz gained 22% energy in 2s.
        # Let b=dt*I^-1*(-w_old x Iw_old), d=2*w_mid-w_old. Solve
        # |I*(lambda*d-b)|^2=|I*d|^2, then apply I*(lambda*d-b-w_old)/dt.
        # This is the documented PhysX discrete operator, not damping.
        # See computeLinkInternalAcceleration in DyFeatherstoneForwardDynamic.cpp.
        desired = 2*midpoint-old
        desired_l = (inertia @ desired.unsqueeze(-1)).squeeze(-1)
        ib = -dt*torch.linalg.cross(old, old_iw)
        a = desired_l.square().sum(-1)
        dot = (desired_l*ib).sum(-1)
        discriminant = dot.square()+a*(a-ib.square().sum(-1))
        scale = (dot+discriminant.clamp(min=0).sqrt())/a.clamp(min=1e-12)
        scale = torch.where(a>1e-12,scale,torch.ones_like(scale))
        correction = correction + (scale-1)[:,None]*desired_l/dt
    return gyro, correction


def cayley_body_orientation(old_quaternion, old_body_rate, new_body_rate, dt):
    """Lie midpoint orientation paired with the body angular-velocity solve.

    R_new=R_old*Cayley(dt*w_mid). For unforced constant H, midpoint Euler
    gives L_new-L_old=-dt*w_mid x (L_new+L_old)/2; the Cayley rotation thus
    preserves WORLD angular momentum as well as the velocity solve's energy.
    No preferred attitude, rate limit, controller or landing goal is involved.
    """
    from tvc_env.common.frames import frd_to_isaac
    from tvc_env.common.quaternions import normalize, multiply
    delta = torch.cat((torch.ones_like(old_body_rate[:,:1]),
                       .25*dt*frd_to_isaac(old_body_rate+new_body_rate)),dim=-1)
    return normalize(multiply(old_quaternion,normalize(delta)))
