"""Torch-batched fin aerodynamics matching simulation.dynamics.fin_model."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class FinAeroParams:
    cl_alpha: float
    cd0: float
    aspect_ratio: float
    stall_angle: float
    max_deflection: float
    planform_area: float
    v_exhaust_nominal: float
    omega_fan_max: float
    exhaust_velocity_ratio: bool


def _rodrigues_batch(
    k: torch.Tensor,       # (1, 4, 3) or (4, 3) — per-fin hinge axes (unit)
    v: torch.Tensor,       # (1, 4, 3) or (4, 3) — vectors to rotate
    theta: torch.Tensor,   # (N, 4) — rotation angles
) -> torch.Tensor:
    """Rodrigues' rotation: rotate *v* by *theta* around *k*, batched.

    Returns shape (N, 4, 3).
    """
    cos_t = theta.unsqueeze(-1)    # (N, 4, 1)
    sin_t = theta.sin().unsqueeze(-1)
    cos_t_val = theta.cos().unsqueeze(-1)

    k3 = k if k.ndim == 3 else k.unsqueeze(0)   # (1, 4, 3)
    v3 = v if v.ndim == 3 else v.unsqueeze(0)    # (1, 4, 3)

    k_cross_v = torch.linalg.cross(
        k3.expand(theta.shape[0], -1, -1),
        v3.expand(theta.shape[0], -1, -1),
    )  # (N, 4, 3)
    k_dot_v = (k3 * v3).sum(-1, keepdim=True)  # (1, 4, 1) or (N, 4, 1)

    return v3 * cos_t_val + k_cross_v * sin_t + k3 * k_dot_v * (1.0 - cos_t_val)


def compute_fin_forces_body(
    *,
    delta_rad: torch.Tensor,  # (N, 4)
    omega_fan: torch.Tensor,  # (N,)
    rho: torch.Tensor,  # (N,)
    lift_dirs_body: torch.Tensor,  # (4, 3)
    drag_dirs_body: torch.Tensor,  # (4, 3)
    hinge_axes_body: torch.Tensor | None = None,  # (4, 3) — enables cos/sin decomposition
    params: FinAeroParams,
) -> torch.Tensor:
    """Compute per-fin force vectors in body frame, shape (N, 4, 3).

    When *hinge_axes_body* is provided, lift and drag directions are rotated
    by the mechanical deflection δ around each fin's hinge axis (Rodrigues'
    formula).  This captures the cos(δ)/sin(δ) coupling where lift "leaks"
    into thrust loss at large deflections.
    """
    if delta_rad.ndim != 2 or delta_rad.shape[1] != 4:
        raise ValueError(f"delta_rad must be (N,4), got {tuple(delta_rad.shape)}")
    if omega_fan.ndim != 1 or rho.ndim != 1:
        raise ValueError("omega_fan and rho must be 1D tensors.")
    if omega_fan.shape[0] != delta_rad.shape[0] or rho.shape[0] != delta_rad.shape[0]:
        raise ValueError("Batch dimensions must match.")

    delta = delta_rad.clamp(-params.max_deflection, params.max_deflection)
    alpha_eff = params.stall_angle * torch.tanh(delta / max(params.stall_angle, 1e-6))
    c_l = params.cl_alpha * alpha_eff
    c_d = params.cd0 + (c_l.square() / (torch.pi * max(params.aspect_ratio, 1e-6)))

    if params.exhaust_velocity_ratio:
        omega_ratio = (omega_fan / max(params.omega_fan_max, 1e-6)).clamp(0.0, 1.0)
        v_ex = params.v_exhaust_nominal * omega_ratio
    else:
        v_ex = torch.full_like(omega_fan, float(params.v_exhaust_nominal))

    q_dyn = 0.5 * rho * v_ex.square() * params.planform_area  # (N,)

    # Rotate lift/drag basis by mechanical deflection when hinge axes known.
    if hinge_axes_body is not None:
        nL = _rodrigues_batch(hinge_axes_body, lift_dirs_body, delta)   # (N, 4, 3)
        nD = _rodrigues_batch(hinge_axes_body, drag_dirs_body, delta)   # (N, 4, 3)
    else:
        nL = lift_dirs_body.unsqueeze(0)  # (1, 4, 3)
        nD = drag_dirs_body.unsqueeze(0)  # (1, 4, 3)

    lift = c_l.unsqueeze(-1) * nL   # (N, 4, 3)
    drag = c_d.unsqueeze(-1) * nD   # (N, 4, 3)
    forces = q_dyn.view(-1, 1, 1) * (lift + drag)
    return forces
