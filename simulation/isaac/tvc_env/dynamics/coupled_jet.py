"""Finite-momentum EDF wake and passive jet vanes, all in body FRD.

Reduced-order control-volume model, not CFD or a fitted fan map. Four equal
streamtubes share mass flux T/U; a vane removes a fraction of velocity normal
to its *actual* plane. Exponential attenuation matches the small-loading lift
slope but cannot extract more transverse momentum than enters its streamtube.
This fixes the 2026-09-14 audit: independent q*S*CNa forces exceeded the jet's
turning momentum, and angular damping .27 was an artificial stabilizer.

Swirl and reaction use the SAME angular-momentum flux, Q=mdot*r*vtheta.
Reference: Drela, QPROP Formulation (MIT, 2006), sections 1 and 3:
https://web.mit.edu/drela/Public/web/qprop/qprop_theory.pdf
Equal streamtube mass and residual stator torque are explicit model estimates.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class JetOutput:
    forces: Tensor
    reaction_torque: Tensor
    incoming_velocity: Tensor
    outgoing_velocity: Tensor
    mass_flow_per_fin: Tensor
    dissipated_power: Tensor
    swirl_power: Tensor


class CoupledJet:
    def __init__(self, config: dict, aero, normals: Tensor):
        self.aero = aero
        self.normals = normals
        self.residual_fraction = float(config.get('residual_swirl_fraction', .1))
        self.shaft_power = float(config.get('shaft_power_at_max_w', 3072.))
        self.exit_position = normals.new_tensor(config.get('exit_position_frd', [0, 0, .05]))
        if not 0 <= self.residual_fraction <= 1 or self.shaft_power <= 0:
            raise ValueError('Invalid swirl fraction or shaft power')

    def compute(self, angles: Tensor, fraction: Tensor, omega: Tensor,
                raw_thrust: Tensor, cops: Tensor, relative_cop_velocity: Tensor,
                body_rate: Tensor) -> JetOutput:
        """COP velocity is relative to translating Body origin, expressed FRD.

        Jet translation follows its moving nozzle. The frozen convected-wake
        approximation retains nozzle velocity and subtracts the actual COP
        velocity, including articulation motion. This includes rotation over
        nozzle-to-vane distance without subtracting vehicle translation twice.
        """
        # Shaft torque that the stator has not recovered. Spool I*domega/dt
        # remains a separate reaction; it must not be multiplied by this ratio.
        residual_q = self.residual_fraction * self.shaft_power * fraction.pow(3) / omega.clamp(min=1e-8)
        radial = cops.clone()
        radial[..., 2] = 0
        mean_r2 = radial.square().sum(-1).mean(-1).clamp(min=1e-8)
        # Share the ideal axial wake power with swirl instead of adding free
        # swirl energy: P=.5*T*U + .5*Q^2*U/(T*<r^2>), mdot=T/U.
        u = self.aero.exhaust_speed * fraction / (1 + residual_q.square() /
            (raw_thrust.square() * mean_r2).clamp(min=1e-12))
        mdot = raw_thrust / u.clamp(min=1e-8)
        flux = mdot[:, None, None] / 4
        swirl_rate = residual_q / (mdot * mean_r2).clamp(min=1e-8)
        axis = torch.zeros_like(radial)
        axis[..., 2] = swirl_rate[:, None]
        swirl = torch.linalg.cross(axis, radial)
        incoming = swirl + torch.linalg.cross(body_rate, self.exit_position.expand_as(body_rate))[:, None] - relative_cop_velocity
        incoming[..., 2] += u[:, None]
        normal = self.normals[None] * angles.cos()[..., None]
        normal[..., 2] -= angles.sin()
        chord = self.normals[None] * angles.sin()[..., None]
        chord[..., 2] += angles.cos()
        # Integrating d(v_normal)/dx=-k*v_normal yields 1-exp(-loading).
        # At low loading this recovers q*S*CNa*alpha, at high loading it
        # approaches alignment with the vane instead of reversing the flow.
        q_area = .5 * self.aero.air_density * u.square() * self.aero.fin_area * self.aero.duct_confinement_factor
        loading = q_area * self.aero.C_N_alpha / (raw_thrust / 4).clamp(min=1e-8)
        turn = -torch.expm1(-loading)
        normal_speed = (incoming * normal).sum(-1)
        after_turn = incoming - turn[:, None, None] * normal_speed[..., None] * normal
        along = (after_turn * chord).sum(-1)
        # Profile drag removes streamwise momentum once. Incidence-induced
        # loss already follows the normal projection; do not add CD_alpha2.
        drag_loading = q_area * self.aero.C_D_0 / (raw_thrust / 4).clamp(min=1e-8)
        drag = -torch.expm1(-drag_loading)
        outgoing = after_turn - drag[:, None, None] * along[..., None] * chord
        forces = flux * (incoming - outgoing)
        reaction = torch.zeros_like(body_rate)
        reaction[:, 2] = -residual_q
        dissipation = (.5 * flux[..., 0] * (incoming.square().sum(-1) - outgoing.square().sum(-1))).sum(-1)
        swirl_power = (.5 * flux[..., 0] * swirl.square().sum(-1)).sum(-1)
        return JetOutput(forces, reaction, incoming, outgoing, flux[..., 0].expand(-1, 4), dissipation, swirl_power)


def cylinder_rotational_drag(rate: Tensor, length: float, diameter: float,
                             cd: float = 1., rho: float = 1.225) -> Tensor:
    """Still-air slender-cylinder crossflow integral; no tuned linear damping.

    dF=.5*rho*Cd*D*|omega*z|*(omega*z) dz integrated over [-L/2,L/2].
    Axial spin has no crossflow; omitted skin friction needs measurement.
    """
    result = torch.zeros_like(rate)
    transverse = rate[:, :2]
    result[:, :2] = -(rho * cd * diameter * length**4 / 64) * transverse.norm(dim=-1, keepdim=True) * transverse
    return result
