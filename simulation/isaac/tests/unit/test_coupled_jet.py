"""Conservation/passivity regressions for the 2026-09-14 wake revision."""
import torch
import pytest
from tvc_env.dynamics.coupled_jet import CoupledJet, cylinder_rotational_drag
from tvc_env.dynamics.fin_aero import FinAeroModel


def scenario(angles=None, fraction=1., residual=.1, rate=None, velocity=None):
    cops = torch.tensor([[[.04,0,.10],[0,.04,.10],[-.04,0,.10],[0,-.04,.10]]], dtype=torch.float64)
    normals = torch.tensor([[0,-1,0],[1,0,0],[0,1,0],[-1,0,0]], dtype=torch.float64)
    aero = FinAeroModel(.002,.262,exhaust_speed=128.)
    model = CoupledJet({'residual_swirl_fraction':residual}, aero, normals)
    angles = torch.tensor([angles or [0]*4], dtype=torch.float64)
    speed = torch.tensor([fraction], dtype=torch.float64)
    rate = torch.tensor([rate or [0]*3], dtype=torch.float64)
    vel = torch.linalg.cross(rate[:,None].expand(-1,4,-1), cops) if velocity is None else velocity
    return model.compute(angles,speed,speed*4649.56,48*speed.square(),cops,vel,rate), cops


def test_swirl_reaction_closes_angular_momentum():
    out, cops = scenario()
    incoming_angular_flux = torch.linalg.cross(cops, out.mass_flow_per_fin[...,None]*out.incoming_velocity).sum(1)
    assert torch.allclose(incoming_angular_flux + out.reaction_torque, torch.zeros_like(incoming_angular_flux), atol=1e-12)
    # Including vanes, body moment equals negative outgoing angular flux.
    body = out.reaction_torque + torch.linalg.cross(cops,out.forces).sum(1)
    outgoing = torch.linalg.cross(cops,out.mass_flow_per_fin[...,None]*out.outgoing_velocity).sum(1)
    assert torch.allclose(body+outgoing,torch.zeros_like(body),atol=1e-12)


@pytest.mark.parametrize('fraction',[0.,.1,.5,1.])
def test_wake_power_includes_swirl_without_free_energy(fraction):
    out,_ = scenario(fraction=fraction)
    power=(.5*out.mass_flow_per_fin*out.incoming_velocity.square().sum(-1)).sum()
    assert float(power) == pytest.approx(3072*fraction**3,abs=1e-9)


def test_vanes_are_passive_and_share_finite_momentum():
    for residual in [0.,.1,.2,1.]:
        out,_=scenario([.262,-.21,.08,-.15],residual=residual,rate=[1,-2,3])
        assert (out.dissipated_power >= 0).all()
        assert (out.outgoing_velocity.norm(dim=-1) <= out.incoming_velocity.norm(dim=-1)+1e-10).all()
        assert torch.allclose(out.forces,out.mass_flow_per_fin[...,None]*(out.incoming_velocity-out.outgoing_velocity))
    out,_=scenario([.262]*4,residual=0)
    # Each vane shares 12N of axial momentum. It cannot extract the old
    # >15N sideforce from this streamtube at only 15 degrees of deflection.
    assert out.forces[0,:,:2].norm(dim=-1).max() < 12*torch.sin(torch.tensor(.262))


def test_no_jet_no_directional_authority_or_reaction():
    out,_=scenario([.2]*4,fraction=0,rate=[2,3,4])
    assert torch.count_nonzero(out.forces)==0
    assert torch.count_nonzero(out.reaction_torque)==0
    assert out.dissipated_power.item()==0


@pytest.mark.parametrize('axis,angles',[(0,[-.1,0,.1,0]),(1,[0,-.1,0,.1]),(2,[.1]*4)])
def test_radial_vanes_have_correct_positive_axis_authority(axis,angles):
    out,cops=scenario(angles,residual=0)
    moment=torch.linalg.cross(cops,out.forces).sum(1)[0]
    assert moment[axis]>0
    assert moment[torch.arange(3)!=axis].abs().max()<1e-10


def test_local_motion_creates_damping_without_tuned_linear_torque():
    for axis in range(3):
        rate=[0.,0.,0.];rate[axis]=1.
        out,cops=scenario(residual=0,rate=rate)
        moment=torch.linalg.cross(cops,out.forces).sum(1)[0]
        assert moment[axis]<0
    rates=torch.tensor([[1.,-2.,3.],[-1.,2.,-3.]])
    drag=cylinder_rotational_drag(rates,.35,.12)
    assert ((drag*rates).sum(-1)<0).all()
    assert drag.abs().max()<.001  # geometry-derived, not the old .27 Nms/rad
