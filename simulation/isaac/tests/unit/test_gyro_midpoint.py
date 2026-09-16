import pytest
import torch
from tvc_env.dynamics.rotor_reaction import compute_midpoint_gyroscopic_torque


@pytest.mark.parametrize('dt',[1/120,1/240,1/480])
def test_midpoint_preserves_rotational_energy_at_full_rotor_momentum(dt):
    inertia=torch.diag(torch.tensor([.05,.06,.02],dtype=torch.float64))[None]
    rate=torch.tensor([[.2,-.1,.3]],dtype=torch.float64)
    initial=.5*torch.einsum('bi,bij,bj->b',rate,inertia,rate)
    for _ in range(round(2/dt)):
        torque=compute_midpoint_gyroscopic_torque(torch.tensor([4300.],dtype=torch.float64),rate,.0002,
            torch.tensor([0.,0.,1.],dtype=torch.float64),inertia,dt)
        rate=rate+dt*torch.linalg.solve(inertia,torque.unsqueeze(-1)).squeeze(-1)
    final=.5*torch.einsum('bi,bij,bj->b',rate,inertia,rate)
    assert torch.allclose(initial,final,atol=1e-13,rtol=1e-12)


def test_midpoint_recovers_continuous_sign_in_small_step_limit():
    rate=torch.tensor([[1.,2.,3.]],dtype=torch.float64)
    axis=torch.tensor([0.,0.,1.],dtype=torch.float64)
    torque=compute_midpoint_gyroscopic_torque(torch.tensor([4000.],dtype=torch.float64),rate,.0002,axis,
        torch.eye(3,dtype=torch.float64)[None]*.05,1e-9)
    assert torch.allclose(torque,torch.tensor([[-1.6,.8,0.]],dtype=torch.float64),atol=1e-7)


@pytest.mark.parametrize('rotor_momentum', [0., .84])
def test_coupled_cayley_preserves_energy_and_world_momentum_at_adverse_rates(rotor_momentum):
    from tvc_env.dynamics.rotor_reaction import compute_coupled_midpoint_torques, cayley_body_orientation
    from tvc_env.common.frames import frd_to_isaac
    from tvc_env.common.quaternions import rotate_vector
    inertia = torch.diag(torch.tensor([.05, .06, .02], dtype=torch.float64))[None]
    rate = torch.tensor([[6.28, 6.28, 40.], [6.28, -6.28, -40.]], dtype=torch.float64)
    q = torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64).repeat(2, 1)
    h = rate.new_tensor([[0., 0., rotor_momentum]]).expand_as(rate)
    energy = lambda w: .5*torch.einsum('bi,bij,bj->b', w, inertia.expand(2,-1,-1), w)
    world_h = lambda w, pose: rotate_vector(pose, frd_to_isaac((inertia @ w[...,None]).squeeze(-1)+h))
    initial_energy, initial_h = energy(rate), world_h(rate, q)
    dt = 1/480
    for _ in range(240):
        gyro, correction = compute_coupled_midpoint_torques(rate, inertia, h, torch.zeros_like(rate), dt)
        iw = (inertia @ rate[...,None]).squeeze(-1)
        torque = gyro + correction - torch.linalg.cross(rate, iw)
        new_rate = rate + dt*torch.linalg.solve(inertia, torque[...,None]).squeeze(-1)
        q = cayley_body_orientation(q, rate, new_rate, dt)
        rate = new_rate
    assert torch.allclose(energy(rate), initial_energy, rtol=1e-10, atol=1e-10)
    assert torch.allclose(world_h(rate, q), initial_h, rtol=1e-10, atol=1e-10)
