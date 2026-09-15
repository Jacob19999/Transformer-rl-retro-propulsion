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
