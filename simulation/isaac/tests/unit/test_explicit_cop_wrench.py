from types import SimpleNamespace
import torch
from tvc_env.sim.link_force_interface import LinkForceInterface


class Composer:
    def set_forces_and_torques(self,**kwargs):self.kwargs=kwargs


def test_off_com_force_adds_to_external_torque_without_position_kernel():
    composer=Composer()
    art=SimpleNamespace(device='cpu',instantaneous_wrench_composer=composer,
        data=SimpleNamespace(body_com_pos_w=torch.tensor([[[.01,0.,.01]]])))
    iface=LinkForceInterface(art,None,torch.zeros(4,3))
    iface.apply_body_wrench(torch.tensor([[0.,0.,48.]]),torch.tensor([[.2,.3,.4]]),0,torch.zeros(1,3))
    assert 'positions' not in composer.kwargs
    assert torch.allclose(composer.kwargs['torques'],torch.tensor([[[.2,.78,.4]]]))
