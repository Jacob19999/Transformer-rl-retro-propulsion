import torch
from tvc_env.sim.wrench_dispatch import WrenchDispatch


class Recorder:
    def apply_fin_forces_at_cop(self,*args):self.fins=args
    def apply_body_wrench(self,force,torque,**kwargs):self.body=(force,torque,kwargs)


def test_both_dispatch_paths_apply_resultant_at_body_origin():
    q=torch.tensor([[1.,0,0,0]]);p=torch.tensor([[7.,3.,5.]])
    force=torch.tensor([[[1.,2.,3.]]*4]);cops=torch.tensor([[.04,0,.1]]*4)
    for mode in ['per_link_force','collapsed_body_wrench']:
        recorder=Recorder();dispatch=WrenchDispatch(mode,link_force_interface=recorder)
        dispatch.dispatch(force,cops,q,p,torch.tensor([[0.,0,-48.]]),torch.zeros(1,3))
        assert torch.equal(recorder.body[2]['position_world'],p)
        if mode=='collapsed_body_wrench':
            expected=torch.linalg.cross(cops,force[0]).sum(0)*torch.tensor([1.,-1.,-1.])
            assert torch.allclose(recorder.body[1][0],expected)
