import torch
from tvc_env.controllers.ppo_model import ActorCritic


def test_throttle_initialization_preserves_fin_policy_and_remains_trainable():
    model=ActorCritic(28)
    with torch.no_grad():model.actor[-1].weight.normal_()
    obs=torch.randn(32,28)
    before=model.act(obs).detach()
    model.initialize_throttle_prior(.79)
    after=model.act(obs)
    assert torch.equal(before[:,:4],after[:,:4])
    assert torch.allclose((after[:,4]+1)/2,torch.full((32,),.79))
    # No runtime override: an ordinary actor gradient changes throttle.
    after[:,4].sum().backward()
    with torch.no_grad():
        model.actor[-1].weight -= .01*model.actor[-1].weight.grad
    assert not torch.allclose(model.act(obs)[:,4],after[:,4])
