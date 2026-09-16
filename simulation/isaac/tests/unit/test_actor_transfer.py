import torch
from tvc_env.controllers.ppo_model import ActorCritic


def test_explicit_actuator_coordinate_transfer_preserves_parent_function():
    torch.manual_seed(2)
    parent = ActorCritic(24)
    with torch.no_grad():
        parent.actor[-1].weight.normal_(0,.05)
        parent.actor[-1].bias.normal_()
    child = ActorCritic(28)
    critic_before = {k:v.clone() for k,v in child.critic.state_dict().items()}
    child.initialize_actor(parent.state_dict(), [3,0,1,2])
    old = torch.randn(8,24)
    extended = torch.cat([old,torch.randn(8,4)],dim=1)
    torch.testing.assert_close(child.actor(extended),parent.actor(old)[:,[3,0,1,2,4]])
    torch.testing.assert_close(child.log_std,parent.log_std[[3,0,1,2,4]])
    for key,value in critic_before.items():
        torch.testing.assert_close(value,child.critic.state_dict()[key])
    child.actor(extended).sum().backward()
    assert child.actor[0].weight.grad[:,24:].abs().sum() > 0


def test_transferred_actor_exploration_can_be_reinitialized():
    model = ActorCritic(43)
    mean_before = {key: value.clone() for key, value in model.actor.state_dict().items()}
    model.initialize_exploration(-2.3, -1.7)
    torch.testing.assert_close(model.log_std, torch.tensor([-2.3] * 4 + [-1.7]))
    for key, value in mean_before.items():
        torch.testing.assert_close(value, model.actor.state_dict()[key])
