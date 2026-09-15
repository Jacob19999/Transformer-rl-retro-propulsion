"""Optimizer migration must retain learned Adam state and update semantics."""
import copy
import torch
from tvc_env.controllers.ppo_model import ActorCritic, make_optimizer


def step(model, optimizer, obs):
    optimizer.zero_grad()
    actor, value = model(obs)
    (actor.square().mean() + value.square().mean() + model.log_std.square().mean()).backward()
    optimizer.step()


def test_legacy_adam_migration_preserves_next_update_and_moments():
    torch.manual_seed(2)
    original = ActorCritic(28)
    optimizer = torch.optim.Adam(original.parameters(), lr=3e-5, eps=1e-5)
    obs = torch.randn(12, 28)
    step(original, optimizer, obs)
    restored = copy.deepcopy(original)
    migrated = make_optimizer(restored, 3e-5, saved_state=copy.deepcopy(optimizer.state_dict()))
    for a, b in zip(original.parameters(), restored.parameters()):
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            assert torch.equal(optimizer.state[a][key], migrated.state[b][key])
    step(original, optimizer, obs)
    step(restored, migrated, obs)
    for a, b in zip(original.parameters(), restored.parameters()):
        assert torch.equal(a, b)
    resumed = make_optimizer(restored, 3e-5, 1e-3, copy.deepcopy(migrated.state_dict()))
    assert [g['lr'] for g in resumed.param_groups] == [3e-5, 1e-3]
    for p in restored.parameters():
        assert torch.equal(resumed.state[p]['exp_avg'], migrated.state[p]['exp_avg'])
