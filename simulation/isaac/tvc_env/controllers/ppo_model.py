"""Feed-forward PPO model with a correctly transformed Gaussian policy."""
from __future__ import annotations

import math
from functools import lru_cache
import torch
from torch import nn
from torch.distributions import Normal


@lru_cache(maxsize=8)
def _hermite_rule(points):
    from numpy.polynomial.hermite import hermgauss
    return hermgauss(points)


def bounded_action_mean(mean: torch.Tensor, std: torch.Tensor, points: int = 20) -> torch.Tensor:
    """Deterministic expectation E[tanh(z)] of the learned Gaussian policy.

    Diagnostic: the 20M policy scored 139/512 with sampled actions versus
    3/512 with tanh(mean). Near throttle saturation tanh(mean) is the median,
    not the bounded distribution's expectation. Gauss-Hermite quadrature
    evaluates that expectation without introducing a controller or sampling.
    """
    nodes, weights = _hermite_rule(points)
    nodes = torch.as_tensor(nodes.copy(), device=mean.device, dtype=mean.dtype)
    weights = torch.as_tensor(weights.copy(), device=mean.device, dtype=mean.dtype)
    values = torch.tanh(mean.unsqueeze(-1) + math.sqrt(2.0) * std.unsqueeze(-1) * nodes)
    return (values * weights).sum(-1) / math.sqrt(math.pi)


def anneal_log_std(parameter, initial, target, progress):
    """Project learned log standard deviations onto a gradually decreasing cap.

    This changes exploration during learning, not the commanded mean. Initial
    and final caps are explicit checkpointed algorithm parameters.
    """
    if not torch.all(target <= initial):
        raise ValueError('Exploration annealing must not increase its cap')
    cap = torch.lerp(initial, target, min(1., max(0., float(progress))))
    with torch.no_grad():
        parameter.copy_(torch.minimum(parameter, cap))
    return cap


def tanh_log_prob(dist: Normal, latent: torch.Tensor) -> torch.Tensor:
    """Stable log density of tanh(latent), including its Jacobian."""
    log_jacobian = 2.0 * (math.log(2.0) - latent - torch.nn.functional.softplus(-2.0 * latent))
    return (dist.log_prob(latent) - log_jacobian).sum(-1)


def policy_kl(old_mean, old_log_std, new_mean, new_log_std):
    """Exact KL(old || new) for diagonal Gaussians (also their tanh transforms).

    The shared invertible tanh map leaves KL unchanged. Computing this from
    distribution parameters avoids sampling noise in the PPO update guard.
    """
    return (new_log_std - old_log_std
            + .5 * ((2 * (old_log_std - new_log_std)).exp()
                    + (old_mean - new_mean).square() * (-2 * new_log_std).exp() - 1)).sum(-1)


def value_loss(values, returns, old_values=None, clip_range=None):
    """Value regression with optional clipping in return units, not ratio units."""
    error = (values - returns).square()
    if clip_range is not None:
        if clip_range <= 0 or old_values is None:
            raise ValueError('Value clipping requires a positive range and old values')
        clipped = old_values + (values - old_values).clamp(-clip_range, clip_range)
        error = torch.maximum(error, (clipped - returns).square())
    return .5 * error.mean()


def make_optimizer(model, actor_lr, critic_lr=None, saved_state=None):
    """Independent actor/critic learning rates, preserving legacy Adam moments.

    Radial transfer at 4.06M: explained variance was -0.0004 and the fresh
    tanh critic's absolute output bound was only 20.8 versus +475 terminals.
    A conservative actor LR need not prevent the critic fitting return units.
    This changes optimization only; rewards and inference remain unchanged.
    """
    critic_lr = actor_lr if critic_lr is None else critic_lr
    if not all(math.isfinite(x) and x > 0 for x in (actor_lr, critic_lr)):
        raise ValueError('Actor and critic learning rates must be positive and finite')
    groups = [dict(params=list(model.actor.parameters()) + [model.log_std], name='actor', lr=actor_lr),
              dict(params=list(model.critic.parameters()), name='critic', lr=critic_lr)]
    if saved_state is not None and len(saved_state['param_groups']) == 1:
        optimizer = torch.optim.Adam(model.parameters(), lr=actor_lr, eps=1e-5)
        optimizer.load_state_dict(saved_state)
        # Existing state is keyed by Parameter objects, so replacing groups
        # keeps each parameter's exp_avg, exp_avg_sq and step exactly intact.
        options = {k: v for k, v in optimizer.param_groups[0].items() if k != 'params'}
        optimizer.param_groups.clear()
        for group in groups:
            optimizer.add_param_group({**options, **group})
    else:
        optimizer = torch.optim.Adam(groups, eps=1e-5)
        if saved_state is not None:
            if [g.get('name') for g in saved_state['param_groups']] != ['actor', 'critic']:
                raise ValueError('Unrecognized saved optimizer group layout')
            optimizer.load_state_dict(saved_state)
            for group, lr in zip(optimizer.param_groups, (actor_lr, critic_lr)):
                group['lr'] = lr
    return optimizer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 24, act_dim: int = 5, throttle_bias: float = 0.0):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, act_dim))
        self.critic = nn.Sequential(nn.Linear(obs_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        with torch.no_grad():
            self.actor[-1].bias[4] = throttle_bias
        init_log_std = torch.full((act_dim,), -1.0)
        # Review v2 lost all stage successes by 0.84M steps. The old initial
        # fin sigma (.262*exp(-2) = .035 rad) applied ~0.45 N m per roll/pitch
        # channel before any feedback was learned. Start at .013 rad (~.75 deg)
        # for this low-inertia vehicle; all four means/stds remain trainable.
        init_log_std[:4] = -3.0
        self.log_std = nn.Parameter(init_log_std)

    def initialize_actor(self, source, fin_permutation=None):
        """Initialize a new task without a runtime action wrapper.

        Optional channel permutation is an explicit physical actuator-coordinate
        change. Extra battery/waypoint input columns start at zero, preserving the
        parent's response initially while allowing PPO to learn their weights.
        Critic, optimizer and training counters are intentionally not copied.
        """
        actor = {k: v.clone() for k, v in source.items() if k.startswith('actor.') or k == 'log_std'}
        old = actor['actor.0.weight']
        target = self.actor[0].weight
        if old.shape != target.shape:
            if old.shape[0] != target.shape[0] or (old.shape[1],target.shape[1]) not in ((24,28),(28,43)):
                raise ValueError('Only explicit 24-to-28 battery or 28-to-43 waypoint observation extension is supported')
            expanded = torch.zeros_like(target)
            expanded[:, :old.shape[1]] = old
            actor['actor.0.weight'] = expanded
        if fin_permutation is not None:
            if sorted(fin_permutation) != list(range(4)):
                raise ValueError('Fin permutation must contain 0,1,2,3 exactly once')
            order = [*fin_permutation, 4]
            for key in ('actor.4.weight', 'actor.4.bias', 'log_std'):
                actor[key] = actor[key][order]
        result = self.load_state_dict(actor, strict=False)
        if result.unexpected_keys or any(not k.startswith('critic.') for k in result.missing_keys):
            raise ValueError('Incompatible actor checkpoint')

    def forward(self, obs):
        return self.actor(obs), self.critic(obs).squeeze(-1)

    def initialize_throttle_prior(self, duty: float):
        """One-time action initialization, never an inference controller.

        Momentum transfer diagnostic at 2.097M: 510/512 stage-1 crashes.
        Initial throttle ~0.001 at 2-4 m produced 8-18 rad/s spool-down yaw
        and insufficient braking distance. Its saturated inherited head and
        narrow distribution almost never sampled hover at those states.
        Reset only this head to a physical hover prior; all weights remain
        trainable, and PPO must learn descent/burn timing from the reward.
        """
        if not math.isfinite(duty) or not 0 < duty < 1:
            raise ValueError('Initial throttle prior must be between zero and one')
        with torch.no_grad():
            self.actor[-1].weight[4].zero_()
            self.actor[-1].bias[4] = math.atanh(2*duty-1)

    def initialize_exploration(self, fin_log_std: float, throttle_log_std: float):
        """Reset transferred actor variance while preserving its mean policy.

        The 132.12M waypoint run inherited/learned latent sigmas of only
        0.039 on fins and 0.070 on throttle, then remained at 6.8% stage
        success for 120.85M transitions.  A fresh task needs enough action
        support to discover waypoint capture and yaw cancellation.  This is
        an initialization prior; PPO continues training every log-std value.
        """
        values = (float(fin_log_std), float(throttle_log_std))
        if any(not math.isfinite(value) or value < -8.0 or value > 1.0 for value in values):
            raise ValueError('Initial log standard deviations must be finite and within [-8, 1]')
        with torch.no_grad():
            self.log_std[:4].fill_(values[0])
            self.log_std[4] = values[1]

    def act(self, obs, mode='deterministic'):
        """Inference with an explicit, recorded action-distribution mode."""
        mean = self.actor(obs)
        if mode == 'mean':
            return bounded_action_mean(mean, self.log_std.exp())
        if mode == 'stochastic':
            return torch.tanh(mean + torch.randn_like(mean) * self.log_std.exp())
        if mode == 'deterministic':
            return torch.tanh(mean)
        raise ValueError(f'Unknown PPO action mode: {mode}')

    def get_action_and_value(self, obs, action_raw=None, deterministic=False, *, latent_action=None, return_latent=False):
        mean, value = self(obs)
        dist = Normal(mean, self.log_std.exp().expand_as(mean))
        if latent_action is not None:
            latent = latent_action
            action_raw = torch.tanh(latent)
        elif action_raw is None:
            latent = mean if deterministic else dist.rsample()
            action_raw = torch.tanh(latent)
        else:
            # Rollout actions are constants during the PPO update.
            latent = torch.atanh(action_raw.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        log_prob = tanh_log_prob(dist, latent)
        # Normal.entropy() is entropy BEFORE tanh. Maximizing it can push
        # bounded actuators into saturation (review run: entropy rose steadily
        # while stage success fell to zero). Use a reparameterized estimate of
        # the actual bounded distribution, also when evaluating stored actions.
        entropy = -tanh_log_prob(dist, dist.rsample())
        result = (action_raw, log_prob, entropy, value)
        return (*result, latent) if return_latent else result
