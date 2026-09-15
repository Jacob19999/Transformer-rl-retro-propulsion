import torch
from torch.distributions import Normal, TransformedDistribution, TanhTransform

from tvc_env.controllers.ppo_model import ActorCritic, tanh_log_prob, value_loss, bounded_action_mean


def test_log_density_matches_torch_transformed_distribution():
    dist = Normal(torch.tensor([[0.2, -0.5]]), torch.tensor([[0.3, 0.7]]))
    z = torch.tensor([[0.8, -1.2]])
    reference = TransformedDistribution(dist, [TanhTransform()]).log_prob(z.tanh()).sum(-1)
    torch.testing.assert_close(tanh_log_prob(dist, z), reference)


def test_saturation_reduces_bounded_entropy():
    torch.manual_seed(1)
    noise = torch.randn(50000, 1) * .2
    center = Normal(torch.zeros_like(noise), torch.full_like(noise, .2))
    saturated = Normal(torch.full_like(noise, 4.0), torch.full_like(noise, .2))
    assert (-tanh_log_prob(saturated, noise + 4)).mean() < (-tanh_log_prob(center, noise)).mean() - 5


def test_stored_rollout_actions_reproduce_policy_ratio_one():
    torch.manual_seed(0)
    model = ActorCritic(throttle_bias=1.3)
    obs = torch.randn(128, 24)
    with torch.no_grad():
        action, old_logp, _, _ = model.get_action_and_value(obs)
    _, new_logp, _, _ = model.get_action_and_value(obs, action)
    torch.testing.assert_close((new_logp - old_logp).exp(), torch.ones(128), atol=1e-5, rtol=1e-5)


def test_saturated_actions_retain_exact_likelihood_via_latents():
    model = ActorCritic(throttle_bias=20.0)
    obs = torch.zeros(4, 24)
    with torch.no_grad():
        action, old_logp, _, _, latent = model.get_action_and_value(obs, return_latent=True)
    assert torch.all(action[:, 4] == 1.0)
    _, new_logp, _, _ = model.get_action_and_value(obs, latent_action=latent)
    torch.testing.assert_close(new_logp, old_logp)


def test_value_regression_retains_gradient_for_large_terminal_returns():
    values = torch.tensor([1.0], requires_grad=True)
    value_loss(values, torch.tensor([475.0]), torch.zeros(1)).backward()
    assert values.grad.item() == -474.


def test_explicit_value_clip_reproduces_flat_gradient_diagnostic():
    values = torch.tensor([1.0], requires_grad=True)
    value_loss(values, torch.tensor([475.0]), torch.zeros(1), clip_range=.2).backward()
    assert values.grad.item() == 0.


def test_bounded_action_expectation_matches_sampling_and_symmetry():
    torch.manual_seed(12)
    mean = torch.tensor([[-1.3, 0., 1.3]])
    std = torch.tensor([.35, .35, .35])
    expected = bounded_action_mean(mean, std)
    samples = torch.tanh(mean + torch.randn(200000, 3) * std).mean(0, keepdim=True)
    torch.testing.assert_close(expected, samples, atol=.002, rtol=0)
    assert abs(float(expected[0, 1])) < 1e-6
    torch.testing.assert_close(expected[:, 0], -expected[:, 2])
    assert float(expected[0, 2]) < float(mean.tanh()[0, 2])
    torch.testing.assert_close(bounded_action_mean(mean, torch.zeros(3)), mean.tanh())


def test_exploration_cap_decays_without_increasing_already_small_variance():
    from tvc_env.controllers.ppo_model import anneal_log_std
    parameter = torch.nn.Parameter(torch.tensor([-2., -5.]))
    initial, target = torch.tensor([-2., -3.]), torch.tensor([-4., -4.])
    anneal_log_std(parameter, initial, target, .5)
    torch.testing.assert_close(parameter, torch.tensor([-3., -5.]))
    anneal_log_std(parameter, initial, target, 2.)
    torch.testing.assert_close(parameter, torch.tensor([-4., -5.]))


def test_policy_kl_matches_torch_and_detects_mean_shift_at_small_variance():
    from torch.distributions import kl_divergence
    from tvc_env.controllers.ppo_model import policy_kl
    torch.manual_seed(7)
    old, new = torch.randn(128, 5), torch.randn(128, 5)
    old_log_std, new_log_std = torch.randn(5) - 3, torch.randn(5) - 3
    expected = kl_divergence(Normal(old, old_log_std.exp()), Normal(new, new_log_std.exp())).sum(-1)
    torch.testing.assert_close(policy_kl(old, old_log_std, new, new_log_std), expected)
    torch.testing.assert_close(policy_kl(old, old_log_std, old, old_log_std), torch.zeros(128))
    # A small shift of .02 latent units is large relative to sigma=exp(-4).
    assert policy_kl(torch.zeros(1), torch.tensor(-4.), torch.tensor([.02]), torch.tensor(-4.)) > .5


def test_inference_sampling_uses_checkpoint_variance_and_deterministic_default():
    model = ActorCritic(throttle_bias=1.3)
    obs = torch.zeros(20000, 24)
    with torch.no_grad():
        model.log_std.fill_(-2.)
        mean = model.actor(obs)
        torch.testing.assert_close(model.act(obs), mean.tanh())
        samples = torch.atanh(model.act(obs, 'stochastic')) - mean
        torch.testing.assert_close(samples.std(0), model.log_std.exp(), atol=.003, rtol=0)
