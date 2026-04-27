"""
Reward term registry per research decision R9.

Maps string term names to reward functions. Registry pattern:
  {"alive_bonus": fn, "position_error": fn, ...}

Each function signature: fn(env_state, config) → Tensor of shape (num_envs,).
"""

from __future__ import annotations
from typing import Callable
from tvc_env.envs.rewards import (
    compute_alive_bonus,
    compute_position_error_reward,
    compute_horizontal_position_error_reward,
    compute_attitude_error_reward,
    compute_angular_velocity_reward,
    compute_control_effort_reward,
    compute_control_rate_reward,
    compute_hover_stability_reward,
    compute_drift_penalty,
    compute_contact_penalty,
    compute_crash_penalty,
    compute_touchdown_softness_reward,
    compute_landing_success_reward,
    compute_pad_accuracy_reward,
    compute_vertical_speed_shaping,
    compute_delta_v_cost,
)

# Registry: term name → reward function
_REGISTRY: dict[str, Callable] = {
    # Shared terms
    "alive_bonus": compute_alive_bonus,
    "position_error": compute_position_error_reward,
    "horizontal_position_error": compute_horizontal_position_error_reward,
    "attitude_error": compute_attitude_error_reward,
    "angular_velocity": compute_angular_velocity_reward,
    "control_effort": compute_control_effort_reward,
    "control_rate": compute_control_rate_reward,

    # Hover-specific terms
    "hover_stability": compute_hover_stability_reward,
    "drift_penalty": compute_drift_penalty,
    "contact_penalty": compute_contact_penalty,

    # Landing-specific terms
    "crash_penalty": compute_crash_penalty,
    "touchdown_softness": compute_touchdown_softness_reward,
    "landing_success": compute_landing_success_reward,
    "pad_accuracy": compute_pad_accuracy_reward,
    "vertical_speed_shaping": compute_vertical_speed_shaping,
    "delta_v_cost": compute_delta_v_cost,
}


def get_reward_fn(term_name: str) -> Callable:
    """Get a reward function by term name.

    Args:
        term_name: String key matching a registered reward term.

    Returns:
        Reward function with signature fn(env_state, config) → Tensor(num_envs,).

    Raises:
        KeyError: If term_name is not registered.
    """
    if term_name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(
            f"Reward term '{term_name}' not registered. Available terms: {available}"
        )
    return _REGISTRY[term_name]


def compute_total_reward(reward_weights: dict[str, float], env_state, config: dict):
    """Compute weighted sum of all active reward terms.

    Args:
        reward_weights: Dict of {term_name: weight} from task config.
        env_state: Current VehicleState or equivalent state container.
        config: Task config dict for term-specific parameters.

    Returns:
        Tensor of shape (num_envs,) — total reward per environment.
    """
    import torch
    total = None
    for term_name, weight in reward_weights.items():
        if weight == 0.0:
            continue
        fn = get_reward_fn(term_name)
        term_reward = fn(env_state, config)
        weighted = weight * term_reward
        total = weighted if total is None else total + weighted
    if total is None:
        # No active terms
        import torch
        return torch.zeros(env_state.position.shape[0], device=env_state.position.device)
    return total


def list_registered_terms() -> list[str]:
    """Return sorted list of all registered reward term names."""
    return sorted(_REGISTRY.keys())
