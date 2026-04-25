"""
PPO action adapter for the TVC environment.

Interprets raw 5-dim network output as:
  [0:4] fin target angles — scaled from tanh output to ±max_fin_angle (rad)
  [4]   throttle          — scaled from tanh output to [0, 1]

The adapter wraps a PPO policy callable and applies action scaling/clipping
to conform to the action_space contract.
"""

from __future__ import annotations
import torch
from torch import Tensor
from typing import Callable, Any

from tvc_env.controllers.base import BaseController


class PPOAdapter(BaseController):
    """Wraps a PPO policy and maps its output to the 5-dim action space."""

    def __init__(
        self,
        policy: Callable[[Tensor], Tensor],
        max_fin_angle: float = 0.262,   # rad (15°) per action_space contract
        config: dict[str, Any] | None = None,
        device: torch.device = None,
    ):
        """
        Args:
            policy:        Callable (num_envs, obs_dim) → (num_envs, 5).
                           Network output is expected in tanh-saturated range (-1, 1).
            max_fin_angle: Maximum fin deflection (rad).
            config:        Optional config dict.
            device:        Target device.
        """
        super().__init__(config)
        self._policy = policy
        self._max_fin_angle = max_fin_angle
        self.device = device

    def compute_action(self, obs: Tensor) -> Tensor:
        """Compute scaled action from raw PPO policy output.

        Args:
            obs: (num_envs, 24) observation tensor per contract.

        Returns:
            Action (num_envs, 5): [fin0, fin1, fin2, fin3, throttle].
        """
        raw = self._policy(obs)  # (num_envs, 5), expected in [-1, 1]

        # Scale fin angles from [-1, 1] → [-max_fin_angle, +max_fin_angle]
        fins = raw[:, :4] * self._max_fin_angle

        # Scale throttle from [-1, 1] → [0, 1]
        throttle = (raw[:, 4:5] + 1.0) * 0.5

        action = torch.cat([fins, throttle], dim=-1)
        return self.validate_action(action)

    def reset(self, env_ids: Tensor | None = None) -> None:
        """PPO policy is stateless between episodes — no-op."""
        pass
