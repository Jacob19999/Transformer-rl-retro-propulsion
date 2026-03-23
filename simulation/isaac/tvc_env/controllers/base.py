"""
Controller base interface for the TVC environment.

All controllers must implement compute_action(obs) → action_tensor.
"""

from __future__ import annotations
import torch
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Any


class BaseController(ABC):
    """Abstract base class for all TVC controllers."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._max_fin_angle = 0.262  # rad (15°) — from action_space contract

    @abstractmethod
    def compute_action(self, obs: Tensor) -> Tensor:
        """Compute action from observation.

        Args:
            obs: Observation tensor of shape (num_envs, obs_dim).

        Returns:
            Action tensor of shape (num_envs, 5):
              [0:4] fin target angles (rad) in [-max_angle, max_angle]
              [4]   throttle [0, 1]
        """
        ...

    def validate_action(self, action: Tensor) -> Tensor:
        """Clamp action to valid bounds per action_space contract."""
        fin_clipped = action[:, :4].clamp(-self._max_fin_angle, self._max_fin_angle)
        throttle_clipped = action[:, 4:5].clamp(0.0, 1.0)
        return torch.cat([fin_clipped, throttle_clipped], dim=-1)

    def reset(self, env_ids: Tensor | None = None) -> None:
        """Reset controller state for specified environments. Override if stateful."""
        pass
