"""
GTrXL-PPO action adapter for the TVC environment.

Interprets raw 5-dim transformer policy output as:
  [0:4] fin target angles — scaled from tanh output to ±max_fin_angle (rad)
  [4]   throttle          — scaled from tanh output to [0, 1]

Handles sequence context required for GTrXL (Gated Transformer-XL) policies:
  - Memory state is maintained across steps within an episode
  - Reset clears memory for specified environments
  - Observation is passed through unbatching/rebatching as needed for the
    transformer's sequence dimension

The adapter wraps a GTrXL policy callable that accepts (obs, memory) and
returns (raw_action, new_memory).
"""

from __future__ import annotations
import torch
from torch import Tensor
from typing import Callable, Any

from tvc_env.controllers.base import BaseController


class GTrXLAdapter(BaseController):
    """Wraps a GTrXL-PPO policy and maps its output to the 5-dim action space.

    The policy callable signature:
        policy(obs: Tensor, memory: Tensor | None) -> (action: Tensor, new_memory: Tensor)

    If the policy does not require explicit memory management (e.g., it manages
    its own internal state), pass a plain callable and set stateless=True.
    """

    def __init__(
        self,
        policy: Callable,
        max_fin_angle: float = 0.262,   # rad (15°) per action_space contract
        stateless: bool = False,        # True if policy manages its own memory
        config: dict[str, Any] | None = None,
        device: torch.device = None,
    ):
        """
        Args:
            policy:        GTrXL policy callable.
                             If stateless=False: (obs, memory) → (raw_action, new_memory)
                             If stateless=True:  (obs,) → raw_action
            max_fin_angle: Maximum fin deflection (rad).
            stateless:     Whether the policy manages memory internally.
            config:        Optional config dict.
            device:        Target device.
        """
        super().__init__(config)
        self._policy = policy
        self._max_fin_angle = max_fin_angle
        self._stateless = stateless
        self.device = device
        self._memory: Tensor | None = None  # (num_envs, memory_dim) or None

    def compute_action(self, obs: Tensor) -> Tensor:
        """Compute scaled action from GTrXL policy output.

        Args:
            obs: (num_envs, 24) observation tensor per contract.

        Returns:
            Action (num_envs, 5): [fin0, fin1, fin2, fin3, throttle].
        """
        if self._stateless:
            raw = self._policy(obs)
        else:
            raw, self._memory = self._policy(obs, self._memory)

        # Scale fin angles from [-1, 1] → [-max_fin_angle, +max_fin_angle]
        fins = raw[:, :4] * self._max_fin_angle

        # Scale throttle from [-1, 1] → [0, 1]
        throttle = (raw[:, 4:5] + 1.0) * 0.5

        action = torch.cat([fins, throttle], dim=-1)
        return self.validate_action(action)

    def reset(self, env_ids: Tensor | None = None) -> None:
        """Reset transformer memory for specified environments.

        Args:
            env_ids: Environment indices to reset. None resets all.
        """
        if self._memory is None or self._stateless:
            return
        if env_ids is None:
            self._memory.zero_()
        else:
            self._memory[env_ids] = 0.0
