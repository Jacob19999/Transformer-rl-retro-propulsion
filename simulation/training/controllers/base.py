"""
Controller base class for EDF landing.

Implements Implementation Tracker Stage 16:
Shared ABC for all controller variants (PID, PPO-MLP, GTrXL-PPO, SCP).
Ref: training.md §2.3
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class Controller(ABC):
    """Abstract base class for all landing controllers."""

    @abstractmethod
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Map observation to action in [-1, 1]^5.

        Parameters
        ----------
        obs : np.ndarray
            Observation vector of shape (20,) from EDFLandingEnv.

        Returns
        -------
        np.ndarray
            Action vector of shape (5,) in [-1, 1]^5:
            [T_cmd_normalized, delta_1, delta_2, delta_3, delta_4].
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for new episode.

        Call once at the start of each episode before the first get_action.
        """
        ...

    def update_memory(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        """Optional: update internal memory (for recurrent / memory-based policies).

        Default implementation is a no-op. Override in GTrXL-PPO or other
        memory-equipped controllers to update hidden state after each step.

        Parameters
        ----------
        obs : np.ndarray
            Observation at the current step.
        action : np.ndarray
            Action taken at the current step.
        reward : float
            Reward received for the transition.
        done : bool
            Whether the episode ended (terminated or truncated).
        """
        pass
