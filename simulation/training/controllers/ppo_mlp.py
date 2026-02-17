"""
PPO-MLP controller wrapper (Stage 19.1).

This module provides a `Controller`-compatible wrapper around an SB3 PPO model
trained with `MlpPolicy`. It is used for evaluation and (later) deployment
interfaces where we want a simple `get_action(obs) -> [-1,1]^5` API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from simulation.training.controllers.base import Controller


class PPOMlpController(Controller):
    """Wrap a Stable-Baselines3 PPO(MlpPolicy) model behind the Controller ABC."""

    def __init__(self, model: Any, *, deterministic: bool = True) -> None:
        """
        Parameters
        ----------
        model
            A `stable_baselines3.PPO` instance (kept as Any to avoid importing SB3
            at module import time in tooling contexts).
        deterministic
            If True, uses deterministic actions (recommended for evaluation).
        """

        self.model = model
        self.deterministic = bool(deterministic)

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        *,
        device: str = "auto",
        deterministic: bool = True,
        **kwargs: Any,
    ) -> "PPOMlpController":
        """Load a PPO-MLP policy from disk.

        Notes
        -----
        - This loads only the PPO policy weights/optimizer state from SB3's `.zip`.
        - If training used `VecNormalize`, evaluation should also load the saved
          normalization statistics and wrap the environment accordingly.
        """

        try:
            from stable_baselines3 import PPO
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "stable-baselines3 is required to load PPO models. "
                "Install with `pip install stable-baselines3`."
            ) from e

        model = PPO.load(str(model_path), device=device, **kwargs)
        return cls(model, deterministic=deterministic)

    def reset(self) -> None:
        # PPO-MLP is memoryless; nothing to reset.
        return None

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        if o.size < 1:
            raise ValueError("obs must be a non-empty array.")

        action, _state = self.model.predict(o, deterministic=self.deterministic)
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size != 5:
            raise ValueError(f"Expected action shape (5,), got {a.shape}.")
        return np.clip(a, -1.0, 1.0)


__all__ = ["PPOMlpController"]

