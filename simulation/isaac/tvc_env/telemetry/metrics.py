"""
Aggregate episode metrics for TVC environment.

Computes per-episode summary statistics:
  - Mean/max position error
  - Mean/max tilt angle
  - Total reward
  - Episode length
  - Outcome: success / crash / timeout
  - Landing accuracy (landing task only)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor


@dataclass
class EpisodeMetrics:
    """Per-episode aggregate statistics."""
    episode_id: int = 0
    task: str = "hover"

    # Position error (m)
    mean_pos_error: float = 0.0
    max_pos_error: float = 0.0

    # Tilt angle (rad)
    mean_tilt: float = 0.0
    max_tilt: float = 0.0

    # Reward
    total_reward: float = 0.0
    mean_reward: float = 0.0

    # Episode length
    episode_length: int = 0

    # Outcome
    outcome: Literal["success", "crash", "timeout", "unknown"] = "unknown"

    # Landing accuracy (m from target — landing task only)
    landing_accuracy: float | None = None

    # Angular rate stats (rad/s)
    mean_ang_rate: float = 0.0
    max_ang_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "mean_pos_error": round(self.mean_pos_error, 4),
            "max_pos_error": round(self.max_pos_error, 4),
            "mean_tilt_deg": round(math.degrees(self.mean_tilt), 2),
            "max_tilt_deg": round(math.degrees(self.max_tilt), 2),
            "total_reward": round(self.total_reward, 4),
            "mean_reward": round(self.mean_reward, 6),
            "episode_length": self.episode_length,
            "outcome": self.outcome,
            "landing_accuracy": round(self.landing_accuracy, 4) if self.landing_accuracy is not None else None,
            "mean_ang_rate": round(self.mean_ang_rate, 4),
            "max_ang_rate": round(self.max_ang_rate, 4),
        }


class EpisodeMetricsTracker:
    """Accumulates per-step data and computes episode-level metrics."""

    def __init__(self, task: str = "hover", env_idx: int = 0):
        self._task = task
        self._env_idx = env_idx
        self._episode_id = 0
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        self._pos_errors: list[float] = []
        self._tilts: list[float] = []
        self._rewards: list[float] = []
        self._ang_rates: list[float] = []
        self._final_pos: list[float] | None = None

    def update(
        self,
        obs: Tensor,        # (num_envs, 24)
        reward: Tensor,     # (num_envs,)
        terminated: Tensor, # (num_envs,) bool
        truncated: Tensor,  # (num_envs,) bool
    ) -> EpisodeMetrics | None:
        """Update accumulators with current step data.

        Returns:
            EpisodeMetrics if the episode just ended, else None.
        """
        i = self._env_idx

        # Position error magnitude
        pos_err = obs[i, 0:3].norm().item()
        self._pos_errors.append(pos_err)

        # Tilt from quaternion w: tilt = 2 * acos(|w|)
        w = obs[i, 3].item()
        tilt = 2.0 * math.acos(min(abs(w), 1.0))
        self._tilts.append(tilt)

        # Reward
        self._rewards.append(reward[i].item())

        # Angular rate
        ang_rate = obs[i, 10:13].norm().item()
        self._ang_rates.append(ang_rate)

        # Track final position for landing accuracy
        self._final_pos = obs[i, 0:3].tolist()

        done = terminated[i].item() or truncated[i].item()
        if done:
            metrics = self._compute_metrics(
                crashed=terminated[i].item(),
                truncated=truncated[i].item(),
            )
            self._episode_id += 1
            self._reset_accumulators()
            return metrics
        return None

    def _compute_metrics(self, crashed: bool, truncated: bool) -> EpisodeMetrics:
        n = len(self._pos_errors)
        if n == 0:
            return EpisodeMetrics(episode_id=self._episode_id, task=self._task)

        if crashed:
            outcome = "crash"
        elif truncated:
            outcome = "timeout"
        else:
            outcome = "success"

        # Landing accuracy: XY distance from target (pos_error is relative to target)
        landing_acc = None
        if self._task == "landing" and self._final_pos is not None:
            # pos_error is distance to target, so final pos_error IS the landing accuracy
            landing_acc = math.sqrt(self._final_pos[0] ** 2 + self._final_pos[1] ** 2)

        return EpisodeMetrics(
            episode_id=self._episode_id,
            task=self._task,
            mean_pos_error=sum(self._pos_errors) / n,
            max_pos_error=max(self._pos_errors),
            mean_tilt=sum(self._tilts) / n,
            max_tilt=max(self._tilts),
            total_reward=sum(self._rewards),
            mean_reward=sum(self._rewards) / n,
            episode_length=n,
            outcome=outcome,
            landing_accuracy=landing_acc,
            mean_ang_rate=sum(self._ang_rates) / n,
            max_ang_rate=max(self._ang_rates),
        )
