"""
Landing task configuration adapter.

Loads landing.yaml, configures landing-specific reward terms/weights,
success criteria (LANDED state + pad distance), termination conditions,
and spawn ranges.
"""

from __future__ import annotations
from pathlib import Path
from tvc_env.envs.task_registry import resolve_task_config


class LandingTask:
    """Adapter for landing task configuration."""

    def __init__(self, sim_root: str | Path | None = None):
        self._config = resolve_task_config("landing", sim_root)
        task = self._config.get("task", {})

        self.name: str = task.get("name", "landing")
        self.target_position: list[float] = task.get("target_position", [0.0, 0.0, 0.0])
        self.episode_length_s: float = task.get("episode_length_s", 60.0)
        self.reward_weights: dict = task.get("reward", {})
        self.success: dict = task.get("success", {})
        self.termination: dict = task.get("termination", {})
        self.spawn: dict = task.get("spawn", {})

    @property
    def config(self) -> dict:
        return self._config

    @property
    def max_pad_distance(self) -> float:
        return self.success.get("max_pad_distance", 0.5)
