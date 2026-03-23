"""
Hover task configuration adapter.

Loads hover.yaml, configures hover-specific reward terms/weights,
success criteria, termination conditions, and spawn ranges.
"""

from __future__ import annotations
from pathlib import Path
from tvc_env.envs.task_registry import resolve_task_config


class HoverTask:
    """Adapter for hover task configuration."""

    def __init__(self, sim_root: str | Path | None = None):
        self._config = resolve_task_config("hover", sim_root)
        task = self._config.get("task", {})

        self.name: str = task.get("name", "hover")
        self.target_position: list[float] = task.get("target_position", [0.0, 0.0, 5.0])
        self.episode_length_s: float = task.get("episode_length_s", 30.0)
        self.reward_weights: dict = task.get("reward", {})
        self.success: dict = task.get("success", {})
        self.termination: dict = task.get("termination", {})
        self.spawn: dict = task.get("spawn", {})

    @property
    def config(self) -> dict:
        return self._config
