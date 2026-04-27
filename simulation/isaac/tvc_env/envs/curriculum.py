"""Training curriculum helpers for task-configured environment schedules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SpawnCurriculumState:
    """Resolved spawn curriculum state at one training step."""

    enabled: bool
    progress: float
    position_range: list[list[float]]
    final_position_range: list[list[float]]


def _as_range(value: Any, *, name: str) -> list[list[float]]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(row, list) and len(row) == 3 for row in value)
    ):
        raise ValueError(f"{name} must be [[x_min, y_min, z_min], [x_max, y_max, z_max]]")
    return [[float(x) for x in row] for row in value]


def _interpolate_range(
    start_range: list[list[float]],
    end_range: list[list[float]],
    progress: float,
) -> list[list[float]]:
    return [
        [
            start_range[row][col] + progress * (end_range[row][col] - start_range[row][col])
            for col in range(3)
        ]
        for row in range(2)
    ]


def resolve_spawn_curriculum(task_config: dict[str, Any], global_step: int) -> SpawnCurriculumState:
    """Return the active spawn range for a task's optional spawn curriculum.

    The curriculum is intentionally limited to reset sampling. Rewards,
    terminations, and observations remain unchanged, and evaluation can force
    ``final_position_range`` to test the full non-curricularized task.
    """

    task = task_config.get("task", task_config)
    spawn = task.get("spawn", {})
    final_range = _as_range(spawn.get("position_range", [[-1.0, -1.0, 4.0], [1.0, 1.0, 6.0]]), name="spawn.position_range")
    curriculum = spawn.get("curriculum", {})
    if not curriculum or not bool(curriculum.get("enabled", False)):
        return SpawnCurriculumState(
            enabled=False,
            progress=1.0,
            position_range=final_range,
            final_position_range=final_range,
        )

    start_range = _as_range(curriculum.get("position_start_range"), name="spawn.curriculum.position_start_range")
    end_range = _as_range(curriculum.get("position_end_range", final_range), name="spawn.curriculum.position_end_range")
    start_step = int(curriculum.get("start_step", 0))
    end_step = int(curriculum.get("end_step", start_step))
    if end_step <= start_step:
        progress = 1.0
    else:
        progress = (int(global_step) - start_step) / float(end_step - start_step)
        progress = max(0.0, min(1.0, progress))

    return SpawnCurriculumState(
        enabled=True,
        progress=progress,
        position_range=_interpolate_range(start_range, end_range, progress),
        final_position_range=end_range,
    )


def apply_spawn_position_range(task_config: dict[str, Any], position_range: list[list[float]]) -> None:
    """Mutate a task config so future resets sample from ``position_range``."""

    task = task_config.setdefault("task", {})
    spawn = task.setdefault("spawn", {})
    spawn["position_range"] = _as_range(position_range, name="position_range")
