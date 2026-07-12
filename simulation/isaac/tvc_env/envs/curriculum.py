"""Training curriculum helpers for task-configured environment schedules."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SpawnCurriculumState:
    """Resolved spawn curriculum state at one training step.

    For ``mode=linear`` the schedule is a pure function of ``global_step``
    and ``progress`` is the lerp factor. For ``mode=staged`` the schedule
    advances on a feedback signal (success fraction) tracked outside this
    module; ``progress`` then reports the fraction of stages completed
    (``stage_index / max(num_stages - 1, 1)``).
    """

    enabled: bool
    progress: float
    position_range: list[list[float]]
    final_position_range: list[list[float]]
    mode: str = "linear"
    stage_index: int = 0
    num_stages: int = 1


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


def resolve_spawn_curriculum(
    task_config: dict[str, Any],
    global_step: int,
    stage_index: int = 0,
) -> SpawnCurriculumState:
    """Return the active spawn range for a task's optional spawn curriculum.

    Two modes are supported:

    * ``linear`` (default, backward-compatible): linearly interpolates
      ``position_start_range`` → ``position_end_range`` between ``start_step``
      and ``end_step``. Pure function of ``global_step``.
    * ``staged``: returns the ``position_range`` of the requested
      ``stage_index`` from ``curriculum.stages``. Stage advancement is
      driven by feedback (success fraction) tracked outside this module
      via :class:`StagedCurriculumTracker`.

    The curriculum is intentionally limited to reset sampling. Rewards,
    terminations, and observations remain unchanged, and evaluation forces
    the ``final_position_range`` (last stage / linear end_range) so that
    reported metrics reflect the un-curricularized task.
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
            mode="linear",
            stage_index=0,
            num_stages=1,
        )

    mode = str(curriculum.get("mode", "linear")).lower()
    if mode == "staged":
        stages = curriculum.get("stages") or []
        if not stages:
            raise ValueError("spawn.curriculum.mode='staged' requires a non-empty 'stages' list")
        num_stages = len(stages)
        idx = max(0, min(int(stage_index), num_stages - 1))
        stage = stages[idx]
        stage_range = _as_range(stage.get("position_range"), name=f"spawn.curriculum.stages[{idx}].position_range")
        last_range = _as_range(stages[-1].get("position_range"), name=f"spawn.curriculum.stages[-1].position_range")
        progress = idx / max(num_stages - 1, 1)
        return SpawnCurriculumState(
            enabled=True,
            progress=progress,
            position_range=stage_range,
            final_position_range=last_range,
            mode="staged",
            stage_index=idx,
            num_stages=num_stages,
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
        mode="linear",
        stage_index=0,
        num_stages=1,
    )


def apply_spawn_position_range(task_config: dict[str, Any], position_range: list[list[float]]) -> None:
    """Mutate a task config so future resets sample from ``position_range``."""

    task = task_config.setdefault("task", {})
    spawn = task.setdefault("spawn", {})
    spawn["position_range"] = _as_range(position_range, name="position_range")


@dataclass
class StagedCurriculumTracker:
    """Feedback-driven stage advancement for ``mode=staged`` curricula.

    The trainer feeds terminal events (one bool per env per RL step:
    ``True`` if that env just completed a successful landing in the
    current step, ``False`` otherwise — including non-terminal steps,
    crashes, and off-pad / hard-touchdown landings). The tracker
    maintains a rolling-window success fraction and, after a stage's
    ``min_steps`` budget has elapsed, signals advancement when the
    success fraction crosses ``advance_success_fraction`` or when
    ``max_steps`` has elapsed (whichever comes first).

    ``min_steps`` and ``max_steps`` are counted in environment steps
    spent in the current stage (i.e. ``num_envs * rl_steps``).

    The stage list itself lives in the task YAML under
    ``spawn.curriculum.stages``; this tracker only owns the *which
    stage / when to advance* state.
    """

    stages: list[dict[str, Any]]
    success_window_size: int = 1024
    stage_index: int = 0
    steps_in_stage: int = 0
    _successes: deque = field(default_factory=deque)
    _terminations: deque = field(default_factory=deque)

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("StagedCurriculumTracker requires at least one stage")
        # Use bounded deques so memory is constant. The window tracks the most
        # recent ``success_window_size`` *terminal events* (LANDED or CRASHED),
        # not all rollout steps — non-terminal steps are uninformative for
        # success-fraction estimation.
        self._successes = deque(maxlen=self.success_window_size)
        self._terminations = deque(maxlen=self.success_window_size)

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @property
    def at_final_stage(self) -> bool:
        return self.stage_index >= len(self.stages) - 1

    def current_stage_cfg(self) -> dict[str, Any]:
        return self.stages[self.stage_index]

    def record_step(self, num_env_steps: int, success_count: int, termination_count: int) -> None:
        """Account for one rollout slice.

        ``num_env_steps`` is ``num_envs * rl_steps_in_slice`` and counts
        toward ``min_steps`` / ``max_steps`` thresholds. ``success_count``
        is the number of successful landings in the slice;
        ``termination_count`` is the total number of LANDED+CRASHED
        events in the slice. These are summarized to one window entry per
        terminal event so that the success fraction is *over terminal
        events*, not rollout steps.
        """
        if num_env_steps < 0 or success_count < 0 or termination_count < 0:
            raise ValueError("counts must be non-negative")
        if success_count > termination_count:
            raise ValueError("success_count cannot exceed termination_count")
        self.steps_in_stage += int(num_env_steps)
        # Push one bool per terminal event. The deque is bounded so old
        # events are evicted automatically.
        for _ in range(success_count):
            self._successes.append(True)
            self._terminations.append(True)
        for _ in range(termination_count - success_count):
            self._successes.append(False)
            self._terminations.append(True)

    def success_fraction(self) -> float:
        if not self._terminations:
            return 0.0
        return sum(1 for s in self._successes if s) / len(self._terminations)

    def termination_count(self) -> int:
        return len(self._terminations)

    def should_advance(self) -> bool:
        if self.at_final_stage:
            return False
        cfg = self.current_stage_cfg()
        min_steps = int(cfg.get("min_steps", 0))
        max_steps = int(cfg.get("max_steps", 0))  # 0 = no cap
        threshold = float(cfg.get("advance_success_fraction", 1.0))
        min_terminations = int(cfg.get("min_terminations", 64))
        if self.steps_in_stage < min_steps:
            return False
        if max_steps > 0 and self.steps_in_stage >= max_steps:
            return True
        if self.termination_count() < min_terminations:
            return False
        return self.success_fraction() >= threshold

    def advance(self) -> int:
        """Advance to the next stage and reset stage-local state.

        Returns the new stage index. No-op when already at the final stage.
        """
        if self.at_final_stage:
            return self.stage_index
        self.stage_index += 1
        self.steps_in_stage = 0
        self._successes.clear()
        self._terminations.clear()
        return self.stage_index
