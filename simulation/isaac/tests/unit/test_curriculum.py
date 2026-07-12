import pytest

from tvc_env.envs.curriculum import (
    StagedCurriculumTracker,
    apply_spawn_position_range,
    resolve_spawn_curriculum,
)


def _landing_config():
    return {
        "task": {
            "spawn": {
                "position_range": [[-2.0, -2.0, 8.0], [2.0, 2.0, 12.0]],
                "curriculum": {
                    "enabled": True,
                    "position_start_range": [[-0.5, -0.5, 8.0], [0.5, 0.5, 12.0]],
                    "position_end_range": [[-2.0, -2.0, 8.0], [2.0, 2.0, 12.0]],
                    "start_step": 0,
                    "end_step": 3000000,
                },
            }
        }
    }


def test_spawn_curriculum_starts_close_to_pad():
    state = resolve_spawn_curriculum(_landing_config(), global_step=0)

    assert state.enabled is True
    assert state.progress == 0.0
    assert state.position_range == [[-0.5, -0.5, 8.0], [0.5, 0.5, 12.0]]
    assert state.final_position_range == [[-2.0, -2.0, 8.0], [2.0, 2.0, 12.0]]


def test_spawn_curriculum_interpolates_xy_range():
    state = resolve_spawn_curriculum(_landing_config(), global_step=1500000)

    assert state.progress == 0.5
    assert state.position_range == [[-1.25, -1.25, 8.0], [1.25, 1.25, 12.0]]


def test_spawn_curriculum_clamps_to_full_task_range():
    state = resolve_spawn_curriculum(_landing_config(), global_step=6000000)

    assert state.progress == 1.0
    assert state.position_range == [[-2.0, -2.0, 8.0], [2.0, 2.0, 12.0]]


def test_apply_spawn_position_range_preserves_curriculum_metadata():
    config = _landing_config()

    apply_spawn_position_range(config, [[-1.0, -1.0, 8.0], [1.0, 1.0, 12.0]])

    assert config["task"]["spawn"]["position_range"] == [[-1.0, -1.0, 8.0], [1.0, 1.0, 12.0]]
    assert config["task"]["spawn"]["curriculum"]["position_end_range"] == [
        [-2.0, -2.0, 8.0],
        [2.0, 2.0, 12.0],
    ]


# ----- Staged curriculum -----


def _staged_landing_config():
    return {
        "task": {
            "spawn": {
                "position_range": [[-2.0, -2.0, 16.0], [2.0, 2.0, 20.0]],
                "curriculum": {
                    "enabled": True,
                    "mode": "staged",
                    "stages": [
                        {
                            "position_range": [[-0.5, -0.5, 16.0], [0.5, 0.5, 20.0]],
                            "min_steps": 100,
                            "max_steps": 1000,
                            "advance_success_fraction": 0.5,
                            "min_terminations": 4,
                        },
                        {
                            "position_range": [[-1.0, -1.0, 16.0], [1.0, 1.0, 20.0]],
                            "min_steps": 100,
                            "max_steps": 1000,
                            "advance_success_fraction": 0.5,
                            "min_terminations": 4,
                        },
                        {
                            "position_range": [[-2.0, -2.0, 16.0], [2.0, 2.0, 20.0]],
                            "min_steps": 100,
                            "max_steps": 0,
                            "advance_success_fraction": 1.0,
                            "min_terminations": 4,
                        },
                    ],
                },
            }
        }
    }


def test_staged_curriculum_returns_first_stage_by_default():
    state = resolve_spawn_curriculum(_staged_landing_config(), global_step=0)

    assert state.enabled is True
    assert state.mode == "staged"
    assert state.stage_index == 0
    assert state.num_stages == 3
    assert state.position_range == [[-0.5, -0.5, 16.0], [0.5, 0.5, 20.0]]
    assert state.final_position_range == [[-2.0, -2.0, 16.0], [2.0, 2.0, 20.0]]


def test_staged_curriculum_returns_requested_stage():
    state = resolve_spawn_curriculum(_staged_landing_config(), global_step=0, stage_index=1)

    assert state.stage_index == 1
    assert state.position_range == [[-1.0, -1.0, 16.0], [1.0, 1.0, 20.0]]


def test_staged_curriculum_clamps_stage_index():
    state = resolve_spawn_curriculum(_staged_landing_config(), global_step=0, stage_index=99)

    # Out-of-range stage_index clamps to the last stage.
    assert state.stage_index == 2
    assert state.position_range == [[-2.0, -2.0, 16.0], [2.0, 2.0, 20.0]]


def test_staged_curriculum_requires_stages_list():
    config = {
        "task": {
            "spawn": {
                "position_range": [[-1.0, -1.0, 16.0], [1.0, 1.0, 20.0]],
                "curriculum": {"enabled": True, "mode": "staged"},
            }
        }
    }
    with pytest.raises(ValueError, match="non-empty 'stages'"):
        resolve_spawn_curriculum(config, global_step=0)


def test_tracker_does_not_advance_before_min_steps():
    cfg = _staged_landing_config()
    tracker = StagedCurriculumTracker(stages=cfg["task"]["spawn"]["curriculum"]["stages"])
    # Plenty of successes but not enough env steps yet.
    tracker.record_step(num_env_steps=50, success_count=10, termination_count=10)

    assert tracker.should_advance() is False
    assert tracker.stage_index == 0


def test_tracker_advances_when_success_fraction_crosses_threshold():
    cfg = _staged_landing_config()
    tracker = StagedCurriculumTracker(stages=cfg["task"]["spawn"]["curriculum"]["stages"])
    tracker.record_step(num_env_steps=200, success_count=8, termination_count=10)

    # 80% success, > 50% threshold; min_steps=100 satisfied; 10 terminations >= min 4.
    assert tracker.should_advance() is True
    assert tracker.advance() == 1
    assert tracker.steps_in_stage == 0
    assert tracker.termination_count() == 0


def test_tracker_force_advances_at_max_steps_even_with_low_success():
    cfg = _staged_landing_config()
    tracker = StagedCurriculumTracker(stages=cfg["task"]["spawn"]["curriculum"]["stages"])
    tracker.record_step(num_env_steps=1000, success_count=1, termination_count=10)

    # max_steps=1000 hit even though success fraction (10%) is below threshold.
    assert tracker.should_advance() is True


def test_tracker_does_not_advance_past_final_stage():
    cfg = _staged_landing_config()
    tracker = StagedCurriculumTracker(stages=cfg["task"]["spawn"]["curriculum"]["stages"])
    tracker.advance()
    tracker.advance()
    assert tracker.stage_index == 2
    assert tracker.at_final_stage is True

    tracker.record_step(num_env_steps=10000, success_count=100, termination_count=100)
    assert tracker.should_advance() is False
    assert tracker.advance() == 2  # idempotent at final stage


def test_tracker_window_evicts_old_terminations():
    cfg = _staged_landing_config()
    tracker = StagedCurriculumTracker(
        stages=cfg["task"]["spawn"]["curriculum"]["stages"],
        success_window_size=4,
    )
    # First slice: 4 failures saturate the window.
    tracker.record_step(num_env_steps=200, success_count=0, termination_count=4)
    assert tracker.success_fraction() == 0.0
    # Second slice: 4 successes evict all prior failures.
    tracker.record_step(num_env_steps=10, success_count=4, termination_count=4)
    assert tracker.success_fraction() == 1.0
    assert tracker.termination_count() == 4  # window stays at 4


def test_tracker_rejects_inconsistent_counts():
    cfg = _staged_landing_config()
    tracker = StagedCurriculumTracker(stages=cfg["task"]["spawn"]["curriculum"]["stages"])
    with pytest.raises(ValueError):
        tracker.record_step(num_env_steps=10, success_count=5, termination_count=4)
    with pytest.raises(ValueError):
        tracker.record_step(num_env_steps=-1, success_count=0, termination_count=0)
