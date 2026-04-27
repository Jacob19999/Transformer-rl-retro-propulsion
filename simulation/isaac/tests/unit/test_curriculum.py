from tvc_env.envs.curriculum import apply_spawn_position_range, resolve_spawn_curriculum


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
