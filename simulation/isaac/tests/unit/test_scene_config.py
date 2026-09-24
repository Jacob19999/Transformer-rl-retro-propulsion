"""Offline tests for Isaac scene configuration mapping."""

from tvc_env.sim.scene_builder import SceneConfig


def test_env_clock_overrides_default_solver_file_dt():
    cfg = SceneConfig.from_yaml(
        {
            "env": {"physics_dt": 0.002, "decimation": 16},
            "physics": {"dt": 1.0 / 120.0, "gpu_pipeline": False},
        }
    )
    assert cfg.physics_dt == 0.002
    assert cfg.decimation == 16
    assert cfg.device == "cpu"


def test_gpu_pipeline_maps_to_cuda_device():
    cfg = SceneConfig.from_yaml({"env": {}, "physics": {"gpu_pipeline": True}})
    assert cfg.device == "cuda:0"


def test_explicit_environment_solver_survives_automatic_defaults(tmp_path):
    from tvc_env.envs.base_env import BaseEnvConfig
    config = tmp_path / 'env.yaml'
    config.write_text('env:\n  num_envs: 4\nphysics:\n  enable_external_forces_every_iteration: false\n')
    assert BaseEnvConfig('landing', config).config['physics']['enable_external_forces_every_iteration'] is False
    explicit = tmp_path / 'solver.yaml'
    explicit.write_text('physics:\n  enable_external_forces_every_iteration: true\n')
    assert BaseEnvConfig('landing', config, explicit).config['physics']['enable_external_forces_every_iteration'] is True


def test_ground_plane_covers_env_grid_and_lateral_spawn_reach():
    """A drone with no ground beneath it free-falls and reads as a control failure.

    The ground was a hard-coded 200x200 m cuboid while the Isaac Lab env grid
    and the task spawn box both grew independently of it. At 4 m spacing the
    grid half-extent is 90 m for 2048 envs but 180 m for 8192, so a
    ppo_waypoints_staged_v4 stage-0 evaluation landed only 25.7% of episodes
    against 87.6% for the identical policy at 2048 envs, with max downward
    speed 59.9 m/s -- environments off the plane, not a worse policy.
    """
    import math

    from tvc_env.sim.scene_builder import DEFAULT_GROUND_MARGIN_M, SceneConfig

    widest_spawn_reach_m = 100.0  # configs/env/train_2048_8s_waypoints.yaml
    for num_envs in (128, 512, 2048, 4096, 8192, 16384):
        cfg = SceneConfig(num_envs=num_envs, env_spacing=4.0)
        columns = math.ceil(math.sqrt(num_envs))
        assert cfg.grid_half_extent_m == (columns - 1) * 4.0 / 2.0
        ground_half = cfg.grid_half_extent_m + DEFAULT_GROUND_MARGIN_M
        assert ground_half >= cfg.grid_half_extent_m + widest_spawn_reach_m, num_envs


def test_ground_half_extent_is_explicitly_overridable():
    from tvc_env.sim.scene_builder import SceneConfig

    assert SceneConfig.from_yaml({"env": {"num_envs": 8}}).ground_half_extent_m is None
    cfg = SceneConfig.from_yaml({"env": {"num_envs": 8, "ground_half_extent_m": 500.0}})
    assert cfg.ground_half_extent_m == 500.0


def test_physx_gpu_buffers_scale_with_environment_count():
    """PhysX GPU capacities are hard limits that fail silently when exceeded.

    physx_train.yaml sizes them for 2048 envs (found_lost_pairs_capacity is
    literally 8192). Raising --num-envs without raising these degrades
    broad-phase pairs and contact patches, which surfaces as missed landings
    rather than an error.
    """
    import pathlib

    import yaml

    from tvc_env.sim.scene_builder import (
        GPU_BUFFER_REFERENCE_ENVS,
        SceneConfig,
        scale_gpu_buffer,
    )

    sim_root = pathlib.Path(__file__).resolve().parents[2]
    physics = yaml.safe_load(
        (sim_root / "configs/physics/physx_train.yaml").read_text(encoding="utf-8")
    )["physics"]
    base = SceneConfig.from_yaml({"env": {"num_envs": GPU_BUFFER_REFERENCE_ENVS},
                                  "physics": physics})
    wide = SceneConfig.from_yaml({"env": {"num_envs": 4 * GPU_BUFFER_REFERENCE_ENVS},
                                  "physics": physics})
    for field in ("gpu_temp_buffer_capacity", "gpu_max_rigid_contact_count",
                  "gpu_max_rigid_patch_count", "gpu_found_lost_pairs_capacity"):
        assert getattr(wide, field) == 4 * getattr(base, field), field

    # Fewer environments must never shrink a capacity below its configured value.
    narrow = SceneConfig.from_yaml({"env": {"num_envs": 64}, "physics": physics})
    assert narrow.gpu_found_lost_pairs_capacity == physics["gpu_found_lost_pairs_capacity"]
    assert scale_gpu_buffer(None, 8192) is None
