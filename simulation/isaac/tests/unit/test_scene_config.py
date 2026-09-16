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
