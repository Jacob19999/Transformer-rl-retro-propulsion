"""Refinement must not silently change the policy period or landing detector."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from tvc_env.envs.evaluation_contract import refine_physics_clock


def config():
    return SimpleNamespace(physics_dt=1/240, decimation=8,
        config={'env': {'physics_dt':1/240, 'decimation':8},
                'task':{'contact':{'dwell_frames':30,'min_contact_force':1.0}}})


def test_refinement_preserves_physical_intervals():
    cfg = config()
    report = refine_physics_clock(cfg, 960)
    assert cfg.physics_dt == 1/960
    assert cfg.decimation == 32
    assert cfg.config['task']['contact']['dwell_frames'] == 120
    assert cfg.config['task']['contact']['min_contact_force'] == 1.0
    assert cfg.physics_dt * cfg.decimation == pytest.approx(1/30)
    assert report['contact_dwell_s'] == .125


@pytest.mark.parametrize('hz', [120, 300, -240])
def test_invalid_refinement_does_not_mutate_task(hz):
    cfg = config()
    before = deepcopy(cfg)
    with pytest.raises(ValueError):
        refine_physics_clock(cfg, hz)
    assert vars(cfg) == vars(before)
