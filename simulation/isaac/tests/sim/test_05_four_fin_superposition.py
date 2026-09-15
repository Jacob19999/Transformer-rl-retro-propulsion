"""Radial-hinge force superposition; pure tensor checks also run without Kit.

Body-FRD order is FWD, RIGHT, AFT, LEFT. Positive roll uses -FWD/+AFT;
positive pitch uses -RIGHT/+LEFT; common positive deflection gives +yaw.
These signs follow r cross F and the user-confirmed radial hinge spans.
Actual articulation motion and net lift are checked by test_14 and recorded
pose verification, not inferred solely from this aerodynamic calculation.
"""
from pathlib import Path
import pytest
import torch
import yaml


@pytest.fixture
def dispatch():
    from tvc_env.asset.usd_loader import load_asset_metadata
    from tvc_env.dynamics.fin_force_dispatch import FinForceDispatch
    root = Path(__file__).parents[2]
    return FinForceDispatch.from_metadata_and_config(
        load_asset_metadata(root / 'assets/metadata/edf_drone_v2.asset.yaml'),
        yaml.safe_load((root / 'configs/vehicle/edf_drone_v2.yaml').read_text(encoding='utf-8')),
        yaml.safe_load((root / 'configs/params/edf_90mm.yaml').read_text(encoding='utf-8')))


def torque(dispatch, angles):
    result = dispatch.compute_body_frame_forces(torch.tensor([angles]), torch.ones(1))
    return torch.linalg.cross(result.cop_positions[None], result.forces_body).sum(1)[0]


@pytest.mark.parametrize('axis,angles', [(0,[-.15,0.,.15,0.]),
                                        (1,[0.,-.15,0.,.15]), (2,[.15]*4)])
def test_radial_patterns_have_correct_axis_and_sign(dispatch, axis, angles):
    value = torque(dispatch, angles)
    assert value[axis] > 1e-4
    other = [i for i in range(3) if i != axis]
    assert value[other].abs().max() < 1e-6
    torch.testing.assert_close(torque(dispatch, [-a for a in angles]), -value)


def test_neutral_fins_have_no_side_moment_but_retain_axial_drag(dispatch):
    result = dispatch.compute_body_frame_forces(torch.zeros(1,4), torch.ones(1))
    assert result.forces_body.abs().max() == 0
    assert result.thrust_loss.item() > 0


def test_individual_fin_moments_superpose(dispatch):
    angles = [.1,-.2,.03,.15]
    individual = [torque(dispatch, [angles[j] if i == j else 0. for j in range(4)]) for i in range(4)]
    torch.testing.assert_close(torque(dispatch, angles), torch.stack(individual).sum(0))
