"""Unit tests for fin force orientation and mixer sign conventions."""

from pathlib import Path

import torch
import yaml

from tvc_env.asset.usd_loader import load_asset_metadata
from tvc_env.controllers.pid_fin_mixer import PIDFinMixer
from tvc_env.dynamics.fin_force_dispatch import FinForceDispatch


SIM_ROOT = Path(__file__).parents[2]
METADATA_PATH = SIM_ROOT / "assets/metadata/edf_drone_v2.asset.yaml"
VEHICLE_PATH = SIM_ROOT / "configs/vehicle/edf_drone_v2.yaml"
EDF_PATH = SIM_ROOT / "configs/params/edf_90mm.yaml"


def _load_dispatch() -> FinForceDispatch:
    metadata = load_asset_metadata(METADATA_PATH)
    with open(VEHICLE_PATH, "r", encoding="utf-8") as f:
        vehicle_config = yaml.safe_load(f)
    with open(EDF_PATH, "r", encoding="utf-8") as f:
        edf_config = yaml.safe_load(f)
    return FinForceDispatch.from_metadata_and_config(metadata, vehicle_config, edf_config)


def _body_torque(result) -> torch.Tensor:
    cops = result.cop_positions.unsqueeze(0).expand_as(result.forces_body)
    return torch.linalg.cross(cops, result.forces_body).sum(dim=1)


def test_symmetric_fin_deflection_has_no_axial_force_and_positive_thrust_loss():
    dispatch = _load_dispatch()
    angles = torch.full((1, 4), 0.10)
    result = dispatch.compute_body_frame_forces(angles, torch.ones(1))

    body_force = result.forces_body.sum(dim=1)
    assert abs(body_force[0, 2].item()) < 1e-6
    assert result.thrust_loss[0].item() > 0.0


def test_positive_pitch_command_produces_positive_y_body_torque():
    dispatch = _load_dispatch()
    mixer = PIDFinMixer(max_fin_angle=1.0)
    angles = mixer.mix(torch.zeros(1), torch.ones(1) * 0.10, torch.zeros(1))
    result = dispatch.compute_body_frame_forces(angles, torch.ones(1))
    torque = _body_torque(result)

    assert torque[0, 1].item() > 0.0


def test_positive_roll_command_produces_positive_x_body_torque():
    dispatch = _load_dispatch()
    mixer = PIDFinMixer(max_fin_angle=1.0)
    angles = mixer.mix(torch.ones(1) * 0.10, torch.zeros(1), torch.zeros(1))
    result = dispatch.compute_body_frame_forces(angles, torch.ones(1))
    torque = _body_torque(result)

    assert torque[0, 0].item() > 0.0


def test_vane_reaction_opposes_downstream_chord_deflection():
    dispatch = _load_dispatch()
    result = dispatch.compute_body_frame_forces(torch.full((1, 4), 0.1), torch.ones(1))
    # d(chord)/d(angle) = hinge x chord = metadata normal. The downstream
    # jet turns toward that normal; momentum reaction on the vane is negative.
    assert torch.all((result.forces_body * dispatch.normal_dirs).sum(-1) < 0.0)


def test_radial_hinges_generate_yaw_and_have_three_axis_authority():
    dispatch = _load_dispatch()
    result = dispatch.compute_body_frame_forces(torch.full((1,4), .1), torch.ones(1))
    torque = _body_torque(result)
    assert torque[0,2] > 0
    torch.testing.assert_close(torque[0,:2], torch.zeros(2), atol=1e-7, rtol=0)
    columns = dispatch.compute_body_frame_forces(torch.eye(4)*.01, torch.ones(4))
    assert torch.linalg.matrix_rank(_body_torque(columns)) == 3
