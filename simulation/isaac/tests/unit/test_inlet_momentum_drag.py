"""Intake momentum drag of the EDF (physics review 2026-09-29)."""

import math
from pathlib import Path

import torch
import yaml

from tvc_env.dynamics.propulsion_edf import inlet_momentum_drag

ROOT = Path(__file__).resolve().parents[2]


def test_force_opposes_air_relative_inlet_velocity_and_never_adds_energy():
    torch.manual_seed(0)
    mass_flow = torch.rand(64) * 0.5
    velocity = torch.randn(64, 3) * 4.0
    force = inlet_momentum_drag(mass_flow, velocity)
    assert torch.allclose(force, -mass_flow[:, None] * velocity)
    power = (force * velocity).sum(-1)
    assert torch.all(power <= 0.0)
    assert torch.allclose(power, -mass_flow * velocity.square().sum(-1))


def test_hover_crosswind_drag_dominates_body_form_drag_at_low_speed():
    # Planned 8S: 48 N / 128 m/s at full rotor, hover thrust ~ weight.
    weight = 3.104 * 9.81
    fraction = math.sqrt(weight / 48.0)
    mass_flow = weight / (128.0 * fraction)             # T / u, u scales with rotor speed
    ram = float(inlet_momentum_drag(torch.tensor([mass_flow]), torch.tensor([[1.0, 0.0, 0.0]]))[0, 0])
    body = yaml.safe_load((ROOT / 'configs/vehicle/edf_drone_v2.yaml').read_text())['vehicle']['body']
    form = 0.5 * 1.225 * body['cd_lateral'] * body['lateral_reference_area'] * 1.0 ** 2
    assert 0.25 < -ram < 0.35
    assert -ram > 8.0 * form


def test_mission_plant_enables_it_at_the_inlet_above_the_com():
    plant = yaml.safe_load((ROOT / 'configs/env/mission_plant.yaml').read_text())
    inlet = plant['dynamics']['inlet_momentum_drag']
    assert inlet['enabled'] is True
    vehicle = yaml.safe_load((ROOT / 'configs/vehicle/edf_drone_v2.yaml').read_text())['vehicle']
    # FRD z is down: the intake lip sits above the COM, so a crosswind pitches the nose away from it.
    assert inlet['inlet_position_frd'][2] < vehicle['body_com_offset'][2]


def test_physx_default_does_not_clamp_physical_body_rates():
    from tvc_env.sim.scene_builder import SceneConfig
    cfg = SceneConfig.from_yaml({'env': {}, 'physics': {}})
    # A warm rotor spawn hands the body ~680 deg/s of yaw; the old 100 deg/s
    # default silently clipped (and removed energy from) every faster rotation.
    assert cfg.max_angular_velocity_deg_s >= 3600.0
