"""Electrical invariants and physical coupling, independent of Isaac graphics."""
import copy
import pytest
import torch
from mission_control.models import DEFAULTS, battery_config, validate_mission, hardware_overrides
from tvc_env.dynamics.battery_lipo import LiPoBattery
from tvc_env.dynamics.propulsion_edf import EDFModel


def pack(**overrides):
    c = battery_config(validate_mission({}))
    c.update(overrides)
    return LiPoBattery(c, 2)


def test_load_obeys_electrical_equations_and_coulomb_counting():
    b = pack()
    before = b.soc.clone()
    emf = b.ocv()
    v, i, cutoff, limited = b.solve_load(torch.tensor([400., 800.]))
    assert torch.allclose(v * i, torch.tensor([400., 800.]), atol=.01)
    assert torch.allclose(v, emf - i * .024, atol=1e-5)
    b.integrate(v, i, cutoff, limited, 10.)
    assert torch.allclose(b.soc, before - i * 10 / (3600 * 5), atol=1e-6)
    assert torch.allclose(b.energy_wh, v * i * 10 / 3600, atol=1e-6)


def test_current_limit_and_cutoff_are_physical_and_latched():
    b = pack(max_current_a=20)
    v, i, cut, limit = b.solve_load(torch.tensor([10000., 10000.]))
    assert torch.all(i <= 20.001) and limit.all()
    b.integrate(v, i, cut, limit, .1)
    b.soc[1] = 0
    v, i, cut, limit = b.solve_load(torch.ones(2) * 100)
    assert cut[1] and v[1] == i[1] == 0
    b.integrate(v, i, cut, limit, .1)
    b.soc[1] = 1
    assert b.solve_load(torch.ones(2))[2][1]


def test_low_charge_reduces_real_rotor_speed_and_thrust():
    b = pack()
    b.soc[1] = .15
    b.voltage_v = b.ocv()
    edf = EDFModel(max_thrust=48, omega_max=4649.56, rotor_inertia=.0002)
    omega = torch.zeros(2)
    for _ in range(360):
        omega = b.update_motor(edf, omega, torch.ones(2) * .9, 1/120)
    thrust = edf.compute_thrust(omega)
    assert thrust[0] > thrust[1] + 4
    assert (b.energy_wh > 0).all() and (b.temperature_c > 25).all()


def test_motor_cannot_create_energy_when_current_limited():
    b = pack(max_current_a=5)
    edf = EDFModel(max_thrust=48, omega_max=4649.56, rotor_inertia=.0002)
    omega = torch.zeros(2)
    for _ in range(120):
        old = omega.clone()
        aero = b.config['shaft_power_at_max_w'] * (old / edf.omega_max)**3
        omega = b.update_motor(edf, old, torch.ones(2), 1/120)
        kinetic_w = .5 * edf.rotor_inertia * (omega.square()-old.square()) * 120
        available = (b.power_w - b.config['auxiliary_power_w']).clamp(min=0) * b.config['motor_efficiency']
        assert (kinetic_w + aero <= available + .05).all()
    assert (edf.compute_thrust(omega) < 10).all()


def test_partial_reset_does_not_recharge_other_environment():
    b = pack()
    b.soc[:] = .4
    b.energy_wh[:] = 7
    b.reset(torch.tensor([1]))
    assert b.soc[0] == .4 and b.soc[1] == 1
    assert b.energy_wh[0] == 7 and b.energy_wh[1] == 0


def test_policy_observes_normalized_electrical_state_without_future_values():
    b = pack()
    b.soc[:] = torch.tensor([.8, .4])
    b.voltage_v[:] = torch.tensor([30., 28.])
    b.current_a[:] = torch.tensor([60., 90.])
    b.polarization_v[:] = torch.tensor([.3, .6])
    obs = b.observation()
    assert obs.shape == (2, 4)
    assert torch.allclose(obs[:, 0], b.soc)
    assert torch.allclose(obs[:, 1] * 29.6, b.voltage_v)
    assert torch.allclose(obs[:, 2] * 120, b.current_a)
    assert torch.allclose(obs[:, 3] * 29.6, b.polarization_v)


def test_efficiency_reward_integrates_physical_units_and_preserves_terminal_priority():
    from types import SimpleNamespace
    from tvc_env.envs.rewards import compute_battery_energy_cost, compute_propulsive_delta_v_cost
    import yaml
    from pathlib import Path
    config = yaml.safe_load((Path(__file__).resolve().parents[2] / 'configs/env/train_512_8s_radial.yaml').read_text())
    weights = config['task']['reward']
    totals = []
    for steps in (900, 3600):
        dt = 30 / steps
        state = SimpleNamespace(battery_energy_step_wh=torch.tensor([2500 * dt / 3600]),
                                propulsive_delta_v_step=torch.tensor([10 * dt]),
                                motor_omega=torch.zeros(1))
        cost = steps * (weights['battery_energy_cost'] * compute_battery_energy_cost(state, {})
                        + weights['propulsive_delta_v_cost'] * compute_propulsive_delta_v_cost(state, {}))
        totals.append(float(cost))
    assert totals[0] == pytest.approx(totals[1], abs=1e-5)
    assert totals[0] == pytest.approx(-16.4166667, abs=1e-4)
    assert abs(totals[0]) < 200 * .1


def test_pack_and_esc_ratings_bound_current_and_profiles_are_explicit():
    mission = validate_mission({'battery': {'capacity_ah': 1, 'c_rating': 10}})
    assert battery_config(mission)['max_current_a'] == 10
    assert battery_config(mission)['cells'] == 8
    assert hardware_overrides(mission)['edf']['max_thrust'] == 48
    legacy = validate_mission({'hardware_profile': 'legacy_6s'})
    assert battery_config(legacy)['cells'] == 6
    assert not hardware_overrides(legacy)
    assert DEFAULTS['battery']['capacity_ah'] == 5


@pytest.mark.parametrize('bad', [{'position':[0,0,float('nan')]}, {'seed':1.1}, {'seed':True},
                               {'controller':'../../evil'}, {'battery':{'cells':12}},
                               {'battery':{'max_current_a':150}}, {'duration_s':1000}])
def test_mission_rejects_unsafe_or_inconsistent_inputs(bad):
    with pytest.raises(ValueError):
        validate_mission(bad)
