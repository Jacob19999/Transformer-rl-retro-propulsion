"""Whole-flight rotation must survive reversals, substep spikes and resets."""
import math
from types import SimpleNamespace

import pytest
import torch

from tvc_env.envs.rotation_metrics import RotationTracker
from tvc_env.envs.rewards import compute_excess_rotation_cost, compute_rotation_quality_reward
from tvc_env.common.constants import ContactState


def rates(*values):
    return torch.tensor([values], dtype=torch.float32) * math.pi / 180


def test_constant_rates_have_axis_specific_limits_and_correct_units():
    tracker = RotationTracker(1, 'cpu')
    w = rates(180, 45, -360)
    tracker.update(w, w, 2., torch.tensor([True]))
    r = tracker.record()
    assert r['peak_rate_deg_s'] == pytest.approx([180,45,360])
    assert r['angular_travel_deg'] == pytest.approx([360,90,720])
    assert r['excess_rotation_deg'] == pytest.approx([180,0,360])
    assert r['time_above_limit_s'] == pytest.approx([2,0,2])
    assert r['excess_cost_s'] == pytest.approx(1.)
    assert compute_excess_rotation_cost(SimpleNamespace(excess_rotation_cost_step_s=tracker.step_cost_s), {}).item() == pytest.approx(1.)


def test_rate_reversal_does_not_cancel_travel_and_crossings_are_interpolated():
    tracker = RotationTracker(1, 'cpu')
    tracker.update(rates(-180,0,0), rates(180,0,0), 2., torch.tensor([True]))
    r = tracker.record()
    assert r['angular_travel_deg'][0] == pytest.approx(180)
    assert r['excess_rotation_deg'][0] == pytest.approx(45)
    assert r['time_above_limit_s'][0] == pytest.approx(1.)


def test_substep_peak_is_retained_when_control_step_ends_at_zero():
    tracker = RotationTracker(1, 'cpu')
    zero, high = rates(0,0,0), rates(0,0,720)
    tracker.update(zero, high, .01, torch.tensor([True]))
    tracker.update(high, zero, .01, torch.tensor([True]))
    assert tracker.record()['peak_rate_deg_s'][2] == pytest.approx(720)
    assert tracker.record()['angular_travel_deg'][2] == pytest.approx(7.2)
    assert tracker.step_cost_s.item() > 0


def test_terminal_rotation_quality_uses_worst_whole_flight_axis():
    peak = rates(45, 90, 720).expand(2, 3)
    state = SimpleNamespace(
        rotation_peak_rate_rad_s=peak,
        contact_state=torch.tensor([ContactState.LANDED, ContactState.AIRBORNE]),
        position=torch.zeros(2, 3),
        touchdown_speed=torch.zeros(2),
        mission_ready_to_land=torch.ones(2, dtype=torch.bool),
    )
    score = compute_rotation_quality_reward(state, {'task': {'rotation': {
        'soft_limits_deg_s': [90, 90, 180]}}})
    assert score.tolist() == pytest.approx([.25, 0.])


def test_pre_reset_snapshot_is_independent_and_post_terminal_updates_are_masked():
    tracker = RotationTracker(2, 'cpu')
    w = rates(180,0,0).expand(2,3)
    tracker.update(w, w, 1., torch.tensor([True,False]))
    snapshot = tracker.snapshot()
    tracker.reset(torch.tensor([0]))
    assert snapshot['angular_travel_rad'][0,0].item() == pytest.approx(math.pi)
    assert tracker.episode['angular_travel_rad'].sum().item() == 0
    assert tracker.step_cost_s.sum().item() == 0


def test_refining_constant_rate_clock_preserves_integrals_and_cost_budget():
    def accumulate(hz):
        tracker = RotationTracker(1, 'cpu')
        w = rates(2000,2000,2000)
        for _ in range(hz):
            tracker.begin_step()
            tracker.update(w, w, 1/hz, torch.tensor([True]))
        return tracker.record()
    coarse, fine = accumulate(30), accumulate(240)
    for key in ('angular_travel_deg','excess_rotation_deg','time_above_limit_s','excess_cost_s'):
        assert coarse[key] == pytest.approx(fine[key], rel=1e-5)
    assert 0 <= fine['excess_cost_s'] <= 1


@pytest.mark.parametrize('limits', [(0,90,180),(90,float('nan'),180),(90,180)])
def test_invalid_limits_are_rejected(limits):
    with pytest.raises(ValueError):
        RotationTracker(1, 'cpu', limits)
