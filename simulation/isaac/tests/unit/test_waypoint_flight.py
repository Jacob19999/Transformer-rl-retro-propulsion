import copy
import math
from pathlib import Path

import pytest
import torch
import yaml

from tvc_env.common.constants import ContactState
from tvc_env.common.quaternions import from_euler
from tvc_env.envs import waypoint_flight as wf
from tvc_env.envs.task_registry import load_merged_config

ROOT = Path(__file__).resolve().parents[2]
DT = 1 / 30


def config():
    env = yaml.safe_load((ROOT / 'configs/env/train_waypoint_flight.yaml').read_text())
    return load_merged_config('waypoint_flight', env_config=env, sim_root=ROOT)


def task(n=4, cfg=None, missions=None, start=None):
    cfg = cfg or config()
    flight = wf.WaypointFlightTask(n, 'cpu', cfg, torch.zeros(n, 3), DT)
    if missions is not None:
        flight.set_explicit_missions(missions)
    start = torch.tensor([[0., 0., 5.]] * n) if start is None else start
    flight.reset(torch.arange(n), start)
    return flight, start


def step(flight, before, after, velocity=None, rates=None, contact=None, touchdown=None,
         tilt=None, altitude=None):
    n = before.shape[0]
    zeros3 = torch.zeros(n, 3)
    return flight.step(before, after, zeros3 if velocity is None else velocity,
                       zeros3 if rates is None else rates,
                       torch.full((n,), int(ContactState.AIRBORNE)) if contact is None else contact,
                       torch.zeros(n) if touchdown is None else touchdown, torch.zeros(n),
                       torch.zeros(n, dtype=torch.bool) if tilt is None else tilt,
                       torch.zeros(n, dtype=torch.bool) if altitude is None else altitude)


def test_rule_two_budget_and_energy_weighted_efficiency():
    budget = wf.reward_budget(config(), hover_fraction=0.83)
    assert 0.7 <= budget['energy_share_at_hover'] <= 0.8
    assert budget['worst_episode_step_cost'] < -budget['failure']
    assert budget['worst_episode_step_cost'] < budget['mission_success']
    bad = config()
    bad['task']['reward']['failure'] = -50.0
    with pytest.raises(ValueError, match='Rule-2'):
        wf.reward_budget(bad, hover_fraction=0.83)


def test_reward_sign_contract():
    cfg = config()
    cfg['task']['reward']['energy'] = 1.3
    with pytest.raises(ValueError, match='wrong sign'):
        wf.reward_weights(cfg)


def test_every_curriculum_stage_is_a_valid_task_and_final_restores():
    cfg = config()
    final = wf.final_task_snapshot(cfg)
    for stage in cfg['task']['waypoint_flight']['curriculum']['stages']:
        wf.apply_stage(cfg, final, stage)
    wf.apply_stage(cfg, final, None)
    assert cfg['task']['spawn'] == final['spawn']
    assert cfg['task']['waypoint_flight']['generator'] == final['waypoint_flight']['generator']
    assert cfg['task']['episode_length_s'] == final['episode_length_s']
    with pytest.raises(ValueError, match='Unknown curriculum stage'):
        wf.apply_stage(cfg, final, {'reward': {}})


def test_random_missions_respect_generator_bounds():
    torch.manual_seed(0)
    n = 4096
    cfg = config()
    g = cfg['task']['waypoint_flight']['generator']
    start = torch.rand(n, 3) * torch.tensor([80., 80., 22.]) - torch.tensor([40., 40., -3.])
    flight, _ = task(n, cfg, start=start)
    lo, hi = g['count_range']
    assert int(flight.count.min()) == lo and int(flight.count.max()) == hi
    ids = torch.arange(n)
    final_kind = flight.kinds[ids, flight.count - 1]
    assert set(final_kind.unique().tolist()) == {wf.HOVER, wf.LAND}
    assert 0.3 < float((final_kind == wf.LAND).float().mean()) < 0.5
    slots = torch.arange(wf.MAX_WAYPOINTS)[None]
    valid = slots < flight.count[:, None]
    assert bool((flight.kinds[~valid] == wf.NONE).all())
    land = flight.kinds == wf.LAND
    assert int((land & (slots != (flight.count - 1)[:, None])).sum()) == 0
    airborne = valid & ~land
    z = flight.positions[..., 2]
    assert float(z[airborne].min()) >= g['altitude_m'][0] - 1e-5
    assert float(z[airborne].max()) <= g['altitude_m'][1] + 1e-5
    assert torch.allclose(z[land], torch.full_like(z[land], 0.3125))
    assert float(flight.positions[..., :2][valid].abs().max()) <= g['arena_half_width_m'] + 1e-4
    route = torch.cat((start[:, None], flight.positions), 1)
    legs = (route[:, 1:] - route[:, :-1]).norm(dim=-1)
    # Minimum length holds for airborne legs away from the arena wall.
    inside = flight.positions[..., :2].abs().max(-1).values < g['arena_half_width_m'] - 1e-3
    assert float(legs[airborne & inside].min()) >= g['min_leg_m'] - 1e-4
    hover_holds = flight.holds[flight.kinds == wf.HOVER]
    assert float(hover_holds.min()) >= g['hover_hold_s'][0] and float(hover_holds.max()) <= g['hover_hold_s'][1]


def test_flypass_capture_is_swept_and_advances_to_next_waypoint():
    mission = [dict(position=[10, 0, 5], type='flypass', radius_m=0.5),
               dict(position=[10, 10, 5], type='hover', radius_m=0.5, hold_s=1.0)]
    flight, start = task(2, missions=[mission])
    # One step jumps past the waypoint: a swept segment test still captures it.
    before = torch.tensor([[9., 0.2, 5.], [9., 3., 5.]])
    after = torch.tensor([[11., 0.2, 5.], [11., 3., 5.]])
    terminated = step(flight, before, after)
    assert flight.captured_now.tolist() == [True, False]
    assert flight.index.tolist() == [1, 0]
    assert not terminated.any()


def test_spinning_hover_is_not_a_hold_and_hold_completes_mission():
    mission = [dict(position=[0, 0, 5], type='hover', radius_m=0.5, hold_s=0.5)]
    flight, start = task(2, missions=[mission])
    spin = torch.tensor([[0., 0., 3.0], [0., 0., 0.]])  # 172 deg/s yaw vs 60 deg/s gate
    terminated = torch.zeros(2, dtype=torch.bool)
    for _ in range(int(0.5 / DT) + 1):
        terminated = step(flight, start, start, rates=spin)
    assert flight.outcome.tolist() == [wf.RUNNING, wf.SUCCESS]
    assert terminated.tolist() == [False, True]
    assert flight.success_now.tolist() == [False, True]


def test_landing_gates_and_premature_landing():
    land = [dict(position=[0, 0, 5], type='hover', radius_m=1.0, hold_s=0.0),
            dict(position=[0, 0, 0], type='land', radius_m=0.5)]
    flight, start = task(4, missions=[land])
    step(flight, start, start)  # zero hold -> captured, now on LAND
    assert flight.index.tolist() == [1, 1, 1, 1]
    ground = torch.tensor([[0.1, 0, .3125], [0.1, 0, .3125], [0.9, 0, .3125], [0., 0, .3125]])
    landed = torch.full((4,), int(ContactState.LANDED))
    touchdown = torch.tensor([0.2, 0.4, 0.1, 0.2])
    contact = landed.clone()
    contact[3] = int(ContactState.CRASHED)
    step(flight, ground, ground, contact=contact, touchdown=touchdown)
    assert flight.outcome.tolist() == [wf.SUCCESS, wf.BAD_LANDING, wf.BAD_LANDING, wf.CRASH]
    assert float(flight.landing_quality_now[0]) > float(flight.landing_quality_now[1]) > 0
    # LANDED on a hover leg is a premature landing, never success.
    flight, start = task(1, missions=[[dict(position=[5, 0, 5], type='hover')]])
    step(flight, start, start, contact=torch.tensor([int(ContactState.LANDED)]))
    assert flight.outcome.tolist() == [wf.PREMATURE_LANDING]


def test_spin_geofence_and_tilt_classification():
    flight, start = task(4, missions=[[dict(position=[5, 0, 5], type='hover')]])
    rates = torch.zeros(4, 3)
    rates[0, 2] = math.radians(400)
    after = start.clone()
    after[1, 0] = 75.0  # arena 60 + margin 10
    tilt = torch.tensor([False, False, True, False])
    contact = torch.full((4,), int(ContactState.AIRBORNE))
    contact[2] = int(ContactState.CRASHED)  # CrashDetector marks excessive tilt CRASHED
    step(flight, start, after, rates=rates, contact=contact, tilt=tilt)
    assert flight.outcome.tolist() == [wf.SPIN, wf.GEOFENCE, wf.TILT, wf.RUNNING]
    assert flight.failed_now.tolist() == [True, True, True, False]


def test_dense_terms_sum_to_potential_change_without_terminal_refund():
    torch.manual_seed(1)
    mission = [dict(position=[6, 0, 5], type='hover', radius_m=1.0, hold_s=0.2),
               dict(position=[6, 6, 8], type='hover', radius_m=0.5, hold_s=0.2)]
    cfg = config()
    dense = ('progress', 'hover_settle', 'hover_hold')
    cfg['task']['reward'] = {k: (1.0 if k in dense else 0.0) for k in wf.REWARD_TERMS}
    flight, start = task(1, cfg, missions=[mission])
    psi0 = {k: float(flight.last_potential[k][0]) for k in dense}
    position, total, banked_after_first = start.clone(), 0.0, None
    for _ in range(600):
        target = flight.positions[0, flight.index[0]]
        after = position + (target - position[0]).clamp(-0.3, 0.3)
        index_before = int(flight.index[0])
        terminated = step(flight, position, after)
        total += float(flight.reward(after, torch.zeros(1, 3), terminated, torch.zeros(1, 5), torch.zeros(1)))
        if index_before == 0 and int(flight.index[0]) == 1:
            banked_after_first = float(flight.banked['hover_hold'][0])
        position = after
        if bool(terminated[0]):
            break
    assert flight.outcome.tolist() == [wf.SUCCESS]
    # A capture banks the completed hold, so it never reads as a loss.
    assert banked_after_first == pytest.approx(1.0)
    psi_t = {k: float(flight.last_potential[k][0]) for k in dense}
    assert total == pytest.approx(sum(psi_t[k] - psi0[k] for k in dense), rel=1e-4)
    assert psi_t['hover_hold'] == pytest.approx(2.0)          # two completed holds, no refund to zero


def test_failure_keeps_lost_progress():
    cfg = config()
    cfg['task']['reward'] = {k: (1.0 if k == 'progress' else 0.0) for k in wf.REWARD_TERMS}
    flight, start = task(1, cfg, missions=[[dict(position=[5, 0, 5], type='hover')]])
    away = start.clone(); away[0, 0] = -20.0                  # 20 m the wrong way: geofence
    terminated = step(flight, start, away)
    reward = float(flight.reward(away, torch.zeros(1, 3), terminated, torch.zeros(1, 5), torch.zeros(1)))
    assert flight.outcome.tolist() == [wf.GEOFENCE]
    assert reward == pytest.approx(-20.0, abs=1e-4)           # no terminal refund of the potential


def test_observation_is_body_frame_and_sized_to_contract():
    flight, start = task(2, missions=[[dict(position=[10, 0, 5], type='flypass', radius_m=1.0),
                                       dict(position=[10, 0, 1], type='land')]])
    # Env 1 is yawed +90 deg about world Z: the waypoint (world +X) is then to
    # its right in FRD (+Y); env 0 sees it straight ahead (+X).
    yaw = torch.tensor([0.0, math.pi / 2])
    q = from_euler(torch.zeros(2), torch.zeros(2), yaw)
    obs = flight.observation(start, q, torch.zeros(2, 3), torch.zeros(2, 3), start[:, 2],
                             torch.zeros(2, 4), torch.zeros(2, 4), torch.full((2,), 0.83),
                             torch.zeros(2, dtype=torch.long), torch.zeros(2, 4), torch.zeros(2, 5),
                             0.262, 6.98)
    assert obs.shape == (2, wf.OBS_DIM)
    assert obs[0, 0:3].tolist() == pytest.approx([0.5, 0.0, 0.0], abs=1e-5)
    assert obs[1, 0:3].tolist() == pytest.approx([0.0, 0.5, 0.0], abs=1e-5)
    assert obs[:, 9:12].tolist() == [[1, 0, 0]] * 2             # active: flypass
    assert obs[:, 12:16].tolist() == [[0, 0, 1, 0]] * 2         # next: land
    assert obs[:, 16:20].tolist() == [[0, 0, 0, 1]] * 2         # next2: none
    assert obs[0, 22:25].tolist() == pytest.approx([0, 0, 1], abs=1e-6)  # level: gravity along +Z_frd
    # Fine-scale copy of the target vector (2 m scale, saturated at 1.5).
    assert obs[0, 51:54].tolist() == pytest.approx([1.5, 0.0, 0.0], abs=1e-5)


def test_explicit_mission_validation():
    with pytest.raises(ValueError, match='only allowed as the final'):
        wf.parse_mission([dict(position=[0, 0, 0], type='land'), dict(position=[1, 0, 5], type='hover')],
                         .3125, 1.0)
    with pytest.raises(ValueError, match='must be hover or land'):
        wf.parse_mission([dict(position=[0, 0, 5], type='flypass')], .3125, 1.0)
    with pytest.raises(ValueError, match='z >= 1.0'):
        wf.parse_mission([dict(position=[0, 0, 0.5], type='hover')], .3125, 1.0)
    parsed = wf.parse_mission([dict(position=[3, 4, 9], type='land')], .3125, 1.0)
    assert parsed['positions'][0].tolist() == pytest.approx([3, 4, .3125])


def test_curriculum_can_relax_the_hover_rate_gate():
    cfg = config()
    final = wf.final_task_snapshot(cfg)
    mission = [dict(position=[0, 0, 5], type='hover', radius_m=0.5, hold_s=0.2)]
    spin = torch.tensor([[0., 0., math.radians(100)]])
    for stage, expected in (({'hover_capture': {'max_body_rate_deg_s': 240.0}}, wf.SUCCESS), (None, wf.RUNNING)):
        wf.apply_stage(cfg, final, stage)
        flight, start = task(1, cfg, missions=[mission])
        for _ in range(int(0.2 / DT) + 1):
            step(flight, start, start, rates=spin)
        assert flight.outcome.tolist() == [expected]


def test_geofence_is_the_route_bounding_box_plus_margin():
    mission = [dict(position=[10, 0, 8], type='flypass', radius_m=1.0), dict(position=[10, 5, 6], type='hover')]
    flight, start = task(3, missions=[mission])      # start [0, 0, 5]; box x 0..10, y 0..5, z 5..8
    after = start.clone()
    after[0, 2] = 17.9                                # 8 + 10 m margin: inside
    after[1, 2] = 18.1                                # above the mission ceiling
    after[2, 1] = -10.5                               # beyond y_min - 10 m
    step(flight, start, after)
    assert flight.outcome.tolist() == [wf.RUNNING, wf.GEOFENCE, wf.GEOFENCE]


def test_throttle_action_is_centred_on_hover_duty():
    raw = torch.tensor([[1.0, -1.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, -1.0]])
    action = wf.policy_to_env_action(raw, 0.262, 0.78, 0.25)
    assert action[0, :4].tolist() == pytest.approx([0.262, -0.262, 0.131, 0.0])
    assert action[:, 4].tolist() == pytest.approx([0.78, 1.0, 0.53])   # 1.03 clamps to 1.0
