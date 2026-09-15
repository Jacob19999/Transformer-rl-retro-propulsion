"""Independent frame, momentum and terminal-event regression checks."""

from types import SimpleNamespace

import torch

from tvc_env.common.constants import ContactState
from tvc_env.common.frames import isaac_velocity_to_frd
from tvc_env.common.quaternions import from_euler, inverse, rotate_vector
from tvc_env.controllers.pid_adapter import PIDController
from tvc_env.controllers.pid_fin_mixer import PIDFinMixer
from tvc_env.envs.rewards import compute_vertical_speed_shaping, compute_drift_penalty, compute_landing_descent_tracking
from tvc_env.sim.contacts import ContactStateMachine
from tvc_env.sim.link_force_interface import LinkForceInterface


def test_crash_wins_on_last_contact_dwell_frame():
    sm = ContactStateMachine(1, dwell_frames=2)
    sm.update(torch.tensor([True]), torch.tensor([False]), torch.tensor([20.0]))
    state = sm.update(torch.tensor([True]), torch.tensor([True]), torch.tensor([20.0]))
    assert state.item() == ContactState.CRASHED


def test_airborne_safety_termination_receives_failure_penalty():
    from tvc_env.envs.rewards import compute_crash_penalty
    from tvc_env.envs.terminations import check_failure_terminations
    state = SimpleNamespace(
        position=torch.tensor([[0., 0., 30.15], [4., 0., 1.], [8., 0., .3125], [12., 0., 1.]]),
        quaternion_wxyz=from_euler(torch.tensor([0., 1.6, 0., 0.]), torch.zeros(4), torch.zeros(4)),
        contact_state=torch.tensor([ContactState.AIRBORNE, ContactState.AIRBORNE,
                                    ContactState.LANDED, ContactState.AIRBORNE]),
    )
    targets = torch.tensor([[0., 0., 0.], [4., 0., 0.], [8., 0., 0.], [12., 0., 0.]])
    config = {'_target_position_world': targets, 'task': {'termination': {'max_altitude_error': 30.}}}
    expected = torch.tensor([1., 1., 0., 0.])
    torch.testing.assert_close(compute_crash_penalty(state, config), expected)
    torch.testing.assert_close(check_failure_terminations(state.quaternion_wxyz, state.position,
                                                         targets, state.contact_state, config).float(), expected)


def test_bounce_does_not_erase_earlier_impact_speed():
    from tvc_env.envs.direct_rl_env import TVCDirectRLEnv
    false = lambda *args: torch.tensor([False])
    env = SimpleNamespace(
        _contact_sm=ContactStateMachine(1, dwell_frames=3),
        _touchdown_speed=torch.zeros(1),
        _body_iface=SimpleNamespace(get_angular_velocity_body_frd=lambda: torch.zeros(1, 3),
                                   get_root_quaternion_wxyz=lambda: torch.tensor([[1., 0., 0., 0.]])),
        _crash_detector=SimpleNamespace(check_impact_speed=false, check_tilt_at_contact=false,
                                         check_angular_rate_at_contact=false, check_excessive_tilt=false),
    )
    update = lambda force, speed: TVCDirectRLEnv._update_contact_state(
        env, torch.tensor([force]), torch.tensor([False]), torch.tensor([speed]))
    update(20., 1.5)
    update(0., 0.)  # leave contact
    update(20., .01)  # gentle re-contact cannot erase the hard arrival
    assert env._touchdown_speed.item() == 1.5
    update(0., 0.)
    update(20., 2.)  # a worse subsequent impact must also count
    assert env._touchdown_speed.item() == 2.


def test_pid_pure_vertical_descent_does_not_command_lateral_translation():
    q = from_euler(torch.tensor([0.15]), torch.tensor([-0.2]), torch.tensor([1.3]))
    obs = torch.zeros(1, 24)
    obs[:, 2] = -18.0
    obs[:, 3:7] = q
    obs[:, 7:10] = isaac_velocity_to_frd(rotate_vector(inverse(q), torch.tensor([[0.0, 0.0, -4.0]])))
    pid = PIDController()
    pid.compute_action(obs)
    debug = pid.get_debug_state()
    assert abs(debug['alt_vel_down'] - 4.0) < 1e-5
    assert torch.tensor(debug['desired_attitude_rp_unsat']).abs().max() < 1e-6


def test_radial_hinges_allocate_yaw_as_common_mode():
    mixer = PIDFinMixer()
    actual = mixer.mix(torch.tensor([0.04]), torch.tensor([0.02]), torch.tensor([.03]))
    no_yaw = mixer.mix(torch.tensor([0.04]), torch.tensor([0.02]), torch.tensor([0.0]))
    torch.testing.assert_close(actual - no_yaw, torch.full((1,4), .03))


def test_descent_and_drift_rewards_are_ground_relative():
    state = SimpleNamespace(
        linear_vel_world=torch.tensor([[0.0, 0.0, -5.0], [0.0, 0.0, -5.0]]),
        linear_vel_frd=torch.tensor([[0.0, 0.0, 5.0], [5.0, 0.0, 0.0]]),
    )
    assert torch.allclose(compute_vertical_speed_shaping(state, {}), torch.tensor([4.5, 4.5]))
    assert torch.equal(compute_drift_penalty(state, {}), torch.zeros(2))


def test_cop_follows_fin_rotation_about_its_hinge():
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    pos = torch.tensor([[2.0, 3.0, 5.0]])
    data = SimpleNamespace(body_link_quat_w=q[:, None].repeat(1, 4, 1), body_link_pos_w=pos[:, None].repeat(1, 4, 1))
    art = SimpleNamespace(data=data)
    iface = LinkForceInterface(art, SimpleNamespace(fin_body_indices=[0, 1, 2, 3]), torch.tensor([[0.0, 0.0, 0.2]]).repeat(4, 1))
    neutral = iface.get_fin_cop_positions_world(q, pos)
    assert torch.allclose(neutral, (pos + torch.tensor([0.0, 0.0, -0.2]))[:, None].repeat(1, 4, 1))
    fin_q = from_euler(torch.zeros(1), torch.tensor([torch.pi / 2]), torch.zeros(1))
    data.body_link_quat_w = fin_q[:, None].repeat(1, 4, 1)
    rotated = iface.get_fin_cop_positions_world(q, pos)
    assert torch.allclose(rotated, (pos + torch.tensor([-0.2, 0.0, 0.0]))[:, None].repeat(1, 4, 1), atol=1e-6)


def test_descent_reward_tracks_world_speed_and_brakes_at_foot_clearance():
    state = SimpleNamespace(
        position=torch.tensor([[0., 0., .3125], [0., 0., .6125], [0., 0., 18.]]),
        linear_vel_world=torch.tensor([[0., 0., -.15], [0., 0., -(.15**2 + 1.6*.3)**.5], [0., 0., -1.]]),
        # Body speed intentionally unrelated: descent must use the ground frame.
        linear_vel_frd=torch.full((3, 3), 50.),
    )
    torch.testing.assert_close(compute_landing_descent_tracking(state, {}), torch.zeros(3), atol=1e-6, rtol=0)
    state.linear_vel_world[:, 2] = 100.
    torch.testing.assert_close(compute_landing_descent_tracking(state, {}), torch.full((3,), 3.))


def test_landing_reward_episode_budget_does_not_outweigh_terminals():
    from pathlib import Path
    import yaml
    from tvc_env.envs.reward_registry import compute_total_reward
    cfg = yaml.safe_load((Path(__file__).resolve().parents[2] / 'configs/tasks/landing.yaml').read_text())
    state = SimpleNamespace(
        position=torch.tensor([[2., 0., 18.]]),
        quaternion_wxyz=from_euler(torch.tensor([.15]), torch.zeros(1), torch.zeros(1)),
        linear_vel_world=torch.zeros(1, 3), angular_vel_frd=torch.tensor([[1., 0., 0.]]),
        fin_angles=torch.full((1, 4), .1), fin_rates=torch.ones(1, 4),
        motor_omega=torch.tensor([.9*4300]), touchdown_speed=torch.zeros(1),
        contact_state=torch.tensor([ContactState.AIRBORNE]),
    )
    weights = cfg['task']['reward']
    episode_cost = -float(compute_total_reward(weights, state, cfg)) * 900
    assert 150 < episode_cost < 155
    assert episode_cost < abs(weights['crash_penalty'])
    state.position.zero_()
    state.contact_state[:] = ContactState.LANDED
    terminal_reward = float(compute_total_reward(weights, state, cfg))
    assert terminal_reward > 3 * episode_cost
    # Pad accuracy cannot make a landing outside the impact-speed gate pay.
    state.touchdown_speed[:] = cfg['task']['success']['max_touchdown_speed'] + .001
    assert float(compute_total_reward(weights, state, cfg)) < 0
