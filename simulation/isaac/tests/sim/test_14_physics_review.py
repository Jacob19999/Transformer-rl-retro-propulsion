"""PhysX checks of net lift, vane reaction, full spool torque and soft contact."""
import torch


def run():
    from tvc_env.envs.base_env import BaseEnvConfig
    from tvc_env.envs.direct_rl_env import TVCDirectRLEnv
    from tvc_env.common.constants import ContactState
    cfg = BaseEnvConfig('landing', overrides={
        'env': {'num_envs': 4, 'reset_on_crash': False},
        'task': {'spawn': {'position_range': [[0, 0, 5], [0, 0, 5]],
                           'velocity_range': [[0, 0, 0], [0, 0, 0]],
                           'attitude_range': [[0, 0, 0], [0, 0, 0]],
                           'curriculum': {'enabled': False}}}})
    env = TVCDirectRLEnv(cfg)
    try:
        env.reset(seed=2)
        hover = env.nominal_hover_throttle()
        env._reset_manager._omega_state[:] = hover * env._omega_max
        env._reset_manager._omega_prev[:] = env._reset_manager.omega_state
        action = torch.zeros(4, 5, device=env.device)
        action[:, 4] = hover
        for _ in range(10):
            obs, _, term, trunc, _ = env.step(action)
            assert not (term | trunc).any()
        velocity = env._body_iface.get_root_linear_velocity_world()
        assert velocity.abs().max() < .03, velocity
        debug = env._last_dynamics_debug
        mass = env._drone.root_physx_view.get_masses()[0].sum()
        net_lift = debug['edf_raw_thrust_N'] - debug['fin_force_body_frd_N'][:, 2]
        assert torch.allclose(net_lift.cpu(), torch.full((4,), float(mass) * 9.81), atol=.1)
        print(f'PASS equilibrium: throttle={hover:.6f}, max speed={float(velocity.abs().max()):.6f} m/s', flush=True)

        angles = torch.zeros(4, 4, device=env.device)
        angles[:, 0] = .05
        env._body_iface.write_fin_joint_state(angles, torch.zeros_like(angles))
        env._reset_manager._servo_state[:] = angles
        action[:, :4] = angles
        env.step(action)
        assert torch.all(env._last_dynamics_debug['fin_force_body_frd_N'][:, 1] > 0)
        assert torch.all(env._body_iface.get_angular_velocity_body_frd()[:, 0] < 0)
        assert torch.all(env._last_dynamics_debug['fin_torque_body_frd_Nm'][:, 2] > 0)
        print('PASS radial hinge: positive forward-fin rotation gives -roll and +yaw torque', flush=True)

        env.reset(seed=2)
        action.zero_()
        action[:, 4] = .5
        env.step(action)
        assert torch.all(env._last_dynamics_debug['edf_dynamic_torque_body_frd_Nm'][:, 2] < 0)
        assert torch.all(env._body_iface.get_angular_velocity_body_frd()[:, 2] < 0)
        print('PASS cold spool: body yaw reaction opposes rotor acceleration', flush=True)

        env.reset(seed=2)
        # Loaded collision geometry has neutral root-to-foot height 0.3124945 m.
        # A low, stationary drop tests actual PhysX contact, not an altitude fallback.
        positions = env._body_iface.get_root_position()
        positions[0, 2] = .314
        env._body_iface.set_root_state(positions, env._body_iface.get_root_quaternion_wxyz(),
                                      torch.zeros(4, 3, device=env.device), torch.zeros(4, 3, device=env.device))
        action.zero_()
        for _ in range(30):
            _, _, term, _, info = env.step(action)
            if bool(term[0]):
                break
        assert int(info['contact_state_pre_reset'][0]) == int(ContactState.LANDED), info
        assert float(info['touchdown_speed_pre_reset'][0]) < .25
        assert torch.all(info['contact_state_pre_reset'][1:] == int(ContactState.AIRBORNE))
        print(f"PASS soft contact and env isolation: impact={float(info['touchdown_speed_pre_reset'][0]):.6f} m/s", flush=True)
        return True
    finally:
        env.close()
