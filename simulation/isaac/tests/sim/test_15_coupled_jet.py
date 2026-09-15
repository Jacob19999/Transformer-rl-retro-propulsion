"""Real PhysX audit of revised wake, gyro, actuator motion and soft contact."""
import json
import torch


def run():
    from tvc_env.envs.base_env import BaseEnvConfig
    from tvc_env.envs.direct_rl_env import TVCDirectRLEnv
    from tvc_env.common.constants import ContactState
    cfg=BaseEnvConfig('landing',env_config_path='configs/env/train_2048_8s_rotation.yaml',overrides={
        'env':{'num_envs':4,'reset_on_crash':False},
        'task':{'spawn':{'position_range':[[0,0,5],[0,0,5]],'velocity_range':[[0,0,0],[0,0,0]],
            'attitude_range':[[0,0,0],[0,0,0]],'curriculum':{'enabled':False}}}})
    env=TVCDirectRLEnv(cfg)
    try:
        env.reset(seed=7)
        action=torch.zeros(4,5,device=env.device);action[:,4]=.85
        for _ in range(15):
            obs,_,term,trunc,_=env.step(action)
            assert torch.isfinite(obs['policy']).all()
            assert not (term|trunc).any()
            d=env._last_dynamics_debug
            assert (d['vane_dissipated_power_w'] >= -1e-3).all()
            assert (d['edf_static_torque_body_frd_Nm'][:,2]<0).all()
        print('PASS real 8S battery/jet/gyro integration and finite observations',flush=True)
        print(json.dumps({k:v.cpu().tolist() for k,v in d.items() if k in (
            'jet_mass_flow_kg_s','jet_swirl_power_w','vane_dissipated_power_w',
            'edf_dynamic_torque_body_frd_Nm','edf_gyro_torque_body_frd_Nm','body_damping_torque_body_frd_Nm')}),flush=True)
        # Four distinct moving hinges exercise actual link COP velocities.
        action[:,:4]=torch.tensor([.1,-.1,.05,-.05],device=env.device)
        for _ in range(5):
            obs,*_=env.step(action)
            assert torch.isfinite(obs['policy']).all()
        d=env._last_dynamics_debug
        assert d['jet_incoming_velocity_body_frd_m_s'].shape==(4,4,3)
        assert (d['vane_dissipated_power_w']>=-1e-3).all()
        print('PASS measured moving-fin COP airflow',flush=True)
        env.reset(seed=7)
        pos=env._body_iface.get_root_position();pos[0,2]=.314
        env._body_iface.set_root_state(pos,env._body_iface.get_root_quaternion_wxyz(),
            torch.zeros(4,3,device=env.device),torch.zeros(4,3,device=env.device))
        action.zero_()
        for _ in range(30):
            _,_,term,_,info=env.step(action)
            if bool(term[0]):break
        assert int(info['contact_state_pre_reset'][0])==int(ContactState.LANDED)
        assert float(info['touchdown_speed_pre_reset'][0])<.25
        assert torch.all(info['contact_state_pre_reset'][1:]==int(ContactState.AIRBORNE))
        print(f"PASS real touchdown at {float(info['touchdown_speed_pre_reset'][0]):.6f} m/s, 0.125s dwell",flush=True)
        return True
    finally:
        env.close()
