"""Body-frame response must be independent of heading; preserve rotor torques."""
import torch


def run():
    from tvc_env.envs.base_env import BaseEnvConfig
    from tvc_env.envs.direct_rl_env import TVCDirectRLEnv
    cfg=BaseEnvConfig('landing',env_config_path='configs/env/train_2048_8s_rotation.yaml',overrides={
        'env':{'num_envs':4,'decimation':1,'observe_battery':False},'battery':{'enabled':False},
        'dynamics':{'enable_wind_force':False,'enable_edf_static_torque':False,
                    'enable_edf_dynamic_torque':False,'enable_edf_gyro_torque':False},
        'task':{'spawn':{'position_range':[[0,0,5],[0,0,5]],'velocity_range':[[0,0,0],[0,0,0]],
            'attitude_range':[[0,0,0],[0,0,0]],'curriculum':{'enabled':False}}}})
    env=TVCDirectRLEnv(cfg)
    try:
        env.reset(seed=9)
        headings=torch.tensor([0.,.8,1.7,3.1],device=env.device)
        q=torch.zeros(4,4,device=env.device);q[:,0]=(headings/2).cos();q[:,3]=(headings/2).sin()
        env._body_iface.set_root_state(env._body_iface.get_root_position(),q,
            torch.zeros(4,3,device=env.device),torch.zeros(4,3,device=env.device))
        fins=torch.zeros(4,4,device=env.device);fins[:,0]=.1
        env._body_iface.write_fin_joint_state(fins,torch.zeros_like(fins))
        env._reset_manager._servo_state[:]=fins
        env._reset_manager._omega_state.fill_(env._omega_max)
        env._reset_manager._omega_prev.fill_(env._omega_max)
        act=torch.cat([fins,torch.ones(4,1,device=env.device)],dim=1)
        env.step(act)
        rates=env._body_iface.get_angular_velocity_body_frd()
        assert rates[0,0]<0 and rates[0,2]>0,rates
        assert (rates-rates[:1]).abs().max()<.00015,rates
        print(f'PASS heading-invariant fin wrench: {rates.cpu().tolist()}',flush=True)
        env.reset(seed=9)
        cfg.config['dynamics']['enable_edf_dynamic_torque']=True
        act.zero_();act[:,4]=.5
        env.step(act)
        rate=env._body_iface.get_angular_velocity_body_frd()[:,2]
        assert (rate<-.1).all(),rate
        print(f'PASS spool torque survives simultaneous off-COM force: {rate.cpu().tolist()}',flush=True)
        return True
    finally:env.close()
