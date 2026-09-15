"""Real off-center COM must feel EDF thrust-line torque, not a COM-applied force."""
import torch


def run():
    from tvc_env.envs.base_env import BaseEnvConfig
    from tvc_env.envs.direct_rl_env import TVCDirectRLEnv
    cfg=BaseEnvConfig('landing',env_config_path='configs/env/train_2048_8s_rotation.yaml',overrides={
        'env':{'num_envs':4,'decimation':1,'observe_battery':False},'battery':{'enabled':False},
        'dynamics':{'enable_fin_forces':False,'enable_wind_force':False,
            'enable_edf_static_torque':False,'enable_edf_dynamic_torque':False,'enable_edf_gyro_torque':False},
        'task':{'spawn':{'position_range':[[0,0,5],[0,0,5]],'velocity_range':[[0,0,0],[0,0,0]],
            'attitude_range':[[0,0,0],[0,0,0]],'curriculum':{'enabled':False}}}})
    env=TVCDirectRLEnv(cfg)
    try:
        env.reset(seed=8)
        offsets=torch.tensor([[0.,0,0],[.01,0,0],[-.01,0,0],[0,.01,0]],device=env.device)
        env._body_iface.set_body_com_offset_frd(offsets,torch.arange(4,device=env.device))
        env._reset_manager._omega_state.fill_(env._omega_max)
        env._reset_manager._omega_prev.fill_(env._omega_max)
        act=torch.zeros(4,5,device=env.device);act[:,4]=1
        env.step(act)
        rates=env._body_iface.get_angular_velocity_body_frd()
        expected=48*.01/.05*cfg.physics_dt
        assert rates[0].norm()<.001,rates
        assert abs(float(rates[1,1])+expected)<.003,rates
        assert abs(float(rates[2,1])-expected)<.003,rates
        assert abs(float(rates[3,0])-expected)<.003,rates
        print(f'PASS thrust/COM torque: expected +/- {expected:.6f} rad/s after one step; actual {rates.cpu().tolist()}',flush=True)
        return True
    finally:env.close()
