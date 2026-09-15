"""Real Isaac integration of mission observations, inverted starts and contact gates."""
import torch


def run():
    from tvc_env.envs.base_env import BaseEnvConfig
    from tvc_env.envs.direct_rl_env import TVCDirectRLEnv
    from tvc_env.common.quaternions import tilt_angle
    from tvc_env.common.constants import ContactState
    from tvc_env.controllers.ppo_model import ActorCritic
    cfg=BaseEnvConfig('landing',env_config_path='configs/env/train_2048_8s_waypoints.yaml',overrides={
        'env':{'num_envs':4,'reset_on_crash':False},
        'task':{'spawn':{'position_range':[[0,0,50]]*2,'velocity_range':[[0,0,0]]*2,
            'attitude_range':[[3.14159265,0,0]]*2,'angular_velocity_range':[[6.28,0,0]]*2,
            'waypoint_count_range':[0,0],'initial_motor_omega_fraction':0.,'curriculum':{'enabled':False}}}})
    env=TVCDirectRLEnv(cfg)
    try:
        obs=env.reset(seed=9717)[0]['policy']
        assert obs.shape==(4,43) and env.observation_space.shape==(43,)
        assert (tilt_angle(env._body_iface.get_root_quaternion_wxyz())>3.13).all()
        source=ActorCritic(28).to(env.device)
        model=ActorCritic(43).to(env.device);model.initialize_actor(source.state_dict())
        assert torch.allclose(source.act(obs[:,:28]),model.act(obs),atol=1e-6)
        action=torch.zeros(4,5,device=env.device);action[:,4]=.4
        obs,reward,terminated,truncated,info=env.step(action)
        assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
        assert not (terminated|truncated).any(), 'Inverted airborne state must be recoverable in task definition'
        assert (info['rotation_pre_reset']['peak_rate_rad_s'][:,0]>6.).all()
        print('PASS 43D actor transfer; inverted/high-roll-rate state survives first real step; rotation recorded',flush=True)
        cfg.config['task']['spawn'].update(position_range=[[0,0,.314]]*2,velocity_range=[[0,0,0]]*2,
            attitude_range=[[0,0,0]]*2,angular_velocity_range=[[0,0,0]]*2)
        cfg.config['task']['navigation']['waypoints']=[dict(position=[1,0,5],type='hover',hold_s=2.,radius_m=.5)]
        env.reset(seed=9718);action.zero_()
        for _ in range(30):
            _,reward,terminated,_,info=env.step(action)
            if bool(terminated.all()):break
        assert (info['contact_state_pre_reset']==int(ContactState.LANDED)).all()
        assert not info['mission_ready_to_land_pre_reset'].any()
        assert (reward < -900).all(),reward
        print('PASS real soft contact before hover waypoint is a failed mission with no landing payout',flush=True)
        return True
    finally:
        env.close()
