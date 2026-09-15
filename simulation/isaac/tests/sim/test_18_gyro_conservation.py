"""Measure unforced gyro energy/momentum drift and permit timestep refinement."""
import json
import os
from pathlib import Path
import torch


def combined_midpoint_experiment(omega, body_angular_vel, rotor_inertia, spin_axis, body_inertia, dt):
    """Experimental whole-body Euler/rotor midpoint with explicit Euler correction.

    Used only when TVC_GYRO_TEST_METHOD=combined_midpoint. This is an integrator
    comparison, not a policy or training-plant modification.
    """
    def cross_matrix(v):
        result = v.new_zeros((v.shape[0],3,3))
        result[:,0,1] = -v[:,2]; result[:,0,2] = v[:,1]
        result[:,1,0] = v[:,2]; result[:,1,2] = -v[:,0]
        result[:,2,0] = -v[:,1]; result[:,2,1] = v[:,0]
        return result
    inertia = body_inertia.to(body_angular_vel)
    h = spin_axis.to(body_angular_vel)[None]*(rotor_inertia*omega[:,None])
    old = body_angular_vel
    midpoint = old.clone()
    for _ in range(4):
        iw = (inertia @ midpoint.unsqueeze(-1)).squeeze(-1)
        f = -torch.linalg.cross(midpoint, iw+h)
        residual = (inertia @ (midpoint-old).unsqueeze(-1)).squeeze(-1)-.5*dt*f
        derivative = cross_matrix(iw+h)-cross_matrix(midpoint)@inertia
        jacobian = inertia-.5*dt*derivative
        midpoint -= torch.linalg.solve(jacobian,residual.unsqueeze(-1)).squeeze(-1)
    iw = (inertia @ midpoint.unsqueeze(-1)).squeeze(-1)
    old_iw = (inertia @ old.unsqueeze(-1)).squeeze(-1)
    return -torch.linalg.cross(midpoint,iw+h)+torch.linalg.cross(old,old_iw)


def inverse_projection_experiment(omega, body_angular_vel, rotor_inertia, spin_axis, body_inertia, dt):
    tau = combined_midpoint_experiment(omega,body_angular_vel,rotor_inertia,spin_axis,body_inertia,dt)
    old = body_angular_vel
    old_iw = (body_inertia @ old.unsqueeze(-1)).squeeze(-1)
    b = dt*torch.linalg.solve(body_inertia,(-torch.linalg.cross(old,old_iw)).unsqueeze(-1)).squeeze(-1)
    desired = old + b + dt*torch.linalg.solve(body_inertia,tau.unsqueeze(-1)).squeeze(-1)
    desired_l = (body_inertia @ desired.unsqueeze(-1)).squeeze(-1)
    ib = (body_inertia @ b.unsqueeze(-1)).squeeze(-1)
    a = desired_l.square().sum(-1)
    dot = (desired_l*ib).sum(-1)
    scale = (dot + (dot.square()+a*(a-ib.square().sum(-1))).clamp(min=0).sqrt())/a.clamp(min=1e-12)
    scale = torch.where(a>1e-12,scale,torch.ones_like(scale))
    return tau + ((scale-1)[:,None]*desired_l)/dt


def run():
    from tvc_env.common.frames import frd_to_isaac
    from tvc_env.common.quaternions import rotate_vector, normalize
    from tvc_env.envs.base_env import BaseEnvConfig
    from tvc_env.envs.direct_rl_env import TVCDirectRLEnv
    hz = int(os.environ.get('TVC_GYRO_TEST_HZ', '240'))
    stress = os.environ.get('TVC_GYRO_TEST_STRESS') == '1'
    spin = float(os.environ.get('TVC_GYRO_TEST_SPIN_RAD_S', '12'))
    rotor_fraction = float(os.environ.get('TVC_GYRO_TEST_ROTOR_FRACTION', '.9'))
    roll = float(os.environ.get('TVC_GYRO_TEST_ROLL_RAD_S','.2'))
    pitch = float(os.environ.get('TVC_GYRO_TEST_PITCH_RAD_S','.1'))
    method = os.environ.get('TVC_GYRO_TEST_METHOD', 'rotor_only')
    every_iteration = os.environ.get('TVC_GYRO_TEST_EVERY_ITERATION', '1') == '1'
    if method == 'combined_midpoint':
        import tvc_env.dynamics.rotor_reaction as rotor
        rotor.compute_midpoint_gyroscopic_torque = combined_midpoint_experiment
    if method == 'inverse_projection':
        import tvc_env.dynamics.rotor_reaction as rotor
        rotor.compute_midpoint_gyroscopic_torque = inverse_projection_experiment
    cfg=BaseEnvConfig('landing',env_config_path='configs/env/train_512_8s_momentum.yaml',overrides={
        'physics': {'enable_external_forces_every_iteration': every_iteration},
        'env':{'num_envs':4,'physics_dt':1/hz,'decimation':1,'observe_battery':False,'reset_on_crash':False},'battery':{'enabled':False},
        'dynamics':{'enable_wind_force':False,'enable_fin_forces':False,'enable_edf_static_torque':False,
                    'gyro_integration': 'coupled_midpoint' if method == 'production_coupled' else 'rotor_midpoint',
                    'enable_edf_dynamic_torque':False},
        'task':{'spawn':{'position_range':[[0,0,100],[0,0,100]],'velocity_range':[[0,0,0],[0,0,0]],
            'attitude_range':[[0,0,0],[0,0,0]],'curriculum':{'enabled':False}}}})
    env=TVCDirectRLEnv(cfg)
    try:
        # This is a conservation test, so remove the aerodynamic crossflow
        # torque as well as wind/vane forces; it otherwise dissipates energy.
        import tvc_env.dynamics.coupled_jet as jet_module
        original_drag = jet_module.cylinder_rotational_drag
        jet_module.cylinder_rotational_drag = lambda rates, *args, **kwargs: torch.zeros_like(rates)
        env.reset(seed=4)
        # 0.2 / 0.1 rad/s in FRD; gravity/thrust act axially and cannot supply
        # angular work at the centered COM. No fan spool or vane loads.
        w=torch.tensor([[roll,-pitch,0.]]*4,device=env.device)
        if stress:
            w[:,2] = w.new_tensor([0., -spin, spin, -3.])
        env._body_iface.set_root_state(env._body_iface.get_root_position(),env._body_iface.get_root_quaternion_wxyz(),
            torch.zeros_like(w),w)
        env._reset_manager._omega_state.fill_(rotor_fraction*env._omega_max)
        env._reset_manager._omega_prev.fill_(rotor_fraction*env._omega_max)
        act=torch.zeros(4,5,device=env.device);act[:,4]=rotor_fraction
        inertia=env._locked_body_inertia
        def invariants():
            rates = env._body_iface.get_angular_velocity_body_frd()
            momentum = (inertia @ rates.unsqueeze(-1)).squeeze(-1)
            energy = .5*(rates*momentum).sum(-1)
            rotor_h = env._edf_model.thrust_axis.to(rates)[None] * (
                env._edf_model.rotor_inertia*env._reset_manager._omega_state[:,None])
            momentum_w = rotate_vector(normalize(env._body_iface.get_root_quaternion_wxyz()),
                frd_to_isaac(momentum+rotor_h))
            return energy, momentum_w, rates
        before, h_before, initial_rates = invariants()
        # A counter-rotating body can nearly cancel rotor momentum. Report
        # relative-to-net error, but condition the tolerance on the sum of
        # component magnitudes so the test is defined even at zero net H.
        h_scale = ((inertia @ initial_rates.unsqueeze(-1)).squeeze(-1).norm(dim=-1)
                   + env._edf_model.rotor_inertia*env._reset_manager.omega_state.abs())
        env._pre_physics_step(act)
        for _ in range(2*hz):
            env._apply_action();env._sim_scene.step()
            if _ == 0:
                first_rates = env._body_iface.get_angular_velocity_body_frd().cpu().tolist()
        after, h_after, final_rates = invariants()
        h_error = (h_after-h_before).norm(dim=-1)/h_before.norm(dim=-1)
        h_scaled_error = (h_after-h_before).norm(dim=-1)/h_scale.clamp(min=1e-9)
        report=dict(physics_dt=cfg.physics_dt,duration_s=2.,stress=stress,spin_rad_s=spin,method=method,
            rotor_fraction=rotor_fraction, first_rates=first_rates, every_iteration=every_iteration,
            roll_rad_s=roll,pitch_rad_s=pitch,
            rotational_energy_ratio=(after/before).cpu().tolist(),
            transverse_rate_squared_ratio=(final_rates[:,:2].square().sum(-1)/initial_rates[:,:2].square().sum(-1)).cpu().tolist(),
            world_momentum_relative_error=h_error.cpu().tolist(),
            world_momentum_absolute_error_Nms=(h_after-h_before).norm(dim=-1).cpu().tolist(),
            world_momentum_component_scaled_error=h_scaled_error.cpu().tolist(),
            initial_rates=initial_rates.cpu().tolist(),final_rates=final_rates.cpu().tolist())
        print(json.dumps(report),flush=True)
        path=Path('runs/mission_control/gyro_conservation_latest.json')
        path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        path.with_name(f'gyro_conservation_{hz}hz_stress{int(stress)}_spin{spin:g}_rotor{rotor_fraction:g}_{method}_every{int(every_iteration)}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        assert torch.all((after/before-1).abs()<.03),report
        assert torch.all(h_scaled_error<.002),report
        if stress:
            assert max(abs(value-1) for value in report['transverse_rate_squared_ratio']) < .1, report
        return True
    finally:
        jet_module.cylinder_rotational_drag = original_drag
        env.close()
