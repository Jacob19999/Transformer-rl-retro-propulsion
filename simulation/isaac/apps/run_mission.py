"""One real Isaac mission, streaming replay poses and telemetry to disk."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from runner_safety import WallClockWatchdog, force_process_exit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mission_control.models import validate_mission, battery_config, hardware_overrides, disturbance_config, policy_paths


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=None,
                        help='Local diagnostic checkpoint override; does not publish or qualify a policy')
    parser.add_argument('--action-mode', choices=['stochastic','deterministic','mean'], default=None)
    args = parser.parse_args()
    request = validate_mission(json.loads(args.request.read_text()))
    if args.checkpoint is not None and request['controller'] == 'pid':
        raise ValueError('Checkpoint diagnostics require a PPO mission')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status = dict(state='starting', phase='Starting Isaac Sim', frames=0, sim_time_s=0., name=request['name'])
    def update(**values):
        status.update(values, wall_time_s=round(time.time() - started, 2))
        atomic_json(output / 'status.json', status)
    update()
    watchdog = WallClockWatchdog(480, label='Mission control simulation')
    watchdog.start()
    env = app = None
    try:
        from isaac_launcher import launch_simulation_app, close_simulation_app
        app = launch_simulation_app(headless=True)
        import torch
        from tvc_env.envs.base_env import BaseEnvConfig
        from tvc_env.envs.direct_rl_env import TVCDirectRLEnv
        from tvc_env.controllers.ppo_model import ActorCritic
        from tvc_env.common.frames import isaac_position_to_frd, frd_velocity_to_isaac
        from tvc_env.common.quaternions import normalize, inverse, rotate_vector, tilt_angle
        from tvc_env.common.constants import ContactState

        angles = [math.radians(x) for x in request['attitude_deg']]
        spawn = dict(position_range=[request['position']] * 2, velocity_range=[request['velocity']] * 2,
                     attitude_range=[angles] * 2, initial_motor_omega_fraction=request['initial_motor_fraction'])
        overrides = hardware_overrides(request)
        overrides.update(disturbance_config(request))
        overrides.update(env=dict(reset_on_crash=False, gizmos_enabled=False),
                         task=dict(spawn=spawn, episode_length_s=request['duration_s']),
                         battery=battery_config(request))
        policies = policy_paths()
        saved = None
        if request['controller'] in policies:
            checkpoint = ROOT / (args.checkpoint or policies[request['controller']])
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            obs_dim = saved['model']['actor.0.weight'].shape[1]
            if saved.get('args', {}).get('residual_pid') or saved.get('args', {}).get('landing_guidance'):
                raise ValueError('This mission runner requires a direct-action PPO checkpoint')
            overrides['env']['observe_battery'] = obs_dim == 28
            if saved.get('task_config', {}).get('dynamics', {}).get('coupled_jet', {}).get('enabled'):
                # A qualified policy must replay its actual training plant.
                # User-selected initial conditions/battery/disturbances remain
                # explicit overrides; never silently run new weights on legacy
                # damping or independent unlimited vane-force equations.
                from tvc_env.envs.task_registry import deep_merge
                trained = saved['task_config']
                plant = {key: trained[key] for key in ('dynamics','physics','env','task')}
                overrides = deep_merge(plant, overrides)
                overrides['env'].update(num_envs=1, replicate_physics=False)
                overrides['task']['spawn']['curriculum'] = {'enabled':False}
        torch.manual_seed(request['seed'])
        config = BaseEnvConfig(task_name='landing', sim_root=ROOT,
                               env_config_path=ROOT / 'configs/env/single_env_debug.yaml',
                               overrides=overrides)
        update(phase='Building physics scene')
        env = TVCDirectRLEnv(config)
        dt = config.physics_dt * config.decimation
        device = env.device
        model = pid = guidance = None
        policy_meta = dict(controller=request['controller'])
        if saved is not None:
            model = ActorCritic(obs_dim).to(device)
            model.load_state_dict(saved['model'])
            model.eval()
            policy_meta.update(checkpoint=str(checkpoint), step=saved['step'],
                               diagnostic_checkpoint_override=args.checkpoint is not None,
                               checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                               body_frame_position_error=saved['args'].get('body_frame_position_error', False),
                               observation_dim=obs_dim,
                               action_mode=args.action_mode or ('deterministic' if request['controller'] == 'ppo_deterministic' else 'stochastic'))
        else:
            from tvc_env.controllers.pid_adapter import PIDController
            from tvc_env.controllers.landing_guidance import LandingGuidance
            import yaml
            pid_settings = yaml.safe_load((ROOT / 'configs/controllers/pid_radial.yaml').read_text())
            hover = env.nominal_hover_throttle()
            pid = PIDController(**pid_settings['pid'], throttle_hover=hover, num_envs=1, device=device, dt=dt)
            guidance = LandingGuidance(1, device, env._target_position,
                                       **pid_settings['guidance'], throttle_hover=hover, dt=dt)
            policy_meta.update(guidance='Explicit PID landing guidance', parameters=pid_settings)
        obs = env.reset(seed=request['seed'])[0]['policy']
        # User input specifies gyro rates in body FRD; PhysX root writer takes world rates.
        rates = torch.tensor([[math.radians(x) for x in request['angular_rate_deg_s']]], device=device)
        initial_quat = env._body_iface.get_root_quaternion_wxyz()
        world_rates = rotate_vector(initial_quat, frd_velocity_to_isaac(rates))
        env._body_iface.set_root_state(env._body_iface.get_root_position(), initial_quat,
                                      torch.tensor([request['velocity']], device=device), world_rates)
        obs = env._get_observations()['policy']
        if guidance:
            pid.reset()
            guidance.reset(obs=obs)
        metadata = dict(schema_version=2, request=request, policy=policy_meta, dt=dt,
                        physics_dt=config.physics_dt, decimation=config.decimation,
                        hinge_layout='radial_span_v1',
                        asset_sha256=hashlib.sha256((ROOT / 'assets/usd/drone_v2_physics.usd').read_bytes()).hexdigest(),
                        battery_model=config.config['battery'],
                        hardware_profile=request['hardware_profile'],
                        disturbances=config.config['disturbances'],
                        fin_link_names=env._art_map.fin_link_names,
                        physics_parameters=env._resolved_hardware,
                        dynamics=config.config.get('dynamics', {}),
                        physics=config.config.get('physics', {}),
                        coordinate_frame='World XYZ, Z up; body rates FRD; quaternion wxyz',
                        source='Isaac Sim / PhysX', model_asset='drone.glb',
                        battery_calibration='Estimated 1-RC LiPo model; voltage and power coupled to EDF',
                        body_rate_command=None,
                        terminal_procedure='After LANDED: zero fin/throttle commands for 2 seconds; record actual spool-down and verify final settling. This procedure is outside the PPO episode.',
                        telemetry_notes='Fin command rate is servo target motion per control interval; body-rate targets are not commanded by these policies.',
                        initial_conditions_outside_training=(request['controller'] in policies and request['controller'] != 'ppo_radial') or request['position'][2] < 16 or request['position'][2] > 20 or request['battery']['initial_soc'] != 1.,
                        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for folder in ('apps', 'tvc_env', 'configs') for p in (ROOT / folder).rglob('*')
                                       if p.suffix in ('.py', '.yaml')})
        atomic_json(output / 'metadata.json', metadata)
        last_target = env._reset_manager.servo_state.clone()
        frames = 0
        cumulative_delta_v = 0.
        latest = None
        control_phase = 'CONTROLLER'
        landing_frame = None
        settled_after_shutdown = None
        with (output / 'frames.jsonl').open('w', encoding='utf-8', buffering=1) as stream:
            def capture(t, action, rate):
                nonlocal frames, latest
                state = env._build_vehicle_state()
                def vec(x):
                    return x[0].detach().cpu().tolist()
                art = env._drone.data
                ids = env._art_map.fin_body_indices
                debug = env._last_dynamics_debug if hasattr(env, '_last_dynamics_debug') else {}
                raw = float(env._edf_model.compute_thrust(state.motor_omega)[0])
                applied = float(debug['edf_applied_thrust_N'][0]) if debug else raw
                battery = {k: v[0].item() for k, v in env._battery_model.telemetry().items()} if env._battery_model else None
                frame = dict(t=t, control_phase=control_phase, position=vec(state.position), quaternion=vec(state.quaternion_wxyz),
                             velocity=vec(state.linear_vel_world), gyro=vec(state.angular_vel_frd),
                             observed_gyro=vec(obs[:, 10:13]),
                             fin_angles=vec(state.fin_angles), fin_rates=vec(state.fin_rates),
                             fin_commands=vec(action[:, :4]), fin_command_rates=vec(rate),
                             fin_positions=vec(art.body_pos_w[:, ids]), fin_quaternions=vec(art.body_quat_w[:, ids]),
                             throttle=float(action[0, 4]), rotor_rpm=float(state.motor_omega[0]) * 60 / (2 * math.pi),
                             thrust_n=applied, raw_thrust_n=raw, battery=battery,
                             contact=int(state.contact_state[0]), impact_speed=float(env._touchdown_speed[0]),
                             pad_distance=float((state.position[0, :2] - env._target_position[0, :2]).norm()),
                             contact_force_n=float(env._landing_contact_force_step[0]),
                             propulsive_delta_v_m_s=cumulative_delta_v,
                             rotation=env._rotation.record(),
                             body_rate_command=None)
                stream.write(json.dumps(frame, allow_nan=False) + '\n')
                frames += 1
                latest = frame
            capture(0., torch.zeros(1, 5, device=device), torch.zeros(1, 4, device=device))
            update(state='running', phase='Simulating', frames=frames)
            cancelled = False
            terminated = truncated = torch.tensor([False], device=device)
            with torch.no_grad():
                for step in range(math.ceil(request['duration_s'] / dt)):
                    if (output / 'STOP').exists():
                        cancelled = True
                        break
                    if model:
                        po = obs.clone()
                        if policy_meta['body_frame_position_error']:
                            po[:, :3] = isaac_position_to_frd(rotate_vector(inverse(normalize(obs[:, 3:7])), obs[:, :3]))
                        raw = model.act(po, policy_meta['action_mode'])
                        action = torch.cat([raw[:, :4] * env._servo_model.max_command_angle, (raw[:, 4:] + 1) / 2], dim=-1)
                    else:
                        if env._battery_model is not None and pid_settings['voltage_feedforward']:
                            # PID-only feedforward: account for the same loaded
                            # bus voltage used by the EDF's duty-to-speed model.
                            battery = env._battery_model
                            corrected_hover = (hover * battery.config['reference_voltage_v']
                                               / battery.voltage_v.clamp(min=1.)).clamp(0., 1.)
                            pid.throttle_hover = guidance.throttle_hover = corrected_hover
                        action = guidance.post_action(pid.compute_action(guidance.modify_obs(obs)), obs)
                    obs_dict, _, terminated, truncated, info = env.step(action)
                    cumulative_delta_v += float(info['propulsive_delta_v_step'][0])
                    obs = obs_dict['policy']
                    servo = env._reset_manager.servo_state
                    capture((step + 1) * dt, action, (servo - last_target) / dt)
                    last_target = servo.clone()
                    if step % 10 == 0:
                        update(frames=frames, sim_time_s=latest['t'])
                    env.render()
                    if bool((terminated | truncated)[0]):
                        break
                if latest['contact'] == int(ContactState.LANDED) and not cancelled:
                    # A finite PPO episode ends at physical LANDED. The flight
                    # executive then disarms explicitly; no controller alters
                    # the learned approach or generates a fake landing event.
                    landing_frame = latest
                    control_phase = 'POST_TOUCHDOWN_DISARM'
                    update(phase='Motor shutdown and settling', frames=frames)
                    start_t = latest['t']
                    stable_samples = []
                    unsafe_shutdown = False
                    for tail_step in range(math.ceil(2. / dt)):
                        if (output / 'STOP').exists():
                            cancelled = True
                            break
                        action = torch.zeros(1, 5, device=device)
                        obs_dict, _, _, _, info = env.step(action)
                        cumulative_delta_v += float(info['propulsive_delta_v_step'][0])
                        obs = obs_dict['policy']
                        servo = env._reset_manager.servo_state
                        capture(start_t + (tail_step + 1) * dt, action, (servo - last_target) / dt)
                        last_target = servo.clone()
                        attitude = float(tilt_angle(env._body_iface.get_root_quaternion_wxyz())[0])
                        unsafe_shutdown |= bool(env._unsafe_contact_step[0]) or attitude > .5
                        stable_samples.append(latest['contact_force_n'] >= 1.
                                              and math.sqrt(sum(v*v for v in latest['velocity'])) < .05
                                              and math.sqrt(sum(v*v for v in latest['gyro'])) < .15
                                              and attitude < .2 and latest['pad_distance'] <= .5)
                        env.render()
                    window = math.ceil(.5 / dt)
                    settled_after_shutdown = (not cancelled and not unsafe_shutdown
                                              and len(stable_samples) >= window
                                              and all(stable_samples[-window:]))
        landed = latest['contact'] == int(ContactState.LANDED)
        landing_event_success = bool(landing_frame and landing_frame['impact_speed'] <= .25
                                     and landing_frame['pad_distance'] <= .5)
        success = landing_event_success and latest['impact_speed'] <= .25 and latest['pad_distance'] <= .5 and settled_after_shutdown is True
        outcome = ('CANCELLED' if cancelled else 'POST_LANDING_FAILURE' if landed and not settled_after_shutdown
                   else 'LANDED' if landed else 'CRASHED' if bool(terminated[0]) else 'TIMEOUT')
        summary = dict(outcome=outcome, success=success, duration_s=latest['t'], frames=frames,
                       landing_duration_s=landing_frame['t'] if landing_frame else None,
                       landing_event_success=landing_event_success,
                       settled_after_shutdown=settled_after_shutdown,
                       flight_energy_wh=(landing_frame['battery']['energy_wh'] if landing_frame and landing_frame['battery'] else None),
                       propulsive_delta_v_m_s=cumulative_delta_v,
                       flight_rotation=env._rotation.record(),
                       impact_speed=latest['impact_speed'], pad_distance=latest['pad_distance'], battery=latest['battery'])
        atomic_json(output / 'summary.json', summary)
        update(state='cancelled' if cancelled else 'complete', phase=outcome, frames=frames,
               sim_time_s=latest['t'], summary=summary)
        return 0
    except Exception as exc:
        update(state='failed', phase='Simulation failed', error=str(exc))
        raise
    finally:
        watchdog.reset(30, label='Mission cleanup')
        if env is not None:
            env.close()
        if app is not None:
            close_simulation_app(app)
        watchdog.stop()


if __name__ == '__main__':
    force_process_exit(main())
