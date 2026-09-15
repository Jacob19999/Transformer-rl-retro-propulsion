"""Independent, batched PPO landing evaluation with one outcome per episode."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

from runner_safety import WallClockWatchdog, force_process_exit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--env-config', default='configs/env/train_512.yaml')
    parser.add_argument('--disturbance', default='configs/disturbances/nominal.yaml')
    parser.add_argument('--episodes', type=int, default=512)
    parser.add_argument('--num-envs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--action-mode', choices=['deterministic', 'stochastic', 'mean'], default='deterministic',
                        help='Use tanh(latent mean), sample, or the true bounded-action expectation; recorded in results.')
    parser.add_argument('--curriculum-stage', type=int, default=None)
    parser.add_argument('--battery-soc', type=float, default=None)
    parser.add_argument('--residual-swirl', type=float, default=None,
                        help='Explicit new-model stator residual torque sensitivity in [0,1]')
    parser.add_argument('--physics-hz', type=float, default=None,
                        help='Explicit integer-multiple timestep refinement, retaining policy/contact periods')
    parser.add_argument('--trace-episodes', type=int, default=4)
    parser.add_argument('--allow-physics-transfer', action='store_true',
                        help='Explicitly evaluate a legacy checkpoint against changed asset physics.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-wall-time', type=float, default=900)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError('Episode count must be positive')
    if args.num_envs is not None and args.num_envs <= 0:
        raise ValueError('Environment count must be positive')
    if args.battery_soc is not None and not 0 <= args.battery_soc <= 1:
        raise ValueError('Battery SOC must be between zero and one')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    checkpoint = root / args.checkpoint
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'episodes.jsonl').exists():
        raise ValueError('Choose a new output directory; evaluation records already exist')
    watchdog = WallClockWatchdog(args.max_wall_time, label='PPO batch evaluation')
    watchdog.start()
    env = app = None
    try:
        from isaac_launcher import launch_simulation_app, close_simulation_app
        app = launch_simulation_app(headless=True)
        import torch
        from tvc_env.common.constants import ContactState
        from tvc_env.common.frames import isaac_position_to_frd
        from tvc_env.common.quaternions import inverse, normalize, rotate_vector, tilt_angle
        from tvc_env.controllers.ppo_model import ActorCritic
        from tvc_env.envs.base_env import BaseEnvConfig
        from tvc_env.envs.curriculum import apply_spawn_stage
        from tvc_env.envs.direct_rl_env import TVCDirectRLEnv

        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        train_args = saved.get('args', {})
        if train_args.get('task') != 'landing' or train_args.get('residual_pid') or train_args.get('landing_guidance'):
            raise ValueError('This evaluator requires a pure-PPO landing checkpoint')
        config = BaseEnvConfig(task_name='landing', env_config_path=root / args.env_config,
                               disturbance_config_path=root / args.disturbance, sim_root=root,
                               overrides={'env':{'num_envs':args.num_envs}} if args.num_envs else None)
        if args.battery_soc is not None:
            if not config.config.get('battery', {}).get('enabled'):
                raise ValueError('SOC evaluation requires an enabled battery model')
            config.config['battery']['initial_soc'] = args.battery_soc
        from tvc_env.envs.evaluation_contract import plant_mismatches
        mismatches = plant_mismatches(saved, config.config, root)
        if mismatches and not args.allow_physics_transfer:
            raise ValueError(f'Checkpoint plant mismatch: {mismatches}; use --allow-physics-transfer for a documented transfer diagnostic')
        if args.residual_swirl is not None:
            if not 0 <= args.residual_swirl <= 1 or not config.config.get('dynamics', {}).get('coupled_jet', {}).get('enabled'):
                raise ValueError('Residual swirl sensitivity requires a coupled jet and fraction in [0,1]')
            config.config['dynamics']['coupled_jet']['residual_swirl_fraction'] = args.residual_swirl
        asset_hashes = {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                        for path in ('assets/usd/drone_v2_physics.usd',
                                     'assets/metadata/edf_drone_v2.asset.yaml')}
        if not args.allow_physics_transfer:
            manifest = saved.get('source_manifest', {})
            for path, digest in asset_hashes.items():
                if manifest.get(path) != digest:
                    raise ValueError(f'Checkpoint asset mismatch: {path}; use --allow-physics-transfer for an explicit transfer diagnostic')
        if args.curriculum_stage is not None:
            spawn = deepcopy(config.config['task']['spawn'])
            stages = spawn.get('curriculum', {}).get('stages', [])
            if not 0 <= args.curriculum_stage < len(stages):
                raise ValueError('Requested curriculum stage is outside the task definition')
            apply_spawn_stage(config.config, spawn, stages[args.curriculum_stage])
        refinement = None
        if args.physics_hz is not None:
            from tvc_env.envs.evaluation_contract import refine_physics_clock
            refinement = refine_physics_clock(config, args.physics_hz)
        audit = dict(args=vars(args), trained_steps=saved['step'], task_config=deepcopy(config.config),
                     plant_mismatches=mismatches,
                     physics_refinement=refinement,
                     checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     asset_sha256=asset_hashes)
        (output / 'evaluation_config.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        env = TVCDirectRLEnv(config)
        n = config.num_envs
        device = env.device
        model = ActorCritic(env.observation_space.shape[0]).to(device)
        model.load_state_dict(saved['model'])
        model.eval()
        dt = config.physics_dt * config.decimation
        episode_steps = int(config.config['task']['episode_length_s'] / dt)
        gates = config.config['task']['success']
        max_pad = float(gates['max_pad_distance'])
        max_speed = float(gates['max_touchdown_speed'])
        max_angle = env._servo_model.max_command_angle
        records = []

        for batch in range((args.episodes + n - 1) // n):
            count = min(n, args.episodes - len(records))
            obs = env.reset(seed=args.seed + batch)[0]['policy']
            initial = obs.clone()
            initial_position = (env._body_iface.get_root_position()-env._env_origins).cpu().tolist()
            initial_quaternion = env._body_iface.get_root_quaternion_wxyz().cpu().tolist()
            initial_velocity = env._body_iface.get_linear_velocity_body_frd().cpu().tolist()
            initial_rates = env._body_iface.get_angular_velocity_body_frd().cpu().tolist()
            initial_waypoints = [env._navigation.record(i)['waypoints'] for i in range(count)] if env._navigation else [[] for _ in range(count)]
            finished = torch.arange(n, device=device) >= count
            max_down = torch.zeros(n, device=device)
            max_tilt = torch.zeros(n, device=device)
            throttle_sum = torch.zeros(n, device=device)
            energy_wh = torch.zeros(n, device=device)
            propulsive_delta_v = torch.zeros(n, device=device)
            low_hover_time = torch.zeros(n, device=device)
            max_body_rate = torch.zeros(n, device=device)
            with torch.no_grad():
                for step in range(episode_steps):
                    active = ~finished
                    policy_obs = obs.clone()
                    if train_args.get('body_frame_position_error', False):
                        policy_obs[:, :3] = isaac_position_to_frd(rotate_vector(
                            inverse(normalize(obs[:, 3:7])), obs[:, :3]))
                    raw = model.act(policy_obs, args.action_mode)
                    actions = torch.cat([raw[:, :4] * max_angle, (raw[:, 4:] + 1) * .5], dim=-1)
                    obs_dict, _, terminated, truncated, info = env.step(actions)
                    after = info['observation_pre_reset']
                    max_down = torch.maximum(max_down, -info['linear_vel_world_pre_reset'][:, 2] * active)
                    max_tilt = torch.maximum(max_tilt, tilt_angle(after[:, 3:7]) * active)
                    throttle_sum += actions[:, 4] * active
                    energy_wh += info['battery_energy_step_wh'] * active
                    propulsive_delta_v += info['propulsive_delta_v_step'] * active
                    clearance = info['position_pre_reset'][:, 2] - env._target_position[:, 2] - .3125
                    low_hover = active & (clearance > .03) & (clearance < 1.) & (info['linear_vel_world_pre_reset'][:, 2].abs() < .15)
                    low_hover_time += low_hover * dt
                    max_body_rate = torch.maximum(max_body_rate,
                        info['rotation_pre_reset']['peak_rate_norm_rad_s'][:, 0] * active)
                    if args.trace_episodes:
                        with (output/'traces.jsonl').open('a', encoding='utf-8') as trace_file:
                            for i in range(min(count, max(0,args.trace_episodes-batch*n))):
                                if bool(active[i]):
                                    trace_file.write(json.dumps(dict(episode=batch*n+i,time_s=(step+1)*dt,
                                        position=(info['position_pre_reset'][i]-env._target_position[i]).cpu().tolist(),
                                        velocity=info['linear_vel_world_pre_reset'][i].cpu().tolist(),
                                        actions=actions[i].cpu().tolist(),
                                        body_rate=info['angular_vel_frd_pre_reset'][i].cpu().tolist(),
                                        observed_body_rate=after[i,10:13].cpu().tolist(),
                                        rotor_fraction=float(info['motor_omega_pre_reset'][i]/env._omega_max),
                                        energy_wh=float(energy_wh[i]),delta_v_m_s=float(propulsive_delta_v[i])))+'\n')
                    complete = active & (terminated | truncated)
                    if step == episode_steps - 1:
                        complete |= active  # Explicit timeout for any unfinished trial.
                    ids = complete.nonzero(as_tuple=False).squeeze(-1).tolist()
                    for i in ids:
                        contact = int(info['contact_state_pre_reset'][i])
                        landed = contact == int(ContactState.LANDED)
                        crashed = bool(terminated[i]) and not landed
                        speed = float(info['touchdown_speed_pre_reset'][i])
                        pos = info['position_pre_reset'][i] - env._target_position[i]
                        distance = float(pos[:2].norm())
                        mission_complete = bool(info['mission_ready_to_land_pre_reset'][i])
                        success = landed and speed <= max_speed and distance <= max_pad and mission_complete
                        reason = ('success' if success else
                                  'premature_landing' if landed and not mission_complete else
                                  'hard_off_pad' if landed and speed > max_speed and distance > max_pad else
                                  'hard_landing' if landed and speed > max_speed else
                                  'off_pad' if landed else
                                  'contact_or_tilt_failure' if contact == int(ContactState.CRASHED) else
                                  'altitude_or_tilt_limit' if crashed else 'timeout')
                        rec = dict(episode=batch * n + i, reset_seed=args.seed + batch, batch_env=i,
                                   outcome='LANDED' if landed else 'CRASHED' if crashed else 'TIMEOUT',
                                   success=success, failure_reason=reason, duration_s=(step + 1) * dt,
                                   touchdown_speed=speed, pad_distance=distance,
                                   max_downward_speed=float(max_down[i]), max_tilt=float(max_tilt[i]),
                                   mean_throttle=float(throttle_sum[i]) / (step + 1),
                                   energy_wh=float(energy_wh[i]),
                                   propulsive_delta_v_m_s=float(propulsive_delta_v[i]),
                                   low_altitude_hover_time_s=float(low_hover_time[i]),
                                   max_body_rate_rad_s=float(max_body_rate[i]),
                                   waypoints_completed=int(info['waypoints_completed_pre_reset'][i]),
                                   mission_ready_to_land=mission_complete,
                                   spawn_position=initial_position[i],
                                   spawn_quaternion=initial_quaternion[i],
                                   spawn_body_velocity=initial_velocity[i],
                                   spawn_body_rates=initial_rates[i],
                                   waypoints=initial_waypoints[i],
                                   spawn_motor_fraction=float(initial[i, 22]),
                                   terminal_position=pos.cpu().tolist())
                        records.append(rec)
                        from tvc_env.envs.rotation_metrics import rotation_record
                        rec['rotation'] = rotation_record(info['rotation_pre_reset'], env._rotation.limits, i)
                        with (output / 'episodes.jsonl').open('a', encoding='utf-8') as fh:
                            fh.write(json.dumps(rec) + '\n')
                    finished |= complete
                    obs = obs_dict['policy']
                    env.render()
                    if bool(finished.all()):
                        break
            print(f'Evaluated {len(records)}/{args.episodes} independent episodes', flush=True)
        assert len(records) == args.episodes
        landed_records = [r for r in records if r['outcome'] == 'LANDED']
        successful_records = [r for r in records if r['success']]
        reasons = {reason: sum(r['failure_reason'] == reason for r in records)
                   for reason in sorted({r['failure_reason'] for r in records})}
        summary = dict(checkpoint=str(checkpoint), trained_steps=saved['step'], seed=args.seed,
                       action_mode=args.action_mode,
                       curriculum_stage=args.curriculum_stage, episodes=len(records),
                       success_count=sum(r['success'] for r in records),
                       success_fraction=sum(r['success'] for r in records) / len(records),
                       landed_fraction=len(landed_records) / len(records),
                       crashed_fraction=sum(r['outcome'] == 'CRASHED' for r in records) / len(records),
                       timeout_fraction=sum(r['outcome'] == 'TIMEOUT' for r in records) / len(records),
                       failure_counts=reasons, success_max_pad_distance=max_pad,
                       success_max_touchdown_speed=max_speed)
        for key in ('touchdown_speed', 'pad_distance'):
            values = [r[key] for r in landed_records]
            summary[key + '_landed'] = dict(mean=sum(values) / len(values), max=max(values)) if values else None
        # Compare efficiency only for successful trials: early crashes consume little energy.
        summary['successful_landing_efficiency'] = {}
        for key in ('energy_wh', 'propulsive_delta_v_m_s', 'duration_s', 'low_altitude_hover_time_s'):
            values = [r[key] for r in successful_records]
            summary['successful_landing_efficiency'][key] = (
                dict(mean=sum(values) / len(values), min=min(values), max=max(values)) if values else None)
        summary['passed'] = summary['success_fraction'] >= .8 and summary['crashed_fraction'] <= .05
        summary['rotation'] = {}
        for group, subset in (('all_episodes', records), ('successful_landings', successful_records)):
            summary['rotation'][group] = {
                key: (dict(mean=torch.tensor([r['rotation'][key] for r in subset]).mean(0).tolist(),
                           max=torch.tensor([r['rotation'][key] for r in subset]).max(0).values.tolist())
                      if subset else None)
                for key in ('peak_rate_deg_s','angular_travel_deg','excess_rotation_deg','time_above_limit_s','excess_cost_s')}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2), flush=True)
        return 0
    except Exception:
        # Kit's teardown can mask an unhandled Python exception's exit code.
        # Preserve an explicit failed-run result for the sequential suite.
        import traceback
        traceback.print_exc()
        return 2
    finally:
        watchdog.reset(30, label='PPO batch evaluation cleanup')
        if env is not None:
            env.close()
        if app is not None:
            close_simulation_app(app)
        watchdog.stop()


if __name__ == '__main__':
    force_process_exit(main())
