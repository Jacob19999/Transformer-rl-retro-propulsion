"""Repeatable PID gain sweep against the unchanged hover/landing task physics."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

from runner_safety import force_process_exit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['hover', 'landing'], default='hover')
    parser.add_argument('--seeds', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--kp-att', type=float, nargs='+', default=[.10, .20, .30])
    parser.add_argument('--kd-att', type=float, nargs='+', default=[.02, .05, .10])
    parser.add_argument('--gyro-comp-rp', type=float, nargs='+', default=[0, .03, .06])
    parser.add_argument('--min-fin-cmd-xy', type=float, default=.018)
    parser.add_argument('--kp-alt', type=float, default=.22)
    parser.add_argument('--ki-alt', type=float, default=.01)
    parser.add_argument('--kd-alt', type=float, default=.10)
    parser.add_argument('--k-pos-xy', type=float, default=.125)
    parser.add_argument('--k-vel-xy', type=float, default=.21)
    parser.add_argument('--descent-rate', type=float, default=1.0)
    parser.add_argument('--flare-alt', type=float, default=.5)
    parser.add_argument('--flare-descent-rate', type=float, default=.25)
    parser.add_argument('--output', default='logs/physics_review_pid_sweep.json')
    args = parser.parse_args()
    from isaac_launcher import launch_simulation_app
    app = launch_simulation_app(headless=True)
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tvc_env.envs.base_env import BaseEnvConfig
    from tvc_env.envs.direct_rl_env import TVCDirectRLEnv
    from tvc_env.controllers.pid_adapter import PIDController
    from tvc_env.controllers.landing_guidance import LandingGuidance
    from tvc_env.common.constants import ContactState
    from tvc_env.common.quaternions import tilt_angle

    if args.seeds < 1:
        raise ValueError('--seeds must be positive')
    gains = list(itertools.product(args.kp_att, args.kd_att, args.gyro_comp_rp))
    n = len(gains) * args.seeds
    config = BaseEnvConfig(args.task, overrides={'env': {'num_envs': n, 'reset_on_crash': False},
                                              'task': {'spawn': {'curriculum': {'enabled': False}}}})
    env = TVCDirectRLEnv(config)
    dt = config.physics_dt * config.decimation
    obs, _ = env.reset(seed=args.seed)
    state = env._build_vehicle_state()
    # Give each candidate exactly the same sampled initial conditions.
    local_pos = state.position[:args.seeds] - env._env_origins[:args.seeds]
    initial_cases = dict(position=local_pos.cpu().tolist(),
                         velocity=state.linear_vel_world[:args.seeds].cpu().tolist(),
                         quaternion=state.quaternion_wxyz[:args.seeds].cpu().tolist())
    env._body_iface.set_root_state(
        local_pos.repeat(len(gains), 1) + env._env_origins,
        state.quaternion_wxyz[:args.seeds].repeat(len(gains), 1),
        state.linear_vel_world[:args.seeds].repeat(len(gains), 1),
        state.angular_vel_world[:args.seeds].repeat(len(gains), 1),
    )
    obs = env._get_observations()['policy']
    hover = env.nominal_hover_throttle()
    controllers = [PIDController(num_envs=args.seeds, device=env.device, kp_att=kp, kd_att=kd,
                                 gyro_comp_rp=gc, throttle_hover=hover, dt=dt,
                                 kp_alt=args.kp_alt, ki_alt=args.ki_alt, kd_alt=args.kd_alt,
                                 k_pos_xy=args.k_pos_xy, k_vel_xy=args.k_vel_xy,
                                 min_fin_cmd_xy=args.min_fin_cmd_xy) for kp, kd, gc in gains]
    guidance = LandingGuidance(n, env.device, env._target_position, throttle_hover=hover, dt=dt,
                               descent_rate=args.descent_rate, flare_alt=args.flare_alt,
                               flare_descent_rate=args.flare_descent_rate) if args.task == 'landing' else None
    if guidance:
        guidance.reset(obs=obs)
    finished = torch.zeros(n, dtype=torch.bool, device=env.device)
    landed = finished.clone()
    crashed = finished.clone()
    success = finished.clone()
    max_tilt = torch.zeros(n, device=env.device)
    final_error = torch.zeros(n, device=env.device)
    impact = torch.full((n,), float('nan'), device=env.device)
    finish_time = torch.full((n,), float(config.config['task']['episode_length_s']), device=env.device)
    final_height = torch.zeros(n, device=env.device)
    sq_error = torch.zeros(n, device=env.device)
    sample_count = torch.zeros(n, device=env.device)
    seconds = float(config.config['task']['episode_length_s'])
    for step in range(int(seconds / dt)):
        pid_obs = guidance.modify_obs(obs) if guidance else obs
        actions = torch.cat([pid.compute_action(pid_obs[i*args.seeds:(i+1)*args.seeds]) for i, pid in enumerate(controllers)])
        if guidance:
            actions = guidance.post_action(actions)
        obs_dict, _, term, trunc, info = env.step(actions)
        obs = obs_dict['policy']
        active = ~finished
        max_tilt = torch.maximum(max_tilt, tilt_angle(obs[:, 3:7]) * active)
        error = obs[:, :3].norm(dim=-1) if not guidance else obs[:, :2].norm(dim=-1)
        final_error[active] = error[active]
        final_height[active] = obs[active, 13]
        if step * dt > seconds - 5:
            sq_error += error.square() * active
            sample_count += active
        new_done = (term | trunc) & active
        finish_time[new_done] = (step + 1) * dt
        landed_now = info['contact_state_pre_reset'] == int(ContactState.LANDED)
        crashed |= new_done & (info['contact_state_pre_reset'] == int(ContactState.CRASHED))
        landed |= new_done & landed_now
        impact[new_done & landed_now] = info['touchdown_speed_pre_reset'][new_done & landed_now]
        success |= new_done & landed_now & (error <= .5) & (info['touchdown_speed_pre_reset'] <= .25)
        finished |= new_done
        if finished.all():
            break
    rows = []
    for i, (kp, kd, gc) in enumerate(gains):
        sl = slice(i*args.seeds, (i+1)*args.seeds)
        rows.append(dict(kp_att=kp, kd_att=kd, gyro_comp_rp=gc, throttle_hover=hover,
                         landed=int(landed[sl].sum()), crashed=int(crashed[sl].sum()), success=int(success[sl].sum()),
                         final_error_mean=float(final_error[sl].mean()),
                         rms_error_last_5s=float((sq_error[sl].sum()/sample_count[sl].sum().clamp(min=1)).sqrt()),
                         max_tilt_rad=float(max_tilt[sl].max()), touchdown_speed=impact[sl].tolist(),
                         cases=[dict(success=bool(success[j]), landed=bool(landed[j]), crashed=bool(crashed[j]),
                                     duration_s=float(finish_time[j]), pad_error_m=float(final_error[j]),
                                     height_m=float(final_height[j]), touchdown_speed=float(impact[j]))
                                for j in range(i*args.seeds, (i+1)*args.seeds)]))
    rows.sort(key=lambda r: (-r['success'], r['crashed'], r['rms_error_last_5s'] if args.task == 'hover' else r['final_error_mean']))
    payload = dict(task=args.task, seed=args.seed, cases_per_candidate=args.seeds, duration_s=seconds,
                   args=vars(args), initial_cases=initial_cases, candidates=rows)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2), flush=True)
    env.close()
    return 0


if __name__ == '__main__':
    force_process_exit(main())
