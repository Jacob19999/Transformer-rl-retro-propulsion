"""Evaluate a waypoint-flight PPO policy: full random task and fixed benchmarks.

Library use (trainer): ``evaluate_policy(env, model, config, final_task, ...)``.
CLI use: ``python apps/waypoint_eval.py --checkpoint runs/.../ppo_best.pt``.

Every environment flies exactly one episode, so quick failures are never
counted twice. Metrics are grouped by outcome, final-waypoint kind and, for
benchmarks, by mission.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from runner_safety import WallClockWatchdog, force_process_exit

ROOT = Path(__file__).resolve().parents[1]

# Fixed benchmark routes in env-local metres, all starting from BENCHMARK_SPAWN.
BENCHMARK_SPAWN = dict(position_range=[[-0.05, -0.05, 9.95], [0.05, 0.05, 10.05]],
                       velocity_range=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                       attitude_range=[[0.0, 0.0, -3.141592653589793], [0.0, 0.0, 3.141592653589793]],
                       angular_velocity_range=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                       initial_motor_omega_fraction='hover', initial_motor_omega_jitter=0.0,
                       initial_soc_range=[1.0, 1.0], wind_speed_range=[0.0, 0.0])


def _fly(x, y, z, radius=1.0):
    return dict(position=[x, y, z], type='flypass', radius_m=radius)


def _hover(x, y, z, hold=2.0, radius=0.5):
    return dict(position=[x, y, z], type='hover', radius_m=radius, hold_s=hold)


def _land(x, y, radius=0.5):
    return dict(position=[x, y, 0.0], type='land', radius_m=radius)


BENCHMARKS = {
    'station_keeping': [_hover(0, 0, 10, hold=5.0)],
    'straight_line': [_fly(15, 0, 10), _hover(30, 0, 10)],
    'square': [_fly(10, 0, 10), _fly(10, 10, 10), _fly(0, 10, 10), _hover(0, 0, 10)],
    'climb_descend': [_fly(5, 0, 20), _fly(10, 0, 5), _hover(15, 0, 10)],
    'reversal': [_fly(15, 0, 10), _hover(0, 0, 10)],
    'vertical_landing': [_land(0, 0)],
    'landing_approach': [_fly(8, 0, 6), _fly(12, 0, 3), _land(14, 0)],
    'zigzag_to_land': [_fly(8, 5, 10), _fly(16, -5, 10), _fly(24, 5, 12), _land(28, 0)],
}


def evaluate_policy(env, model, config, final_task, action_mode='deterministic', seed=0,
                    missions: dict | None = None, max_seconds: float | None = None, action_fn=None) -> dict:
    """One episode per env on the full task, or on named explicit missions.

    ``max_seconds`` is a diagnostic cap only (smoke tests); unfinished
    episodes then count as TIMEOUT. Restores the full task afterwards; the
    caller reinstalls its training stage. ``action_fn`` decodes the tanh
    actor output with the checkpoint's action contract.
    """
    import torch
    from tvc_env.envs import waypoint_flight as wf

    stage = {'spawn': BENCHMARK_SPAWN} if missions else None
    wf.apply_stage(config.config, final_task, stage)
    names = list(missions) if missions else None
    env._flight.set_explicit_missions([missions[name] for name in names] if missions else None)
    try:
        obs = env.reset(seed=seed)[0]['policy']
        n, device = env._config.num_envs, env.device
        rl_dt = config.physics_dt * config.decimation
        horizon = float(config.config['task']['episode_length_s'])
        max_steps = math.ceil(min(horizon, max_seconds or horizon) / rl_dt) + 2
        finished = torch.zeros(n, dtype=torch.bool, device=device)
        keys = ('outcome', 'energy_wh', 'time_s', 'captured', 'count', 'route_m', 'flown_m',
                'touchdown_speed', 'pad_distance', 'final_land')
        result = {key: torch.zeros(n, device=device) for key in keys}
        peak = torch.zeros(n, 3, device=device)
        with torch.no_grad():
            for _ in range(max_steps):
                raw = model.act(obs, action_mode)
                action = action_fn(raw)
                obs_dict, _, terminated, truncated, info = env.step(action)
                obs = obs_dict['policy']
                done = terminated | truncated
                newly = done & ~finished
                flight = info['flight_pre_reset']
                for key in keys:
                    result[key] = torch.where(newly, flight[key].float(), result[key])
                peak = torch.where(newly[:, None], info['rotation_pre_reset']['peak_rate_rad_s'], peak)
                finished |= done
                env.render()
                if bool(finished.all()):
                    break
        result['outcome'] = torch.where(finished, result['outcome'],
                                        torch.full_like(result['outcome'], wf.TIMEOUT))
    finally:
        env._flight.set_explicit_missions(None)
        wf.apply_stage(config.config, final_task, None)
    data = {key: value.cpu() for key, value in result.items()}
    data['peak_deg_s'] = peak.cpu() * (180 / math.pi)
    metrics = summarize(data, wf, config.config)
    if names:
        group = torch.arange(n) % len(names)
        metrics['missions'] = {name: summarize({k: v[group == i] for k, v in data.items()}, wf, config.config)
                               for i, name in enumerate(names)}
    return metrics


def summarize(data: dict, wf, config: dict) -> dict:
    import torch
    outcome = data['outcome'].long()
    episodes = int(outcome.numel())
    success = outcome == wf.SUCCESS
    failure = (outcome != wf.SUCCESS) & (outcome != wf.TIMEOUT) & (outcome != wf.RUNNING)
    land_final = data['final_land'] > 0.5
    yaw_soft = float(config['task']['waypoint_flight']['rate_soft_limits_deg_s'][2])

    def mean(values, mask):
        return float(values[mask].mean()) if bool(mask.any()) else float('nan')

    count = data['count'].clamp(min=1)
    metrics = dict(
        episodes=episodes,
        success_fraction=float(success.float().mean()),
        failure_fraction=float(failure.float().mean()),
        outcomes={name: float((outcome == code).float().mean()) for code, name in enumerate(wf.OUTCOMES)
                  if name != 'RUNNING'},
        capture_fraction=float((data['captured'] / count).mean()),
        land_mission_success_fraction=mean(success.float(), land_final),
        hover_mission_success_fraction=mean(success.float(), ~land_final),
        success_mean_time_s=mean(data['time_s'], success),
        success_mean_energy_wh=mean(data['energy_wh'], success),
        success_mean_energy_wh_per_m=mean(data['energy_wh'] / data['route_m'].clamp(min=1e-3), success),
        success_mean_route_speed_m_s=mean(data['route_m'] / data['time_s'].clamp(min=1e-3), success),
        success_mean_path_efficiency=mean(data['route_m'] / data['flown_m'].clamp(min=1e-3), success),
        landing_success_mean_touchdown_speed=mean(data['touchdown_speed'], success & land_final),
        landing_success_mean_pad_distance=mean(data['pad_distance'], success & land_final),
        mean_peak_rate_deg_s=data['peak_deg_s'].mean(0).tolist() if episodes else [],
        p95_peak_yaw_deg_s=float(torch.quantile(data['peak_deg_s'][:, 2], 0.95)) if episodes else float('nan'),
        fraction_peak_yaw_over_soft_limit=float((data['peak_deg_s'][:, 2] > yaw_soft).float().mean()),
    )
    metrics['passed'] = metrics['success_fraction'] >= 0.8 and metrics['failure_fraction'] <= 0.1
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=512)
    parser.add_argument('--action-mode', choices=['deterministic', 'stochastic', 'mean'], default='deterministic')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--suite', choices=['random', 'benchmarks', 'both'], default='both')
    parser.add_argument('--disturbance', default='configs/disturbances/nominal.yaml')
    parser.add_argument('--max-seconds', type=float, default=None,
                        help='Diagnostic cap on episode length; unfinished episodes count as TIMEOUT.')
    parser.add_argument('--output', type=Path, default=None, help='JSON output path (default next to checkpoint)')
    parser.add_argument('--headless', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    watchdog = WallClockWatchdog(3600, label='Waypoint evaluation')
    watchdog.start()
    sys.path.insert(0, str(ROOT))
    app = env = None
    try:
        from isaac_launcher import launch_simulation_app, close_simulation_app
        app = launch_simulation_app(headless=args.headless)
        import torch
        import yaml
        from tvc_env.controllers.ppo_model import ActorCritic
        from tvc_env.envs.base_env import BaseEnvConfig
        from tvc_env.envs.direct_rl_env import TVCDirectRLEnv
        from tvc_env.envs import waypoint_flight as wf
        from tvc_env.envs.task_registry import deep_merge
        checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved.get('observation_contract') != wf.OBSERVATION_CONTRACT:
            raise ValueError('Checkpoint is not a waypoint_flight_v1 policy')
        trained = saved['task_config']
        overrides = {key: trained[key] for key in ('env', 'edf', 'servo', 'battery', 'physics', 'dynamics', 'task')}
        overrides['env'] = dict(overrides['env'], num_envs=args.num_envs)
        disturbance = yaml.safe_load((ROOT / args.disturbance).read_text()) if args.disturbance else {}
        overrides = deep_merge(overrides, disturbance)
        config = BaseEnvConfig(task_name='waypoint_flight', sim_root=ROOT, overrides=overrides)
        final_task = wf.final_task_snapshot(config.config)
        env = TVCDirectRLEnv(config)
        model = ActorCritic(saved['obs_dim'], saved['act_dim']).to(env.device)
        model.load_state_dict(saved['model'])
        model.eval()
        contract = saved['action_contract']
        max_angle = float(env._servo_model.max_command_angle)

        def action_fn(raw):
            return wf.policy_to_env_action(raw, max_angle, contract['throttle_center'], contract['throttle_span'])
        report = dict(checkpoint=str(checkpoint), step=saved['step'], action_mode=args.action_mode,
                      disturbance=args.disturbance, num_envs=args.num_envs)
        report['max_seconds'] = args.max_seconds
        if args.suite in ('random', 'both'):
            report['random_task'] = evaluate_policy(env, model, config, final_task, args.action_mode, args.seed,
                                                    max_seconds=args.max_seconds, action_fn=action_fn)
        if args.suite in ('benchmarks', 'both'):
            report['benchmarks'] = evaluate_policy(env, model, config, final_task, args.action_mode, args.seed,
                                                   missions=BENCHMARKS, max_seconds=args.max_seconds,
                                                   action_fn=action_fn)
        output = args.output or checkpoint.with_name(checkpoint.stem + f'_eval_{args.action_mode}.json')
        output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
        print(f'Wrote {output}', flush=True)
        return 0
    finally:
        watchdog.reset(30, label='Evaluation cleanup')
        if env is not None:
            env.close()
        if app is not None:
            close_simulation_app(app)
        watchdog.stop()


if __name__ == '__main__':
    force_process_exit(main())
