"""Train the goal-conditioned waypoint-flight PPO policy (feed-forward, direct actions).

Task: configs/tasks/waypoint_flight.yaml via tvc_env/envs/waypoint_flight.py.
Actions are the physical commands (4 vane angles + throttle); there is no
controller, mixer or guidance wrapper at training or inference time.

Example (8192 envs, ~16k env-steps/s on one RTX 5070):
    python apps/run_train_waypoints.py --num-envs 8192 --rollout-steps 32 --total-steps 400000000
Create a file named STOP in the run directory for a clean early stop.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

from runner_safety import WallClockWatchdog, force_process_exit


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env-config', default='configs/env/train_waypoint_flight.yaml')
    p.add_argument('--disturbance', default='configs/disturbances/nominal.yaml',
                   help='Disturbance YAML; sensor noise/COM/gusts apply to every stage. Steady wind '
                        'and battery SOC are already randomized per episode by the task.')
    p.add_argument('--num-envs', type=int, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--total-steps', type=int, default=400_000_000)
    p.add_argument('--rollout-steps', type=int, default=64)
    p.add_argument('--minibatches', type=int, default=8)
    p.add_argument('--update-epochs', type=int, default=4)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--critic-learning-rate', type=float, default=None)
    p.add_argument('--gamma', type=float, default=0.999)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--clip-coef', type=float, default=0.2)
    p.add_argument('--value-clip-range', type=float, default=None)
    p.add_argument('--ent-coef', type=float, default=0.002)
    p.add_argument('--vf-coef', type=float, default=0.5)
    p.add_argument('--max-grad-norm', type=float, default=0.5)
    p.add_argument('--target-kl', type=float, default=0.03)
    # Adam's bias-corrected first steps move every weight by the full LR. In run
    # 20260923_143257 that broke the KL guard after one minibatch for updates
    # 1-9, i.e. nine unfiltered sign steps on the actor. Ramp the actor LR.
    p.add_argument('--actor-lr-warmup-updates', type=int, default=10)
    # Adaptive KL learning rate (rsl_rl / Isaac Lab PPO schedule). Run
    # 20260923_201446 reached 60% stage-0 success at 42M transitions, then from
    # update 169 one fixed-LR Adam step already exceeded target_kl on every
    # update. The pre-minibatch guard cannot stop that first step, so each
    # update became one noisy 1/8-batch step; throttle random-walked 0.806 ->
    # 0.878 and success fell to 0 within ~15 updates. With latent sd ~0.09 a
    # 3e-4 step is too large; adapt the actor LR to the measured full-batch KL.
    p.add_argument('--desired-kl', type=float, default=0.01)
    p.add_argument('--actor-lr-min', type=float, default=1e-5)
    p.add_argument('--actor-lr-max', type=float, default=1e-3)
    p.add_argument('--value-norm-beta', type=float, default=0.01,
                   help='PopArt return-statistics EMA rate per update (critic trains on standardized returns).')
    # Initialization prior (CLAUDE.md rule 4). Throttle noise moves rotor
    # speed and so body yaw at 46.5 rad/s per unit throttle; a -2.0 latent
    # log-std gives a ~0.035 throttle sd, whose 60 s yaw peaks stay below
    # 170 deg/s (p99) without feedback, versus 96% of episodes past the
    # 360 deg/s spin limit at the old -1.0. Fin sigma keeps the old 0.75 deg.
    # Throttle is now hover duty + span * tanh(z) (waypoint_flight.policy_to_env_action),
    # so -2.0 gives a 0.25 * 0.135 = 0.034 duty sd, the same yaw-safe level.
    p.add_argument('--fin-log-std', type=float, default=-2.5)
    p.add_argument('--throttle-log-std', type=float, default=-2.0)
    p.add_argument('--throttle-span', type=float, default=0.25,
                   help='Throttle action = hover duty + span * tanh(z) (action contract).')
    # Exploration floors. Run 20260923_152010: throttle latent sd collapsed
    # 0.135 -> 0.055 and fin sd stayed at 0.05 while stage 1 had zero successes
    # for 160M transitions; translation was never sampled. The floors keep a
    # minimum action support; the means and the sd above the floor all train.
    p.add_argument('--fin-log-std-floor', type=float, default=-3.0)
    p.add_argument('--throttle-log-std-floor', type=float, default=-2.3)
    # A full-task evaluation flies one episode per env (up to 90 s simulated,
    # ~2700 policy steps); the serial PhysX chain makes that ~20 min wall time.
    # Curriculum advancement uses rollout outcomes and never waits for it.
    p.add_argument('--eval-interval', type=int, default=50_000_000)
    p.add_argument('--eval-max-seconds', type=float, default=None,
                   help='Diagnostic cap on evaluation episode length (smoke tests); unfinished = TIMEOUT.')
    p.add_argument('--save-interval', type=int, default=10_000_000)
    p.add_argument('--eval-action-mode', choices=['deterministic', 'stochastic', 'mean'], default='deterministic')
    p.add_argument('--output-dir', default='runs/waypoint_flight')
    p.add_argument('--resume', default=None, help='Resume a waypoint_flight checkpoint (optimizer, curriculum, steps).')
    p.add_argument('--allow-future-curriculum-change', action='store_true',
                   help='With --resume: accept revised curriculum stages after the checkpoint stage; '
                        'everything else, including the stages already trained on, must match.')
    p.add_argument('--headless', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--max-wall-time', type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    watchdog = WallClockWatchdog(args.max_wall_time or max(600.0, args.total_steps / 2000.0 + 900.0),
                                 label='Waypoint PPO training')
    watchdog.start()
    sim_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(sim_root))
    run_name = f"ppo_waypoint_flight_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = sim_root / args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'args.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    source_manifest = {str(path.relative_to(sim_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for directory in ('apps', 'tvc_env', 'configs')
                       for path in (sim_root / directory).rglob('*') if path.suffix in ('.py', '.yaml')}
    for relative in ('assets/usd/drone_v2_physics.usd', 'assets/metadata/edf_drone_v2.asset.yaml'):
        source_manifest[relative] = hashlib.sha256((sim_root / relative).read_bytes()).hexdigest()
    (output_dir / 'source_manifest.json').write_text(json.dumps(source_manifest, indent=2), encoding='utf-8')
    with zipfile.ZipFile(output_dir / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for relative in source_manifest:
            if not relative.startswith('assets/'):
                archive.write(sim_root / relative, relative.replace('\\', '/'))

    def append_jsonl(name, record):
        with (output_dir / name).open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(record, sort_keys=True) + '\n')

    app = env = None
    try:
        from isaac_launcher import launch_simulation_app
        app = launch_simulation_app(headless=args.headless)
    except ImportError:
        print('ERROR: Isaac Sim not available.', file=sys.stderr, flush=True)
        watchdog.stop()
        return 1

    try:
        import torch
        import torch.nn as nn
        from tvc_env.controllers.ppo_model import (ActorCritic, PopArtValueNormalizer, make_optimizer,
                                                   policy_kl, value_loss)
        from tvc_env.envs.base_env import BaseEnvConfig
        from tvc_env.envs.curriculum import StagedCurriculumTracker
        from tvc_env.envs.direct_rl_env import TVCDirectRLEnv
        from tvc_env.envs import waypoint_flight as wf
        from waypoint_eval import evaluate_policy

        torch.manual_seed(args.seed)
        overrides = {'env': {'num_envs': int(args.num_envs)}} if args.num_envs else None
        config = BaseEnvConfig(task_name='waypoint_flight', env_config_path=sim_root / args.env_config,
                               disturbance_config_path=sim_root / args.disturbance if args.disturbance else None,
                               overrides=overrides, sim_root=sim_root)
        config.validate_for_training()
        flight_cfg = config.config['task']['waypoint_flight']
        task_config = copy.deepcopy(config.config)
        final_task = wf.final_task_snapshot(config.config)
        (output_dir / 'task_config.json').write_text(json.dumps(task_config, indent=2), encoding='utf-8')
        for name in ('params/edf_90mm.yaml', 'params/servo_mg996r.yaml', 'vehicle/edf_drone_v2.yaml'):
            (output_dir / Path(name).name).write_bytes((sim_root / 'configs' / name).read_bytes())

        stages = flight_cfg['curriculum']['stages'] if flight_cfg['curriculum'].get('enabled') else [{}]
        tracker = StagedCurriculumTracker(stages=copy.deepcopy(stages),
                                          success_window_size=int(flight_cfg['curriculum'].get('success_window_size', 2048)))

        def install_stage():
            wf.apply_stage(config.config, final_task, stages[tracker.stage_index])

        install_stage()
        env = TVCDirectRLEnv(config)
        num_envs, device = config.num_envs, env.device
        obs_dim, act_dim = wf.OBS_DIM, 5
        rl_dt = config.physics_dt * config.decimation
        max_angle = float(env._servo_model.max_command_angle)

        hover_fraction = env.nominal_hover_throttle()
        # Budget the un-curricularized task: the live config holds a stage's
        # shorter episode length.
        budget = wf.reward_budget(task_config, hover_fraction)
        (output_dir / 'reward_budget.json').write_text(json.dumps(budget, indent=2), encoding='utf-8')
        print(f'Reward budget (rule 2): {json.dumps(budget)}', flush=True)

        # Throttle-head prior at level hover, corrected for loaded pack voltage
        # (duty = rotor fraction * reference / bus voltage). Not a controller:
        # every actor weight trains normally (CLAUDE.md rule 4).
        battery = env._battery_model
        bc = battery.config
        requested = torch.full_like(battery.soc, bc['shaft_power_at_max_w'] * hover_fraction ** 3
                                    / bc['motor_efficiency'] + bc['auxiliary_power_w'])
        voltage = battery.solve_load(requested)[0]
        hover_duty = min(0.98, hover_fraction * bc['reference_voltage_v'] / float(voltage.mean()))
        # Throttle latent 0 decodes to hover duty (CLAUDE.md rule 4 prior).
        model = ActorCritic(obs_dim, act_dim, throttle_bias=0.0).to(device)
        model.initialize_exploration(args.fin_log_std, args.throttle_log_std)
        log_std_floor = torch.tensor([args.fin_log_std_floor] * 4 + [args.throttle_log_std_floor], device=device)

        def to_env_action(raw):
            return wf.policy_to_env_action(raw, max_angle, hover_duty, args.throttle_span)
        optimizer = make_optimizer(model, args.learning_rate, args.critic_learning_rate)
        value_norm = PopArtValueNormalizer(model.critic[-1], beta=args.value_norm_beta)
        print(f'Hover rotor fraction {hover_fraction:.4f}, throttle prior duty {hover_duty:.4f}, '
              f'log_std {model.log_std.detach().cpu().tolist()}', flush=True)

        global_step = update = 0
        actor_lr = args.learning_rate
        best_eval = None
        if args.resume:
            path = Path(args.resume)
            path = path if path.is_absolute() else sim_root / path
            saved = torch.load(path, map_location=device, weights_only=False)
            if saved.get('observation_contract') != wf.OBSERVATION_CONTRACT:
                raise ValueError('Resume requires a waypoint_flight_v1 checkpoint')
            previous = copy.deepcopy(saved['task_config'])
            previous['env']['num_envs'] = task_config['env']['num_envs']
            if previous != task_config:
                index = int(saved['curriculum']['stage_index'])
                old, new = copy.deepcopy(previous), copy.deepcopy(task_config)
                old_stages = old['task']['waypoint_flight']['curriculum'].pop('stages')
                new_stages = new['task']['waypoint_flight']['curriculum'].pop('stages')
                if (not args.allow_future_curriculum_change or old != new
                        or old_stages[:index + 1] != new_stages[:index + 1]):
                    raise ValueError('Resolved task/plant config differs from the checkpoint; start a new run')
                print(f'[resume] curriculum stages after stage {index} revised: '
                      f'{[s.get("name") for s in old_stages[index + 1:]]} -> '
                      f'{[s.get("name") for s in new_stages[index + 1:]]}', flush=True)
            for key in ('gamma', 'gae_lambda', 'value_clip_range'):
                if saved['args'].get(key) != getattr(args, key):
                    raise ValueError(f'Resume changes the learning definition: {key}')
            model.load_state_dict(saved['model'])
            optimizer = make_optimizer(model, args.learning_rate, args.critic_learning_rate, saved['optimizer'])
            tracker.load_state_dict(saved['curriculum'], allow_future_changes=args.allow_future_curriculum_change)
            value_norm.load_state_dict(saved['value_normalizer'])
            global_step, update, best_eval = int(saved['step']), int(saved['update']), saved.get('best_eval')
            actor_lr = float(saved['actor_lr'])
            install_stage()
            print(f'Resumed step {global_step:,}, stage {tracker.stage_index}', flush=True)

        def checkpoint(path):
            payload = dict(format_version=3, task='waypoint_flight', observation_contract=wf.OBSERVATION_CONTRACT,
                           obs_dim=obs_dim, act_dim=act_dim,
                           action_contract=dict(fins='tanh(z) * max_command_angle rad, order +X,+Y,-X,-Y',
                                                throttle='clamp(throttle_center + throttle_span * tanh(z), 0, 1)',
                                                throttle_center=hover_duty, throttle_span=args.throttle_span),
                           model=model.state_dict(), optimizer=optimizer.state_dict(), args=vars(args),
                           step=global_step, update=update, task_config=task_config,
                           curriculum=tracker.state_dict(), value_normalizer=value_norm.state_dict(), actor_lr=actor_lr,
                           source_manifest=source_manifest,
                           reward_budget=budget, best_eval=best_eval,
                           physical_parameters={name: (output_dir / name).read_text(encoding='utf-8')
                                                for name in ('edf_90mm.yaml', 'servo_mg996r.yaml', 'edf_drone_v2.yaml')})
            temporary = path.with_suffix(path.suffix + '.tmp')
            torch.save(payload, temporary)
            temporary.replace(path)

        def run_eval(tag):
            nonlocal best_eval
            metrics = evaluate_policy(env, model, config, final_task, args.eval_action_mode, args.seed,
                                      max_seconds=args.eval_max_seconds, action_fn=to_env_action)
            record = dict(type=tag, global_step=global_step, update=update, stage_index=tracker.stage_index,
                          action_mode=args.eval_action_mode, **metrics)
            append_jsonl('eval_log.jsonl', record)
            (output_dir / 'eval_latest.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
            key = (metrics['success_fraction'], -metrics['failure_fraction'])
            if best_eval is None or key > (best_eval['success_fraction'], -best_eval['failure_fraction']):
                best_eval = dict(metrics, global_step=global_step)
                checkpoint(output_dir / 'ppo_best.pt')
            print(f"[eval] step={global_step:,} success={metrics['success_fraction']:.3f} "
                  f"failure={metrics['failure_fraction']:.3f} outcomes={metrics['outcomes']}", flush=True)
            install_stage()
            return env.reset(seed=args.seed + update)[0]['policy']

        obs = env.reset(seed=args.seed)[0]['policy']
        T = args.rollout_steps
        batch_size = T * num_envs
        minibatch_size = batch_size // args.minibatches
        obs_buf = torch.zeros(T, num_envs, obs_dim, device=device)
        latent_buf = torch.zeros(T, num_envs, act_dim, device=device)
        logprob_buf = torch.zeros(T, num_envs, device=device)
        reward_buf = torch.zeros(T, num_envs, device=device)
        done_buf = torch.zeros(T, num_envs, device=device)
        value_buf = torch.zeros(T, num_envs, device=device)
        success_buf = torch.zeros(T, num_envs, dtype=torch.bool, device=device)
        last_eval_bucket = global_step // args.eval_interval
        last_save_bucket = global_step // args.save_interval
        start_time, start_step = time.time(), global_step

        while global_step < args.total_steps:
            if (output_dir / 'STOP').exists():
                print('Graceful stop requested through STOP file.', flush=True)
                break
            update += 1
            optimizer.param_groups[0]['lr'] = actor_lr * min(
                1.0, update / max(args.actor_lr_warmup_updates, 1))
            outcome_counts = torch.zeros(len(wf.OUTCOMES), device=device)
            term_sums = torch.zeros(len(wf.REWARD_TERMS), device=device)
            ep_keys = ('energy_wh', 'time_s', 'route_m', 'captured', 'count')
            success_sums = torch.zeros(len(ep_keys), device=device)
            capture_fraction_sum = torch.zeros((), device=device)
            peak_yaw_sum = torch.zeros((), device=device)
            peak_yaw_max = torch.zeros((), device=device)
            throttle_sum = torch.zeros((), device=device)
            for t in range(T):
                global_step += num_envs
                obs_buf[t] = obs
                with torch.no_grad():
                    action_raw, logprob, _, value, latent = model.get_action_and_value(obs, return_latent=True)
                latent_buf[t], logprob_buf[t], value_buf[t] = latent, logprob, value_norm.denormalize(value)
                action = to_env_action(action_raw)
                throttle_sum += action[:, 4].sum()
                obs_dict, reward, terminated, truncated, info = env.step(action)
                obs = obs_dict['policy']
                done = terminated | truncated
                with torch.no_grad():
                    terminal_value = value_norm.denormalize(model(info['observation_pre_reset'])[1])
                reward_buf[t] = reward + args.gamma * terminal_value * truncated.float()
                done_buf[t] = done.float()
                flight = info['flight_pre_reset']
                success = done & (flight['outcome'] == wf.SUCCESS)
                success_buf[t] = success
                outcome_counts += (torch.nn.functional.one_hot(flight['outcome'], len(wf.OUTCOMES))
                                   * done[:, None]).sum(0)
                term_sums += torch.stack([info['reward_terms'][k].sum() for k in wf.REWARD_TERMS])
                success_sums += torch.stack([(flight[k].float() * success).sum() for k in ep_keys])
                capture_fraction_sum += ((flight['captured'] / flight['count'].clamp(min=1)) * done).sum()
                peak_yaw = info['rotation_pre_reset']['peak_rate_rad_s'][:, 2] * done
                peak_yaw_sum += peak_yaw.sum()
                peak_yaw_max = torch.maximum(peak_yaw_max, peak_yaw.max())
                env.render()

            # One host synchronisation per rollout.
            tracker.record_outcomes(batch_size, success_buf[done_buf.bool()].tolist())
            completed_stage = tracker.stage_index
            stage_success = tracker.success_fraction()
            advanced = tracker.should_advance()

            with torch.no_grad():
                next_value = value_norm.denormalize(model(obs)[1])
                advantages = torch.zeros_like(reward_buf)
                last = torch.zeros(num_envs, device=device)
                for t in reversed(range(T)):
                    nonterminal = 1.0 - done_buf[t]
                    following = next_value if t == T - 1 else value_buf[t + 1]
                    delta = reward_buf[t] + args.gamma * following * nonterminal - value_buf[t]
                    last = delta + args.gamma * args.gae_lambda * nonterminal * last
                    advantages[t] = last
                returns = advantages + value_buf
            b_obs = obs_buf.reshape(-1, obs_dim)
            b_latent = latent_buf.reshape(-1, act_dim)
            b_logprob = logprob_buf.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = value_buf.reshape(-1)
            b_adv = advantages.reshape(-1)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
            # Rescales the critic's last layer so predictions are preserved.
            value_norm.update(b_returns)
            n_returns, n_values = value_norm.normalize(b_returns), value_norm.normalize(b_values)
            with torch.no_grad():
                old_means = model.actor(b_obs)
                old_log_std = model.log_std.detach().clone()
            clipfracs, kls = [], []
            sums = dict(pg=0.0, v=0.0, ent=0.0)
            minibatches, kl_stop = 0, False
            for _ in range(args.update_epochs):
                order = torch.randperm(batch_size, device=device)
                for start in range(0, batch_size, minibatch_size):
                    mb = order[start:start + minibatch_size]
                    # Pre-update guard: stop before applying more gradient once
                    # the exact Gaussian KL from the rollout policy exceeds target.
                    with torch.no_grad():
                        current_kl = policy_kl(old_means[mb], old_log_std, model.actor(b_obs[mb]), model.log_std).mean()
                    if not torch.isfinite(current_kl):
                        raise FloatingPointError('Non-finite PPO policy KL')
                    if current_kl > args.target_kl:
                        kl_stop = True
                        break
                    _, newlogprob, entropy, newvalue = model.get_action_and_value(b_obs[mb], latent_action=b_latent[mb])
                    logratio = newlogprob - b_logprob[mb]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        kls.append(((ratio - 1) - logratio).mean())
                        clipfracs.append(((ratio - 1).abs() > args.clip_coef).float().mean())
                    pg_loss = torch.max(-b_adv[mb] * ratio,
                                        -b_adv[mb] * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()
                    v_loss = value_loss(newvalue, n_returns[mb], n_values[mb], args.value_clip_range)
                    loss = pg_loss - args.ent_coef * entropy.mean() + args.vf_coef * v_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(model.actor.parameters()) + [model.log_std], args.max_grad_norm)
                    nn.utils.clip_grad_norm_(model.critic.parameters(), args.max_grad_norm)
                    optimizer.step()
                    with torch.no_grad():
                        model.log_std.copy_(torch.maximum(model.log_std, log_std_floor))
                    sums['pg'] += float(pg_loss.detach())
                    sums['v'] += float(v_loss.detach())
                    sums['ent'] += float(entropy.mean().detach())
                    minibatches += 1
                if kl_stop:
                    break
            with torch.no_grad():
                sample = torch.randperm(batch_size, device=device)[:65536]
                full_kl = float(policy_kl(old_means[sample], old_log_std, model.actor(b_obs[sample]),
                                          model.log_std).mean())
            if full_kl > 2.0 * args.desired_kl:
                actor_lr = max(actor_lr / 1.5, args.actor_lr_min)
            elif full_kl < 0.5 * args.desired_kl:
                actor_lr = min(actor_lr * 1.5, args.actor_lr_max)

            finished = float(outcome_counts.sum())
            successes = float(outcome_counts[wf.SUCCESS])
            steps = T * num_envs
            elapsed = time.time() - start_time
            record = dict(
                type='train_update', update=update, global_step=global_step, wall_s=round(elapsed, 1),
                sps=round((global_step - start_step) / max(elapsed, 1e-6), 1),
                stage_index=completed_stage, stage_name=stages[completed_stage].get('name'),
                stage_success_fraction=round(stage_success, 4), stage_steps=tracker.steps_in_stage,
                pg_loss=sums['pg'] / max(minibatches, 1), v_loss=sums['v'] / max(minibatches, 1),
                entropy=sums['ent'] / max(minibatches, 1), minibatches=minibatches, kl_early_stop=kl_stop,
                approx_kl=float(torch.stack(kls).mean()) if kls else 0.0,
                clipfrac=float(torch.stack(clipfracs).mean()) if clipfracs else 0.0,
                explained_variance=float(1 - (b_returns - b_values).var() / b_returns.var().clamp(min=1e-8)),
                value_norm_mean=float(value_norm.mean), value_norm_std=float(value_norm.std),
                actor_lr=optimizer.param_groups[0]['lr'], policy_kl_full=full_kl,
                reward_mean=float(reward_buf.mean()), throttle_mean=float(throttle_sum) / steps,
                policy_latent_std=model.log_std.detach().exp().cpu().tolist(),
                reward_terms_per_step={k: float(v) / steps for k, v in zip(wf.REWARD_TERMS, term_sums.tolist())},
                episodes=int(finished),
                outcomes={name: int(c) for name, c in zip(wf.OUTCOMES, outcome_counts.tolist()) if name != 'RUNNING'},
                rollout_success_fraction=successes / max(finished, 1),
                capture_fraction=float(capture_fraction_sum) / max(finished, 1),
                mean_peak_yaw_deg_s=math.degrees(float(peak_yaw_sum) / max(finished, 1)),
                max_peak_yaw_deg_s=math.degrees(float(peak_yaw_max)),
                success_means={k: (float(v) / successes if successes else None)
                               for k, v in zip(ep_keys, success_sums.tolist())},
            )
            print(json.dumps(record, separators=(',', ':')), flush=True)
            append_jsonl('train_log.jsonl', record)

            if advanced:
                checkpoint(output_dir / f'ppo_stage_{completed_stage}_mastered.pt')
                tracker.advance()
                install_stage()
                obs = env.reset()[0]['policy']
                print(f'[curriculum] stage {completed_stage} mastered at {stage_success:.3f}; now stage '
                      f'{tracker.stage_index} ({stages[tracker.stage_index].get("name")})', flush=True)
            if global_step // args.save_interval > last_save_bucket:
                last_save_bucket = global_step // args.save_interval
                checkpoint(output_dir / f'ppo_step_{global_step}.pt')
            if global_step // args.eval_interval > last_eval_bucket:
                last_eval_bucket = global_step // args.eval_interval
                obs = run_eval('eval')

        run_eval('final_eval')
        checkpoint(output_dir / 'ppo_final.pt')
        print(f'Training complete: {global_step:,} steps in {time.time() - start_time:.0f}s. Run: {output_dir}', flush=True)
        return 0
    except Exception as exc:
        print(f'\nERROR: waypoint PPO training failed: {exc}', file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        return 2
    finally:
        watchdog.reset(30.0, label='Training cleanup')
        if env is not None:
            env.close()
        if app is not None:
            from isaac_launcher import close_simulation_app
            close_simulation_app(app)
        watchdog.stop()


if __name__ == '__main__':
    force_process_exit(main())
