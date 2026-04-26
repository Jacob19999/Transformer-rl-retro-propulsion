"""Train a feed-forward PPO hover policy for the TVC environment."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from runner_safety import WallClockWatchdog, force_process_exit


def parse_args():
    parser = argparse.ArgumentParser(description="PPO training for TVC environment")
    parser.add_argument("--task", default="hover", choices=["hover", "landing"])
    parser.add_argument("--env-config", default="configs/env/train_128.yaml")
    parser.add_argument("--disturbance", default="configs/disturbances/nominal.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=250_000)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--minibatches", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--bc-steps", type=int, default=2_000)
    parser.add_argument("--bc-batch-size", type=int, default=2048)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--eval-seconds", type=float, default=30.0)
    parser.add_argument("--save-interval", type=int, default=50_000)
    parser.add_argument("--output-dir", default="runs", help="Base output directory under simulation/isaac")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--fixed-hover-spawn", action="store_true")
    parser.add_argument("--residual-pid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument(
        "--max-wall-time",
        type=float,
        default=None,
        help="Maximum wall-clock seconds before forcing process exit.",
    )
    return parser.parse_args()


@dataclass
class EvalMetrics:
    mean_pos: float
    max_pos: float
    mean_tilt: float
    max_tilt: float
    mean_rate: float
    max_rate: float
    passed: bool


def main():
    args = parse_args()
    max_wall_time = args.max_wall_time
    if max_wall_time is None:
        max_wall_time = max(300.0, args.total_steps / 2500.0 + 240.0)
    watchdog = WallClockWatchdog(max_wall_time, label="PPO training")
    watchdog.start()

    sim_root = Path(__file__).parent.parent
    sys.path.insert(0, str(sim_root))

    run_name = f"ppo_{args.task}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = sim_root / args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    simulation_app = None
    env = None
    try:
        print(
            f"PPO training starting: task={args.task}, total_steps={args.total_steps}, "
            f"bc_steps={args.bc_steps}, headless={args.headless}",
            flush=True,
        )
        from isaacsim import SimulationApp

        simulation_app = SimulationApp({"headless": args.headless})
    except ImportError:
        print("ERROR: Isaac Sim not available.", file=sys.stderr, flush=True)
        watchdog.stop()
        return 1

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.distributions import Normal

        from tvc_env.common.quaternions import to_euler
        from tvc_env.controllers.pid_adapter import PIDController
        from tvc_env.envs.base_env import BaseEnvConfig
        from tvc_env.envs.direct_rl_env import TVCDirectRLEnv

        random.seed(args.seed)
        torch.manual_seed(args.seed)

        overrides = None
        if args.fixed_hover_spawn:
            overrides = {
                "task": {
                    "spawn": {
                        "position_range": [[0.0, 0.0, 5.0], [0.0, 0.0, 5.0]],
                        "velocity_range": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        "attitude_range": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    }
                }
            }

        config = BaseEnvConfig(
            task_name=args.task,
            env_config_path=sim_root / args.env_config,
            disturbance_config_path=sim_root / args.disturbance if args.disturbance else None,
            overrides=overrides,
            sim_root=sim_root,
        )
        env = TVCDirectRLEnv(config)
        num_envs = config.num_envs
        device = env.device
        obs_dim = 24
        act_dim = 5
        max_fin_angle = float(env._servo_model.max_command_angle)

        class ActorCritic(nn.Module):
            def __init__(self):
                super().__init__()
                actor_out = nn.Linear(256, act_dim)
                nn.init.zeros_(actor_out.weight)
                nn.init.zeros_(actor_out.bias)
                self.actor = nn.Sequential(
                    nn.Linear(obs_dim, 256),
                    nn.Tanh(),
                    nn.Linear(256, 256),
                    nn.Tanh(),
                    actor_out,
                )
                self.critic = nn.Sequential(
                    nn.Linear(obs_dim, 256),
                    nn.Tanh(),
                    nn.Linear(256, 256),
                    nn.Tanh(),
                    nn.Linear(256, 1),
                )
                self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))

            def forward(self, obs):
                mean = torch.tanh(self.actor(obs))
                value = self.critic(obs).squeeze(-1)
                return mean, value

            def get_action_and_value(self, obs, action_raw=None, deterministic=False):
                mean, value = self(obs)
                std = self.log_std.exp().expand_as(mean)
                dist = Normal(mean, std)
                if action_raw is None:
                    action_raw = mean if deterministic else dist.rsample()
                logprob = dist.log_prob(action_raw).sum(-1)
                entropy = dist.entropy().sum(-1)
                action_raw = action_raw.clamp(-1.0, 1.0)
                return action_raw, logprob, entropy, value

        def raw_to_env_action(action_raw, obs_for_pid=None, pid_for_residual=None):
            if args.residual_pid:
                if obs_for_pid is None or pid_for_residual is None:
                    raise ValueError("residual PID action conversion requires obs and PID controller")
                with torch.no_grad():
                    base_action = pid_for_residual.compute_action(obs_for_pid)
                residual_fins = action_raw[:, :4].clamp(-1.0, 1.0) * max_fin_angle
                residual_throttle = action_raw[:, 4:5].clamp(-1.0, 1.0) * 0.5
                residual = torch.cat([residual_fins, residual_throttle], dim=-1)
                action = base_action + args.residual_scale * residual
                action[:, :4] = action[:, :4].clamp(-max_fin_angle, max_fin_angle)
                action[:, 4] = action[:, 4].clamp(0.0, 1.0)
                return action
            fins = action_raw[:, :4].clamp(-1.0, 1.0) * max_fin_angle
            throttle = (action_raw[:, 4:5].clamp(-1.0, 1.0) + 1.0) * 0.5
            action = torch.cat([fins, throttle], dim=-1)
            return action

        def env_to_raw_action(action):
            fins = (action[:, :4] / max_fin_angle).clamp(-1.0, 1.0)
            throttle = (action[:, 4:5].clamp(0.0, 1.0) * 2.0 - 1.0).clamp(-1.0, 1.0)
            return torch.cat([fins, throttle], dim=-1)

        eval_pid = PIDController(num_envs=num_envs, device=device)

        def evaluate(policy, seconds: float) -> EvalMetrics:
            obs_dict, _ = env.reset(seed=args.seed)
            obs = obs_dict["policy"]
            eval_pid.reset()
            n_steps = int(seconds / (config.physics_dt * config.decimation))
            pos_errors: list[float] = []
            tilts: list[float] = []
            rates: list[float] = []
            with torch.no_grad():
                for _ in range(n_steps):
                    raw_action, _, _, _ = policy.get_action_and_value(obs, deterministic=True)
                    obs_dict, _, done, trunc, _ = env.step(raw_to_env_action(raw_action, obs, eval_pid))
                    reset_ids = (done | trunc).nonzero(as_tuple=False).squeeze(-1)
                    if len(reset_ids) > 0:
                        eval_pid.reset(reset_ids)
                    obs = obs_dict["policy"]
                    pos_errors.extend(obs[:, 0:3].norm(dim=-1).detach().cpu().tolist())
                    roll, pitch, _yaw = to_euler(obs[:, 3:7])
                    tilts.extend(torch.sqrt(roll * roll + pitch * pitch).detach().cpu().tolist())
                    rates.extend(obs[:, 10:13].norm(dim=-1).detach().cpu().tolist())
                    simulation_app.update()
            mean_pos = sum(pos_errors) / len(pos_errors)
            max_pos = max(pos_errors)
            mean_tilt = sum(tilts) / len(tilts)
            max_tilt = max(tilts)
            mean_rate = sum(rates) / len(rates)
            max_rate = max(rates)
            return EvalMetrics(
                mean_pos=mean_pos,
                max_pos=max_pos,
                mean_tilt=mean_tilt,
                max_tilt=max_tilt,
                mean_rate=mean_rate,
                max_rate=max_rate,
                passed=mean_pos < 0.5 and max_tilt < 0.262 and mean_rate < 1.0,
            )

        model = ActorCritic().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
        obs_dict, _ = env.reset(seed=args.seed)
        obs = obs_dict["policy"]

        if args.bc_steps > 0 and not args.residual_pid:
            print(f"PID behavior-cloning warm start: {args.bc_steps} gradient steps", flush=True)
            pid = PIDController(num_envs=num_envs, device=device)
            pid.reset()
            for step in range(args.bc_steps):
                with torch.no_grad():
                    teacher_action = pid.compute_action(obs)
                    target_raw = env_to_raw_action(teacher_action)
                pred_raw, _value = model(obs)
                loss = F.mse_loss(pred_raw, target_raw)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                obs_dict, _, done, trunc, _ = env.step(teacher_action)
                reset_ids = (done | trunc).nonzero(as_tuple=False).squeeze(-1)
                if len(reset_ids) > 0:
                    pid.reset(reset_ids)
                obs = obs_dict["policy"]
                simulation_app.update()

                if (step + 1) % max(1, args.bc_steps // 5) == 0:
                    print(f"  bc_step={step + 1}/{args.bc_steps} loss={loss.item():.5f}", flush=True)

            bc_metrics = evaluate(model, min(args.eval_seconds, 10.0))
            print(f"BC eval: {asdict(bc_metrics)}", flush=True)
            obs_dict, _ = env.reset(seed=args.seed)
            obs = obs_dict["policy"]
        elif args.residual_pid:
            print("Residual-PID PPO enabled: policy outputs bounded residuals around the stable PID baseline", flush=True)

        rollout_steps = args.rollout_steps
        batch_size = rollout_steps * num_envs
        minibatch_size = batch_size // args.minibatches
        if minibatch_size <= 0:
            raise ValueError("--minibatches is too large for rollout batch")

        obs_buf = torch.zeros((rollout_steps, num_envs, obs_dim), device=device)
        action_buf = torch.zeros((rollout_steps, num_envs, act_dim), device=device)
        logprob_buf = torch.zeros((rollout_steps, num_envs), device=device)
        reward_buf = torch.zeros((rollout_steps, num_envs), device=device)
        done_buf = torch.zeros((rollout_steps, num_envs), device=device)
        value_buf = torch.zeros((rollout_steps, num_envs), device=device)

        global_step = 0
        update = 0
        best_eval = None
        start_time = time.time()
        train_pid = PIDController(num_envs=num_envs, device=device)
        train_pid.reset()

        while global_step < args.total_steps:
            update += 1
            for t in range(rollout_steps):
                global_step += num_envs
                obs_buf[t] = obs
                with torch.no_grad():
                    action_raw, logprob, _entropy, value = model.get_action_and_value(obs)
                action_buf[t] = action_raw
                logprob_buf[t] = logprob
                value_buf[t] = value

                obs_dict, reward, terminated, truncated, _ = env.step(
                    raw_to_env_action(action_raw, obs, train_pid)
                )
                done = terminated | truncated
                reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
                if len(reset_ids) > 0:
                    train_pid.reset(reset_ids)
                reward_buf[t] = reward
                done_buf[t] = done.float()
                obs = obs_dict["policy"]
                simulation_app.update()

            with torch.no_grad():
                _next_mean, next_value = model(obs)
                advantages = torch.zeros_like(reward_buf)
                lastgaelam = torch.zeros(num_envs, device=device)
                for t in reversed(range(rollout_steps)):
                    if t == rollout_steps - 1:
                        next_nonterminal = 1.0 - done.float()
                        next_values = next_value
                    else:
                        next_nonterminal = 1.0 - done_buf[t + 1]
                        next_values = value_buf[t + 1]
                    delta = reward_buf[t] + args.gamma * next_values * next_nonterminal - value_buf[t]
                    lastgaelam = delta + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
                    advantages[t] = lastgaelam
                returns = advantages + value_buf

            b_obs = obs_buf.reshape((-1, obs_dim))
            b_actions = action_buf.reshape((-1, act_dim))
            b_logprobs = logprob_buf.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = value_buf.reshape(-1)
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            inds = torch.randperm(batch_size, device=device)
            clipfracs = []
            approx_kl = torch.tensor(0.0, device=device)
            for _epoch in range(args.update_epochs):
                for start in range(0, batch_size, minibatch_size):
                    mb_inds = inds[start : start + minibatch_size]
                    _a, newlogprob, entropy, newvalue = model.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - logratio).mean()
                        clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                    pg_loss1 = -b_advantages[mb_inds] * ratio
                    pg_loss2 = -b_advantages[mb_inds] * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + (newvalue - b_values[mb_inds]).clamp(
                        -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                if approx_kl > args.target_kl:
                    break

            if global_step % args.save_interval < num_envs or global_step >= args.total_steps:
                ckpt = output_dir / f"ppo_step_{global_step}.pt"
                torch.save({"model": model.state_dict(), "args": vars(args), "step": global_step}, ckpt)

            if global_step % args.eval_interval < num_envs or global_step >= args.total_steps:
                metrics = evaluate(model, args.eval_seconds)
                best_eval = metrics if best_eval is None or metrics.mean_pos < best_eval.mean_pos else best_eval
                (output_dir / "eval_latest.json").write_text(
                    json.dumps(asdict(metrics), indent=2), encoding="utf-8"
                )
                print(
                    f"step={global_step:,} reward_mean={reward_buf.mean().item():.3f} "
                    f"kl={approx_kl.item():.5f} clipfrac={sum(clipfracs)/max(len(clipfracs),1):.3f} "
                    f"eval={asdict(metrics)}",
                    flush=True,
                )
                obs_dict, _ = env.reset(seed=args.seed)
                obs = obs_dict["policy"]

        elapsed = time.time() - start_time
        final_metrics = evaluate(model, args.eval_seconds)
        torch.save({"model": model.state_dict(), "args": vars(args), "step": global_step}, output_dir / "ppo_final.pt")
        (output_dir / "eval_final.json").write_text(json.dumps(asdict(final_metrics), indent=2), encoding="utf-8")
        print(f"Training complete in {elapsed:.1f}s, steps={global_step:,}", flush=True)
        print(f"Final eval: {asdict(final_metrics)}", flush=True)
        print(f"Run directory: {output_dir}", flush=True)
        return 0 if final_metrics.passed else 1

    except Exception as exc:
        print(f"\nERROR: PPO training failed: {exc}", file=sys.stderr, flush=True)
        import traceback

        traceback.print_exc()
        return 2
    finally:
        watchdog.reset(30.0, label="PPO training cleanup")
        if env is not None:
            print("Closing TVC environment...", flush=True)
            env.close()
        if simulation_app is not None:
            print("Closing Isaac Sim...", flush=True)
            simulation_app.close()
            print("Isaac Sim closed.", flush=True)
        watchdog.stop()


if __name__ == "__main__":
    force_process_exit(main())
