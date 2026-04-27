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
    # Learning rate: 3e-4 is the standard PPO LR for normalized-advantage
    # control tasks (CleanRL, SB3). The previous default of 1e-5 produced
    # per-update KL ≈ 3e-5 — two orders of magnitude below target_kl=0.03 —
    # so the policy effectively did not move across millions of env steps.
    # target_kl already serves as the safety governor against oversized updates.
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    # Small entropy bonus keeps the throttle/fin distributions from collapsing
    # around the init bias before the policy has explored the descent basin.
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,
        help=(
            "Scalar multiplier on the env reward before PPO. Default 1.0 lets the "
            "YAML reward weights speak directly; setting it << 1 collapses the "
            "relative magnitude of one-time terminal rewards versus integrated "
            "per-step costs and is the usual cause of 'agent learns to minimize "
            "throttle and ride out the clock'."
        ),
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--bc-steps", type=int, default=0)
    parser.add_argument("--bc-batch-size", type=int, default=2048)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--eval-seconds", type=float, default=30.0)
    parser.add_argument("--save-interval", type=int, default=50_000)
    parser.add_argument("--output-dir", default="runs", help="Base output directory under simulation/isaac")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--fixed-hover-spawn", action="store_true")
    parser.add_argument(
        "--body-frame-position-error",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Feed the policy target-current position error in body-FRD coordinates. "
            "Defaults on for landing so lateral guidance is expressed in the same "
            "control frame as body velocity and fin authority."
        ),
    )
    parser.add_argument("--residual-pid", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument(
        "--landing-guidance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Explicitly wrap landing training/eval with the scripted moving-altitude "
            "guidance profile. Default is pure PPO against the task reward."
        ),
    )
    parser.add_argument("--landing-descent-rate", type=float, default=1.0)
    parser.add_argument("--landing-flare-alt", type=float, default=0.5)
    parser.add_argument("--landing-flare-descent-rate", type=float, default=0.25)
    parser.add_argument("--landing-xy-gate-radius", type=float, default=0.75)
    parser.add_argument("--landing-far-descent-rate", type=float, default=0.15)
    parser.add_argument("--landing-descent-brake-gain", type=float, default=0.35)
    parser.add_argument("--landing-min-descent-throttle", type=float, default=0.30)
    parser.add_argument("--landing-touchdown-speed-limit", type=float, default=1.5)
    parser.add_argument("--landing-pad-distance-limit", type=float, default=0.5)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-3)
    parser.add_argument("--early-stop-warmup-updates", type=int, default=8)
    parser.add_argument("--early-stop-ema-alpha", type=float, default=0.2)
    parser.add_argument("--no-early-stop", action="store_true")
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


@dataclass
class LandingEvalMetrics:
    landed_fraction: float
    crashed_fraction: float
    mean_touchdown_speed: float
    max_touchdown_speed: float
    mean_pad_distance: float
    max_pad_distance: float
    mean_delta_v_proxy: float
    mean_eval_throttle: float
    max_eval_throttle: float
    max_upward_speed: float
    max_downward_speed: float
    passed: bool


def main():
    args = parse_args()
    if args.body_frame_position_error is None:
        args.body_frame_position_error = args.task == "landing"
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

    def append_jsonl(name: str, record: dict) -> None:
        with (output_dir / name).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

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

        from tvc_env.common.constants import ContactState
        from tvc_env.common.frames import isaac_position_to_frd
        from tvc_env.common.quaternions import inverse as quat_inv, normalize, rotate_vector
        from tvc_env.common.quaternions import to_euler
        from tvc_env.controllers.landing_guidance import LandingGuidance
        from tvc_env.controllers.pid_adapter import PIDController
        from tvc_env.envs.base_env import BaseEnvConfig
        from tvc_env.envs.curriculum import apply_spawn_position_range, resolve_spawn_curriculum
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
                        "curriculum": {"enabled": False},
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

        def set_training_spawn(global_step_value: int):
            curriculum_state = resolve_spawn_curriculum(config.config, global_step_value)
            if curriculum_state.enabled:
                apply_spawn_position_range(config.config, curriculum_state.position_range)
            return curriculum_state

        def set_eval_spawn():
            curriculum_state = resolve_spawn_curriculum(config.config, args.total_steps)
            if curriculum_state.enabled:
                apply_spawn_position_range(config.config, curriculum_state.final_position_range)
            return curriculum_state

        def spawn_xy_half_width(position_range: list[list[float]]) -> float:
            return max(abs(float(position_range[row][col])) for row in range(2) for col in range(2))

        initial_spawn_curriculum = set_training_spawn(0)
        if initial_spawn_curriculum.enabled:
            print(
                "Spawn curriculum enabled: "
                f"xy_half_width starts at {spawn_xy_half_width(initial_spawn_curriculum.position_range):.2f} m "
                f"and anneals to {spawn_xy_half_width(initial_spawn_curriculum.final_position_range):.2f} m "
                f"by step {int(config.config['task']['spawn']['curriculum'].get('end_step', 0)):,}. "
                "Eval uses the final full spawn range.",
                flush=True,
            )
        if args.body_frame_position_error:
            print(
                "PPO policy observation uses body-FRD position error for obs[0:3]; "
                "environment observations and PID/guidance adapters remain unchanged.",
                flush=True,
            )

        def policy_observation(obs_raw: torch.Tensor) -> torch.Tensor:
            if not args.body_frame_position_error:
                return obs_raw
            obs_policy = obs_raw.clone()
            pos_error_world = obs_policy[:, 0:3]
            q_inv = quat_inv(normalize(obs_policy[:, 3:7]))
            pos_error_body_isaac = rotate_vector(q_inv, pos_error_world)
            obs_policy[:, 0:3] = isaac_position_to_frd(pos_error_body_isaac)
            return obs_policy

        def atanh_clamped(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            x = x.clamp(-1.0 + eps, 1.0 - eps)
            return 0.5 * (torch.log1p(x) - torch.log1p(-x))

        # Initial bias for the throttle channel of the actor's output.
        # Action layout: [0:4]=fins (mean=0 ⇒ centered), [4]=throttle.
        # Throttle mapping is throttle = (tanh(z) + 1) / 2, and the EDF model is
        # T = max_thrust * throttle² (T = k_T * (throttle * ω_max)²). For the
        # 3.1 kg / 39.2 N drone, hover throttle = sqrt(m·g / T_max) ≈ 0.88.
        # We bias slightly below hover (~0.78 mean) so initial exploration
        # straddles hover instead of free-falling from throttle=0.5, which is
        # the prior cause of the policy never finding the descent basin.
        hover_throttle_init = 0.78
        throttle_bias_init = float(
            torch.atanh(torch.tensor(2.0 * hover_throttle_init - 1.0)).item()
        )

        class ActorCritic(nn.Module):
            def __init__(self):
                super().__init__()
                actor_out = nn.Linear(256, act_dim)
                nn.init.zeros_(actor_out.weight)
                nn.init.zeros_(actor_out.bias)
                with torch.no_grad():
                    actor_out.bias[4] = throttle_bias_init
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
                # Per-dim exploration std: fins start quieter than throttle so
                # the policy isn't drowning in random ±7.5° lateral noise (the
                # symptom of a shared log_std=-1.0 init was a "land softly
                # anywhere" plateau at ~2 m pad distance — see prior run
                # ppo_landing_seed0_20260426_182954). Fins log_std=-2 ⇒ std≈0.14
                # (tanh-bounded ±0.05·max_fin_angle range per step), throttle
                # log_std=-1 ⇒ std≈0.37 keeps thrust exploration alive.
                init_log_std = torch.full((act_dim,), -1.0)
                init_log_std[:4] = -2.0  # fin channels quieter
                self.log_std = nn.Parameter(init_log_std)

            def forward(self, obs):
                mean = self.actor(obs)
                value = self.critic(obs).squeeze(-1)
                return mean, value

            def get_action_and_value(self, obs, action_raw=None, deterministic=False):
                mean, value = self(obs)
                std = self.log_std.exp().expand_as(mean)
                dist = Normal(mean, std)
                if action_raw is None:
                    pre_tanh = mean if deterministic else dist.rsample()
                    action_raw = torch.tanh(pre_tanh)
                else:
                    action_raw = action_raw.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
                    pre_tanh = atanh_clamped(action_raw)
                log_det_jacobian = torch.log(1.0 - action_raw.pow(2) + 1e-6).sum(-1)
                logprob = dist.log_prob(pre_tanh).sum(-1) - log_det_jacobian
                entropy = dist.entropy().sum(-1)
                return action_raw, logprob, entropy, value

        def raw_to_env_action(action_raw, obs_for_pid=None, pid_for_residual=None, guidance=None):
            if args.residual_pid:
                if obs_for_pid is None or pid_for_residual is None:
                    raise ValueError("residual PID action conversion requires obs and PID controller")
                pid_obs = guidance.modify_obs(obs_for_pid) if guidance is not None else obs_for_pid
                with torch.no_grad():
                    base_action = pid_for_residual.compute_action(pid_obs)
                residual_fins = action_raw[:, :4].clamp(-1.0, 1.0) * max_fin_angle
                residual_throttle = action_raw[:, 4:5].clamp(-1.0, 1.0) * 0.5
                residual = torch.cat([residual_fins, residual_throttle], dim=-1)
                action = base_action + args.residual_scale * residual
                action[:, :4] = action[:, :4].clamp(-max_fin_angle, max_fin_angle)
                action[:, 4] = action[:, 4].clamp(0.0, 1.0)
                if guidance is not None:
                    action = guidance.post_action(action, obs_for_pid)
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
        is_landing = args.task == "landing"
        rl_dt = config.physics_dt * config.decimation

        def make_guidance() -> LandingGuidance:
            return LandingGuidance(
                num_envs=num_envs,
                device=device,
                target_position=env._target_position,
                descent_rate=args.landing_descent_rate,
                flare_alt=args.landing_flare_alt,
                flare_descent_rate=args.landing_flare_descent_rate,
                xy_gate_radius=args.landing_xy_gate_radius,
                far_descent_rate=args.landing_far_descent_rate,
                descent_brake_gain=args.landing_descent_brake_gain,
                min_descent_throttle=args.landing_min_descent_throttle,
                dt=rl_dt,
            )

        guidance_enabled = bool(is_landing and args.landing_guidance)
        if args.residual_pid:
            print(
                "Residual-PID PPO enabled explicitly: policy learns bounded residuals around the PID baseline.",
                flush=True,
            )
        if guidance_enabled:
            print(
                "Landing guidance enabled explicitly: moving-altitude observations and touchdown disarm are part of this task definition.",
                flush=True,
            )
        eval_guidance = make_guidance() if guidance_enabled else None
        train_guidance = make_guidance() if guidance_enabled else None

        def evaluate_landing(policy, seconds: float) -> LandingEvalMetrics:
            set_eval_spawn()
            obs_dict, _ = env.reset(seed=args.seed)
            obs = obs_dict["policy"]
            eval_pid.reset()
            if eval_guidance is not None:
                eval_guidance.reset(obs=obs)
            n_steps = int(seconds / rl_dt)
            landed_step = torch.full((num_envs,), -1, dtype=torch.int64, device=device)
            touchdown_speed = torch.zeros(num_envs, device=device)
            touchdown_pad_dist = torch.zeros(num_envs, device=device)
            crashed = torch.zeros(num_envs, dtype=torch.bool, device=device)
            delta_v_proxy = torch.zeros(num_envs, device=device)
            throttle_sum = 0.0
            throttle_count = 0
            max_eval_throttle = 0.0
            max_upward_speed = 0.0
            max_downward_speed = 0.0
            prev_contact = torch.full((num_envs,), int(ContactState.AIRBORNE), dtype=torch.int64, device=device)
            omega_max = max(float(env._omega_max), 1.0)
            with torch.no_grad():
                for step in range(n_steps):
                    raw_action, _, _, _ = policy.get_action_and_value(
                        policy_observation(obs), deterministic=True
                    )
                    env_action = raw_to_env_action(raw_action, obs, eval_pid, eval_guidance)
                    throttle = env_action[:, 4]
                    throttle_sum += float(throttle.sum().item())
                    throttle_count += int(throttle.numel())
                    max_eval_throttle = max(max_eval_throttle, float(throttle.max().item()))
                    obs_dict, _, done, trunc, info = env.step(env_action)
                    next_obs = obs_dict["policy"]
                    # Use the pre-reset contact state (env auto-resets terminated
                    # envs to AIRBORNE before observations are computed, so
                    # reading next_obs[:, 23] would always miss LANDED/CRASHED).
                    contact_pre = info["contact_state_pre_reset"].long()
                    vel_frd_pre = info["linear_vel_frd_pre_reset"]
                    pos_pre = info["position_pre_reset"]
                    just_landed = (
                        (landed_step == -1)
                        & (prev_contact != int(ContactState.LANDED))
                        & (contact_pre == int(ContactState.LANDED))
                    )
                    if just_landed.any():
                        landed_step = torch.where(just_landed, torch.full_like(landed_step, step), landed_step)
                        td_speed = vel_frd_pre[:, 2].clamp(min=0.0)
                        touchdown_speed = torch.where(just_landed, td_speed, touchdown_speed)
                        pad_xy = env._target_position[:, :2]
                        pad_dist = (pos_pre[:, :2] - pad_xy).norm(dim=-1)
                        touchdown_pad_dist = torch.where(just_landed, pad_dist, touchdown_pad_dist)
                    crashed = crashed | (contact_pre == int(ContactState.CRASHED))
                    state = env._build_vehicle_state()
                    vertical_down_speed = vel_frd_pre[:, 2]
                    max_downward_speed = max(max_downward_speed, float(vertical_down_speed.max().item()))
                    max_upward_speed = max(max_upward_speed, float((-vertical_down_speed).max().item()))
                    ratio = (state.motor_omega / omega_max).clamp(min=0.0, max=1.0).pow(2)
                    delta_v_proxy = delta_v_proxy + ratio * rl_dt
                    prev_contact = contact_pre
                    reset_ids = (done | trunc).nonzero(as_tuple=False).squeeze(-1)
                    if len(reset_ids) > 0:
                        eval_pid.reset(reset_ids)
                        if eval_guidance is not None:
                            eval_guidance.reset(obs=next_obs, env_ids=reset_ids)
                    obs = next_obs
                    simulation_app.update()
            landed_mask = landed_step >= 0
            landed_fraction = float(landed_mask.float().mean().item())
            crashed_fraction = float(crashed.float().mean().item())
            mean_td = float(touchdown_speed[landed_mask].mean().item()) if landed_mask.any() else float("nan")
            max_td = float(touchdown_speed[landed_mask].max().item()) if landed_mask.any() else float("nan")
            mean_pd = float(touchdown_pad_dist[landed_mask].mean().item()) if landed_mask.any() else float("nan")
            max_pd = float(touchdown_pad_dist[landed_mask].max().item()) if landed_mask.any() else float("nan")
            mean_dv = float(delta_v_proxy.mean().item())
            passed = (
                landed_fraction >= 0.8
                and crashed_fraction <= 0.05
                and (max_td if max_td == max_td else float("inf")) < args.landing_touchdown_speed_limit
                and (max_pd if max_pd == max_pd else float("inf")) < args.landing_pad_distance_limit
            )
            return LandingEvalMetrics(
                landed_fraction=landed_fraction,
                crashed_fraction=crashed_fraction,
                mean_touchdown_speed=mean_td,
                max_touchdown_speed=max_td,
                mean_pad_distance=mean_pd,
                max_pad_distance=max_pd,
                mean_delta_v_proxy=mean_dv,
                mean_eval_throttle=throttle_sum / max(throttle_count, 1),
                max_eval_throttle=max_eval_throttle,
                max_upward_speed=max_upward_speed,
                max_downward_speed=max_downward_speed,
                passed=passed,
            )

        def evaluate(policy, seconds: float) -> EvalMetrics:
            set_eval_spawn()
            obs_dict, _ = env.reset(seed=args.seed)
            obs = obs_dict["policy"]
            eval_pid.reset()
            n_steps = int(seconds / (config.physics_dt * config.decimation))
            pos_errors: list[float] = []
            tilts: list[float] = []
            rates: list[float] = []
            with torch.no_grad():
                for _ in range(n_steps):
                    raw_action, _, _, _ = policy.get_action_and_value(
                        policy_observation(obs), deterministic=True
                    )
                    obs_dict, _, done, trunc, _ = env.step(
                        raw_to_env_action(raw_action, obs, eval_pid, eval_guidance)
                    )
                    next_obs = obs_dict["policy"]
                    reset_ids = (done | trunc).nonzero(as_tuple=False).squeeze(-1)
                    if len(reset_ids) > 0:
                        eval_pid.reset(reset_ids)
                        if eval_guidance is not None:
                            eval_guidance.reset(obs=next_obs, env_ids=reset_ids)
                    obs = next_obs
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
        set_training_spawn(0)
        obs_dict, _ = env.reset(seed=args.seed)
        obs = obs_dict["policy"]
        if train_guidance is not None:
            train_guidance.reset(obs=obs)

        if args.bc_steps > 0 and not args.residual_pid:
            print(f"PID behavior-cloning warm start: {args.bc_steps} gradient steps", flush=True)
            pid = PIDController(num_envs=num_envs, device=device)
            pid.reset()
            bc_guidance = make_guidance() if is_landing else None
            if bc_guidance is not None:
                bc_guidance.reset(obs=obs)
            for step in range(args.bc_steps):
                with torch.no_grad():
                    pid_obs = bc_guidance.modify_obs(obs) if bc_guidance is not None else obs
                    teacher_action = pid.compute_action(pid_obs)
                    if bc_guidance is not None:
                        teacher_action = bc_guidance.post_action(teacher_action, obs)
                    target_raw = env_to_raw_action(teacher_action)
                pred_latent, _value = model(policy_observation(obs))
                pred_raw = torch.tanh(pred_latent)
                loss = F.mse_loss(pred_raw, target_raw)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                obs_dict, _, done, trunc, _ = env.step(teacher_action)
                next_obs = obs_dict["policy"]
                reset_ids = (done | trunc).nonzero(as_tuple=False).squeeze(-1)
                if len(reset_ids) > 0:
                    pid.reset(reset_ids)
                    if bc_guidance is not None:
                        bc_guidance.reset(obs=next_obs, env_ids=reset_ids)
                obs = next_obs
                simulation_app.update()

                if (step + 1) % max(1, args.bc_steps // 5) == 0:
                    print(f"  bc_step={step + 1}/{args.bc_steps} loss={loss.item():.5f}", flush=True)

            bc_metrics = evaluate(model, min(args.eval_seconds, 10.0))
            print(f"BC eval: {asdict(bc_metrics)}", flush=True)
            set_training_spawn(0)
            obs_dict, _ = env.reset(seed=args.seed)
            obs = obs_dict["policy"]
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
        throttle_buf = torch.zeros((rollout_steps, num_envs), device=device)

        global_step = 0
        update = 0
        best_eval = None
        last_eval_bucket = -1
        last_save_bucket = -1
        start_time = time.time()
        loss_ema: float | None = None
        loss_plateau_count = 0
        loss_plateau_delta = float("inf")
        stop_for_loss_plateau = False
        train_pid = PIDController(num_envs=num_envs, device=device)
        train_pid.reset()
        if train_guidance is not None:
            train_guidance.reset(obs=obs)

        while global_step < args.total_steps:
            update += 1
            spawn_curriculum = set_training_spawn(global_step)
            for t in range(rollout_steps):
                global_step += num_envs
                obs_policy = policy_observation(obs)
                obs_buf[t] = obs_policy
                with torch.no_grad():
                    action_raw, logprob, _entropy, value = model.get_action_and_value(obs_policy)
                action_buf[t] = action_raw
                logprob_buf[t] = logprob
                value_buf[t] = value

                env_action = raw_to_env_action(action_raw, obs, train_pid, train_guidance)
                throttle_buf[t] = env_action[:, 4]
                obs_dict, reward, terminated, truncated, _ = env.step(env_action)
                done = terminated | truncated
                next_obs = obs_dict["policy"]
                reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
                if len(reset_ids) > 0:
                    train_pid.reset(reset_ids)
                    if train_guidance is not None:
                        train_guidance.reset(obs=next_obs, env_ids=reset_ids)
                reward_buf[t] = reward * args.reward_scale
                done_buf[t] = done.float()
                obs = next_obs
                simulation_app.update()

            with torch.no_grad():
                _next_mean, next_value = model(policy_observation(obs))
                advantages = torch.zeros_like(reward_buf)
                lastgaelam = torch.zeros(num_envs, device=device)
                for t in reversed(range(rollout_steps)):
                    next_nonterminal = 1.0 - done_buf[t]
                    if t == rollout_steps - 1:
                        next_values = next_value
                    else:
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
            pg_loss_acc = 0.0
            v_loss_acc = 0.0
            entropy_acc = 0.0
            loss_acc = 0.0
            n_minibatches = 0
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

                    pg_loss_acc += float(pg_loss.detach().item())
                    v_loss_acc += float(v_loss.detach().item())
                    entropy_acc += float(entropy_loss.detach().item())
                    loss_acc += float(loss.detach().item())
                    n_minibatches += 1
                if approx_kl > args.target_kl:
                    break

            denom = max(n_minibatches, 1)
            mean_pg = pg_loss_acc / denom
            mean_v = v_loss_acc / denom
            mean_ent = entropy_acc / denom
            mean_loss = loss_acc / denom
            mean_clipfrac = sum(clipfracs) / max(len(clipfracs), 1)
            wall_elapsed = time.time() - start_time
            sps = global_step / max(wall_elapsed, 1e-6)
            if loss_ema is None:
                loss_ema = mean_loss
                loss_plateau_delta = float("inf")
            else:
                previous_loss_ema = loss_ema
                loss_ema = (
                    args.early_stop_ema_alpha * mean_loss
                    + (1.0 - args.early_stop_ema_alpha) * loss_ema
                )
                loss_plateau_delta = abs(loss_ema - previous_loss_ema)
                if (
                    not args.no_early_stop
                    and update >= args.early_stop_warmup_updates
                    and loss_plateau_delta <= args.early_stop_min_delta
                ):
                    loss_plateau_count += 1
                else:
                    loss_plateau_count = 0
                stop_for_loss_plateau = (
                    not args.no_early_stop
                    and loss_plateau_count >= args.early_stop_patience
                )
            log_record = {
                "type": "train_update",
                "update": update,
                "global_step": global_step,
                "wall_s": round(wall_elapsed, 2),
                "sps": round(sps, 1),
                "loss": round(mean_loss, 6),
                "loss_ema": round(float(loss_ema), 6),
                "loss_plateau_delta": (
                    None if loss_plateau_delta == float("inf") else round(float(loss_plateau_delta), 6)
                ),
                "loss_plateau_count": loss_plateau_count,
                "pg_loss": round(mean_pg, 6),
                "v_loss": round(mean_v, 6),
                "entropy": round(mean_ent, 6),
                "approx_kl": round(float(approx_kl.item()), 6),
                "clipfrac": round(mean_clipfrac, 4),
                "reward_mean": round(float(reward_buf.mean().item()), 4),
                "reward_std": round(float(reward_buf.std().item()), 4),
                "reward_min": round(float(reward_buf.min().item()), 4),
                "reward_max": round(float(reward_buf.max().item()), 4),
                "reward_scale": args.reward_scale,
                "throttle_mean": round(float(throttle_buf.mean().item()), 4),
                "throttle_min": round(float(throttle_buf.min().item()), 4),
                "throttle_max": round(float(throttle_buf.max().item()), 4),
                "spawn_curriculum_enabled": spawn_curriculum.enabled,
                "spawn_curriculum_progress": round(float(spawn_curriculum.progress), 4),
                "spawn_xy_half_width_m": round(spawn_xy_half_width(spawn_curriculum.position_range), 3),
            }
            print(json.dumps(log_record, separators=(",", ":")), flush=True)
            append_jsonl("train_log.jsonl", log_record)

            save_bucket = global_step // max(args.save_interval, 1)
            if save_bucket > last_save_bucket or global_step >= args.total_steps:
                ckpt = output_dir / f"ppo_step_{global_step}.pt"
                torch.save({"model": model.state_dict(), "args": vars(args), "step": global_step}, ckpt)
                last_save_bucket = save_bucket

            eval_bucket = global_step // max(args.eval_interval, 1)
            if eval_bucket > last_eval_bucket or global_step >= args.total_steps:
                last_eval_bucket = eval_bucket
                if is_landing:
                    metrics = evaluate_landing(model, args.eval_seconds)
                    best_eval = (
                        metrics
                        if best_eval is None or metrics.mean_delta_v_proxy < best_eval.mean_delta_v_proxy
                        else best_eval
                    )
                else:
                    metrics = evaluate(model, args.eval_seconds)
                    best_eval = (
                        metrics
                        if best_eval is None or metrics.mean_pos < best_eval.mean_pos
                        else best_eval
                    )
                (output_dir / "eval_latest.json").write_text(
                    json.dumps(asdict(metrics), indent=2), encoding="utf-8"
                )
                eval_record = {
                    "type": "eval",
                    "global_step": global_step,
                    "update": update,
                    "wall_s": round(time.time() - start_time, 2),
                    **asdict(metrics),
                }
                append_jsonl("eval_log.jsonl", eval_record)
                print(
                    f"step={global_step:,} reward_mean={reward_buf.mean().item():.3f} "
                    f"kl={approx_kl.item():.5f} clipfrac={sum(clipfracs)/max(len(clipfracs),1):.3f} "
                    f"eval={asdict(metrics)}",
                    flush=True,
                )
                set_training_spawn(global_step)
                obs_dict, _ = env.reset(seed=args.seed)
                obs = obs_dict["policy"]
                if train_guidance is not None:
                    train_guidance.reset(obs=obs)

            if stop_for_loss_plateau:
                print(
                    f"Early stop: loss EMA plateaued for {loss_plateau_count} updates "
                    f"(delta <= {args.early_stop_min_delta}).",
                    flush=True,
                )
                break

        elapsed = time.time() - start_time
        final_metrics = evaluate_landing(model, args.eval_seconds) if is_landing else evaluate(model, args.eval_seconds)
        torch.save({"model": model.state_dict(), "args": vars(args), "step": global_step}, output_dir / "ppo_final.pt")
        (output_dir / "eval_final.json").write_text(json.dumps(asdict(final_metrics), indent=2), encoding="utf-8")
        append_jsonl(
            "eval_log.jsonl",
            {
                "type": "final_eval",
                "global_step": global_step,
                "update": update,
                "wall_s": round(time.time() - start_time, 2),
                **asdict(final_metrics),
            },
        )
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
