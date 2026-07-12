"""Replay a trained PPO landing policy in a single environment for N episodes.

Loads a saved actor-critic checkpoint, runs the policy deterministically against
the landing task in single-env mode (headed by default), and reports per-episode
landed/crashed/touchdown-speed/pad-distance plus a summary over all episodes.

Usage (typical):
    python apps/run_eval_ppo.py \
        --checkpoint runs/<run_dir>/ppo_step_<N>.pt \
        --episodes 100 --no-headless
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from runner_safety import WallClockWatchdog, force_process_exit


def parse_args():
    parser = argparse.ArgumentParser(description="PPO landing policy evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--task", default="landing", choices=["hover", "landing"])
    parser.add_argument(
        "--env-config",
        default="configs/env/single_env_debug.yaml",
        help="Single-env YAML (num_envs must be 1).",
    )
    parser.add_argument("--disturbance", default="configs/disturbances/nominal.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-episode-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument(
        "--max-wall-time",
        type=float,
        default=None,
        help="Max wall-clock seconds before forcing process exit.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to write per-episode JSONL + summary JSON.",
    )
    return parser.parse_args()


@dataclass
class EpisodeRecord:
    episode: int
    outcome: str               # "LANDED" | "CRASHED" | "TIMEOUT"
    duration_s: float
    touchdown_speed: float     # m/s downward at termination (NaN if not landed)
    pad_distance: float        # m horizontal at termination (NaN if not landed)
    max_downward_speed: float
    mean_throttle: float
    # Phase-1 lateral-accuracy diagnosis: signed positions let us tell
    # "aim and miss" (touchdown_xy points roughly from spawn toward pad)
    # apart from "open-loop drift" (touchdown_xy ~ spawn_xy + initial_v_xy * t).
    spawn_x: float
    spawn_y: float
    spawn_xy_dist: float
    touchdown_x: float         # NaN if not landed/crashed
    touchdown_y: float
    touchdown_throttle: float  # commanded throttle on the terminal step


def main():
    args = parse_args()
    if args.max_wall_time is None:
        # generous: ~30s startup + episodes * (max_episode_s + reset overhead)
        args.max_wall_time = 60.0 + args.episodes * (args.max_episode_seconds + 2.0)
    watchdog = WallClockWatchdog(args.max_wall_time, label="PPO eval")
    watchdog.start()

    sim_root = Path(__file__).parent.parent
    sys.path.insert(0, str(sim_root))

    simulation_app = None
    env = None
    try:
        from isaac_launcher import launch_simulation_app
        simulation_app = launch_simulation_app(headless=args.headless)
    except ImportError:
        print("ERROR: Isaac Sim not available.", file=sys.stderr, flush=True)
        watchdog.stop()
        return 1

    try:
        import torch
        import torch.nn as nn
        from torch.distributions import Normal

        from tvc_env.common.constants import ContactState
        from tvc_env.common.frames import isaac_position_to_frd
        from tvc_env.common.quaternions import inverse as quat_inv, normalize, rotate_vector
        from tvc_env.envs.base_env import BaseEnvConfig
        from tvc_env.envs.direct_rl_env import TVCDirectRLEnv

        torch.manual_seed(args.seed)

        config = BaseEnvConfig(
            task_name=args.task,
            env_config_path=sim_root / args.env_config,
            disturbance_config_path=sim_root / args.disturbance if args.disturbance else None,
            sim_root=sim_root,
        )
        if config.num_envs != 1:
            raise ValueError(f"--env-config must specify num_envs=1, got {config.num_envs}")
        env = TVCDirectRLEnv(config)
        device = env.device
        obs_dim = 24
        act_dim = 5
        max_fin_angle = float(env._servo_model.max_command_angle)
        rl_dt = config.physics_dt * config.decimation
        max_episode_steps = int(args.max_episode_seconds / rl_dt)

        # Match the actor-critic architecture from run_train_ppo.py so the
        # checkpoint state-dict loads cleanly.
        class ActorCritic(nn.Module):
            def __init__(self):
                super().__init__()
                self.actor = nn.Sequential(
                    nn.Linear(obs_dim, 256),
                    nn.Tanh(),
                    nn.Linear(256, 256),
                    nn.Tanh(),
                    nn.Linear(256, act_dim),
                )
                self.critic = nn.Sequential(
                    nn.Linear(obs_dim, 256),
                    nn.Tanh(),
                    nn.Linear(256, 256),
                    nn.Tanh(),
                    nn.Linear(256, 1),
                )
                self.log_std = nn.Parameter(torch.zeros(act_dim))

            def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
                mean = self.actor(obs)
                return torch.tanh(mean)

        def raw_to_env_action(action_raw: torch.Tensor) -> torch.Tensor:
            fins = action_raw[:, :4].clamp(-1.0, 1.0) * max_fin_angle
            throttle = (action_raw[:, 4:5].clamp(-1.0, 1.0) + 1.0) * 0.5
            return torch.cat([fins, throttle], dim=-1)

        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = sim_root / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        train_args = ckpt.get("args", {})
        body_frame_position_error = bool(train_args.get("body_frame_position_error", False))
        model = ActorCritic().to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(
            f"Loaded checkpoint {ckpt_path.name} "
            f"(trained for {ckpt.get('step', '?')} steps); "
            f"running {args.episodes} episodes in single env, headless={args.headless}.",
            flush=True,
        )

        def policy_observation(obs_raw: torch.Tensor) -> torch.Tensor:
            if not body_frame_position_error:
                return obs_raw
            obs_policy = obs_raw.clone()
            pos_error_world = obs_policy[:, 0:3]
            q_inv = quat_inv(normalize(obs_policy[:, 3:7]))
            pos_error_body_isaac = rotate_vector(q_inv, pos_error_world)
            obs_policy[:, 0:3] = isaac_position_to_frd(pos_error_body_isaac)
            return obs_policy

        out_dir = None
        if args.output_dir is not None:
            out_dir = Path(args.output_dir)
            if not out_dir.is_absolute():
                out_dir = sim_root / out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

        episodes: list[EpisodeRecord] = []
        obs_dict, _ = env.reset(seed=args.seed)
        obs = obs_dict["policy"]

        for ep_idx in range(1, args.episodes + 1):
            step = 0
            outcome = "TIMEOUT"
            touchdown_speed = float("nan")
            pad_distance = float("nan")
            max_downward = 0.0
            throttle_sum = 0.0
            throttle_count = 0
            t0 = time.time()
            # Spawn pose: env auto-resets at episode boundary, so the obs we
            # already hold reflects the new spawn. position_error = target - pos
            # in world frame (3D), so spawn xy = pad_xy - obs[0,0:2].
            pad_xy0 = env._target_position[0, :2]
            spawn_xy = (pad_xy0 - obs[0, 0:2]).detach().cpu().tolist()
            spawn_x = float(spawn_xy[0])
            spawn_y = float(spawn_xy[1])
            spawn_xy_dist = float((pad_xy0 - obs[0, 0:2]).norm().item())
            touchdown_x = float("nan")
            touchdown_y = float("nan")
            touchdown_throttle = float("nan")
            episode_finished = False

            with torch.no_grad():
                while step < max_episode_steps:
                    raw_action = model.deterministic_action(policy_observation(obs))
                    env_action = raw_to_env_action(raw_action)
                    throttle_cmd = float(env_action[0, 4].item())
                    throttle_sum += throttle_cmd
                    throttle_count += 1
                    obs_dict, _, terminated, truncated, info = env.step(env_action)
                    next_obs = obs_dict["policy"]

                    contact_pre = info["contact_state_pre_reset"].long()
                    vel_frd_pre = info["linear_vel_frd_pre_reset"]
                    pos_pre = info["position_pre_reset"]
                    impact_speed_pre = info["touchdown_speed_pre_reset"]
                    max_downward = max(max_downward, float(vel_frd_pre[0, 2].clamp(min=0.0).item()))

                    landed_now = bool((contact_pre == int(ContactState.LANDED))[0])
                    crashed_now = bool((contact_pre == int(ContactState.CRASHED))[0])
                    done = bool((terminated | truncated)[0])

                    if landed_now or crashed_now:
                        outcome = "LANDED" if landed_now else "CRASHED"
                        touchdown_speed = float(impact_speed_pre[0].item())
                        pad_xy = env._target_position[0, :2]
                        td_offset = pos_pre[0, :2] - pad_xy
                        pad_distance = float(td_offset.norm().item())
                        touchdown_x = float(pos_pre[0, 0].item())
                        touchdown_y = float(pos_pre[0, 1].item())
                        touchdown_throttle = throttle_cmd

                    obs = next_obs
                    simulation_app.update()
                    step += 1
                    if done:
                        episode_finished = True
                        break

            if not episode_finished:
                obs_dict, _ = env.reset()
                obs = obs_dict["policy"]

            duration = time.time() - t0
            mean_throttle = throttle_sum / max(throttle_count, 1)

            def _r(x: float, n: int = 4) -> float:
                return round(x, n) if x == x else float("nan")

            rec = EpisodeRecord(
                episode=ep_idx,
                outcome=outcome,
                duration_s=round(step * rl_dt, 3),
                touchdown_speed=_r(touchdown_speed),
                pad_distance=_r(pad_distance),
                max_downward_speed=round(max_downward, 3),
                mean_throttle=round(mean_throttle, 4),
                spawn_x=round(spawn_x, 4),
                spawn_y=round(spawn_y, 4),
                spawn_xy_dist=round(spawn_xy_dist, 4),
                touchdown_x=_r(touchdown_x),
                touchdown_y=_r(touchdown_y),
                touchdown_throttle=_r(touchdown_throttle),
            )
            episodes.append(rec)
            print(
                f"ep={ep_idx:>3}/{args.episodes} outcome={rec.outcome:<7} "
                f"spawn_xy=({rec.spawn_x:>+5.2f},{rec.spawn_y:>+5.2f}) d0={rec.spawn_xy_dist:>4.2f} "
                f"td_xy=({rec.touchdown_x:>+5.2f},{rec.touchdown_y:>+5.2f}) pad_d={rec.pad_distance:>5.2f} "
                f"td_speed={rec.touchdown_speed:>5.2f} td_thr={rec.touchdown_throttle:>4.2f} "
                f"dur={rec.duration_s:>5.2f}s wall={duration:>4.1f}s",
                flush=True,
            )
            if out_dir is not None:
                with (out_dir / "episodes.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(rec)) + "\n")

        # Summary
        n = len(episodes)
        landed = [r for r in episodes if r.outcome == "LANDED"]
        crashed = [r for r in episodes if r.outcome == "CRASHED"]
        timeouts = [r for r in episodes if r.outcome == "TIMEOUT"]
        on_pad = [r for r in landed if r.pad_distance < 0.5]

        def _stats(values):
            if not values:
                return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
            return {
                "mean": round(sum(values) / len(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }

        # Aim-and-miss vs. drift diagnostic: bin landed episodes by spawn distance
        # and summarize how much horizontal range was closed. Closure ratio
        # 1.0 = touched down on pad, 0.0 = ended at the spawn xy (no closure),
        # negative = drifted further from pad than spawn.
        def _closure_stats(records):
            ratios = []
            for r in records:
                d0 = r.spawn_xy_dist
                if d0 < 1e-3 or r.pad_distance != r.pad_distance:
                    continue
                ratios.append((d0 - r.pad_distance) / d0)
            if not ratios:
                return {"mean": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
            return {
                "mean": round(sum(ratios) / len(ratios), 4),
                "min": round(min(ratios), 4),
                "max": round(max(ratios), 4),
                "n": len(ratios),
            }

        spawn_bins = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]
        by_spawn = {}
        for lo, hi in spawn_bins:
            bucket = [r for r in landed if lo <= r.spawn_xy_dist < hi]
            if not bucket:
                continue
            by_spawn[f"d0_{lo:.1f}_{hi:.1f}"] = {
                "n": len(bucket),
                "mean_pad_distance": round(sum(r.pad_distance for r in bucket) / len(bucket), 4),
                "max_pad_distance": round(max(r.pad_distance for r in bucket), 4),
                "mean_closure_ratio": _closure_stats(bucket)["mean"],
                "on_pad_fraction": round(sum(1 for r in bucket if r.pad_distance < 0.5) / len(bucket), 4),
            }

        summary = {
            "checkpoint": str(ckpt_path),
            "trained_steps": ckpt.get("step", None),
            "episodes": n,
            "landed_fraction": round(len(landed) / n, 4),
            "crashed_fraction": round(len(crashed) / n, 4),
            "timeout_fraction": round(len(timeouts) / n, 4),
            "on_pad_fraction": round(len(on_pad) / n, 4),  # landed AND within 0.5 m
            "touchdown_speed_landed": _stats([r.touchdown_speed for r in landed]),
            "pad_distance_landed": _stats([r.pad_distance for r in landed]),
            "spawn_xy_distance": _stats([r.spawn_xy_dist for r in episodes]),
            "closure_ratio_landed": _closure_stats(landed),
            "by_spawn_distance": by_spawn,
            "duration_s": _stats([r.duration_s for r in episodes]),
        }
        print("=" * 72, flush=True)
        print(json.dumps(summary, indent=2), flush=True)

        if out_dir is not None:
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"Wrote results to {out_dir}", flush=True)

        return 0

    except Exception as exc:
        print(f"\nERROR: PPO eval failed: {exc}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        return 2
    finally:
        watchdog.reset(30.0, label="PPO eval cleanup")
        if env is not None:
            print("Closing TVC environment...", flush=True)
            env.close()
        if simulation_app is not None:
            print("Closing Isaac Sim...", flush=True)
            from isaac_launcher import close_simulation_app
            close_simulation_app(simulation_app)
        watchdog.stop()


if __name__ == "__main__":
    force_process_exit(main())
