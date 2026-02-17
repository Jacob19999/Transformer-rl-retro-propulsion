"""
PPO-MLP training entry point (Stage 19.4–19.6).

Implements training.md §13.3 for SB3:
- Vectorized envs via SubprocVecEnv
- Observation normalization via VecNormalize (freeze stats for eval)
- TensorBoard logging
- Checkpointing every 500K steps and "best model" saving
- Periodic evaluation (50 eps every 100K steps) logging success rate + CEP

Run (example):
    python -m simulation.training.scripts.train_ppo_mlp --seed 0
"""

from __future__ import annotations

import argparse
import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import time

from simulation.config_loader import load_config
from simulation.training.edf_landing_env import EDFLandingEnv


def _repo_root() -> Path:
    # train_ppo_mlp.py -> scripts/ -> training/ -> simulation/ -> repo root
    return Path(__file__).resolve().parents[4]


def _sim_configs_dir() -> Path:
    # train_ppo_mlp.py -> scripts/ -> training/ -> simulation/
    return Path(__file__).resolve().parents[2] / "configs"


def _linear_schedule(initial_value: float) -> Callable[[float], float]:
    """SB3 linear schedule: progress_remaining goes 1 -> 0."""

    init = float(initial_value)

    def f(progress_remaining: float) -> float:
        return float(progress_remaining) * init

    return f


def _activation_fn(name: str):
    try:
        import torch as th
        import torch.nn as nn
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "torch is required for PPO-MLP training. Install with `pip install torch`."
        ) from e

    n = str(name).strip().lower()
    if n in ("tanh", "nn.tanh"):
        return nn.Tanh
    if n in ("relu", "nn.relu"):
        return nn.ReLU
    if n in ("elu", "nn.elu"):
        return nn.ELU
    raise ValueError(f"Unknown activation_fn {name!r}. Supported: tanh, relu, elu.")


@dataclass(frozen=True, slots=True)
class PpoMlpConfig:
    total_timesteps: int
    n_envs: int
    hyper: dict[str, Any]
    policy: dict[str, Any]
    schedule: dict[str, Any]
    ckpt: dict[str, Any]


def _load_ppo_mlp_yaml(path: Path | None) -> PpoMlpConfig:
    p = path or (_sim_configs_dir() / "ppo_mlp.yaml")
    raw = load_config(p)
    cfg = dict(raw.get("ppo_mlp", raw))

    return PpoMlpConfig(
        total_timesteps=int(cfg.get("total_timesteps", 10_000_000)),
        n_envs=int(cfg.get("n_envs", 16)),
        hyper=dict(cfg.get("hyperparameters", {})),
        policy=dict(cfg.get("policy", {})),
        schedule=dict(cfg.get("schedule", {})),
        ckpt=dict(cfg.get("checkpointing", {})),
    )


def _make_env_fn(*, env_root_config: Mapping[str, Any] | str | Path | None, seed: int):
    """Closure factory for SubprocVecEnv."""

    def _thunk():
        try:
            from stable_baselines3.common.monitor import Monitor
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "stable-baselines3 is required for PPO-MLP training. "
                "Install with `pip install stable-baselines3`."
            ) from e

        env = EDFLandingEnv(env_root_config)
        env = Monitor(env)
        # Gymnasium seeding happens in reset().
        env.reset(seed=int(seed))
        return env

    return _thunk


class _CheckpointAndNormalizeCallback:
    """Save model checkpoints AND VecNormalize stats on the same cadence."""

    def __init__(self, *, save_freq: int, out_dir: Path, vecnormalize) -> None:
        self.save_freq = int(save_freq)
        self.out_dir = Path(out_dir)
        self.vecnormalize = vecnormalize
        self._last_saved_at = 0

        # SB3 will set these on callback init
        self.model = None  # type: ignore[assignment]
        self.num_timesteps = 0

    def init_callback(self, model) -> None:
        self.model = model

    def _maybe_save(self) -> None:
        if (self.num_timesteps - self._last_saved_at) < self.save_freq:
            return
        self._last_saved_at = int(self.num_timesteps)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        step = int(self.num_timesteps)
        self.model.save(str(self.out_dir / f"checkpoint_{step}.zip"))
        self.vecnormalize.save(str(self.out_dir / f"vecnormalize_{step}.pkl"))
        print(f"[checkpoint] steps={step} saved to {self.out_dir}", flush=True)


class _EvalSuccessCallback:
    """Periodic evaluation: log success rate and CEP; save best model."""

    def __init__(
        self,
        *,
        eval_env,
        train_vecnormalize,
        eval_freq: int,
        n_eval_episodes: int,
        out_dir: Path,
    ) -> None:
        self.eval_env = eval_env
        self.train_vecnormalize = train_vecnormalize
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.out_dir = Path(out_dir)

        self._last_eval_at = 0
        self._best_success = -np.inf

        self.model = None  # type: ignore[assignment]
        self.num_timesteps = 0
        self.logger = None  # set by SB3

    def init_callback(self, model) -> None:
        self.model = model

    def _sync_norm(self) -> None:
        # Use current training normalization stats for evaluation.
        if hasattr(self.eval_env, "obs_rms") and hasattr(self.train_vecnormalize, "obs_rms"):
            self.eval_env.obs_rms = self.train_vecnormalize.obs_rms
        if hasattr(self.eval_env, "ret_rms") and hasattr(self.train_vecnormalize, "ret_rms"):
            self.eval_env.ret_rms = self.train_vecnormalize.ret_rms

    def _evaluate(self) -> dict[str, float]:
        self._sync_norm()

        successes: list[bool] = []
        ceps: list[float] = []
        ep_rewards: list[float] = []
        ep_lengths: list[int] = []

        obs = self.eval_env.reset()
        n_done = 0
        cur_rew = 0.0
        cur_len = 0

        while n_done < self.n_eval_episodes:
            action, _state = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = self.eval_env.step(action)
            # VecEnv returns arrays even for n_env=1
            r0 = float(np.asarray(rewards).reshape(-1)[0])
            d0 = bool(np.asarray(dones).reshape(-1)[0])
            info0 = (infos[0] if isinstance(infos, (list, tuple)) and infos else {}) or {}

            cur_rew += r0
            cur_len += 1

            if d0:
                successes.append(bool(info0.get("landed", False)))
                ceps.append(float(info0.get("cep", float("nan"))))
                ep_rewards.append(float(cur_rew))
                ep_lengths.append(int(cur_len))
                cur_rew = 0.0
                cur_len = 0
                n_done += 1

        success_rate = float(np.mean(successes)) if successes else 0.0
        mean_cep_success = float(
            np.mean([c for c, s in zip(ceps, successes, strict=True) if s])
        ) if any(successes) else float("inf")
        mean_cep_all = float(np.nanmean(ceps)) if ceps else float("nan")
        mean_ep_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        mean_ep_len = float(np.mean(ep_lengths)) if ep_lengths else 0.0

        return {
            "eval/success_rate": success_rate,
            "eval/mean_cep_success": mean_cep_success,
            "eval/mean_cep_all": mean_cep_all,
            "eval/mean_ep_reward": mean_ep_reward,
            "eval/mean_ep_len": mean_ep_len,
        }

    def _maybe_eval_and_log(self) -> None:
        if (self.num_timesteps - self._last_eval_at) < self.eval_freq:
            return
        self._last_eval_at = int(self.num_timesteps)

        metrics = self._evaluate()
        print(
            "[eval] "
            f"steps={int(self.num_timesteps)} "
            f"success_rate={metrics['eval/success_rate']:.3f} "
            f"mean_cep_success={metrics['eval/mean_cep_success']:.3f} "
            f"mean_ep_len={metrics['eval/mean_ep_len']:.1f}",
            flush=True,
        )
        if self.logger is not None:
            for k, v in metrics.items():
                self.logger.record(k, float(v))
            self.logger.dump(int(self.num_timesteps))

        # Save best model by success rate, with CEP tiebreak.
        success = float(metrics["eval/success_rate"])
        is_best = success > float(self._best_success) + 1e-12
        if is_best:
            self._best_success = success
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.out_dir / "best_model.zip"))
            self.train_vecnormalize.save(str(self.out_dir / "best_vecnormalize.pkl"))
            print(
                f"[best] steps={int(self.num_timesteps)} success_rate={success:.3f} "
                f"saved best_model.zip + best_vecnormalize.pkl to {self.out_dir}",
                flush=True,
            )


class _RealtimeTrainPrintCallback:
    """Print one line per PPO update/iteration for realtime monitoring."""

    def __init__(self) -> None:
        self._last_n_updates: int | None = None
        self._t0 = time.perf_counter()
        self._last_steps = 0

        self.model = None  # set by SB3
        self.num_timesteps = 0

    def init_callback(self, model) -> None:
        self.model = model

    def _fmt(self, x: object, *, default: str = "NA") -> str:
        try:
            if x is None:
                return default
            v = float(x)
            if np.isnan(v) or np.isinf(v):
                return default
            return f"{v:.4g}"
        except Exception:
            return default

    def _maybe_print(self) -> None:
        n_updates = getattr(self.model, "_n_updates", None)
        if n_updates is None:
            return
        if self._last_n_updates is None:
            self._last_n_updates = int(n_updates)
            self._last_steps = int(self.num_timesteps)
            return
        if int(n_updates) == int(self._last_n_updates):
            return

        now = time.perf_counter()
        dt = max(1e-9, now - self._t0)
        dsteps = int(self.num_timesteps) - int(self._last_steps)
        sps = float(dsteps) / dt
        self._t0 = now
        self._last_steps = int(self.num_timesteps)
        self._last_n_updates = int(n_updates)

        # Pull whatever SB3 recorded most recently; keys vary slightly across versions.
        name_to_value = getattr(getattr(self.model, "logger", None), "name_to_value", {}) or {}
        ep_rew = name_to_value.get("rollout/ep_rew_mean", None)
        ep_len = name_to_value.get("rollout/ep_len_mean", None)
        kl = name_to_value.get("train/approx_kl", None)
        clipfrac = name_to_value.get("train/clip_fraction", None)
        ent = name_to_value.get("train/entropy_loss", None)
        vf = name_to_value.get("train/value_loss", None)
        pg = name_to_value.get("train/policy_gradient_loss", None)
        ev = name_to_value.get("train/explained_variance", None)
        lr = name_to_value.get("train/learning_rate", None)

        print(
            "[train] "
            f"update={int(n_updates)} "
            f"steps={int(self.num_timesteps)} "
            f"steps/s={sps:.0f} "
            f"ep_rew_mean={self._fmt(ep_rew)} "
            f"ep_len_mean={self._fmt(ep_len)} "
            f"kl={self._fmt(kl)} "
            f"clipfrac={self._fmt(clipfrac)} "
            f"ent={self._fmt(ent)} "
            f"vf_loss={self._fmt(vf)} "
            f"pg_loss={self._fmt(pg)} "
            f"ev={self._fmt(ev)} "
            f"lr={self._fmt(lr)}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 19 PPO-MLP training (SB3).")
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    p.add_argument(
        "--ppo_config",
        type=str,
        default=str(_sim_configs_dir() / "ppo_mlp.yaml"),
        help="Path to PPO-MLP YAML (simulation/configs/ppo_mlp.yaml).",
    )
    p.add_argument(
        "--env_config",
        type=str,
        default="",
        help="Optional path to a root env YAML config. If omitted, uses EDFLandingEnv defaults.",
    )
    p.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Output directory (logs/checkpoints). Defaults to runs/ppo_mlp/<timestamp>.",
    )
    p.add_argument("--device", type=str, default="auto", help="SB3 device (auto/cpu/cuda).")
    p.add_argument(
        "--num_envs",
        type=int,
        default=0,
        help="Override n_envs from YAML (0 = use YAML).",
    )
    args = p.parse_args(argv)

    try:
        import torch  # noqa: F401
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "PPO-MLP training requires stable-baselines3, torch, and tensorboard. "
            "Install from requirements.txt."
        ) from e

    ppo_cfg = _load_ppo_mlp_yaml(Path(args.ppo_config))
    n_envs = int(args.num_envs) if int(args.num_envs) > 0 else int(ppo_cfg.n_envs)

    # Environment root config: EDFLandingEnv can load defaults when None.
    env_root_cfg: Mapping[str, Any] | str | Path | None
    env_root_cfg = Path(args.env_config) if str(args.env_config).strip() else None

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if str(args.run_dir).strip() else (_repo_root() / "runs" / "ppo_mlp" / ts)
    tb_dir = run_dir / "tb"
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build vectorized envs
    env_fns = [
        _make_env_fn(env_root_config=env_root_cfg, seed=int(args.seed) + i)
        for i in range(int(n_envs))
    ]
    venv = SubprocVecEnv(env_fns)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Separate eval env with frozen normalization (stats synced from training venv)
    eval_env = DummyVecEnv([_make_env_fn(env_root_config=env_root_cfg, seed=int(args.seed) + 10_000)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    # Policy kwargs
    pol = dict(ppo_cfg.policy)
    net_arch = pol.get("net_arch", [256, 256])
    activation = _activation_fn(str(pol.get("activation_fn", "tanh")))
    ortho_init = bool(pol.get("ortho_init", True))
    log_std_init = float(pol.get("log_std_init", -0.5))
    policy_kwargs = dict(
        net_arch=list(net_arch),
        activation_fn=activation,
        ortho_init=ortho_init,
        log_std_init=log_std_init,
    )

    # Hyperparameters + schedules
    hyper = dict(ppo_cfg.hyper)
    lr = float(hyper.get("learning_rate", 3e-4))
    sched = dict(ppo_cfg.schedule)
    if str(sched.get("lr_schedule", "linear")).strip().lower() == "linear":
        learning_rate = _linear_schedule(lr)
    else:
        learning_rate = lr

    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=learning_rate,
        n_steps=int(hyper.get("n_steps", 2048)),
        batch_size=int(hyper.get("batch_size", 256)),
        n_epochs=int(hyper.get("n_epochs", 10)),
        gamma=float(hyper.get("gamma", 0.99)),
        gae_lambda=float(hyper.get("gae_lambda", 0.95)),
        clip_range=float(hyper.get("clip_range", 0.2)),
        ent_coef=float(hyper.get("ent_coef", 0.01)),
        vf_coef=float(hyper.get("vf_coef", 0.5)),
        max_grad_norm=float(hyper.get("max_grad_norm", 0.5)),
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tb_dir),
        device=str(args.device),
        seed=int(args.seed),
        verbose=1,
    )

    ckpt = dict(ppo_cfg.ckpt)
    save_freq = int(ckpt.get("save_freq", 500_000))
    eval_freq = int(ckpt.get("eval_freq", 100_000))
    eval_eps = int(ckpt.get("eval_episodes", 50))

    class _AdapterCallback(BaseCallback):
        """Adapt our small callbacks to SB3 BaseCallback API."""

        def __init__(self, inner) -> None:
            super().__init__()
            self.inner = inner

        def _on_training_start(self) -> None:
            self.inner.init_callback(self.model)
            if hasattr(self.inner, "logger"):
                self.inner.logger = self.logger

        def _on_step(self) -> bool:
            # Keep num_timesteps mirrored.
            self.inner.num_timesteps = int(self.num_timesteps)
            if hasattr(self.inner, "_maybe_save"):
                self.inner._maybe_save()
            if hasattr(self.inner, "_maybe_eval_and_log"):
                self.inner._maybe_eval_and_log()
            if hasattr(self.inner, "_maybe_print"):
                self.inner._maybe_print()
            return True

    callbacks = CallbackList(
        [
            _AdapterCallback(_RealtimeTrainPrintCallback()),
            _AdapterCallback(
                _CheckpointAndNormalizeCallback(
                    save_freq=save_freq, out_dir=ckpt_dir, vecnormalize=venv
                )
            ),
            _AdapterCallback(
                _EvalSuccessCallback(
                    eval_env=eval_env,
                    train_vecnormalize=venv,
                    eval_freq=eval_freq,
                    n_eval_episodes=eval_eps,
                    out_dir=run_dir,
                )
            ),
        ]
    )

    model.learn(
        total_timesteps=int(ppo_cfg.total_timesteps),
        callback=callbacks,
        progress_bar=True,
        log_interval=1,  # print every PPO iteration via SB3 logger + our callback
    )

    # Final save
    model.save(str(run_dir / "final_model.zip"))
    venv.save(str(run_dir / "final_vecnormalize.pkl"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

