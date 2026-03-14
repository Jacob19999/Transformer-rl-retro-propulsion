"""
edf_isaac_env.py -- Gymnasium-compatible wrapper for EdfLandingTask.

Implements T021:
- Wraps EdfLandingTask (IsaacLab DirectRLEnv)
- Converts PyTorch tensors to numpy arrays
- Exposes observation_space = Box(-inf, inf, (20,), float32)
- Exposes action_space = Box(-1, 1, (5,), float32)
- Implements auto-reset on done (SB3 VecEnv contract)
- VRAM guard for large num_envs on RTX 5070 (T031)

Contract: specs/001-isaac-sim-env/contracts/gymnasium-env-interface.md
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import gymnasium as gym
from gymnasium import spaces

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from simulation.config_loader import load_config  # noqa: E402
from simulation.isaac.conventions import ACTION_DIM, OBS_DIM, OBS_H_AGL  # noqa: E402

logger = logging.getLogger(__name__)

# VRAM guard threshold (GB)
_VRAM_WARN_ENVS = 512
_VRAM_WARN_GB   = 16.0


class EDFIsaacEnv(gym.Env):
    """Gymnasium wrapper around EdfLandingTask.

    Usage::

        env = EDFIsaacEnv(config_path="simulation/isaac/configs/isaac_env_single.yaml")
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
        env.close()

    For vectorized use (num_envs > 1), returns batched numpy arrays with
    leading dimension num_envs.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config_path: str | Path = "simulation/isaac/configs/isaac_env_single.yaml",
        render_mode: str | None = None,
        seed: int | None = None,
        *,
        disable_wind: bool = False,
        disable_gyro: bool = False,
        disable_anti_torque: bool = False,
        disable_gravity: bool = False,
    ) -> None:
        super().__init__()

        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = REPO_ROOT / config_path

        self._cfg_raw = load_config(config_path)
        self._num_envs = int(self._cfg_raw.get("num_envs", 1))
        self._render_mode = render_mode
        self._seed = seed
        self._runtime_disable_wind = bool(disable_wind)
        self._runtime_disable_gyro = bool(disable_gyro)
        self._runtime_disable_anti_torque = bool(disable_anti_torque)
        self._runtime_disable_gravity = bool(disable_gravity)

        # VRAM guard (T031)
        self._check_vram()

        # Build IsaacLab task config
        from simulation.isaac.tasks.edf_landing_task import EdfLandingTask, EdfLandingTaskCfg

        task_cfg = EdfLandingTaskCfg()
        task_cfg.scene.num_envs = self._num_envs
        task_cfg.scene.env_spacing = float(self._cfg_raw.get("env_spacing", 4.0))

        # Apply YAML overrides
        spawn_alt  = self._cfg_raw.get("spawn_altitude_range", [5.0, 10.0])
        spawn_vel  = self._cfg_raw.get("spawn_velocity_magnitude_range", [0.0, 5.0])
        task_cfg.spawn_altitude_min = float(spawn_alt[0])
        task_cfg.spawn_altitude_max = float(spawn_alt[1])
        task_cfg.spawn_vel_mag_min  = float(spawn_vel[0])
        task_cfg.spawn_vel_mag_max  = float(spawn_vel[1])
        task_cfg.episode_length_s   = float(
            self._cfg_raw.get("episode_length_steps", 600)
        ) * task_cfg.sim.dt
        target_pos = self._cfg_raw.get("target_position", None)
        if target_pos is not None:
            task_cfg.target_pos_world = tuple(float(v) for v in target_pos)

        vehicle_path = self._cfg_raw.get("vehicle_config_path", None)
        if vehicle_path:
            p = Path(vehicle_path)
            task_cfg.vehicle_config_path = str(REPO_ROOT / p if not p.is_absolute() else p)

        reward_path = self._cfg_raw.get("reward_config_path", None)
        if reward_path:
            p = Path(reward_path)
            task_cfg.reward_config_path = str(REPO_ROOT / p if not p.is_absolute() else p)

        # T024: Pass environment config path for wind model instantiation
        env_path = self._cfg_raw.get("environment_config_path", None)
        if env_path:
            p = Path(env_path)
            task_cfg.environment_config_path = str(REPO_ROOT / p if not p.is_absolute() else p)

        usd_path = self._cfg_raw.get("drone_usd_path", None)
        if usd_path:
            from simulation.isaac.tasks.edf_landing_task import EdfSceneCfg
            import isaaclab.sim as sim_utils
            p = Path(usd_path)
            abs_usd = REPO_ROOT / p if not p.is_absolute() else p
            task_cfg.scene.robot.spawn.usd_path = str(abs_usd)

        # Instantiate IsaacLab task
        self._task = EdfLandingTask(cfg=task_cfg, render_mode=render_mode)

        # Apply runtime overrides (PID tuning / evaluation scripts).
        self._task.set_runtime_overrides(
            disable_wind=self._runtime_disable_wind,
            disable_gyro=self._runtime_disable_gyro,
            disable_anti_torque=self._runtime_disable_anti_torque,
            disable_gravity=self._runtime_disable_gravity,
        )

        # Gymnasium spaces
        obs_dim = OBS_DIM
        act_dim = ACTION_DIM
        if self._num_envs == 1:
            obs_shape = (obs_dim,)
            act_shape = (act_dim,)
        else:
            obs_shape = (self._num_envs, obs_dim)
            act_shape = (self._num_envs, act_dim)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=act_shape, dtype=np.float32
        )

        self._done = np.zeros(self._num_envs, dtype=bool)

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._seed = seed
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        obs_dict, _ = self._task.reset()
        obs = self._to_numpy(obs_dict["policy"])

        if self._num_envs == 1:
            obs = obs.squeeze(0)

        info = {
            "episode_step": 0,
            "env_ids": list(range(self._num_envs)),
        }
        self._done[:] = False
        return obs, info

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | float, np.ndarray | bool, np.ndarray | bool, dict]:
        # Convert action to torch tensor -- always (num_envs, action_dim)
        action_t = torch.from_numpy(action).float().to(self._task.device)
        if action_t.ndim == 1:
            action_t = action_t.unsqueeze(0).expand(self._num_envs, -1)

        obs_dict, reward_t, terminated_t, truncated_t, info_dict = self._task.step(action_t)

        obs  = self._to_numpy(obs_dict["policy"])
        rew  = reward_t.cpu().numpy().astype(np.float32)
        term = terminated_t.cpu().numpy()
        trunc = truncated_t.cpu().numpy()

        done = term | trunc
        self._done = done

        # Build info
        h_agl = obs[:, OBS_H_AGL] if obs.ndim == 2 else obs[OBS_H_AGL:OBS_H_AGL + 1]
        info = {
            "episode_step": self._task._episode_step.cpu().numpy(),
            "h_agl": h_agl,
            "is_success": self._task._last_landed.cpu().numpy(),
            "landed": self._task._last_landed.cpu().numpy(),
            "crashed": self._task._last_crashed.cpu().numpy(),
            "out_of_bounds": self._task._last_out_of_bounds.cpu().numpy(),
            "ground_hit": self._task._last_ground_hit.cpu().numpy(),
            "impact_speed": self._task._last_impact_speed.cpu().numpy(),
            "lateral_dist": self._task._last_lateral_dist.cpu().numpy(),
        }

        if self._num_envs == 1:
            obs   = obs.squeeze(0)
            rew   = float(rew[0])
            term  = bool(term[0])
            trunc = bool(trunc[0])
            info = {
                "episode_step": int(np.asarray(info["episode_step"]).reshape(-1)[0]),
                "h_agl": float(np.asarray(info["h_agl"]).reshape(-1)[0]),
                "is_success": bool(np.asarray(info["is_success"]).reshape(-1)[0]),
                "landed": bool(np.asarray(info["landed"]).reshape(-1)[0]),
                "crashed": bool(np.asarray(info["crashed"]).reshape(-1)[0]),
                "out_of_bounds": bool(np.asarray(info["out_of_bounds"]).reshape(-1)[0]),
                "ground_hit": bool(np.asarray(info["ground_hit"]).reshape(-1)[0]),
                "impact_speed": float(np.asarray(info["impact_speed"]).reshape(-1)[0]),
                "lateral_dist": float(np.asarray(info["lateral_dist"]).reshape(-1)[0]),
            }

        return obs, rew, term, trunc, info

    # ------------------------------------------------------------------
    # close
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._task.close()

    def set_reset_perturbation(
        self,
        *,
        altitude_offset_m: float = 0.0,
        roll_offset_rad: float = 0.0,
        pitch_offset_rad: float = 0.0,
        ang_vel_frd: tuple[float, float, float] | list[float] | None = None,
    ) -> None:
        """Forward deterministic hover perturbations to the underlying task."""
        self._task.set_reset_perturbation(
            altitude_offset_m=altitude_offset_m,
            roll_offset_rad=roll_offset_rad,
            pitch_offset_rad=pitch_offset_rad,
            ang_vel_frd=ang_vel_frd,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _to_numpy(self, t: torch.Tensor) -> np.ndarray:
        return t.cpu().detach().numpy().astype(np.float32)

    def _check_vram(self) -> None:
        """Emit advisory if num_envs may exceed RTX 5070 VRAM budget (T031)."""
        if self._num_envs <= _VRAM_WARN_ENVS:
            return
        try:
            import torch
            if not torch.cuda.is_available():
                return
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if vram_gb < _VRAM_WARN_GB:
                logger.warning(
                    "[EDFIsaacEnv] num_envs=%d may exceed GPU VRAM budget "
                    "(detected %.1f GB < %.1f GB threshold). "
                    "Consider using isaac_env_training.yaml (num_envs=256).",
                    self._num_envs, vram_gb, _VRAM_WARN_GB,
                )
        except Exception:
            pass

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> str:
        return str(self._task.device)
