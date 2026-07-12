"""
Per-step telemetry logger for TVC environment.

Logs observation vector, action vector, reward, torque contributions, contact
state, and episode metrics per step to a structured CSV format. Designed for
lightweight, real-time logging during evaluation runs.

Output file: <output_dir>/telemetry_<episode_id>.csv
"""

from __future__ import annotations
import csv
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


# Column layout for the CSV log
_COLUMNS = [
    # Step metadata
    "episode_id", "step", "wall_time",
    # Observation (24 dims)
    "pos_err_x", "pos_err_y", "pos_err_z",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "lin_vel_x", "lin_vel_y", "lin_vel_z",
    "ang_vel_x", "ang_vel_y", "ang_vel_z",
    "height",
    "fin0", "fin1", "fin2", "fin3",
    "fin_rate0", "fin_rate1", "fin_rate2", "fin_rate3",
    "rpm_norm",
    "contact_state",
    # Action (5 dims)
    "act_fin0", "act_fin1", "act_fin2", "act_fin3", "throttle",
    # Scalar episode signals
    "reward", "terminated", "truncated",
    # Optional torque contributions (logged if provided)
    "torque_fin_x", "torque_fin_y", "torque_fin_z",
    "torque_static_x", "torque_static_y", "torque_static_z",
    "torque_gyro_x", "torque_gyro_y", "torque_gyro_z",
    "torque_spool_x", "torque_spool_y", "torque_spool_z",
    "force_wind_x", "force_wind_y", "force_wind_z",
]


class TelemetryLogger:
    """Per-step CSV logger for a single environment (env_idx=0)."""

    def __init__(
        self,
        output_dir: str | Path,
        env_idx: int = 0,
        flush_every: int = 100,
    ):
        """
        Args:
            output_dir:  Directory to write CSV files.
            env_idx:     Which environment index to log (default 0).
            flush_every: Flush to disk every N steps.
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._env_idx = env_idx
        self._flush_every = flush_every

        self._episode_id = 0
        self._step = 0
        self._file = None
        self._writer = None
        self._t0 = time.time()

        self._open_new_episode()

    def _open_new_episode(self) -> None:
        """Open a new CSV file for the current episode."""
        if self._file is not None:
            self._file.close()

        path = self._output_dir / f"telemetry_ep{self._episode_id:04d}.csv"
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=_COLUMNS)
        self._writer.writeheader()
        self._step = 0

    def log_step(
        self,
        obs: Tensor,            # (num_envs, 24)
        action: Tensor,         # (num_envs, 5)
        reward: Tensor,         # (num_envs,)
        terminated: Tensor,     # (num_envs,) bool
        truncated: Tensor,      # (num_envs,) bool
        torques: dict[str, Tensor] | None = None,  # optional torque contributions
    ) -> None:
        """Log one step for env_idx.

        Args:
            obs:        Observation tensor (num_envs, 24).
            action:     Action tensor (num_envs, 5).
            reward:     Reward tensor (num_envs,).
            terminated: Termination flags (num_envs,).
            truncated:  Truncation flags (num_envs,).
            torques:    Optional dict of torque contributions, each (3,) or (num_envs, 3).
                        Expected keys: "fin", "static", "gyro", "spool", "wind".
        """
        i = self._env_idx
        o = obs[i].tolist()
        a = action[i].tolist()

        row: dict[str, Any] = {
            "episode_id": self._episode_id,
            "step": self._step,
            "wall_time": time.time() - self._t0,
            # Observation
            "pos_err_x": o[0], "pos_err_y": o[1], "pos_err_z": o[2],
            "quat_w": o[3], "quat_x": o[4], "quat_y": o[5], "quat_z": o[6],
            "lin_vel_x": o[7], "lin_vel_y": o[8], "lin_vel_z": o[9],
            "ang_vel_x": o[10], "ang_vel_y": o[11], "ang_vel_z": o[12],
            "height": o[13],
            "fin0": o[14], "fin1": o[15], "fin2": o[16], "fin3": o[17],
            "fin_rate0": o[18], "fin_rate1": o[19], "fin_rate2": o[20], "fin_rate3": o[21],
            "rpm_norm": o[22],
            "contact_state": o[23],
            # Action
            "act_fin0": a[0], "act_fin1": a[1], "act_fin2": a[2], "act_fin3": a[3],
            "throttle": a[4],
            # Signals
            "reward": reward[i].item(),
            "terminated": int(terminated[i].item()),
            "truncated": int(truncated[i].item()),
        }

        # Torque contributions (fill with 0 if not provided)
        for key, prefix in [
            ("fin",    "torque_fin"),
            ("static", "torque_static"),
            ("gyro",   "torque_gyro"),
            ("spool",  "torque_spool"),
            ("wind",   "force_wind"),
        ]:
            if torques and key in torques:
                t = torques[key]
                if t.dim() > 1:
                    t = t[i]
                vals = t.tolist()
            else:
                vals = [0.0, 0.0, 0.0]
            row[f"{prefix}_x"] = vals[0]
            row[f"{prefix}_y"] = vals[1]
            row[f"{prefix}_z"] = vals[2]

        self._writer.writerow(row)
        self._step += 1

        if self._step % self._flush_every == 0:
            self._file.flush()

        # Auto-advance episode
        done = terminated[i].item() or truncated[i].item()
        if done:
            self._episode_id += 1
            self._open_new_episode()

    def close(self) -> None:
        """Flush and close the current log file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()
