"""
Episode data export for TVC environment.

Exports episode telemetry to JSON and CSV formats for offline analysis.
Includes metadata: task name, config hash, seed, git hash, and timestamps.

Output structure:
  <output_dir>/
    episode_<id>_metadata.json  — episode metadata + summary statistics
    episode_<id>_steps.csv      — per-step telemetry (same format as logger.py)
"""

from __future__ import annotations
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from tvc_env.telemetry.metrics import EpisodeMetrics


def _get_git_hash() -> str:
    """Return current git commit hash (short), or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[4],
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _hash_config_file(path: str | Path | None) -> str:
    """Compute MD5 hash of a config file for traceability."""
    if path is None:
        return "none"
    try:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()[:8]
    except Exception:
        return "error"


def export_episode(
    episode_id: int,
    steps: list[dict[str, Any]],
    metrics: EpisodeMetrics,
    output_dir: str | Path,
    task: str = "hover",
    env_config_path: str | Path | None = None,
    disturbance_config_path: str | Path | None = None,
    seed: int = 0,
) -> dict[str, Path]:
    """Export a complete episode to JSON metadata + CSV steps.

    Args:
        episode_id:              Episode index.
        steps:                   List of per-step dicts (from TelemetryLogger.log_step).
        metrics:                 EpisodeMetrics summary for this episode.
        output_dir:              Directory to write output files.
        task:                    Task name (hover/landing).
        env_config_path:         Path to env config YAML (for traceability hash).
        disturbance_config_path: Path to disturbance config YAML.
        seed:                    Random seed used.

    Returns:
        Dict with "metadata" and "steps" keys pointing to output file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()

    metadata = {
        "episode_id": episode_id,
        "task": task,
        "seed": seed,
        "timestamp": timestamp,
        "git_hash": _get_git_hash(),
        "env_config_hash": _hash_config_file(env_config_path),
        "disturbance_config_hash": _hash_config_file(disturbance_config_path),
        "env_config_path": str(env_config_path) if env_config_path else None,
        "disturbance_config_path": str(disturbance_config_path) if disturbance_config_path else None,
        "metrics": metrics.to_dict(),
    }

    # Write metadata JSON
    meta_path = output_dir / f"episode_{episode_id:04d}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Write steps CSV
    steps_path = output_dir / f"episode_{episode_id:04d}_steps.csv"
    if steps:
        fieldnames = list(steps[0].keys())
        with open(steps_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(steps)

    return {"metadata": meta_path, "steps": steps_path}


def load_episode_metadata(metadata_path: str | Path) -> dict[str, Any]:
    """Load episode metadata JSON.

    Args:
        metadata_path: Path to episode_<id>_metadata.json.

    Returns:
        Dict with episode metadata and metrics.
    """
    with open(metadata_path) as f:
        return json.load(f)


def export_run_summary(
    output_dir: str | Path,
    all_metrics: list[EpisodeMetrics],
    run_config: dict[str, Any] | None = None,
) -> Path:
    """Export a summary JSON for the entire run (all episodes).

    Args:
        output_dir:   Directory to write summary.
        all_metrics:  List of EpisodeMetrics for every episode in the run.
        run_config:   Optional run-level configuration dict.

    Returns:
        Path to the summary JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outcomes = [m.outcome for m in all_metrics]
    n = len(all_metrics)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "git_hash": _get_git_hash(),
        "run_config": run_config or {},
        "n_episodes": n,
        "outcomes": {
            "success": outcomes.count("success"),
            "crash": outcomes.count("crash"),
            "timeout": outcomes.count("timeout"),
            "unknown": outcomes.count("unknown"),
        },
        "aggregate": {
            "mean_pos_error": sum(m.mean_pos_error for m in all_metrics) / n if n else 0,
            "max_pos_error": max((m.max_pos_error for m in all_metrics), default=0),
            "mean_tilt_deg": sum(
                __import__("math").degrees(m.mean_tilt) for m in all_metrics
            ) / n if n else 0,
            "mean_total_reward": sum(m.total_reward for m in all_metrics) / n if n else 0,
            "mean_episode_length": sum(m.episode_length for m in all_metrics) / n if n else 0,
        },
    }

    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary_path
