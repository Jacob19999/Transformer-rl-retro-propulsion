"""
Task name-to-config resolver with deep-merge config loading.

Resolves task name string ("hover"/"landing") to task YAML path,
loads and merges configs in priority order:
  base → env → task → disturbance → CLI overrides
"""

from __future__ import annotations
import yaml
import copy
from pathlib import Path
from typing import Any


# Default task name → YAML path mapping (relative to simulation/isaac/)
_TASK_YAML_MAP = {
    "hover": "configs/tasks/hover.yaml",
    "landing": "configs/tasks/landing.yaml",
}


def resolve_task_config(
    task_name: str,
    sim_root: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve task name to task config dict.

    Args:
        task_name: Task identifier ("hover" or "landing").
        sim_root: Path to simulation/isaac/ directory. Auto-detected if None.

    Returns:
        Parsed task config dict.

    Raises:
        KeyError: If task_name is not registered.
        FileNotFoundError: If task YAML file does not exist.
    """
    if task_name not in _TASK_YAML_MAP:
        available = sorted(_TASK_YAML_MAP.keys())
        raise KeyError(f"Task '{task_name}' not registered. Available: {available}")

    if sim_root is None:
        # Auto-detect: assume this file is in tvc_env/envs/
        sim_root = Path(__file__).parents[2]

    yaml_path = Path(sim_root) / _TASK_YAML_MAP[task_name]
    if not yaml_path.exists():
        raise FileNotFoundError(f"Task config not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, overlay: dict) -> dict:
    """Deep merge overlay dict into base dict (overlay values win on conflict).

    Args:
        base: Base config dict.
        overlay: Overlay dict whose values override base.

    Returns:
        New merged dict (base unchanged).
    """
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_merged_config(
    task_name: str,
    env_config: dict | None = None,
    disturbance_config: dict | None = None,
    overrides: dict | None = None,
    sim_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load and merge configs in priority order.

    Priority (highest last): task → env → disturbance → overrides

    Args:
        task_name: Task name string.
        env_config: Parsed env YAML dict (optional).
        disturbance_config: Parsed disturbance YAML dict (optional).
        overrides: Dict of CLI overrides (highest priority).
        sim_root: Path to simulation/isaac/ directory.

    Returns:
        Merged config dict.
    """
    config = resolve_task_config(task_name, sim_root)

    if env_config:
        config = deep_merge(config, env_config)
    if disturbance_config:
        config = deep_merge(config, disturbance_config)
    if overrides:
        config = deep_merge(config, overrides)

    return config


def register_task(name: str, yaml_path: str) -> None:
    """Register a new task name → YAML path mapping."""
    _TASK_YAML_MAP[name] = yaml_path


def list_tasks() -> list[str]:
    """Return sorted list of registered task names."""
    return sorted(_TASK_YAML_MAP.keys())
