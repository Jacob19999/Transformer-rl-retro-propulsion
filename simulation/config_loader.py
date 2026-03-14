from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge mappings while letting override values win."""
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a Python dict.

    Supports an optional `extends` key that points to one or more base YAML
    files. Relative paths are resolved from the current config file directory,
    and child values override base values using recursive deep-merge semantics.
    Returns `{}` if the YAML file is empty.
    """
    p = Path(path).resolve()
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    extends = data.pop("extends", None)
    if extends is None:
        return data

    base_paths = extends if isinstance(extends, list) else [extends]
    merged: dict[str, Any] = {}
    for base_path in base_paths:
        resolved_base = Path(base_path)
        if not resolved_base.is_absolute():
            resolved_base = p.parent / resolved_base
        merged = _deep_merge(merged, load_config(resolved_base))
    return _deep_merge(merged, data)

