from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a Python dict.

    Notes:
    - Uses `yaml.safe_load`.
    - Returns `{}` if the YAML file is empty.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}

