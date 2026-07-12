"""Consistent Isaac Sim application startup for command-line drivers."""

from __future__ import annotations

import os
from pathlib import Path


def launch_simulation_app(headless: bool, **overrides):
    """Launch Isaac Sim with the minimal matching Isaac Lab experience."""
    from isaacsim import SimulationApp

    experience_name = "isaaclab.python.headless.kit" if headless else "isaaclab.python.kit"
    experience = None
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "IsaacLab" / "apps" / experience_name
        if candidate.exists():
            experience = candidate
            break

    launch_config = {
        "headless": headless,
        "multi_gpu": False,
        "disable_viewport_updates": headless,
        **overrides,
    }
    return SimulationApp(
        launch_config,
        experience=str(experience) if experience is not None else "",
    )


def close_simulation_app(simulation_app) -> bool:
    """Close Kit when requested; default to process-level fast shutdown on Windows."""
    if simulation_app is None:
        return False
    if os.getenv("TVC_ISAAC_FAST_CLOSE", "1") == "1":
        return False
    simulation_app.close()
    return True
