"""Consistent Isaac Sim application startup for command-line drivers."""

from __future__ import annotations

import os


def launch_simulation_app(headless: bool, **overrides):
    """Launch Isaac Sim through Isaac Lab's supported application bootstrap.

    AppLauncher applies the extension-order and renderer patches required before
    SimulationApp starts.  Raw SimulationApp construction can segfault in the
    headed UI preparation path on Windows/Isaac Sim 5.1.
    """
    from isaaclab.app import AppLauncher

    experience_name = "isaaclab.python.headless.kit" if headless else "isaaclab.python.kit"
    app_launcher = AppLauncher(
        headless=headless,
        experience=experience_name,
        multi_gpu=False,
        **overrides,
    )
    # Retain the launcher for the lifetime of its SimulationApp and its event
    # subscriptions; callers intentionally operate on the familiar app object.
    app_launcher.app._tvc_app_launcher = app_launcher
    return app_launcher.app


def close_simulation_app(simulation_app) -> bool:
    """Close Kit when requested; default to process-level fast shutdown on Windows."""
    if simulation_app is None:
        return False
    if os.getenv("TVC_ISAAC_FAST_CLOSE", "1") == "1":
        return False
    simulation_app.close()
    return True
