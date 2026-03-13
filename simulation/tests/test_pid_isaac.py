"""Isaac Sim integration tests for PID tuning/evaluation scaffolding.

Marked with pytest \"isaac\" marker so they can be skipped when IsaacLab
is not available. These are smoke tests for the new tune_pid_isaac and
test_pid_isaac entry points and for EDFIsaacEnv wiring.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


pytestmark = pytest.mark.isaac


def _run_module(module: str, *args: str) -> int:
    cmd = ["python", "-m", module, *args]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return result.returncode


def test_tune_pid_isaac_scaffold_runs_help() -> None:
    code = _run_module("simulation.isaac.scripts.tune_pid_isaac", "--help")
    assert code == 0


def test_test_pid_isaac_scaffold_runs_help() -> None:
    code = _run_module("simulation.isaac.scripts.test_pid_isaac", "--help")
    assert code == 0


def test_edf_isaac_env_runtime_overrides_accept_flags() -> None:
    # Smoke test that EDFIsaacEnv accepts runtime override kwargs and
    # does not crash on construction. We do not step the env here to
    # avoid requiring a full Isaac runtime in unit tests.
    from simulation.isaac.envs.edf_isaac_env import EDFIsaacEnv

    env = EDFIsaacEnv(
        config_path="simulation/isaac/configs/isaac_env_single.yaml",
        seed=0,
        disable_wind=True,
        disable_gyro=True,
        disable_anti_torque=True,
        disable_gravity=True,
    )
    assert env.num_envs >= 1
    env.close()


