"""
Test runner for individual Isaac Sim simulation validation tests.

Bootstraps the Isaac Sim runtime, then runs a selected test module from tests/sim/.
Each test in the 13-step validation ladder can be run independently.

Usage:
    python apps/run_single_test.py --test test_00_asset_validation
    python apps/run_single_test.py --test test_01_joint_axes --no-headless
    python apps/run_single_test.py --test test_06_edf_spool_and_reaction --physics configs/physics/solver_high_fidelity.yaml
"""

import argparse
import importlib
import sys
import os
from pathlib import Path

from runner_safety import WallClockWatchdog, force_process_exit


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a single Isaac Sim simulation test from the validation ladder."
    )
    parser.add_argument(
        "--test",
        required=True,
        help="Test module name (without .py) from tests/sim/, e.g. test_00_asset_validation",
    )
    parser.add_argument(
        "--physics",
        default=None,
        help="Path to PhysX solver config YAML (optional override)",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Headless (no viewport) by default; use --no-headless for interactive Kit UI",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        default=False,
        help="Visual inspection mode: renders every step, prints per-fin status, pauses 2s between fins",
    )
    parser.add_argument(
        "--max-wall-time",
        type=float,
        default=300.0,
        help="Maximum wall-clock seconds before forcing process exit.",
    )
    return parser.parse_args()


def bootstrap_isaac_sim(headless: bool = True, physics_config: str | None = None):
    """Initialize Isaac Sim application context."""
    from isaac_launcher import launch_simulation_app
    return launch_simulation_app(headless=headless)


def run_test(test_name: str, physics_config: str | None = None):
    """Locate and run the given test module."""
    # Add the simulation/isaac root to sys.path
    sim_root = Path(__file__).parent.parent
    sys.path.insert(0, str(sim_root))
    tests_sim_dir = sim_root / "tests" / "sim"
    sys.path.insert(0, str(tests_sim_dir))

    # Import the test module
    try:
        test_module = importlib.import_module(test_name)
    except ImportError as e:
        print(f"ERROR: Could not import test module '{test_name}': {e}", file=sys.stderr)
        print(f"Available tests in {tests_sim_dir}:")
        for f in sorted(tests_sim_dir.glob("test_*.py")):
            print(f"  {f.stem}")
        sys.exit(1)

    # Pass physics config to module if it supports it
    if hasattr(test_module, "set_physics_config") and physics_config:
        test_module.set_physics_config(physics_config)

    # Run the module's main test function
    if hasattr(test_module, "run"):
        result = test_module.run()
        return result
    else:
        # Fall back to pytest-style test collection and execution
        import pytest
        exit_code = pytest.main([
            str(tests_sim_dir / f"{test_name}.py"),
            "-v",
            "--tb=long",
        ])
        return exit_code == 0


def main():
    args = parse_args()
    watchdog = WallClockWatchdog(args.max_wall_time, label=f"Isaac Sim validation test {args.test}")
    watchdog.start()
    print(f"=== Isaac Sim Validation Test: {args.test} ===", flush=True)

    simulation_app = None
    try:
        if args.slow:
            os.environ["ISAAC_VIZ_SLOW"] = "1"
            print("[--slow] Visual inspection mode: 200 steps/fin, rendered, 2s pause between fins", flush=True)

        print("Bootstrapping Isaac Sim...", flush=True)
        simulation_app = bootstrap_isaac_sim(
            headless=args.headless,
            physics_config=args.physics,
        )
        print("Isaac Sim ready.", flush=True)

        success = run_test(args.test, physics_config=args.physics)

        if success:
            print(f"\nPASS: {args.test}", flush=True)
            return 0
        else:
            print(f"\nFAIL: {args.test}", flush=True)
            return 1

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        return 2
    finally:
        watchdog.stop()
        if simulation_app is not None:
            print("Closing Isaac Sim...", flush=True)
            from isaac_launcher import close_simulation_app
            closed = close_simulation_app(simulation_app)
            print("Isaac Sim closed." if closed else "Isaac Sim fast shutdown requested.", flush=True)


if __name__ == "__main__":
    force_process_exit(main())
