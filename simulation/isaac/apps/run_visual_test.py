"""
Runner for scripted visual-validation scenarios.

Usage examples:
    python apps/run_visual_test.py --scenario fin_sweep
    python apps/run_visual_test.py --scenario edf_spool_gyro --episode-steps 100
    python apps/run_visual_test.py --scenario all --no-headless
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path


SCENARIO_MODULES = {
    "fin_sweep": "test_viz_01_fin_sweep",
    "edf_spool_gyro": "test_viz_02_edf_spool_and_gyro",
    "wind_drop": "test_viz_03_wind_drop",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run scripted Isaac Sim visual-validation scenarios.")
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["all", *SCENARIO_MODULES.keys()],
        help="Which visual scenario to run.",
    )
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=100,
        help="Number of steps per visual episode.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=0,
        help="Number of times to repeat the episode. 0 = loop forever (default).",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Visual runner defaults to viewport-on mode; use --headless for terminal-only playback.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print state vectors every N steps in the terminal.",
    )
    parser.add_argument(
        "--pause-between",
        type=float,
        default=1.0,
        help="Pause between scenarios when running --scenario all.",
    )
    return parser.parse_args()


def bootstrap_isaac_sim(headless: bool = False):
    """Initialize Isaac Sim application context."""
    from isaac_launcher import launch_simulation_app

    return launch_simulation_app(headless=headless, renderer="RaytracedLighting")


def main():
    args = parse_args()
    sim_root = Path(__file__).parent.parent
    visual_dir = sim_root / "tests" / "visual"
    sys.path.insert(0, str(sim_root))
    sys.path.insert(0, str(visual_dir))

    scenario_names = list(SCENARIO_MODULES.keys()) if args.scenario == "all" else [args.scenario]

    simulation_app = None
    try:
        print("=== Isaac Sim Visual Validation Runner ===")
        print("Bootstrapping Isaac Sim...")
        simulation_app = bootstrap_isaac_sim(headless=args.headless)
        print("Isaac Sim ready.")

        all_ok = True
        for idx, scenario_name in enumerate(scenario_names):
            module = importlib.import_module(SCENARIO_MODULES[scenario_name])
            ok = bool(
                module.run(
                    simulation_app=simulation_app,
                    episode_steps=args.episode_steps,
                    print_every=args.print_every,
                    sim_root=sim_root,
                    num_episodes=args.num_episodes,
                )
            )
            all_ok = all_ok and ok
            if idx < len(scenario_names) - 1 and args.pause_between > 0.0:
                time.sleep(args.pause_between)

        if not all_ok:
            sys.exit(1)
    finally:
        if simulation_app is not None:
            from isaac_launcher import close_simulation_app
            close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
