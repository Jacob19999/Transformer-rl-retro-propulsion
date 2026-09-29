"""Time the convex flight stack on the computer it will fly on (Raspberry Pi 5, Jetson Orin Nano, ...).

Runs the offline closed-loop replica (tests/unit/test_convex_guidance.py: momentum vanes, servo and
joint lag, rotor gyro) and a 4-waypoint route plan, and reports:

* the tracking + attitude step (per 30 Hz control period, re-plan steps excluded),
* landing re-plan and route plan solve times,
* a guidance.plan_latency_s to validate in Isaac before flying async (route p95 x margin, rounded
  up to a control period).

Needs numpy, scipy, clarabel, torch, pyyaml and pytest (the replica lives in the unit tests).

    python tools/bench_flight_computer.py --workers 2 --repeats 5 --json bench.json
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests/unit'))


def _stats(values_s):
    import numpy as np
    x = np.asarray(values_s, dtype=float) * 1e3
    return dict(n=int(x.size), median_ms=round(float(np.median(x)), 2), p95_ms=round(float(np.percentile(x, 95)), 2),
                max_ms=round(float(x.max()), 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=None, help='guidance.solver_workers (default: the YAML value)')
    parser.add_argument('--repeats', type=int, default=3, help='closed-loop landings and route plans to time')
    parser.add_argument('--margin', type=float, default=1.25, help='plan_latency_s = route p95 x margin')
    parser.add_argument('--json', type=Path, default=None, help='also write the report here')
    args = parser.parse_args()

    import test_convex_guidance as T
    from tvc_env.controllers.convex_adapter import ConvexGuidanceController
    from tvc_env.controllers.convex_guidance import ObjectiveWeights, RouteWaypoint

    settings = T.settings()
    if args.workers is not None:
        settings['guidance']['solver_workers'] = args.workers
    workers = int(settings['guidance'].get('solver_workers', 4))
    control_dt = 1 / 30

    tracking, replans, touchdowns = [], [], []
    for _ in range(args.repeats):
        controller = ConvexGuidanceController(settings, T._with_vanes(T.MOMENTUM_VANES), (0., 0., 0.), .3125,
                                              control_dt, servo_deadband_rad=.017)
        solve, compute = controller.guidance.plan, controller.compute_action
        solving = []

        def timed_plan(*a, **k):
            started = time.perf_counter()
            plan = solve(*a, **k)
            solving.append(time.perf_counter() - started)
            return plan

        def timed_step(*a, **k):
            count, started = len(solving), time.perf_counter()
            action = compute(*a, **k)
            if len(solving) == count:                     # a step without a solve: the tracking loop alone
                tracking.append(time.perf_counter() - started)
            return action
        controller.guidance.plan, controller.compute_action = timed_plan, timed_step
        touchdown, _ = T._land_on_vanes(controller, T.MOMENTUM_VANES, damping=0.)
        controller.close()
        replans += solving[1:]                            # solving[0] is the pad plan
        touchdowns.append(touchdown)

    start = [0., 0., 3.]
    route = (RouteWaypoint((8., 6., 6.), 'flypass', 1., 3.), RouteWaypoint((16., -6., 8.), 'flypass', 1., 3.),
             RouteWaypoint((24., 6., 6.), 'flypass', 1., 3.), RouteWaypoint((24., 6., 4.), 'hover', .5, 2., 2.))
    routes, solves = [], 0
    for _ in range(args.repeats):
        guidance = T._tracking_planner(ObjectiveWeights(path=10., smoothness=1.), flypass_min_speed_fraction=.5,
                                       flypass_heading_tolerance_rad=math.radians(20), workers=workers)
        started = time.perf_counter()
        plan = guidance.plan(start, [0., 0., 0.], T.WEIGHT, T.GATE, route, path=T._drawn(start, route))
        routes.append(time.perf_counter() - started)
        solves = plan.solves
        guidance.close()

    route_stats = _stats(routes)
    latency = math.ceil(route_stats['p95_ms'] / 1e3 * args.margin / control_dt) * control_dt
    report = dict(
        machine=dict(platform=platform.platform(), processor=platform.processor() or platform.machine(),
                     python=platform.python_version()),
        solver_workers=workers,
        tracking_step=_stats(tracking), tracking_share_of_period=round(_stats(tracking)['p95_ms'] / 1e3 / control_dt, 3),
        landing_replan=_stats(replans), route_plan=dict(**route_stats, socps=solves),
        replica_landings=[None if t is None else {k: round(v, 3) for k, v in t.items()} for t in touchdowns],
        suggested_plan_latency_s=round(latency, 3))
    print(json.dumps(report, indent=2))
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
