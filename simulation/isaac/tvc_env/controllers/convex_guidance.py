"""Convex powered-descent guidance for the EDF vehicle (SOCP, lossless convexification).

Reference: B. Açıkmeşe and S. R. Ploen, "Convex Programming Approach to
Powered Descent Guidance for Mars Landing", JGCD 30(5), 2007
(Paper/Reference/Convex Programming.pdf).

The vehicle is planned as a point mass with thrust acceleration u = T/m:
r'' = u + g. The non-convex thrust annulus rho1 <= ||T|| <= rho2 (the vanes
lose authority without jet flow, so rho1 > 0) is replaced by the slack sigma:

    ||u_k|| <= sigma_k,   rho1/m <= sigma_k <= rho2/m.

The paper's Lemma 1 shows the relaxation is lossless: at the optimum of any
cost that increases with sigma, ||u_k|| = sigma_k. `GuidancePlan.convexification_gap`
reports max (sigma - ||u||)/sigma so every solve checks this numerically.
The battery vehicle has constant mass, so the paper's z = ln m change of
variables is not needed and the problem is an exact SOCP.

The thrust is discretized with a first-order hold (linear between nodes, the
dynamics integrated exactly), where the paper's Problem 4 uses a zero-order
hold: a piecewise-constant plan steps the thrust direction at every node,
which the rate-limited rotor and the attitude loop cannot follow.

Constraints (all convex, imposed at the nodes; the convex ones then hold
between them too):
  * thrust pointing: u_z >= sigma cos(max_tilt), i.e. tilt <= max_tilt (Sec. VI);
  * glide slope: ||r_xy - pad_xy|| <= tan(theta) (r_z - touchdown_z) (eq. 11),
    widened for a start outside it and narrowed back before the gate;
  * speed: ||v_k|| <= V (eq. 10), braking back under it from a faster start;
  * thrust rate: ||u_{k+1} - u_k|| <= sigma_dot dt, starting at the
    rotor's present thrust vector. Not in the paper: a rotor speed change yaws
    this body (rotor/body angular-momentum exchange) and the bound keeps that
    torque inside the vanes' yaw authority. Bounding the vector (not the
    slack) keeps the executed thrust rate-limited and also caps the
    thrust-direction slew the attitude loop must follow;
  * upright arrival: the last thrust vector is vertical (eq. 12 / 37);
  * sink-rate envelope (optional): on the landing leg -v_z <= sqrt(v_gate^2 +
    2 a (z - z_gate)), a a share of the planned braking acceleration, so the
    descent never outruns the braking it could still fly (_sink_envelope);
  * waypoints: a fly-through node lies inside a capture ball and crosses it
    at the leg speed along the route tangent (or, when configured, inside a
    cone around the tangent with a bounded along-route speed); a hover node
    is at rest on the point and stays there for the remaining hold time;
  * leg speed: every node of a waypoint leg (and of the landing leg, when an
    approach speed is set) stays under that leg's speed, after braking an
    entry faster than it at the planner's braking acceleration;
  * route corridor (optional, soft): each node stays within its leg's
    half-width of the drawn route, the mission sequencer's Catmull-Rom curve,
    measured from a chord of it (the distance to a segment is convex: an SOC
    per node). Any excess is a penalized slack, so a start the corridor
    cannot contain still has a plan.

A plan that follows a corridor may use the full thrust (the soft-terminal
ceiling) while it brakes an over-speed start back under the speed bound.
Planned at the reserved ceiling, a 20 m/s descent braked at 2 m/s^2 and sank
26 m below its waypoint (Isaac mission 835c3de32185). Without a corridor the
energy objective brakes no harder than the floor forces, so the reserve stays.

Objective: 'energy' minimizes electrical energy with the simulator's
momentum-theory power law P = P_ref (T/T_ref)^1.5 (a power cone per node,
trapezoid rule); 'delta_v' minimizes the integral of ||T||/m, the paper's fuel analog and the
project's propulsive delta-v metric. The corridor penalty is priced in
seconds of hover cost per metre-second outside it.

Secondary objectives (`ObjectiveWeights`, all zero by default) are priced in
the same unit, seconds of hover cost, so each weight reads as "how many
seconds of hovering one unit of this is worth" whatever the primary cost:
  * path: cross-track distance from the drawn route, per m s (an SOC per
    node). Inside the corridor the primary cost alone is indifferent to where
    the plan runs, so energy plans ride the corridor edge; this term pulls
    them onto the centreline and keeps working where the corridor is wide;
  * time: plan duration, per s. Also searches shorter waypoint-leg timings;
  * smoothness: squared thrust-vector jerk, normalized so one second at the
    planned attitude-slew bound costs one unit. Smooth plans leave vane
    authority for feedback, which is what the tracking loop follows best;
  * tilt: squared horizontal thrust acceleration, normalized so one second
    at the planned tilt limit costs one unit (keeps attitude margin for
    gust rejection).
The quadratic terms keep the problem a convex QP-SOCP (Clarabel's P matrix).
They do not involve sigma, so the Lemma 1 argument still holds and every
plan still reports its convexification gap.

The final time is free, so the landing
leg duration is found by a line search (the paper's Algorithm 1; the cost is
unimodal in t_f, Remark 10). If no duration is feasible, a soft-terminal
problem returns the closest reachable plan instead of no plan.

Solved with Clarabel, a primal-dual interior-point conic solver with
deterministic convergence properties, as the paper recommends for onboard use.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import time

import numpy as np
from scipy import sparse

GRAVITY = 9.81
FLYPASS, HOVER = 'flypass', 'hover'
TAKEOFF, DESCENT = 'takeoff', 'descent'
STOP_KINDS = (HOVER, TAKEOFF, DESCENT)
_OK = ('Solved', 'AlmostSolved')
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
CORRIDOR_WINDOW = 0.1   # refined corridor chords span +-10% of the leg around each node


def _clarabel():
    try:
        import clarabel
    except ImportError as exc:  # pragma: no cover - exercised only without the package
        raise ImportError('Convex guidance needs the Clarabel conic solver: install '
                          'simulation/isaac/mission_control/requirements.txt') from exc
    return clarabel


def _trapezoid_weights(dts: np.ndarray) -> np.ndarray:
    """Node weights of the trapezoid rule over intervals dts (N intervals, N+1 nodes)."""
    weights = np.zeros(len(dts) + 1)
    weights[:-1] += 0.5 * dts
    weights[1:] += 0.5 * dts
    return weights


@dataclass(frozen=True)
class GuidanceLimits:
    """Physical bounds for one plan (SI units, world frame, Z up)."""

    mass_kg: float
    thrust_min_n: float        # rho1
    thrust_max_n: float        # rho2
    thrust_rate_n_s: float     # |dT/dt|
    max_tilt_rad: float
    max_speed_m_s: float
    glide_slope_rad: float | None = None  # cone half-angle from vertical; None disables
    glide_slope_final_s: float = 3.0      # a widened cone is back to nominal this long before the gate
    tilt_rate_rad_s: float | None = None  # thrust-direction slew the attitude actuators can drive; None: unbounded
    route_floor_m: float = 0.8            # minimum body-origin altitude on waypoint legs
    gravity: float = GRAVITY
    # Soft-terminal (no safe plan) problems may use the full thrust and the
    # faster spool slew: avoiding ground impact outranks the yaw transient.
    # Optimal plans may use the full thrust (at the normal slew) while they
    # brake an over-speed start back under max_speed_m_s.
    emergency_thrust_max_n: float | None = None
    emergency_thrust_rate_n_s: float | None = None
    # Landing-leg sink-rate envelope: the powered descent never sinks faster
    # than it could brake to the gate speed using this share of the planned
    # braking acceleration (ceiling - weight) / m. None disables it.
    landing_sink_brake_fraction: float | None = None

    def __post_init__(self):
        if not 0.0 < self.thrust_min_n < self.mass_kg * self.gravity < self.thrust_max_n:
            raise ValueError('Guidance needs thrust_min < weight < thrust_max, got '
                             f'{self.thrust_min_n:.2f} / {self.mass_kg * self.gravity:.2f} / '
                             f'{self.thrust_max_n:.2f} N')
        if self.thrust_rate_n_s <= 0.0 or not 0.0 < self.max_tilt_rad < math.pi / 2:
            raise ValueError('Invalid thrust rate or tilt limit')


@dataclass(frozen=True)
class EnergyModel:
    """Electrical power P(T) = power_ref (T / thrust_ref)^1.5 + auxiliary."""

    thrust_ref_n: float
    power_ref_w: float
    auxiliary_w: float = 0.0


@dataclass(frozen=True)
class ObjectiveWeights:
    """Secondary costs added to the primary objective, in seconds of hover cost.

    path: per metre-second of cross-track distance from the drawn route
    (legs with a route curve only). time: per second of plan duration.
    smoothness: per second spent at the attitude-slew bound (quadratic in
    thrust jerk). tilt: per second spent at the planned tilt limit
    (quadratic in horizontal thrust acceleration).
    """

    path: float = 0.0
    time: float = 0.0
    smoothness: float = 0.0
    tilt: float = 0.0

    def __post_init__(self):
        for name in ('path', 'time', 'smoothness', 'tilt'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f'Objective weight {name} must be finite and non-negative')


@dataclass(frozen=True)
class RouteWaypoint:
    position: tuple[float, float, float]
    kind: str = FLYPASS
    radius_m: float = 1.0
    speed_m_s: float = 3.0
    hold_s: float = 0.0      # remaining hold time for a hover waypoint
    corridor_m: float | None = None   # corridor half-width of the leg to this waypoint (None: planner default)


@dataclass(frozen=True)
class LandingGate:
    """End of powered descent: a point above the pad, descending vertically."""

    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    apex: tuple[float, float, float]   # glide-slope apex, the touchdown point


@dataclass
class GuidancePlan:
    times: np.ndarray          # (N+1,) node times from plan start, s
    position: np.ndarray       # (N+1, 3)
    velocity: np.ndarray       # (N+1, 3)
    thrust_accel: np.ndarray   # (N+1, 3) u_k = T_k / m at the nodes, linear in between
    sigma: np.ndarray          # (N+1,)
    landing_start_s: float     # when the powered-descent leg starts
    mode: str                  # 'optimal' | 'soft_terminal'
    status: str
    cost: float                # Wh for 'energy', m/s for 'delta_v' (plus miss penalties if soft)
    energy_wh: float
    delta_v_m_s: float
    convexification_gap: float
    terminal_miss_m: float
    solve_time_s: float = 0.0
    solves: int = 0
    iterations: int = 0
    gravity: float = GRAVITY
    waypoint_nodes: list[int] = field(default_factory=list)
    final_target: str = 'gate'  # 'gate' (landing) or 'waypoint' (soft route fallback)
    corridor_excess_m: float = 0.0  # largest planned distance outside the route corridor
    route_deviation_m: float = 0.0  # largest planned node distance from the drawn route curves
    # Cost breakdown in the primary objective's unit (Wh or m/s): the primary
    # cost and each priced term (corridor, path, time, smoothness, tilt).
    cost_terms: dict = field(default_factory=dict)
    segments: list = field(default_factory=list, repr=False)

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reference position, velocity and thrust acceleration at plan time t.

        Exact within an interval: the thrust is linear between nodes.
        """
        g = np.array([0.0, 0.0, -self.gravity])
        if t >= self.times[-1]:
            return self.position[-1].copy(), self.velocity[-1].copy(), self.thrust_accel[-1].copy()
        k = max(0, int(np.searchsorted(self.times, t, side='right')) - 1)
        tau = max(0.0, t - float(self.times[k]))
        a = self.thrust_accel[k] + g
        jerk = (self.thrust_accel[k + 1] - self.thrust_accel[k]) / float(self.times[k + 1] - self.times[k])
        return (self.position[k] + self.velocity[k] * tau + 0.5 * a * tau ** 2 + jerk * tau ** 3 / 6.0,
                self.velocity[k] + a * tau + 0.5 * jerk * tau ** 2, self.thrust_accel[k] + jerk * tau)


@dataclass
class _Segment:
    kind: str                  # 'flypass' | 'hover' | 'hold' | 'land'
    duration: float
    nodes: int
    target: np.ndarray | None = None
    radius: float = 0.0
    curve: np.ndarray | None = None   # (M, 3) drawn route of this leg (corridor)
    along: np.ndarray | None = None   # (nodes,) arc fractions of the leg's nodes on the curve
    speed: float | None = None        # leg speed limit
    arrival_velocity: np.ndarray | None = None
    corridor: float | None = None     # corridor half-width around `curve`


def _arc(curve: np.ndarray) -> np.ndarray:
    """Cumulative arc length of a polyline, from 0."""
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))])


def curve_point(curve: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Point and unit tangent at an arc-length fraction of a polyline."""
    arc = _arc(curve)
    s = min(max(float(fraction), 0.0), 1.0) * arc[-1]
    i = int(min(max(np.searchsorted(arc, s, side='right') - 1, 0), len(curve) - 2))
    step = curve[i + 1] - curve[i]
    length = float(np.linalg.norm(step))
    alpha = 0.0 if length < 1e-12 else (s - arc[i]) / length
    return curve[i] + alpha * step, step / max(length, 1e-12)


def curve_fraction(curve: np.ndarray, points) -> np.ndarray:
    """Arc-length fraction of the nearest point on a polyline, per point."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    a, step = curve[:-1], np.diff(curve, axis=0)
    lengths2 = np.maximum(np.sum(step * step, axis=1), 1e-24)
    alpha = np.clip(np.einsum('pmi,mi->pm', points[:, None, :] - a[None], step) / lengths2, 0.0, 1.0)
    nearest = a[None] + alpha[..., None] * step[None]
    j = np.argmin(np.sum((nearest - points[:, None, :]) ** 2, axis=2), axis=1)
    arc = _arc(curve)
    return (arc[j] + alpha[np.arange(len(points)), j] * np.sqrt(lengths2[j])) / max(arc[-1], 1e-12)


def curve_distance(curve: np.ndarray, points) -> np.ndarray:
    """Distance of each point from a polyline (its cross-track error)."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    a, step = curve[:-1], np.diff(curve, axis=0)
    lengths2 = np.maximum(np.sum(step * step, axis=1), 1e-24)
    alpha = np.clip(np.einsum('pmi,mi->pm', points[:, None, :] - a[None], step) / lengths2, 0.0, 1.0)
    nearest = a[None] + alpha[..., None] * step[None]
    return np.sqrt(np.min(np.sum((nearest - points[:, None, :]) ** 2, axis=2), axis=1))


def remaining_curve_distance(curve: np.ndarray, start) -> float:
    """Distance to rejoin the curve plus its unflown arc, for leg timing only."""
    fraction = float(curve_fraction(curve, start)[0])
    nearest, _ = curve_point(curve, fraction)
    return float(np.linalg.norm(np.asarray(start) - nearest) + (1.0 - fraction) * _arc(curve)[-1])


def catmull_rom_leg(points, leg: int, samples: int = 49) -> np.ndarray:
    """Leg `leg` (points[leg] -> points[leg + 1]) of the route the mission
    sequencer and launch planner draw: a uniform Catmull-Rom spline with the
    end points repeated (tvc_env.envs.waypoints.catmull_rom, samplePlannerSpline)."""
    p = np.asarray(points, dtype=float)
    a, b = p[max(0, leg - 1)], p[leg]
    c, d = p[leg + 1], p[min(len(p) - 1, leg + 2)]
    if np.linalg.norm(c - b) < 1e-9:
        return np.repeat(b[None, :], samples, axis=0)
    t = np.linspace(0.0, 1.0, samples)[:, None]
    return 0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t * t + (-a + 3 * b - 3 * c + d) * t ** 3)


class _Cones:
    """Rows of A x + s = b grouped by cone, in Clarabel's cone order."""

    def __init__(self):
        self.blocks: dict[str, list] = {'zero': [], 'nonneg': [], 'soc': [], 'pow': []}

    def eq(self, terms, rhs):
        self.blocks['zero'].append((terms, rhs))

    def le(self, terms, rhs):
        """sum(coef * x) <= rhs."""
        self.blocks['nonneg'].append((terms, rhs))

    def soc(self, rows):
        """rows[0] >= ||rows[1:]|| where each row is (terms, constant): value = sum + constant."""
        self.blocks['soc'].append(rows)

    def power(self, rows, alpha):
        self.blocks['pow'].append((rows, alpha))

    def assemble(self, n):
        clarabel = _clarabel()
        ri, ci, vals, rhs, cones = [], [], [], [], []

        # Runs once per SOCP with ~10^4 terms (a route plan solves ~24 SOCPs):
        # per-row list extends instead of three appends per term.
        def put(terms, b, sign=1.0):
            if terms:
                cols, coefs = zip(*terms)
                ri.extend([len(rhs)] * len(cols))
                ci.extend(cols)
                vals.extend(coefs if sign > 0.0 else [-v for v in coefs])
            rhs.append(b)

        for terms, b in self.blocks['zero']:
            put(terms, b)
        if self.blocks['zero']:
            cones.append(clarabel.ZeroConeT(len(self.blocks['zero'])))
        for terms, b in self.blocks['nonneg']:
            put(terms, b)                 # s = b - a.x >= 0
        if self.blocks['nonneg']:
            cones.append(clarabel.NonnegativeConeT(len(self.blocks['nonneg'])))
        for rows in self.blocks['soc']:
            for terms, constant in rows:  # s = constant + a.x, so A = -a
                put(terms, constant, -1.0)
            cones.append(clarabel.SecondOrderConeT(len(rows)))
        for rows, alpha in self.blocks['pow']:
            for terms, constant in rows:
                put(terms, constant, -1.0)
            cones.append(clarabel.PowerConeT(alpha))
        A = sparse.csc_matrix((vals, (ri, ci)), shape=(len(rhs), n))
        return A, np.asarray(rhs, dtype=float), cones


class ConvexGuidance:
    """Free-final-time SOCP planner for landing and waypoint routes."""

    def __init__(self, limits: GuidanceLimits, energy: EnergyModel | None = None,
                 objective: str = 'energy', landing_nodes: int = 24, route_dt_s: float = 0.35,
                 max_leg_nodes: int = 40, flypass_capture_fraction: float = 0.5,
                 max_solve_time_s: float = 0.5, corridor_m: float | None = None,
                 corridor_weight: float = 10.0, workers: int = 4, corridor_mode: str = 'soft',
                 weights: ObjectiveWeights | None = None, flypass_min_speed_fraction: float = 1.0,
                 flypass_heading_tolerance_rad: float = 0.0):
        if objective not in ('energy', 'delta_v'):
            raise ValueError("objective must be 'energy' or 'delta_v'")
        if objective == 'energy' and energy is None:
            raise ValueError('The energy objective needs an EnergyModel')
        if corridor_m is not None and (corridor_m <= 0.0 or corridor_weight <= 0.0):
            raise ValueError('The route corridor needs a positive half-width and weight')
        self.limits = limits
        self.energy = energy
        self.objective = objective
        self.landing_nodes = int(landing_nodes)
        self.route_dt_s = float(route_dt_s)
        self.max_leg_nodes = int(max_leg_nodes)
        self.flypass_capture_fraction = float(flypass_capture_fraction)
        self.max_solve_time_s = float(max_solve_time_s)
        self.corridor_m = None if corridor_m is None else float(corridor_m)
        self.corridor_weight = float(corridor_weight)
        if corridor_mode not in ('soft', 'strict'):
            raise ValueError('Corridor mode must be soft or strict')
        self.corridor_mode = corridor_mode
        self.weights = weights or ObjectiveWeights()
        # Fly-through arrival: the velocity at a gate points along the drawn
        # route within this heading tolerance, with its along-route speed
        # between this fraction of the leg speed and the leg speed. 1.0 / 0
        # is the exact arrival velocity (leg speed along the tangent).
        if not 0.0 < flypass_min_speed_fraction <= 1.0 or not 0.0 <= flypass_heading_tolerance_rad < math.pi / 2:
            raise ValueError('Fly-through speed fraction must be in (0, 1] and heading tolerance in [0, 90) deg')
        self.flypass_min_speed_fraction = float(flypass_min_speed_fraction)
        self.flypass_heading_tolerance_rad = float(flypass_heading_tolerance_rad)
        # Independent SOCPs of one plan (the free-final-time sweep, the
        # soft-terminal durations) are solved on this many threads. Every
        # problem has the same equations as a serial solve. Wall-time solver
        # limits can still affect feasibility under CPU contention.
        self.workers = max(1, int(workers))
        self._executor = None

    def _pool(self):
        if self._executor is None:
            from concurrent.futures import ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(self.workers, thread_name_prefix='socp')
        return self._executor

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _solve_many(self, problems, stats, **options) -> list:
        """_solve for each (r0, v0, thrust, gate, segments), in order; concurrently when workers > 1."""
        if self.workers == 1 or len(problems) == 1:
            return [self._solve(*problem, stats, **options) for problem in problems]
        runs = [dict(solves=0) for _ in problems]
        plans = list(self._pool().map(lambda job: self._solve(*job[0], job[1], **options), zip(problems, runs)))
        stats['solves'] += sum(run['solves'] for run in runs)
        return plans

    # ---- public API ----

    def plan(self, position, velocity, thrust_now_n, gate: LandingGate,
             route: tuple[RouteWaypoint, ...] | list = (), landing_time_hint: float | None = None,
             path=None, landing_corridor_m: float | None = None,
             landing_speed_m_s: float | None = None) -> GuidancePlan | None:
        """Plan from the current state through `route` to `gate`.

        `thrust_now_n` is the rotor's present thrust, a world-frame vector in
        N (a scalar is taken as vertical): the thrust profile starts there and
        changes at the bounded rate. `landing_time_hint` (the previous plan's
        remaining powered-descent time) narrows the line search. `path` is the
        drawn route, one (M, 3) curve per leg: path[i] leads to route[i] and
        path[len(route)] to the gate. A leg follows its curve within its
        corridor half-width: route[i].corridor_m (landing_corridor_m for the
        landing leg), else the planner's corridor_m; a leg with neither has
        no corridor. `landing_speed_m_s` caps the landing leg's speed.
        """
        r0 = np.asarray(position, dtype=float)
        v0 = np.asarray(velocity, dtype=float)
        started = time.perf_counter()
        stats = dict(solves=0)
        route = tuple(route)
        curves = [None] * (len(route) + 1)
        widths = [wp.corridor_m if wp.corridor_m is not None else self.corridor_m for wp in route]
        widths.append(landing_corridor_m if landing_corridor_m is not None else self.corridor_m)
        if any(w is not None and w <= 0.0 for w in widths):
            raise ValueError('Corridor half-widths must be positive')
        if path is not None:
            if len(path) != len(route) + 1:
                raise ValueError(f'path needs one curve per leg: {len(route) + 1}, got {len(path)}')
            curves = [None if c is None or width is None or _arc(np.asarray(c, dtype=float))[-1] < 1e-6
                      else np.asarray(c, dtype=float) for c, width in zip(path, widths)]
        land_speed = None if landing_speed_m_s is None else min(float(landing_speed_m_s), self.limits.max_speed_m_s)
        best = None
        # Waypoint legs are timed from their speeds (_route_segments) and only
        # lengthened when infeasible. A time price makes their timing part of
        # the objective: a shorter allocation is also tried and the cheaper
        # plan (time cost included) wins.
        tiers = ((0.8, 1.0), (1.5,), (2.25,)) if route and self.weights.time > 0.0 else ((1.0,), (1.5,), (2.25,))
        for tier in tiers:
            for scale in tier:
                segments = self._route_segments(r0, v0, route, scale, curves, widths)
                plan = self._search_landing_time(r0, v0, thrust_now_n, gate, segments, landing_time_hint, stats,
                                                 land_curve=curves[-1], land_corridor=widths[-1],
                                                 land_speed=land_speed)
                if plan is not None and (best is None or plan.cost < best.cost):
                    best = plan
            if best is not None:
                break
        if best is not None and any(c is not None for c in curves):
            best = self._refine_corridor(r0, v0, thrust_now_n, gate, best, stats)
        # Strict corridors never silently degrade into an unconstrained
        # emergency trajectory. The adapter reports HOLD when no plan exists.
        if best is None and not (self.corridor_mode == 'strict' and any(c is not None for c in curves)):
            best = self._soft_terminal(r0, v0, thrust_now_n, gate, route, stats)
        if best is not None:
            best.route_deviation_m = self._route_deviation(best)
            best.solve_time_s = time.perf_counter() - started
            best.solves = stats['solves']
        return best

    @staticmethod
    def _route_deviation(plan) -> float:
        """Largest distance of a plan's nodes from their legs' drawn curves."""
        worst, node = 0.0, 0
        for seg in plan.segments:
            first, node = node, node + seg.nodes
            if seg.curve is not None and node > first:
                worst = max(worst, float(curve_distance(seg.curve, plan.position[first + 1:node + 1]).max()))
        return worst

    def _initial_thrust_accel(self, thrust_now, sig_max) -> np.ndarray:
        """Present thrust as an acceleration vector inside the problem's bounds.

        A rotor above the ceiling (the tracking loop may use the margin) or a
        body tilted past the pointing cone starts the plan on the nearest
        admissible vector; the duty slew and attitude loop close the rest.
        """
        lim = self.limits
        t = np.asarray(thrust_now, dtype=float)
        u = (np.array([0.0, 0.0, float(t)]) if t.ndim == 0 else t.copy()) / lim.mass_kg
        magnitude = float(np.linalg.norm(u))
        if magnitude < 1e-9:
            return np.zeros(3)
        if magnitude > sig_max:
            u *= sig_max / magnitude
            magnitude = sig_max
        horizontal = float(np.linalg.norm(u[:2]))
        if math.atan2(horizontal, u[2]) > lim.max_tilt_rad:
            azimuth = u[:2] / horizontal if horizontal > 1e-9 else np.zeros(2)
            u = magnitude * np.array([*(math.sin(lim.max_tilt_rad) * azimuth), math.cos(lim.max_tilt_rad)])
        return u

    # ---- time allocation ----

    def _lateral_accel(self) -> float:
        lim = self.limits
        return 0.5 * lim.gravity * math.tan(lim.max_tilt_rad)

    def _route_segments(self, r0, v0, route, scale, curves=None, widths=None) -> list[_Segment]:
        """Waypoint legs with durations from each leg's reference speed.

        Rest-to-rest hover legs also get acceleration and braking time. The
        planner retries with longer legs (scale) before giving up on a route.
        A leg on a corridor is timed along its curve, which a hard speed
        limit makes longer than the chord.
        """
        curves = curves or [None] * (len(route) + 1)
        widths = widths or [self.corridor_m] * (len(route) + 1)
        segments = []
        start, speed_in = r0, float(np.linalg.norm(v0))
        accel = self._lateral_accel()
        for i, wp in enumerate(route):
            target = np.asarray(wp.position, dtype=float)
            if wp.kind in (TAKEOFF, DESCENT) and curves[i] is not None:
                curves = list(curves)
                curves[i] = np.linspace(curves[i][0], target, 49)
            distance = float(np.linalg.norm(target - start))
            if curves[i] is not None:
                # Mission 8243dfb69c0d timed out at waypoint 0: every replan
                # allocated the full ~100 m incoming arc even beside the hover,
                # perpetually postponing arrival (~36 s). Time only the unflown
                # distance, retaining the complete curve for corridor checks.
                distance = max(distance, remaining_curve_distance(curves[i], start))
            cruise = min(max(float(wp.speed_m_s), 0.1), self.limits.max_speed_m_s)
            if wp.kind in STOP_KINDS:
                # Rest to rest: trapezoidal speed profile, or bang-bang when
                # the hop is too short to reach the cruise speed.
                duration = (distance / cruise + cruise / accel if distance > cruise * cruise / accel
                            else 2.0 * math.sqrt(distance / accel))
            else:
                duration = distance / cruise
            if i == 0:
                duration += speed_in / accel
            duration = max(duration * scale, 1.0 if wp.kind in STOP_KINDS else 0.6)
            direction = target - start
            if curves[i] is not None:
                _, direction = curve_point(curves[i], 1.0)
            direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
            segments.append(_Segment(wp.kind, duration, self._nodes(duration), target,
                                     float(wp.radius_m), curve=curves[i], speed=cruise,
                                     arrival_velocity=direction * cruise if wp.kind == FLYPASS else None,
                                     corridor=None if curves[i] is None else widths[i]))
            if wp.kind == HOVER:
                hold = max(float(wp.hold_s), 0.0) + 0.5
                segments.append(_Segment('hold', hold, max(2, self._nodes(hold)), target))
            start = target
        return segments

    def _nodes(self, duration):
        return int(min(self.max_leg_nodes, max(3, math.ceil(duration / self.route_dt_s))))

    # ---- free final time (Algorithm 1) ----

    def _search_landing_time(self, r0, v0, thrust_now, gate, segments, hint, stats, land_curve=None,
                             land_corridor=None, land_speed=None):
        lim = self.limits
        start = segments[-1].target if segments else r0
        g = np.asarray(gate.position, dtype=float)
        distance = float(np.linalg.norm(g - start))
        if land_curve is not None:
            distance = max(distance, remaining_curve_distance(land_curve, start))
        cruise = lim.max_speed_m_s if land_speed is None else land_speed
        speed_cap = max(cruise, float(np.linalg.norm(v0)))
        t_lo = max(0.8, 0.8 * distance / speed_cap)
        t_hi = max(3.0 * t_lo, distance / min(0.75, 0.8 * cruise) + 6.0)
        cache = {}

        def land(t_land):
            return _Segment('land', t_land, self.landing_nodes, curve=land_curve, speed=land_speed,
                            corridor=None if land_curve is None else land_corridor)

        def cost(t_land):
            key = round(t_land, 4)
            if key not in cache:
                cache[key] = self._solve(r0, v0, thrust_now, gate, segments + [land(t_land)], stats)
            plan = cache[key]
            return math.inf if plan is None else plan.cost

        def sweep(candidates):
            # The sweep's durations are independent problems: solve them
            # concurrently (Clarabel releases the GIL while it iterates).
            # Each problem, and so each plan, is identical to a serial solve.
            fresh = {}
            for t in candidates:
                fresh.setdefault(round(t, 4), t)       # the duration a serial cost(t) would solve
            fresh = {key: t for key, t in fresh.items() if key not in cache}
            if len(fresh) > 1:
                for key, plan in zip(fresh, self._solve_many(
                        [(r0, v0, thrust_now, gate, segments + [land(t)]) for t in fresh.values()], stats)):
                    cache[key] = plan
            return [cost(t) for t in candidates]

        if hint is not None and hint > 0.3:
            candidates = [max(0.5, hint * f) for f in (0.85, 1.0, 1.2)]
        else:
            candidates = list(np.geomspace(t_lo, t_hi, 8))
        values = sweep(candidates)
        if not any(math.isfinite(v) for v in values) and hint is not None:
            candidates = list(np.geomspace(t_lo, t_hi, 8))
            values = sweep(candidates)
        if not any(math.isfinite(v) for v in values):
            return None
        j = int(np.argmin(values))
        # Golden-section refinement in the bracket around the best sample.
        a = candidates[j - 1] if j > 0 else candidates[j] * 0.8
        b = candidates[j + 1] if j + 1 < len(candidates) else candidates[j] * 1.25
        c, d = b - _GOLDEN * (b - a), a + _GOLDEN * (b - a)
        for _ in range(6):
            fc, fd = sweep([c, d])
            # Infeasibility lies at short durations: with both probes
            # infeasible the minimum is to the right.
            if math.isfinite(fc) and fc <= fd:
                b, d = d, c
                c = b - _GOLDEN * (b - a)
            else:
                a, c = c, d
                d = a + _GOLDEN * (b - a)
        feasible = [plan for plan in cache.values() if plan is not None]
        return min(feasible, key=lambda plan: plan.cost)

    # ---- SOCP ----

    def _solve(self, r0, v0, thrust_now, gate, segments, stats, soft_weight=None, velocity_weight=None):
        """One fixed-duration SOCP (Problem 4 with the segment constraints).

        With `soft_weight`, the terminal position/velocity become penalized
        slacks and the state constraints are dropped (see _soft_terminal).
        """
        clarabel = _clarabel()
        lim = self.limits
        m, g = lim.mass_kg, lim.gravity
        gvec = np.array([0.0, 0.0, -g])
        dts = np.concatenate([np.full(s.nodes, s.duration / s.nodes) for s in segments])
        n_int = len(dts)
        times = np.concatenate([[0.0], np.cumsum(dts)])
        # Variables: [r_k, v_k] per node, then [u_k, sigma_k] per node (the
        # thrust is linear in between), then energy epigraphs c_k per node,
        # then soft-terminal miss slacks.
        X = lambda k, i: 6 * k + i                       # noqa: E731  r: i<3, v: 3..5
        off_u = 6 * (n_int + 1)
        U = lambda k, i: off_u + 4 * k + i               # noqa: E731  u: i<3, sigma: 3
        off_c = off_u + 4 * (n_int + 1)
        n_c = n_int + 1 if self.objective == 'energy' else 0
        off_e = off_c + n_c
        # Route corridor: (node, chord end points). Each node is measured from
        # a chord of its leg's curve, the distance to a segment being convex.
        # The chord caps the leg at its waypoint, so running past it counts
        # (a tangent-line corridor let a near-vertical leg sink 20 m below its
        # waypoint for free). Without placements the chord is the whole leg,
        # which leaves the timing free; _refine_corridor then measures each
        # node from the short chord around its planned point on the curve
        # (+-CORRIDOR_WINDOW of the leg), 0.2-0.6 m from the curve where the
        # whole chord is 1-3 m.
        corridor = []
        if soft_weight is None:
            node = 0
            for seg in segments:
                first, node = node, node + seg.nodes
                if seg.curve is None or seg.corridor is None:
                    continue
                for j, k in enumerate(range(first + 1, node + 1)):
                    if seg.along is None:
                        corridor.append((k, seg.curve[0], seg.curve[-1], seg.corridor, None, None))
                    else:
                        lo, _ = curve_point(seg.curve, seg.along[j] - CORRIDOR_WINDOW)
                        hi, _ = curve_point(seg.curve, seg.along[j] + CORRIDOR_WINDOW)
                        point, tangent = curve_point(seg.curve, seg.along[j])
                        corridor.append((k, lo, hi, seg.corridor, point, tangent))
        off_w = off_e + (3 if soft_weight is not None else 0)
        # Path cost: one cross-track epigraph per corridor node.
        path_nodes = corridor if self.weights.path > 0.0 else []
        off_p = off_w + 2 * len(corridor)      # per corridor node: excess, chord parameter
        # Sink-rate envelope: one bound s_k <= sqrt(v_gate^2 + 2 a (z_k - z_gate))
        # per landing-leg node (see _sink_envelope).
        sink_envelope = (soft_weight is None and bool(lim.landing_sink_brake_fraction)
                         and segments[-1].kind == 'land')
        off_s = off_p + len(path_nodes)
        n_var = off_s + (segments[-1].nodes if sink_envelope else 0)
        cones = _Cones()

        for i in range(3):
            cones.eq([(X(0, i), 1.0)], r0[i])
            cones.eq([(X(0, 3 + i), 1.0)], v0[i])
        for k, dt in enumerate(dts):
            for i in range(3):
                # First-order hold, exact for u linear from u_k to u_k+1:
                # r+ = r + dt v + dt^2 (u_k/3 + u_k+1/6 + g/2);  v+ = v + dt ((u_k + u_k+1)/2 + g)
                cones.eq([(X(k + 1, i), 1.0), (X(k, i), -1.0), (X(k, 3 + i), -dt),
                          (U(k, i), -dt * dt / 3.0), (U(k + 1, i), -dt * dt / 6.0)], 0.5 * dt * dt * gvec[i])
                cones.eq([(X(k + 1, 3 + i), 1.0), (X(k, 3 + i), -1.0), (U(k, i), -0.5 * dt),
                          (U(k + 1, i), -0.5 * dt)], dt * gvec[i])

        sig_min, sig_max = lim.thrust_min_n / m, lim.thrust_max_n / m
        rate = lim.thrust_rate_n_s / m
        sig_full = sig_max if lim.emergency_thrust_max_n is None else max(sig_max, lim.emergency_thrust_max_n / m)
        if soft_weight is not None:
            sig_max = sig_full
            rate = (lim.emergency_thrust_rate_n_s or lim.thrust_rate_n_s) / m
        u_now = self._initial_thrust_accel(thrust_now, sig_full)
        sig_now = float(np.linalg.norm(u_now))
        # Speed bound terms (see below). Until the rate-limited thrust returns
        # to weight the vehicle keeps accelerating by up to (T_now - W)^2 /
        # (2 m Tdot).
        weight = m * g
        settle = abs(sig_now * m - weight) / (rate * m)
        overshoot = (sig_now * m - weight) ** 2 / (2.0 * m * rate * m)
        entry_speed = float(np.linalg.norm(v0)) + overshoot + 0.1
        brake = 0.5 * min(sig_max - g, g - sig_min, g * math.tan(lim.max_tilt_rad))
        # On a route corridor an over-speed start may brake at full thrust
        # until the speed bound below is back at max_speed: the reserved
        # ceiling (half the excess thrust) braked a 20 m/s descent at 2 m/s^2,
        # and the plan sank 26 m below its waypoint and climbed back (Isaac
        # mission 835c3de32185). Without a corridor the energy objective only
        # brakes as hard as the floor forces: offline, a full-thrust allowance
        # turned that mission's maximum-braking fallback into a plan that
        # skimmed the 0.8 m floor, with no thrust left for tracking.
        braking_until = -math.inf
        if soft_weight is None and corridor and float(np.linalg.norm(v0)) > lim.max_speed_m_s:
            braking_until = settle + (entry_speed - lim.max_speed_m_s) / brake
        cos_tilt = math.cos(lim.max_tilt_rad)
        for k in range(n_int + 1):
            # A spooling rotor cannot reach rho1 at once: ramp the lower bound.
            cones.le([(U(k, 3), -1.0)], -min(sig_min, sig_now + rate * times[k]))
            # A rotor above the ceiling (a braking plan, or tracking margin)
            # comes back down at the thrust rate, not in one step.
            ceiling = sig_full if times[k] <= braking_until else sig_max
            cones.le([(U(k, 3), 1.0)], max(ceiling, min(sig_now - rate * times[k], sig_full)))
            cones.le([(U(k, 3), cos_tilt), (U(k, 2), -1.0)], 0.0)
            cones.soc([([(U(k, 3), 1.0)], 0.0)] + [([(U(k, i), 1.0)], 0.0) for i in range(3)])
        # The thrust profile starts at the rotor's present thrust vector and
        # is continuous (first-order hold), so the rate bound below bounds the
        # thrust the vehicle actually has to follow. The paper's zero-order
        # hold stepped it at every node: up to 2.2 m/s^2 (8.6 deg of thrust
        # direction) every 0.46 s on a 30 m approach, and 12.6 deg at once
        # when a hover handed over to the landing leg (mission e7d7fbb81708:
        # an 11.6 deg tilt command in one frame and 48 deg/s body rates).
        for i in range(3):
            cones.eq([(U(0, i), 1.0)], u_now[i])
        # Thrust-rate bound on the thrust vector itself. A bound on the slack
        # sigma is not one on ||u|| once sigma > ||u||, and would let the plan
        # change the real thrust faster than the rotor can follow. The vector
        # form also bounds the thrust-direction slew (attitude rate), and
        # implies | ||u_k+1|| - ||u_k|| | <= rate dt.
        for k in range(n_int):
            cones.soc([([], rate * dts[k])] + [([(U(k + 1, i), 1.0), (U(k, i), -1.0)], 0.0) for i in range(3)])
        # Attitude slew: the thrust tilts with the body, and the vanes can only
        # precess the rotor's angular momentum so fast. Bounding the rate of the
        # horizontal thrust acceleration by g * tilt_rate bounds the tilt rate
        # (to first order in tilt) with a convex constraint.
        if lim.tilt_rate_rad_s is not None:
            slew = g * lim.tilt_rate_rad_s
            for k in range(n_int):
                cones.soc([([], slew * dts[k])] + [([(U(k + 1, i), 1.0), (U(k, i), -1.0)], 0.0) for i in range(2)])

        # Speed bound (paper eq. 10). A start at or near the bound (a fast
        # entry, tracking error) may keep that speed while the thrust settles,
        # but must then brake back under the bound at half the slowest braking
        # the limits allow. One |v0|-based cap
        # for every node let each re-plan fly faster than the last, since the
        # energy objective flies at the cap: 4.0 -> 5.9 m/s in 0.25 m/s steps
        # on a 30 m approach (Isaac mission 0289417f5d60), which then had no
        # feasible plan to the gate 2.9 m above the pad.
        if soft_weight is None:
            for k in range(1, n_int + 1):
                cap = max(lim.max_speed_m_s, entry_speed - brake * max(0.0, float(times[k]) - settle))
                cones.soc([([], cap)] + [([(X(k, 3 + i), 1.0)], 0.0) for i in range(3)])

        # Segment constraints.
        node, waypoint_nodes = 0, []
        z_floor = min(lim.route_floor_m, float(r0[2]))
        apex = np.asarray(gate.apex, dtype=float)
        previous_speed = float(np.linalg.norm(v0))
        horizontal_slew = min(rate, g * lim.tilt_rate_rad_s) if lim.tilt_rate_rad_s else rate
        # A newly requested slower leg cannot brake before thrust has tilted
        # into the braking direction. Include that actuator ramp in the entry
        # envelope (a 3.5 m/s crosswind entry otherwise has no feasible route
        # at a 3 m/s leg limit despite a soft spatial corridor).
        brake_ramp = g * math.tan(lim.max_tilt_rad) / horizontal_slew
        def leg_speed_cap(seg, first, k):
            # A waypoint speed is a leg cap, not just a time-allocation hint.
            # Retain the physical entry braking envelope on replans.
            return max(seg.speed, entry_speed - brake * max(0., float(times[k]) - settle - brake_ramp),
                       previous_speed - brake * float(times[k] - times[first]))

        for seg in segments:
            first, node = node, node + seg.nodes
            if seg.kind == 'land':
                land_first = first
                if soft_weight is None and seg.speed is not None:
                    # Landing approach speed (a Landing step's setting).
                    for k in range(first + 1, node + 1):
                        cones.soc([([], leg_speed_cap(seg, first, k))] + [([(X(k, 3+i), 1.)], 0.) for i in range(3)])
                continue
            if soft_weight is None:
                for k in range(first + 1, node + 1):
                    cones.le([(X(k, 2), -1.0)], -z_floor)
                    if seg.speed is not None:
                        cones.soc([([], leg_speed_cap(seg, first, k))] + [([(X(k, 3+i), 1.)], 0.) for i in range(3)])
                    if seg.kind in (TAKEOFF, DESCENT):
                        # Vertical legs stay over their column; speed can ramp
                        # during acceleration/braking and the endpoint is at rest.
                        cones.soc([([], seg.radius)] + [
                            ([(X(k, i), 1.)], -seg.target[i]) for i in range(2)])
            if seg.kind == FLYPASS:
                waypoint_nodes.append(node)
                ball = self.flypass_capture_fraction * seg.radius
                if soft_weight is None:
                    cones.soc([([], ball)] + [([(X(node, i), 1.0)], -seg.target[i]) for i in range(3)])
                    if seg.arrival_velocity is not None:
                        self._arrival(cones, X, node, seg.arrival_velocity)
            elif seg.kind in STOP_KINDS:
                # At rest on the point: position, velocity and hover thrust.
                # Under the first-order hold, zero velocity at every hold node
                # only fixes u_k + u_k+1 = 2 g, so without the thrust the plan
                # alternated +/-0.9 N node to node through a hold (Isaac
                # mission f1b3c7b587da: a 1.5 Hz, 6 deg/s wobble at a hover).
                waypoint_nodes.append(node)
                for i in range(3):
                    cones.eq([(X(node, i), 1.0)], seg.target[i])
                    cones.eq([(X(node, 3 + i), 1.0)], 0.0)
                    cones.eq([(U(node, i), 1.0)], -gvec[i])
            elif seg.kind == 'hold':
                # Hover thrust from rest keeps the vehicle on the point.
                for k in range(first + 1, node + 1):
                    for i in range(3):
                        cones.eq([(U(k, i), 1.0)], -gvec[i])
            previous_speed = seg.speed if seg.kind == FLYPASS else 0.0

        gate_r = np.asarray(gate.position, dtype=float)
        gate_v = np.asarray(gate.velocity, dtype=float)
        N = n_int
        if soft_weight is None:
            for i in range(3):
                cones.eq([(X(N, i), 1.0)], gate_r[i])
                cones.eq([(X(N, 3 + i), 1.0)], gate_v[i])
            # The powered-descent leg reaches the gate from above. The glide
            # cone's apex is at touchdown, so a centered plan could sink below
            # the gate and climb back: after a fast final approach the plan of
            # Isaac mission 614cb15ab14a dipped to 0.52 m (gate 0.81 m,
            # touchdown 0.31 m) and the vehicle sank to 0.48 m before climbing
            # back to the gate at 0.5 m/s.
            if segments[-1].kind == 'land':
                leg_start = r0 if land_first == 0 else segments[-2].target
                gate_floor = min(float(gate_r[2]), float(leg_start[2]))
                for k in range(land_first + 1, N + 1):
                    cones.le([(X(k, 2), -1.0)], -gate_floor)
                if sink_envelope:
                    self._sink_envelope(cones, X, off_s, land_first, N, times, gate_floor,
                                        float(abs(gate_v[2])), r0, v0, sig_max, sig_now, rate)
            # Glide slope on the powered-descent leg (paper eq. 11). A start
            # outside the cone is not rejected: the cone is widened to contain
            # it, then narrows back to the nominal half-angle by the final
            # glide_slope_final_s before the gate (still convex: a fixed
            # half-angle per node). Widened for the whole leg, every re-plan
            # rode the wide cone to the pad: from 30 m out and 18 m up (a 60
            # deg cone) mission 0289417f5d60 was still flying 2 m/s sideways
            # 1 m above the pad and reached the gate 0.4 m off center.
            if lim.glide_slope_rad is not None and segments[-1].kind == 'land':
                start = r0 if land_first == 0 else (
                    segments[-2].target if len(segments) > 1 else r0)
                offset = np.asarray(start, dtype=float) - apex
                height = max(offset[2], 1e-6)
                needed = math.atan2(float(np.linalg.norm(offset[:2])), height) + math.radians(2.0)
                tan_nominal = math.tan(lim.glide_slope_rad)
                tan_start = math.tan(min(max(lim.glide_slope_rad, needed), math.radians(88.0)))
                t0, leg = float(times[land_first]), float(times[N] - times[land_first])
                narrowed = max(leg - lim.glide_slope_final_s, 0.5 * leg)
                for k in range(land_first + 1, N + 1):
                    share = min(1.0, (float(times[k]) - t0) / narrowed)
                    tan_gs = tan_start + (tan_nominal - tan_start) * share
                    cones.soc([([(X(k, 2), tan_gs)], -tan_gs * apex[2]),
                               ([(X(k, 0), 1.0)], -apex[0]), ([(X(k, 1), 1.0)], -apex[1])])
            # Upright, steady arrival: the thrust at the gate is vertical
            # (paper eq. 37) and equals weight, so the gate hands over to the
            # constant-rate terminal descent without a thrust step the
            # rate-limited rotor could not follow.
            cones.eq([(U(N, 0), 1.0)], 0.0)
            cones.eq([(U(N, 1), 1.0)], 0.0)
            cones.eq([(U(N, 2), 1.0)], g)
        else:
            e_r, e_v, e_ground = off_e, off_e + 1, off_e + 2
            cones.soc([([(e_r, 1.0)], 0.0)] + [([(X(N, i), 1.0)], -gate_r[i]) for i in range(3)])
            cones.soc([([(e_v, 1.0)], 0.0)] + [([(X(N, 3 + i), 1.0)], -gate_v[i]) for i in range(3)])
            # The ground stays a (heavily penalized) constraint: when contact
            # cannot be avoided the plan degrades to maximum braking instead
            # of a path through the pad.
            cones.le([(e_ground, -1.0)], 0.0)
            for k in range(1, N + 1):
                cones.le([(X(k, 2), -1.0), (e_ground, -1.0)], -apex[2])

        q = np.zeros(n_var)
        weights = _trapezoid_weights(dts)
        if self.objective == 'energy':
            for k in range(n_int + 1):
                # c_k >= sigma_k^1.5  <=>  (c_k, 1, sigma_k) in K_pow(2/3)
                cones.power([([(off_c + k, 1.0)], 0.0), ([], 1.0), ([(U(k, 3), 1.0)], 0.0)], 2.0 / 3.0)
                q[off_c + k] = weights[k]
        else:
            for k in range(n_int + 1):
                q[U(k, 3)] = weights[k]
        if soft_weight is not None:
            q[off_e] = soft_weight
            q[off_e + 1] = soft_weight if velocity_weight is None else velocity_weight
            q[off_e + 2] = 10.0 * soft_weight
        # Corridor: ||r_k - (a + lambda (b - a))|| <= half-width + excess with
        # 0 <= lambda <= 1, the excess priced at corridor_weight seconds of
        # hover cost per m s.
        hover_cost = g ** 1.5 if self.objective == 'energy' else g
        for j, (k, a, b, width, _, _) in enumerate(corridor):
            excess, lam = off_w + 2 * j, off_w + 2 * j + 1
            cones.le([(excess, -1.0)], 0.0)
            if self.corridor_mode == 'strict':
                cones.eq([(excess, 1.0)], 0.0)
            cones.le([(lam, -1.0)], 0.0)
            cones.le([(lam, 1.0)], 1.0)
            cones.soc([([(excess, 1.0)], width)]
                      + [([(X(k, i), 1.0), (lam, -float(b[i] - a[i]))], -float(a[i])) for i in range(3)])
            q[excess] = self.corridor_weight * hover_cost * weights[k]
        # Path: cross-track distance d_k from the drawn route. Before the
        # nodes are placed on the curve it is the distance to the leg's chord
        # (sharing the corridor's chord parameter); once placed, the distance
        # to the curve's tangent line at the node's point, which measures
        # cross-track error only and leaves the along-track timing free (the
        # corridor chord still caps the leg at its waypoint).
        for j, (k, a, b, _, point, tangent) in enumerate(path_nodes):
            d = off_p + j
            if point is None:
                lam = off_w + 2 * j + 1
                rows = [([(X(k, i), 1.0), (lam, -float(b[i] - a[i]))], -float(a[i])) for i in range(3)]
            else:
                perp = np.eye(3) - np.outer(tangent, tangent)
                offset = perp @ point
                rows = [([(X(k, c), float(perp[i, c])) for c in range(3) if abs(perp[i, c]) > 1e-12],
                         -float(offset[i])) for i in range(3)]
            cones.soc([([(d, 1.0)], 0.0)] + rows)
            q[d] = self.weights.path * hover_cost * weights[k]
        # Quadratic terms, 0.5 x'Px (upper triangle), accumulated as COO
        # triplets (duplicates sum on conversion; lil item updates cost ~5 us each).
        p_row, p_col, p_val = [], [], []
        if soft_weight is None and self.weights.smoothness > 0.0:
            # Thrust jerk j_k = (u_k+1 - u_k) / dt_k, priced (|j| / J_ref)^2 dt:
            # J_ref is the planned attitude-slew bound as a horizontal jerk
            # (the thrust-rate bound without one).
            j_ref = g * lim.tilt_rate_rad_s if lim.tilt_rate_rad_s else rate
            for k, dt in enumerate(dts):
                c = 2.0 * self.weights.smoothness * hover_cost / (dt * j_ref ** 2)
                for i in range(3):
                    lo, hi = U(k, i), U(k + 1, i)
                    p_row += (lo, hi, lo)
                    p_col += (lo, hi, hi)
                    p_val += (c, c, -c)
        if soft_weight is None and self.weights.tilt > 0.0:
            # Horizontal thrust acceleration, priced (|u_xy| / a_max)^2 dt at
            # the planned tilt limit a_max = g tan(max_tilt).
            a_max = g * math.tan(lim.max_tilt_rad)
            for k in range(n_int + 1):
                c = 2.0 * self.weights.tilt * hover_cost * weights[k] / a_max ** 2
                for i in range(2):
                    p_row.append(U(k, i))
                    p_col.append(U(k, i))
                    p_val.append(c)

        A, b, cone_list = cones.assemble(n_var)
        P = sparse.triu(sparse.csc_matrix((np.asarray(p_val, dtype=float), (np.asarray(p_row, dtype=np.int64),
                        np.asarray(p_col, dtype=np.int64))), shape=(n_var, n_var)), format='csc')
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = 200
        settings.time_limit = self.max_solve_time_s
        solution = clarabel.DefaultSolver(P, q, A, b, cone_list, settings).solve()
        stats['solves'] += 1
        status = str(solution.status)
        if status not in _OK:
            return None
        x = np.asarray(solution.x)
        states = x[:off_u].reshape(n_int + 1, 6)
        controls = x[off_u:off_c].reshape(n_int + 1, 4)
        u, sigma = controls[:, :3], controls[:, 3]
        miss = None if soft_weight is None else float(q[off_e:off_w] @ x[off_e:off_w])
        plan = self._package(times, states, u, sigma, dts, segments, gate_r, mode=(
            'optimal' if soft_weight is None else 'soft_terminal'), status=status,
            iterations=int(solution.iterations), waypoint_nodes=waypoint_nodes, soft=miss)
        # The line search ranks durations by cost: add every priced term, in
        # Wh for the energy objective (its sigma^1.5 s is P_ref (m/T_ref)^1.5 J).
        unit = 1.0
        if self.objective == 'energy':
            e = self.energy
            unit = e.power_ref_w * (m / e.thrust_ref_n) ** 1.5 / 3600.0
        terms = {self.objective: plan.cost if miss is None else plan.cost - miss}
        if miss is not None:
            terms['miss'] = miss
        if corridor:
            plan.corridor_excess_m = float(max(x[off_w:off_p:2].max(), 0.0))
            terms['corridor'] = unit * float(q[off_w:off_p] @ x[off_w:off_p])
        if path_nodes:
            end = off_p + len(path_nodes)
            terms['path'] = unit * float(q[off_p:end] @ x[off_p:end])
        if self.weights.time > 0.0:
            terms['time'] = unit * self.weights.time * hover_cost * float(times[-1])
        if P.nnz:
            # Split the quadratic cost by term: smoothness couples thrust
            # nodes, tilt is diagonal on the horizontal components.
            quadratic = 0.5 * float(x @ (P @ x) + x @ (P.T @ x) - x @ (P.diagonal() * x))
            if self.weights.tilt > 0.0:
                a_max = g * math.tan(lim.max_tilt_rad)
                tilt = self.weights.tilt * hover_cost / a_max ** 2 * float(
                    weights @ np.sum(u[:, :2] ** 2, axis=1))
                terms['tilt'] = unit * tilt
                quadratic -= tilt
            if self.weights.smoothness > 0.0:
                terms['smoothness'] = unit * quadratic
        plan.cost = float(sum(terms.values()))
        plan.cost_terms = {key: round(value, 6) for key, value in terms.items() if math.isfinite(value)}
        return plan

    def _arrival(self, cones, X, node, arrival_velocity):
        """Fly-through velocity: exact, or inside a cone around the route tangent.

        The exact velocity (leg speed along the tangent) fixes the shape of
        every leg around a gate: offline, energy plans rode the 1 m corridor
        edge (0.98 m) whatever the path weight, and a 10 m zigzag had no
        feasible plan under the 10 deg/s attitude-slew bound. The cone
        frees the plan to slow down and cut in by up to the heading
        tolerance while still passing the gate in the route's direction.
        """
        speed = float(np.linalg.norm(arrival_velocity))
        if self.flypass_min_speed_fraction >= 1.0 and self.flypass_heading_tolerance_rad <= 0.0 or speed < 1e-9:
            for i in range(3):
                cones.eq([(X(node, 3 + i), 1.0)], float(arrival_velocity[i]))
            return
        tangent = np.asarray(arrival_velocity, dtype=float) / speed
        along = [(X(node, 3 + i), float(tangent[i])) for i in range(3)]
        cones.le([(c, -v) for c, v in along], -self.flypass_min_speed_fraction * speed)
        cones.le(along, speed)
        # Two unit vectors across the tangent span the cross-route velocity.
        helper = np.eye(3)[int(np.argmin(np.abs(tangent)))]
        e1 = np.cross(tangent, helper)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(tangent, e1)
        across = [[(X(node, 3 + i), float(e[i])) for i in range(3)] for e in (e1, e2)]
        if self.flypass_heading_tolerance_rad <= 0.0:
            for terms in across:
                cones.eq(terms, 0.0)
        else:
            slope = math.tan(self.flypass_heading_tolerance_rad)
            cones.soc([([(c, slope * v) for c, v in along], 0.0)] + [(terms, 0.0) for terms in across])

    def _sink_envelope(self, cones, X, off_s, land_first, N, times, z_gate, v_gate, r0, v0, sig_max, sig_now, rate):
        """Landing-leg sink-rate envelope: -v_z <= sqrt(v_gate^2 + 2 a (z - z_gate)).

        a is landing_sink_brake_fraction of the planned braking acceleration
        (sig_max - g): from every node of the plan the vehicle could still
        reach the gate speed using only that share of its braking margin. The
        rest is left for disturbances and for the actuator lag the tracking
        loop has to make up. The set {(z, v): v^2 <= c + 2 a z} is convex (a
        rotated second-order cone through an auxiliary bound s_k per node).

        Evidence (Isaac 751dd0214086, c41efd7ff96f; offline replica
        2026-09-28): an energy-optimal landing leg from a 4.6 m hover sank at
        2.0 m/s and braked on the thrust-rate bound, the same slew the duty
        limiter allows, so the tracking loop had no slew left to make up a
        lag. The vehicle reached the gate at 2.5 m/s and the soft-terminal
        fallback touched down at 1.7-2.1 m/s. In the replica (bus sag, 0.8 s
        downdrafts of 1.5 m/s^2 over 3-8 m landing legs) 0.35 cut hard
        touchdowns from 13 to 3 of 32 together with the adapter's braking
        emergency (6 of 32 with the emergency alone).

        A leg that starts outside the envelope (a fast direct descent) gets
        the excess sink rate as a relaxation that shrinks at the full planned
        braking acceleration once the rate-limited thrust has settled back to
        weight and ramped up to the braking thrust.
        """
        lim = self.limits
        g = lim.gravity
        a_brake = max(sig_max - g, 1e-3)
        a = float(lim.landing_sink_brake_fraction) * a_brake
        extra, settle = 0.0, 0.0
        # In velocity, a linear thrust ramp to a_brake equals full braking
        # delayed by half the ramp.
        ramp = 0.5 * a_brake / rate
        if land_first == 0:
            height = max(float(r0[2]) - z_gate, 0.0)
            # Thrust below weight keeps accelerating the descent until the
            # rate-limited rotor has returned to weight.
            settle = max(0.0, g - sig_now) / rate
            sink = max(0.0, -float(v0[2])) + 0.5 * max(0.0, g - sig_now) * settle
            extra = max(0.0, sink - math.sqrt(v_gate ** 2 + 2.0 * a * height))
        base = v_gate ** 2 - 2.0 * a * z_gate
        for j, k in enumerate(range(land_first + 1, N + 1)):
            s = off_s + j
            # s_k^2 <= L_k = v_gate^2 + 2 a (z_k - z_gate):  ||(2 s_k, L_k - 1)|| <= L_k + 1
            cones.soc([([(X(k, 2), 2.0 * a)], base + 1.0), ([(s, 2.0)], 0.0), ([(X(k, 2), 2.0 * a)], base - 1.0)])
            relax = max(0.0, extra - a_brake * max(0.0, float(times[k]) - settle - ramp))
            cones.le([(X(k, 5), -1.0), (s, -1.0)], relax)

    def _refine_corridor(self, r0, v0, thrust_now, gate, plan, stats):
        """Re-solve once with each node's corridor point where the plan is.

        The first solve places a leg's nodes along its curve in proportion to
        time; a plan that brakes hard and then cruises runs ahead of that
        schedule, and on a curved leg the corridor would then be measured
        across the tangent of the wrong point.
        """
        segments, node = [], 0
        for seg in plan.segments:
            first, node = node, node + seg.nodes
            if seg.curve is not None:
                along = np.maximum.accumulate(curve_fraction(seg.curve, plan.position[first + 1:node + 1]))
                seg = replace(seg, along=along)
            segments.append(seg)
        refined = self._solve(r0, v0, thrust_now, gate, segments, stats)
        candidate = plan if refined is None else refined
        # Chord constraints approximate a curved tube. Strict mode additionally
        # checks the actual sampled centerline, including between SOCP nodes,
        # so a shortcut cannot be labelled a feasible corridor plan.
        if self.corridor_mode == 'strict':
            node = 0
            for seg in candidate.segments:
                first, node = node, node + seg.nodes
                if seg.curve is None or seg.corridor is None:
                    continue
                samples = np.linspace(candidate.times[first], candidate.times[node], seg.nodes * 8 + 1)
                positions = np.array([candidate.sample(float(t))[0] for t in samples])
                delta = seg.curve[1:] - seg.curve[:-1]
                alpha = np.clip(np.sum((positions[:, None] - seg.curve[:-1]) * delta, axis=2)
                                / np.maximum(np.sum(delta * delta, axis=1), 1e-12), 0, 1)
                distance = np.linalg.norm(positions[:, None] - seg.curve[:-1] - alpha[:, :, None] * delta, axis=2).min(axis=1)
                if distance.max() > seg.corridor + 1e-4:
                    return None
        return candidate

    def _soft_terminal(self, r0, v0, thrust_now, gate, route, stats):
        """No feasible duration: minimize the miss of the next target.

        Keeps every actuator constraint (thrust bounds, tilt, rate) and drops
        the state constraints, so the closest reachable approach is flown and
        the next replan can recover the full problem. On a route the next
        target is the active waypoint (at rest for a hover), never the pad,
        so a waypoint is not skipped.
        """
        target = gate
        if route:
            wp = route[0]
            target = LandingGate(position=tuple(wp.position), velocity=(0.0, 0.0, 0.0), apex=gate.apex)
        best = None
        plans = self._solve_many([(r0, v0, thrust_now, target, [_Segment('land', t_land, self.landing_nodes)])
                                  for t_land in (2.0, 4.0, 8.0, 14.0)], stats, soft_weight=1.0e3,
                                 velocity_weight=1.0e3 if not route or route[0].kind == HOVER else 50.0)
        for plan in plans:
            if plan is not None and (best is None or plan.cost < best.cost):
                best = plan
        if best is not None and route:
            best.final_target = 'waypoint'
        return best

    def _package(self, times, states, u, sigma, dts, segments, gate_r, mode, status, iterations,
                 waypoint_nodes, soft):
        lim = self.limits
        norm_u = np.linalg.norm(u, axis=1)
        gap = float(np.max((sigma - norm_u) / np.maximum(sigma, 1e-9)))
        weights = _trapezoid_weights(dts)
        delta_v = float(np.sum(sigma * weights))
        thrust = sigma * lim.mass_kg
        if self.energy is not None:
            e = self.energy
            power = e.power_ref_w * np.power(np.maximum(thrust, 0.0) / e.thrust_ref_n, 1.5) + e.auxiliary_w
            energy_wh = float(np.sum(power * weights) / 3600.0)
        else:
            energy_wh = math.nan
        cost = energy_wh if self.objective == 'energy' else delta_v
        if soft is not None:
            cost += soft
        landing_start = float(times[len(times) - 1 - segments[-1].nodes]) if segments[-1].kind == 'land' else float(times[-1])
        return GuidancePlan(times=times, position=states[:, :3].copy(), velocity=states[:, 3:].copy(),
                            thrust_accel=u.copy(), sigma=sigma.copy(), landing_start_s=landing_start,
                            mode=mode, status=status, cost=float(cost), energy_wh=energy_wh,
                            delta_v_m_s=delta_v, convexification_gap=max(0.0, gap),
                            terminal_miss_m=float(np.linalg.norm(states[-1, :3] - gate_r)),
                            iterations=iterations, gravity=lim.gravity, waypoint_nodes=waypoint_nodes,
                            segments=list(segments))
