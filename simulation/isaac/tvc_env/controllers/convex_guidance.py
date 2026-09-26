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
  * waypoints: a fly-through node lies inside a capture ball; a hover node is
    at rest on the point and stays there for the remaining hold time;
  * route corridor (optional, soft): each node stays within a half-width of
    the drawn route, the mission sequencer's Catmull-Rom curve, measured from
    a chord of it (the distance to a segment is convex: an SOC per node). Any
    excess is a penalized slack, so a start the corridor cannot contain still
    has a plan.

A plan that follows a corridor may use the full thrust (the soft-terminal
ceiling) while it brakes an over-speed start back under the speed bound.
Planned at the reserved ceiling, a 20 m/s descent braked at 2 m/s^2 and sank
26 m below its waypoint (Isaac mission 835c3de32185). Without a corridor the
energy objective brakes no harder than the floor forces, so the reserve stays.

Objective: 'energy' minimizes electrical energy with the simulator's
momentum-theory power law P = P_ref (T/T_ref)^1.5 (a power cone per node,
trapezoid rule); 'delta_v' minimizes the integral of ||T||/m, the paper's fuel analog and the
project's propulsive delta-v metric. The corridor penalty is priced in
seconds of hover cost per metre-second outside it. The final time is free, so the landing
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
class RouteWaypoint:
    position: tuple[float, float, float]
    kind: str = FLYPASS
    radius_m: float = 1.0
    speed_m_s: float = 3.0
    hold_s: float = 0.0      # remaining hold time for a hover waypoint


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


def catmull_rom_leg(points, leg: int, samples: int = 49) -> np.ndarray:
    """Leg `leg` (points[leg] -> points[leg + 1]) of the route the mission
    sequencer and launch planner draw: a uniform Catmull-Rom spline with the
    end points repeated (tvc_env.envs.waypoints.catmull_rom, samplePlannerSpline)."""
    p = np.asarray(points, dtype=float)
    a, b = p[max(0, leg - 1)], p[leg]
    c, d = p[leg + 1], p[min(len(p) - 1, leg + 2)]
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
        row = 0

        def put(terms, b):
            nonlocal row
            for col, coef in terms:
                ri.append(row)
                ci.append(col)
                vals.append(coef)
            rhs.append(b)
            row += 1

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
                put([(c, -v) for c, v in terms], constant)
            cones.append(clarabel.SecondOrderConeT(len(rows)))
        for rows, alpha in self.blocks['pow']:
            for terms, constant in rows:
                put([(c, -v) for c, v in terms], constant)
            cones.append(clarabel.PowerConeT(alpha))
        A = sparse.csc_matrix((vals, (ri, ci)), shape=(row, n))
        return A, np.asarray(rhs, dtype=float), cones


class ConvexGuidance:
    """Free-final-time SOCP planner for landing and waypoint routes."""

    def __init__(self, limits: GuidanceLimits, energy: EnergyModel | None = None,
                 objective: str = 'energy', landing_nodes: int = 24, route_dt_s: float = 0.35,
                 max_leg_nodes: int = 40, flypass_capture_fraction: float = 0.5,
                 max_solve_time_s: float = 0.5, corridor_m: float | None = None,
                 corridor_weight: float = 10.0):
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

    # ---- public API ----

    def plan(self, position, velocity, thrust_now_n, gate: LandingGate,
             route: tuple[RouteWaypoint, ...] | list = (), landing_time_hint: float | None = None,
             path=None) -> GuidancePlan | None:
        """Plan from the current state through `route` to `gate`.

        `thrust_now_n` is the rotor's present thrust, a world-frame vector in
        N (a scalar is taken as vertical): the thrust profile starts there and
        changes at the bounded rate. `landing_time_hint` (the previous plan's
        remaining powered-descent time) narrows the line search. `path` is the
        drawn route, one (M, 3) curve per leg: path[i] leads to route[i] and
        path[len(route)] to the gate; with a corridor half-width set, the plan
        stays near it.
        """
        r0 = np.asarray(position, dtype=float)
        v0 = np.asarray(velocity, dtype=float)
        started = time.perf_counter()
        stats = dict(solves=0)
        route = tuple(route)
        curves = [None] * (len(route) + 1)
        if path is not None and self.corridor_m is not None:
            if len(path) != len(route) + 1:
                raise ValueError(f'path needs one curve per leg: {len(route) + 1}, got {len(path)}')
            curves = [None if c is None or _arc(np.asarray(c, dtype=float))[-1] < 1e-6
                      else np.asarray(c, dtype=float) for c in path]
        best = None
        for scale in (1.0, 1.5, 2.25):
            segments = self._route_segments(r0, v0, route, scale, curves)
            best = self._search_landing_time(r0, v0, thrust_now_n, gate, segments, landing_time_hint, stats,
                                             land_curve=curves[-1])
            if best is not None:
                break
        if best is not None and any(c is not None for c in curves):
            best = self._refine_corridor(r0, v0, thrust_now_n, gate, best, stats)
        if best is None:
            best = self._soft_terminal(r0, v0, thrust_now_n, gate, route, stats)
        if best is not None:
            best.solve_time_s = time.perf_counter() - started
            best.solves = stats['solves']
        return best

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

    def _route_segments(self, r0, v0, route, scale, curves=None) -> list[_Segment]:
        """Waypoint legs with durations from each leg's reference speed.

        Rest-to-rest hover legs also get acceleration and braking time. The
        planner retries with longer legs (scale) before giving up on a route.
        """
        curves = curves or [None] * (len(route) + 1)
        segments = []
        start, speed_in = r0, float(np.linalg.norm(v0))
        accel = self._lateral_accel()
        for i, wp in enumerate(route):
            target = np.asarray(wp.position, dtype=float)
            distance = float(np.linalg.norm(target - start))
            cruise = min(max(float(wp.speed_m_s), 0.3), self.limits.max_speed_m_s)
            if wp.kind == HOVER:
                # Rest to rest: trapezoidal speed profile, or bang-bang when
                # the hop is too short to reach the cruise speed.
                duration = (distance / cruise + cruise / accel if distance > cruise * cruise / accel
                            else 2.0 * math.sqrt(distance / accel))
            else:
                duration = distance / cruise
            if i == 0:
                duration += speed_in / accel
            duration = max(duration * scale, 1.0 if wp.kind == HOVER else 0.6)
            segments.append(_Segment(wp.kind, duration, self._nodes(duration), target,
                                     float(wp.radius_m), curve=curves[i]))
            if wp.kind == HOVER:
                hold = max(float(wp.hold_s), 0.0) + 0.5
                segments.append(_Segment('hold', hold, max(2, self._nodes(hold)), target))
            start = target
        return segments

    def _nodes(self, duration):
        return int(min(self.max_leg_nodes, max(3, math.ceil(duration / self.route_dt_s))))

    # ---- free final time (Algorithm 1) ----

    def _search_landing_time(self, r0, v0, thrust_now, gate, segments, hint, stats, land_curve=None):
        lim = self.limits
        start = segments[-1].target if segments else r0
        g = np.asarray(gate.position, dtype=float)
        distance = float(np.linalg.norm(g - start))
        speed_cap = max(lim.max_speed_m_s, float(np.linalg.norm(v0)))
        t_lo = max(0.8, 0.8 * distance / speed_cap)
        t_hi = max(3.0 * t_lo, distance / 0.75 + 6.0)
        cache = {}

        def cost(t_land):
            key = round(t_land, 4)
            if key not in cache:
                cache[key] = self._solve(r0, v0, thrust_now, gate, segments + [
                    _Segment('land', t_land, self.landing_nodes, curve=land_curve)], stats)
            plan = cache[key]
            return math.inf if plan is None else plan.cost

        if hint is not None and hint > 0.3:
            candidates = [max(0.5, hint * f) for f in (0.85, 1.0, 1.2)]
        else:
            candidates = list(np.geomspace(t_lo, t_hi, 8))
        values = [cost(t) for t in candidates]
        if not any(math.isfinite(v) for v in values) and hint is not None:
            candidates = list(np.geomspace(t_lo, t_hi, 8))
            values = [cost(t) for t in candidates]
        if not any(math.isfinite(v) for v in values):
            return None
        j = int(np.argmin(values))
        # Golden-section refinement in the bracket around the best sample.
        a = candidates[j - 1] if j > 0 else candidates[j] * 0.8
        b = candidates[j + 1] if j + 1 < len(candidates) else candidates[j] * 1.25
        c, d = b - _GOLDEN * (b - a), a + _GOLDEN * (b - a)
        for _ in range(6):
            fc, fd = cost(c), cost(d)
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
        if soft_weight is None and self.corridor_m is not None:
            node = 0
            for seg in segments:
                first, node = node, node + seg.nodes
                if seg.curve is None:
                    continue
                for j, k in enumerate(range(first + 1, node + 1)):
                    if seg.along is None:
                        corridor.append((k, seg.curve[0], seg.curve[-1]))
                    else:
                        lo, _ = curve_point(seg.curve, seg.along[j] - CORRIDOR_WINDOW)
                        hi, _ = curve_point(seg.curve, seg.along[j] + CORRIDOR_WINDOW)
                        corridor.append((k, lo, hi))
        off_w = off_e + (3 if soft_weight is not None else 0)
        n_var = off_w + 2 * len(corridor)      # per corridor node: excess, chord parameter
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
        for seg in segments:
            first, node = node, node + seg.nodes
            if seg.kind == 'land':
                land_first = first
                continue
            if soft_weight is None:
                for k in range(first + 1, node + 1):
                    cones.le([(X(k, 2), -1.0)], -z_floor)
            if seg.kind == FLYPASS:
                waypoint_nodes.append(node)
                ball = self.flypass_capture_fraction * seg.radius
                if soft_weight is None:
                    cones.soc([([], ball)] + [([(X(node, i), 1.0)], -seg.target[i]) for i in range(3)])
            elif seg.kind == HOVER:
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
        for j, (k, a, b) in enumerate(corridor):
            excess, lam = off_w + 2 * j, off_w + 2 * j + 1
            cones.le([(excess, -1.0)], 0.0)
            cones.le([(lam, -1.0)], 0.0)
            cones.le([(lam, 1.0)], 1.0)
            cones.soc([([(excess, 1.0)], self.corridor_m)]
                      + [([(X(k, i), 1.0), (lam, -float(b[i] - a[i]))], -float(a[i])) for i in range(3)])
            q[excess] = self.corridor_weight * hover_cost * weights[k]

        A, b, cone_list = cones.assemble(n_var)
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = 200
        settings.time_limit = self.max_solve_time_s
        solution = clarabel.DefaultSolver(sparse.csc_matrix((n_var, n_var)), q, A, b,
                                          cone_list, settings).solve()
        stats['solves'] += 1
        status = str(solution.status)
        if status not in _OK:
            return None
        x = np.asarray(solution.x)
        states = x[:off_u].reshape(n_int + 1, 6)
        controls = x[off_u:off_c].reshape(n_int + 1, 4)
        u, sigma = controls[:, :3], controls[:, 3]
        plan = self._package(times, states, u, sigma, dts, segments, gate_r, mode=(
            'optimal' if soft_weight is None else 'soft_terminal'), status=status,
            iterations=int(solution.iterations), waypoint_nodes=waypoint_nodes,
            soft=None if soft_weight is None else float(q[off_e:off_w] @ x[off_e:off_w]))
        if corridor:
            plan.corridor_excess_m = float(max(x[off_w::2].max(), 0.0))
            # The line search ranks durations by cost: include the corridor
            # penalty, in Wh (the objective's sigma^1.5 s is P_ref (m/T_ref)^1.5 J).
            penalty = float(q[off_w:] @ x[off_w:])
            if self.objective == 'energy':
                e = self.energy
                penalty *= e.power_ref_w * (m / e.thrust_ref_n) ** 1.5 / 3600.0
            plan.cost += penalty
        return plan

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
        return plan if refined is None else refined

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
        for t_land in (2.0, 4.0, 8.0, 14.0):
            plan = self._solve(r0, v0, thrust_now, target, [_Segment('land', t_land, self.landing_nodes)],
                               stats, soft_weight=1.0e3,
                               velocity_weight=1.0e3 if not route or route[0].kind == HOVER else 50.0)
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
