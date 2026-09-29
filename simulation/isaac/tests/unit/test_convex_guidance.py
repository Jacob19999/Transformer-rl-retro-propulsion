"""Convex (SOCP) powered-descent guidance and its tracking controller."""
import math

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip('clarabel')

from tvc_env.controllers.convex_adapter import ConvexGuidanceController, VehicleModel, _FRD_TO_ISAAC  # noqa: E402
from tvc_env.controllers.convex_guidance import (  # noqa: E402
    ConvexGuidance, EnergyModel, GuidanceLimits, LandingGate, ObjectiveWeights, RouteWaypoint, _Segment, catmull_rom_leg,
    curve_distance, curve_fraction, curve_point,
)
from tvc_env.controllers.pid_fin_mixer import PIDFinMixer  # noqa: E402
from tvc_env.common.quaternions import from_euler, to_euler, to_rotation_matrix  # noqa: E402

ROOT = __import__('pathlib').Path(__file__).resolve().parents[2]
MASS, G = 3.104, 9.81
WEIGHT = MASS * G
FULL = WEIGHT / 0.81 ** 2          # net thrust at full rotor speed for a 0.81 hover duty
RATE = 2 * math.sqrt(0.6 * WEIGHT * FULL) * 0.25
GATE = LandingGate(position=(0., 0., 0.8125), velocity=(0., 0., -0.15), apex=(0., 0., 0.3125))


def planner(objective='energy'):
    limits = GuidanceLimits(mass_kg=MASS, thrust_min_n=0.6 * WEIGHT, thrust_max_n=0.85 * FULL,
                            thrust_rate_n_s=RATE, max_tilt_rad=math.radians(15), max_speed_m_s=4.,
                            glide_slope_rad=math.radians(70))
    return ConvexGuidance(limits, EnergyModel(FULL, 3072 / .88, 10.), objective)


def settings(law=None):
    s = yaml.safe_load((ROOT / 'configs/controllers/convex_guidance.yaml').read_text())
    if law is not None:
        s['attitude']['law'] = law
    return s


def bare_vehicle():
    """Planned 8S vehicle without an attitude model (enough for the PD law)."""
    return VehicleModel(mass_kg=MASS, hover_throttle=.81, reference_voltage_v=29.6,
                        shaft_power_at_max_w=3072., motor_efficiency=.88, auxiliary_power_w=10.)


# Attitude model of the _fly test plant: effort torques (20, 20, 1.1) N m/rad
# at full rotor, the 0.27 N m s/rad damper, a 0.05 s servo, no rotor gyro.
FLY_ATTITUDE = dict(vane_authority_nm_per_rad=20., yaw_authority_nm_per_rad=1.1, inertia_rp_kg_m2=.05,
                    inertia_yaw_kg_m2=.02, angular_damping_nm_s_per_rad=.27, servo_lag_s=.05)


def vehicle():
    return VehicleModel(**{**bare_vehicle().__dict__, **FLY_ATTITUDE})


def test_landing_plan_meets_every_constraint_and_is_lossless():
    guidance = planner()
    plan = guidance.plan([-.28, .82, 18.], [0., 0., -1.], WEIGHT, GATE)
    lim = guidance.limits
    assert plan.mode == 'optimal'
    np.testing.assert_allclose(plan.position[-1], GATE.position, atol=1e-5)
    np.testing.assert_allclose(plan.velocity[-1], GATE.velocity, atol=1e-5)
    thrust = plan.sigma * MASS
    assert thrust.min() >= lim.thrust_min_n - 1e-3 and thrust.max() <= lim.thrust_max_n + 1e-3
    norm = np.linalg.norm(plan.thrust_accel, axis=1)
    # Lemma 1 (lossless convexification): the slack is tight at the optimum.
    assert plan.convexification_gap < 5e-3
    np.testing.assert_allclose(norm, plan.sigma, rtol=5e-3)
    tilt = np.arccos(np.clip(plan.thrust_accel[:, 2] / norm, -1, 1))
    assert tilt.max() <= lim.max_tilt_rad + 1e-4
    assert np.linalg.norm(plan.velocity, axis=1).max() <= lim.max_speed_m_s + 1e-4
    # The rate bound holds for the executed thrust vector, not just the slack,
    # and the thrust starts at the present thrust (continuous, first-order hold).
    dt = np.diff(plan.times)
    change = np.linalg.norm(np.diff(plan.thrust_accel, axis=0), axis=1) * MASS
    assert (change / dt).max() <= lim.thrust_rate_n_s * (1 + 1e-3)
    np.testing.assert_allclose(plan.thrust_accel[0], [0., 0., G], atol=1e-5)
    horizontal = np.linalg.norm(plan.position[1:, :2], axis=1)
    height = plan.position[1:, 2] - GATE.apex[2]
    assert np.all(horizontal <= math.tan(lim.glide_slope_rad) * height + 1e-4)
    # Upright, steady hand-over to the terminal descent.
    np.testing.assert_allclose(plan.thrust_accel[-1], [0., 0., G], atol=1e-5)
    # Exact first-order-hold sampling reproduces the nodes, from either side.
    r, v, u = plan.sample(float(plan.times[5]))
    np.testing.assert_allclose(r, plan.position[5], atol=1e-6)
    np.testing.assert_allclose(v, plan.velocity[5], atol=1e-6)
    np.testing.assert_allclose(u, plan.thrust_accel[5], atol=1e-9)
    r, v, u = plan.sample(float(plan.times[6]) - 1e-9)
    np.testing.assert_allclose(r, plan.position[6], atol=1e-6)
    np.testing.assert_allclose(v, plan.velocity[6], atol=1e-6)
    np.testing.assert_allclose(u, plan.thrust_accel[6], atol=1e-6)


def test_free_final_time_search_finds_the_cheapest_duration():
    guidance = planner()
    plan = guidance.plan([2., -1., 12.], [.5, 0., -2.], WEIGHT, GATE)
    stats = dict(solves=0)
    for factor in (.9, 1.1):
        other = guidance._solve(np.array([2., -1., 12.]), np.array([.5, 0., -2.]), WEIGHT, GATE,
                                [_Segment('land', plan.duration * factor, guidance.landing_nodes)], stats)
        assert other is None or other.cost >= plan.cost - 1e-6


def test_route_plan_captures_flypass_and_holds_hover_before_landing():
    route = (RouteWaypoint((6., 2., 6.), 'flypass', 1., 3.), RouteWaypoint((10., -4., 4.), 'hover', 1., 2., 2.))
    plan = planner().plan([0., 0., 3.], [0., 0., 0.], WEIGHT, GATE, route=route)
    assert plan.mode == 'optimal'
    flypass, hover = plan.waypoint_nodes
    assert np.linalg.norm(plan.position[flypass] - route[0].position) <= .5 + 1e-4
    np.testing.assert_allclose(plan.position[hover], route[1].position, atol=1e-5)
    held = (plan.times >= plan.times[hover]) & (plan.times <= plan.times[hover] + 2.)
    np.testing.assert_allclose(plan.velocity[held], 0., atol=1e-5)
    # Hover thrust throughout: zero velocity at the nodes alone let the
    # first-order-hold thrust alternate node to node (mission f1b3c7b587da).
    np.testing.assert_allclose(plan.thrust_accel[held], np.tile([0., 0., G], (held.sum(), 1)), atol=1e-5)
    np.testing.assert_allclose(plan.position[held], np.tile(route[1].position, (held.sum(), 1)), atol=1e-5)
    assert plan.landing_start_s >= plan.times[hover] + 2.
    assert plan.position[:, 2].min() >= .8 - 1e-4


def test_replans_at_the_speed_bound_do_not_ratchet_it_up():
    guidance = planner()
    V = guidance.limits.max_speed_m_s
    r, v, thrust = np.array([0., -35., 20.]), np.array([0., 3.9, -1.]), np.array([0., 0., WEIGHT])
    peaks = []
    for _ in range(8):   # re-plan every 0.5 s from the plan's own reference, as the adapter does
        plan = guidance.plan(r, v, thrust, GATE)
        assert plan.mode == 'optimal'
        peaks.append(np.linalg.norm(plan.velocity, axis=1).max())
        r, v, u = plan.sample(.5)
        thrust = u * MASS
    # Mission 0289417f5d60: +0.25 m/s per re-plan, 4.0 -> 5.9 m/s.
    assert max(peaks) <= V + .15
    # A start above the bound keeps its speed, then brakes back under it.
    plan = guidance.plan([0., -35., 20.], [0., 4.6, 0.], WEIGHT, GATE)
    speed = np.linalg.norm(plan.velocity, axis=1)
    assert plan.mode == 'optimal' and speed[0] > V
    assert np.all(speed[plan.times >= 2.] <= V + 1e-4)


def test_widened_glide_slope_narrows_to_nominal_before_the_gate():
    limits = GuidanceLimits(**{**planner().limits.__dict__, 'glide_slope_rad': math.radians(45.)})
    guidance = ConvexGuidance(limits, EnergyModel(FULL, 3072 / .88, 10.))
    start = np.array([0., -30., 18.6])   # the last hover waypoint of mission 0289417f5d60
    plan = guidance.plan(start, [0., 0., 0.], WEIGHT, GATE)
    assert plan.mode == 'optimal'
    horizontal = np.linalg.norm(plan.position[:, :2], axis=1)
    height = plan.position[:, 2] - GATE.apex[2]
    widened = math.tan(math.atan2(30., 18.6 - GATE.apex[2]) + math.radians(2.))
    assert np.all(horizontal[1:] <= widened * height[1:] + 1e-4)
    final = plan.times >= plan.duration - limits.glide_slope_final_s
    assert np.all(horizontal[final] <= height[final] + 1e-4)   # tan 45 deg


def test_landing_leg_reaches_the_gate_from_above():
    # Mission 614cb15ab14a at t = 93.9 s: fast and low on the final approach;
    # its plan sank to 0.52 m, below the 0.81 m gate, and climbed back.
    plan = planner().plan([-.08, -2.12, 2.48], [-.25, 2.97, -2.21], [0., 0., WEIGHT], GATE)
    assert plan.mode == 'optimal'
    assert plan.position[:, 2].min() >= GATE.position[2] - 1e-4
    # ... and between the nodes (exact first-order-hold sampling).
    heights = [plan.sample(t)[0][2] for t in np.linspace(0., plan.duration, 400)]
    assert min(heights) >= GATE.position[2] - 2e-3


def test_unreachable_gate_degrades_to_maximum_braking():
    guidance = planner()
    guidance.limits = GuidanceLimits(**{**guidance.limits.__dict__, 'emergency_thrust_max_n': FULL,
                                        'emergency_thrust_rate_n_s': 8 * RATE})
    plan = guidance.plan([0., 0., 3.], [0., 0., -6.], WEIGHT, GATE)
    assert plan.mode == 'soft_terminal'
    assert np.all(np.isfinite(plan.position)) and np.all(np.isfinite(plan.thrust_accel))
    early = plan.times < .5
    assert plan.sigma[early].max() * MASS > .95 * FULL   # full braking at once, past the 85% ceiling


def _corridor_planner():
    base = planner()
    limits = GuidanceLimits(**{**base.limits.__dict__, 'emergency_thrust_max_n': FULL,
                               'emergency_thrust_rate_n_s': 8 * RATE})
    return ConvexGuidance(limits, base.energy, corridor_m=1.)


def _off_route(plan):
    """Largest distance of a plan's route-leg nodes from their drawn curves."""
    worst, node = 0., 0
    for seg in plan.segments:
        first, node = node, node + seg.nodes
        for r in plan.position[first + 1:node + 1] if seg.curve is not None else ():
            point, _ = curve_point(seg.curve, curve_fraction(seg.curve, r)[0])
            worst = max(worst, float(np.linalg.norm(r - point)))
    return worst


def test_route_corridor_keeps_the_plan_on_the_drawn_route():
    start = [0., 0., 20.]
    route = (RouteWaypoint((20., 0., 15.), 'flypass', 1., 3.), RouteWaypoint((20., 20., 10.), 'flypass', 1., 3.))
    points = [start] + [w.position for w in route] + [[0., 0., 0.]]
    path = [catmull_rom_leg(points, leg) for leg in range(3)]
    guidance = _corridor_planner()
    on = guidance.plan(start, [0., 0., 0.], WEIGHT, GATE, route, path=path)
    assert on.mode == 'optimal' and on.corridor_excess_m < .05
    assert _off_route(on) <= 1.05
    # Without it the plan only meets the waypoints: 2.6 m off the curve at the turn.
    free = guidance.plan(start, [0., 0., 0.], WEIGHT, GATE, route)
    free.segments = guidance._route_segments(np.array(start), np.zeros(3), route, 1., path) + free.segments[2:]
    assert _off_route(free) > 2.


def test_route_corridor_brakes_a_fast_descent_at_its_waypoint():
    # Isaac 835c3de32185: from a 20 m/s descent the plan sank 26 m below its
    # fly-through and climbed back. The corridor caps the leg at the
    # waypoint, and the plan may then brake at full thrust.
    guidance = _corridor_planner()
    start, velocity = [4., 0., 40.], [0., 0., -12.]
    route = (RouteWaypoint((0., 0., 18.), 'flypass', 1., 3.),)
    path = [catmull_rom_leg([start, route[0].position, [0., 0., 0.]], leg) for leg in range(2)]
    free = guidance.plan(start, velocity, [0., 0., WEIGHT], GATE, route)
    on = guidance.plan(start, velocity, [0., 0., WEIGHT], GATE, route, path=path)
    assert free.mode == on.mode == 'optimal'
    before = lambda plan: plan.position[:plan.waypoint_nodes[0] + 1, 2].min()  # noqa: E731
    assert before(free) < 1.        # the energy optimum rides down to the 0.8 m floor
    assert before(on) > 16.         # stops within ~2 m of the 18 m waypoint
    ceiling = guidance.limits.thrust_max_n
    assert free.sigma.max() * MASS <= ceiling + 1e-3 < on.sigma.max() * MASS
    np.testing.assert_array_less(on.sigma[on.times > 8.] * MASS, ceiling + 1e-3)   # reserve back after braking


def test_route_corridor_is_soft_for_a_start_it_cannot_contain():
    guidance = _corridor_planner()
    start = [0., 0., 12.]
    route = (RouteWaypoint((10., 0., 10.), 'flypass', 1., 3.),)
    path = [catmull_rom_leg([start, route[0].position, [0., 0., 0.]], leg) for leg in range(2)]
    plan = guidance.plan(start, [0., 3.5, 0.], [0., 0., WEIGHT], GATE, route, path=path)
    assert plan.mode == 'optimal' and plan.corridor_excess_m > .2


def test_corridor_curves_match_the_mission_sequencer_and_never_undershoot_a_leg():
    from tvc_env.envs.waypoints import catmull_rom
    points = [[-1.2, .8, 50.4], [-.5, 23.6, 6.], [7.3, -1.4, 4.5], [-.1, -29.9, 18.6], [0., 0., 0.]]
    p = torch.tensor(points, dtype=torch.float64)
    for leg in range(4):
        # WaypointMission._refresh_curves: p0 = start for the first two legs, p3 clamps to the pad.
        ends = [p[max(0, leg - 1)], p[leg], p[leg + 1], p[min(4, leg + 2)]]
        expected = catmull_rom(*(e[None] for e in ends))[0].numpy()
        np.testing.assert_allclose(catmull_rom_leg(points, leg), expected, atol=1e-9)
    # The trial route's hover-to-hover leg (6 m -> 4.5 m) dips to 1.5 m on the
    # drawn curve; the controller's corridor stays at or above 4.5 m.
    assert catmull_rom_leg(points, 1)[:, 2].min() < 2.
    controller = ConvexGuidanceController(settings(), vehicle(), (0., 0., 0.), .3125, 1 / 30)
    curves = controller._path(points, 1, 2)
    assert len(curves) == 3 and curves[0][:, 2].min() >= 4.5 - 1e-9
    np.testing.assert_allclose(curves[0][[0, -1]], np.array(points[1:3]), atol=1e-9)
    direct = controller._path(points[:1] + points[-1:], 0, 0)  # explicit landing-only route
    np.testing.assert_allclose(direct[0], np.linspace(points[0], points[-1], 49))


def test_geometric_attitude_efforts_match_pid_euler_convention():
    s = settings('pd')
    controller = ConvexGuidanceController(s, bare_vehicle(), (0., 0., 0.), .3125, 1 / 30)
    kp, kd, comp = (s['attitude'][k] for k in ('kp_att', 'kd_att', 'gyro_comp_rp'))
    mixer = PIDFinMixer(max_fin_angle=controller._vane_limit)
    for roll, pitch in ((.05, 0.), (0., .05), (-.04, .03)):
        q = from_euler(torch.tensor(roll), torch.tensor(pitch), torch.tensor(0.))
        R = to_rotation_matrix(q.double()).numpy()
        fins = controller._attitude_fins(R, np.array([0., 0., 1.]), np.zeros(3), .88)
        # PIDController: FRD pitch is the negated Isaac Euler pitch; level target.
        r_isaac, p_isaac, _ = to_euler(q)
        expected = mixer.mix(torch.tensor([kp * -float(r_isaac)]), torch.tensor([kp * float(p_isaac)]),
                             torch.tensor([0.]))[0]
        # Equal to first order; SO(3) and Euler errors differ by O(angle^2).
        np.testing.assert_allclose(fins, expected.numpy(), atol=1e-3)
        assert np.all(np.sign(fins) == np.sign(expected.numpy()))
    rates = np.array([.3, -.2, .5])
    fins = controller._attitude_fins(np.eye(3), np.array([0., 0., 1.]), rates, .88)
    expected = mixer.mix(torch.tensor([-kd * .3 + comp * -.2]), torch.tensor([-kd * -.2 - comp * .3]),
                         torch.tensor([-s['attitude']['yaw_damper_gain'] * .5]))[0]
    np.testing.assert_allclose(fins, expected.numpy(), atol=1e-6)


def test_gyroscopic_cross_feed_cancels_the_identified_rotor_coupling():
    s = settings('pd')
    s['attitude']['vane_rp_authority_nm_per_rad'] = authority = 20.7   # identified value, opt-in
    with_rotor = VehicleModel(**{**bare_vehicle().__dict__, 'rotor_inertia': 2e-4, 'omega_max': 4649.56})
    controller = ConvexGuidanceController(s, with_rotor, (0., 0., 0.), .3125, 1 / 30)
    for f in (.6, .88, 1.):
        k = controller._gyro_compensation(f)
        # Cross-feed torque K f^2 k q equals the gyroscopic torque H q, H = I_r omega_max f.
        assert authority * f * f * k == pytest.approx(2e-4 * 4649.56 * f)
    assert controller._gyro_compensation(.88) == pytest.approx(.051, abs=.002)
    # Without rotor parameters, or by default, the fixed PID cross-feed is used.
    assert ConvexGuidanceController(s, bare_vehicle(), (0., 0., 0.), .3125, 1 / 30)._gyro_compensation(.88) == s['attitude']['gyro_comp_rp']
    default = ConvexGuidanceController(settings('pd'), with_rotor, (0., 0., 0.), .3125, 1 / 30)
    assert default._gyro_compensation(.88) == settings()['attitude']['gyro_comp_rp']


def test_thrust_margins_scale_with_thrust_to_weight():
    s = settings()
    controller = ConvexGuidanceController(s, vehicle(), (0., 0., 0.), .3125, 1 / 30)
    for available in (42.78, 34.91):          # planned 8S and legacy 6S at the reference voltage
        ceiling = controller._planning_ceiling(available)
        floor = controller._planning_floor(ceiling)
        assert WEIGHT < ceiling < available
        # Half the excess thrust stays in reserve for tracking ...
        assert available - ceiling == pytest.approx(ceiling - WEIGHT)
        # ... and the plan never falls faster than it can brake.
        assert WEIGHT - floor == pytest.approx(ceiling - WEIGHT)
        assert floor >= s['guidance']['thrust_min_weight_fraction'] * WEIGHT
    with pytest.raises(ValueError, match='thrust above weight'):
        controller._planning_ceiling(1.01 * WEIGHT, strict=True)
    assert controller._planning_ceiling(0.9 * WEIGHT) > WEIGHT   # in flight: degrade, never raise


def test_throttle_inverts_the_duty_voltage_thrust_model():
    v = vehicle()
    for thrust in (.5 * WEIGHT, WEIGHT, 1.3 * WEIGHT):
        for volts in (27., 29.6, 33.6):
            duty = v.throttle_for_thrust(thrust, volts)
            assert v.thrust_from_rotor(duty * volts / v.reference_voltage_v) == pytest.approx(thrust)
    assert v.available_thrust_n(33.6) == pytest.approx(FULL)
    assert v.available_thrust_n(26.64) == pytest.approx(FULL * .81)


def _quaternion(R):
    w = math.sqrt(max(0., 1 + np.trace(R))) / 2
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])


def _fly(controller, r0, v0, rotor, duration=25., volts=33.):
    """Point-mass + rigid-body plant with rotor/servo lag and vane torque."""
    from scipy.spatial.transform import Rotation
    inertia, dt = np.diag([.05, .05, .02]), 1 / 120
    r, v, R, w = np.array(r0, float), np.array(v0, float), np.eye(3), np.zeros(3)
    fins, t, commands = np.zeros(4), 0., []
    while t < duration:
        obs = np.concatenate([-r, _quaternion(R), _FRD_TO_ISAAC.T @ (R.T @ v), _FRD_TO_ISAAC.T @ w, [r[2]],
                              fins, np.zeros(4), [rotor], [0.]])
        action = controller.compute_action(torch.tensor(obs[None], dtype=torch.float32), bus_voltage_v=volts)[0].numpy()
        commands.append(action.copy())
        for _ in range(4):
            fins = np.clip(fins + np.clip((action[:4] - fins) / .05, -7, 7) * dt, -.262, .262)
            rotor = min(1., max(0., rotor + (min(1., action[4] * volts / 29.6) - rotor) / .15 * dt))
            effort = np.array([(fins[2] - fins[0]) / 2, (fins[3] - fins[1]) / 2, fins.mean()])
            torque = _FRD_TO_ISAAC @ (np.array([20., 20., 1.1]) * effort * (rotor / .81) ** 2) - .27 * w
            w = w + np.linalg.solve(inertia, torque - np.cross(w, inertia @ w)) * dt
            R = R @ Rotation.from_rotvec(w * dt).as_matrix()
            v = v + (FULL * rotor ** 2 / MASS * R[:, 2] + [0., 0., -G]) * dt
            r = r + v * dt
            t += dt
            if r[2] <= .3125:
                return dict(impact=-v[2], pad=float(np.linalg.norm(r[:2])), t=t), np.array(commands)
    return None, np.array(commands)


def test_closed_loop_landing_is_soft_on_the_pad_with_bounded_commands():
    s = settings()
    controller = ConvexGuidanceController(s, vehicle(), (0., 0., 0.), .3125, 1 / 30)
    touchdown, commands = _fly(controller, [1.5, -1., 10.], [0., .5, -1.], rotor=.84)
    assert touchdown is not None, 'no touchdown'
    assert touchdown['impact'] <= .25 and touchdown['pad'] <= .2
    assert np.abs(commands[:, :4]).max() <= s['attitude']['max_fin_angle'] + 1e-6
    assert commands[:, 4].min() >= 0. and commands[:, 4].max() <= 1.
    slew = np.abs(np.diff(commands[:, 4])).max()
    assert slew <= s['guidance']['throttle_rate_per_s'] / 30 + 1e-6   # yaw-safe duty slew outside spool-up
    assert controller.phase == 'TERMINAL_DESCENT'


def _servo(deadband=.017):
    from tvc_env.dynamics.actuator_servo import ServoModel
    return ServoModel(tau_servo=.05, max_angular_velocity=6.98, max_command_angle=.262, deadband=deadband)


def test_vane_deadband_inverse_lands_the_servo_on_the_desired_angle():
    controller = ConvexGuidanceController(settings(), vehicle(), (0., 0., 0.), .3125, 1 / 30, servo_deadband_rad=.017)
    servo = _servo()
    for start, desired in ((0., .03), (.03, -.01), (-.05, -.035), (.1, .105)):
        plain = compensated = torch.full((1, 4), start)
        for _ in range(15):   # 0.5 s of control steps, four servo substeps each
            command = controller._deadband_inverse(np.full(4, desired), compensated[0].double().numpy())
            for _ in range(4):
                plain = servo.update(plain, torch.full((1, 4), desired), 1 / 120)
                compensated = servo.update(compensated, torch.tensor(command[None], dtype=torch.float32), 1 / 120)
        # Commanded directly, the servo stalls a deadband short (or never moves) ...
        if abs(desired - start) > .017:
            assert abs(float(plain[0, 0]) - desired) == pytest.approx(.017, abs=.003)
        # ... pre-compensated, it lands on the desired angle.
        assert abs(float(compensated[0, 0]) - desired) < .004
    # A vane already on its angle is left alone, and commands stay inside the servo range.
    np.testing.assert_allclose(controller._deadband_inverse(np.full(4, .02), np.full(4, .02)), .02)
    assert controller._deadband_inverse(np.full(4, .26), np.zeros(4)).max() <= .262
    # Without a servo deadband (or with the compensation disabled) commands pass through.
    plain = ConvexGuidanceController(settings(), vehicle(), (0., 0., 0.), .3125, 1 / 30)
    np.testing.assert_allclose(plain._deadband_inverse(np.full(4, .03), np.zeros(4)), .03)


def _coning_rate_rms(controller, deadband, duration=6.):
    """Roll/pitch rate RMS (deg/s) while descending from rest at 8 m.

    Plant as _fly, plus what sustains the coning seen in Isaac: the
    tvc_env ServoModel (lag, slew, error deadband), the ~25 ms PhysX vane-joint
    lag, the identified vane authority 20.7 N m/rad f^2 and the rotor's
    gyroscopic torque -(w x H), H = I_rotor omega_max f.
    """
    from scipy.spatial.transform import Rotation
    servo, inertia, dt = _servo(deadband), np.diag([.05, .05, .02]), 1 / 120
    r, v, w = np.array([0., 0., 8.]), np.zeros(3), np.zeros(3)
    R = Rotation.from_euler('xy', [2., 1.], degrees=True).as_matrix()
    target, fins, rotor, t, rates = torch.zeros(1, 4), np.zeros(4), .81, 0., []
    while t < duration:
        obs = np.concatenate([-r, _quaternion(R), _FRD_TO_ISAAC.T @ (R.T @ v), _FRD_TO_ISAAC.T @ w, [r[2]],
                              fins, np.zeros(4), [rotor], [0.]])
        action = controller.compute_action(torch.tensor(obs[None], dtype=torch.float32), bus_voltage_v=29.6)[0].numpy()
        for _ in range(4):
            target = servo.update(target, torch.tensor(action[None, :4]), dt)
            fins = fins + np.clip((target[0].double().numpy() - fins) / .025, -6.98, 6.98) * dt
            rotor = min(1., max(0., rotor + (min(1., float(action[4])) - rotor) / .15 * dt))
            effort = np.array([(fins[2] - fins[0]) / 2, (fins[3] - fins[1]) / 2, fins.mean()])
            w_frd = _FRD_TO_ISAAC.T @ w
            torque_frd = (np.array([20.7, 20.7, 1.1]) * effort * rotor ** 2
                          - np.cross(w_frd, [0., 0., 2e-4 * 4649.56 * rotor]))
            torque = _FRD_TO_ISAAC @ torque_frd - .27 * w
            w = w + np.linalg.solve(inertia, torque - np.cross(w, inertia @ w)) * dt
            R = R @ Rotation.from_rotvec(w * dt).as_matrix()
            v = v + (FULL * rotor ** 2 / MASS * R[:, 2] + [0., 0., -G]) * dt
            r = r + v * dt
            t += dt
        if t >= 2.:
            rates.append(np.degrees(_FRD_TO_ISAAC.T @ w)[:2])
    return float(np.sqrt(np.mean(np.square(rates))))


def test_vane_deadband_inverse_removes_the_gyroscopic_coning_limit_cycle():
    # The PD law on the legacy vanes (Isaac mission 0289417f5d60).
    with_rotor = VehicleModel(**{**bare_vehicle().__dict__, 'rotor_inertia': 2e-4, 'omega_max': 4649.56})

    def controller(deadband_rad):
        return ConvexGuidanceController(settings('pd'), with_rotor, (0., 0., 0.), .3125, 1 / 30,
                                        servo_deadband_rad=deadband_rad)

    # Isaac mission 0289417f5d60 coned at ~15 deg/s RMS with the 1 deg servo deadband.
    coning = _coning_rate_rms(controller(0.), .017)
    assert coning > 8.
    assert _coning_rate_rms(controller(.017), .017) < 2.
    # Over-compensating (no real deadband) only dithers, far below the coning it removes.
    assert _coning_rate_rms(controller(.017), 0.) < .5 * coning


LEGACY_VANES, MOMENTUM_VANES = (20.04, 14.58), (2.365, 1.72)   # probed N m/rad at full rotor (planned 8S)


def _with_vanes(authority, damping=0.):
    """Planned 8S vehicle with these vanes, the rotor, and the Isaac attitude model (servo + 25 ms joint lag)."""
    return VehicleModel(**{**bare_vehicle().__dict__, 'rotor_inertia': 2e-4, 'omega_max': 4649.56,
                           'vane_authority_nm_per_rad': authority[0], 'yaw_authority_nm_per_rad': authority[1],
                           'inertia_rp_kg_m2': .05, 'inertia_yaw_kg_m2': .02, 'servo_lag_s': .05,
                           'vane_joint_lag_s': .025, 'angular_damping_nm_s_per_rad': damping})


def test_vane_efforts_scale_to_the_plant_authority():
    s = settings('pd')
    legacy = ConvexGuidanceController(s, _with_vanes(LEGACY_VANES), (0., 0., 0.), .3125, 1 / 30)
    jet = ConvexGuidanceController(s, _with_vanes(MOMENTUM_VANES), (0., 0., 0.), .3125, 1 / 30)
    R = to_rotation_matrix(from_euler(torch.tensor(.01), torch.tensor(-.006), torch.tensor(0.)).double()).numpy()
    rates = np.array([.02, -.01, .05])
    small = np.array(legacy._attitude_fins(R, np.array([0., 0., 1.]), rates, .84))
    # The gains were tuned on the legacy vanes: identical there, 8.47x the deflection on the coupled jet.
    np.testing.assert_allclose(small, ConvexGuidanceController(s, bare_vehicle(), (0., 0., 0.), .3125, 1 / 30)
                               ._attitude_fins(R, np.array([0., 0., 1.]), rates, .84), atol=1e-9)
    # (Roll/pitch scale by 20.04/2.365, the yaw common mode by 14.58/1.72: 8.474 vs 8.477.)
    np.testing.assert_allclose(jet._attitude_fins(R, np.array([0., 0., 1.]), rates, .84),
                               small * 20.04 / 2.365, rtol=1e-3)
    # A large error becomes a slew at the rate the vanes sustain against the rotor, not saturated vanes.
    R = to_rotation_matrix(from_euler(torch.tensor(.2), torch.tensor(0.), torch.tensor(0.)).double()).numpy()
    fins = np.array(jet._attitude_fins(R, np.array([0., 0., 1.]), np.zeros(3), .84))
    limit = s['attitude']['max_fin_angle']
    slew = .6 * 2.365 * .81 * .81 * limit / (2e-4 * 4649.56 * .81)
    assert jet._slew_rate == pytest.approx(slew)
    assert np.abs(fins).max() == pytest.approx(s['attitude']['kd_att'] * slew * 20.04 / 2.365, rel=1e-6)
    assert np.abs(fins).max() < limit
    assert legacy._slew_rate > math.radians(120.)          # inactive on the legacy vanes


def test_duty_slew_and_yaw_reserve_follow_the_yaw_authority():
    s = settings()
    legacy = ConvexGuidanceController(s, _with_vanes(LEGACY_VANES, damping=.27), (0., 0., 0.), .3125, 1 / 30)
    jet = ConvexGuidanceController(s, _with_vanes(MOMENTUM_VANES), (0., 0., 0.), .3125, 1 / 30)
    limit = s['attitude']['max_fin_angle']
    reaction = 2e-4 * 4649.56 * jet._duty_rate(29.6)          # N m of rotor reaction at the slew
    assert reaction == pytest.approx(.4 * 1.72 * .81 * .81 * limit)
    # The yaw LQR is designed around its design torque on any vanes, so the
    # legacy vanes' authority buys no faster slew (their 0.25 /s wobbled the
    # legacy replica at 13-23 deg/s); the PD law keeps 0.25 /s there.
    assert 2e-4 * 4649.56 * legacy._duty_rate(29.6) == pytest.approx(.4 * s['attitude']['lqr']['yaw_design_torque_nm'])
    pd = ConvexGuidanceController(settings('pd'), _with_vanes(LEGACY_VANES), (0., 0., 0.), .3125, 1 / 30)
    assert pd._duty_rate(29.6) == s['guidance']['throttle_rate_per_s']
    assert jet._duty_rate(33.6) < jet._duty_rate(29.6)      # a full pack spins the rotor faster per duty
    assert jet._yaw_reserve == pytest.approx(.4 * limit)
    assert legacy._yaw_reserve < math.radians(1.5)
    # The plan's tilt rate: ~12 deg/s on the coupled jet, inactive (~100 deg/s) on legacy vanes.
    assert math.degrees(jet._limits.tilt_rate_rad_s) == pytest.approx(
        math.degrees(.4 * 2.365 * .81 * limit / (2e-4 * 4649.56)), rel=1e-6)
    assert math.degrees(legacy._limits.tilt_rate_rad_s) > 80.


def test_planned_tilt_rate_respects_the_attitude_slew_bound():
    limits = GuidanceLimits(**{**planner().limits.__dict__, 'tilt_rate_rad_s': math.radians(10.)})
    plan = ConvexGuidance(limits, EnergyModel(FULL, 3072 / .88, 10.)).plan([6., -4., 12.], [0., 0., 0.], WEIGHT, GATE)
    assert plan.mode == 'optimal'
    slew = np.linalg.norm(np.diff(plan.thrust_accel[:, :2], axis=0), axis=1) / np.diff(plan.times)
    assert slew.max() <= G * math.radians(10.) * (1 + 1e-3)


def _land_on_vanes(controller, authority, damping, r0=(2., -1.5, 10.), duration=25., com_xy=(0., 0.),
                   yaw_torque=0., stats=None, route=()):
    """Point mass + rigid body on vanes of the given authority, with the rotor's
    gyroscopic torque (implicit midpoint, energy-conserving like Isaac's
    integrator), its spool reaction I_rotor domega/dt, the servo model with its
    deadband and a 25 ms vane-joint lag; optionally a lateral CoM offset (m)
    and a steady yaw torque (N m, the jet's residual swirl), and a fixed
    route handed to the controller (never advanced). Returns
    (touchdown or None, peak |attitude error| deg); `stats` receives the
    roll/pitch and yaw rate RMS (deg/s) after the first 2 s."""
    from scipy.spatial.transform import Rotation
    servo, inertia, dt, h = _servo(), np.diag([.05, .05, .02]), 1 / 120, 2e-4 * 4649.56
    r, v, R, w = np.array(r0, float), np.zeros(3), np.eye(3), np.zeros(3)
    target, fins, rotor, t, worst, pq, yaw = torch.zeros(1, 4), np.zeros(4), .81, 0., 0., [], []
    joint_rate = np.zeros(4)
    while t < duration:
        obs = np.concatenate([-r, _quaternion(R), _FRD_TO_ISAAC.T @ (R.T @ v), _FRD_TO_ISAAC.T @ w, [r[2]],
                              fins, joint_rate, [rotor], [0.]])
        action = controller.compute_action(torch.tensor(obs[None], dtype=torch.float32), bus_voltage_v=29.6,
                                           route=route)[0].numpy()
        worst = max(worst, controller.last_telemetry.get('attitude_error_deg') or 0.)
        if t >= 2.:
            w_frd = np.degrees(_FRD_TO_ISAAC.T @ w)
            pq.append(w_frd[:2])
            yaw.append(w_frd[2])
        for _ in range(4):
            target = servo.update(target, torch.tensor(action[None, :4]), dt)
            joint_rate = np.clip((target[0].double().numpy() - fins) / .025, -6.98, 6.98)
            fins = fins + joint_rate * dt
            previous = rotor
            rotor = min(1., max(0., rotor + (min(1., float(action[4])) - rotor) / .15 * dt))
            effort = np.array([(fins[2] - fins[0]) / 2, (fins[3] - fins[1]) / 2, fins.mean()])
            thrust = FULL * rotor ** 2
            # A CoM offset d (FRD) under the thrust -T z_b (FRD) torques d x (0, 0, -T).
            com_torque = np.cross([com_xy[0], com_xy[1], 0.], [0., 0., -thrust])
            torque_frd = (np.array([authority[0], authority[0], authority[1]]) * effort * rotor ** 2 + com_torque
                          + [0., 0., yaw_torque * rotor ** 2] - [0., 0., h * (rotor - previous) / dt])
            H = np.array([0., 0., h * rotor])
            Hx = np.array([[0., -H[2], H[1]], [H[2], 0., -H[0]], [-H[1], H[0], 0.]])
            I_frd = _FRD_TO_ISAAC.T @ inertia @ _FRD_TO_ISAAC
            w_frd = _FRD_TO_ISAAC.T @ w
            rhs = I_frd @ w_frd + dt * (torque_frd - damping * w_frd - np.cross(w_frd, I_frd @ w_frd)) + .5 * dt * Hx @ w_frd
            w_new = _FRD_TO_ISAAC @ np.linalg.solve(I_frd - .5 * dt * Hx, rhs)
            R = R @ Rotation.from_rotvec(.5 * (w + w_new) * dt).as_matrix()
            w = w_new
            v = v + (thrust / MASS * R[:, 2] + [0., 0., -G]) * dt
            r = r + v * dt
            t += dt
            if r[2] <= .3125:
                break
        if r[2] <= .3125 or abs(w).max() > 20.:
            break
    if stats is not None:
        stats.update(pq_rms=float(np.sqrt(np.mean(np.square(pq)))) if pq else 0.,
                     r_rms=float(np.sqrt(np.mean(np.square(yaw)))) if yaw else 0.)
    if r[2] <= .3125:
        return dict(impact=-v[2], pad=float(np.linalg.norm(r[:2]))), worst
    return None, worst


def test_plant_aware_controller_lands_on_momentum_bounded_vanes():
    # Isaac 8abd7233a4ec / offline replica: tuned for the legacy vanes, the
    # controller lost attitude on the coupled jet, whose vanes have 8.5x less
    # authority and no artificial damper.
    blind = ConvexGuidanceController(settings('pd'), bare_vehicle(), (0., 0., 0.), .3125, 1 / 30,
                                     servo_deadband_rad=.017)
    touchdown, worst = _land_on_vanes(blind, MOMENTUM_VANES, damping=0.)
    assert touchdown is None or touchdown['impact'] > .25 or worst > 10.
    aware = ConvexGuidanceController(settings(), _with_vanes(MOMENTUM_VANES), (0., 0., 0.), .3125, 1 / 30,
                                     servo_deadband_rad=.017)
    stats = {}
    touchdown, worst = _land_on_vanes(aware, MOMENTUM_VANES, damping=0., stats=stats)
    assert touchdown is not None and touchdown['impact'] <= .25 and touchdown['pad'] <= .2
    assert worst < 5. and stats['pq_rms'] < 5.


def _pd_and_lqr_modes(fraction):
    """Slowest closed-loop roll/pitch decay rates (1/s) of the mission PD law and the LQR on the momentum-bounded plant."""
    import cmath
    from tvc_env.controllers.attitude_lqr import _zoh
    controller = ConvexGuidanceController(settings(), _with_vanes(MOMENTUM_VANES), (0., 0., 0.), .3125, 1 / 30)
    lqr = controller._lqr
    A, B = lqr.roll_pitch_model(fraction)
    Ad, Bd = _zoh(A, B, 1 / 30)
    pd = settings('pd')['attitude']
    kp, kd, comp, scale = pd['kp_att'], pd['kd_att'], pd['gyro_comp_rp'], 20.04 / 2.365
    gain = np.zeros((2, 10))                  # u = scale [-kd p - kp ex + comp q, -kd q - kp ey - comp p]
    gain[0, 2], gain[0, 4], gain[0, 5] = -kp, -kd, comp
    gain[1, 3], gain[1, 5], gain[1, 4] = -kp, -kd, -comp
    pd_decay = max((cmath.log(z) * 30).real for z in np.linalg.eigvals(Ad[2:, 2:] + Bd[2:] @ (scale * gain)[:, 2:]))
    lqr_modes = lqr.modes(fraction)
    return pd_decay, lqr_modes


def test_attitude_lqr_stabilizes_the_gyroscopic_plant_where_the_pd_law_cannot():
    # 2026-09-26: on momentum-bounded vanes the mission PD law is linearly
    # unstable at every rotor speed (Isaac: 4 Hz and 1-1.4 Hz limit cycles,
    # missions addd8752dc7c and e9c203457bba). The rotor's nutation sits at
    # H / I ~2.5 Hz, where the vane servo + joint lag ~0.075 s.
    for fraction in (.7, .845, 1.):
        pd_decay, modes = _pd_and_lqr_modes(fraction)
        assert pd_decay > .2
        assert max(decay for decay, _ in modes) < -1.
        zeta = min(-d / math.hypot(d, 2 * math.pi * hz) for d, hz in modes if hz > .3)
        assert zeta > .4
    # Stable (zeta > 0.2) when any single model parameter is off: +/-30% vane
    # authority, +50% servo or joint lag, +/-20% inertia, 8 ms more delay.
    import dataclasses
    lqr = ConvexGuidanceController(settings(), _with_vanes(MOMENTUM_VANES), (0., 0., 0.), .3125, 1 / 30)._lqr
    base = lqr.plant
    for change in (dict(rp_authority_nm_per_rad=base.rp_authority_nm_per_rad * .7),
                   dict(rp_authority_nm_per_rad=base.rp_authority_nm_per_rad * 1.3),
                   dict(servo_lag_s=base.servo_lag_s * 1.5), dict(servo_lag_s=base.servo_lag_s + 1 / 120),
                   dict(joint_lag_s=base.joint_lag_s * 1.5), dict(inertia_rp_kg_m2=base.inertia_rp_kg_m2 * .8),
                   dict(inertia_rp_kg_m2=base.inertia_rp_kg_m2 * 1.2)):
        modes = lqr.modes(.845, dataclasses.replace(base, **change))
        assert max(d for d, _ in modes) < 0.
        assert min(-d / math.hypot(d, 2 * math.pi * hz) for d, hz in modes if hz > .3) > .2, change


def test_attitude_lqr_holds_a_com_trim_and_stops_the_swirl_spin():
    # A 4.5 mm CoM offset needs ~0.09 rad of steady vane trim at hover, and the
    # jet's residual swirl is a steady ~0.05 N m yaw torque. The attitude
    # integral and the yaw-rate integral hold both without a standing
    # attitude error or a steady spin.
    def land(yaw_integral_weight):
        s = settings()
        s['attitude']['lqr']['yaw_integral_weight'] = yaw_integral_weight
        controller = ConvexGuidanceController(s, _with_vanes(MOMENTUM_VANES), (0., 0., 0.), .3125, 1 / 30,
                                              servo_deadband_rad=.017)
        stats = {}
        touchdown, _ = _land_on_vanes(controller, MOMENTUM_VANES, damping=0., com_xy=(.004, -.002),
                                      yaw_torque=.07, stats=stats)
        return controller, touchdown, stats

    controller, touchdown, stats = land(settings()['attitude']['lqr']['yaw_integral_weight'])
    assert touchdown is not None and touchdown['impact'] <= .25 and touchdown['pad'] <= .25
    assert stats['pq_rms'] < 8.            # manoeuvring to trim; the PD law's limit cycles ran at 15+ deg/s
    trim = controller._attitude_integral
    assert np.linalg.norm(controller._lqr.roll_pitch_gain(.81)[:, :2] @ trim) > .02   # the integral holds the trim
    # What yaw rate remains is the rotor's reaction to the descent's duty changes;
    # without the yaw-rate integral the swirl torque doubled it (25 vs 12 deg/s).
    _, _, spinning = land(0.)
    assert stats['r_rms'] < 15. and stats['r_rms'] < .6 * spinning['r_rms']


def test_cold_rotor_spools_up_before_the_first_plan():
    controller = ConvexGuidanceController(settings(), vehicle(), (0., 0., 0.), .3125, 1 / 30)
    obs = torch.zeros(1, 24)
    obs[0, :3] = torch.tensor([0., 0., -18.])
    obs[0, 3] = 1.
    action = controller.compute_action(obs, bus_voltage_v=33.)
    assert controller.phase == 'SPOOL_UP' and controller.plan is None
    assert 0. < float(action[0, 4]) <= settings()['spool_up']['throttle_rate_per_s'] / 30 + 1e-6


def test_attitude_lqr_is_the_same_torque_loop_on_any_vanes():
    # Bryson's rule on vane torque: weighted per rad of effort instead, the
    # legacy vanes' 8.5x authority made the design 72x more aggressive (a
    # 7 Hz loop that limit-cycled on the servo slew limit, offline replica).
    legacy = ConvexGuidanceController(settings(), _with_vanes(LEGACY_VANES), (0., 0., 0.), .3125, 1 / 30)._lqr
    jet = ConvexGuidanceController(settings(), _with_vanes(MOMENTUM_VANES), (0., 0., 0.), .3125, 1 / 30)._lqr
    for f in (.5, .845, 1.):
        a, b = legacy.roll_pitch_gain(f), jet.roll_pitch_gain(f)
        # Torque per unit of integral, error and rate is the same; the actuator
        # states are already in effort units, so their gains are equal.
        np.testing.assert_allclose(a[:, :6] * LEGACY_VANES[0], b[:, :6] * MOMENTUM_VANES[0], rtol=1e-5, atol=1e-9)
        np.testing.assert_allclose(a[:, 6:], b[:, 6:], rtol=1e-5, atol=1e-9)


def test_duty_slew_bounds_the_rotor_command_not_the_voltage_compensation():
    # Isaac mission 0a04e2549b26: a warm-rotor start computed its first duty
    # on the unloaded pack; the slew then held back the duty that offsets the
    # sagging bus, and the rotor spun down 1600 rpm, yawing the body 30 deg/s.
    controller = ConvexGuidanceController(settings(), _with_vanes(MOMENTUM_VANES), (0., 0., 0.), .3125, 1 / 30)
    rate = controller._duty_rate(29.6)
    controller._throttle = .844 * 29.6 / 33.6
    for volts in (33.6, 32.6, 31.8):
        controller._slew_duty(.844 * 29.6 / volts, rate, volts)
        assert controller._throttle * volts / 29.6 == pytest.approx(.844)   # the rotor command holds
    # A thrust change still slews: rate x V_bus / V_ref of rotor command per second.
    controller._slew_duty(.95 * 29.6 / 31.8, rate, 31.8)
    assert controller._rotor_command == pytest.approx(.844 + rate * 31.8 / 29.6 / 30)
    # The duty never exceeds 1: a sagged bus caps the rotor command.
    for _ in range(300):
        controller._slew_duty(2., rate, 25.)
    assert controller._throttle == pytest.approx(1.) and controller._rotor_command == pytest.approx(25. / 29.6)


def test_replans_hand_over_the_thrust_feedforward_smoothly():
    # Each re-plan starts at the present thrust with its own slope, so the
    # lead-sampled feedforward stepped at re-plans on route legs (offline
    # replica: up to 1.6 deg of tilt, against 0.05 deg per step). It now fades
    # from the replaced plan.
    class Recorder(ConvexGuidanceController):
        def _reference(self, position, waypoints):
            r, v, u = super()._reference(position, waypoints)
            self.trace.append((self._new_plan, np.asarray(u, dtype=float)))
            return r, v, u

    route = [dict(position=[6., 3., 6.], type='flypass', radius_m=1., speed_m_s=3.)]

    def jumps(blend_s):
        s = settings()
        s['guidance']['replan_blend_s'] = blend_s
        controller = Recorder(s, _with_vanes(MOMENTUM_VANES), (0., 0., 0.), .3125, 1 / 30, servo_deadband_rad=.017)
        controller.trace = []
        _land_on_vanes(controller, MOMENTUM_VANES, damping=0., r0=(0., 0., 8.), duration=3., route=route)
        pairs = list(zip(controller.trace, controller.trace[1:]))
        return (max(np.linalg.norm((b[1] - a[1])[:2]) for a, b in pairs if b[0]),
                float(np.median([np.linalg.norm((b[1] - a[1])[:2]) for a, b in pairs if not b[0]])))

    abrupt, typical = jumps(0.)
    smooth, _ = jumps(settings()['guidance']['replan_blend_s'])
    assert abrupt > 3. * typical            # re-plans stepped the feedforward
    assert smooth < .4 * abrupt


def test_replan_times_only_unflown_arc_and_retains_corridor():
    guidance = planner()
    target = np.array([0., 0., 12.3])
    curve = np.linspace([40., -40., 94.4], target, 49)
    route = [RouteWaypoint(tuple(target), kind='hover', hold_s=10., speed_m_s=3.)]
    early = guidance._route_segments(curve[0], np.zeros(3), route, 1., [curve, None])
    late = guidance._route_segments(target + [0., 0., 1.], np.zeros(3), route, 1., [curve, None])
    assert early[0].duration > 30.
    assert late[0].duration < 4.
    assert late[0].curve is curve  # timing must not shorten the enforced corridor
    assert late[1].duration == early[1].duration == 10.5


@pytest.mark.parametrize('objective', ['energy', 'delta_v'])
def test_receding_horizon_captures_hover_after_long_incoming_leg(objective):
    # Regression for mission 8243dfb69c0d: the reference itself hovered outside
    # capture, despite <0.1 m tracking error. Exercise the actual sequencer.
    from tvc_env.envs.waypoints import WaypointMission
    guidance = planner(objective)
    guidance.corridor_m = 2.
    target = [-.1, .2, 12.3]
    points = [[41.4, -39.4, 94.4], target, list(GATE.position)]
    curves = [catmull_rom_leg(points, i) for i in range(2)]
    for curve in curves:
        curve[:, 2] = np.maximum(curve[:, 2], min(curve[0, 2], curve[-1, 2]))
    config = {'task': {'navigation': {'waypoints': [dict(type='hover', position=target,
                    radius_m=1., hold_s=2., speed_m_s=3.)]}}}
    nav = WaypointMission(1, 'cpu', config, torch.zeros(1, 3), torch.zeros(1, 3))
    r = np.array([-.7, -.7, 14.4]); v = np.zeros(3); thrust = np.array([0., 0., WEIGHT])
    nav.reset(torch.tensor([0]), torch.tensor([points[0]]))
    try:
        for _ in range(40):
            route = [RouteWaypoint(tuple(target), kind='hover', hold_s=max(0., 2.-float(nav.hold_elapsed[0])), speed_m_s=3.)]
            plan = guidance.plan(r, v, thrust, GATE, route, path=curves)
            assert plan is not None and plan.mode == 'optimal'
            before = torch.tensor(r[None], dtype=torch.float32)
            r, v, u = plan.sample(.5)
            thrust = u * MASS
            nav.advance(before, torch.tensor(r[None], dtype=torch.float32),
                        torch.tensor(v[None], dtype=torch.float32), .5, torch.tensor([True]))
            if nav.ready_to_land[0]:
                break
        assert nav.ready_to_land[0], 'Repeated replans postponed the hover indefinitely'
    finally:
        guidance.close()


# ---- secondary objectives and the fly-through arrival cone (2026-09-28) ----

def _tracking_planner(weights=None, **kwargs):
    """Planned 8S limits with the 10 deg/s attitude-slew bound and a 1 m corridor."""
    limits = GuidanceLimits(mass_kg=MASS, thrust_min_n=0.8 * WEIGHT, thrust_max_n=0.86 * FULL, thrust_rate_n_s=RATE,
                            max_tilt_rad=math.radians(15), max_speed_m_s=4., glide_slope_rad=math.radians(45),
                            tilt_rate_rad_s=math.radians(10), emergency_thrust_max_n=FULL,
                            emergency_thrust_rate_n_s=8 * RATE)
    return ConvexGuidance(limits, EnergyModel(FULL, 3072 / .88, 10.), corridor_m=1., corridor_weight=25.,
                          weights=weights, **kwargs)


def _drawn(start, route):
    """The controller's corridor curves for a route ending at the pad (never below a leg's lower end)."""
    points = [start] + [w.position for w in route] + [[0., 0., 0.]]
    path = [catmull_rom_leg(points, leg) for leg in range(len(route) + 1)]
    for curve in path:
        curve[:, 2] = np.maximum(curve[:, 2], min(curve[0, 2], curve[-1, 2]))
    return path


def _leg_deviation(plan, legs=None):
    """Largest node distance from the drawn curve over the first `legs` fly-through legs."""
    worst, node = 0., 0
    for i, seg in enumerate(plan.segments):
        first, node = node, node + seg.nodes
        if (legs is None or i < legs) and seg.kind == 'flypass' and seg.curve is not None:
            worst = max(worst, float(curve_distance(seg.curve, plan.position[first + 1:node + 1]).max()))
    return worst


def test_path_weight_pulls_the_plan_from_the_corridor_edge_onto_the_route():
    # Inside the corridor the energy optimum is indifferent to position and
    # rides the edge (0.98 m of a 1 m corridor); a path price moves the plan
    # onto the drawn route for a small energy premium.
    start = [0., 0., 20.]
    route = (RouteWaypoint((20., 0., 15.), 'flypass', 1., 3.), RouteWaypoint((20., 20., 10.), 'flypass', 1., 3.))
    path = _drawn(start, route)
    energy = _tracking_planner().plan(start, [0., 0., 0.], WEIGHT, GATE, route, path=path)
    tracked = _tracking_planner(ObjectiveWeights(path=10.)).plan(start, [0., 0., 0.], WEIGHT, GATE, route, path=path)
    assert energy.mode == tracked.mode == 'optimal'
    assert _leg_deviation(energy, 1) > .9 and _leg_deviation(tracked, 1) < .1
    assert tracked.energy_wh < 1.01 * energy.energy_wh
    assert set(tracked.cost_terms) >= {'energy', 'path'} and tracked.convexification_gap < 1e-4
    assert math.isclose(sum(tracked.cost_terms.values()), tracked.cost, rel_tol=1e-4)
    assert tracked.route_deviation_m >= _leg_deviation(tracked) - 1e-6


def test_arrival_cone_makes_a_tight_zigzag_feasible_and_keeps_gate_direction():
    start = [0., 0., 3.]
    route = (RouteWaypoint((8., 6., 6.), 'flypass', 1., 3.), RouteWaypoint((16., -6., 8.), 'flypass', 1., 3.),
             RouteWaypoint((24., 6., 6.), 'flypass', 1., 3.), RouteWaypoint((24., 6., 4.), 'hover', .5, 2., 2.))
    path = _drawn(start, route)
    exact = _tracking_planner().plan(start, [0., 0., 0.], WEIGHT, GATE, route, path=path)
    assert exact.mode == 'soft_terminal'     # exact gate velocities: no feasible plan under the slew bound
    tolerance = math.radians(20)
    guidance = _tracking_planner(ObjectiveWeights(path=10.), flypass_min_speed_fraction=.5,
                                 flypass_heading_tolerance_rad=tolerance)
    plan = guidance.plan(start, [0., 0., 0.], WEIGHT, GATE, route, path=path)
    assert plan.mode == 'optimal' and plan.convexification_gap < 1e-4
    segments = guidance._route_segments(np.array(start), np.zeros(3), route, 1., path)
    for node, seg in zip(plan.waypoint_nodes[:3], segments[:3]):
        v, tangent = plan.velocity[node], seg.arrival_velocity / np.linalg.norm(seg.arrival_velocity)
        along = float(v @ tangent)
        assert .5 * 3. - 1e-3 <= along <= 3. + 1e-3
        assert math.atan2(float(np.linalg.norm(v - along * tangent)), along) <= tolerance + 1e-3
    assert _leg_deviation(plan) < .3


def test_default_arrival_is_the_exact_leg_velocity():
    start = [0., 0., 20.]
    route = (RouteWaypoint((20., 0., 15.), 'flypass', 1., 3.), RouteWaypoint((20., 20., 10.), 'flypass', 1., 3.))
    guidance = _tracking_planner()
    plan = guidance.plan(start, [0., 0., 0.], WEIGHT, GATE, route, path=_drawn(start, route))
    assert plan.mode == 'optimal'
    segments = guidance._route_segments(np.array(start), np.zeros(3), route, 1., _drawn(start, route))
    for node, seg in zip(plan.waypoint_nodes, segments):
        np.testing.assert_allclose(plan.velocity[node], seg.arrival_velocity, atol=1e-5)


def test_smoothness_and_tilt_weights_trade_energy_for_actuator_margin():
    start = [0., 0., 20.]
    route = (RouteWaypoint((20., 0., 15.), 'flypass', 1., 3.), RouteWaypoint((20., 20., 10.), 'flypass', 1., 3.))
    path = _drawn(start, route)

    def jerk(plan):
        return float(np.sqrt(np.mean(np.square(np.linalg.norm(np.diff(plan.thrust_accel, axis=0), axis=1)
                                               / np.diff(plan.times)))))

    def tilt(plan):
        return float(np.mean(np.linalg.norm(plan.thrust_accel[:, :2], axis=1)))

    base = _tracking_planner().plan(start, [0., 0., 0.], WEIGHT, GATE, route, path=path)
    smooth = _tracking_planner(ObjectiveWeights(smoothness=5.)).plan(start, [0., 0., 0.], WEIGHT, GATE, route, path=path)
    level = _tracking_planner(ObjectiveWeights(tilt=5.)).plan(start, [0., 0., 0.], WEIGHT, GATE, route, path=path)
    assert jerk(smooth) < .6 * jerk(base) and smooth.cost_terms['smoothness'] > 0.
    assert tilt(level) < tilt(base) and level.cost_terms['tilt'] > 0.
    for plan in (smooth, level):
        assert plan.mode == 'optimal' and plan.convexification_gap < 1e-4
        assert math.isclose(sum(plan.cost_terms.values()), plan.cost, rel_tol=1e-4)


def test_time_weight_prices_duration_in_hover_seconds():
    start = [6., -4., 12.]
    route = (RouteWaypoint((0., 8., 8.), 'hover', .5, 3., 1.),)
    path = _drawn(start, route)
    timed = _tracking_planner(ObjectiveWeights(time=1.)).plan(start, [0., 0., 0.], WEIGHT, GATE, route, path=path)
    assert timed.mode == 'optimal'
    # weight x hover cost x duration, in Wh (the auxiliary load is not part of the priced hover cost).
    e = _tracking_planner().energy
    hover_wh = e.power_ref_w * (WEIGHT / e.thrust_ref_n) ** 1.5 / 3600.
    assert math.isclose(timed.cost_terms['time'], hover_wh * timed.duration, rel_tol=1e-3)
    free = _tracking_planner().plan(start, [0., 0., 0.], WEIGHT, GATE, route, path=path)
    assert timed.duration <= free.duration + 1e-6


def test_objective_weights_and_arrival_settings_are_validated():
    with pytest.raises(ValueError):
        ObjectiveWeights(path=-1.)
    with pytest.raises(ValueError):
        _tracking_planner(flypass_min_speed_fraction=0.)


def test_controller_reports_cost_terms_and_flown_cross_track():
    s = settings()
    s['guidance'].update(path_weight=10., smoothness_weight=2.)
    controller = ConvexGuidanceController(s, _with_vanes(MOMENTUM_VANES), (0., 0., 0.), .3125, 1 / 30)
    assert controller.guidance.weights == ObjectiveWeights(path=10., smoothness=2.)
    route = [dict(position=[6., 0., 8.], type='hover', hold_s=2., radius_m=.5, speed_m_s=2.)]
    obs = np.concatenate([[6., 0., 0.], [1., 0., 0., 0.], np.zeros(6), [8.], np.zeros(8), [.81], [0.]])
    controller.compute_action(torch.tensor(obs[None], dtype=torch.float32), reference_position=[6., 0., 8.],
                              route=route, path_points=[[0., 0., 8.], [6., 0., 8.], [0., 0., .3125]], path_index=0)
    telemetry = controller.last_telemetry
    assert telemetry['phase'] == 'ROUTE' and 'path' in telemetry['solver']['cost_terms']
    assert telemetry['cross_track_m'] < 1e-3 and telemetry['solver']['route_deviation_m'] >= 0.


def test_landing_leg_never_outruns_its_braking_envelope():
    # Isaac 751dd0214086: from a 4.6 m hover the energy optimum sank at 2.0
    # m/s and braked on the thrust-rate bound; the vehicle touched down at
    # 2.1 m/s. The envelope keeps -v_z <= sqrt(v_gate^2 + 2 a (z - z_gate)).
    fraction = 0.35
    limits = GuidanceLimits(**{**planner().limits.__dict__, 'landing_sink_brake_fraction': fraction})
    guidance = ConvexGuidance(limits, EnergyModel(FULL, 3072 / .88, 10.))
    free = planner().plan([0., .3, 4.6], [0., 0., 0.], WEIGHT, GATE)
    plan = guidance.plan([0., .3, 4.6], [0., 0., 0.], WEIGHT, GATE)
    assert plan.mode == 'optimal' and plan.convexification_gap < 1e-3
    a = fraction * (limits.thrust_max_n - WEIGHT) / MASS
    envelope = np.sqrt(.15 ** 2 + 2 * a * (plan.position[:, 2] - GATE.position[2]))
    assert np.all(-plan.velocity[:, 2] <= envelope + 1e-3)
    assert -free.velocity[:, 2].min() > -plan.velocity[:, 2].min() + .2   # the unconstrained plan sinks faster
    # A fast direct descent starts outside the envelope and is still planned.
    fast = guidance.plan([2., -1., 12.], [0., 0., -5.], WEIGHT, GATE)
    assert fast is not None and fast.mode == 'optimal'


def test_braking_emergency_fires_only_when_the_yaw_safe_slew_cannot_stop():
    s = settings()
    controller = ConvexGuidanceController(s, _with_vanes(MOMENTUM_VANES), (0., 0., 0.), .3125, 1 / 30)
    controller._phase = 'POWERED_DESCENT'
    available = controller.vehicle.available_thrust_n(29.6)
    fires = lambda z, sink, thrust: controller._braking_emergency(  # noqa: E731
        np.array([0., 0., z]), np.array([0., 0., -sink]), thrust, available, 29.6)
    assert fires(2.6, 2.6, .9 * WEIGHT)           # Isaac 751dd0214086 at 69.4 s
    assert not fires(4.6, .5, WEIGHT)             # a calm start of the landing leg
    assert not fires(.6, .18, WEIGHT)             # the terminal descent's own rate
    s['guidance']['braking_emergency_height_fraction'] = None
    off = ConvexGuidanceController(s, _with_vanes(MOMENTUM_VANES), (0., 0., 0.), .3125, 1 / 30)
    off._phase = 'POWERED_DESCENT'
    assert not off._braking_emergency(np.array([0., 0., 2.6]), np.array([0., 0., -2.6]), .9 * WEIGHT, available, 29.6)


# ---- flight-computer cost and planning latency (physics review 2026-09-29) ----

def test_scheduled_lqr_gain_is_the_per_entry_interpolation():
    lqr = ConvexGuidanceController(settings(), _with_vanes(MOMENTUM_VANES), (0., 0., 0.), .3125, 1 / 30)._lqr
    for fraction in (0.0, 0.3, 0.3125, 0.47, 0.8, 0.99, 1.0, 1.4):
        for table, gain in ((lqr._rp, lqr.roll_pitch_gain(fraction)), (lqr._yaw, lqr.yaw_gain(fraction))):
            reference = np.array([[np.interp(fraction, lqr.fractions, table[:, i, j]) for j in range(table.shape[2])]
                                  for i in range(table.shape[1])])
            assert np.allclose(gain, reference, rtol=0, atol=1e-12)


def test_constraint_assembly_matches_the_row_by_row_definition():
    from scipy import sparse
    from tvc_env.controllers.convex_guidance import _Cones
    rng = np.random.default_rng(3)

    def terms():
        return [(int(c), float(v)) for c, v in zip(rng.choice(40, rng.integers(0, 5), replace=False), rng.normal(size=5))]

    cones = _Cones()
    for _ in range(7):
        cones.eq(terms(), float(rng.normal()))
    for _ in range(9):
        cones.le(terms(), float(rng.normal()))
    for _ in range(4):
        cones.soc([(terms(), float(rng.normal())) for _ in range(4)])
    cones.power([(terms(), 1.0), (terms(), 0.0), (terms(), 0.5)], 2 / 3)
    A, b, cone_list = cones.assemble(40)
    rows, rhs = [], []
    for block, sign in (('zero', 1.0), ('nonneg', 1.0)):
        for t, c in cones.blocks[block]:
            rows.append([(col, sign * v) for col, v in t]); rhs.append(c)
    for group in cones.blocks['soc'] + [rows_alpha[0] for rows_alpha in cones.blocks['pow']]:
        for t, c in group:
            rows.append([(col, -v) for col, v in t]); rhs.append(c)
    dense = np.zeros((len(rows), 40))
    for r, t in enumerate(rows):
        for col, v in t:
            dense[r, col] += v
    assert sparse.issparse(A) and np.array_equal(A.toarray(), dense) and np.array_equal(b, rhs)
    assert [type(c).__name__ for c in cone_list] == ['ZeroConeT', 'NonnegativeConeT'] + ['SecondOrderConeT'] * 4 + ['PowerConeT']


def test_planning_latency_keeps_flying_the_old_plan_until_the_solve_lands():
    s = settings()
    s['guidance']['plan_latency_s'] = 0.3
    controller = ConvexGuidanceController(s, _with_vanes(MOMENTUM_VANES), (0., 0., 0.), .3125, 1 / 30,
                                          servo_deadband_rad=.017)
    installed = []
    original = controller._install

    def spy(new, requested_t, *args):
        installed.append((round(controller._t, 6), round(requested_t, 6)))
        return original(new, requested_t, *args)
    controller._install = spy
    touchdown, worst = _land_on_vanes(controller, MOMENTUM_VANES, damping=0.)
    first, later = installed[0], installed[1:]
    assert first[0] == first[1] == 0.0                               # the pad plan is solved before launch
    assert later and all(t - requested >= 0.3 - 1e-6 for t, requested in later)
    assert all(t - requested < 0.3 + 1 / 30 + 1e-6 for t, requested in later)
    # The plan's clock starts at the request, so tracking continues where it was solved from.
    assert touchdown is not None and touchdown['impact'] <= .25 and touchdown['pad'] <= .2 and worst < 5.


def test_zero_planning_latency_is_the_synchronous_controller():
    def fly(latency):
        s = settings()
        if latency is None:
            s['guidance'].pop('plan_latency_s', None)          # configs that predate the setting
        else:
            s['guidance']['plan_latency_s'] = latency
        controller = ConvexGuidanceController(s, _with_vanes(MOMENTUM_VANES), (0., 0., 0.), .3125, 1 / 30,
                                              servo_deadband_rad=.017)
        return _land_on_vanes(controller, MOMENTUM_VANES, damping=0., duration=6.)
    assert fly(None) == fly(0.0)
    with pytest.raises(ValueError):
        s = settings()
        s['guidance']['plan_latency_s'] = -0.1
        ConvexGuidanceController(s, _with_vanes(MOMENTUM_VANES), (0., 0., 0.), .3125, 1 / 30)


@pytest.mark.parametrize('mode', ['thread', 'process'])
def test_async_replanning_keeps_the_control_step_short_and_still_lands(mode):
    import sys
    import time
    s = settings()
    s['guidance']['async_replan'] = mode
    switch = sys.getswitchinterval()
    controller = ConvexGuidanceController(s, _with_vanes(MOMENTUM_VANES), (0., 0., 0.), .3125, 1 / 30,
                                          servo_deadband_rad=.017)
    steps, compute = [], controller.compute_action

    def paced(*args, **kwargs):
        started = time.perf_counter()
        action = compute(*args, **kwargs)
        steps.append(time.perf_counter() - started)
        time.sleep(0.004)                 # let background solves land within a few control steps
        return action
    controller.compute_action = paced
    try:
        assert controller.planner_ready(60)
        touchdown, worst = _land_on_vanes(controller, MOMENTUM_VANES, damping=0.)
    finally:
        controller.close()
        sys.setswitchinterval(switch)
    assert controller._plan_id >= 2                                   # background re-plans were taken over
    flight = sorted(steps[1:])                                        # steps[0] solves the pad plan
    # A thread shares the GIL with the solve: typically <= 6 ms, but C-level
    # work stalls the odd step ~35 ms. A planner process never blocks the loop.
    assert flight[int(0.95 * len(flight))] < 0.01
    if mode == 'process':
        assert flight[-2] < 0.02
    assert touchdown is not None and touchdown['impact'] <= .25 and touchdown['pad'] <= .2 and worst < 5.


def test_async_replan_mode_is_validated():
    s = settings()
    s['guidance']['async_replan'] = 'fibre'
    with pytest.raises(ValueError):
        ConvexGuidanceController(s, _with_vanes(MOMENTUM_VANES), (0., 0., 0.), .3125, 1 / 30)
