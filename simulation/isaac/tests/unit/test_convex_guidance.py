"""Convex (SOCP) powered-descent guidance and its tracking controller."""
import math

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip('clarabel')

from tvc_env.controllers.convex_adapter import ConvexGuidanceController, VehicleModel, _FRD_TO_ISAAC  # noqa: E402
from tvc_env.controllers.convex_guidance import (  # noqa: E402
    ConvexGuidance, EnergyModel, GuidanceLimits, LandingGate, RouteWaypoint, _Segment,
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


def settings():
    return yaml.safe_load((ROOT / 'configs/controllers/convex_guidance.yaml').read_text())


def vehicle():
    return VehicleModel(mass_kg=MASS, hover_throttle=.81, reference_voltage_v=29.6,
                        shaft_power_at_max_w=3072., motor_efficiency=.88, auxiliary_power_w=10.)


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


def test_geometric_attitude_efforts_match_pid_euler_convention():
    s = settings()
    controller = ConvexGuidanceController(s, vehicle(), (0., 0., 0.), .3125, 1 / 30)
    kp, kd, comp = (s['attitude'][k] for k in ('kp_att', 'kd_att', 'gyro_comp_rp'))
    mixer = PIDFinMixer(max_fin_angle=controller._max_fin_angle)
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
    s = settings()
    s['attitude']['vane_rp_authority_nm_per_rad'] = authority = 20.7   # identified value, opt-in
    with_rotor = VehicleModel(**{**vehicle().__dict__, 'rotor_inertia': 2e-4, 'omega_max': 4649.56})
    controller = ConvexGuidanceController(s, with_rotor, (0., 0., 0.), .3125, 1 / 30)
    for f in (.6, .88, 1.):
        k = controller._gyro_compensation(f)
        # Cross-feed torque K f^2 k q equals the gyroscopic torque H q, H = I_r omega_max f.
        assert authority * f * f * k == pytest.approx(2e-4 * 4649.56 * f)
    assert controller._gyro_compensation(.88) == pytest.approx(.051, abs=.002)
    # Without rotor parameters, or by default, the fixed PID cross-feed is used.
    assert ConvexGuidanceController(s, vehicle(), (0., 0., 0.), .3125, 1 / 30)._gyro_compensation(.88) == s['attitude']['gyro_comp_rp']
    default = ConvexGuidanceController(settings(), with_rotor, (0., 0., 0.), .3125, 1 / 30)
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
    with_rotor = VehicleModel(**{**vehicle().__dict__, 'rotor_inertia': 2e-4, 'omega_max': 4649.56})

    def controller(deadband_rad):
        return ConvexGuidanceController(settings(), with_rotor, (0., 0., 0.), .3125, 1 / 30,
                                        servo_deadband_rad=deadband_rad)

    # Isaac mission 0289417f5d60 coned at ~15 deg/s RMS with the 1 deg servo deadband.
    coning = _coning_rate_rms(controller(0.), .017)
    assert coning > 8.
    assert _coning_rate_rms(controller(.017), .017) < 2.
    # Over-compensating (no real deadband) only dithers, far below the coning it removes.
    assert _coning_rate_rms(controller(.017), 0.) < .5 * coning


LEGACY_VANES, MOMENTUM_VANES = (20.04, 14.58), (2.365, 1.72)   # probed N m/rad at full rotor (planned 8S)


def _with_vanes(authority):
    return VehicleModel(**{**vehicle().__dict__, 'rotor_inertia': 2e-4, 'omega_max': 4649.56,
                           'vane_authority_nm_per_rad': authority[0], 'yaw_authority_nm_per_rad': authority[1]})


def test_vane_efforts_scale_to_the_plant_authority():
    s = settings()
    legacy = ConvexGuidanceController(s, _with_vanes(LEGACY_VANES), (0., 0., 0.), .3125, 1 / 30)
    jet = ConvexGuidanceController(s, _with_vanes(MOMENTUM_VANES), (0., 0., 0.), .3125, 1 / 30)
    R = to_rotation_matrix(from_euler(torch.tensor(.01), torch.tensor(-.006), torch.tensor(0.)).double()).numpy()
    rates = np.array([.02, -.01, .05])
    small = np.array(legacy._attitude_fins(R, np.array([0., 0., 1.]), rates, .84))
    # The gains were tuned on the legacy vanes: identical there, 8.47x the deflection on the coupled jet.
    np.testing.assert_allclose(small, ConvexGuidanceController(s, vehicle(), (0., 0., 0.), .3125, 1 / 30)
                               ._attitude_fins(R, np.array([0., 0., 1.]), rates, .84), atol=1e-9)
    # (Roll/pitch scale by 20.04/2.365, the yaw common mode by 14.58/1.72: 8.474 vs 8.477.)
    np.testing.assert_allclose(jet._attitude_fins(R, np.array([0., 0., 1.]), rates, .84),
                               small * 20.04 / 2.365, rtol=1e-3)
    # A large error becomes a slew at the rate the vanes sustain against the rotor, not saturated vanes.
    R = to_rotation_matrix(from_euler(torch.tensor(.2), torch.tensor(0.), torch.tensor(0.)).double()).numpy()
    fins = np.array(jet._attitude_fins(R, np.array([0., 0., 1.]), np.zeros(3), .84))
    slew = .6 * 2.365 * .81 * .81 * .2 / (2e-4 * 4649.56 * .81)
    assert jet._slew_rate == pytest.approx(slew)
    assert np.abs(fins).max() == pytest.approx(s['attitude']['kd_att'] * slew * 20.04 / 2.365, rel=1e-6)
    assert np.abs(fins).max() < s['attitude']['max_fin_angle']
    assert legacy._slew_rate > math.radians(120.)          # inactive on the legacy vanes


def test_duty_slew_and_yaw_reserve_follow_the_yaw_authority():
    s = settings()
    legacy = ConvexGuidanceController(s, _with_vanes(LEGACY_VANES), (0., 0., 0.), .3125, 1 / 30)
    jet = ConvexGuidanceController(s, _with_vanes(MOMENTUM_VANES), (0., 0., 0.), .3125, 1 / 30)
    assert legacy._duty_rate(29.6) == s['guidance']['throttle_rate_per_s']
    reaction = 2e-4 * 4649.56 * jet._duty_rate(29.6)          # N m of rotor reaction at the slew
    assert reaction == pytest.approx(.4 * 1.72 * .81 * .81 * .2)
    assert jet._duty_rate(33.6) < jet._duty_rate(29.6)      # a full pack spins the rotor faster per duty
    assert jet._yaw_reserve == pytest.approx(.4 * .2)
    assert legacy._yaw_reserve < math.radians(1.5)
    # The plan's tilt rate: ~10 deg/s on the coupled jet, inactive (~83 deg/s) on legacy vanes.
    assert math.degrees(jet._limits.tilt_rate_rad_s) == pytest.approx(10., abs=1.)
    assert math.degrees(legacy._limits.tilt_rate_rad_s) > 80.


def test_planned_tilt_rate_respects_the_attitude_slew_bound():
    limits = GuidanceLimits(**{**planner().limits.__dict__, 'tilt_rate_rad_s': math.radians(10.)})
    plan = ConvexGuidance(limits, EnergyModel(FULL, 3072 / .88, 10.)).plan([6., -4., 12.], [0., 0., 0.], WEIGHT, GATE)
    assert plan.mode == 'optimal'
    slew = np.linalg.norm(np.diff(plan.thrust_accel[:, :2], axis=0), axis=1) / np.diff(plan.times)
    assert slew.max() <= G * math.radians(10.) * (1 + 1e-3)


def _land_on_vanes(controller, authority, damping, r0=(2., -1.5, 10.), duration=25.):
    """Point mass + rigid body on vanes of the given authority, with the rotor's
    gyroscopic torque, its spool reaction I_rotor domega/dt, the servo model
    and a 25 ms vane-joint lag. Returns (touchdown or None, peak |attitude error| deg)."""
    from scipy.spatial.transform import Rotation
    servo, inertia, dt, h = _servo(), np.diag([.05, .05, .02]), 1 / 120, 2e-4 * 4649.56
    r, v, R, w = np.array(r0, float), np.zeros(3), np.eye(3), np.zeros(3)
    target, fins, rotor, t, worst = torch.zeros(1, 4), np.zeros(4), .81, 0., 0.
    while t < duration:
        obs = np.concatenate([-r, _quaternion(R), _FRD_TO_ISAAC.T @ (R.T @ v), _FRD_TO_ISAAC.T @ w, [r[2]],
                              fins, np.zeros(4), [rotor], [0.]])
        action = controller.compute_action(torch.tensor(obs[None], dtype=torch.float32), bus_voltage_v=29.6)[0].numpy()
        worst = max(worst, controller.last_telemetry.get('attitude_error_deg') or 0.)
        for _ in range(4):
            target = servo.update(target, torch.tensor(action[None, :4]), dt)
            fins = fins + np.clip((target[0].double().numpy() - fins) / .025, -6.98, 6.98) * dt
            previous = rotor
            rotor = min(1., max(0., rotor + (min(1., float(action[4])) - rotor) / .15 * dt))
            effort = np.array([(fins[2] - fins[0]) / 2, (fins[3] - fins[1]) / 2, fins.mean()])
            w_frd = _FRD_TO_ISAAC.T @ w
            torque_frd = (np.array([authority[0], authority[0], authority[1]]) * effort * rotor ** 2
                          - np.cross(w_frd, [0., 0., h * rotor]) - [0., 0., h * (rotor - previous) / dt])
            torque = _FRD_TO_ISAAC @ torque_frd - damping * w
            w = w + np.linalg.solve(inertia, torque - np.cross(w, inertia @ w)) * dt
            R = R @ Rotation.from_rotvec(w * dt).as_matrix()
            v = v + (FULL * rotor ** 2 / MASS * R[:, 2] + [0., 0., -G]) * dt
            r = r + v * dt
            t += dt
            if r[2] <= .3125:
                return dict(impact=-v[2], pad=float(np.linalg.norm(r[:2]))), worst
    return None, worst


def test_plant_aware_controller_lands_on_momentum_bounded_vanes():
    # Isaac 8abd7233a4ec / offline replica: tuned for the legacy vanes, the
    # controller lost attitude on the coupled jet, whose vanes have 8.5x less
    # authority and no artificial damper.
    blind = ConvexGuidanceController(settings(), vehicle(), (0., 0., 0.), .3125, 1 / 30, servo_deadband_rad=.017)
    touchdown, worst = _land_on_vanes(blind, MOMENTUM_VANES, damping=0.)
    assert touchdown is None or touchdown['impact'] > .25 or worst > 10.
    aware = ConvexGuidanceController(settings(), _with_vanes(MOMENTUM_VANES), (0., 0., 0.), .3125, 1 / 30,
                                     servo_deadband_rad=.017)
    touchdown, worst = _land_on_vanes(aware, MOMENTUM_VANES, damping=0.)
    assert touchdown is not None and touchdown['impact'] <= .25 and touchdown['pad'] <= .3
    assert worst < 5.


def test_cold_rotor_spools_up_before_the_first_plan():
    controller = ConvexGuidanceController(settings(), vehicle(), (0., 0., 0.), .3125, 1 / 30)
    obs = torch.zeros(1, 24)
    obs[0, :3] = torch.tensor([0., 0., -18.])
    obs[0, 3] = 1.
    action = controller.compute_action(obs, bus_voltage_v=33.)
    assert controller.phase == 'SPOOL_UP' and controller.plan is None
    assert 0. < float(action[0, 4]) <= settings()['spool_up']['throttle_rate_per_s'] / 30 + 1e-6
