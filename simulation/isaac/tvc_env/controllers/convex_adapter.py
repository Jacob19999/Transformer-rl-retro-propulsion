"""Convex-guidance flight controller: closed-loop SOCP plans, tracked by
attitude and throttle loops.

Outer loop (every replan_period_s): `ConvexGuidance` re-solves the landing /
route SOCP from the measured state, as the paper intends for onboard use.
Inner loop (control rate): the plan's thrust acceleration is the
feedforward; position/velocity feedback (with slow integrators for
thrust-model bias and wind) corrects deviations between replans. The thrust
vector sets the desired body axis (geometric SO(3) attitude error, a
full-state gyroscopic LQR over the body and vane-actuator model
(attitude_lqr.py), the radial-vane mixer, then an inverse of the vane
servos' deadband from the measured servo state) and, projected on the actual
body axis, the throttle through the EDF duty/bus-voltage model.

Flight phases:
  SPOOL_UP          rotor below the planner's minimum thrust (cold start):
                    level attitude, duty ramps to hover, no plan yet
  ROUTE             tracking a plan whose waypoint legs are not complete
  POWERED_DESCENT   tracking the plan's final leg to the landing gate
  TERMINAL_DESCENT  constant-rate vertical descent over the pad to contact;
                    the descent pauses while the vehicle is off center
  HOLD              no feasible plan yet: hold the current position

On a mission with waypoints every plan also stays near the drawn route (the
sequencer's Catmull-Rom curve), a soft corridor in the SOCP.

The controller reads only the observation (so sensor noise reaches it), the
mission sequencer's route and the measured bus voltage. It never reads
simulator ground truth and never alters the environment.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
import torch
from torch import Tensor

from tvc_env.common.constants import ContactState
from tvc_env.common.frames import get_frd_to_isaac_matrix
from tvc_env.common.quaternions import to_rotation_matrix
from tvc_env.controllers.attitude_lqr import AttitudePlant, GyroAttitudeLQR, LQRWeights
from tvc_env.controllers.base import BaseController
from tvc_env.controllers.convex_guidance import (
    HOVER, ConvexGuidance, EnergyModel, GuidanceLimits, LandingGate, RouteWaypoint, catmull_rom_leg,
)
from tvc_env.controllers.pid_fin_mixer import PIDFinMixer

_FRD_TO_ISAAC = get_frd_to_isaac_matrix(dtype=torch.float64).numpy()
SPOOL_UP, ROUTE, POWERED_DESCENT, TERMINAL_DESCENT, HOLD = (
    'SPOOL_UP', 'ROUTE', 'POWERED_DESCENT', 'TERMINAL_DESCENT', 'HOLD')


@dataclass(frozen=True)
class VehicleModel:
    """What the flight computer knows about the plant (resolved hardware)."""

    mass_kg: float
    hover_throttle: float            # level equilibrium duty at the reference voltage
    reference_voltage_v: float = 1.0
    shaft_power_at_max_w: float = 0.0
    motor_efficiency: float = 1.0
    auxiliary_power_w: float = 0.0
    rotor_inertia: float = 0.0       # kg m^2, for gyroscopic decoupling
    omega_max: float = 0.0           # rad/s at full rotor fraction
    # Vane torque per rad of effort at full rotor speed (scales with rotor
    # fraction squared); 0 = unknown. TVCDirectRLEnv.vane_authority() probes it.
    vane_authority_nm_per_rad: float = 0.0   # roll/pitch
    yaw_authority_nm_per_rad: float = 0.0    # common mode
    # Attitude dynamics for the LQR attitude law (attitude_lqr.py); 0 = unknown.
    inertia_rp_kg_m2: float = 0.0            # transverse moment of inertia
    inertia_yaw_kg_m2: float = 0.0
    angular_damping_nm_s_per_rad: float = 0.0
    servo_lag_s: float = 0.0                 # vane servo first-order time constant
    vane_joint_lag_s: float = 0.0            # vane linkage lag behind the servo
    gravity: float = 9.81

    @property
    def weight_n(self) -> float:
        return self.mass_kg * self.gravity

    @property
    def full_thrust_n(self) -> float:
        """Net thrust at full rotor speed, neutral vane drag included."""
        return self.weight_n / self.hover_throttle ** 2

    def available_thrust_n(self, bus_voltage_v: float) -> float:
        return self.full_thrust_n * min(1.0, bus_voltage_v / self.reference_voltage_v) ** 2

    def thrust_from_rotor(self, rotor_fraction: float) -> float:
        return self.full_thrust_n * min(max(rotor_fraction, 0.0), 1.0) ** 2

    def throttle_for_thrust(self, thrust_n: float, bus_voltage_v: float) -> float:
        """Invert T = T_full (duty V_bus / V_ref)^2 (battery_lipo.update_motor)."""
        rotor = math.sqrt(max(thrust_n, 0.0) / self.full_thrust_n)
        return rotor * self.reference_voltage_v / max(bus_voltage_v, 1e-3)

    def energy_model(self) -> EnergyModel | None:
        if self.shaft_power_at_max_w <= 0.0:
            return None
        return EnergyModel(thrust_ref_n=self.full_thrust_n,
                           power_ref_w=self.shaft_power_at_max_w / self.motor_efficiency,
                           auxiliary_w=self.auxiliary_power_w)


def _vee_error(R_des: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Geometric attitude error 0.5 vee(R_d^T R - R^T R_d), body frame (Lee et al., 2010)."""
    E = R_des.T @ R - R.T @ R_des
    return 0.5 * np.array([E[2, 1], E[0, 2], E[1, 0]])


class ConvexGuidanceController(BaseController):
    """Single-vehicle controller (the SOCP is solved per vehicle)."""

    def __init__(self, settings: dict, vehicle: VehicleModel, pad_position, touchdown_height_m: float,
                 dt: float, max_command_angle: float = 0.262, servo_deadband_rad: float = 0.0):
        super().__init__(settings)
        if dt <= 0.0:
            raise ValueError('Controller dt must be positive')
        self.vehicle, self.dt = vehicle, float(dt)
        g, tr, att = settings['guidance'], settings['tracking'], settings['attitude']
        self.g, self.tr, self.att = g, tr, att
        self.spool_rate = float(settings['spool_up']['throttle_rate_per_s'])
        self.pad = np.asarray(pad_position, dtype=float)
        self.touchdown_z = float(self.pad[2] + touchdown_height_m)
        self.touchdown_speed = float(g['touchdown_speed_m_s'])
        self.gate = LandingGate(position=(self.pad[0], self.pad[1], self.touchdown_z + float(g['gate_height_m'])),
                                velocity=(0.0, 0.0, -self.touchdown_speed),
                                apex=(self.pad[0], self.pad[1], self.touchdown_z))
        self._servo_limit = float(max_command_angle)
        # Attitude efforts stay inside the vane limit; the deadband inverse may
        # add its deadband on top, up to the servo's range (validate_action).
        self._vane_limit = min(float(att['max_fin_angle']), self._servo_limit)
        self._max_fin_angle = self._servo_limit
        self._mixer = PIDFinMixer(max_fin_angle=self._vane_limit)
        self._law = str(att.get('law', 'pd'))
        if self._law not in ('pd', 'lqr'):
            raise ValueError(f"attitude.law must be 'pd' or 'lqr', not {self._law!r}")
        self._lqr = self._attitude_lqr(vehicle, att, dt) if self._law == 'lqr' else None
        # Vane servo deadband inverse (_deadband_inverse): the servo's
        # deadband is a hardware property, the compensated share a setting.
        self._vane_deadband = float(att.get('deadband_compensation', 0.0)) * max(float(servo_deadband_rad), 0.0)
        if self._vane_deadband > 0.0 and not 0.0 < float(att.get('deadband_smoothing', 0.25)) <= 1.0:
            raise ValueError('attitude.deadband_smoothing must be within (0, 1]')
        self._deadband_band = float(att.get('deadband_smoothing', 0.25)) * self._vane_deadband
        # The attitude gains are vane efforts for the plant they were tuned
        # on (reference authority). The same torque on a plant with weaker
        # vanes needs proportionally more deflection.
        self._rp_scale = self._authority_ratio(att.get('reference_vane_authority_nm_per_rad'),
                                               vehicle.vane_authority_nm_per_rad)
        self._yaw_scale = self._authority_ratio(att.get('reference_yaw_authority_nm_per_rad'),
                                                vehicle.yaw_authority_nm_per_rad)
        slew = self._precession_rate(att.get('slew_rate_authority_fraction'))
        self._slew_rate = math.inf if slew is None else slew
        W = vehicle.weight_n
        self.thrust_min_n = float(g['thrust_min_weight_fraction']) * W
        self.track_thrust_min_n = float(tr['thrust_min_weight_fraction']) * W
        self.throttle_rate = float(g['throttle_rate_per_s'])
        self._yaw_reserve = self._yaw_travel(self._duty_rate(vehicle.reference_voltage_v), vehicle.reference_voltage_v)
        self.lead_s = float(g.get('thrust_lead_s', 0.15))
        self._limits = GuidanceLimits(
            mass_kg=vehicle.mass_kg, thrust_min_n=self.thrust_min_n,
            thrust_max_n=self._planning_ceiling(vehicle.full_thrust_n, strict=True),
            thrust_rate_n_s=self._thrust_rate(vehicle.reference_voltage_v),
            max_tilt_rad=math.radians(float(g['max_tilt_deg'])),
            max_speed_m_s=float(g['max_speed_m_s']),
            glide_slope_rad=(None if g.get('glide_slope_deg') is None else math.radians(float(g['glide_slope_deg']))),
            glide_slope_final_s=float(g.get('glide_slope_final_s', 3.0)),
            tilt_rate_rad_s=self._tilt_rate_limit(),
            route_floor_m=float(g['route_floor_m']), gravity=vehicle.gravity)
        self.guidance = ConvexGuidance(
            self._limits, vehicle.energy_model(), objective=str(g['objective']),
            landing_nodes=int(g['landing_nodes']), route_dt_s=float(g['route_dt_s']),
            max_leg_nodes=int(g['max_leg_nodes']), flypass_capture_fraction=float(g['flypass_capture_fraction']),
            max_solve_time_s=float(g['max_solve_time_s']),
            corridor_m=None if g.get('route_corridor_m') is None else float(g['route_corridor_m']),
            corridor_weight=float(g.get('route_corridor_weight', 10.0)))
        self.reset()

    def _planning_ceiling(self, available_n: float, strict: bool = False) -> float:
        """rho2: weight plus the unreserved share of the thrust above weight.

        `strict` rejects a vehicle that cannot out-thrust its weight at the
        reference voltage. In flight a sagging pack degrades the ceiling to
        just above weight instead; the soft-terminal fallback may still use
        everything available.
        """
        weight = self.vehicle.weight_n
        if strict and available_n <= 1.02 * weight:
            raise ValueError(f'Convex guidance needs thrust above weight: {available_n:.2f} N available for '
                             f'{weight:.2f} N weight')
        return max(1.01 * weight,
                   weight + (1.0 - float(self.g['thrust_excess_reserve_fraction'])) * (available_n - weight))

    def _planning_floor(self, ceiling_n: float) -> float:
        """rho1: never plan to accelerate downward harder than the plan can brake.

        W - rho1 <= ratio (rho2 - W). A 6S pack (T/W ~1.15) planned at 0.6 W
        fell at 3.9 m/s^2 against 0.7 m/s^2 of braking; its current-limited
        spool-up then braked too late (Isaac mission 201f2c58bb9e).
        """
        weight = self.vehicle.weight_n
        ratio = float(self.g['descent_brake_ratio'])
        return max(self.thrust_min_n, weight - ratio * (ceiling_n - weight))

    def _thrust_rate(self, bus_voltage_v: float, duty_rate: float | None = None) -> float:
        """Thrust-rate bound implied by the duty-rate bound at the lowest planned duty.

        T = T_full r^2 with rotor fraction r = duty V_bus / V_ref, so
        dT/dt = 2 T_full r (V_bus / V_ref) d(duty)/dt, smallest at T_min.
        """
        ratio = bus_voltage_v / self.vehicle.reference_voltage_v
        rate = self._duty_rate(bus_voltage_v) if duty_rate is None else duty_rate
        return 2.0 * math.sqrt(self.thrust_min_n * self.vehicle.full_thrust_n) * ratio * rate

    def _duty_rate(self, bus_voltage_v: float) -> float:
        """Duty slew whose rotor reaction stays inside a share of the vanes' yaw authority.

        Slewing the duty at d changes rotor speed at omega_max (V_bus/V_ref) d,
        a reaction torque I_rotor omega_max (V_bus/V_ref) d about the thrust
        axis. Only the vanes' common mode can hold it: K_yaw f^2 delta_max at
        hover. The configured throttle_rate_per_s (0.25 /s, set from the
        waypoint_flight analysis) is kept whenever it is smaller. Legacy
        vanes: always. Momentum-bounded vanes: ~0.09 /s. There, 0.25 /s at the
        gate spun the body to 34 deg/s with every vane saturated in common
        mode, and roll/pitch fell into a 2.5 Hz nutation oscillation (Isaac
        mission 8abd7233a4ec: 0.29 m/s touchdown, CRASHED).
        """
        fraction = self.g.get('throttle_rate_yaw_authority_fraction')
        v = self.vehicle
        if not fraction or v.yaw_authority_nm_per_rad <= 0.0 or v.rotor_inertia <= 0.0 or v.omega_max <= 0.0:
            return self.throttle_rate
        rotor = v.hover_throttle          # rotor fraction at hover, reference voltage
        torque = float(fraction) * v.yaw_authority_nm_per_rad * rotor * rotor * self._vane_limit
        if self._lqr is not None:
            # The yaw loop is designed around its design torque, however
            # strong the vanes: the legacy vanes' 0.25 /s slew (8x that
            # torque) spun the body and wobbled roll/pitch at 13-23 deg/s
            # in the offline replica; at the momentum vanes' rate, 2-7 deg/s.
            torque = min(torque, float(fraction) * float(self.att['lqr']['yaw_design_torque_nm']))
        reaction = v.rotor_inertia * v.omega_max * max(bus_voltage_v, 1e-3) / v.reference_voltage_v
        return min(self.throttle_rate, torque / reaction)

    def _yaw_travel(self, duty_rate: float, bus_voltage_v: float) -> float:
        """Common-mode vane travel that holds the rotor's reaction at this duty slew (0 if unknown).

        Momentum-bounded vanes: 40% of the travel (by construction of
        _duty_rate). Legacy vanes: ~1.3 deg, which roll/pitch never contest.
        """
        v = self.vehicle
        if v.yaw_authority_nm_per_rad <= 0.0 or v.rotor_inertia <= 0.0 or v.omega_max <= 0.0:
            return 0.0
        rotor = v.hover_throttle
        reaction = v.rotor_inertia * v.omega_max * bus_voltage_v / v.reference_voltage_v * duty_rate
        return min(self._vane_limit, reaction / (v.yaw_authority_nm_per_rad * rotor * rotor))

    @staticmethod
    def _attitude_lqr(vehicle: VehicleModel, att: dict, dt: float) -> GyroAttitudeLQR:
        v = vehicle
        if v.inertia_rp_kg_m2 <= 0.0 or v.vane_authority_nm_per_rad <= 0.0 or v.servo_lag_s <= 0.0:
            raise ValueError("attitude.law 'lqr' needs the vehicle's inertia, vane authority and servo lag")
        w = att['lqr']
        plant = AttitudePlant(
            inertia_rp_kg_m2=v.inertia_rp_kg_m2, inertia_yaw_kg_m2=v.inertia_yaw_kg_m2,
            rp_authority_nm_per_rad=v.vane_authority_nm_per_rad, yaw_authority_nm_per_rad=v.yaw_authority_nm_per_rad,
            rotor_momentum_max_nms=v.rotor_inertia * v.omega_max, servo_lag_s=v.servo_lag_s,
            joint_lag_s=v.vane_joint_lag_s, damping_nms_per_rad=v.angular_damping_nm_s_per_rad)
        # Bryson's rule on the input: R = 1 / (design torque)^2.
        weights = LQRWeights(error=float(w['error_weight']), rate=float(w['rate_weight']),
                             integral=float(w['integral_weight']), torque=float(w['design_torque_nm']) ** -2,
                             yaw_rate=float(w['yaw_rate_weight']), yaw_torque=float(w['yaw_design_torque_nm']) ** -2,
                             yaw_integral=float(w.get('yaw_integral_weight', 0.0)))
        return GyroAttitudeLQR(plant, weights, dt)

    @staticmethod
    def _authority_ratio(reference, actual) -> float:
        """Effort scale from the gains' reference vane authority to this plant's (1 if either is unknown)."""
        if not reference or actual <= 0.0:
            return 1.0
        return float(reference) / float(actual)

    def _precession_rate(self, fraction) -> float | None:
        """`fraction` of the slew rate full vane deflection sustains against the rotor at hover.

        Tilting the body at rate w precesses the rotor's angular momentum H,
        which takes a vane torque H w. At hover the vanes' largest torque is
        K f^2 delta_max against H = I_rotor omega_max f.
        """
        v = self.vehicle
        if not fraction or v.vane_authority_nm_per_rad <= 0.0 or v.rotor_inertia <= 0.0 or v.omega_max <= 0.0:
            return None
        rotor = v.hover_throttle          # rotor fraction at hover, reference voltage
        torque = v.vane_authority_nm_per_rad * rotor * rotor * self._vane_limit
        return float(fraction) * torque / (v.rotor_inertia * v.omega_max * rotor)

    def _tilt_rate_limit(self) -> float | None:
        """Planned thrust-direction slew the vanes can drive against the rotor's momentum.

        Tilting the body at rate w precesses the rotor's angular momentum H,
        which takes a vane torque H w. At hover the vanes' largest torque is
        K f^2 delta_max against H = I_rotor omega_max f, so the plan may use
        `tilt_rate_authority_fraction` of K f delta_max / (I_rotor omega_max),
        leaving the rest for disturbance rejection. Legacy vanes: ~83 deg/s
        (inactive). Momentum-bounded vanes: ~10 deg/s. There, planned tilts
        at the thrust-rate bound (~30 deg/s) saturated every vane and lost
        the vehicle in the offline replica (2026-09-25).
        """
        return self._precession_rate(self.g.get('tilt_rate_authority_fraction'))

    def reset(self, env_ids: Tensor | None = None) -> None:
        self._t = 0.0
        self._phase = None
        self._plan = None
        self._plan_t0 = 0.0
        self._previous_plan = None      # (plan, t0) replaced at the last re-plan: feedforward cross-fade
        self._plan_id = 0
        self._plan_route_len = None
        self._new_plan = False
        self._plan_origin = None
        self._b3_des = self._attitude_error = None
        self._last_attempt_t = -math.inf
        self._terminal_z = None
        self._hold_position = None
        self._integral = np.zeros(3)
        self._vz_integral = 0.0
        self._attitude_integral = np.zeros(2)   # LQR: integral of the roll/pitch attitude error
        self._yaw_integral = 0.0                # LQR: integral of the yaw rate (heading drift, rad)
        self._R_plan_prev = None                # LQR: the plan's attitude last step (rate feedforward)
        self._rate_ref = np.zeros(2)            # LQR: filtered desired roll/pitch rate (FRD, rad/s)
        self._touchdown = False
        self._emergency = False
        self._velocity = np.zeros(3)
        self._throttle = None
        self._rotor_command = None      # duty x V_bus / V_ref: the rotor speed the duty asks for
        self._path_key = self._path_curves = None   # route corridor curves for the active leg onward
        self.last_telemetry: dict = {}

    @property
    def phase(self) -> str | None:
        return self._phase

    @property
    def plan(self):
        return self._plan

    # ---- main entry ----

    def compute_action(self, obs: Tensor, *, reference_position=None, route=(), hold_elapsed_s: float = 0.0,
                       bus_voltage_v: float | None = None, path_points=None, path_index: int = 0) -> Tensor:
        """Fin angles and duty for one vehicle.

        Args:
            obs: (1, D) observation; obs[:, :3] is (reference - position).
            reference_position: world point the position error refers to
                (the pad, or the mission sequencer's active goal).
            route: remaining waypoints, active first (mission sequencer).
            hold_elapsed_s: dwell already accrued on an active hover waypoint.
            bus_voltage_v: measured pack voltage (None: ideal reference bus).
            path_points: the mission's drawn route, [start, every waypoint,
                landing target] (world); its Catmull-Rom legs are the route
                corridor. None, or no waypoints: no corridor.
            path_index: the active leg (the sequencer's waypoint index).
        """
        o = obs[0].detach().to('cpu', torch.float64).numpy()
        reference = self.pad if reference_position is None else np.asarray(reference_position, dtype=float)
        position = reference - o[0:3]
        R = to_rotation_matrix(torch.as_tensor(o[3:7])).numpy()
        velocity = R @ (_FRD_TO_ISAAC @ o[7:10])
        rates = o[10:13]
        vanes, vane_rates = o[14:18], o[18:22]
        rotor = float(o[22])
        contact = int(round(float(o[23])))
        volts = self.vehicle.reference_voltage_v if bus_voltage_v is None else float(bus_voltage_v)
        thrust_now = self.vehicle.thrust_from_rotor(rotor)
        if self._throttle is None:
            self._throttle = min(1.0, rotor * self.vehicle.reference_voltage_v / max(volts, 1e-3))
        self._new_plan = False
        waypoints = tuple(self._route(route, hold_elapsed_s))
        path = self._path(path_points, path_index, len(waypoints))

        if contact == int(ContactState.LANDED):
            # The flight executive disarms after LANDED; never push on the pad.
            self._phase = 'LANDED'
            self._throttle = self._rotor_command = 0.0
            action = torch.zeros(1, 5, dtype=obs.dtype, device=obs.device)
            self._record(position, velocity, position, np.zeros(3), np.zeros(3), 0.0, 0.0)
            self._t += self.dt
            return action

        if self._phase is None:
            self._phase = SPOOL_UP if thrust_now < 0.95 * self.thrust_min_n else POWERED_DESCENT
        if self._phase == SPOOL_UP and thrust_now >= 0.95 * self.thrust_min_n:
            self._phase = POWERED_DESCENT

        if self._phase == SPOOL_UP:
            return self._spool_up(obs, R, rates, vanes, vane_rates, volts, position, velocity)

        # Present thrust: magnitude from rotor speed, direction the body axis.
        self._velocity = velocity
        self._maybe_replan(position, velocity, thrust_now * R[:, 2], waypoints, volts, path)
        r_ref, v_ref, u_ff = self._reference(position, waypoints)

        # Feedback around the plan, world frame.
        g_accel = self.vehicle.gravity
        e_r, e_v = r_ref - position, v_ref - velocity
        kp = np.array([self.tr['kp_xy'], self.tr['kp_xy'], self.tr['kp_z']], dtype=float)
        kd = np.array([self.tr['kd_xy'], self.tr['kd_xy'], self.tr['kd_z']], dtype=float)
        ki = np.array([self.tr['ki_xy'], self.tr['ki_xy'], self.tr['ki_z']], dtype=float)
        # Integrator states are bounded by the acceleration they may command,
        # and only integrate near steady flight (near-hover feedforward, small
        # error): they must learn steady disturbances (CoM trim, wind, thrust
        # bias), not maneuver lag, which they carried into the terminal phase
        # as a bias (100% lateral overshoot, mission 4df48c2427e7).
        authority = float(self.tr['integral_limit_m_s2'])
        limit = authority / np.maximum(ki, 1e-9)
        steady = float(np.linalg.norm(u_ff - np.array([0.0, 0.0, g_accel]))) <= float(self.tr['integrate_max_accel_m_s2'])
        # The integrated error saturates instead of switching off, so a large
        # standing offset is still removed, only at a bounded rate.
        window = float(self.tr['integrate_max_error_m'])
        e_int = e_r.copy()
        horizontal = float(np.linalg.norm(e_int[:2]))
        if horizontal > window:
            e_int[:2] *= window / horizontal
        e_int[2] = min(max(e_int[2], -window), window)
        candidate = np.clip(self._integral + (e_int if steady else 0.0) * self.dt, -limit, limit)
        correction = kp * e_r + kd * e_v + ki * candidate
        if self._phase == TERMINAL_DESCENT:
            # Velocity-commanded descent: position error only shapes a
            # clamped rate command, so neither a thrust bias nor a lagging
            # reference can stall the descent or speed it past the gate.
            # Its integrator acts on the rate error; it is seeded at the gate
            # with the position integrator's thrust-bias estimate.
            candidate[2] = 0.0
            v_cmd = min(max(v_ref[2] + kp[2] / kd[2] * e_r[2], -float(self.g['terminal_max_descent_m_s'])),
                        float(self.g['terminal_max_climb_m_s']))
            rate_error = v_cmd - float(velocity[2])
            vz_limit = authority / float(self.tr['ki_vz'])
            if abs(rate_error) <= float(self.tr['integrate_max_rate_error_m_s']):
                # Conditional integration: a bounce off the gate wound this up
                # and overshot the descent to 0.32 m/s (mission ff68c9c5430e).
                self._vz_integral = min(max(self._vz_integral + rate_error * self.dt, -vz_limit), vz_limit)
            correction[2] = kd[2] * rate_error + float(self.tr['ki_vz']) * self._vz_integral
            # Land detector: legs loaded (contact candidate). Latch it, so a
            # bounce does not return the vehicle to steering while it skids
            # (mission d3869dbed58c); release only if it climbs clear again.
            commit = self.touchdown_z + float(self.g['terminal_commit_height_m'])
            if contact == int(ContactState.GROUND_CONTACT_CANDIDATE):
                self._touchdown = True
            elif float(position[2]) > commit + 0.1:
                self._touchdown = False
            if self._touchdown:
                # Stop steering against the pad and unload the rotor so the
                # contact dwell completes.
                correction = np.array([0.0, 0.0, -g_accel * (1.0 - float(self.g['touchdown_thrust_weight_fraction']))])
                u_ff = np.array([0.0, 0.0, g_accel])
        cap = float(self.tr['max_correction_m_s2'])
        horizontal = np.linalg.norm(correction[:2])
        if horizontal > cap:
            correction[:2] *= cap / horizontal
        correction[2] = min(max(correction[2], -cap), cap)
        thrust_vec = self.vehicle.mass_kg * (u_ff + correction)

        available = self.vehicle.available_thrust_n(volts)
        tilt_limit = float(self.g['terminal_max_tilt_deg'] if self._phase == TERMINAL_DESCENT
                           else self.tr['max_tilt_deg'])
        thrust_vec, limited = self._limit_thrust(thrust_vec, available, tilt_limit)
        if not limited:
            self._integral = candidate
        b3_des = thrust_vec / np.linalg.norm(thrust_vec)
        # Rate feedforward direction: the plan's thrust direction plus a
        # share of the feedback correction (see _update_rate_reference). Only
        # an optimal plan's turn is fed forward: soft-terminal fallbacks (no
        # feasible plan) swing between re-plans, and chasing them tipped a 6S
        # vehicle (T/W 1.15) 14 deg off its command (Isaac 766270f3320d).
        plan_dir = (u_ff / np.linalg.norm(u_ff) if self._plan is not None and self._plan.mode == 'optimal'
                    else np.array([0.0, 0.0, 1.0]))
        share = float(self.att['lqr'].get('feedback_rate_fraction', 0.0)) if self._lqr is not None else 0.0
        b3_ref = (1.0 - share) * plan_dir + share * b3_des
        fins = self._attitude_fins(R, b3_des, rates, rotor, vanes, vane_rates, b3_ref=b3_ref / np.linalg.norm(b3_ref))
        self._b3_des, self._attitude_error = b3_des, math.degrees(math.acos(min(1.0, max(-1.0, float(b3_des @ R[:, 2])))))

        # Throttle realizes the thrust component along the actual body axis.
        # A soft-terminal plan means no safe plan exists: the duty may then
        # slew as fast as a spool-up, trading a yaw transient for braking.
        along = max(float(thrust_vec @ R[:, 2]), self.track_thrust_min_n)
        duty = self.vehicle.throttle_for_thrust(along, volts)
        # The fast slew stays latched after the fallback until the duty has
        # converged: dropping to the yaw-safe rate at full braking thrust
        # ballooned the vehicle to 2 m (mission 0a222db0855b).
        if self._phase != HOLD and self._plan is not None and self._plan.mode == 'soft_terminal':
            self._emergency = True
        elif abs(duty - self._throttle) < 0.03:
            self._emergency = False
        self._slew_duty(duty, self.spool_rate if self._emergency else self._duty_rate(volts), volts)
        action = torch.tensor([[*fins, self._throttle]], dtype=obs.dtype, device=obs.device)
        tilt = math.degrees(math.acos(min(1.0, max(-1.0, b3_des[2]))))
        self._record(position, velocity, r_ref, v_ref, thrust_vec, tilt, available)
        self._t += self.dt
        return self.validate_action(action)

    # ---- phases ----

    def _spool_up(self, obs, R, rates, vanes, vane_rates, volts, position, velocity):
        """Cold rotor: hold level and ramp the duty toward hover, then plan."""
        fins = self._attitude_fins(R, np.array([0.0, 0.0, 1.0]), rates, float(obs[0, 22]), vanes, vane_rates)
        self._slew_duty(self.vehicle.throttle_for_thrust(self.vehicle.weight_n, volts), self.spool_rate, volts)
        self._record(position, velocity, position, np.zeros(3), np.zeros(3), 0.0,
                     self.vehicle.available_thrust_n(volts))
        self._t += self.dt
        return self.validate_action(torch.tensor([[*fins, self._throttle]], dtype=obs.dtype, device=obs.device))

    def _slew_duty(self, duty, rate_per_s, volts):
        """Move the duty toward `duty`, slewing the rotor command (duty x V_bus / V_ref) at `rate_per_s` x V_bus / V_ref.

        The slew bounds the rotor's acceleration, whose reaction torque yaws
        the body, and the rotor follows the rotor command, not the raw duty: a
        duty change that only offsets a bus-voltage change moves no rotor
        speed, so it passes at once. Slewing the raw duty held that
        compensation back. A warm-rotor start computed its first duty on the
        unloaded 33.6 V pack, the loaded bus sagged 3%, and the rotor spun
        down 1600 rpm, yawing the body to 30 deg/s (Isaac mission 0a04e2549b26).
        """
        ratio = max(volts, 1e-3) / self.vehicle.reference_voltage_v
        command = self._throttle * ratio if self._rotor_command is None else self._rotor_command
        step = rate_per_s * ratio * self.dt
        command = min(max(duty * ratio, command - step, 0.0), command + step, ratio)   # duty <= 1
        self._rotor_command = command
        self._throttle = float(command / ratio)

    @staticmethod
    def _route(route, hold_elapsed_s):
        """Remaining waypoints; an active hover keeps only its unserved hold."""
        result = []
        for i, wp in enumerate(route):
            if isinstance(wp, dict):  # WaypointMission.record() entries
                wp = RouteWaypoint(position=tuple(float(x) for x in wp['position']), kind=wp.get('type', 'flypass'),
                                   radius_m=float(wp.get('radius_m', 1.0)), speed_m_s=float(wp.get('speed_m_s', 3.0)),
                                   hold_s=float(wp.get('hold_s', 0.0)))
            hold = wp.hold_s if wp.kind == HOVER else 0.0
            if i == 0 and wp.kind == HOVER:
                hold = max(0.0, hold - hold_elapsed_s)
            result.append(replace(wp, hold_s=hold))
        return result

    def _path(self, points, index, remaining):
        """The route corridor's curves, one per remaining leg (landing included).

        A direct landing draws no route and keeps the glide-slope approach.
        """
        if points is None or self.guidance.corridor_m is None or len(points) <= 2:
            return None
        key = (tuple(tuple(float(x) for x in p) for p in points), int(index))
        if self._path_key != key:
            self._path_key = key
            self._path_curves = []
            for leg in range(int(index), len(points) - 1):
                # A Catmull-Rom leg can undershoot both of its ends after a
                # steep arrival: the trial route's hover-to-hover leg (6 m and
                # 4.5 m, after the 50 m start) dips to 1.5 m, and the corridor
                # pulled the plan down to the 0.8 m floor at 2.3 m/s (offline
                # replica: touchdown 17 m from the pad). The corridor never
                # runs below the lower end of its leg.
                curve = catmull_rom_leg(points, leg)
                curve[:, 2] = np.maximum(curve[:, 2], min(curve[0, 2], curve[-1, 2]))
                self._path_curves.append(curve)
        if len(self._path_curves) != remaining + 1:
            raise ValueError(f'Route path has {len(self._path_curves)} legs left for {remaining} waypoints')
        return self._path_curves

    def _maybe_replan(self, position, velocity, thrust_now, waypoints, volts, path=None):
        if self._phase == TERMINAL_DESCENT:
            return
        plan, elapsed = self._plan, self._t - self._plan_t0
        period = float(self.g['replan_period_s'])
        if self._phase == HOLD and self._t - self._last_attempt_t < period:
            return
        need = plan is None or self._phase == HOLD or elapsed >= period
        need |= len(waypoints) != self._plan_route_len
        if plan is not None and not need:
            r_ref, _, _ = plan.sample(elapsed)
            need = float(np.linalg.norm(r_ref - position)) > float(self.g['replan_error_m'])
        if (need and plan is not None and not waypoints and len(waypoints) == self._plan_route_len
                and plan.duration - elapsed < float(self.g['freeze_time_s'])):
            need = False
        if not need:
            return
        hint = None
        start = (position, velocity, thrust_now)
        origin = 'measured'
        if plan is not None:
            hint = plan.duration - max(elapsed, plan.landing_start_s)
            if self._phase != HOLD:
                # While the vehicle tracks the plan, re-solve from the plan's
                # own reference state: feedback and integrators keep working
                # on the deviation. Re-solving from the measured state zeroed
                # the tracking error every period, so each new plan started
                # wherever the vehicle had drifted and it orbited the pad
                # (Isaac mission f708e98f6031). Large deviations still
                # re-plan from the measured state.
                r_ref, v_ref, u_ref = plan.sample(elapsed)
                if (np.linalg.norm(r_ref - position) <= float(self.g['replan_error_m'])
                        and np.linalg.norm(v_ref - velocity) <= float(self.g['replan_velocity_error_m_s'])):
                    start = (r_ref, v_ref, u_ref * self.vehicle.mass_kg)
                    origin = 'reference'
        available = self.vehicle.available_thrust_n(volts)
        ceiling = self._planning_ceiling(available)
        self.guidance.limits = replace(
            self._limits, thrust_rate_n_s=self._thrust_rate(volts),
            thrust_min_n=self._planning_floor(ceiling), thrust_max_n=ceiling,
            emergency_thrust_max_n=available,
            emergency_thrust_rate_n_s=self._thrust_rate(volts, self.spool_rate))
        self._last_attempt_t = self._t
        new = self.guidance.plan(*start, self.gate, waypoints, landing_time_hint=hint, path=path)
        self._plan_origin = origin
        if new is None:
            if plan is None or len(waypoints) != self._plan_route_len:
                if self._phase != HOLD:
                    self._hold_position = position.copy()
                self._phase = HOLD
            return
        self._previous_plan = None if plan is None or self._phase == HOLD else (plan, self._plan_t0)
        self._plan, self._plan_t0 = new, self._t
        self._plan_id += 1
        self._plan_route_len = len(waypoints)
        self._new_plan = True
        if self._phase == HOLD:
            self._phase = POWERED_DESCENT

    def _reference(self, position, waypoints):
        g = self.vehicle.gravity
        hover = np.array([0.0, 0.0, g])
        if self._phase == HOLD:
            return self._hold_position.copy(), np.zeros(3), hover
        plan, elapsed = self._plan, self._t - self._plan_t0
        if self._phase != TERMINAL_DESCENT:
            # Hand over when the plan's clock runs out, or when the vehicle is
            # physically at the gate: re-solving from there only restarts an
            # approach it has already flown.
            at_gate = (abs(float(position[2]) - float(self.gate.position[2])) <= float(self.g['gate_capture_height_m'])
                       and float(np.linalg.norm(position[:2] - self.pad[:2])) <= float(self.g['gate_capture_radius_m'])
                       and float(np.linalg.norm(self._velocity)) <= float(self.g['gate_capture_speed_m_s']))
            if not waypoints and plan.final_target == 'gate' and (elapsed >= plan.duration or at_gate):
                self._phase = TERMINAL_DESCENT
                self._terminal_z = min(float(self.gate.position[2]), float(position[2]))
                # Hand the thrust-bias estimate to the rate integrator:
                # dropping it at the gate (it holds real losses such as
                # deflected-vane drag) sagged the descent to 0.47 m/s in
                # mission d3869dbed58c.
                self._vz_integral = float(self.tr['ki_z']) * self._integral[2] / float(self.tr['ki_vz'])
                self._integral[2] = 0.0
            else:
                self._phase = ROUTE if waypoints else POWERED_DESCENT
                r_ref, v_ref, _ = plan.sample(elapsed)
                _, _, u_ff = plan.sample(elapsed + self.lead_s)
                # A new plan starts at the present thrust but with its own
                # slope, so the lead-sampled feedforward stepped at re-plans:
                # up to 1.4 deg of tilt (mean 0.36 deg against 0.11 deg per
                # step otherwise, 21 re-plans on the route, offline replica),
                # each a small attitude kick. Cross-fade from the replaced plan.
                blend = float(self.g.get('replan_blend_s', 0.0))
                if self._previous_plan is not None and elapsed < blend:
                    old, old_t0 = self._previous_plan
                    _, _, u_old = old.sample(self._t - old_t0 + self.lead_s)
                    weight = elapsed / blend
                    u_ff = (1.0 - weight) * u_old + weight * u_ff
                return r_ref, v_ref, u_ff
        # Constant-rate vertical descent over the pad; pause while off center.
        # The reference never leads the vehicle by more than terminal_lead_m,
        # so a disturbance cannot turn position error into a fast descent:
        # the descent rate stays the velocity reference.
        # Below the commit height the descent never pauses: hovering on the
        # legs' clearance let a tilted leg catch the pad and skid the vehicle
        # (Isaac mission f708e98f6031).
        miss = float(np.linalg.norm(position[:2] - self.pad[:2]))
        center = float(self.g['terminal_center_radius_m'])
        committed = float(position[2]) <= self.touchdown_z + float(self.g['terminal_commit_height_m'])
        scale = 1.0 if committed else min(1.0, max(0.0, 1.0 - (miss - center) / 0.3))
        rate = self.touchdown_speed * scale
        self._terminal_z -= rate * self.dt
        return (np.array([self.pad[0], self.pad[1], self._terminal_z]), np.array([0.0, 0.0, -rate]), hover)

    # ---- inner loops ----

    def _limit_thrust(self, thrust_vec, available, max_tilt_deg):
        """Tilt cone, then magnitude bounds; returns (vector, saturated)."""
        limited = False
        tan_tilt = math.tan(math.radians(max_tilt_deg))
        vertical = max(float(thrust_vec[2]), self.track_thrust_min_n * 0.5)
        horizontal = float(np.linalg.norm(thrust_vec[:2]))
        if thrust_vec[2] != vertical:
            limited = True
        if horizontal > vertical * tan_tilt:
            thrust_vec = np.array([*(thrust_vec[:2] * vertical * tan_tilt / horizontal), vertical])
            limited = True
        else:
            thrust_vec = np.array([thrust_vec[0], thrust_vec[1], vertical])
        magnitude = float(np.linalg.norm(thrust_vec))
        bounded = min(max(magnitude, self.track_thrust_min_n), available)
        if abs(bounded - magnitude) > 1e-9:
            thrust_vec = thrust_vec * bounded / magnitude
            limited = True
        return thrust_vec, limited

    def _gyro_compensation(self, rotor_fraction: float) -> float:
        """Roll/pitch cross-feed gain that cancels the rotor gyroscopic torque.

        Identified plant: I p' = K f^2 e_roll - c p - H q (pitch mirrored),
        H = I_rotor omega_max f. Feeding e_roll += k q cancels -H q when
        k = I_rotor omega_max / (K f). A fixed gyro_comp_rp is used when no
        vane authority is configured.
        """
        authority = self.att.get('vane_rp_authority_nm_per_rad')
        if not authority or self.vehicle.rotor_inertia <= 0.0:
            return float(self.att['gyro_comp_rp'])
        f = max(rotor_fraction, 0.3)
        return self.vehicle.rotor_inertia * self.vehicle.omega_max / (float(authority) * f)

    def _deadband_inverse(self, fins: np.ndarray, measured: np.ndarray) -> np.ndarray:
        """Pre-compensate the vane servos' deadband (a backlash inverse).

        The servo moves only while |command - position| exceeds its deadband
        b, and stops b short of the command, so each vane trails its command
        by up to b. Commanding desired + b sat((desired - measured) / band)
        from the measured servo position lands the vane on the desired angle;
        inside the linear band a vane already there is left alone.

        Evidence (Isaac mission 0289417f5d60, planned 8S, no disturbances):
        vanes froze while their commands swept +/-1 deg, which kept the
        attitude loop in a 1.0-1.1 Hz retrograde coning limit cycle through
        every phase: 3.2 deg tilt, 15 deg/s body-rate RMS, and +/-0.5 m/s of
        lateral velocity (vane side force) that stalled the terminal descent
        into a timeout. An offline replica of that plant reproduced it (3.0
        deg, 14 deg/s), and removing the deadband removed it.
        """
        b = self._vane_deadband
        if b <= 0.0:
            return fins
        offset = b * np.clip((fins - measured) / self._deadband_band, -1.0, 1.0)
        return np.clip(fins + offset, -self._servo_limit, self._servo_limit)

    def _attitude_fins(self, R, b3_des, rates_frd, rotor_fraction, measured_fins=None, measured_fin_rates=None,
                       b3_ref=None):
        """Geometric attitude error -> roll/pitch/yaw vane efforts -> radial-vane mixer.

        Heading is free: the desired frame keeps the current body heading,
        so yaw is rate-controlled only. FRD sign conventions match
        PIDController (validated by tests against its Euler errors). With the
        measured vane angles and rates, the servo deadband is pre-compensated
        from the servo state they imply. `b3_ref`, the direction whose turn
        rate is fed forward, sets the LQR's rate reference.
        """
        R_des = self._frame(R, b3_des)
        error = _FRD_TO_ISAAC.T @ _vee_error(R_des, R)   # Isaac body -> FRD components
        joint = servo = None
        if measured_fins is not None:
            joint = np.asarray(measured_fins, dtype=float)
            # The vane joint trails the servo by a first-order lag, j' = (s - j) / tau_j,
            # so the servo angle is s = j + tau_j j' (Isaac mission e9c203457bba: 0.009 deg RMS).
            servo = joint if measured_fin_rates is None else (
                joint + self.vehicle.vane_joint_lag_s * np.asarray(measured_fin_rates, dtype=float))
        if self._lqr is not None:
            self._update_rate_reference(None if b3_ref is None else self._frame(R, b3_ref), rotor_fraction)
            roll, pitch, yaw_demand, error_used = self._lqr_efforts(error, rates_frd, rotor_fraction, joint, servo)
        else:
            roll, pitch, yaw_demand = self._pd_efforts(error, rates_frd, rotor_fraction)
        # Each vane carries roll or pitch plus the common-mode yaw. Roll/pitch
        # have priority, except for a yaw reserve sized to hold the rotor's
        # reaction to duty changes (_yaw_reserve). On momentum-bounded vanes a
        # body left to spin up lost roll/pitch control: with a CoM trim
        # saturating roll/pitch, the offline replica spun to 400 deg/s. The
        # reserve must not grow past that: reserving the swirl trim as well
        # starved a 5 mm CoM trim, and the vehicle tipped 20 deg (a spinning
        # but upright vehicle recovers; a tilted one does not).
        cap = self._vane_limit - min(abs(yaw_demand), self._yaw_reserve)
        largest = max(abs(roll), abs(pitch))
        saturated = largest > cap
        if saturated:
            roll, pitch = roll * cap / largest, pitch * cap / largest   # keep the torque direction
        free = max(0.0, self._vane_limit - max(abs(roll), abs(pitch)))
        yaw = min(max(yaw_demand, -free), free)
        if self._lqr is not None and not self._touchdown:
            if not saturated:
                self._integrate_attitude(error_used, rotor_fraction)
            if abs(yaw) >= abs(yaw_demand) - 1e-12:   # yaw effort not clipped (anti-windup)
                self._integrate_yaw(float(rates_frd[2]), rotor_fraction)
        efforts = torch.tensor([[roll, pitch, yaw]], dtype=torch.float32)
        fins = self._mixer.mix(efforts[:, 0], efforts[:, 1], efforts[:, 2])[0].double().numpy()
        if servo is not None:
            fins = self._deadband_inverse(fins, servo)
        return [float(x) for x in fins]

    @staticmethod
    def _frame(R, b3):
        """Attitude with body axis b3 that keeps the current heading (heading is free)."""
        b1 = R[:, 0] - (R[:, 0] @ b3) * b3
        if np.linalg.norm(b1) < 1e-6:
            b1 = np.array([1.0, 0.0, 0.0]) - b3[0] * b3
        b1 /= np.linalg.norm(b1)
        return np.column_stack((b1, np.cross(b3, b1), b3))

    def _pd_efforts(self, error, rates_frd, rotor_fraction):
        """PID-baseline fin-effort law (attitude.law 'pd'), scaled to the plant's vane authority."""
        p, q, r = rates_frd
        kp, kd, comp = float(self.att['kp_att']), float(self.att['kd_att']), self._gyro_compensation(rotor_fraction)
        # Cascade form of the PD law: the attitude error sets a body-rate
        # command, clamped to the slew the vanes can sustain against the
        # rotor's momentum, and the rate loop tracks it. Unclamped,
        # -kd (w - w_cmd) with w_cmd = -(kp/kd) e is the PD law exactly.
        command = -(kp / kd) * error[:2]
        norm = float(np.linalg.norm(command))
        if norm > self._slew_rate:
            command *= self._slew_rate / norm
        # Efforts in the reference plant's units, converted to this plant's.
        roll = (-kd * (p - command[0]) + comp * q) * self._rp_scale
        pitch = (-kd * (q - command[1]) - comp * p) * self._rp_scale
        yaw = -float(self.att['yaw_damper_gain']) * r * self._yaw_scale
        return roll, pitch, yaw

    @staticmethod
    def _effort_space(fins):
        """Mixer inverse: vane angles (+X, +Y, -X, -Y) -> (roll, pitch, yaw) efforts."""
        return np.array([(fins[2] - fins[0]) / 2.0, (fins[3] - fins[1]) / 2.0, float(np.mean(fins))])

    def _lqr_efforts(self, error, rates_frd, rotor_fraction, joint, servo):
        """Full-state gyroscopic LQR (attitude.law 'lqr'); returns roll, pitch, yaw and the error used.

        The attitude error is clamped so that its share of the effort stays
        within error_effort_fraction of the vane limit: a large error becomes
        a precession at a bounded rate, and the rest of the travel keeps
        damping the nutation through the rate and actuator-state feedback.
        """
        gain = self._lqr.roll_pitch_gain(rotor_fraction)
        e = np.asarray(error[:2], dtype=float)
        stiffness = float(np.linalg.svd(gain[:, 2:4], compute_uv=False)[0])
        limit = float(self.att['lqr']['error_effort_fraction']) * self._vane_limit / max(stiffness, 1e-9)
        norm = float(np.linalg.norm(e))
        if norm > limit:
            e = e * limit / norm
        j = np.zeros(3) if joint is None else self._effort_space(joint)
        s = j if servo is None else self._effort_space(servo)
        p, q, r = (float(x) for x in rates_frd)
        # Tracking a turning attitude: regulate the rates about the desired
        # rate, and the vanes about the effort that sustains it. A steady
        # precession at (p_d, q_d) takes K f^2 j = (H q_d + c p_d, c q_d - H p_d).
        v = self.vehicle
        f = max(float(rotor_fraction), 0.05)
        rate_ref = self._rate_ref
        hold = np.array([v.rotor_inertia * v.omega_max * f * rate_ref[1] + v.angular_damping_nm_s_per_rad * rate_ref[0],
                         v.angular_damping_nm_s_per_rad * rate_ref[1] - v.rotor_inertia * v.omega_max * f * rate_ref[0]])
        hold /= v.vane_authority_nm_per_rad * f * f
        x = np.concatenate([self._attitude_integral, e, [p - rate_ref[0], q - rate_ref[1]], s[:2] - hold, j[:2] - hold])
        roll, pitch = hold + gain @ x
        yaw = float((self._lqr.yaw_gain(rotor_fraction) @ np.array([self._yaw_integral, r, s[2], j[2]]))[0])
        return float(roll), float(pitch), yaw, e

    def _update_rate_reference(self, R_plan, rotor_fraction):
        """Filtered roll/pitch rate of the reference direction, bounded by the slew the vanes sustain.

        Without it the regulator treats a turning target as a growing error:
        the attitude then lagged the landing leg's tilt commands, and the
        vehicle circled the pad (offline replica, 2026-09-26). The
        reference is the plan's thrust direction plus only a share
        (attitude.lqr.feedback_rate_fraction) of the feedback correction.
        The lateral velocity feedback carries the vanes' own
        (non-minimum-phase) side force: differentiated whole, it closed a
        1.3 Hz loop through the vanes (linear hover model: zeta 0.03 at
        kd_xy 1.6, 0.18 at 1.2; Isaac: 4-5 deg/s of wobble in terminal
        descent). Left out entirely, the weak 6S vanes lagged every
        correction and the vehicle circled the pad (Isaac 766270f3320d).
        """
        previous, self._R_plan_prev = self._R_plan_prev, R_plan
        if R_plan is None:
            self._rate_ref = np.zeros(2)
            return
        if previous is None:
            return
        M = previous.T @ R_plan
        raw = (_FRD_TO_ISAAC.T @ (0.5 * np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])))[:2] / self.dt
        tau = float(self.att['lqr']['rate_reference_filter_s'])
        self._rate_ref = self._rate_ref + (raw - self._rate_ref) * self.dt / (tau + self.dt)
        v = self.vehicle
        f = max(float(rotor_fraction), 0.05)
        bound = float(self.att['lqr']['error_effort_fraction']) * v.vane_authority_nm_per_rad * f * self._vane_limit / max(
            v.rotor_inertia * v.omega_max, 1e-9)
        if self._limits.tilt_rate_rad_s is not None:
            # A plan never turns faster than its tilt-rate bound; anything
            # faster is a re-plan step, not a turn to anticipate.
            bound = min(bound, float(self._limits.tilt_rate_rad_s))
        norm = float(np.linalg.norm(self._rate_ref))
        if norm > bound:
            self._rate_ref = self._rate_ref * bound / norm

    def _integrate_attitude(self, error, rotor_fraction):
        """Anti-windup integration: only on unsaturated steps, of the error saturated at a window.

        The integral holds the trim torque of a CoM offset; its effort is
        bounded by the vane limit. It takes the proportional path's clamped
        error (~2 deg at hover), so a large manoeuvring error cannot wind it
        up: the raw error saturated at 4-5 deg learned the long legs'
        tracking lag and missed the pad by up to 1.8 m (offline replica),
        and gating on the plan's tilt rate kept a mid-air start from ever
        learning a CoM trim. The price is a slow trim from a mid-air start
        (5.8 mm offset: 1.3 s, tipping ~13 deg); a disturbance-torque
        observer would separate trim from manoeuvre lag.
        """
        window = math.radians(float(self.att['lqr']['integrate_max_error_deg']))
        e = np.asarray(error, dtype=float)
        norm = float(np.linalg.norm(e))
        if norm > window:
            e = e * window / norm
        z = self._attitude_integral + e * self.dt
        effort = float(np.linalg.norm(self._lqr.roll_pitch_gain(rotor_fraction)[:, 0:2] @ z))
        if effort > self._vane_limit:
            z *= self._vane_limit / effort
        self._attitude_integral = z

    def _integrate_yaw(self, rate, rotor_fraction):
        """Integral of the yaw rate, its effort bounded by the yaw reserve (anti-windup)."""
        gain = float(self._lqr.yaw_gain(rotor_fraction)[0, 0])
        z = self._yaw_integral + rate * self.dt
        bound = max(self._yaw_reserve, 0.25 * self._vane_limit) / max(abs(gain), 1e-9)
        self._yaw_integral = min(max(z, -bound), bound)

    # ---- telemetry ----

    def _record(self, position, velocity, r_ref, v_ref, thrust_vec, tilt_deg, available):
        plan = self._plan
        elapsed = self._t - self._plan_t0
        record = dict(
            phase=self._phase, plan_id=self._plan_id, replanned=self._new_plan,
            time_to_go_s=None if plan is None or self._phase in (SPOOL_UP, TERMINAL_DESCENT, 'LANDED')
            else round(max(0.0, plan.duration - elapsed), 3),
            reference_position=[round(float(x), 4) for x in r_ref],
            reference_velocity=[round(float(x), 4) for x in v_ref],
            tracking_error_m=round(float(np.linalg.norm(np.asarray(r_ref) - position)), 4),
            thrust_command_n=round(float(np.linalg.norm(thrust_vec)), 3),
            thrust_available_n=round(float(available), 3),
            tilt_command_deg=round(float(tilt_deg), 3),
            thrust_direction=None if self._b3_des is None else [round(float(x), 4) for x in self._b3_des],
            attitude_error_deg=None if self._attitude_error is None else round(self._attitude_error, 3),
            plan_origin=self._plan_origin,
            throttle_command=round(float(self._throttle or 0.0), 4))
        self._b3_des = self._attitude_error = None
        if plan is not None:
            record['solver'] = dict(
                mode=plan.mode, status=plan.status, objective=self.guidance.objective,
                cost=round(plan.cost, 5), energy_wh=round(plan.energy_wh, 4) if math.isfinite(plan.energy_wh) else None,
                delta_v_m_s=round(plan.delta_v_m_s, 4), duration_s=round(plan.duration, 3),
                landing_start_s=round(plan.landing_start_s, 3), solve_ms=round(plan.solve_time_s * 1e3, 2),
                solves=plan.solves, iterations=plan.iterations,
                convexification_gap=round(plan.convexification_gap, 6),
                terminal_miss_m=round(plan.terminal_miss_m, 4),
                corridor_excess_m=round(plan.corridor_excess_m, 3))
        if self._new_plan:
            record['plan'] = dict(
                times=[round(float(t), 3) for t in plan.times],
                positions=[[round(float(x), 3) for x in p] for p in plan.position],
                thrust_n=[round(float(s * self.vehicle.mass_kg), 3) for s in plan.sigma])
        self.last_telemetry = record
