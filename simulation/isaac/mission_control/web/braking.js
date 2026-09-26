// Launch-form check: where does the start velocity carry the vehicle before
// it can follow the route? A physics estimate for any controller, not the
// planner's solution: straight-line braking at full thrust inside the convex
// planner's tilt cone, once the rotor has spooled up from its start speed.
// Vehicle numbers come from /api/config (mission_control.models.braking_envelopes).
import { samplePlannerSpline } from './planner.js';

const G = 9.81, DT = .01;

// One step of the EDF spool at full duty: the first-order lag, and on the
// momentum-vane plant the motor torque bound against the rotor's drag.
function spool(omega, v) {
  const command = (v.omega_max - omega) / v.motor_time_constant_s;
  if (v.motor_torque_limit_nm == null) return Math.min(v.omega_max, omega + command * DT);
  const drag = v.aero_torque_at_max_nm * (omega / v.omega_max) ** 2;
  const torque = Math.min(v.motor_torque_limit_nm, v.rotor_inertia * command + drag);
  return Math.min(v.omega_max, omega + (torque - drag) / v.rotor_inertia * DT);
}

// Braking along the velocity to `arrival` m/s: distance and time. The net
// acceleration is -a along the velocity, so the thrust is u = g z - a d; the
// largest a keeps |u| <= T/m and tilt(u) <= the cone.
export function brakingDistance(velocity, vehicle, {arrival = 0, rotor = 1} = {}) {
  const speed = Math.hypot(...velocity);
  if (speed <= arrival) return {distance: 0, time: 0};
  const [dx, dy, dz] = velocity.map(x => x / speed), horizontal = Math.hypot(dx, dy);
  const tan = Math.tan(vehicle.max_tilt_deg * Math.PI / 180), cone = horizontal + tan * dz;
  let omega = Math.max(0, Math.min(1, rotor)) * vehicle.omega_max, s = speed, distance = 0, time = 0;
  while (s > arrival && time < 60) {
    const sigma = vehicle.full_thrust_n * (omega / vehicle.omega_max) ** 2 / vehicle.mass_kg;
    let a = G * dz + Math.sqrt(Math.max(0, G * G * (dz * dz - 1) + sigma * sigma));
    if (cone > 1e-9) a = Math.min(a, G * tan / cone);
    s -= a * DT; distance += Math.max(s, 0) * DT; time += DT;
    omega = spool(omega, vehicle);
  }
  return {distance, time};
}

// Warnings for the launch form: the vehicle cannot stop above the ground,
// the start velocity carries it off the route's first leg, or the drawn
// route dips below the lower end of a waypoint leg (the convex corridor
// holds that end instead).
export function checkRoute(initial, waypoints, vehicle, rotor) {
  const result = {warnings: [], stop: null, braking: null};
  const [px, py, pz] = initial.position, velocity = initial.velocity, speed = Math.hypot(...velocity);
  if (vehicle && pz > 1 && speed > .5) {
    const first = waypoints[0];
    const arrival = first?.type === 'flypass' ? Math.min(first.speed_m_s, speed) : 0;
    const braking = brakingDistance(velocity, vehicle, {arrival, rotor});
    const dir = velocity.map(x => x / speed), stop = [px, py, pz].map((x, i) => x + dir[i] * braking.distance);
    result.braking = braking; result.stop = stop;
    if (stop[2] < .5) {
      result.warnings.push({level: 'critical', text: `CANNOT STOP ABOVE THE GROUND · NEEDS ${braking.distance.toFixed(0)} M`});
    } else if (first) {
      const leg = first.position.map((x, i) => x - initial.position[i]), length = Math.hypot(...leg);
      const unit = leg.map(x => x / Math.max(length, 1e-9)), travel = dir.map(x => x * braking.distance);
      const along = travel.reduce((sum, x, i) => sum + x * unit[i], 0);
      const across = Math.hypot(...travel.map((x, i) => x - along * unit[i]));
      const off = Math.max(across, along - length);
      if (off > Math.max(first.radius_m, 1)) result.warnings.push({level: 'warning', off,
        text: `START TOO FAST FOR WAYPOINT 1 · ~${off.toFixed(0)} M OFF ROUTE`});
    }
  }
  if (waypoints.length) {
    const path = samplePlannerSpline(initial.position, waypoints), ends = [initial.position, ...waypoints.map(w => w.position)];
    for (let leg = 0; leg < waypoints.length; leg++) {
      const low = Math.min(...path.slice(leg * 49, leg * 49 + 49).map(p => p[2])), floor = Math.min(ends[leg][2], ends[leg + 1][2]);
      if (low < floor - 1) result.warnings.push({level: 'warning',
        text: `ROUTE DIPS TO ${low.toFixed(1)} M BEFORE WAYPOINT ${leg + 1} · CONVEX HOLDS ${floor.toFixed(1)} M`});
    }
  }
  return result;
}
