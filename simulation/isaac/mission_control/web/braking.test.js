import test from 'node:test';
import assert from 'node:assert/strict';
import { brakingDistance, checkRoute } from './braking.js';

// mission_control.models.braking_envelopes()['planned_8s/momentum']
const VEHICLE = {mass_kg: 3.104, full_thrust_n: 43.07, rotor_inertia: .0002, omega_max: 4649.56,
  motor_time_constant_s: .15, motor_torque_limit_nm: .76, aero_torque_at_max_nm: .6607, max_tilt_deg: 15};
const start = (position, velocity) => ({position, velocity, attitude_deg: [0, 0, 0], angular_rate_deg_s: [0, 0, 0]});
const waypoint = (position, type = 'flypass') => ({position, type, hold_s: 2, radius_m: 1, speed_m_s: 3});

test('a vertical descent at full rotor brakes at T/m - g', () => {
  const {distance, time} = brakingDistance([0, 0, -10], VEHICLE, {rotor: 1});
  const a = 43.07 / 3.104 - 9.81;
  assert.ok(Math.abs(distance - 100 / (2 * a)) < .1, `${distance}`);
  assert.ok(Math.abs(time - 10 / a) < .02, `${time}`);
});

test('a half-speed rotor must spool up first, and brakes later', () => {
  const warm = brakingDistance([0, 0, -20], VEHICLE, {rotor: .84}).distance;
  const half = brakingDistance([0, 0, -20], VEHICLE, {rotor: .5}).distance;
  assert.ok(half > warm + 10, `${half} vs ${warm}`);
});

test('horizontal braking is bounded by the tilt cone, not the thrust', () => {
  const {distance} = brakingDistance([5, 0, 0], VEHICLE, {rotor: 1});
  assert.ok(Math.abs(distance - 25 / (2 * 9.81 * Math.tan(15 * Math.PI / 180))) < .1, `${distance}`);
});

test('Isaac mission 835c3de32185: the fast start carries the vehicle far off its first leg', () => {
  // Flown: 33.7 m off the drawn route (17 m with the route corridor, offline).
  const result = checkRoute(start([19.3, 27.9, 94.3], [-4, -9, -20]), [waypoint([2.4, 23, 32.5])], VEHICLE, .5);
  assert.equal(result.warnings.length, 1);
  assert.match(result.warnings[0].text, /START TOO FAST FOR WAYPOINT 1/);
  assert.ok(result.warnings[0].off > 15 && result.warnings[0].off < 40, `${result.warnings[0].off}`);
  assert.ok(result.stop[2] > 0 && result.stop[2] < 32.5, `${result.stop}`);   // sinks below the waypoint
});

test('the default landing and a gentle route raise no warning', () => {
  assert.deepEqual(checkRoute(start([-.28, .82, 18], [0, 0, -1]), [], VEHICLE, .84).warnings, []);
  const route = [waypoint([6, 3, 6]), waypoint([8, -3, 5], 'hover')];
  assert.deepEqual(checkRoute(start([0, 0, 8], [0, 0, 0]), route, VEHICLE, .84).warnings, []);
});

test('a descent too fast to stop above the ground is critical', () => {
  const result = checkRoute(start([0, 0, 10], [0, 0, -15]), [], VEHICLE, .84);
  assert.equal(result.warnings[0].level, 'critical');
  assert.match(result.warnings[0].text, /CANNOT STOP ABOVE THE GROUND/);
});

test('a route that dips below both ends of a leg is flagged', () => {
  // The 120 s trial: the hover-to-hover leg at 6 m and 4.5 m dips to 1.5 m.
  const route = [waypoint([-.5, 23.6, 6], 'hover'), waypoint([7.3, -1.4, 4.5], 'hover'), waypoint([-.1, -29.9, 18.6], 'hover')];
  const {warnings} = checkRoute(start([-1.2, .8, 50.4], [0, 0, -.5]), route, VEHICLE, .84);
  assert.equal(warnings.length, 1);
  assert.match(warnings[0].text, /ROUTE DIPS TO 1\.5 M BEFORE WAYPOINT 2 · CONVEX HOLDS 4\.5 M/);
});
