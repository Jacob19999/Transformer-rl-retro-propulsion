import test from 'node:test';
import assert from 'node:assert/strict';
import { attitudeError, eulerDelta, hasImu, imuError, sensorNote } from './imu.js';

const level = [1, 0, 0, 0];
const frame = (imu) => ({ position: [1, 2, 3], velocity: [0, 0, -1], quaternion: level, gyro: [0, 0, 0], imu });

test('replays recorded before the IMU channel have no overlay data', () => {
  assert.equal(hasImu([frame(undefined)]), false);
  assert.equal(imuError(frame(undefined)), null);
  assert.equal(hasImu([frame(undefined), frame({})]), true);
});

test('error is IMU minus actual in every channel', () => {
  const half = Math.PI / 360;  // 1° about body X
  const e = imuError(frame({ position: [1.03, 2, 2.96], velocity: [0, .1, -1], quaternion: [Math.cos(half), Math.sin(half), 0, 0], gyro: [Math.PI / 180, 0, 0] }));
  assert.deepEqual(e.position.map(v => +v.toFixed(6)), [.03, 0, -.04]);
  assert.ok(Math.abs(e.distance - .05) < 1e-9);
  assert.ok(Math.abs(e.attitude - 1) < 1e-9);
  assert.ok(Math.abs(e.speed - .1) < 1e-9);
  assert.ok(Math.abs(e.gyro[0] - 1) < 1e-9 && Math.abs(e.rate - 1) < 1e-9);
});

test('attitude error treats q and -q as the same orientation', () => {
  assert.equal(attitudeError(level, [-1, 0, 0, 0]), 0);
  assert.equal(attitudeError(level, level), 0);
});

test('euler deltas wrap across ±180°', () => {
  assert.deepEqual(eulerDelta({ roll: 1, pitch: 0, yaw: -179 }, { roll: 0, pitch: 0, yaw: 179 }), [1, 0, 2]);
});

test('sensor note reports noise sigmas in display units', () => {
  assert.match(sensorNote({ disturbances: { sensor_noise: { enabled: false } } }), /noise off/);
  assert.match(sensorNote({ disturbances: { sensor_noise: { enabled: true, position_std: .01, attitude_std: Math.PI / 180, velocity_std: .05, angular_velocity_std: Math.PI / 90 } } }),
               /position 0\.010 m · attitude 1\.00° · velocity 0\.050 m\/s · gyro 2\.00°\/s/);
});

test('sensor note describes the physical IMU chain when the simulator ran it', () => {
  const noise = { enabled: true, position_std: .01, velocity_std: .05, attitude_std: 0, angular_velocity_std: 0,
                  imu: { enabled: true, sample_rate_hz: 100, bandwidth_hz: 116, latency_s: .005, gyro: { range_dps: 2000 } } };
  const note = sensorNote({ disturbances: { sensor_noise: noise } });
  assert.match(note, /Physical IMU chain/);
  assert.match(note, /100 Hz output · 116 Hz bandwidth · 5\.0 ms latency · gyro ±2000°\/s/);
  assert.match(note, /position 0\.010 m \/ velocity 0\.050 m\/s external noise/);
  assert.match(sensorNote({ disturbances: { sensor_noise: { ...noise, imu: { enabled: false } } } }), /White noise/);
  const inertial = sensorNote({ disturbances: { sensor_noise: { ...noise, imu: { ...noise.imu, nav: { enabled: true } } } } });
  assert.match(inertial, /inertial navigation \(position and velocity integrated/);
  assert.doesNotMatch(inertial, /external noise/);
  const fused = sensorNote({ disturbances: { sensor_noise: { ...noise, imu: { ...noise.imu, fusion: { enabled: true } } } } });
  assert.match(fused, /sensor fusion \(EKF3-style filter: IMU \+ rangefinder \+ optical flow \+ barometer\)/);
  const marked = sensorNote({ disturbances: { sensor_noise: { ...noise, imu: { ...noise.imu, fusion: { enabled: true, marker: { enabled: true } } } } } });
  assert.match(marked, /optical flow \+ barometer \+ pad marker\)/);
});
