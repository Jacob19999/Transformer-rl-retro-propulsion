// IMU estimate vs PhysX truth. frame.imu is the state the controller acted on
// (run_mission.py imu_record): position/velocity in world XYZ, quaternion wxyz,
// gyro in body FRD. It equals the truth unless sensor noise is enabled.
const deg = 180 / Math.PI;
const sub = (a, b) => a.map((v, i) => v - b[i]);
const norm = v => Math.hypot(...v);
const wrap = a => ((a + 540) % 360) - 180;

export const hasImu = frames => frames.some(f => f.imu);

// Smallest rotation between two unit quaternions, in degrees.
export function attitudeError(a, b) {
  const dot = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);
  return 2 * Math.acos(Math.min(1, dot)) * deg;
}

export function imuError(frame) {
  const imu = frame?.imu;
  if (!imu) return null;
  const position = sub(imu.position, frame.position), velocity = sub(imu.velocity, frame.velocity);
  const gyro = sub(imu.gyro, frame.gyro).map(v => v * deg);
  return { position, distance: norm(position), attitude: attitudeError(imu.quaternion, frame.quaternion),
           velocity, speed: norm(velocity), gyro, rate: norm(gyro) };
}

// Euler deltas (IMU - truth) wrapped to ±180°, from attitude() readings.
export const eulerDelta = (imu, truth) => ['roll', 'pitch', 'yaw'].map(k => wrap(imu[k] - truth[k]));

// One-line description of the recorded sensor model, from metadata.disturbances.
export function sensorNote(metadata) {
  const noise = metadata?.disturbances?.sensor_noise;
  if (!noise?.enabled) return 'Sensor noise off: the IMU reports the PhysX state exactly.';
  const f = (v, d) => Number(v ?? 0).toFixed(d);
  return `White noise per control step · σ position ${f(noise.position_std, 3)} m · attitude ${f((noise.attitude_std ?? 0) * deg, 2)}° · velocity ${f(noise.velocity_std, 3)} m/s · gyro ${f((noise.angular_velocity_std ?? 0) * deg, 2)}°/s`;
}
