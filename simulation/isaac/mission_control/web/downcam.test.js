import test from 'node:test';
import assert from 'node:assert/strict';
import { TEXTURE_LAYERS, cameraSettings, flowArrow, focalPx, makeCamera, markerState, project, rotationMatrix, rulerFraction, textureShade } from './downcam.js';

const level = [1, 0, 0, 0];
const near = (a, b, tol = 1e-6) => assert.ok(Math.abs(a - b) < tol, `${a} vs ${b}`);

test('a level vehicle looks straight down: the ground point below projects to the image centre', () => {
  const cam = makeCamera([0, 0, 4], level, 0);
  const frd = cam.toFrd([0, 0, 0]);
  near(frd[2], 4);
  const p = project(frd, 200, 100, 80);
  near(p[0], 100); near(p[1], 80);
  near(cam.height, 4);
});

test('image right is FRD right (Isaac -y) and image up is forward', () => {
  const cam = makeCamera([0, 0, 4], level, 0);
  const ahead = project(cam.toFrd([1, 0, 0]), 200, 100, 80);
  near(ahead[0], 100); near(ahead[1], 80 - 200 / 4);            // 1 m forward at 4 m: 50 px up
  const right = project(cam.toFrd([0, -1, 0]), 200, 100, 80);
  near(right[0], 100 + 50); near(right[1], 80);
});

test('the camera hangs below the root along the body axis, so a tilted body shifts it', () => {
  const half = Math.PI / 12;                                    // 30 deg roll about body x
  const q = [Math.cos(half), Math.sin(half), 0, 0];
  const cam = makeCamera([0, 0, 4], q, 0.5);
  near(cam.height, 4 - 0.5 * Math.cos(Math.PI / 6));
  const R = rotationMatrix(q);
  near(R[2][2], Math.cos(Math.PI / 6));
});

test('points behind the camera do not project', () => {
  assert.equal(project([0, 0, -1], 200, 100, 80), null);
});

test('focal length follows the field of view', () => {
  near(focalPx(270, 60), 135 / Math.tan(Math.PI / 6), 1e-9);
});

test('marker state separates out of view, outside the altitude window, not detected, rejected and tracking', () => {
  const base = { marker_enabled: true, marker_in_fov: true, marker_in_window: true, marker_valid: true, accepted: { marker: true } };
  assert.equal(markerState(null).key, 'off');
  assert.equal(markerState({ ...base, marker_enabled: false }).key, 'off');
  assert.equal(markerState({ ...base, marker_in_fov: false }).key, 'out_of_view');
  assert.equal(markerState({ ...base, marker_in_window: false, height_est_m: 12 }, { altMaxM: 8, altMinM: 0.3 }).label, 'MARKER VISIBLE · USED BELOW 8 m');
  assert.equal(markerState({ ...base, marker_in_window: false, height_est_m: 0.2 }, { altMaxM: 8, altMinM: 0.3 }).label, 'MARKER OFF BELOW 0.3 m');
  assert.equal(markerState({ ...base, marker_valid: false }).key, 'not_detected');
  assert.equal(markerState({ ...base, accepted: { marker: false } }).key, 'rejected');
  assert.equal(markerState(base).key, 'tracking');
});

test('flow arrow shows the scene moving (right, up) = (-f_x, f_y) and is length-limited', () => {
  const [dx, dy] = flowArrow([-0.5, 0.25], 30, 60);              // scene moves right 0.5 and up 0.25 rad/s
  near(dx, 15); near(dy, -7.5);                                   // screen y is down
  const [lx, ly] = flowArrow([-10, 0], 30, 60);
  near(Math.hypot(lx, ly), 60, 1e-9);
  assert.deepEqual(flowArrow(undefined), [0, 0]);
});

test('height ruler clamps to its range', () => {
  near(rulerFraction(6, 12), 0.5); near(rulerFraction(-1, 12), 0); near(rulerFraction(30, 12), 1);
});

test('camera settings come from the recorded fusion configuration, with defaults', () => {
  const s = cameraSettings({ disturbances: { sensor_noise: { imu: { fusion: { enabled: true, mount_down_m: 0.2,
    marker: { enabled: true, fov_deg: 70, alt_max_m: 6 }, flow: { max_height_m: 9 } } } } } });
  assert.deepEqual([s.enabled, s.markerEnabled, s.fovDeg, s.altMaxM, s.mountDownM, s.maxHeightM], [true, true, 70, 6, 0.2, 9]);
  const d = cameraSettings(undefined);
  assert.deepEqual([d.enabled, d.markerEnabled, d.fovDeg, d.altMaxM], [false, false, 60, 8]);
});

test('the ground texture is fixed to the world: the same cell has the same shade at every altitude', () => {
  for (const { cell, salt } of TEXTURE_LAYERS) {
    const shades = [3, 5, 8, 12].map(() => textureShade(17, -9, salt));
    assert.equal(new Set(shades).size, 1);
  }
  assert.deepEqual(TEXTURE_LAYERS.map(l => l.cell), [1.0, 0.25]);                 // cell sizes are constants, not functions of height
  const varied = new Set(Array.from({ length: 200 }, (_, k) => textureShade(k, 3 * k % 11, 0).toFixed(3)));
  assert.ok(varied.size > 100);                                                     // and the texture actually varies from cell to cell
});

test('rays through the image border hit the ground at heights proportional to the altitude', () => {
  const at = h => makeCamera([0, 0, h], level, 0).groundHit([0, 0.5, 1]);          // 0.5 to the right of the optical axis
  const low = at(2), high = at(8);
  near(low[0], 0); near(low[1], -1);                                                // FRD right is world -y (Isaac y is left)
  near(high[1], -4);
  assert.equal(makeCamera([0, 0, 4], level, 0).groundHit([0, 0, -1]), null);       // straight up never meets the ground
});
