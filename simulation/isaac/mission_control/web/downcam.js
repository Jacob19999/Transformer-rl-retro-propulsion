// CAM 05: what the downward camera (MTF-01P optical flow + pad-marker detector) sees, and the height
// each sensor reports. Drawn from the recorded frame: physics pose (frame.position / quaternion), the
// fusion telemetry (frame.fusion: sensor readings, their truth, marker visibility) and the run metadata
// (camera field of view, marker size, altitude window).
//
// Camera frame is body FRD: x forward (image up), y right (image right), z down (optical axis).

const RAD = Math.PI / 180;
export const DEFAULTS = { fovDeg: 60, markerSizeM: 0.4, altMaxM: 8, altMinM: 0.3, mountDownM: 0.15, padSizeM: 1.6, maxHeightM: 12 };

// Isaac wxyz quaternion (body -> world) as a row-major 3x3.
export function rotationMatrix([w, x, y, z]) {
  return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
          [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
          [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]];
}

// A world point expressed in the camera's FRD axes. The camera hangs mountDownM below the root along body z.
export function makeCamera(position, quaternion, mountDownM = DEFAULTS.mountDownM) {
  const R = rotationMatrix(quaternion);
  const cam = [0, 1, 2].map(i => position[i] - mountDownM * R[i][2]);      // root + R * (0, 0, -d) in Isaac body axes
  return {
    toFrd(point) {
      const v = [point[0] - cam[0], point[1] - cam[1], point[2] - cam[2]];
      const body = [0, 1, 2].map(j => R[0][j] * v[0] + R[1][j] * v[1] + R[2][j] * v[2]);   // R^T v, Isaac body axes
      return [body[0], -body[1], -body[2]];                                              // to FRD
    },
    height: cam[2],
    origin: cam,
    // Where a ray leaving the camera along an FRD direction meets the ground (z = 0); null if it points at the sky.
    groundHit(dir) {
      const body = [dir[0], -dir[1], -dir[2]];                                          // FRD -> Isaac body axes
      const w = [0, 1, 2].map(i => R[i][0] * body[0] + R[i][1] * body[1] + R[i][2] * body[2]);
      if (w[2] >= -1e-6) return null;
      const t = -cam[2] / w[2];
      return [cam[0] + t * w[0], cam[1] + t * w[1]];
    },
  };
}

// The ground texture is fixed to the world: two square-cell layers whose size never changes. Altitude only
// changes how large the cells appear, never what the texture is.
export const TEXTURE_LAYERS = [{ cell: 1.0, salt: 7919, keep: 0.5, base: 58 }, { cell: 0.25, salt: 0, keep: 0.45, base: 70 }];
export const textureShade = (i, j, salt) => cellHash(i + salt, j - 3 * salt);

// Pinhole projection of a camera-frame point: image right = +y, image up = +x. Null behind the camera.
export function project(frd, focalPx, cx, cy) {
  if (frd[2] <= 1e-3) return null;
  return [cx + focalPx * frd[1] / frd[2], cy - focalPx * frd[0] / frd[2]];
}

export const focalPx = (widthPx, fovDeg) => 0.5 * widthPx / Math.tan(fovDeg * RAD / 2);

export function cameraSettings(metadata) {
  const fusion = metadata?.disturbances?.sensor_noise?.imu?.fusion;
  const marker = fusion?.marker;
  return {
    enabled: !!fusion?.enabled,
    markerEnabled: !!marker?.enabled,
    fovDeg: marker?.fov_deg ?? DEFAULTS.fovDeg,
    markerSizeM: marker?.marker_size_m ?? DEFAULTS.markerSizeM,
    altMaxM: marker?.alt_max_m ?? DEFAULTS.altMaxM,
    altMinM: marker?.alt_min_m ?? DEFAULTS.altMinM,
    mountDownM: fusion?.mount_down_m ?? DEFAULTS.mountDownM,
    maxHeightM: fusion?.flow?.max_height_m ?? DEFAULTS.maxHeightM,
    minPixels: marker?.min_pixels ?? 24,
    resolutionPx: marker?.resolution_px ?? 640,
  };
}

// What the marker detector is doing, from one frame's fusion telemetry.
export function markerState(fusion, settings = DEFAULTS) {
  if (!fusion || !fusion.marker_enabled) return { key: 'off', label: 'PAD MARKER OFF' };
  if (!fusion.marker_in_fov) return { key: 'out_of_view', label: 'MARKER OUT OF VIEW' };
  if (!fusion.marker_in_window) {
    const min = settings.altMinM ?? DEFAULTS.altMinM, max = settings.altMaxM ?? DEFAULTS.altMaxM;
    return { key: 'outside_window', label: fusion.height_est_m < min ? `MARKER OFF BELOW ${min} m` : `MARKER VISIBLE · USED BELOW ${max} m` };
  }
  if (!fusion.marker_valid) return { key: 'not_detected', label: 'MARKER NOT DETECTED' };
  if (!fusion.accepted?.marker) return { key: 'rejected', label: 'DETECTED · REJECTED BY GATE' };
  return { key: 'tracking', label: 'MARKER TRACKING' };
}

// Optical-flow arrow in image pixels (right, up positive; +y_screen down): the sensor's flow is
// (v_y/r - w_x, -v_x/r - w_y) about the FRD axes, i.e. the scene moves (right, up) = (-f_x, f_y).
export function flowArrow(flow, pxPerRadS = 30, maxPx = 60) {
  if (!flow) return [0, 0];
  let dx = -flow[0] * pxPerRadS, dy = -flow[1] * pxPerRadS;                     // screen: down is +y
  const length = Math.hypot(dx, dy);
  if (length > maxPx) { dx *= maxPx / length; dy *= maxPx / length; }
  return [dx, dy];
}

// Height ruler position (0 at the bottom, 1 at the top) for a height in metres.
export const rulerFraction = (heightM, maxM = DEFAULTS.maxHeightM) => Math.max(0, Math.min(1, heightM / maxM));

const TAG_BITS = '101101011001110010101101001011010010';
const cellHash = (i, j) => {
  let x = Math.imul(i, 374761393) ^ Math.imul(j, 668265263);
  x = Math.imul(x ^ (x >>> 13), 1274126177);
  return ((x ^ (x >>> 16)) >>> 0) / 4294967296;
};

export function createDownCam(root) {
  const canvas = root.querySelector('canvas'), status = root.querySelector('.down-cam-status'), mode = root.querySelector('.down-cam-mode');
  const ctx = canvas.getContext('2d');
  root.querySelector('.down-cam-bar').onclick = () => root.classList.toggle('collapsed');
  const W = canvas.width, H = canvas.height, RULER = 52, CW = W - RULER, cx = CW / 2, cy = H / 2;
  let last = null, lastMetadata = null;

  function ground(camera, settings) {
    const focal = focalPx(CW, settings.fovDeg);
    // Ground covered by the image: hit points of rays through the border, bounded for rays that miss the ground.
    const xs = [], ys = [];
    for (const px of [0, CW / 2, CW]) for (const py of [0, H / 2, H]) {
      const hit = camera.groundHit([(cy - py) / focal, (px - cx) / focal, 1]) ?? [camera.origin[0] + 30, camera.origin[1] + 30];
      xs.push(hit[0]); ys.push(hit[1]);
    }
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    for (const { cell, salt, keep, base } of TEXTURE_LAYERS) {
      const i0 = Math.floor(x0 / cell) - 1, i1 = Math.ceil(x1 / cell) + 1, j0 = Math.floor(y0 / cell) - 1, j1 = Math.ceil(y1 / cell) + 1;
      if ((i1 - i0) * (j1 - j0) > 12000) continue;                     // only when the view spans far more than the flight envelope
      for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) {
        const shade = textureShade(i, j, salt);
        if (shade < 1 - keep) continue;
        const q = quad(camera, focal, [[i * cell, j * cell, 0], [(i + 1) * cell, j * cell, 0], [(i + 1) * cell, (j + 1) * cell, 0], [i * cell, (j + 1) * cell, 0]]);
        if (!q) continue;
        const minX = Math.min(q[0][0], q[1][0], q[2][0], q[3][0]), maxX = Math.max(q[0][0], q[1][0], q[2][0], q[3][0]);
        const minY = Math.min(q[0][1], q[1][1], q[2][1], q[3][1]), maxY = Math.max(q[0][1], q[1][1], q[2][1], q[3][1]);
        if (maxX < 0 || minX > CW || maxY < 0 || minY > H || (maxX - minX) * (maxY - minY) < 0.25) continue;
        const level = Math.round(base + shade * 70);
        ctx.fillStyle = `rgb(${level - 20},${level},${level - 30})`;
        ctx.beginPath(); ctx.moveTo(q[0][0], q[0][1]); ctx.lineTo(q[1][0], q[1][1]); ctx.lineTo(q[2][0], q[2][1]); ctx.lineTo(q[3][0], q[3][1]); ctx.closePath(); ctx.fill();
      }
    }
    return focal;
  }

  function quad(camera, focal, corners) {
    const pts = corners.map(c => project(camera.toFrd(c), focal, cx, cy));
    if (pts.some(p => !p)) return null;
    return pts;
  }

  function padAndMarker(camera, focal, marker, settings) {
    const pad = DEFAULTS.padSizeM / 2, [mx, my] = marker;
    const padQuad = quad(camera, focal, [[mx - pad, my - pad, 0], [mx + pad, my - pad, 0], [mx + pad, my + pad, 0], [mx - pad, my + pad, 0]]);
    if (padQuad) { ctx.beginPath(); padQuad.forEach((p, k) => (k ? ctx.lineTo(...p) : ctx.moveTo(...p))); ctx.closePath(); ctx.fillStyle = '#5b6168'; ctx.fill(); }
    if (!settings.markerEnabled) return null;
    const half = settings.markerSizeM / 2, cells = 8, step = settings.markerSizeM / cells;
    for (let a = 0; a < cells; a++) for (let b = 0; b < cells; b++) {
      const inner = a > 0 && a < cells - 1 && b > 0 && b < cells - 1;
      const white = inner && TAG_BITS[(a - 1) * 6 + (b - 1)] === '1';
      const x0 = mx - half + a * step, y0 = my - half + b * step;
      const q = quad(camera, focal, [[x0, y0, 0], [x0 + step, y0, 0], [x0 + step, y0 + step, 0], [x0, y0 + step, 0]]);
      if (!q) continue;
      ctx.beginPath(); q.forEach((p, k) => (k ? ctx.lineTo(...p) : ctx.moveTo(...p))); ctx.closePath();
      ctx.fillStyle = white ? '#f2f2f2' : '#101214'; ctx.fill();
    }
    return quad(camera, focal, [[mx - half, my - half, 0], [mx + half, my - half, 0], [mx + half, my + half, 0], [mx - half, my + half, 0]]);
  }

  function arrow(x, y, dx, dy, color, width) {
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + dx, y + dy); ctx.stroke();
    const len = Math.hypot(dx, dy);
    if (len < 3) return;
    const ux = dx / len, uy = dy / len;
    ctx.beginPath(); ctx.moveTo(x + dx, y + dy); ctx.lineTo(x + dx - 6 * ux - 3 * uy, y + dy - 6 * uy + 3 * ux); ctx.lineTo(x + dx - 6 * ux + 3 * uy, y + dy - 6 * uy - 3 * ux); ctx.closePath(); ctx.fill();
  }

  function ruler(f, fusion, settings) {
    const x0 = CW + 8, top = 22, bottom = H - 24, span = bottom - top, y = m => bottom - rulerFraction(m, settings.maxHeightM) * span;
    ctx.fillStyle = '#0d1218'; ctx.fillRect(CW, 0, RULER, H);
    if (settings.markerEnabled) { ctx.fillStyle = 'rgba(95,212,160,.16)'; ctx.fillRect(x0, y(settings.altMaxM), 14, y(settings.altMinM) - y(settings.altMaxM)); }
    ctx.strokeStyle = '#3a4550'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0 + 7, top); ctx.lineTo(x0 + 7, bottom); ctx.stroke();
    ctx.font = '9px Bahnschrift, Arial'; ctx.fillStyle = '#8b98a5'; ctx.textAlign = 'left';
    for (let m = 0; m <= settings.maxHeightM; m += 4) { ctx.fillRect(x0 + 3, y(m), 8, 1); ctx.fillText(`${m}`, x0 + 16, y(m) + 3); }
    ctx.fillStyle = '#f2f2f2'; ctx.fillRect(x0 - 2, y(fusion.height_truth_m) - 1, 18, 2);                       // truth
    ctx.fillStyle = '#5fd4ff'; ctx.beginPath(); ctx.moveTo(x0 + 16, y(fusion.height_est_m)); ctx.lineTo(x0 + 24, y(fusion.height_est_m) - 4); ctx.lineTo(x0 + 24, y(fusion.height_est_m) + 4); ctx.fill();   // estimate
    const rangeHeight = fusion.range_m?.[0];
    if (rangeHeight != null) { ctx.strokeStyle = fusion.range_valid ? '#f2b84b' : '#6b5a2e'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x0 + 7, y(rangeHeight), 3, 0, 2 * Math.PI); ctx.stroke(); }
    ctx.fillStyle = '#8b98a5'; ctx.font = '8px Bahnschrift, Arial'; ctx.textAlign = 'center';
    ctx.fillText('HEIGHT m', x0 + 12, 12); ctx.fillStyle = '#5fd4ff'; ctx.fillText('EST', x0 + 12, H - 14);
    ctx.fillStyle = '#f2b84b'; ctx.fillText('RANGE', x0 + 12, H - 5);
  }

  function draw(f, settings) {
    const fusion = f.fusion, camera = makeCamera(f.position, f.quaternion, settings.mountDownM);
    ctx.clearRect(0, 0, W, H);
    ctx.save(); ctx.beginPath(); ctx.rect(0, 0, CW, H); ctx.clip();
    ctx.fillStyle = '#2d3a30'; ctx.fillRect(0, 0, CW, H);
    const focal = ground(camera, settings);
    const tag = padAndMarker(camera, focal, fusion.marker_world_m, settings);
    const state = markerState(fusion, settings);
    // Optical flow field: the scene moves the same way everywhere for a flat floor.
    const [mx, my] = flowArrow(fusion.flow_rad_s), [tx, ty] = flowArrow(fusion.flow_truth_rad_s);
    for (const gx of [0.2, 0.5, 0.8]) for (const gy of [0.25, 0.5, 0.75]) {
      arrow(gx * CW, gy * H, mx, my, fusion.flow_valid ? 'rgba(255,214,120,.9)' : 'rgba(150,120,70,.7)', 1.6);
    }
    arrow(cx, cy, tx, ty, 'rgba(255,255,255,.75)', 1);
    if (tag && settings.markerEnabled) {
      const xs = tag.map(p => p[0]), ys = tag.map(p => p[1]), color = { tracking: '#7fe0a0', rejected: '#ff8a80', not_detected: '#ff8a80', outside_window: '#8b98a5' }[state.key] ?? '#8b98a5';
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(state.key === 'tracking' ? [] : [4, 3]);
      ctx.strokeRect(Math.min(...xs) - 4, Math.min(...ys) - 4, Math.max(...xs) - Math.min(...xs) + 8, Math.max(...ys) - Math.min(...ys) + 8); ctx.setLineDash([]);
      if (fusion.marker_valid) {                                        // what the detector reported (angles -> pixels)
        const px = cx + focal * Math.tan(fusion.marker_rad[1]), py = cy - focal * Math.tan(fusion.marker_rad[0]);
        ctx.strokeStyle = '#5fd4ff'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(px - 8, py); ctx.lineTo(px + 8, py); ctx.moveTo(px, py - 8); ctx.lineTo(px, py + 8); ctx.stroke();
      }
    } else if (settings.markerEnabled && state.key === 'out_of_view') {
      const angle = Math.atan2(-Math.tan(fusion.marker_truth_rad[0]), Math.tan(fusion.marker_truth_rad[1]));
      const ex = cx + Math.cos(angle) * (CW / 2 - 16), ey = cy + Math.sin(angle) * (H / 2 - 16);
      arrow(ex - Math.cos(angle) * 18, ey - Math.sin(angle) * 18, Math.cos(angle) * 18, Math.sin(angle) * 18, '#ff8a80', 2.5);
    }
    ctx.strokeStyle = 'rgba(255,255,255,.35)'; ctx.lineWidth = 1;                                              // optical axis
    ctx.beginPath(); ctx.moveTo(cx - 10, cy); ctx.lineTo(cx + 10, cy); ctx.moveTo(cx, cy - 10); ctx.lineTo(cx, cy + 10); ctx.stroke();
    ctx.restore();
    ruler(f, fusion, settings);
    ctx.font = '9px Bahnschrift, Arial'; ctx.fillStyle = 'rgba(255,255,255,.75)'; ctx.textAlign = 'left';
    ctx.fillText('FWD ↑', 6, 12);
    const flowSpeed = fusion.flow_rad_s ? Math.hypot(...fusion.flow_rad_s) : 0;
    mode.textContent = `FOV ${settings.fovDeg}°`;
    status.innerHTML = `<span>HEIGHT <b>${fusion.height_est_m.toFixed(2)}</b> m est · <b>${fusion.range_valid ? fusion.range_m[0].toFixed(2) : '—'}</b> m range</span>`
      + `<span>FLOW <b>${flowSpeed.toFixed(2)}</b> rad/s${fusion.flow_valid ? '' : ' · <i>no lock</i>'}</span>`
      + `<span class="marker-${state.key}">${state.label}</span>`;
  }

  return {
    update(f, metadata) {
      // Older replays recorded fusion readings without the truth fields this view needs.
      root.hidden = !f?.fusion?.marker_world_m;
      if (root.hidden || (f === last && metadata === lastMetadata)) return;
      last = f; lastMetadata = metadata;
      draw(f, cameraSettings(metadata));
    },
  };
}
