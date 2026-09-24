// Webcast-style flight instruments drawn on 2D canvases. Every value shown is
// read from recorded Isaac frames; nothing here estimates or smooths state.
export const ink = { primary: '#f4f6f7', secondary: '#b9c2c8', muted: '#7c878f', track: 'rgba(255,255,255,.13)', grid: 'rgba(255,255,255,.07)' };
export const series = ['#3987e5', '#d95926', '#199e70', '#c98500'];
export const status = { good: '#0ca30c', warning: '#fab219', serious: '#ec835a', critical: '#d03b3b' };
const sans = 'Bahnschrift, "DIN Alternate", "Segoe UI", Arial, sans-serif';
const deg = 180 / Math.PI;

export function fitCanvas(canvas, width = canvas.clientWidth, height = canvas.clientHeight) {
  const dpr = Math.min(devicePixelRatio, 2);
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
  }
  const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
  return ctx;
}

/** ZYX Euler angles and tilt from a w,x,y,z world quaternion (Z up). */
export function attitude(q) {
  const [w, x, y, z] = q;
  return {
    roll: Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)) * deg,
    pitch: Math.asin(Math.max(-1, Math.min(1, 2 * (w * y - z * x)))) * deg,
    yaw: Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) * deg,
    tilt: Math.acos(Math.max(-1, Math.min(1, 1 - 2 * (x * x + y * y)))) * deg,
  };
}

// 270° arc gauge: value in the centre, label beneath, the arc fills with value.
function gauge(ctx, cx, cy, r, { label, value, unit, max, digits = 0, alert = false }) {
  const start = Math.PI * .75, sweep = Math.PI * 1.5;
  const fraction = Number.isFinite(value) ? Math.max(0, Math.min(1, Math.abs(value) / max)) : 0;
  ctx.lineCap = 'round'; ctx.lineWidth = 3;
  ctx.strokeStyle = ink.track; ctx.beginPath(); ctx.arc(cx, cy, r, start, start + sweep); ctx.stroke();
  if (fraction > 0) { ctx.strokeStyle = alert ? status.warning : ink.primary; ctx.beginPath(); ctx.arc(cx, cy, r, start, start + sweep * fraction); ctx.stroke(); }
  ctx.textAlign = 'center'; ctx.fillStyle = ink.primary;
  ctx.font = `${Math.round(r * .52)}px ${sans}`;
  ctx.fillText(Number.isFinite(value) ? value.toFixed(digits) : '—', cx, cy + r * .12);
  ctx.fillStyle = ink.muted; ctx.font = `${Math.max(8, Math.round(r * .2))}px ${sans}`;
  ctx.fillText(unit, cx, cy + r * .45);
  ctx.fillStyle = ink.secondary; ctx.font = `${Math.max(9, Math.round(r * .22))}px ${sans}`;
  ctx.fillText(label, cx, cy + r + 14);
}

/**
 * Bottom band under the camera array, like a launch webcast: speed and
 * altitude on the left, the mission timeline centred, thrust and pack on the right.
 */
export function drawWebcast(canvas, { frame, time, end, milestones, maxima, width, height }) {
  width ??= canvas.clientWidth; height ??= canvas.clientHeight;
  const ctx = fitCanvas(canvas, width, height);
  const gradient = ctx.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, '#05070a'); gradient.addColorStop(1, '#000');
  ctx.fillStyle = gradient; ctx.fillRect(0, 0, width, height);
  const r = Math.min(34, height * .3), cy = height * .42, gap = r * 2.6;
  const f = frame;
  const speed = f ? Math.hypot(...f.velocity) : NaN;
  const left = [
    { label: 'SPEED', value: speed, unit: 'm/s', max: maxima.speed, digits: 1 },
    { label: 'ALTITUDE', value: f?.position[2], unit: 'm', max: maxima.altitude, digits: 1 },
  ];
  const right = [
    { label: 'THRUST', value: f?.thrust_n, unit: 'N', max: maxima.thrust, digits: 1 },
    { label: 'LIPO', value: f?.battery ? f.battery.soc * 100 : NaN, unit: '% SOC', max: 100, digits: 0, alert: !!f?.battery?.current_limited },
  ];
  const compact = width < 640;
  const shown = compact ? [left[1]] : left, shownRight = compact ? [right[0]] : right;
  shown.forEach((g, i) => gauge(ctx, 16 + r + i * gap, cy, r, g));
  shownRight.forEach((g, i) => gauge(ctx, width - 16 - r - (shownRight.length - 1 - i) * gap, cy, r, g));
  const x0 = 16 + shown.length * gap + 18, x1 = width - 16 - shownRight.length * gap - 18;
  if (x1 - x0 > 80) drawTimeline(ctx, x0, x1, height * .46, time, end, milestones);
}

// Milestones alternate above and below the line so neighbouring labels never collide.
function drawTimeline(ctx, x0, x1, y, time, end, milestones) {
  const span = Math.max(end, 1e-3), toX = t => x0 + Math.max(0, Math.min(1, t / span)) * (x1 - x0), now = toX(time);
  ctx.lineCap = 'butt'; ctx.lineWidth = 2;
  ctx.strokeStyle = ink.track; ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
  ctx.strokeStyle = ink.primary; ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(now, y); ctx.stroke();
  let lastX = [-1e9, -1e9];
  milestones.forEach((m, i) => {
    const x = toX(m.t), passed = m.t <= time + 1e-6, lane = i % 2;
    ctx.fillStyle = '#000'; ctx.strokeStyle = m.tone ? status[m.tone] : passed ? ink.primary : ink.muted; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    if (passed) { ctx.fillStyle = ctx.strokeStyle; ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fill(); }
    if (x - lastX[lane] < 70) return;
    lastX[lane] = x;
    ctx.textAlign = 'center'; ctx.font = `10px ${sans}`; ctx.fillStyle = passed ? ink.primary : ink.muted;
    const ty = lane ? y + 22 : y - 14;
    ctx.fillText(m.label, x, ty);
    ctx.font = `9px ${sans}`; ctx.fillStyle = ink.muted; ctx.fillText(`T+${m.t.toFixed(1)}`, x, lane ? ty + 12 : ty - 12);
  });
  // Current time: a short bar through the line, clear of milestone labels.
  ctx.fillStyle = ink.primary; ctx.fillRect(now - 1, y - 7, 2, 14);
}

/** Attitude director: horizon disc, pitch ladder, roll pointer and heading. */
export function drawAdi(canvas, a) {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (Math.min(w, h) < 40) return; // hidden page or collapsed layout
  const ctx = fitCanvas(canvas, w, h), cx = w / 2, cy = h / 2, r = Math.min(w, h) / 2 - 12, ppd = r / 35;
  ctx.save(); ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.clip();
  if (a) {
    ctx.translate(cx, cy); ctx.rotate(-a.roll / deg); ctx.translate(0, a.pitch * ppd);
    ctx.fillStyle = '#0f2233'; ctx.fillRect(-2 * r, -4 * r, 4 * r, 4 * r);
    ctx.fillStyle = '#21180f'; ctx.fillRect(-2 * r, 0, 4 * r, 4 * r);
    ctx.strokeStyle = ink.primary; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(-2 * r, 0); ctx.lineTo(2 * r, 0); ctx.stroke();
    ctx.lineWidth = 1; ctx.font = `9px ${sans}`; ctx.fillStyle = ink.secondary; ctx.textAlign = 'left';
    for (let p = -80; p <= 80; p += 10) {
      if (!p) continue;
      const y = -p * ppd, half = p % 20 ? r * .13 : r * .24;
      ctx.strokeStyle = 'rgba(244,246,247,.55)'; ctx.beginPath(); ctx.moveTo(-half, y); ctx.lineTo(half, y); ctx.stroke();
      if (!(p % 20)) ctx.fillText(String(Math.abs(p)), half + 4, y + 3);
    }
  } else { ctx.fillStyle = '#0b0f13'; ctx.fillRect(0, 0, w, h); }
  ctx.restore();
  ctx.strokeStyle = ink.track; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
  // Fixed roll scale, pointer rotates with the vehicle.
  ctx.strokeStyle = ink.muted;
  for (const t of [-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60]) {
    const angle = (t - 90) / deg, inner = r + (t % 30 ? 3 : 0);
    ctx.beginPath(); ctx.moveTo(cx + Math.cos(angle) * inner, cy + Math.sin(angle) * inner); ctx.lineTo(cx + Math.cos(angle) * (r + 8), cy + Math.sin(angle) * (r + 8)); ctx.stroke();
  }
  if (a) {
    const angle = (-a.roll - 90) / deg;
    ctx.fillStyle = ink.primary; ctx.beginPath();
    ctx.moveTo(cx + Math.cos(angle) * (r - 1), cy + Math.sin(angle) * (r - 1));
    ctx.lineTo(cx + Math.cos(angle + .07) * (r - 11), cy + Math.sin(angle + .07) * (r - 11));
    ctx.lineTo(cx + Math.cos(angle - .07) * (r - 11), cy + Math.sin(angle - .07) * (r - 11)); ctx.fill();
  }
  // Vehicle reference symbol.
  ctx.strokeStyle = ink.primary; ctx.lineWidth = 2.5; ctx.beginPath();
  ctx.moveTo(cx - r * .45, cy); ctx.lineTo(cx - r * .15, cy); ctx.lineTo(cx - r * .07, cy + 7); ctx.moveTo(cx + r * .45, cy); ctx.lineTo(cx + r * .15, cy); ctx.lineTo(cx + r * .07, cy + 7); ctx.stroke();
  ctx.fillStyle = ink.primary; ctx.beginPath(); ctx.arc(cx, cy, 2.5, 0, Math.PI * 2); ctx.fill();
}
