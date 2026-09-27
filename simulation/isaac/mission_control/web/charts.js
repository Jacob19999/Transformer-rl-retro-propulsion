// Small-multiple telemetry strip charts on one shared time axis. Each chart is
// one quantity with one y-scale; hovering shows a crosshair with readouts,
// clicking or dragging seeks the replay to that time.
import { fitCanvas, ink, series, status } from './instruments.js';
import { imuError } from './imu.js';

const deg = 180 / Math.PI;
const sans = 'Bahnschrift, "DIN Alternate", "Segoe UI", Arial, sans-serif';
const fmt = (n, d = 1) => Number.isFinite(n) ? n.toFixed(d) : '—';

export const CHARTS = [
  // PLAN / COMMAND exist only in convex-guidance recordings (frame.guidance).
  { title: 'ALTITUDE', unit: 'm', digits: 2, zero: true, series: [
    { name: 'Z', get: f => f.position[2] }, { name: 'PLAN', get: f => f.guidance?.reference_position?.[2] }] },
  { title: 'VELOCITY', unit: 'm/s', digits: 2, zero: true, series: [
    { name: 'VERTICAL', get: f => f.velocity[2] }, { name: 'HORIZONTAL', get: f => Math.hypot(f.velocity[0], f.velocity[1]) }] },
  { title: 'DISTANCE', unit: 'm', digits: 2, zero: true, series: [
    { name: 'PAD', get: f => f.pad_distance }, { name: 'CROSS-TRACK', get: f => f.mission?.cross_track_error_m }] },
  { title: 'THRUST', unit: 'N', digits: 1, zero: true, series: [
    { name: 'THRUST', get: f => f.thrust_n }, { name: 'COMMAND', get: f => f.guidance?.thrust_command_n }], limit: c => c.maxThrust },
  { title: 'BODY RATES · FRD', unit: '°/s', digits: 0, zero: true, series: [
    { name: 'P', get: f => f.gyro[0] * deg }, { name: 'Q', get: f => f.gyro[1] * deg }, { name: 'R', get: f => f.gyro[2] * deg }],
    bands: c => c.softLimits },
  { title: 'FIN ANGLES', unit: '°', digits: 1, zero: true, series: ['FWD', 'RIGHT', 'AFT', 'LEFT'].map((name, i) => ({ name, get: f => f.fin_angles[i] * deg })),
    bands: c => [c.finLimit] },
  { title: 'BUS VOLTAGE', unit: 'V', digits: 2, series: [{ name: 'V', get: f => f.battery?.voltage_v }] },
  { title: 'PACK CURRENT', unit: 'A', digits: 1, zero: true, series: [{ name: 'A', get: f => f.battery?.current_a }], limit: c => c.maxCurrent },
  // IMU estimate minus PhysX truth (frame.imu); empty for replays recorded before it.
  { title: 'IMU POSITION ERROR', unit: 'm', digits: 3, zero: true, series: ['X', 'Y', 'Z'].map((name, i) => ({ name, get: f => imuError(f)?.position[i] })) },
  { title: 'IMU VELOCITY ERROR', unit: 'm/s', digits: 3, zero: true, series: ['X', 'Y', 'Z'].map((name, i) => ({ name, get: f => imuError(f)?.velocity[i] })) },
  { title: 'IMU ATTITUDE ERROR', unit: '°', digits: 2, zero: true, series: [{ name: 'ANGLE', get: f => imuError(f)?.attitude }] },
  { title: 'IMU BODY-RATE ERROR', unit: '°/s', digits: 1, zero: true, series: ['P', 'Q', 'R'].map((name, i) => ({ name, get: f => imuError(f)?.gyro[i] })) },
];

function niceStep(range) {
  const raw = range / 3, power = 10 ** Math.floor(Math.log10(raw)), n = raw / power;
  return (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * power;
}

export function createCharts(root, { onSeek, titles }) {
  let frames = [], context = {}, events = [], version = 0, hover = null;
  const specs = titles ? titles.map(title => CHARTS.find(spec => spec.title === title)) : CHARTS;
  const charts = specs.map(spec => {
    const el = document.createElement('div'); el.className = 'chart';
    const multi = spec.series.length > 1;
    el.innerHTML = `<div class="chart-head"><b>${spec.title}</b><span class="chart-unit">${spec.unit}</span><span class="chart-legend">${spec.series.map((s, i) =>
      `<span>${multi ? `<i style="background:${series[i]}"></i>${s.name} ` : ''}<em>—</em></span>`).join('')}</span></div><canvas aria-label="${spec.title} time history in ${spec.unit}; click to seek"></canvas>`;
    root.append(el);
    const canvas = el.querySelector('canvas'), values = [...el.querySelectorAll('em')];
    const chart = { spec, el, canvas, values, cache: document.createElement('canvas'), cacheKey: '' };
    const timeAt = e => { const r = canvas.getBoundingClientRect(), g = chart.geometry; return g ? Math.max(0, Math.min(g.end, (e.clientX - r.left - g.left) / g.plotW * g.end)) : 0; };
    let dragging = false;
    canvas.onpointerdown = e => { dragging = true; canvas.setPointerCapture(e.pointerId); onSeek(timeAt(e)); };
    canvas.onpointermove = e => { hover = timeAt(e); if (dragging) onSeek(hover); };
    canvas.onpointerup = canvas.onpointercancel = () => { dragging = false; };
    canvas.onpointerleave = () => { hover = null; };
    return chart;
  });

  function renderStatic(chart, w, h) {
    const { spec } = chart, dpr = Math.min(devicePixelRatio, 2);
    chart.cache.width = Math.round(w * dpr); chart.cache.height = Math.round(h * dpr);
    const ctx = chart.cache.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const left = 38, right = 8, top = 6, bottom = 18, plotW = w - left - right, plotH = h - top - bottom;
    const end = Math.max(frames.at(-1)?.t ?? 1, 1e-3);
    let lo = Infinity, hi = -Infinity;
    for (const f of frames) for (const s of spec.series) { const v = s.get(f); if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }
    const bands = spec.bands?.(context)?.filter(Number.isFinite) ?? [], limit = spec.limit?.(context);
    if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
    if (spec.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
    if (Number.isFinite(limit) && limit < hi * 1.6) hi = Math.max(hi, limit);
    const bandMax = Math.max(0, ...bands);
    if (bandMax && bandMax < Math.max(Math.abs(lo), Math.abs(hi)) * 1.6) { hi = Math.max(hi, bandMax); if (lo < 0) lo = Math.min(lo, -bandMax); }
    if (hi - lo < 1e-6) { hi += .5; lo -= .5; }
    const step = niceStep(hi - lo); lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
    const y = v => top + (hi - v) / (hi - lo) * plotH, x = t => left + t / end * plotW;
    chart.geometry = { left, plotW, end, top, plotH, y, x };
    ctx.font = `9px ${sans}`; ctx.fillStyle = ink.muted; ctx.textAlign = 'right'; ctx.lineWidth = 1;
    for (let v = lo; v <= hi + step / 2; v += step) {
      ctx.strokeStyle = Math.abs(v) < step / 1e3 && lo < 0 ? 'rgba(255,255,255,.22)' : ink.grid;
      ctx.beginPath(); ctx.moveTo(left, Math.round(y(v)) + .5); ctx.lineTo(w - right, Math.round(y(v)) + .5); ctx.stroke();
      ctx.fillText(fmt(v, step < 1 ? Math.max(1, Math.ceil(-Math.log10(step) - 1e-9)) : 0), left - 5, y(v) + 3);
    }
    ctx.textAlign = 'center';
    const tStep = niceStep(end * 1.4);
    for (let t = 0; t <= end + 1e-6; t += tStep) ctx.fillText(`${fmt(t, tStep < 1 ? 1 : 0)}s`, x(t), h - 5);
    // Soft limits are thresholds, not data: dashed, in warning ink, labelled on the axis side.
    ctx.setLineDash([3, 4]); ctx.strokeStyle = 'rgba(250,178,25,.55)';
    for (const b of bands) for (const v of [b, -b]) if (v >= lo && v <= hi) { ctx.beginPath(); ctx.moveTo(left, y(v)); ctx.lineTo(w - right, y(v)); ctx.stroke(); }
    if (Number.isFinite(limit) && limit <= hi) { ctx.beginPath(); ctx.moveTo(left, y(limit)); ctx.lineTo(w - right, y(limit)); ctx.stroke(); }
    ctx.setLineDash([]);
    for (const e of events) { ctx.strokeStyle = e.tone ? status[e.tone] + '88' : 'rgba(255,255,255,.16)'; ctx.beginPath(); ctx.moveTo(x(e.t), top); ctx.lineTo(x(e.t), top + plotH); ctx.stroke(); }
    ctx.lineWidth = 1.6; ctx.lineJoin = 'round';
    spec.series.forEach((s, i) => {
      ctx.strokeStyle = series[i]; ctx.beginPath(); let pen = false;
      for (const f of frames) { const v = s.get(f); if (!Number.isFinite(v)) { pen = false; continue; } pen ? ctx.lineTo(x(f.t), y(v)) : ctx.moveTo(x(f.t), y(v)); pen = true; }
      ctx.stroke();
    });
  }

  function draw(time) {
    if (!root.offsetParent) return; // page hidden: nothing to lay out
    for (const chart of charts) {
      const w = chart.canvas.clientWidth, h = chart.canvas.clientHeight;
      if (!w || !h) continue;
      const key = `${version}:${w}:${h}`;
      if (chart.cacheKey !== key) { renderStatic(chart, w, h); chart.cacheKey = key; }
      const ctx = fitCanvas(chart.canvas, w, h), g = chart.geometry;
      ctx.drawImage(chart.cache, 0, 0, w, h);
      const cursor = hover ?? time;
      if (frames.length) {
        const f = frameAt(cursor);
        ctx.strokeStyle = hover == null ? 'rgba(244,246,247,.8)' : ink.secondary; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(g.x(cursor), g.top); ctx.lineTo(g.x(cursor), g.top + g.plotH); ctx.stroke();
        chart.spec.series.forEach((s, i) => {
          const v = s.get(f), label = fmt(v, chart.spec.digits);
          if (chart.values[i].textContent !== label) chart.values[i].textContent = label;
          if (Number.isFinite(v)) { ctx.fillStyle = '#000'; ctx.strokeStyle = series[i]; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(g.x(f.t), g.y(v), 3.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); }
        });
        chart.el.classList.toggle('hovering', hover != null);
      } else chart.values.forEach(v => { v.textContent = '—'; });
    }
  }

  function frameAt(t) {
    let lo = 0, hi = frames.length - 1;
    while (lo < hi) { const m = Math.ceil((lo + hi) / 2); if (frames[m].t <= t) lo = m; else hi = m - 1; }
    return frames[lo];
  }

  return {
    setData(nextFrames, nextContext, nextEvents) { frames = nextFrames; context = nextContext; events = nextEvents; version++; },
    draw, hoverTime: () => hover, canvases: () => charts.map(c => c.canvas),
  };
}
