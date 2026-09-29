import { fitCanvas } from './instruments.js';
import { createGuidanceDiagnostics } from './guidance-diagnostics.js';

const colors = { plan: '#48dba2', actual: '#f4f6f7', reference: '#6ab7ff', warning: '#fab219', grid: '#20292f', muted: '#8d9da7' };
const fmt = (value, digits = 1) => Number.isFinite(value) ? value.toFixed(digits) : '—';

// Plans are emitted only on re-solves. Index them once, then use the plan
// available at replay time, including when scrubbing backwards.
export function collectPlans(frames) {
  return frames.filter(f => f.guidance?.plan?.positions?.length).map(f => ({
    t: f.t, id: f.guidance.plan_id, ...f.guidance.plan,
  }));
}

export function planAt(plans, time) {
  let lo = 0, hi = plans.length;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (plans[mid].t <= time) lo = mid + 1; else hi = mid; }
  return plans[lo - 1] ?? null;
}

export function guidanceStatus(g) {
  if (!g) return { label: 'NO GUIDANCE', tone: 'idle' };
  if (g.phase === 'SPOOL_UP') return { label: 'SPOOLING · AWAITING PLAN', tone: 'idle' };
  if (g.phase === 'HOLD') return { label: 'HOLD · PLAN UNAVAILABLE', tone: 'warning' };
  if (g.phase === 'TERMINAL_DESCENT' || g.phase === 'LANDED') return { label: 'LAST PLAN · ' + g.phase.replaceAll('_', ' '), tone: 'idle' };
  if (!g.solver) return { label: 'AWAITING SOLVER', tone: 'idle' };
  if (g.solver.mode === 'soft_terminal') return { label: 'FALLBACK · MAXIMUM BRAKING', tone: 'warning' };
  // Preserve the actual solver status instead of calling every result optimal.
  return { label: (g.solver.status ?? 'STATUS NOT RECORDED').toUpperCase(), tone: g.solver.status === 'Solved' ? 'good' : 'warning' };
}

// Plan cost split by objective term (primary energy/delta-v plus any priced
// secondary terms), in the primary objective's unit, largest first.
export function costBreakdown(solver) {
  const terms = solver?.cost_terms;
  if (!terms || typeof terms !== 'object') return '';
  const unit = solver.objective === 'delta_v' ? 'm/s' : 'Wh';
  return Object.entries(terms).filter(([, v]) => Number.isFinite(v) && Math.abs(v) >= 5e-4)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .map(([k, v]) => `${k.replace('_', '-')} ${fmt(v, 2)} ${unit}`).join(', ');
}

export function createGuidancePanel(root, { onSeek }) {
  root.innerHTML = `
    <div class="optimization-heading"><div><div class="eyebrow">RECEDING-HORIZON GUIDANCE</div><h2>Convex optimization</h2></div><span class="optimization-status" data-field="status"></span></div>
    <div class="optimization-toolbar"><span data-field="identity">Awaiting first plan</span><div><button type="button" data-action="previous" aria-label="Previous optimization plan">← PREV PLAN</button><button type="button" data-action="next" aria-label="Next optimization plan">NEXT PLAN →</button></div></div>
    <div class="optimization-metrics">${[
      ['cost', 'PLANNED ENERGY'], ['remaining', 'TIME TO GATE'], ['error', 'TRACKING ERROR'], ['solve', 'LAST SOLVE'], ['gap', 'RELAXATION GAP'], ['headroom', 'THRUST HEADROOM'],
    ].map(([key, label]) => `<div><span data-label="${key}">${label}</span><strong data-field="${key}">—</strong></div>`).join('')}</div>
    <div class="guidance-diagnostics" aria-label="Solver, tracking, energy and thrust diagnostics"></div>
    <div class="optimization-legend"><span><i class="plan"></i>Optimized plan</span><span><i class="actual"></i>Flown to replay time</span><span><i class="reference"></i>Current reference</span><span><i class="available"></i>Available thrust now</span></div>
    <div class="optimization-plots">
      <figure><figcaption>GROUND TRACK <span>X / Y · m · equal scale</span></figcaption><canvas data-plot="track" role="img" aria-label="Top view of optimized trajectory, flown path, current reference and landing pad"></canvas></figure>
      <figure><figcaption>ALTITUDE HORIZON <span>Z · m</span></figcaption><canvas data-plot="altitude" role="img" aria-label="Planned and recorded altitude against mission time"></canvas></figure>
      <figure><figcaption>THRUST HORIZON <span>N</span></figcaption><canvas data-plot="thrust" role="img" aria-label="Planned thrust slack and actual thrust against mission time, with current available thrust"></canvas></figure>
    </div>
    <div class="optimization-footnote" data-field="note"></div>`;
  const field = key => root.querySelector(`[data-field="${key}"]`);
  const put = (key, value) => { const el = field(key); if (el.textContent !== value) el.textContent = value; };
  const previous = root.querySelector('[data-action="previous"]'), next = root.querySelector('[data-action="next"]');
  const canvases = Object.fromEntries([...root.querySelectorAll('canvas')].map(c => [c.dataset.plot, c]));
  const diagnostics = createGuidanceDiagnostics(root.querySelector('.guidance-diagnostics'), onSeek);
  let frames = [], plans = [], time = 0, pad = [0,0,0];
  previous.onclick = () => { const p = plans.filter(p => p.t < time - .001).at(-1); if (p) onSeek(p.t); };
  next.onclick = () => { const p = plans.find(p => p.t > time + .001); if (p) onSeek(p.t); };

  function update(frame, replayTime) {
    time = replayTime;
    const g = frame?.guidance;
    root.hidden = !g;
    if (!g) return;
    const plan = planAt(plans, time), s = g.solver, status = guidanceStatus(g);
    put('status', status.label); field('status').dataset.tone = status.tone;
    root.querySelector('.optimization-legend .plan').style.borderColor = status.tone === 'warning' ? colors.warning : colors.plan;
    const phase = g.phase?.replaceAll('_', ' ') ?? 'UNKNOWN PHASE';
    put('identity', plan ? `PLAN #${plan.id} · ${phase} · age ${fmt(time - plan.t)} s` : phase);
    previous.disabled = !plans.some(p => p.t < time - .001);
    next.disabled = !plans.some(p => p.t > time + .001);
    root.querySelector('[data-label="cost"]').textContent = s?.objective === 'delta_v' ? 'PLANNED ΔV' : 'PLANNED ENERGY';
    put('cost', s?.objective === 'delta_v' ? `${fmt(s.delta_v_m_s, 2)} m/s` : `${fmt(s?.energy_wh, 2)} Wh`);
    put('remaining', `${fmt(g.time_to_go_s)} s`);
    put('error', `${fmt(g.tracking_error_m, 2)} m`);
    put('solve', `${fmt(s?.solve_ms, 0)} ms`);
    put('gap', `${fmt(Number.isFinite(s?.convexification_gap) ? s.convexification_gap * 100 : null, 3)} %`);
    const headroom = Number.isFinite(g.thrust_available_n) && Number.isFinite(g.thrust_command_n) ? g.thrust_available_n - g.thrust_command_n : null;
    put('headroom', `${fmt(headroom)} N`); field('headroom').classList.toggle('negative', headroom != null && headroom < 0);
    const tracking = [s && Number.isFinite(s.route_deviation_m) ? `planned route deviation ${fmt(s.route_deviation_m, 2)} m` : null,
      Number.isFinite(g.cross_track_m) ? `flown cross-track ${fmt(g.cross_track_m, 2)} m` : null].filter(Boolean).join(' · ');
    const terms = costBreakdown(s);
    const details = s ? `${s.solves ?? '—'} candidate solves · ${s.iterations ?? '—'} iterations · terminal miss ${fmt(s.terminal_miss_m, 2)} m · corridor excess ${fmt(s.corridor_excess_m, 3)} m${tracking ? ' · ' + tracking : ''}. ${terms ? 'Cost terms: ' + terms + '. ' : ''}` : '';
    put('note', (status.tone === 'warning' && s?.mode === 'soft_terminal' ? 'No safe terminal plan; showing the maximum-braking fallback. ' : '') + details +
      (plan ? 'Thrust plan shows the optimizer’s slack σ. Amber line is current available thrust, not a recorded optimization bound. Headroom = available − commanded thrust.' : 'Waiting for the first recorded trajectory. Live reference and vehicle position are shown when available.'));
    if (!root.getClientRects().length) return;
    diagnostics.update(frames, frame, time);
    const flown = frames.filter(f => f.t <= time);
    drawTrack(canvases.track, plan, flown, frame, status, pad);
    drawHorizon(canvases.altitude, 'altitude', plan, flown, frame, time, status);
    drawHorizon(canvases.thrust, 'thrust', plan, flown, frame, time, status);
  }
  return { setData(data, indexedPlans, landingPad=[0,0,0]) { frames = data; plans = indexedPlans; pad=landingPad; }, update };
}

function plot(canvas, xRange, yRange, equalScale = false) {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return null;
  const ctx = fitCanvas(canvas, w, h), box = { left: 42, right: w - 16, top: 16, bottom: h - 32 };
  let [xmin, xmax] = xRange, [ymin, ymax] = yRange;
  if (equalScale) {
    const scale = Math.max((xmax - xmin) / (box.right - box.left), (ymax - ymin) / (box.bottom - box.top));
    const cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2;
    const dx = scale * (box.right - box.left) / 2, dy = scale * (box.bottom - box.top) / 2;
    xmin = cx - dx; xmax = cx + dx; ymin = cy - dy; ymax = cy + dy;
  }
  const x = v => box.left + (v - xmin) / (xmax - xmin) * (box.right - box.left);
  const y = v => box.bottom - (v - ymin) / (ymax - ymin) * (box.bottom - box.top);
  ctx.font = '10px Consolas, monospace'; ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const xx = box.left + (box.right - box.left) * i / 3, yy = box.bottom - (box.bottom - box.top) * i / 3;
    ctx.strokeStyle = colors.grid; ctx.beginPath(); ctx.moveTo(xx, box.top); ctx.lineTo(xx, box.bottom); ctx.moveTo(box.left, yy); ctx.lineTo(box.right, yy); ctx.stroke();
    ctx.fillStyle = colors.muted; ctx.textAlign = 'center'; ctx.fillText(fmt(xmin + (xmax - xmin) * i / 3), xx, h - 15);
    ctx.textAlign = 'right'; ctx.fillText(fmt(ymin + (ymax - ymin) * i / 3), box.left - 6, yy + 3);
  }
  ctx.save(); ctx.beginPath(); ctx.rect(box.left, box.top, box.right - box.left, box.bottom - box.top); ctx.clip();
  return { ctx, x, y, box, w, h };
}

function line(p, points, color, dash = [], width = 1.8) {
  p.ctx.strokeStyle = color; p.ctx.lineWidth = width; p.ctx.setLineDash(dash); p.ctx.beginPath();
  let started = false;
  for (const [a, b] of points) {
    if (!Number.isFinite(a) || !Number.isFinite(b)) { started = false; continue; }
    if (started) p.ctx.lineTo(p.x(a), p.y(b)); else p.ctx.moveTo(p.x(a), p.y(b));
    started = true;
  }
  p.ctx.stroke(); p.ctx.setLineDash([]);
}

function marker(p, point, color, hollow = false) {
  if (!point?.every(Number.isFinite)) return;
  const ctx = p.ctx;
  ctx.beginPath(); ctx.arc(p.x(point[0]), p.y(point[1]), hollow ? 5 : 3.5, 0, Math.PI * 2);
  ctx.fillStyle = hollow ? '#07090b' : color; ctx.fill(); ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.stroke();
}

function range(values, minimumSpan = 2) {
  let min = Infinity, max = -Infinity;
  for (const v of values) if (Number.isFinite(v)) { min = Math.min(min, v); max = Math.max(max, v); }
  if (!Number.isFinite(min)) return [0, minimumSpan];
  const pad = Math.max(minimumSpan, max - min) * .15;
  return [min - pad, max + pad];
}

function drawTrack(canvas, plan, flown, frame, status, pad) {
  const points = [...(plan?.positions ?? []), ...flown.map(f => f.position), [pad[0]-1.25,pad[1]-1.25,0], [pad[0]+1.25,pad[1]+1.25,0], frame.guidance.reference_position].filter(Boolean);
  const p = plot(canvas, range(points.map(v => v[0])), range(points.map(v => v[1])), true);
  if (!p) return;
  const color = status.tone === 'warning' ? colors.warning : colors.plan;
  // Use the same actual pad radius as the 3D view, rather than inventing a
  // glide cone or presenting an unrecorded constraint as measured telemetry.
  p.ctx.strokeStyle = colors.muted; p.ctx.beginPath(); p.ctx.arc(p.x(pad[0]), p.y(pad[1]), Math.abs(p.x(1.25) - p.x(0)), 0, Math.PI * 2); p.ctx.stroke();
  line(p, flown.map(f => [f.position[0], f.position[1]]), colors.actual);
  line(p, (plan?.positions ?? []).map(v => [v[0], v[1]]), color, [5, 4], 2);
  const actual = frame.position.slice(0, 2), ref = frame.guidance.reference_position?.slice(0, 2);
  if (ref) { line(p, [actual, ref], colors.reference, [2, 3]); marker(p, ref, colors.reference, true); }
  marker(p, actual, colors.actual); marker(p, plan?.positions.at(-1)?.slice(0, 2), color, true);
  p.ctx.restore();
  p.ctx.fillStyle = colors.muted; p.ctx.textAlign = 'left'; p.ctx.fillText('○ pad / plan endpoint   ● vehicle', 12, p.h - 2);
}

function drawHorizon(canvas, kind, plan, flown, frame, time, status) {
  const isThrust = kind === 'thrust', times = plan?.times ?? [];
  const end = plan ? plan.t + (times.at(-1) ?? 0) : time + 1;
  const start = plan?.t ?? Math.max(0, time - 5), last = Math.max(start + 1, end, time);
  const history = flown.filter(f => f.t >= start);
  const planned = times.map((t, i) => [plan.t + t, isThrust ? plan.thrust_n?.[i] : plan.positions[i]?.[2]]);
  const actual = history.map(f => [f.t, isThrust ? f.thrust_n : f.position[2]]);
  const available = frame.guidance.thrust_available_n;
  const values = [...planned, ...actual].map(v => v[1]);
  if (isThrust && Number.isFinite(available)) values.push(available);
  const bounds = range([0, ...values]);
  const p = plot(canvas, [start, last], [values.some(v => v < 0) ? bounds[0] : 0, bounds[1]]);
  if (!p) return;
  if (isThrust && Number.isFinite(available)) line(p, [[start, available], [last, available]], colors.warning, [3, 4], 1);
  line(p, planned, status.tone === 'warning' ? colors.warning : colors.plan, [5, 4], 2);
  line(p, actual, colors.actual);
  line(p, [[time, bounds[0]], [time, bounds[1]]], '#647987', [2, 3], 1);
  marker(p, [frame.t, isThrust ? frame.thrust_n : frame.position[2]], colors.actual);
  if (!isThrust) marker(p, [frame.t, frame.guidance.reference_position?.[2]], colors.reference, true);
  p.ctx.restore(); p.ctx.fillStyle = colors.muted; p.ctx.textAlign = 'left'; p.ctx.fillText('MISSION TIME / s', 12, p.h - 2);
}
