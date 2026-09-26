import { fitCanvas } from './instruments.js';

const ink = { green: '#48dba2', blue: '#6ab7ff', amber: '#fab219', white: '#f4f6f7', muted: '#8d9da7', grid: '#263037' };
const finite = Number.isFinite;
const fmt = (n, digits = 2) => finite(n) ? n.toFixed(digits) : '—';
const headroom = g => finite(g?.thrust_available_n) && finite(g?.thrust_command_n) ? g.thrust_available_n - g.thrust_command_n : null;

// Use a prefix of the recording for every statistic. A backwards seek must
// never retain the peak, consumed energy, or solver result from a later frame.
export function diagnosticSnapshot(frames, time) {
  const history = [], solves = [], seen = new Set();
  let squared = 0, count = 0, peak = null;
  for (const f of frames) {
    if (f.t > time) break;
    const g = f.guidance, error = g?.tracking_error_m;
    history.push({ t: f.t, error, energy: f.battery?.energy_wh, reserve: headroom(g) });
    if (finite(error)) { squared += error * error; count++; peak = Math.max(peak ?? 0, error); }
    // Repeated solver telemetry describes the same solve. Only the frame
    // carrying its newly recorded plan is a solve event, not every frame.
    if (g?.plan && g.solver && !seen.has(g.plan_id)) {
      seen.add(g.plan_id);
      solves.push({ t: f.t, id: g.plan_id, ms: g.solver.solve_ms,
        energy: g.solver.objective === 'energy' ? g.solver.energy_wh : null,
        status: g.solver.status, fallback: g.solver.mode === 'soft_terminal' });
    }
  }
  return { history, solves, rms: count ? Math.sqrt(squared / count) : null, peak };
}

export function createGuidanceDiagnostics(root, onSeek) {
  const cards = [
    ['solver', '01 / SOLVER', 'Solve history', 'Bar height = solve time per recorded plan. Failed attempts are not recorded.', '<span class="diag-green">● Solved</span><span class="diag-amber">! Fallback / other status</span>'],
    ['tracking', '02 / TRACKING', 'Position error', 'Recorded tracking error · statistics through replay time', '<span class="diag-blue">━ Error magnitude</span><span>┄ Sample RMS</span>'],
    ['energy', '03 / ENERGY', 'Energy history', 'Plan cost covers its horizon at solve time; it is not cumulative.', '<span>━ Consumed</span><span class="diag-green">◇ Plan cost at solve</span>'],
    ['reserve', '04 / THRUST', 'Control reserve', 'Gauge = commanded / available. History = available − commanded; zero means no reserve.', '<span class="diag-green">+ Reserve</span><span class="diag-amber">− Demand exceeds available</span>'],
  ];
  root.innerHTML = cards.map(([id, index, title, note, legend]) => `<section class="diagnostic-card" data-card="${id}" aria-label="${title}">
    <div class="diagnostic-heading"><div><span class="eyebrow">${index}</span><h3>${title}</h3></div><strong data-summary="${id}">—</strong></div>
    <div class="diagnostic-legend">${legend}</div>
    ${id === 'reserve' ? '<div class="reserve-gauge" role="img" aria-label="Thrust demand unavailable"><i></i><b></b></div>' : ''}
    <canvas data-diagnostic="${id}" role="img" aria-label="${title} through replay time"></canvas>
    <div class="diagnostic-detail" data-detail="${id}"></div><p class="diagnostic-note">${note}</p>
    ${id === 'solver' ? '<div class="solve-jumps" aria-label="Recent recorded plans"></div>' : ''}
  </section>`).join('');
  const canvas = Object.fromEntries([...root.querySelectorAll('canvas')].map(c => [c.dataset.diagnostic, c]));
  const summaries = Object.fromEntries([...root.querySelectorAll('[data-summary]')].map(c => [c.dataset.summary, c]));
  const details = Object.fromEntries([...root.querySelectorAll('[data-detail]')].map(c => [c.dataset.detail, c]));
  const gauge = root.querySelector('.reserve-gauge'), jumps = root.querySelector('.solve-jumps');
  let jumpKey = '';
  const text = (el, value) => { if (el.textContent !== value) el.textContent = value; };
  return {
    update(frames, frame, time) {
      const { history, solves, rms, peak } = diagnosticSnapshot(frames, time);
      const g = frame.guidance, reserve = headroom(g);
      const demand = finite(g.thrust_command_n) && g.thrust_available_n > 0 ? g.thrust_command_n / g.thrust_available_n : null;
      text(summaries.solver, `${fmt(g.solver?.solve_ms, 0)} ms`);
      text(details.solver, `${solves.length} recorded ${solves.length === 1 ? 'plan' : 'plans'} · ${solves.filter(s => s.fallback).length} fallback · ${g.solver?.iterations ?? '—'} iterations in last plan`);
      text(summaries.tracking, `${fmt(g.tracking_error_m)} m`);
      text(details.tracking, `Sample RMS ${fmt(rms, 3)} m · peak ${fmt(peak, 3)} m`);
      text(summaries.energy, `${fmt(frame.battery?.energy_wh)} Wh used`);
      text(details.energy, g.solver?.objective === 'delta_v' ? 'Δv objective · no energy cost recorded for this plan' : `Last plan ${fmt(g.solver?.energy_wh)} Wh · consumed ${fmt(frame.battery?.energy_wh)} Wh`);
      text(summaries.reserve, `${fmt(reserve, 1)} N ${reserve == null ? '' : reserve < 0 ? 'deficit' : 'reserve'}`);
      summaries.reserve.classList.toggle('diag-amber', reserve != null && reserve < 0);
      text(details.reserve, `Command ${fmt(g.thrust_command_n, 1)} / available ${fmt(g.thrust_available_n, 1)} N · demand ${fmt(demand == null ? null : demand * 100, 0)}%`);
      gauge.querySelector('i').style.width = `${demand == null ? 0 : Math.max(0, Math.min(1, demand)) * 100}%`;
      gauge.dataset.over = String(reserve != null && reserve < 0);
      gauge.dataset.known = String(demand != null);
      gauge.setAttribute('aria-label', demand == null ? 'Thrust demand unavailable' : `Command uses ${fmt(demand * 100, 0)} percent of available thrust; ${fmt(reserve, 1)} newtons headroom`);
      const key = solves.slice(-8).map(s => `${s.id}:${s.t}:${s.ms}:${s.status}:${s.fallback}`).join('|');
      if (key !== jumpKey) {
        jumpKey = key;
        jumps.replaceChildren(...solves.slice(-8).map(s => {
          const button = document.createElement('button'); button.type = 'button';
          const warning = s.fallback || s.status !== 'Solved';
          button.textContent = `${warning ? '!' : '●'} #${s.id}`; button.dataset.warning = String(warning);
          button.title = `Seek to plan #${s.id} at ${fmt(s.t)} s · ${s.fallback ? 'fallback' : s.status ?? 'unknown'} · ${fmt(s.ms, 0)} ms`;
          button.setAttribute('aria-label', button.title); button.onclick = () => onSeek(s.t);
          return button;
        }));
      }
      drawSolver(canvas.solver, solves, time);
      drawTrend(canvas.tracking, history.map(f => [f.t, f.error]), [], time, { color: ink.blue, rms, minimum: .01 });
      drawTrend(canvas.energy, history.map(f => [f.t, f.energy]), solves.map(s => [s.t, s.energy]), time, { color: ink.white, minimum: .1 });
      drawTrend(canvas.reserve, history.map(f => [f.t, f.reserve]), [], time, { color: ink.green, minimum: 1, signed: true });
      for (const id of Object.keys(canvas)) canvas[id].setAttribute('aria-label', `${summaries[id].textContent}. ${details[id].textContent}. History through ${fmt(time)} seconds.`);
    },
  };
}

function axes(canvas, time, low, high) {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return null;
  const ctx = fitCanvas(canvas, w, h), left = 46, right = w - 14, top = 12, bottom = h - 26;
  const x = t => left + t / Math.max(time, 1) * (right - left);
  const y = v => bottom - (v - low) / (high - low) * (bottom - top);
  ctx.font = '10px Consolas, monospace'; ctx.lineWidth = 1;
  const decimals = high - low < .1 ? 3 : high - low < 1 ? 2 : 1;
  for (let i = 0; i <= 2; i++) {
    const value = low + (high - low) * i / 2, yy = y(value);
    ctx.strokeStyle = ink.grid; ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(right, yy); ctx.stroke();
    ctx.fillStyle = ink.muted; ctx.textAlign = 'right'; ctx.fillText(fmt(value, decimals), left - 6, yy + 3);
    ctx.textAlign = 'center'; ctx.fillText(`${fmt(Math.max(time, 1) * i / 2, 1)}s`, left + (right - left) * i / 2, h - 7);
  }
  ctx.save(); ctx.beginPath(); ctx.rect(left - 1, top - 1, right - left + 2, bottom - top + 2); ctx.clip();
  return { ctx, x, y, left, right, top, bottom, w, h };
}

function empty(p, label = 'No recorded samples yet') {
  p.ctx.fillStyle = ink.muted; p.ctx.textAlign = 'center'; p.ctx.fillText(label, (p.left + p.right) / 2, (p.top + p.bottom) / 2);
}

function drawSolver(canvas, solves, time) {
  const values = solves.filter(s => finite(s.ms));
  const maximum = values.reduce((n, s) => Math.max(n, s.ms), 1);
  const p = axes(canvas, time, 0, maximum * 1.25); if (!p) return;
  if (!values.length) empty(p, 'Awaiting a recorded solve');
  for (let i = 0; i < values.length; i++) {
    const s = values[i], warning = s.fallback || s.status !== 'Solved';
    const spacing = Math.min(i ? s.t - values[i - 1].t : Infinity, i + 1 < values.length ? values[i + 1].t - s.t : Infinity);
    const width = Math.max(1, Math.min(14, (p.x(spacing) - p.x(0)) * .65));
    const xx = Math.min(p.right - width / 2, Math.max(p.left + width / 2, p.x(s.t)));
    p.ctx.fillStyle = warning ? ink.amber : ink.green;
    p.ctx.globalAlpha = i === values.length - 1 ? 1 : .55;
    p.ctx.fillRect(xx - width / 2, p.y(s.ms), width, p.bottom - p.y(s.ms));
    p.ctx.globalAlpha = 1;
    if (warning) { p.ctx.textAlign = 'center'; p.ctx.fillText('!', xx, p.y(s.ms) - 4); }
  }
  p.ctx.restore();
}

function path(p, points, color, dashed = false) {
  p.ctx.strokeStyle = color; p.ctx.lineWidth = 1.8; p.ctx.setLineDash(dashed ? [4, 3] : []); p.ctx.beginPath();
  let connected = false;
  for (const [t, v] of points) {
    if (!finite(v)) { connected = false; continue; }
    if (connected) p.ctx.lineTo(p.x(t), p.y(v)); else p.ctx.moveTo(p.x(t), p.y(v));
    connected = true;
  }
  p.ctx.stroke(); p.ctx.setLineDash([]);
}

function drawTrend(canvas, points, secondary, time, options) {
  const values = [...points, ...secondary].map(p => p[1]).filter(finite);
  let low = 0, high = options.minimum;
  for (const v of values) { low = Math.min(low, v); high = Math.max(high, v); }
  const margin = (high - low) * .15;
  const p = axes(canvas, time, low < 0 ? low - margin : 0, high + margin); if (!p) return;
  if (!values.length) empty(p);
  if (options.signed) {
    p.ctx.fillStyle = 'rgba(72,219,162,.06)'; p.ctx.fillRect(p.left, p.top, p.right - p.left, p.y(0) - p.top);
    p.ctx.fillStyle = 'rgba(250,178,25,.12)'; p.ctx.fillRect(p.left, p.y(0), p.right - p.left, p.bottom - p.y(0));
    path(p, [[0, 0], [Math.max(time, 1), 0]], ink.muted, true);
  }
  if (finite(options.rms)) path(p, [[0, options.rms], [time, options.rms]], ink.muted, true);
  path(p, points, options.color);
  if (options.signed) {
    p.ctx.save(); p.ctx.beginPath(); p.ctx.rect(p.left, p.y(0), p.right - p.left, p.bottom - p.y(0)); p.ctx.clip();
    path(p, points, ink.amber); p.ctx.restore();
  }
  // Energy plan costs are discrete points: no interpolation or accumulation
  // between different horizons and no fabricated battery-capacity estimate.
  for (const [t, v] of secondary) {
    if (!finite(v)) continue;
    const xx = p.x(t), yy = p.y(v); p.ctx.beginPath();
    p.ctx.moveTo(xx, yy - 3); p.ctx.lineTo(xx + 3, yy); p.ctx.lineTo(xx, yy + 3); p.ctx.lineTo(xx - 3, yy); p.ctx.closePath();
    p.ctx.strokeStyle = ink.green; p.ctx.lineWidth = 1.2; p.ctx.stroke();
  }
  const last = points.at(-1);
  if (finite(last?.[1])) { p.ctx.beginPath(); p.ctx.arc(p.x(last[0]), p.y(last[1]), 3, 0, 2 * Math.PI); p.ctx.fillStyle = options.signed && last[1] < 0 ? ink.amber : options.color; p.ctx.fill(); }
  p.ctx.restore();
}
