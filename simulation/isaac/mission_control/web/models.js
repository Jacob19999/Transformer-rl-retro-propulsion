// Flight software panel: the policy mission control can fly, and the latest
// waypoint_flight training runs read from their logs (inspection only).
import { fitCanvas, ink, series, status } from './instruments.js';

const fmt = (n, d = 1) => Number.isFinite(n) ? n.toFixed(d) : '—';
const pct = n => Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : '—';
const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
// Outcome colours: success is the only good state; every failure mode is reserved status ink.
const outcomeTone = { SUCCESS: status.good, SPIN: status.critical, CRASH: status.serious, GEOFENCE: status.warning, TIMEOUT: '#5d6770' };
const runLabel = name => { const m = /(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/.exec(name); return m ? `${m[2]}-${m[3]} ${m[4]}:${m[5]}` : name; };

export function createModels(root, api) {
  let data = null, selected = null, history = null;
  root.innerHTML = `<div class="panel-title">FLIGHT SOFTWARE <span>MODELS & TRAINING</span></div><div class="models-body"></div>`;
  const body = root.querySelector('.models-body');

  function outcomeBar(outcomes) {
    if (!outcomes) return '<div class="outcome-bar empty"></div>';
    const entries = Object.entries(outcomes).filter(([, v]) => v > .001).sort((a, b) => b[1] - a[1]);
    return `<div class="outcome-bar" role="img" aria-label="${entries.map(([k, v]) => `${k} ${pct(v)}`).join(', ')}">${entries.map(([k, v]) =>
      `<i style="width:${v * 100}%;background:${outcomeTone[k] ?? '#8a6fdf'}" title="${k} ${pct(v)}"></i>`).join('')}</div>
      <div class="outcome-legend">${entries.slice(0, 4).map(([k, v]) => `<span><i style="background:${outcomeTone[k] ?? '#8a6fdf'}"></i>${k} ${pct(v)}</span>`).join('')}</div>`;
  }

  function render() {
    if (!data) return;
    const flyable = data.flyable.map(p => `<div class="model-card flyable">
      <div class="model-row"><b>${escape(p.label ?? (p.key === 'ppo_mission' ? 'PPO · LANDING + WAYPOINTS (43-OBS)' : p.key.toUpperCase()))}</b>
      <span class="badge ${p.validated ? 'go' : 'hold'}">${p.validated ? '✓ VALIDATED' : '▲ ' + escape((p.status ?? 'unqualified').toUpperCase())}</span></div>
      <div class="mono">${escape(p.checkpoint)}</div>
      ${p.note ? `<details><summary>Qualification notes</summary><p>${escape(p.note)}</p></details>` : ''}</div>`).join('')
      || '<div class="hint">No PPO checkpoint is registered for missions. Convex guidance remains available.</div>';
    const runs = data.runs.map(r => {
      const stageCount = r.stages.length, latest = r.checkpoints.at(-1);
      return `<button type="button" class="model-card run ${r.run === selected ? 'selected' : ''}" data-run="${escape(r.run)}">
        <div class="model-row"><b>WAYPOINT FLIGHT · ${escape(runLabel(r.run))}</b>${r.active ? '<span class="badge live">● TRAINING</span>' : r.evaluation ? `<span class="badge ${r.evaluation.passed ? 'go' : 'nogo'}">EVAL ${pct(r.evaluation.success_fraction)}</span>` : ''}</div>
        <div class="stage-track" aria-label="Curriculum stage ${(r.stage ?? 0) + 1} of ${stageCount}">${r.stages.map((name, i) => `<i class="${i < r.stage ? 'done' : i === r.stage ? 'current' : ''}" title="${escape(name)}"></i>`).join('')}</div>
        <div class="model-stats">
          <span>STEPS<b>${fmt((r.step ?? 0) / 1e6, 1)}M</b></span>
          <span>STAGE<b>${r.stage == null ? '—' : r.stage + 1}/${stageCount || '—'}</b></span>
          <span>STAGE SUCCESS<b>${pct(r.stage_success)}</b></span>
          <span>PEAK YAW<b>${fmt(r.peak_yaw, 0)}°/s</b></span>
        </div>
        <div class="stage-name">${escape((r.stage_name ?? '').replaceAll('_', ' ').toUpperCase())}${latest ? ` · ${escape(latest.file)}` : ''}</div>
        ${outcomeBar(r.outcomes)}
      </button>`;
    }).join('');
    const run = data.runs.find(r => r.run === selected);
    const evaluation = run?.evaluation;
    const detail = run ? `<div class="run-detail">
        <div class="section-label">TRAINING HISTORY <span>${escape(runLabel(run.run))}${run.active ? ' · LIVE' : ''}</span></div>
        <div class="chart-legend history-legend"><span><i style="background:${series[0]}"></i>STAGE SUCCESS</span><span><i style="background:${series[1]}"></i>SPIN TERMINATIONS</span><span><i style="background:${series[2]}"></i>EXPLAINED VAR.</span></div>
        <canvas id="historyChart" aria-label="Training history: stage success, spin fraction and explained variance per update"></canvas>
        <div class="chart-legend history-legend"><span><i style="background:${series[3]}"></i>MEAN PEAK YAW °/s</span></div>
        <canvas id="yawChart" aria-label="Training history: mean peak yaw rate per update"></canvas>
        <div class="hint">Dotted hairlines mark curriculum advances. Resumed from ${escape(run.resume ?? 'scratch')} · ${fmt(run.num_envs, 0)} envs · ${fmt(run.sps, 0)} steps/s · EV ${fmt(run.explained_variance, 2)} · throttle ${fmt(run.throttle, 2)}</div>
        <div class="detail-columns">
          <div><div class="section-label">CURRICULUM</div><ol class="stage-list">${run.stages.map((name, i) =>
            `<li class="${i < run.stage ? 'done' : i === run.stage ? 'current' : ''}"><i>${i < run.stage ? '✓' : i === run.stage ? '▶' : '·'}</i>${escape(name.replaceAll('_', ' ').toUpperCase())}</li>`).join('')}</ol></div>
          <div><div class="section-label">FULL-TASK EVALUATION</div>${evaluation ? `<div class="model-stats two">
            <span>SUCCESS<b>${pct(evaluation.success_fraction)}</b></span><span>AT STEP<b>${fmt(evaluation.global_step / 1e6, 1)}M</b></span>
            <span>CAPTURE<b>${pct(evaluation.capture_fraction)}</b></span><span>P95 PEAK YAW<b>${fmt(evaluation.p95_peak_yaw_deg_s, 0)}°/s</b></span></div>
            ${outcomeBar(evaluation.outcomes)}` : '<p class="hint">No full-task evaluation recorded yet.</p>'}
            <div class="section-label">CHECKPOINTS <span>${run.checkpoints.length}</span></div>
            <ul class="checkpoint-list">${run.checkpoints.slice().reverse().map(c => `<li><span class="mono">${escape(c.file)}</span><em>${fmt(c.bytes / 1e6, 1)} MB</em></li>`).join('') || '<li class="hint">None saved yet.</li>'}</ul></div>
        </div></div>` : '<div class="hint">Select a training run.</div>';
    body.innerHTML = `<div class="models-grid"><div class="models-list"><div class="section-label">MISSION-FLYABLE</div>${flyable}
      <div class="section-label">LATEST TRAINING RUNS <span>runs/waypoint_flight</span></div>
      <p class="hint">54-channel waypoint_flight checkpoints are for inspection here. Flying one needs run_mission.py support for explicit waypoint missions.</p>
      <div class="run-list">${runs || '<div class="hint">No waypoint_flight runs found.</div>'}</div></div>
      <div class="models-detail">${detail}</div></div>`;
    body.querySelectorAll('[data-run]').forEach(el => el.onclick = () => select(el.dataset.run));
    drawHistory();
  }

  function drawHistory() {
    const canvas = body.querySelector('#historyChart'), yaw = body.querySelector('#yawChart');
    if (!canvas || !history || history.run !== selected) return;
    const points = history.series.filter(p => Number.isFinite(p.step));
    const plot = (el, lines, maxY, fixed) => {
      const w = el.clientWidth, h = el.clientHeight, ctx = fitCanvas(el, w, h), left = 34, right = 6, top = 6, bottom = 16;
      const x0 = points[0]?.step ?? 0, x1 = Math.max(points.at(-1)?.step ?? 1, x0 + 1), x = s => left + (s - x0) / (x1 - x0) * (w - left - right), y = v => top + (1 - v / maxY) * (h - top - bottom);
      ctx.font = '9px Bahnschrift, "Segoe UI", sans-serif'; ctx.fillStyle = ink.muted; ctx.textAlign = 'right';
      for (const v of [0, maxY / 2, maxY]) { ctx.strokeStyle = ink.grid; ctx.beginPath(); ctx.moveTo(left, y(v)); ctx.lineTo(w - right, y(v)); ctx.stroke(); ctx.fillText(fixed(v), left - 4, y(v) + 3); }
      ctx.textAlign = 'center'; ctx.fillText(`${fmt(x0 / 1e6, 0)}M`, left + 8, h - 3); ctx.fillText(`${fmt(x1 / 1e6, 0)}M`, w - right - 12, h - 3);
      // Curriculum advances as vertical hairlines.
      points.forEach((p, i) => { if (i && p.stage !== points[i - 1].stage) { ctx.strokeStyle = 'rgba(255,255,255,.3)'; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(x(p.step), top); ctx.lineTo(x(p.step), h - bottom); ctx.stroke(); ctx.setLineDash([]); } });
      lines.forEach(([key, color]) => { ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.beginPath(); let pen = false; for (const p of points) { const v = p[key]; if (!Number.isFinite(v)) { pen = false; continue; } const yy = y(Math.max(0, Math.min(maxY, v))); pen ? ctx.lineTo(x(p.step), yy) : ctx.moveTo(x(p.step), yy); pen = true; } ctx.stroke(); });
    };
    plot(canvas, [['stage_success', series[0]], ['spin', series[1]], ['explained_variance', series[2]]], 1, v => `${Math.round(v * 100)}%`);
    const peak = Math.max(180, ...points.map(p => p.peak_yaw).filter(Number.isFinite));
    plot(yaw, [['peak_yaw', series[3]]], Math.ceil(peak / 90) * 90, v => fmt(v, 0));
  }

  async function select(run) {
    selected = run; render();
    try { history = await api(`/api/models/${encodeURIComponent(run)}/history`); if (run === selected) drawHistory(); } catch { history = null; }
  }

  let lastJson = '';
  async function refresh() {
    const next = await api('/api/models'), json = JSON.stringify(next);
    // Re-rendering resets open notes and scroll, so only redraw on new records.
    if (json === lastJson) return data;
    lastJson = json; data = next;
    if (!selected || !data.runs.some(r => r.run === selected)) selected = data.active_run ?? data.runs[0]?.run ?? null;
    if (selected) await select(selected); else render();
    return data;
  }
  new ResizeObserver(() => drawHistory()).observe(root);
  return { refresh };
}
