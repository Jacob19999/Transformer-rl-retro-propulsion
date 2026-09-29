import test from 'node:test';
import assert from 'node:assert/strict';
import { collectPlans, planAt, guidanceStatus, costBreakdown } from './guidance.js';
import { diagnosticSnapshot } from './guidance-diagnostics.js';

test('replay selects only a plan already recorded, including backward seeks', () => {
  const frames = [
    { t: 0, guidance: { phase: 'SPOOL_UP' } },
    { t: 1, guidance: { plan_id: 1, plan: { positions: [[0, 0, 5]], times: [0], thrust_n: [30] } } },
    { t: 1.5, guidance: { plan_id: 1 } },
    { t: 2, guidance: { plan_id: 2, plan: { positions: [[0, 0, 4]], times: [0], thrust_n: [31] } } },
  ];
  const plans = collectPlans(frames);
  assert.equal(plans.length, 2);
  assert.equal(planAt(plans, .9), null);
  assert.equal(planAt(plans, 2).id, 2);
  assert.equal(planAt(plans, 1.9).id, 1);
  assert.deepEqual(planAt(plans, 1).thrust_n, [30]);
  assert.equal(planAt(plans, 30).id, 2);
  assert.equal(planAt(collectPlans([{ t: 0 }]), 0), null);
});

test('failed and approximate solver states cannot appear as solved', () => {
  const g = { phase: 'POWERED_DESCENT', solver: { mode: 'optimal', status: 'Solved' } };
  assert.equal(guidanceStatus(g).tone, 'good');
  assert.equal(guidanceStatus({ ...g, solver: { mode: 'optimal', status: 'AlmostSolved' } }).label, 'ALMOSTSOLVED');
  assert.equal(guidanceStatus({ ...g, solver: { mode: 'soft_terminal', status: 'Solved' } }).tone, 'warning');
  assert.equal(guidanceStatus({ ...g, phase: 'HOLD' }).tone, 'warning');
  assert.match(guidanceStatus({ ...g, phase: 'TERMINAL_DESCENT' }).label, /LAST PLAN/);
  assert.match(guidanceStatus({ ...g, phase: 'LANDED' }).label, /LAST PLAN/);
  assert.equal(guidanceStatus({ phase: 'SPOOL_UP' }).tone, 'idle');
});

test('diagnostics count replans once and discard future peaks and energy on backward seeks', () => {
  const solver = { objective: 'energy', energy_wh: 2, solve_ms: 25, status: 'Solved', mode: 'optimal' };
  const g = { plan_id: 1, plan: { positions: [[0, 0, 5]] }, solver, tracking_error_m: .3, thrust_available_n: 40, thrust_command_n: 30 };
  const frames = [
    { t: 1, battery: { energy_wh: .5 }, guidance: g },
    { t: 2, battery: { energy_wh: .8 }, guidance: { ...g, plan: undefined, tracking_error_m: .4 } },
    { t: 3, battery: { energy_wh: 1.2 }, guidance: { ...g, plan_id: 2, tracking_error_m: 10, thrust_command_n: 45,
      solver: { ...solver, mode: 'soft_terminal', energy_wh: 3 } } },
  ];
  assert.equal(diagnosticSnapshot(frames, 3).peak, 10);
  const back = diagnosticSnapshot(frames, 2);
  assert.equal(back.peak, .4);
  assert.ok(Math.abs(back.rms - Math.sqrt(.125)) < 1e-10);
  assert.equal(back.solves.length, 1);
  assert.equal(back.solves[0].energy, 2);
  assert.equal(back.history.at(-1).energy, .8);
  assert.equal(back.history.at(-1).reserve, 10);
  assert.equal(diagnosticSnapshot(frames, 3).history.at(-1).reserve, -5);
  assert.equal(diagnosticSnapshot(frames, 3).solves[1].fallback, true);
});

test('missing telemetry stays missing; delta-v objectives never become energy points', () => {
  const frames = [{ t: 0 }, { t: 1, guidance: { plan_id: 1, plan: {},
    solver: { objective: 'delta_v', energy_wh: 12 }, thrust_available_n: 40, thrust_command_n: null } }];
  const result = diagnosticSnapshot(frames, 1);
  assert.equal(result.rms, null);
  assert.equal(result.peak, null);
  assert.equal(result.history.at(-1).reserve, null);
  assert.equal(result.solves[0].energy, null);
  assert.equal(result.history.at(-1).energy, undefined);
  assert.deepEqual(diagnosticSnapshot(frames, -1).history, []);
});

test('cost breakdown lists priced terms largest first in the objective unit', () => {
  const solver = { objective: 'energy', cost_terms: { energy: 19.1, path: 0.8, smoothness: 1.2, corridor: 0, time: NaN } };
  assert.equal(costBreakdown(solver), 'energy 19.10 Wh, smoothness 1.20 Wh, path 0.80 Wh');
  assert.equal(costBreakdown({ objective: 'delta_v', cost_terms: { delta_v: 400.123 } }), 'delta-v 400.12 m/s');
  assert.equal(costBreakdown({ objective: 'energy' }), '');
  assert.equal(costBreakdown(null), '');
});
