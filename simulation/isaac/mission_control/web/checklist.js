// Pre-flight checklist for the physical EDF testbed. Each tick is an operator
// attestation stored in this browser with the time it was made; nothing here is
// read from hardware and it never gates an Isaac simulation run.
const storageKey = 'edfMissionControl.preflight.v1';
const groups = [
  ['POWER', [
    ['charge', 'Battery charged & balanced', c => `${c.cells}S pack reads ${(c.cells * 4.2).toFixed(1)} V when full; plan assumes ${c.soc}% SOC`],
    ['pack_secure', 'Pack secured', 'Strap tight, main connector fully seated, leads clear of the EDF intake'],
    ['servo_supply', 'Servo supply verified', 'Regulated 6 V at the servo rail with all four fins moving'],
  ]],
  ['SENSORS', [
    ['imu_alignment', 'IMU alignment', 'Autopilot board orientation matches body FRD; level calibration done on a flat surface'],
    ['gyro_bias', 'Gyro at rest', 'Vehicle still after power-up; body rates read ≈ 0 °/s on the ground station'],
    ['heading', 'Heading reference', 'Compass or mocap heading agrees with the pad axes'],
    ['position_fix', 'Position & altitude fix', 'Position source healthy; altitude reads ≈ 0 m on the pad'],
  ]],
  ['ACTUATORS', [
    ['fin_neutral', 'Fins at neutral', 'All four fins centred at 0° with the servo horns square'],
    ['fin_sweep', 'Fin sweep & sign', c => `Command ±${c.finLimit.toFixed(0)}° on FWD / RIGHT / AFT / LEFT; motion matches the mixer sign with no binding`],
    ['linkages', 'Linkages tight', 'Horns, pushrods and hinge pins secure with no play'],
  ]],
  ['PROPULSION', [
    ['edf_inspect', 'EDF rotor & duct', 'No chipped blades or debris in the intake; duct and motor mount screws tight'],
    ['esc_calibration', 'ESC calibrated', 'Throttle endpoints set; rotor spins the correct way at idle'],
  ]],
  ['AIRFRAME', [
    ['mass_cg', 'Mass & CG', c => `Weigh the assembled vehicle (simulation assumes ${c.mass.toFixed(3)} kg); CG on the thrust axis`],
    ['structure', 'Structure & legs', 'Carbon tubes, fasteners and landing legs undamaged'],
  ]],
  ['RANGE & SAFETY', [
    ['range_clear', 'Range clear', 'Personnel behind the safety line; landing pad marked at the origin'],
    ['kill_switch', 'Kill switch tested', 'Disarm and RC failsafe both drop throttle to zero'],
    ['telemetry_link', 'Telemetry & logging', 'Ground station connected; onboard log recording'],
    ['mission_plan', 'Mission plan reviewed', c => `Controller ${c.controller}; route, disturbances and initial state match the test card`],
  ]],
];
const items = groups.flatMap(([, list]) => list);

function load() { try { return JSON.parse(localStorage.getItem(storageKey)) ?? {}; } catch { return {}; } }
function save(checked) { try { localStorage.setItem(storageKey, JSON.stringify(checked)); } catch { /* checks still work for this page view */ } }
const stamp = ms => new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });

export function createChecklist(root, { context, onChange }) {
  let checked = load();
  root.innerHTML = `<div class="panel-title">PRE-FLIGHT CHECKLIST <span id="checklistCount"></span></div><div class="card-body">
    <div class="bar"><i id="checklistBar"></i></div>
    <div class="checklist-groups">${groups.map(([name, list], g) => `<div class="check-group"><div class="section-label">${name} <span id="checklistGroup${g}"></span></div>
      ${list.map(([id, label]) => `<label class="check-item"><input type="checkbox" data-check="${id}"><div><b>${label}</b><small data-detail="${id}"></small></div><time data-time="${id}"></time></label>`).join('')}</div>`).join('')}</div>
    <div class="checklist-actions"><span class="hint">Operator checks, saved in this browser. Hardware is not queried.</span><button type="button" id="checklistReset">RESET</button></div></div>`;
  const q = (attr, id) => root.querySelector(`[data-${attr}="${id}"]`);

  function update() {
    const c = context();
    for (const [id, , detail] of items) {
      const text = typeof detail === 'function' ? detail(c) : detail, el = q('detail', id);
      if (el.textContent !== text) el.textContent = text;
      q('time', id).textContent = id in checked ? stamp(checked[id]) : '';
    }
    const { done, total } = summary();
    root.querySelector('#checklistCount').textContent = `${done} / ${total} COMPLETE`;
    root.querySelector('#checklistBar').style.width = `${done / total * 100}%`;
    root.classList.toggle('complete', done === total);
    groups.forEach(([, list], g) => { root.querySelector(`#checklistGroup${g}`).textContent = `${list.filter(([id]) => id in checked).length} / ${list.length}`; });
  }
  function summary() { return { done: items.filter(([id]) => id in checked).length, total: items.length }; }

  root.addEventListener('change', e => {
    const id = e.target.dataset?.check; if (!id) return;
    if (e.target.checked) checked[id] = Date.now(); else delete checked[id];
    save(checked); update(); onChange?.();
  });
  // Boxes are written only here: update() also runs on form input events, which
  // fire before a toggled box's change event and would otherwise undo the click.
  const syncBoxes = () => items.forEach(([id]) => { q('check', id).checked = id in checked; });
  root.querySelector('#checklistReset').onclick = () => { checked = {}; save(checked); syncBoxes(); update(); onChange?.(); };
  syncBoxes(); update();
  return { update, summary };
}
