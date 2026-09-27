import {escapeHtml} from './flight-plan.js';

export const optimizerGroups={
  'Route corridor':'Keep the reference plan near the drawn route. Per-waypoint widths override the default.',
  'Kinematic limits':'Bound planned speed, altitude and thrust direction through the approach.',
  'Thrust':'Leave feedback authority available while limiting thrust and actuator slew.',
  'Landing':'Define the gate where powered descent hands over to terminal pad centering.',
  'Objective':'Choose what the optimizer minimizes along a feasible trajectory.',
  'Discretization':'Balance trajectory resolution, solver budget and CPU concurrency.',
  'Re-planning':'Decide when to solve again from the measured state and how to blend the new plan.',
  'Tracking feedback':'Tune position, velocity and integral feedback around the planned trajectory.',
};

// These diagrams explain configured constraints; they are not solved trajectories.
export function optimizerVisual(group,get){
  const g=key=>get('guidance',key),t=key=>get('tracking',key);
  let drawing='',caption='',metrics=[];
  const text=(x,y,label)=>`<text x="${x}" y="${y}">${escapeHtml(label)}</text>`;
  const line=(x1,y1,x2,y2,cls='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="${cls}"/>`;
  if(group==='Route corridor'){
    const width=10+Math.min(25,g('route_corridor_m'))*2;
    drawing=`<path class="diagram-band" style="stroke-width:${width}" d="M25 110 C85 110 105 40 165 40 S245 80 290 60"/><path class="diagram-route" d="M25 110 C85 110 105 40 165 40 S245 80 290 60"/><circle cx="25" cy="110" r="5"/><circle cx="290" cy="60" r="5"/>${line(165,40-width/2,165,40+width/2,'diagram-dimension')}${text(130,125,'DRAWN REFERENCE')}`;
    caption='Corridor around the reference · schematic';
    metrics=[['Half-width',`${g('route_corridor_m')} m`],['Enforcement',g('route_corridor_mode')],['Fly-through aim',`${g('flypass_capture_fraction')} × radius`]];
  }else if(group==='Kinematic limits'){
    const a=g('max_tilt_deg')*Math.PI/180,dx=95*Math.sin(a),dy=95*Math.cos(a);
    drawing=`<path class="diagram-fill" d="M155 128 L${155-dx} ${128-dy} A95 95 0 0 1 ${155+dx} ${128-dy} Z"/>${line(155,128,155,20,'diagram-dashed')}${line(155,128,155+dx,128-dy,'diagram-route')}<circle cx="155" cy="128" r="5"/>${text(175,95,`${g('max_tilt_deg')}°`)}${text(175,28,'VERTICAL')}`;
    caption='Planned thrust-pointing cone · angle to vertical';
    metrics=[['Speed ceiling',`${g('max_speed_m_s')} m/s`],['Route floor',`${g('route_floor_m')} m`],['Glide half-angle',`${g('glide_slope_deg')}°`]];
  }else if(group==='Thrust'){
    const reserve=g('thrust_excess_reserve_fraction'),floor=g('thrust_min_weight_fraction');
    drawing=`<rect class="diagram-fill" x="25" y="62" width="270" height="24"/><rect class="diagram-reserve" x="${135+160*(1-reserve)}" y="62" width="${160*reserve}" height="24"/>${line(135,50,135,99,'diagram-dimension')}${line(25+110*floor,57,25+110*floor,91,'diagram-route')}${text(25,40,'MIN')}${text(118,118,'WEIGHT')}${text(207,40,'RESERVE')}`;
    caption='Thrust allocation · schematic, no hardware maximum implied';
    metrics=[['Minimum thrust',`${floor} × weight`],['Excess reserved',`${Math.round(reserve*100)}%`],['Duty slew',`${g('throttle_rate_per_s')} /s`]];
  }else if(group==='Landing'){
    const height=35+g('gate_height_m')*22,radius=15+g('gate_capture_radius_m')*30;
    drawing=`${line(28,133,290,133)}${line(160,15,160,133,'diagram-dashed')}<ellipse class="diagram-fill" cx="160" cy="${133-height}" rx="${radius}" ry="10"/>${line(75,133-height,75,133,'diagram-dimension')}${text(20,125,`${g('gate_height_m')} m`)}${text(215,133-height-12,'GATE')}<rect class="diagram-pad" x="140" y="129" width="40" height="5"/>${text(140,153,'PAD')}`;
    caption='Gate height above touchdown · schematic';
    metrics=[['Gate height',`${g('gate_height_m')} m`],['Capture radius',`${g('gate_capture_radius_m')} m`],['Descent clamp',`${g('terminal_max_descent_m_s')} m/s`]];
  }else if(group==='Objective'){
    drawing=`<path class="diagram-band" d="M30 115 Q110 5 290 50"/><path class="diagram-route" d="M30 115 Q110 5 290 50"/><circle cx="30" cy="115" r="5"/><circle cx="290" cy="50" r="5"/>${text(25,145,'START')}${text(251,80,'TARGET')}${text(95,105,g('objective')==='energy'?'∫ electrical power dt':'∫ |T| / m dt')}`;
    caption='Cost integrated along the feasible trajectory · schematic';
    metrics=[['Active cost',g('objective')==='energy'?'Electrical energy':'Delta-v'],['Energy model','Momentum theory'],['Delta-v model','Thrust / mass']];
  }else if(group==='Discretization'){
    drawing=`${line(25,80,295,80,'diagram-route')}${Array.from({length:9},(_,i)=>`<circle cx="${25+i*33.75}" cy="80" r="4"/>`).join('')}${line(25,110,58.75,110,'diagram-dimension')}${text(25,140,`ROUTE INTERVAL ${g('route_dt_s')} s`)}${text(25,45,'DISCRETE PLAN NODES')}`;
    caption='Node spacing along each leg · schematic';
    metrics=[['Landing intervals',g('landing_nodes')],['Nodes / leg cap',g('max_leg_nodes')],['Budget / solve',`${g('max_solve_time_s')} s`]];
  }else if(group==='Re-planning'){
    drawing=`${line(25,80,295,80)}${[35,100,165].map(x=>`${line(x,65,x,95,'diagram-route')}<circle cx="${x}" cy="80" r="4"/>`).join('')}<rect class="diagram-reserve" x="220" y="65" width="65" height="30"/>${text(25,45,'RE-SOLVE')}${text(221,45,'FREEZE')}${text(25,130,`${g('replan_period_s')} s PERIOD`)}${text(211,130,`${g('freeze_time_s')} s`)}`;
    caption='Periodic updates stop near the landing gate · schematic';
    metrics=[['Position trigger',`${g('replan_error_m')} m`],['Velocity trigger',`${g('replan_velocity_error_m_s')} m/s`],['Blend time',`${g('replan_blend_s')} s`]];
  }else{
    drawing=`<path class="diagram-dashed" d="M25 95 Q150 10 290 60"/><path class="diagram-route" d="M25 115 Q150 60 290 60"/>${line(155,90,155,50,'diagram-dimension')}${text(30,45,'REFERENCE')}${text(30,143,'MEASURED → CORRECTION')}`;
    caption='Feedback corrects deviation from the reference · schematic';
    metrics=[['Correction bound',`${t('max_correction_m_s2')} m/s²`],['Command tilt',`${t('max_tilt_deg')}°`],['Planned tilt',`${g('max_tilt_deg')}°`]];
  }
  return `<figure><svg viewBox="0 0 320 165" role="img" aria-label="${escapeHtml(caption)}">${drawing}</svg><figcaption>${escapeHtml(caption)}</figcaption></figure><div class="optimizer-visual-metrics">${metrics.map(([label,v])=>`<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(v??'—')}</strong></div>`).join('')}</div>`;
}

// One schematic of the key constraints (not to scale): thrust cones of the
// planned and feedback tilt, the corridor band and speed cap, the route floor,
// and the landing glide-slope cone with its gate.
export function envelopeVisual(get){
  const g=key=>get('guidance',key),t=key=>get('tracking',key);
  if(g('max_tilt_deg')===undefined)return '';
  const rad=d=>d*Math.PI/180,ground=205,padX=305;
  const cone=(deg,r)=>{const a=rad(Math.min(80,deg));return `M92 70 L${(92-r*Math.sin(a)).toFixed(1)} ${(70-r*Math.cos(a)).toFixed(1)} A${r} ${r} 0 0 1 ${(92+r*Math.sin(a)).toFixed(1)} ${(70-r*Math.cos(a)).toFixed(1)} Z`;};
  const glide=rad(Math.min(85,g('glide_slope_deg'))),height=150,spread=Math.min(170,height*Math.tan(glide));
  const width=Math.max(6,Math.min(46,4+g('route_corridor_m')*9)),strict=g('route_corridor_mode')==='strict';
  const band='M104 84 C170 92 220 70 '+(padX-18)+' 118';
  const text=(x,y,label,cls='')=>`<text x="${x}" y="${y}" class="${cls}">${escapeHtml(label)}</text>`;
  return `<svg viewBox="0 0 420 230" role="img" aria-label="Schematic of the configured guidance envelope">
    <path class="env-glide" d="M${padX} ${ground} L${padX-spread} ${ground-height} L${padX+spread} ${ground-height} Z"/>
    <line class="env-glide-edge" x1="${padX}" y1="${ground}" x2="${padX-spread}" y2="${ground-height}"/><line class="env-glide-edge" x1="${padX}" y1="${ground}" x2="${padX+spread}" y2="${ground-height}"/>
    <path class="env-band ${strict?'is-strict':''}" d="${band}" style="stroke-width:${width}"/><path class="env-route ${strict?'':'is-soft'}" d="${band}"/>
    <line class="env-floor" x1="16" x2="250" y1="${ground-16}" y2="${ground-16}"/>${text(18,ground-21,`route floor ${g('route_floor_m')} m`)}
    <line class="env-ground" x1="8" x2="412" y1="${ground}" y2="${ground}"/><rect class="env-pad" x="${padX-22}" y="${ground-3}" width="44" height="5"/>
    <line class="env-gate" x1="${padX-16}" x2="${padX+16}" y1="${ground-26}" y2="${ground-26}"/>${text(padX+22,ground-22,`gate ${g('gate_height_m')} m`)}
    ${text(padX-spread+4,ground-height-8,`glide ±${g('glide_slope_deg')}°`,'env-label')}
    <path class="env-command" d="${cone(t('max_tilt_deg'),54)}"/><path class="env-plan" d="${cone(g('max_tilt_deg'),46)}"/><line class="env-axis" x1="92" y1="70" x2="92" y2="8"/>
    <rect class="env-vehicle" x="84" y="66" width="16" height="22" rx="3"/>
    ${text(14,112,`plan ≤ ${g('max_tilt_deg')}°`,'env-label env-plan-text')}${text(14,127,`feedback ≤ ${t('max_tilt_deg')}°`,'env-label env-command-text')}
    ${text(150,64,`${strict?'STRICT':'SOFT'} corridor ±${g('route_corridor_m')} m · ≤ ${g('max_speed_m_s')} m/s`,'env-label')}
    ${text(18,224,`cost: ${g('objective')==='delta_v'?'propulsive delta-v':'electrical energy'} · schematic, not to scale`)}
  </svg>`;
}
