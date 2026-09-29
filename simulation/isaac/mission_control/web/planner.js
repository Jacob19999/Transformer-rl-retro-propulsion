// Route design: plan library and samples, the 3D editor with an altitude
// profile, and the flight sequence. Two synchronized projections (precision
// views) keep XYZ dragging unambiguous on a flat screen.
import { fitCanvas } from './instruments.js';
import {waypointTypes,waypointColors,waypointLabel,speedLabel,escapeHtml,normalizeWaypoints,parseFlightPlan,serializeRoute,pickRoute,sampleRouteLegs,routeProfile,categoryNames} from './flight-plan.js';
import {createPlanner3D} from './planner-3d.js';

const stepHelp={takeoff:'Vertical climb from the start. Always the first step.',hover:'Fly to a point, stop, and hold for a set time.',
  flypass:'Pass through a point without stopping.',descent:'Vertical descent below the previous point.',land:'Final step: touch down on a chosen pad.'};
const completion={hover:'Arrival: stay inside the capture radius below 0.4 m/s for the full hold. A speed excursion of up to 1 s (a gust) pauses the timer; leaving the radius or a longer excursion resets it.',
  flypass:'Arrival: pass through the capture radius in the forward direction.',land:'Arrival: contact the selected pad and settle. Position follows the pad.',
  takeoff:'Arrival: reach the target inside the capture radius below 0.4 m/s. X/Y follow the preceding point.',descent:'Arrival: reach the target inside the capture radius below 0.4 m/s. X/Y follow the preceding point.'};

export function createMissionPlanner(root,{readInitial,writeInitial,onChange,overlay,readMission,validateMission,readRoute,writeRoute,validateRoute,applyProfile,applyEnvironment,api,settings=()=>({}),writeSettings,isConvex=()=>true,readDisturbances=()=>null,editStart}){
  let waypoints=[],pads=[{name:'Home pad',position:[0,0,0]}],selected=-1,selectedHandle=null,drag=null,view3d,contextEditor=null,samples=[],sampleFilter='all',undo=null;
  root.innerHTML=`<header class="section-head"><span class="section-index">01</span><div class="section-title"><h2>Route</h2><p>Start from a sample or build a sequence of steps. Select a step in the list, the 3D scene or the altitude profile to edit it.</p></div><div id="routeSummary" class="summary-chips" aria-live="polite"></div></header>
    <details class="sample-gallery" open><summary><span class="fold-title">Sample flight plans</span><span id="sampleCount" class="fold-note"></span></summary>
      <div class="gallery-toolbar"><div class="segmented" role="group" aria-label="Filter sample plans">${[['all','All'],['hop','Hops'],['land','Landings'],['hover','Hover']].map(([key,name])=>`<button type="button" data-sample-filter="${key}" aria-pressed="${key==='all'}">${name}</button>`).join('')}</div><span class="hint">Loading a sample replaces only the route: start state, pads and steps. Its suggested guidance profile and environment are one click each. Undo restores your previous route.</span></div>
      <div id="sampleCards" class="sample-cards" role="list"></div></details>
    <div class="library-bar"><div class="library-group"><label for="savedPlan">My plans</label><select id="savedPlan"><option value="">Saved plans…</option></select><button type="button" id="openSavedPlan">Open</button><button type="button" id="savePlan" title="Save under the mission name; the same name replaces it">Save</button></div><div class="library-group"><button type="button" id="loadPlan">Import JSON</button><button type="button" id="exportPlan">Export JSON</button><input id="planFile" type="file" accept=".json,application/json" hidden></div></div>
    <div class="plan-message" role="status"><span id="planMessage"></span><span id="planSuggestions" class="plan-suggestions"></span><button type="button" id="undoPlan" hidden>Undo</button></div>
    <div class="route-workspace"><div class="route-scene"><div id="planner3D" class="planner-3d"></div>
    <figure class="altitude-profile"><figcaption><b>ALTITUDE PROFILE</b><span>Distance flown along the drawn route · click a marker to select its step</span></figcaption><svg id="altitudeProfile" role="img" aria-label="Altitude against distance along the route"></svg></figure>
    <div class="hint route-check" id="plannerCheck"></div>
    <details class="precision-views"><summary><span class="fold-title">Precision views & start orientation</span><span class="fold-note">Top and side projections with draggable start velocity</span></summary><div class="planner-tools"><label>View range <select id="plannerRange"><option>10</option><option>25</option><option>50</option><option selected>100</option></select> m</label><button type="button" id="invertStart">Invert start</button></div><div class="planner-views"><div><b>TOP · X / Y</b><canvas id="planXY" aria-label="Drag start position and waypoints in X Y; drag the arrow to set initial velocity"></canvas></div><div><b>SIDE · X / Z</b><canvas id="planXZ" aria-label="Drag start height and waypoint altitude; drag the arrow to set vertical velocity"></canvas></div></div></details></div>
    <aside class="route-sequence" aria-label="Flight sequence"><div class="sequence-heading"><h3>Flight sequence</h3><span id="stepCount"></span></div>
      <div class="step-palette" role="group" aria-label="Add a step">${Object.entries(waypointTypes).map(([type,name])=>`<button type="button" data-add="${type}" style="--step-color:${waypointColors[type]}" title="${stepHelp[type]}"><i></i>${name}</button>`).join('')}</div>
      <div id="waypointEditor"></div>
      <label class="route-corridor"><span>Default corridor half-width <small>m · steps without their own width</small></span><input id="defaultCorridor" aria-label="Default CORRIDOR" title="Used by steps with a blank corridor" type="number" min=".2" max="25" step="any" required></label>
      <p class="hint">Capture radius decides when a step is complete; corridor width bounds the planned path around the drawn route.</p></aside></div>
    <details class="route-pads"><summary><span class="fold-title">Landing pads</span><span class="fold-note">Ground targets · up to 4 · at least 3 m apart</span></summary><div id="padEditor"></div><button type="button" id="addPad">+ Landing pad</button></details>
    <details class="route-help"><summary><span class="fold-title">How steps complete & corridor rules</span></summary><div class="hint" id="plannerHint">Takeoff and descent finish inside the capture radius below 0.4 m/s. Hover requires the full hold inside that radius below 0.4 m/s; a speed excursion of up to 1 s pauses the timer, leaving the radius or a longer excursion resets it. Fly-through captures while moving forward through the radius. Landing requires physical contact and settling. Soft corridors penalize excess and permit emergency fallback; strict corridors reject infeasible plans. Actual tracking can deviate.</div></details>`;
  const range=root.querySelector('#plannerRange'),list=root.querySelector('#waypointEditor'),planMessage=root.querySelector('#planMessage'),undoButton=root.querySelector('#undoPlan');
  const canvases=[root.querySelector('#planXY'),root.querySelector('#planXZ')];
  const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
  const corridor=()=>settings().guidance?.route_corridor_m??1;
  // Messages may offer undo and one-click suggestions (a sample's guidance and environment).
  const say=(text,{offerUndo=false,suggestions=[]}={})=>{planMessage.textContent=text;undoButton.hidden=!offerUndo;if(!offerUndo)undo=null;
    const box=root.querySelector('#planSuggestions');box.replaceChildren(...suggestions.map(({label,run})=>{const b=document.createElement('button');b.type='button';b.textContent=label;b.onclick=()=>{run();b.disabled=true;b.textContent='✓ '+label.replace(/^Apply /,'Applied ');};return b;}));};
  const defaultCorridor=root.querySelector('#defaultCorridor');
  defaultCorridor.onchange=()=>{
    if(!defaultCorridor.reportValidity())return;
    const value=settings();writeSettings?.({...value,guidance:{...value.guidance,route_corridor_m:Number(defaultCorridor.value)}});changed();
  };
  const speedLimit=()=>isConvex()?(settings().guidance?.max_speed_m_s??4):15;
  const landingPad=()=>pads[waypoints.find(w=>w.type==='land')?.pad??0]??pads[0];
  function alignVerticalColumns(){let previous=readInitial().position;waypoints.forEach((w,i)=>{if(w.type==='land')w.position=[...pads[w.pad??0].position];if(['takeoff','descent'].includes(w.type)){w.position[0]=previous[0];w.position[1]=previous[1];}syncRow(i);previous=w.position;});}
  function geometry(canvas,axis){
    const r=Number(range.value),w=canvas.clientWidth,h=canvas.clientHeight,p=25;
    return {w,h,toScreen:v=>[p+(v[0]+r)/(2*r)*(w-2*p),p+(axis===1?(r-v[1])/(2*r):1-v[2]/r)*(h-2*p)],
      toWorld:(x,y)=>[(x-p)/(w-2*p)*2*r-r,axis===1?r-(y-p)/(h-2*p)*2*r:(1-(y-p)/(h-2*p))*r]};
  }
  function legs(){return sampleRouteLegs(readInitial().position,waypoints,landingPad().position,{convex:isConvex()});}
  // Altitude against distance flown along the drawn route; markers select steps.
  function drawProfile(){
    const svg=root.querySelector('#altitudeProfile'),width=Math.max(320,svg.clientWidth||640),height=132,pad={l:38,r:14,t:14,b:24};
    const profile=routeProfile(legs()),floor=isConvex()?settings().guidance?.route_floor_m:null;
    const total=Math.max(1,profile.distance),top=Math.max(2,...profile.points.map(p=>p[1]))*1.12;
    const x=d=>pad.l+d/total*(width-pad.l-pad.r),y=z=>height-pad.b-z/top*(height-pad.t-pad.b);
    const ticks=[0,.5,1].map(f=>Math.round(top*f/1.12*10)/10);
    const line=profile.points.map(([d,z],i)=>`${i?'L':'M'}${x(d).toFixed(1)} ${y(z).toFixed(1)}`).join(' ');
    const area=`${line} L${x(profile.distance).toFixed(1)} ${y(0)} L${x(0)} ${y(0)} Z`;
    const markers=profile.ends.map((end,i)=>{const wp=waypoints.filter(w=>w.type!=='land')[i]??waypoints.find(w=>w.type==='land');const index=waypoints.indexOf(wp);
      if(!wp)return `<g class="profile-pad"><rect x="${x(end[0])-9}" y="${y(0)-3}" width="18" height="4"/></g>`;
      const color=waypointColors[wp.type],sel=index===selected;
      return `<g class="profile-step${sel?' selected':''}" data-profile-step="${index}" tabindex="0" role="button" aria-label="Select step ${index+1}"><line x1="${x(end[0])}" x2="${x(end[0])}" y1="${y(end[1])}" y2="${y(0)}" stroke="${color}"/><circle cx="${x(end[0])}" cy="${y(end[1])}" r="${sel?9:7.5}" fill="${color}"/><text x="${x(end[0])}" y="${y(end[1])+3.5}">${index+1}</text></g>`;}).join('');
    const start=readInitial().position;
    svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
    svg.innerHTML=`${ticks.map(t=>`<line class="profile-grid" x1="${pad.l}" x2="${width-pad.r}" y1="${y(t)}" y2="${y(t)}"/><text class="profile-axis" x="${pad.l-6}" y="${y(t)+3}" text-anchor="end">${t}</text>`).join('')}
      ${floor!=null?`<line class="profile-floor" x1="${pad.l}" x2="${width-pad.r}" y1="${y(floor)}" y2="${y(floor)}"/><text class="profile-axis" x="${width-pad.r}" y="${y(floor)-4}" text-anchor="end">route floor ${floor} m</text>`:''}
      <path class="profile-area" d="${area}"/><path class="profile-line" d="${line}"/><line class="profile-ground" x1="${pad.l}" x2="${width-pad.r}" y1="${y(0)}" y2="${y(0)}"/>
      <text class="profile-axis" x="${width-pad.r}" y="${height-6}" text-anchor="end">${profile.distance.toFixed(1)} m</text><text class="profile-axis" x="${pad.l}" y="${height-6}">0 m</text>
      <path class="profile-start" d="M${x(0)} ${y(start[2])-7} l6 7 l-6 7 l-6 -7 Z"/>${markers}`;
    svg.querySelectorAll('[data-profile-step]').forEach(el=>{const pick=()=>selectStep(Number(el.dataset.profileStep));el.onclick=pick;el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();pick();}};});
    return profile;
  }
  function draw(){
    defaultCorridor.disabled=!isConvex()||!writeSettings;
    if(document.activeElement!==defaultCorridor)defaultCorridor.value=corridor();
    alignVerticalColumns();
    const profile=drawProfile();
    updateStepSummaries(profile);
    view3d?.draw();
    const initial=readInitial(),check=overlay?.(waypoints);
    // Braking estimate (braking.js): full thrust from the start velocity; the dashed line ends where it stops.
    root.querySelector('#plannerCheck').textContent=check?.braking?`Full-thrust braking from the start velocity: ~${check.braking.distance.toFixed(0)} m in ${check.braking.time.toFixed(1)} s (dashed line to × in the precision views).`:'';
    if(!root.querySelector('.precision-views').open)return;
    canvases.forEach((canvas,k)=>{
      const axis=k===0?1:2,{w,h,toScreen}=geometry(canvas,axis);if(!w||!h)return; // hidden page
      const ctx=fitCanvas(canvas,w,h);ctx.fillStyle='#030507';ctx.fillRect(0,0,w,h);
      ctx.font='9px Consolas';ctx.strokeStyle='rgba(255,255,255,.08)';ctx.fillStyle='#7c878f';
      const r=Number(range.value);
      for(let i=-4;i<=4;i++){const v=i*r/4;let [x]=toScreen([v,0,0]);ctx.beginPath();ctx.moveTo(x,25);ctx.lineTo(x,h-25);ctx.stroke();ctx.fillText(String(v),x-8,h-8);}
      for(let i=0;i<=4;i++){const v=axis===1?-r+i*r/2:i*r/4;const p=[0,0,0];p[axis]=v;const [,y]=toScreen(p);ctx.beginPath();ctx.moveTo(25,y);ctx.lineTo(w-25,y);ctx.stroke();ctx.fillText(String(v),2,y+3);}
      legs().forEach((path,i)=>{ctx.strokeStyle='#3987e5';ctx.lineWidth=1.5;ctx.beginPath();path.forEach((p,j)=>{const [x,y]=toScreen(p);j?ctx.lineTo(x,y):ctx.moveTo(x,y);});if(isConvex()&&waypoints.length){ctx.save();ctx.globalAlpha=.15;ctx.lineWidth=2*(waypoints[i]?.corridor_m??corridor())*(w-50)/(2*r);ctx.stroke();ctx.restore();}ctx.stroke();});
      pads.forEach(p=>{const [px,py]=toScreen(p.position);ctx.strokeStyle='#48dba2';ctx.strokeRect(px-5,py-3,10,6);ctx.fillText(p.name,px+8,py+12);});
      const start=toScreen(initial.position),end=toScreen(initial.position.map((v,i)=>v+initial.velocity[i]*2));
      ctx.strokeStyle='#d95926';ctx.beginPath();ctx.moveTo(...start);ctx.lineTo(...end);ctx.stroke();
      if(check?.stop){const stop=toScreen(check.stop);ctx.setLineDash([2,3]);ctx.beginPath();ctx.moveTo(...start);ctx.lineTo(...stop);ctx.stroke();ctx.setLineDash([]);
        ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(stop[0]-5,stop[1]-5);ctx.lineTo(stop[0]+5,stop[1]+5);ctx.moveTo(stop[0]+5,stop[1]-5);ctx.lineTo(stop[0]-5,stop[1]+5);ctx.stroke();ctx.lineWidth=1.5;}
      if(Math.hypot(end[0]-start[0],end[1]-start[1])>8){ctx.fillStyle='#d95926';ctx.beginPath();ctx.arc(...end,5,0,2*Math.PI);ctx.fill();ctx.fillText('V',end[0]+8,end[1]-5);}
      const occupied=[[start[0]+10,start[1]-19,45,13]];
      waypoints.forEach((wp,i)=>{const [x,y]=toScreen(wp.position);ctx.fillStyle=i===selected?'#fff':waypointColors[wp.type];ctx.beginPath();ctx.arc(x,y,8,0,Math.PI*2);ctx.fill();ctx.fillStyle='#071119';ctx.textAlign='center';ctx.fillText(String(i+1),x,y+3);ctx.textAlign='left';
        const label=waypointLabel(wp,i),width=ctx.measureText(label).width,lx=clamp(x+12,4,Math.max(4,w-width-4));let ly=clamp(y-12,12,h-18);
        for(let attempt=0;attempt<30;attempt++){const offset=Math.ceil(attempt/2)*14*(attempt%2?-1:1),candidate=clamp(y-12+offset,12,h-18);if(!occupied.some(([a,b,c,d])=>lx<a+c&&lx+width>a&&candidate-10<b+d&&candidate>b)){ly=candidate;break;}}
        occupied.push([lx,ly-10,width,13]);
        ctx.lineWidth=3;ctx.strokeStyle='#030507';ctx.strokeText(label,lx,ly);ctx.fillStyle=waypointColors[wp.type];ctx.fillText(label,lx,ly);ctx.lineWidth=1.5;
      });
      ctx.fillStyle='#f4f6f7';ctx.beginPath();ctx.moveTo(start[0],start[1]-9);ctx.lineTo(start[0]+8,start[1]);ctx.lineTo(start[0],start[1]+9);ctx.lineTo(start[0]-8,start[1]);ctx.closePath();ctx.fill();ctx.fillText('START',start[0]+12,start[1]-8);
    });
  }
  // Pointer moves arrive faster than the display; redraw at most once per frame.
  let pending=0;
  function changed(){if(!pending)pending=requestAnimationFrame(()=>{pending=0;draw();onChange?.();});}
  // Drags update the dragged row's inputs in place instead of rebuilding the editor.
  function syncRow(i){root.querySelectorAll(`[data-index="${i}"] [data-axis]`).forEach(el=>{el.value=waypoints[i].position[Number(el.dataset.axis)];});}
  const stepMeta=wp=>[waypointTypes[wp.type],wp.type==='land'?`on ${escapeHtml(pads[wp.pad??0]?.name??'pad')}`:`Z ${Number(wp.position[2]).toFixed(1)} m`,
    `${speedLabel(wp).toLowerCase()} ${wp.speed_m_s} m/s`,wp.type==='hover'?`hold ${wp.hold_s} s`:''].filter(Boolean).join(' · ');
  function updateStepSummaries(profile=routeProfile(legs())){
    const chips=[[`${waypoints.length}/12`,'steps'],[`${profile.distance.toFixed(1)} m`,'route'],[`${Math.max(readInitial().position[2],...waypoints.map(w=>w.position[2])).toFixed(1)} m`,'max alt'],
      [escapeHtml(landingPad().name),'lands on']];
    root.querySelector('#routeSummary').innerHTML=chips.map(([v,k])=>`<span class="chip"><b>${v}</b>${k}</span>`).join('');
    root.querySelector('#stepCount').textContent=`${waypoints.length} / 12 STEPS`;
    list.querySelectorAll('[data-select-step]').forEach(button=>{
      const i=Number(button.dataset.selectStep),wp=waypoints[i];
      button.innerHTML=`<span class="step-number">${String(i+1).padStart(2,'0')}</span><span class="step-main"><b>${escapeHtml(wp.name||waypointTypes[wp.type])}</b><small>${stepMeta(wp)}</small></span><span class="step-chevron" aria-hidden="true">${i===selected?'▴':'▾'}</span>`;
      button.setAttribute('aria-expanded',String(i===selected));
    });
    const start=readInitial(),startCard=list.querySelector('.step-start small');
    if(startCard)startCard.textContent=`XYZ ${start.position.map(v=>v.toFixed(1)).join(' / ')} m · V ${start.velocity.map(v=>v.toFixed(1)).join(' / ')} m/s`;
    const end=list.querySelector('.step-end small');
    if(end)end.textContent=waypoints.some(w=>w.type==='land')?`${landingPad().name} · X ${landingPad().position[0]} Y ${landingPad().position[1]} m`:`Default 0.15 m/s touchdown on ${pads[0].name}`;
    list.querySelector('.step-start')?.classList.toggle('selected',selectedHandle==='start');
    root.querySelectorAll('[data-add]').forEach(el=>{el.disabled=waypoints.length>=12||(['takeoff','land'].includes(el.dataset.add)&&waypoints.some(w=>w.type===el.dataset.add));});
  }
  function selectStep(i){
    selected=i;selectedHandle=i<0?null:i;
    list.querySelectorAll('[data-index]').forEach(row=>row.classList.toggle('selected',Number(row.dataset.index)===i));
    updateStepSummaries();drawProfile();view3d?.draw();
    if(i>=0)list.querySelector(`[data-index="${i}"]`)?.scrollIntoView({block:'nearest'});
  }
  list.addEventListener('invalid',e=>{const row=e.target.closest('[data-index]');if(row)selectStep(Number(row.dataset.index));},true);
  root.addEventListener('invalid',e=>{for(let parent=e.target.parentElement;parent&&parent!==root;parent=parent.parentElement)if(parent.tagName==='DETAILS')parent.open=true;},true);
  const field=(label,input,extra='')=>`<label class="step-field ${extra}"><span>${label}</span>${input}</label>`;
  function stepFields(wp,i){
    const vertical=['takeoff','descent'].includes(wp.type),landing=wp.type==='land';
    const axes=['X','Y','Z'].map((name,a)=>field(`${name}`,`<input aria-label="Waypoint ${i+1} ${name}" data-axis="${a}" type="number" min="${a===2?1:-100}" max="100" step="any" value="${wp.position[a]}" ${landing||a<2&&vertical?'disabled':''}>`)).join('');
    const typeOptions=Object.entries(waypointTypes).map(([type,name])=>`<option value="${type}" ${wp.type===type?'selected':''} ${((type==='takeoff'&&i!==0)||(type==='land'&&i!==waypoints.length-1))?'disabled':''}>${name}</option>`).join('');
    return `<fieldset class="step-group"><legend>Step</legend>${field('Name',`<input data-key="name" aria-label="Waypoint ${i+1} name" maxlength="40" value="${escapeHtml(wp.name??'')}" placeholder="${waypointTypes[wp.type]}">`,'wide')}${field('Type',`<select data-key="type" aria-label="Waypoint ${i+1} type">${typeOptions}</select>`)}</fieldset>
      <fieldset class="step-group"><legend>Position · m${landing?' <em>follows pad</em>':vertical?' <em>vertical: X/Y follow previous point</em>':''}</legend>${axes}</fieldset>
      <fieldset class="step-group"><legend>Arrival</legend>${field(`${speedLabel(wp)[0]+speedLabel(wp).slice(1).toLowerCase()} speed · m/s`,`<input aria-label="Waypoint ${i+1} speed" data-key="speed_m_s" type="number" min=".1" max="${landing?.5:speedLimit()}" step="any" value="${wp.speed_m_s}">`)}${landing?'':field('Capture radius · m',`<input aria-label="Waypoint ${i+1} radius" data-key="radius_m" type="number" min=".1" max="10" step="any" value="${wp.radius_m}">`)}${wp.type==='hover'?field('Hold · s',`<input aria-label="Waypoint ${i+1} hold seconds" data-key="hold_s" type="number" min=".1" max="60" step="any" value="${wp.hold_s}">`):''}</fieldset>
      <fieldset class="step-group"><legend>Path</legend>${landing?field('Pad',`<select data-key="pad" aria-label="Waypoint ${i+1} landing pad">${pads.map((p,j)=>`<option value="${j}" ${(wp.pad??0)===j?'selected':''}>${escapeHtml(p.name)}</option>`).join('')}</select>`)+field('Approach · m/s',`<input aria-label="Landing approach speed" data-key="approach_speed_m_s" type="number" min=".3" max="${speedLimit()}" step="any" placeholder="${speedLimit()} default" value="${wp.approach_speed_m_s??''}">`):''}${field('Corridor · m',`<input aria-label="Waypoint ${i+1} corridor half-width" data-key="corridor_m" type="number" min=".2" max="25" step="any" placeholder="${corridor()} default" value="${wp.corridor_m??''}" ${isConvex()?'':'disabled'}>`)}</fieldset>
      <p class="step-completion">${completion[wp.type]}</p>
      <div class="step-actions"><button type="button" data-up="${i}" aria-label="Move waypoint ${i+1} earlier" ${i===0||wp.type==='land'||waypoints[i-1]?.type==='takeoff'?'disabled':''}>↑ Earlier</button><button type="button" data-down="${i}" aria-label="Move waypoint ${i+1} later" ${i===waypoints.length-1||wp.type==='takeoff'||waypoints[i+1]?.type==='land'?'disabled':''}>↓ Later</button><button type="button" class="danger" data-remove="${i}" aria-label="Remove waypoint ${i+1}">Remove</button></div>`;
  }
  function editList(){
    const rows=waypoints.map((wp,i)=>`<li class="waypoint-row step-card ${i===selected?'selected':''}" data-index="${i}" style="--step-color:${waypointColors[wp.type]}"><button type="button" class="step-select" data-select-step="${i}" aria-controls="step-fields-${i}"></button><div class="step-fields" id="step-fields-${i}">${stepFields(wp,i)}</div></li>`).join('');
    list.innerHTML=`<ol class="step-list"><li class="step-endpoint step-start" style="--step-color:#fff"><span class="endpoint-mark">◆</span><span class="step-main"><b>Start</b><small></small></span><button type="button" class="edit-start">Edit start</button></li>${rows||'<li class="step-empty hint">Direct landing. Add steps above, right-click in the 3D scene, or load a sample plan.</li>'}${waypoints.some(w=>w.type==='land')?'':'<li class="step-endpoint step-end" style="--step-color:#48dba2"><span class="endpoint-mark">▱</span><span class="step-main"><b>Implicit landing</b><small></small></span></li>'}</ol>`;
    list.querySelector('.edit-start').onclick=()=>{selectedHandle='start';selected=-1;view3d?.draw();updateStepSummaries();editStart?.();};
    updateStepSummaries();bindRows(list);refreshContextEditor();
  }
  function bindRows(container){
    container.querySelectorAll('[data-select-step]').forEach(el=>el.onclick=()=>{const i=Number(el.dataset.selectStep);selectStep(selected===i?-1:i);});
    container.querySelectorAll('input[type="number"]').forEach(el=>{el.required=!['corridor_m','approach_speed_m_s'].includes(el.dataset.key);});
    container.querySelectorAll('input,select').forEach(el=>el.onchange=()=>{if(!el.checkValidity()){el.reportValidity();return;}const row=Number(el.closest('[data-index]').dataset.index),wp=waypoints[row];if(el.dataset.axis!==undefined){const a=Number(el.dataset.axis);wp.position[a]=Number(el.value);}else wp[el.dataset.key]=['type','name'].includes(el.dataset.key)?el.value:el.value===''?null:Number(el.value);
      if(el.dataset.key==='type'){
        if(wp.type==='land'){wp.pad=0;wp.position=[...pads[0].position];wp.speed_m_s=.15;}else{delete wp.pad;delete wp.approach_speed_m_s;if(wp.position[2]<1)wp.position[2]=3;}
      }
      selected=row;selectedHandle=row;
      if(['type','pad'].includes(el.dataset.key))editList();
      else{
        root.querySelectorAll('[data-index]').forEach(r=>r.classList.toggle('selected',Number(r.dataset.index)===row));
        const field=el.dataset.axis!==undefined?`[data-axis="${el.dataset.axis}"]`:`[data-key="${el.dataset.key}"]`;
        root.querySelectorAll(`[data-index="${row}"] ${field}`).forEach(input=>{input.value=el.value;});
      }
      changed();});
    container.querySelectorAll('[data-remove]').forEach(el=>el.onclick=()=>{closeContextEditor();waypoints.splice(Number(el.dataset.remove),1);selected=-1;selectedHandle=null;editList();changed();});
    container.querySelectorAll('[data-down]').forEach(el=>el.onclick=()=>{const i=Number(el.dataset.down);closeContextEditor();[waypoints[i],waypoints[i+1]]=[waypoints[i+1],waypoints[i]];selected=i+1;selectedHandle=selected;editList();changed();});
    container.querySelectorAll('[data-up]').forEach(el=>el.onclick=()=>{const i=Number(el.dataset.up);closeContextEditor();[waypoints[i-1],waypoints[i]]=[waypoints[i],waypoints[i-1]];selected=i-1;selectedHandle=selected;editList();changed();});
  }
  function closeContextEditor(){contextEditor?.close();contextEditor=null;}
  function refreshContextEditor(){
    if(!contextEditor||contextEditor.menu.hidden)return;
    const row=list.querySelector(`[data-index="${contextEditor.id}"]`);
    if(!row){closeContextEditor();return;}
    const body=contextEditor.menu.querySelector('.context-fields'),clone=row.cloneNode(true);clone.querySelector('.step-select')?.remove();clone.querySelector('.step-fields')?.removeAttribute('id');
    const source=row.querySelectorAll('input,select');clone.querySelectorAll('input,select').forEach((el,i)=>{el.value=source[i].value;});
    const wrapper=document.createElement('div');wrapper.className='step-card';wrapper.dataset.index=clone.dataset.index;wrapper.style.cssText=clone.style.cssText;wrapper.append(...clone.childNodes);
    body.replaceChildren(wrapper);bindRows(body);
    const wp=waypoints[contextEditor.id];
    contextEditor.menu.querySelector('.context-note').textContent=wp.type==='land'?'Landing position follows its pad. Select the pad marker to move it.':['takeoff','descent'].includes(wp.type)?'Vertical leg: X/Y follow the previous point. Use the Z arrow to change altitude.':'Use the X, Y or Z arrow to move along one axis.';
  }
  function addWaypoint(type,requestedPosition){
    say('');
    if(waypoints.length>=12){say('Maximum 12 waypoints.');return;}
    if(['takeoff','land'].includes(type)&&waypoints.some(w=>w.type===type)){say(`The plan already has a ${waypointTypes[type].toLowerCase()} step.`);return;}
    const index=type==='takeoff'?0:type==='land'?waypoints.length:waypoints.findIndex(w=>w.type==='land')<0?waypoints.length:waypoints.findIndex(w=>w.type==='land');
    const previous=waypoints[index-1]?.position??readInitial().position;
    const position=type==='land'?[...pads[0].position]:['takeoff','descent'].includes(type)?[previous[0],previous[1],type==='takeoff'?Math.min(100,previous[2]+5):Math.max(1,previous[2]-3)]:[Math.round(previous[0]*.6*10)/10,Math.round(previous[1]*.6*10)/10,Math.max(3,Math.round(previous[2]*.75*10)/10)];
    if(requestedPosition&&type!=='land'){
      for(let a=0;a<3;a++)if(a===2||!['takeoff','descent'].includes(type))position[a]=Math.round(clamp(requestedPosition[a],a===2?1:-100,100)*10)/10;
    }
    closeContextEditor();waypoints.splice(index,0,{type,name:'',position,hold_s:2,radius_m:1,speed_m_s:type==='land'?.15:['takeoff','descent'].includes(type)?1:3});selected=index;selectedHandle=index;editList();changed();
  }
  root.querySelectorAll('[data-add]').forEach(el=>el.onclick=()=>addWaypoint(el.dataset.add));
  function firstInvalid(){document.activeElement?.blur();const invalid=root.closest('form')?.querySelector(':invalid');if(invalid){invalid.reportValidity();return true;}return false;}
  // Plans are route-only: export, save, open and import never touch guidance or environment.
  root.querySelector('#exportPlan').onclick=async()=>{try{
    if(firstInvalid())return;
    const route=await validateRoute(readRoute());const blob=new Blob([serializeRoute(route)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='flight-plan.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);say('Route exported. Guidance and environment are not part of a flight plan.');}catch(error){say(error.message);}
  };
  async function refreshSaved(){const plans=await api('/api/flight-plans');root.querySelector('#savedPlan').replaceChildren(new Option('Saved plans…',''),...plans.map(p=>new Option(p.name,p.id)));}
  root.querySelector('#savePlan').onclick=async()=>{try{if(firstInvalid())return;const saved=await api('/api/flight-plans',readRoute());await refreshSaved();root.querySelector('#savedPlan').value=saved.id;say(`Saved the ${saved.name} route on this computer. Saving the same mission name replaces it.`);}catch(e){say(e.message);}};
  // Loading replaces the route (start state, pads, steps); keep the previous route for one undo.
  async function replaceDraft(route,text,suggestions=[]){
    let previous=null;try{previous=readRoute();}catch{}
    writeRoute(route);selected=-1;selectedHandle=null;editList();changed();requestAnimationFrame(()=>view3d?.fit());
    undo=previous;say(text,{offerUndo:!!previous,suggestions});
    // The route stays loaded even if the current guidance rejects it (e.g. a step faster than its speed limit).
    try{await validateMission(readMission());}catch(error){planMessage.textContent=`${text} With the current guidance and setup it will not launch yet: ${error.message}`;}
  }
  undoButton.onclick=()=>{if(!undo)return;const previous=undo;undo=null;writeRoute(previous);editList();changed();requestAnimationFrame(()=>view3d?.fit());say('Previous route restored.');};
  root.querySelector('#openSavedPlan').onclick=async()=>{try{const key=root.querySelector('#savedPlan').value;if(!key){say('Choose a saved plan first.');return;}const plan=await api(`/api/flight-plans/${encodeURIComponent(key)}`);await replaceDraft(plan.mission,`Loaded the ${plan.mission.name} route. Guidance and environment are unchanged.`);}catch(e){say(e.message);}};
  refreshSaved().catch(e=>say(e.message));
  root.querySelector('#loadPlan').onclick=()=>root.querySelector('#planFile').click();
  root.querySelector('#planFile').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;if(file.size>100000)throw new Error('Plan file is too large.');const plan=parseFlightPlan(await file.text());
    const source=plan.mission??{...plan.initial,waypoints:plan.waypoints,pads:[{name:'Home pad',position:[0,0,0]}]};
    const ignored=plan.mission&&['convex_settings','disturbance','disturbance_settings'].some(key=>key in plan.mission);
    const route=await validateRoute(pickRoute(source));await replaceDraft(route,`Flight plan imported.${ignored?' Its guidance and environment settings were not applied; set them in Guidance and Environment.':''}`);}catch(error){say(error.message);}finally{e.target.value='';}};
  // Sample gallery: repository flight plans with their guidance profile and environment.
  function thumbnail(sample){
    const profile=routeProfile(sample.route.slice(1).map((p,i)=>[sample.route[i],p])),w=172,h=64,top=Math.max(2,sample.max_altitude_m)*1.15,total=Math.max(1,profile.distance);
    const x=d=>6+d/total*(w-12),y=z=>h-8-z/top*(h-16),path=profile.points.map(([d,z],i)=>`${i?'L':'M'}${x(d).toFixed(1)} ${y(z).toFixed(1)}`).join(' ');
    return `<svg class="sample-thumb" viewBox="0 0 ${w} ${h}" aria-hidden="true"><line x1="4" x2="${w-4}" y1="${y(0)}" y2="${y(0)}" class="thumb-ground"/><path d="${path} L${x(profile.distance)} ${y(0)} L${x(0)} ${y(0)} Z" class="thumb-area"/><path d="${path}" class="thumb-line"/>${profile.ends.slice(0,-1).map(([d,z])=>`<circle cx="${x(d)}" cy="${y(z)}" r="2.4"/>`).join('')}<path class="thumb-start" d="M${x(0)} ${y(sample.route[0][2])-4} l3.5 4 l-3.5 4 l-3.5 -4 Z"/><rect class="thumb-pad" x="${x(profile.distance)-7}" y="${y(0)-1.5}" width="14" height="3"/></svg>`;
  }
  function drawSamples(){
    const shown=samples.filter(s=>sampleFilter==='all'||s.category===sampleFilter);
    root.querySelector('#sampleCount').textContent=samples.length?`${samples.length} plans · hops, landings and hover`:'';
    root.querySelectorAll('[data-sample-filter]').forEach(el=>el.setAttribute('aria-pressed',String(el.dataset.sampleFilter===sampleFilter)));
    root.querySelector('#sampleCards').innerHTML=shown.map(s=>`<article class="sample-card cat-${s.category}" role="listitem"><div class="sample-tags"><span class="tag tag-${s.category}">${categoryNames[s.category]??s.category}</span>${s.disturbance.length?`<span class="tag tag-env" title="${escapeHtml(s.disturbance.join(', '))}">${s.disturbance.includes('wind')?'WIND':'DISTURBED'}</span>`:''}</div>${thumbnail(s)}<h4>${escapeHtml(s.name)}</h4><p>${escapeHtml(s.summary)}</p><dl><div><dt>Max alt</dt><dd>${s.max_altitude_m.toFixed(0)} m</dd></div><div><dt>Steps</dt><dd>${s.steps}</dd></div><div><dt>Pads</dt><dd>${s.pads}</dd></div></dl><div class="sample-foot"><span title="Suggested guidance profile (applied only if you choose)">Suggests ${escapeHtml(s.profile_name)}</span><button type="button" data-load-sample="${escapeHtml(s.id)}">Load</button></div></article>`).join('')||'<p class="hint">No sample plans are available from this service.</p>';
    root.querySelectorAll('[data-load-sample]').forEach(el=>el.onclick=async()=>{try{const plan=await api(`/api/flight-plan-samples/${encodeURIComponent(el.dataset.loadSample)}`),sample=plan.sample;
      const suggestions=[{label:`Apply suggested guidance: ${sample.profile_name}`,run:()=>applyProfile?.(sample.profile)}];
      if(sample.environment)suggestions.push({label:`Apply suggested environment: ${sample.environment.selected.map(k=>({wind:'wind + gusts',sensor_noise:'sensor noise',com_shift:'COM offset'}[k])).join(', ')}`,run:()=>applyEnvironment?.(sample.environment)});
      await replaceDraft(plan.mission,`Loaded the sample route “${plan.mission.name}”. Guidance and environment are unchanged.`,suggestions);}catch(e){say(e.message);}});
  }
  root.querySelectorAll('[data-sample-filter]').forEach(el=>el.onclick=()=>{sampleFilter=el.dataset.sampleFilter;drawSamples();});
  // An older service without the sample endpoint still offers everything else.
  api('/api/flight-plan-samples').then(value=>{samples=value;drawSamples();}).catch(()=>{samples=[];drawSamples();});
  root.querySelector('#invertStart').onclick=()=>{const initial=readInitial();initial.attitude_deg[0]=Math.abs(initial.attitude_deg[0])>170?0:180;writeInitial(initial);changed();};
  range.onchange=draw;
  canvases.forEach((canvas,k)=>{
    const axis=k===0?1:2;
    canvas.onpointerdown=e=>{
      const rect=canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top,{toScreen}=geometry(canvas,axis),initial=readInitial();
      const handles=[...waypoints.flatMap((w,i)=>w.type==='land'?[]:[{index:i,point:w.position}]),{index:-1,point:initial.position}];
      const velocityEnd=initial.position.map((v,i)=>v+initial.velocity[i]*2);
      const startScreen=toScreen(initial.position),endScreen=toScreen(velocityEnd);
      if(Math.hypot(endScreen[0]-startScreen[0],endScreen[1]-startScreen[1])>8)handles.push({index:-2,point:velocityEnd});
      const hit=handles.reverse().find(h=>{const p=toScreen(h.point);return Math.hypot(p[0]-x,p[1]-y)<15;});
      if(!hit)return;drag={index:hit.index,axis};selected=hit.index;selectedHandle=hit.index===-1?'start':hit.index>=0?hit.index:null;canvas.setPointerCapture(e.pointerId);editList();draw();e.preventDefault();
    };
    canvas.onpointermove=e=>{
      if(!drag||drag.axis!==axis)return;const rect=canvas.getBoundingClientRect(),[x,y]=geometry(canvas,axis).toWorld(e.clientX-rect.left,e.clientY-rect.top),initial=readInitial();
      const values=[clamp(x,-100,100),clamp(y,axis===2?(drag.index>=0?1:.34):-100,100)];
      if(drag.index===-2){initial.velocity[0]=clamp((x-initial.position[0])/2,-20,20);initial.velocity[axis]=clamp((y-initial.position[axis])/2,-20,20);writeInitial(initial);}
      else if(drag.index===-1){initial.position[0]=Math.round(values[0]*10)/10;initial.position[axis]=Math.round(values[1]*10)/10;writeInitial(initial);}
      else{waypoints[drag.index].position[0]=Math.round(values[0]*10)/10;waypoints[drag.index].position[axis]=Math.round(values[1]*10)/10;syncRow(drag.index);}
      changed();
    };
    canvas.onpointerup=()=>{drag=null;};canvas.onpointercancel=()=>{drag=null;};
  });
  function editPads(){
    root.querySelector('#padEditor').innerHTML=pads.map((p,i)=>`<div class="pad-row"><strong>PAD ${i+1}</strong><label>NAME<input data-pad="${i}" data-field="name" aria-label="Pad ${i+1} name" maxlength="24" value="${escapeHtml(p.name)}" required></label>${['X','Y'].map((a,j)=>`<label>${a} / m<input data-pad="${i}" data-field="${j}" aria-label="Pad ${i+1} ${a}" type="number" min="-100" max="100" step="any" value="${p.position[j]}" required></label>`).join('')}<button type="button" data-remove-pad="${i}" ${pads.length===1?'disabled':''}>Remove</button></div>`).join('');
    root.querySelectorAll('[data-pad]').forEach(el=>el.onchange=()=>{if(!el.reportValidity())return;const p=pads[Number(el.dataset.pad)];if(el.dataset.field==='name')p.name=el.value;else p.position[Number(el.dataset.field)]=Number(el.value);editList();changed();});
    root.querySelectorAll('[data-remove-pad]').forEach(el=>el.onclick=()=>{const i=Number(el.dataset.removePad);pads.splice(i,1);waypoints.filter(w=>w.type==='land').forEach(w=>{w.pad=(w.pad??0)===i?0:(w.pad??0)>i?w.pad-1:w.pad??0;});editPads();editList();changed();});
    root.querySelector('#addPad').disabled=pads.length>=4;
    root.querySelector('.route-pads .fold-note').textContent=`${pads.length} of 4 · ground targets at least 3 m apart`;
  }
  root.querySelector('#addPad').onclick=()=>{if(pads.length>=4)return;pads.push({name:`Pad ${pads.length+1}`,position:[Math.min(100,pads.at(-1).position[0]+5),pads.at(-1).position[1],0]});editPads();editList();changed();};
  view3d=createPlanner3D(root.querySelector('#planner3D'),{
    read:()=>({initial:readInitial(),waypoints,pads,selected:selectedHandle,corridor:corridor(),convex:isConvex(),disturbances:readDisturbances()}),
    select:id=>{selectedHandle=id;selected=typeof id==='number'?id:-1;editList();drawProfile();},
    add:(type,position)=>addWaypoint(type,position),
    remove:i=>{closeContextEditor();waypoints.splice(i,1);selected=Math.min(i,waypoints.length-1);selectedHandle=selected<0?null:selected;editList();changed();},
    context:({id,position,menu,close})=>{
      contextEditor=null;
      if(typeof id==='number'){
        menu.innerHTML=`<div class="context-heading">STEP ${id+1} · ${escapeHtml(waypointTypes[waypoints[id].type].toUpperCase())}<button type="button" class="context-close" aria-label="Close waypoint menu">×</button></div><div class="hint context-note"></div><div class="context-fields"></div>`;
        contextEditor={id,menu,close};menu.hidden=false;refreshContextEditor();
      }else if(id==null){
        menu.innerHTML=`<div class="context-heading">ADD STEP HERE<button type="button" class="context-close" aria-label="Close waypoint menu">×</button></div><div class="hint">X ${position[0].toFixed(1)} · Y ${position[1].toFixed(1)} · Z ${position[2].toFixed(1)} m<br>Placed at the selected step's height, else the last step's (3 m on an empty route). Takeoff/descent keep the previous X/Y; landing uses its pad.</div><div class="context-add">${Object.entries(waypointTypes).map(([type,name])=>`<button type="button" data-context-add="${type}" style="--step-color:${waypointColors[type]}" ${waypoints.length>=12||['takeoff','land'].includes(type)&&waypoints.some(w=>w.type===type)?'disabled':''}><i></i>${name}</button>`).join('')}</div>`;
        menu.querySelectorAll('[data-context-add]').forEach(el=>el.onclick=()=>{close();addWaypoint(el.dataset.contextAdd,position);});
      }else{
        menu.innerHTML=`<div class="context-heading">${id==='start'?'START':'LANDING PAD'}<button type="button" class="context-close" aria-label="Close waypoint menu">×</button></div><div class="hint">Use the axis arrows to move this marker. ${id==='start'?'Velocity, attitude and rotor are under Vehicle & launch → Initial state.':'Rename or remove pads under Landing pads.'}</div>`;
      }
      menu.querySelector('.context-close').onclick=()=>{close();contextEditor=null;};
    },
    move:(id,position,axis)=>{const current=id==='start'?readInitial().position:typeof id==='string'?pads[Number(id.split(':')[1])].position:waypoints[id].position;
      position=position.map((v,i)=>i!==axis?current[i]:Math.round(clamp(v,i===2?(id==='start'?.34:1):-100,100)*10)/10);
      if(id==='start'){const initial=readInitial();initial.position=position;writeInitial(initial);}
      else if(typeof id==='string'){const i=Number(id.split(':')[1]);pads[i].position=[position[0],position[1],0];editPads();}
      else{waypoints[id].position=position;syncRow(id);}changed();}
  });
  root.querySelector('.precision-views').addEventListener('toggle',draw);
  new ResizeObserver(()=>{draw();}).observe(root);editPads();editList();draw();
  return {getWaypoints:()=>{alignVerticalColumns();return structuredClone(waypoints);},getPads:()=>structuredClone(pads),setPads:value=>{closeContextEditor();selectedHandle=null;pads=structuredClone(value??[{name:'Home pad',position:[0,0,0]}]);editPads();},refresh:()=>{editList();draw();},setWaypoints:value=>{closeContextEditor();waypoints=normalizeWaypoints(value??[]);selected=waypoints.length?0:-1;selectedHandle=waypoints.length?0:null;editList();changed();requestAnimationFrame(()=>view3d?.fit());},draw,
    describe:()=>({steps:waypoints.length,pad:landingPad().name,text:waypoints.length?`${waypoints.length} step${waypoints.length===1?'':'s'} → ${landingPad().name}`:`Direct landing → ${landingPad().name}`})};
}

export function samplePlannerSpline(start,waypoints,pad=[0,0,0],options={}){
  return sampleRouteLegs(start,waypoints,pad,options).flat();
}
