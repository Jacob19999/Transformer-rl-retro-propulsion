// Two synchronized projections make XYZ dragging unambiguous on a flat screen.
import { fitCanvas } from './instruments.js';
import {waypointTypes,waypointColors,waypointLabel,speedLabel,escapeHtml,normalizeWaypoints,parseFlightPlan,serializeFlightPlan,sampleRouteLegs} from './flight-plan.js';
import {createPlanner3D} from './planner-3d.js';

export function createMissionPlanner(root,{readInitial,writeInitial,onChange,overlay,readMission,writeMission,validateMission,api,settings=()=>({}),writeSettings,isConvex=()=>true,readDisturbances=()=>null}){
  let waypoints=[],pads=[{name:'Home pad',position:[0,0,0]}],selected=-1,selectedHandle=null,drag=null,view3d,contextEditor=null;
  root.innerHTML=`<div class="panel-title">ROUTE PLANNER <span>DRAG START, VELOCITY & WAYPOINTS</span></div>
    <div class="planner-tools">${Object.entries(waypointTypes).map(([type,name])=>`<button type="button" data-add="${type}">+ ${name.toUpperCase()}</button>`).join('')}<button type="button" id="invertStart">INVERT START</button><label>VIEW RANGE <select id="plannerRange"><option>10</option><option>25</option><option>50</option><option selected>100</option></select> m</label><button type="button" id="savePlan">SAVE PLAN</button><button type="button" id="loadPlan">LOAD PLAN</button><input id="planFile" type="file" accept=".json,application/json" hidden></div>
    <div id="planner3D" class="planner-3d"></div>
    <section id="plannerDisturbances" class="planner-disturbances" aria-label="Visual disturbance designer"></section>
    <details><summary>PRECISION VIEWS · TOP & SIDE</summary><div class="planner-views"><div><b>TOP · X / Y</b><canvas id="planXY" aria-label="Drag start position and waypoints in X Y; drag the arrow to set initial velocity"></canvas></div><div><b>SIDE · X / Z</b><canvas id="planXZ" aria-label="Drag start height and waypoint altitude; drag the arrow to set vertical velocity"></canvas></div></div></details>
    <div class="hint" id="plannerHint">Takeoff first; landing last on the selected pad. Speed limits each incoming leg; fly-through also sets arrival speed. Blue tubes show corridor half-widths around the convex reference. Optimizer enforcement: soft penalizes excess and permits emergency fallback; strict rejects infeasible corridor plans (see Flight diagnostics). Actual tracking can deviate. Blank widths use the optimizer default. Pads lie on the ground and must be 3 m apart.</div><div class="hint" id="planMessage" role="status"></div><div class="hint" id="plannerCheck"></div><div id="padEditor"></div><button type="button" id="addPad">+ LANDING PAD</button><div id="waypointEditor"></div>`;
  root.querySelector('#loadPlan').textContent='IMPORT JSON';
  root.querySelector('.planner-tools').insertAdjacentHTML('beforeend','<button type="button" id="exportPlan">EXPORT JSON</button><label>SAVED PLAN<select id="savedPlan"><option value="">Choose saved plan</option></select></label><button type="button" id="openSavedPlan">LOAD SAVED</button>');
  root.querySelector('.planner-tools').insertAdjacentHTML('beforeend','<label>Default CORRIDOR / m<input id="defaultCorridor" aria-label="Default CORRIDOR" title="Default corridor half-width in meters. Used by waypoints with a blank corridor." type="number" min=".2" max="25" step="any" required></label>');
  const range=root.querySelector('#plannerRange'),list=root.querySelector('#waypointEditor');
  const canvases=[root.querySelector('#planXY'),root.querySelector('#planXZ')];
  const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
  const corridor=()=>settings().guidance?.route_corridor_m??1;
  const defaultCorridor=root.querySelector('#defaultCorridor');
  defaultCorridor.onchange=()=>{
    if(!defaultCorridor.reportValidity())return;
    const value=settings();writeSettings?.({...value,guidance:{...value.guidance,route_corridor_m:Number(defaultCorridor.value)}});changed();
  };
  const speedLimit=()=>isConvex()?(settings().guidance?.max_speed_m_s??4):15;
  function alignVerticalColumns(){let previous=readInitial().position;waypoints.forEach((w,i)=>{if(w.type==='land')w.position=[...pads[w.pad??0].position];if(['takeoff','descent'].includes(w.type)){w.position[0]=previous[0];w.position[1]=previous[1];}syncRow(i);previous=w.position;});}
  function geometry(canvas,axis){
    const r=Number(range.value),w=canvas.clientWidth,h=canvas.clientHeight,p=25;
    return {w,h,toScreen:v=>[p+(v[0]+r)/(2*r)*(w-2*p),p+(axis===1?(r-v[1])/(2*r):1-v[2]/r)*(h-2*p)],
      toWorld:(x,y)=>[(x-p)/(w-2*p)*2*r-r,axis===1?r-(y-p)/(h-2*p)*2*r:(1-(y-p)/(h-2*p))*r]};
  }
  function draw(){
    defaultCorridor.disabled=!isConvex()||!writeSettings;
    if(document.activeElement!==defaultCorridor)defaultCorridor.value=corridor();
    alignVerticalColumns();
    view3d?.draw();
    const initial=readInitial(),check=overlay?.(waypoints);
    // Braking estimate (braking.js): full thrust from the start velocity; the dashed line ends where it stops.
    root.querySelector('#plannerCheck').textContent=check?.braking?`Full-thrust braking from the start velocity: ~${check.braking.distance.toFixed(0)} m in ${check.braking.time.toFixed(1)} s (dashed line to ×).`:'';
    canvases.forEach((canvas,k)=>{
      const axis=k===0?1:2,{w,h,toScreen}=geometry(canvas,axis);if(!w||!h)return; // hidden page
      const ctx=fitCanvas(canvas,w,h);ctx.fillStyle='#030507';ctx.fillRect(0,0,w,h);
      ctx.font='9px Consolas';ctx.strokeStyle='rgba(255,255,255,.08)';ctx.fillStyle='#7c878f';
      const r=Number(range.value);
      for(let i=-4;i<=4;i++){const v=i*r/4;let [x]=toScreen([v,0,0]);ctx.beginPath();ctx.moveTo(x,25);ctx.lineTo(x,h-25);ctx.stroke();ctx.fillText(String(v),x-8,h-8);}
      for(let i=0;i<=4;i++){const v=axis===1?-r+i*r/2:i*r/4;const p=[0,0,0];p[axis]=v;const [,y]=toScreen(p);ctx.beginPath();ctx.moveTo(25,y);ctx.lineTo(w-25,y);ctx.stroke();ctx.fillText(String(v),2,y+3);}
      const legs=sampleRouteLegs(initial.position,waypoints,pads[0].position,{convex:isConvex()});
      legs.forEach((path,i)=>{ctx.strokeStyle='#3987e5';ctx.lineWidth=1.5;ctx.beginPath();path.forEach((p,j)=>{const [x,y]=toScreen(p);j?ctx.lineTo(x,y):ctx.moveTo(x,y);});if(isConvex()&&waypoints.length){ctx.save();ctx.globalAlpha=.15;ctx.lineWidth=2*(waypoints[i]?.corridor_m??corridor())*(w-50)/(2*r);ctx.stroke();ctx.restore();}ctx.stroke();});
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
  function editList(){
    list.innerHTML=waypoints.length?waypoints.map((wp,i)=>`<div class="waypoint-row ${i===selected?'selected':''}" data-index="${i}"><strong style="color:${waypointColors[wp.type]}">${i+1}</strong><label>NAME<input data-key="name" aria-label="Waypoint ${i+1} name" maxlength="40" value="${escapeHtml(wp.name??'')}" placeholder="${waypointTypes[wp.type]}"></label><label>TYPE<select data-key="type" aria-label="Waypoint ${i+1} type">${Object.entries(waypointTypes).map(([type,name])=>`<option value="${type}" ${wp.type===type?'selected':''} ${((type==='takeoff'&&i!==0)||(type==='land'&&i!==waypoints.length-1))?'disabled':''}>${name}</option>`).join('')}</select></label>${['X','Y','Z'].map((name,a)=>`<label>${name} / m<input aria-label="Waypoint ${i+1} ${name}" data-axis="${a}" type="number" min="${a===2?1:-100}" max="100" step="any" value="${wp.position[a]}" ${wp.type==='land'||a<2&&['takeoff','descent'].includes(wp.type)?'disabled':''}></label>`).join('')}<label>HOLD / s<input aria-label="Waypoint ${i+1} hold seconds" data-key="hold_s" type="number" min=".1" max="60" step="any" value="${wp.hold_s}" ${wp.type!=='hover'?'disabled':''}></label><label>RADIUS / m<input aria-label="Waypoint ${i+1} radius" data-key="radius_m" type="number" min=".1" max="10" step="any" value="${wp.radius_m}" ${wp.type==='land'?'disabled':''}></label><label>${speedLabel(wp)} / m/s<input aria-label="Waypoint ${i+1} speed" data-key="speed_m_s" type="number" min=".1" max="${wp.type==='land'?.5:speedLimit()}" step="any" value="${wp.speed_m_s}"></label><label>CORRIDOR / m<input aria-label="Waypoint ${i+1} corridor half-width" data-key="corridor_m" type="number" min=".2" max="25" step="any" placeholder="${corridor()} default" value="${wp.corridor_m??''}" ${isConvex()?'':'disabled'}></label>${wp.type==='land'?`<label>PAD<select data-key="pad" aria-label="Waypoint ${i+1} landing pad">${pads.map((p,j)=>`<option value="${j}" ${(wp.pad??0)===j?'selected':''}>${escapeHtml(p.name)}</option>`).join('')}</select></label><label>APPROACH / m/s<input aria-label="Landing approach speed" data-key="approach_speed_m_s" type="number" min=".3" max="${speedLimit()}" step="any" placeholder="${speedLimit()} default" value="${wp.approach_speed_m_s??''}"></label>`:''}<button type="button" data-down="${i}" aria-label="Move waypoint ${i+1} later" ${i===waypoints.length-1||wp.type==='takeoff'||waypoints[i+1]?.type==='land'?'disabled':''}>↓</button><button type="button" data-up="${i}" aria-label="Move waypoint ${i+1} earlier" ${i===0||wp.type==='land'||waypoints[i-1]?.type==='takeoff'?'disabled':''}>↑</button><button type="button" data-remove="${i}" aria-label="Remove waypoint ${i+1}">×</button></div>`).join(''):'<div class="hint">Direct landing. Add waypoints to create a flight plan. An omitted landing step uses the default 0.15 m/s touchdown on the first pad.</div>';
    bindRows(list);refreshContextEditor();
  }
  function bindRows(container){
    container.querySelectorAll('input[type="number"]').forEach(el=>{el.required=!['corridor_m','approach_speed_m_s'].includes(el.dataset.key);});
    container.querySelectorAll('input,select').forEach(el=>el.onchange=()=>{if(!el.checkValidity()){el.reportValidity();return;}const row=Number(el.closest('[data-index]').dataset.index),wp=waypoints[row];if(el.dataset.axis!==undefined){const a=Number(el.dataset.axis);wp.position[a]=Number(el.value);}else wp[el.dataset.key]=['type','name'].includes(el.dataset.key)?el.value:el.value===''?null:Number(el.value);
      if(el.dataset.key==='type'){
        if(wp.type==='land'){wp.pad=0;wp.position=[...pads[0].position];wp.speed_m_s=.15;}else{delete wp.pad;delete wp.approach_speed_m_s;if(wp.position[2]<1)wp.position[2]=3;}
      }
      selected=row;selectedHandle=row;
      if(el.dataset.key==='type')editList();
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
    const body=contextEditor.menu.querySelector('.context-fields'),clone=row.cloneNode(true);
    const source=row.querySelectorAll('input,select');clone.querySelectorAll('input,select').forEach((el,i)=>{el.value=source[i].value;});
    body.replaceChildren(clone);bindRows(body);
    const wp=waypoints[contextEditor.id];
    contextEditor.menu.querySelector('.context-note').textContent=wp.type==='land'?'Landing position follows its pad. Select the pad marker to move it.':['takeoff','descent'].includes(wp.type)?'Vertical leg: X/Y follow the previous point. Use the Z arrow to change altitude.':'Use the X, Y or Z arrow to move along one axis.';
  }
  function addWaypoint(type,requestedPosition){
    const message=root.querySelector('#planMessage');message.textContent='';
    if(waypoints.length>=12){message.textContent='Maximum 12 waypoints.';return;}
    if(['takeoff','land'].includes(type)&&waypoints.some(w=>w.type===type)){message.textContent=`The plan already has a ${waypointTypes[type].toLowerCase()} step.`;return;}
    const index=type==='takeoff'?0:type==='land'?waypoints.length:waypoints.findIndex(w=>w.type==='land')<0?waypoints.length:waypoints.findIndex(w=>w.type==='land');
    const previous=waypoints[index-1]?.position??readInitial().position;
    const position=type==='land'?[...pads[0].position]:['takeoff','descent'].includes(type)?[previous[0],previous[1],type==='takeoff'?Math.min(100,previous[2]+5):Math.max(1,previous[2]-3)]:[Math.round(previous[0]*.6*10)/10,Math.round(previous[1]*.6*10)/10,Math.max(3,Math.round(previous[2]*.75*10)/10)];
    if(requestedPosition&&type!=='land'){
      for(let a=0;a<3;a++)if(a===2||!['takeoff','descent'].includes(type))position[a]=Math.round(clamp(requestedPosition[a],a===2?1:-100,100)*10)/10;
    }
    closeContextEditor();waypoints.splice(index,0,{type,name:'',position,hold_s:2,radius_m:1,speed_m_s:type==='land'?.15:['takeoff','descent'].includes(type)?1:3});selected=index;selectedHandle=index;editList();changed();
  }
  root.querySelectorAll('[data-add]').forEach(el=>el.onclick=()=>addWaypoint(el.dataset.add));
  root.querySelector('#exportPlan').onclick=async()=>{try{
    document.activeElement?.blur();const invalid=root.closest('form')?.querySelector(':invalid');if(invalid){invalid.reportValidity();return;}
    const mission=await validateMission(readMission());const blob=new Blob([serializeFlightPlan(mission)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='flight-plan.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);root.querySelector('#planMessage').textContent='Complete flight plan saved.';}catch(error){root.querySelector('#planMessage').textContent=error.message;}
  };
  async function refreshSaved(){const plans=await api('/api/flight-plans');root.querySelector('#savedPlan').replaceChildren(new Option('Choose saved plan',''),...plans.map(p=>new Option(p.name,p.id)));}
  root.querySelector('#savePlan').onclick=async()=>{try{document.activeElement?.blur();const invalid=root.closest('form')?.querySelector(':invalid');if(invalid){invalid.reportValidity();return;}const saved=await api('/api/flight-plans',readMission());await refreshSaved();root.querySelector('#savedPlan').value=saved.id;root.querySelector('#planMessage').textContent=`Saved ${saved.name} on this computer. Saving the same mission name replaces it.`;}catch(e){root.querySelector('#planMessage').textContent=e.message;}};
  root.querySelector('#openSavedPlan').onclick=async()=>{try{const key=root.querySelector('#savedPlan').value;if(!key)return;const plan=await api(`/api/flight-plans/${encodeURIComponent(key)}`);writeMission(plan.mission);view3d?.fit();root.querySelector('#planMessage').textContent=`Loaded ${plan.mission.name}.`;}catch(e){root.querySelector('#planMessage').textContent=e.message;}};
  refreshSaved().catch(e=>{root.querySelector('#planMessage').textContent=e.message;});
  root.querySelector('#loadPlan').onclick=()=>root.querySelector('#planFile').click();
  root.querySelector('#planFile').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;if(file.size>100000)throw new Error('Plan file is too large.');const plan=parseFlightPlan(await file.text());const request=plan.mission??{...readMission(),...plan.initial,waypoints:plan.waypoints,pads:[{name:'Home pad',position:[0,0,0]}]};const validated=await validateMission(request);writeMission(validated);selected=-1;editList();changed();view3d?.fit();root.querySelector('#planMessage').textContent='Flight plan loaded.';}catch(error){root.querySelector('#planMessage').textContent=error.message;}finally{e.target.value='';}};
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
    root.querySelector('#padEditor').innerHTML=pads.map((p,i)=>`<div class="pad-row"><strong>PAD ${i+1}</strong><label>NAME<input data-pad="${i}" data-field="name" aria-label="Pad ${i+1} name" maxlength="24" value="${escapeHtml(p.name)}" required></label>${['X','Y'].map((a,j)=>`<label>${a} / m<input data-pad="${i}" data-field="${j}" aria-label="Pad ${i+1} ${a}" type="number" min="-100" max="100" step="any" value="${p.position[j]}" required></label>`).join('')}<button type="button" data-remove-pad="${i}" ${pads.length===1?'disabled':''}>REMOVE PAD ${i+1}</button></div>`).join('');
    root.querySelectorAll('[data-pad]').forEach(el=>el.onchange=()=>{if(!el.reportValidity())return;const p=pads[Number(el.dataset.pad)];if(el.dataset.field==='name')p.name=el.value;else p.position[Number(el.dataset.field)]=Number(el.value);editList();changed();});
    root.querySelectorAll('[data-remove-pad]').forEach(el=>el.onclick=()=>{const i=Number(el.dataset.removePad);pads.splice(i,1);waypoints.filter(w=>w.type==='land').forEach(w=>{w.pad=(w.pad??0)===i?0:(w.pad??0)>i?w.pad-1:w.pad??0;});editPads();editList();changed();});
    root.querySelector('#addPad').disabled=pads.length>=4;
  }
  root.querySelector('#addPad').onclick=()=>{if(pads.length>=4)return;pads.push({name:`Pad ${pads.length+1}`,position:[Math.min(100,pads.at(-1).position[0]+5),pads.at(-1).position[1],0]});editPads();editList();changed();};
  view3d=createPlanner3D(root.querySelector('#planner3D'),{
    read:()=>({initial:readInitial(),waypoints,pads,selected:selectedHandle,corridor:corridor(),convex:isConvex(),disturbances:readDisturbances()}),
    select:id=>{selectedHandle=id;selected=typeof id==='number'?id:-1;editList();},
    context:({id,position,menu,close})=>{
      contextEditor=null;
      if(typeof id==='number'){
        menu.innerHTML=`<div class="context-heading">WAYPOINT ${id+1}<button type="button" class="context-close" aria-label="Close waypoint menu">×</button></div><div class="hint context-note"></div><div class="context-fields"></div>`;
        contextEditor={id,menu,close};menu.hidden=false;refreshContextEditor();
      }else if(id==null){
        menu.innerHTML=`<div class="context-heading">ADD WAYPOINT<button type="button" class="context-close" aria-label="Close waypoint menu">×</button></div><div class="hint">X ${position[0].toFixed(1)} · Y ${position[1].toFixed(1)} · Z ${position[2].toFixed(1)} m<br>Placed at the selected waypoint height, or 3 m. Takeoff/descent keep the previous X/Y; landing uses its pad.</div><div class="context-add">${Object.entries(waypointTypes).map(([type,name])=>`<button type="button" data-context-add="${type}" ${waypoints.length>=12||['takeoff','land'].includes(type)&&waypoints.some(w=>w.type===type)?'disabled':''}>+ ${name}</button>`).join('')}</div>`;
        menu.querySelectorAll('[data-context-add]').forEach(el=>el.onclick=()=>{close();addWaypoint(el.dataset.contextAdd,position);});
      }else{
        menu.innerHTML='<div class="context-heading">POSITION MARKER<button type="button" class="context-close" aria-label="Close waypoint menu">×</button></div><div class="hint">Use the axis arrows to move this marker. Parameters are available in the start / landing pad fields.</div>';
      }
      menu.querySelector('.context-close').onclick=()=>{close();contextEditor=null;};
    },
    move:(id,position,axis)=>{const current=id==='start'?readInitial().position:typeof id==='string'?pads[Number(id.split(':')[1])].position:waypoints[id].position;
      position=position.map((v,i)=>i!==axis?current[i]:Math.round(clamp(v,i===2?(id==='start'?.34:1):-100,100)*10)/10);
      if(id==='start'){const initial=readInitial();initial.position=position;writeInitial(initial);}
      else if(typeof id==='string'){const i=Number(id.split(':')[1]);pads[i].position=[position[0],position[1],0];editPads();}
      else{waypoints[id].position=position;syncRow(id);}changed();}
  });
  root.querySelector('details').addEventListener('toggle',draw);
  new ResizeObserver(draw).observe(root);editPads();editList();draw();
  return {getWaypoints:()=>{alignVerticalColumns();return structuredClone(waypoints);},getPads:()=>structuredClone(pads),setPads:value=>{closeContextEditor();selectedHandle=null;pads=structuredClone(value??[{name:'Home pad',position:[0,0,0]}]);editPads();},refresh:()=>{editList();draw();},setWaypoints:value=>{closeContextEditor();waypoints=normalizeWaypoints(value??[]);selected=-1;selectedHandle=null;editList();changed();requestAnimationFrame(()=>view3d?.fit());},draw};
}

export function samplePlannerSpline(start,waypoints,pad=[0,0,0],options={}){
  return sampleRouteLegs(start,waypoints,pad,options).flat();
}
