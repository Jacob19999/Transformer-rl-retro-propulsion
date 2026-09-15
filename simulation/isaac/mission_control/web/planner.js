// Two synchronized projections make XYZ dragging unambiguous on a flat screen.
export function createMissionPlanner(root,{readInitial,writeInitial,onChange}){
  let waypoints=[],selected=-1,drag=null;
  root.innerHTML=`<div class="panel-title">MISSION PLANNER <span>DRAG START, VELOCITY & WAYPOINTS</span></div>
    <div class="planner-tools"><button type="button" data-add="flypass">+ FLY-THROUGH</button><button type="button" data-add="hover">+ HOVER</button><button type="button" id="invertStart">INVERT START</button><label>VIEW RANGE <select id="plannerRange"><option>10</option><option>25</option><option>50</option><option selected>100</option></select> m</label></div>
    <div class="planner-views"><div><b>TOP · X / Y</b><canvas id="planXY" aria-label="Drag start position and waypoints in X Y; drag the arrow to set initial velocity"></canvas></div><div><b>SIDE · X / Z</b><canvas id="planXZ" aria-label="Drag start height and waypoint altitude; drag the arrow to set vertical velocity"></canvas></div></div>
    <div class="hint" id="plannerHint">Cyan diamond: start. Arrow: initial velocity (2-second scale). Numbered points: route. Landing pad: origin. Ground is Z = 0; starts must stay above it.</div><div id="waypointEditor"></div>`;
  const range=root.querySelector('#plannerRange'),list=root.querySelector('#waypointEditor');
  const canvases=[root.querySelector('#planXY'),root.querySelector('#planXZ')];
  const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
  function geometry(canvas,axis){
    const r=Number(range.value),w=canvas.clientWidth,h=canvas.clientHeight,p=25;
    return {w,h,toScreen:v=>[p+(v[0]+r)/(2*r)*(w-2*p),p+(axis===1?(r-v[1])/(2*r):1-v[2]/r)*(h-2*p)],
      toWorld:(x,y)=>[(x-p)/(w-2*p)*2*r-r,axis===1?r-(y-p)/(h-2*p)*2*r:(1-(y-p)/(h-2*p))*r]};
  }
  function draw(){
    const initial=readInitial();
    canvases.forEach((canvas,k)=>{
      const axis=k===0?1:2,{w,h,toScreen}=geometry(canvas,axis),dpr=Math.min(devicePixelRatio,2);
      canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#0d1821';ctx.fillRect(0,0,w,h);
      ctx.font='9px Consolas';ctx.strokeStyle='#263b49';ctx.fillStyle='#728c9b';
      const r=Number(range.value);
      for(let i=-4;i<=4;i++){const v=i*r/4;let [x]=toScreen([v,0,0]);ctx.beginPath();ctx.moveTo(x,25);ctx.lineTo(x,h-25);ctx.stroke();ctx.fillText(String(v),x-8,h-8);}
      for(let i=0;i<=4;i++){const v=axis===1?-r+i*r/2:i*r/4;const p=[0,0,0];p[axis]=v;const [,y]=toScreen(p);ctx.beginPath();ctx.moveTo(25,y);ctx.lineTo(w-25,y);ctx.stroke();ctx.fillText(String(v),2,y+3);}
      const path=samplePlannerSpline(initial.position,waypoints);ctx.strokeStyle='#7a9bdd';ctx.lineWidth=1.5;ctx.setLineDash([4,3]);ctx.beginPath();path.forEach((p,i)=>{const [x,y]=toScreen(p);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();ctx.setLineDash([]);
      const [px,py]=toScreen([0,0,0]);ctx.strokeStyle='#cbd8df';ctx.strokeRect(px-5,py-3,10,6);
      const start=toScreen(initial.position),end=toScreen(initial.position.map((v,i)=>v+initial.velocity[i]*2));
      ctx.strokeStyle='#ffc06a';ctx.beginPath();ctx.moveTo(...start);ctx.lineTo(...end);ctx.stroke();
      if(Math.hypot(end[0]-start[0],end[1]-start[1])>8){ctx.fillStyle='#ffc06a';ctx.beginPath();ctx.arc(...end,5,0,2*Math.PI);ctx.fill();ctx.fillText('V',end[0]+8,end[1]-5);}
      waypoints.forEach((wp,i)=>{const [x,y]=toScreen(wp.position);ctx.fillStyle=i===selected?'#fff':wp.type==='hover'?'#ffc06a':'#85aaff';ctx.beginPath();ctx.arc(x,y,8,0,Math.PI*2);ctx.fill();ctx.fillStyle='#071119';ctx.textAlign='center';ctx.fillText(String(i+1),x,y+3);ctx.textAlign='left';});
      ctx.fillStyle='#7ef5d2';ctx.beginPath();ctx.moveTo(start[0],start[1]-9);ctx.lineTo(start[0]+8,start[1]);ctx.lineTo(start[0],start[1]+9);ctx.lineTo(start[0]-8,start[1]);ctx.closePath();ctx.fill();ctx.fillText('START',start[0]+12,start[1]-8);
    });
  }
  function changed(){draw();onChange?.();}
  function editList(){
    list.innerHTML=waypoints.length?waypoints.map((wp,i)=>`<div class="waypoint-row ${i===selected?'selected':''}" data-index="${i}"><strong>${i+1}</strong><label>TYPE<select data-key="type" aria-label="Waypoint ${i+1} type"><option value="flypass" ${wp.type==='flypass'?'selected':''}>Fly-through</option><option value="hover" ${wp.type==='hover'?'selected':''}>Hover</option></select></label>${['X','Y','Z'].map((name,a)=>`<label>${name} / m<input aria-label="Waypoint ${i+1} ${name}" data-axis="${a}" type="number" min="${a===2?1:-100}" max="100" step=".1" value="${wp.position[a]}"></label>`).join('')}<label>HOLD / s<input aria-label="Waypoint ${i+1} hold seconds" data-key="hold_s" type="number" min=".1" max="60" step=".1" value="${wp.hold_s}" ${wp.type==='flypass'?'disabled':''}></label><label>RADIUS / m<input aria-label="Waypoint ${i+1} radius" data-key="radius_m" type="number" min=".1" max="10" step=".1" value="${wp.radius_m}"></label><label>SPEED / m/s<input aria-label="Waypoint ${i+1} speed" data-key="speed_m_s" type="number" min=".1" max="15" step=".1" value="${wp.speed_m_s}"></label><button type="button" data-up="${i}" aria-label="Move waypoint ${i+1} earlier" ${i===0?'disabled':''}>↑</button><button type="button" data-remove="${i}" aria-label="Remove waypoint ${i+1}">×</button></div>`).join(''):'<div class="hint">Direct landing. Add waypoints to build a mission; every route ends with landing at the origin.</div>';
    list.querySelectorAll('input,select').forEach(el=>el.onchange=()=>{const row=Number(el.closest('[data-index]').dataset.index),wp=waypoints[row];if(el.dataset.axis!==undefined){const a=Number(el.dataset.axis);wp.position[a]=clamp(Number(el.value),a===2?1:-100,100);}else wp[el.dataset.key]=el.dataset.key==='type'?el.value:Number(el.value);selected=row;editList();changed();});
    list.querySelectorAll('[data-remove]').forEach(el=>el.onclick=()=>{waypoints.splice(Number(el.dataset.remove),1);selected=-1;editList();changed();});
    list.querySelectorAll('[data-up]').forEach(el=>el.onclick=()=>{const i=Number(el.dataset.up);[waypoints[i-1],waypoints[i]]=[waypoints[i],waypoints[i-1]];selected=i-1;editList();changed();});
  }
  root.querySelectorAll('[data-add]').forEach(el=>el.onclick=()=>{
    if(waypoints.length>=12)return;
    const previous=waypoints.at(-1)?.position??readInitial().position;
    waypoints.push({type:el.dataset.add,position:[Math.round(previous[0]*.6*10)/10,Math.round(previous[1]*.6*10)/10,Math.max(3,Math.round(previous[2]*.75*10)/10)],hold_s:2,radius_m:1,speed_m_s:3});selected=waypoints.length-1;editList();changed();
  });
  root.querySelector('#invertStart').onclick=()=>{const initial=readInitial();initial.attitude_deg[0]=Math.abs(initial.attitude_deg[0])>170?0:180;writeInitial(initial);changed();};
  range.onchange=draw;
  canvases.forEach((canvas,k)=>{
    const axis=k===0?1:2;
    canvas.onpointerdown=e=>{
      const rect=canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top,{toScreen}=geometry(canvas,axis),initial=readInitial();
      const handles=[...waypoints.map((w,i)=>({index:i,point:w.position})),{index:-1,point:initial.position}];
      const velocityEnd=initial.position.map((v,i)=>v+initial.velocity[i]*2);
      const startScreen=toScreen(initial.position),endScreen=toScreen(velocityEnd);
      if(Math.hypot(endScreen[0]-startScreen[0],endScreen[1]-startScreen[1])>8)handles.push({index:-2,point:velocityEnd});
      const hit=handles.reverse().find(h=>{const p=toScreen(h.point);return Math.hypot(p[0]-x,p[1]-y)<15;});
      if(!hit)return;drag={index:hit.index,axis};selected=hit.index;canvas.setPointerCapture(e.pointerId);editList();draw();e.preventDefault();
    };
    canvas.onpointermove=e=>{
      if(!drag||drag.axis!==axis)return;const rect=canvas.getBoundingClientRect(),[x,y]=geometry(canvas,axis).toWorld(e.clientX-rect.left,e.clientY-rect.top),initial=readInitial();
      const values=[clamp(x,-100,100),clamp(y,axis===2?(drag.index>=0?1:.34):-100,100)];
      if(drag.index===-2){initial.velocity[0]=clamp((x-initial.position[0])/2,-20,20);initial.velocity[axis]=clamp((y-initial.position[axis])/2,-20,20);writeInitial(initial);}
      else if(drag.index===-1){initial.position[0]=Math.round(values[0]*10)/10;initial.position[axis]=Math.round(values[1]*10)/10;writeInitial(initial);}
      else{waypoints[drag.index].position[0]=Math.round(values[0]*10)/10;waypoints[drag.index].position[axis]=Math.round(values[1]*10)/10;editList();}
      changed();
    };
    canvas.onpointerup=()=>{drag=null;};canvas.onpointercancel=()=>{drag=null;};
  });
  new ResizeObserver(draw).observe(root);editList();draw();
  return {getWaypoints:()=>structuredClone(waypoints),setWaypoints:value=>{waypoints=structuredClone(value??[]);selected=-1;editList();changed();},draw};
}

export function samplePlannerSpline(start,waypoints){
  const points=[start,...waypoints.map(w=>w.position),[0,0,.34]],result=[];
  for(let leg=0;leg<points.length-1;leg++){
    const a=points[Math.max(0,leg-1)],b=points[leg],c=points[leg+1],d=points[Math.min(points.length-1,leg+2)];
    for(let j=0;j<=48;j++){const t=j/48;result.push([0,1,2].map(i=>.5*(2*b[i]+(-a[i]+c[i])*t+(2*a[i]-5*b[i]+4*c[i]-d[i])*t*t+(-a[i]+3*b[i]-3*c[i]+d[i])*t*t*t)));}
  }
  return result;
}
