// Shared display and portable plan contract (SI units, world XYZ, Z up).
export const waypointTypes={takeoff:'Takeoff',hover:'Hover',flypass:'Fly-through',descent:'Descent',land:'Landing'};
export const waypointColors={takeoff:'#48dba2',hover:'#e5ad48',flypass:'#6ab7ff',descent:'#cf94ed',land:'#f4f6f7'};
export const speedLabel=w=>({takeoff:'CLIMB',descent:'DESCENT',land:'TOUCHDOWN',flypass:'PASS'}[w.type]??'APPROACH');
export const waypointLabel=(w,i)=>`${i+1} · ${w.name||waypointTypes[w.type]||w.type}`;
export const waypointDetail=w=>w.type==='hover'?`Hold ${w.hold_s} s · approach ${w.speed_m_s} m/s`:`${speedLabel(w).toLowerCase()} ${w.speed_m_s} m/s`;
export const escapeHtml=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export function normalizeWaypoints(value){
  if(!Array.isArray(value)||value.length>12)throw new Error('A plan supports up to 12 waypoints.');
  return value.map(w=>{
    if(!w||!Object.hasOwn(waypointTypes,w.type)||!Array.isArray(w.position)||w.position.length!==3||
      !w.position.every((v,i)=>Number.isFinite(v)&&v>=(i===2?(w.type==='land'?0:1):-100)&&v<=100))throw new Error('Invalid waypoint type or XYZ position.');
    const point={type:w.type,name:w.name??'',position:[...w.position],hold_s:w.hold_s??2,radius_m:w.radius_m??1,speed_m_s:w.speed_m_s??(w.type==='land'?.15:3)};
    if(typeof point.name!=='string'||point.name.length>40)throw new Error('Waypoint names are limited to 40 characters.');
    for(const [key,low,high] of [['hold_s',.1,60],['radius_m',.1,10],['speed_m_s',.1,w.type==='land'?.5:15]]){
      if(!Number.isFinite(point[key])||point[key]<low||point[key]>high)throw new Error(`Invalid waypoint ${key}.`);
    }
    if(w.corridor_m!=null){
      if(!Number.isFinite(w.corridor_m)||w.corridor_m<.2||w.corridor_m>25)throw new Error('Corridor half-width must be 0.2–25 m.');
      point.corridor_m=w.corridor_m;
    }
    if(w.type==='land'){
      if(w.pad!=null){if(!Number.isInteger(w.pad)||w.pad<0||w.pad>3)throw new Error('Invalid landing pad.');point.pad=w.pad;}
      if(w.approach_speed_m_s!=null){
        if(!Number.isFinite(w.approach_speed_m_s)||w.approach_speed_m_s<.3||w.approach_speed_m_s>10)throw new Error('Invalid landing approach speed.');
        point.approach_speed_m_s=w.approach_speed_m_s;
      }
    }
    return point;
  });
}
export function parseFlightPlan(text){
  const plan=JSON.parse(text);
  if(plan.version===2){
    if(plan.format!=='edf-flight-plan'||!plan.mission||typeof plan.mission!=='object'||Array.isArray(plan.mission))throw new Error('Invalid version 2 flight plan.');
    // Full request validation is performed by /api/flight-plan/validate before
    // mutating the draft. Never discard fields on import.
    return {mission:plan.mission};
  }
  if(plan.version!==1||!plan.initial)throw new Error('Expected a version 1 flight plan.');
  for(const [key,low,high] of [['position',-100,100],['velocity',-20,20],['attitude_deg',-180,180],['angular_rate_deg_s',-720,720]]){
    if(!Array.isArray(plan.initial[key])||plan.initial[key].length!==3||!plan.initial[key].every(v=>Number.isFinite(v)&&v>=low&&v<=high))throw new Error(`Invalid initial ${key}.`);
  }
  if(plan.initial.position[2]<.34)throw new Error('Start altitude must be at least 0.34 m.');
  return {initial:plan.initial,waypoints:normalizeWaypoints(plan.waypoints)};
}
export const serializeFlightPlan=mission=>JSON.stringify({format:'edf-flight-plan',version:2,mission},null,2);
// A flight plan is the route only; guidance and environment are chosen per run.
export const routeFields=['name','duration_s','position','velocity','attitude_deg','angular_rate_deg_s','initial_motor_fraction','pads','waypoints'];
export const pickRoute=value=>Object.fromEntries(routeFields.filter(key=>value?.[key]!==undefined).map(key=>[key,structuredClone(value[key])]));
export const serializeRoute=route=>JSON.stringify({format:'edf-flight-plan',version:2,scope:'route',mission:pickRoute(route)},null,2);

export function sampleRouteLegs(start,waypoints,pad=[0,0,0],{convex=false}={}){
  const landing=waypoints.find(w=>w.type==='land');
  const route=waypoints.filter(w=>w.type!=='land');
  const points=[start,...route.map(w=>w.position),landing?.position??pad];
  return points.slice(1).map((_,leg)=>{
    const a=points[Math.max(0,leg-1)],b=points[leg],c=points[leg+1],d=points[Math.min(points.length-1,leg+2)];
    const straight=['takeoff','descent'].includes(route[leg]?.type)||b.every((v,i)=>v===c[i])||points.length===2;
    return Array.from({length:49},(_,j)=>{const t=j/48;return [0,1,2].map(i=>{
      const v=straight?b[i]+(c[i]-b[i])*t:.5*(2*b[i]+(-a[i]+c[i])*t+(2*a[i]-5*b[i]+4*c[i]-d[i])*t*t+(-a[i]+3*b[i]-3*c[i]+d[i])*t*t*t);
      return convex&&i===2?Math.max(v,Math.min(b[2],c[2])):v;
    });});
  });
}

export const categoryNames={hop:'HOP',land:'LANDING',hover:'HOVER'};

// Altitude against 3D distance flown along sampled route legs (the altitude
// profile and sample thumbnails). `ends` holds each leg's final point.
export function routeProfile(legs){
  const points=[],ends=[];let distance=0,previous=null;
  for(const leg of legs){
    for(const p of leg){
      if(previous)distance+=Math.hypot(p[0]-previous[0],p[1]-previous[1],p[2]-previous[2]);
      if(!previous||p!==leg[0])points.push([distance,p[2]]);
      previous=p;
    }
    if(leg.length)ends.push([distance,leg.at(-1)[2]]);
  }
  return {points,ends,distance};
}

// Mirror the classical mission sequencer's stop gates using recorded physical
// states, not the noisy controller observation or distance to the route curve.
export function convexCaptureStatus(frame){
  const mission=frame?.mission,wp=mission?.waypoints?.[mission.waypoint_index];
  if(!frame?.guidance||mission?.ready_to_land||!['hover','takeoff','descent'].includes(wp?.type))return '';
  if(!frame.position?.every(Number.isFinite)||!frame.velocity?.every(Number.isFinite))return '';
  const distance=Math.hypot(...frame.position.map((v,i)=>v-wp.position[i])),speed=Math.hypot(...frame.velocity);
  const reason=distance>wp.radius_m?'Outside capture radius':speed>.4?'Slowing for capture':wp.type==='hover'?'Hold counting':'Capture conditions met';
  return `${reason} · distance ${distance.toFixed(2)} / ${wp.radius_m.toFixed(2)} m · speed ${speed.toFixed(2)} / 0.40 m/s`;
}
