import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createMissionPlanner, samplePlannerSpline } from './planner.js';

const $ = id => document.getElementById(id);
const deg = 180 / Math.PI;
const finNames = ['FWD', 'RIGHT', 'AFT', 'LEFT'];
const linkNames = ['FwdFin', 'RightFin', 'AftFin', 'LeftFin'];
const colors = ['#7ef5d2', '#72b9fa', '#e7bc79'];
const state = { id: null, frames: [], metadata: null, time: 0, playing: false, live: true, busy: false, recording: false, result: null, training:false };
let config, previousPosition = new THREE.Vector3(), orbitInitialized = false;
const fmt = (n, d = 1) => Number.isFinite(n) ? n.toFixed(d) : '—';
const clock = t => `T+ ${String(Math.floor(t / 60)).padStart(2, '0')}:${(t % 60).toFixed(2).padStart(5, '0')}`;
const text = (id, value) => { $(id).textContent = value; };
function message(value, error = false) { text('runMessage', value); $('runMessage').style.color = error ? '#ffa78e' : ''; }
function updateTraining(c){
  state.training=!!c.training;text('connection',state.training?'PPO TRAINING / REPLAY READY':'ISAAC SERVICE ONLINE');
  const m=c.training_metrics;$('trainingStatus').hidden=!state.training;
  text('trainingStatus',m?.step!=null?`PPO TRAINING · ${fmt(m.step/1e6,2)}M transitions · Spawn stage ${m.stage+1}/${m.stages} · Recent stage success ${fmt(m.stage_success*100,1)}% · Full-height evaluation ${m.full_success==null?'pending':fmt(m.full_success*100,1)+'% success at '+fmt(m.full_eval_step/1e6,2)+'M'}${m.success_energy_wh==null?'':' · Successful full-height landings: '+fmt(m.success_energy_wh,2)+' Wh / '+fmt(m.success_delta_v,1)+' m/s Δv'}`:'PPO TRAINING · Starting simulator and restoring checkpoint');
  $('run').disabled=state.busy||state.recording||state.training;
}
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Mission-Control': 'local' }, body: JSON.stringify(body) });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
  return data;
}

for (const [key, title, labels, initial, min, max] of [
  ['position', 'POSITION / m', ['X', 'Y', 'Z'], [-.28, .82, 18], [-100,-100,.34], [100,100,100]],
  ['velocity', 'VELOCITY / m/s', ['VX','VY','VZ'], [0,0,-1], [-20,-20,-20], [20,20,20]],
  ['attitude_deg', 'ATTITUDE / degrees', ['ROLL','PITCH','YAW'], [0,0,0], [-180,-180,-180], [180,180,180]],
  ['angular_rate_deg_s', 'BODY RATE / degrees/s · FRD', ['P','Q','R'], [0,0,0], [-720,-720,-720], [720,720,720]],
]) {
  const div = document.createElement('div');
  div.innerHTML = `<div class="vector-label">${title}</div><div class="triple">${labels.map((label,i)=>`<label>${label}<input aria-label="${title} ${label}" id="${key}_${i}" type="number" step="any" min="${min[i]}" max="${max[i]}" value="${initial[i]}" required></label>`).join('')}</div>`;
  $('vectors').append(div);
}
const initialKeys=['position','velocity','attitude_deg','angular_rate_deg_s'];
const readPlannerInitial=()=>Object.fromEntries(initialKeys.map(key=>[key,[0,1,2].map(i=>Number($(`${key}_${i}`).value))]));
const planner=createMissionPlanner($('missionPlanner'),{readInitial:readPlannerInitial,
  writeInitial:initial=>initialKeys.forEach(key=>initial[key].forEach((v,i)=>{$(`${key}_${i}`).value=Math.round(v*100)/100;})),
  onChange:()=>updatePlannedRoute()});
$('vectors').addEventListener('input',()=>{planner.draw();updatePlannedRoute();});
$('finRows').innerHTML = finNames.map((n,i)=>`<tr><td>${n}</td><td id="fc${i}">—</td><td id="fa${i}">—</td></tr>`).join('');
$('finRateRows').innerHTML = finNames.map((n,i)=>`<tr><td>${n}</td><td id="fcr${i}">—</td><td id="far${i}">—</td></tr>`).join('');
$('gyro').innerHTML = ['P / ROLL','Q / PITCH','R / YAW'].map((n,i)=>`<div><span>${n}</span><b id="g${i}">—</b></div>`).join('');
const rotationFields=[['peak_rate_deg_s','PEAK °/s',1],['angular_travel_deg','TRAVEL °',0],['excess_rotation_deg','EXCESS °',0],['time_above_limit_s','OVER / s',2]];
$('rotationRows').innerHTML=rotationFields.map(([key,title])=>`<tr><td>${title}</td>${[0,1,2].map(i=>`<td id="rotation_${key}_${i}">—</td>`).join('')}</tr>`).join('');
const batteryFields = [['voltage_v','BUS VOLTAGE','V',2],['current_a','CURRENT','A',1],['power_w','POWER','W',0],['energy_wh','USED ENERGY','Wh',2],['temperature_c','PACK TEMP','°C',1],['ocv_v','OPEN CIRCUIT','V',2]];
$('batteryMetrics').innerHTML = batteryFields.map(([key,name,unit])=>`<div><span>${name}</span><b id="b_${key}">—</b><small>${unit}</small></div>`).join('');
$('batteryMetrics').insertAdjacentHTML('beforeend','<div><span>PROPULSIVE ΔV</span><b id="propulsiveDv">—</b><small>m/s</small></div>');

const renderer = new THREE.WebGLRenderer({ canvas: $('scene'), antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
renderer.setClearColor('#101c27');
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.45;
const scene = new THREE.Scene();
scene.background = new THREE.Color('#14212d');
scene.fog = new THREE.Fog('#14212d', 180, 600);
scene.add(new THREE.HemisphereLight(0xc9e7ff, 0x354b4e, 2.8));
const sun = new THREE.DirectionalLight(0xe6f4ff, 3.2); sun.position.set(4,-6,12); scene.add(sun);
const fill = new THREE.DirectionalLight(0x6db9bd, 2); fill.position.set(-3,4,3); scene.add(fill);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(300,300), new THREE.MeshStandardMaterial({color:0x111b22, roughness:.98}));
floor.position.z = -.006; scene.add(floor);
const grid = new THREE.GridHelper(100,100,0x31515b,0x253840); grid.rotation.x = Math.PI/2; grid.position.z=.001; scene.add(grid);
const pad = new THREE.Mesh(new THREE.CircleGeometry(1.25,80), new THREE.MeshStandardMaterial({color:0x26373d,roughness:.95})); pad.position.z=.004;scene.add(pad);
for (const radius of [.5, 1.15]) { const ring = new THREE.Mesh(new THREE.RingGeometry(radius-.012,radius,96),new THREE.MeshBasicMaterial({color:radius<1?0x7ef5d2:0x839ca5,side:THREE.DoubleSide}));ring.position.z=.007;scene.add(ring); }
for (const angle of [0,Math.PI/2]) { const line = new THREE.Mesh(new THREE.PlaneGeometry(.55,.018),new THREE.MeshBasicMaterial({color:0xc6d8d9})); line.rotation.z=angle;line.position.z=.008;scene.add(line); }
const groundObjects = scene.children.filter(object=>object.isMesh||object===grid);
const trajectory = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({color:0x498785,transparent:true,opacity:.6})); scene.add(trajectory);
const plannedRoute=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0x85aaff,transparent:true,opacity:.65}));scene.add(plannedRoute);
function updatePlannedRoute(){plannedRoute.geometry.dispose();plannedRoute.geometry=new THREE.BufferGeometry().setFromPoints(samplePlannerSpline(readPlannerInitial().position,planner.getWaypoints()).map(v=>new THREE.Vector3(...v)));}
const thrustVector = new THREE.ArrowHelper(new THREE.Vector3(0,0,1),new THREE.Vector3(),.4,0x7ef5d2,.05,.025);scene.add(thrustVector);
const cameras = Array.from({length:4},()=> {const c = new THREE.PerspectiveCamera(40,1,.008,300);c.up.set(0,0,1);return c;});
cameras[3] = new THREE.OrthographicCamera(-.18,.18,.10,-.10,.001,5);
const finLabels = document.createElement('canvas');finLabels.id='finLabels';$('views').append(finLabels);
document.querySelector('.fin-tag b').textContent='FINS / BOTTOM';
const controls = new OrbitControls(cameras[0],$('scene')); controls.enableDamping = true; controls.dampingFactor=.1;controls.minDistance=.3;controls.maxDistance=40;controls.enablePan=false;
const model = await new GLTFLoader().loadAsync('/static/drone.glb');
const geometry = await (await fetch('/static/geometry.json')).json();
const links = {};
model.scene.traverse(object => {if (object.isMesh) {object.material = new THREE.MeshStandardMaterial({color:object.name==='Body'?0xadb8bc:0x7ef5d2,metalness:.38,roughness:.46,side:THREE.DoubleSide}); links[object.name]=object;}});
scene.add(model.scene);
for (const link of geometry.links) { const obj=links[link.name]; if(obj){obj.position.fromArray(link.neutral_position);obj.quaternion.set(link.neutral_quaternion[1],link.neutral_quaternion[2],link.neutral_quaternion[3],link.neutral_quaternion[0]);} }

function pose(obj, aPos, aQuat, bPos, bQuat, alpha) {
  obj.position.fromArray(aPos).lerp(new THREE.Vector3().fromArray(bPos),alpha);
  obj.quaternion.set(aQuat[1],aQuat[2],aQuat[3],aQuat[0]).slerp(new THREE.Quaternion(bQuat[1],bQuat[2],bQuat[3],bQuat[0]),alpha);
}
function sampleAt(t) {
  const frames=state.frames;if(!frames.length)return null;
  let lo=0,hi=frames.length-1;
  while(lo<hi){const m=Math.ceil((lo+hi)/2);if(frames[m].t<=t)lo=m;else hi=m-1;}
  const a=frames[lo],b=frames[Math.min(lo+1,frames.length-1)];
  return {a,b,alpha:a===b?0:THREE.MathUtils.clamp((t-a.t)/(b.t-a.t),0,1)};
}
function renderViews(width=$('views').clientWidth,height=$('views').clientHeight) {
  if(renderer.domElement.width!==Math.round(width*renderer.getPixelRatio())||renderer.domElement.height!==Math.round(height*renderer.getPixelRatio()))renderer.setSize(width,height,false);
  const sample=sampleAt(state.time);
  if(sample){const {a,b,alpha}=sample;pose(links.Body,a.position,a.quaternion,b.position,b.quaternion,alpha);linkNames.forEach((name,i)=>pose(links[name],a.fin_positions[i],a.fin_quaternions[i],b.fin_positions[i],b.fin_quaternions[i],alpha));}
  const pos=links.Body.position.clone();
  if(!orbitInitialized){controls.target.copy(pos);cameras[0].position.copy(pos).add(new THREE.Vector3(.85,-1.05,.43));orbitInitialized=true;previousPosition.copy(pos);}
  const movement=pos.clone().sub(previousPosition);cameras[0].position.add(movement);controls.target.add(movement);previousPosition.copy(pos);controls.update();
  cameras[1].position.set(3,-5,.25);cameras[1].lookAt(pos);cameras[1].fov=THREE.MathUtils.clamp(2*Math.atan(.7/cameras[1].position.distanceTo(pos))*deg,4,45);
  cameras[2].position.copy(pos).add(new THREE.Vector3(0,0,1.6));cameras[2].up.set(0,1,0);cameras[2].lookAt(pos);
  const q=links.Body.quaternion;
  // Body-fixed underside view: +X (forward) is up, camera looks up the EDF.
  // Recorded fin-link quaternions retain the real radial hinge deflections.
  cameras[3].position.copy(pos).add(new THREE.Vector3(0,0,-.55).applyQuaternion(q));
  cameras[3].up.set(1,0,0).applyQuaternion(q);
  cameras[3].lookAt(pos.clone().add(new THREE.Vector3(0,0,-.13).applyQuaternion(q)));
  thrustVector.position.copy(pos);thrustVector.setDirection(new THREE.Vector3(0,0,1).applyQuaternion(q));thrustVector.setLength(.015+(sample?.a.thrust_n??0)/90,.045,.022);
  const split=Math.floor(width*.66),right=width-split,third=height/3;
  const rects=[[0,0,split-1,height],[split+1,2*third,right-1,third-1],[split+1,third,right-1,third-1],[split+1,0,right-1,third-1]];
  renderer.setScissorTest(true);
  rects.forEach(([x,y,w,h],i)=>{renderer.setViewport(x,y,w,h);renderer.setScissor(x,y,w,h);cameras[i].aspect=w/h;if(i===3){cameras[i].left=-.1*w/h;cameras[i].right=.1*w/h;}cameras[i].updateProjectionMatrix();links.Body.visible=i!==3;trajectory.visible=i<3;plannedRoute.visible=i<3;thrustVector.visible=i===0;groundObjects.forEach(object=>{object.visible=i!==3;});renderer.render(scene,cameras[i]);});
  links.Body.visible=true;groundObjects.forEach(object=>{object.visible=true;});renderer.setScissorTest(false);
  finLabels.width=width;finLabels.height=height;
  drawBottomLabels(finLabels.getContext('2d'),width,height,sample);
}

function drawBottomLabels(ctx,width,height,sample){
  if(!sample)return;const split=Math.floor(width*.66)+1,right=width-split,third=height/3;
  const project=v=>{v.project(cameras[3]);return [split+(v.x+1)*right/2,2*third+(1-v.y)*third/2];};
  const radial=[[1,0,0],[0,-1,0],[-1,0,0],[0,1,0]];
  ctx.save();ctx.beginPath();ctx.rect(split,2*third,right,third);ctx.clip();
  if(state.metadata?.hinge_layout!=='radial_span_v1'){
    ctx.fillStyle='#471f15';ctx.fillRect(split,2*third,right,22);ctx.fillStyle='#ffd1b7';
    ctx.font='10px Arial';ctx.textAlign='center';ctx.fillText('OLD HINGE RECORDING · RUN AGAIN',split+right/2,2*third+15);
  }
  linkNames.forEach((name,i)=>{
    const anchor=links[name].position.clone().add(new THREE.Vector3(0,0,-.025).applyQuaternion(links[name].quaternion));
    const label=links.Body.position.clone().add(new THREE.Vector3(radial[i][0]*.077,radial[i][1]*.077,-.13).applyQuaternion(links.Body.quaternion));
    const [ax,ay]=project(anchor),[lx,ly]=project(label);
    ctx.strokeStyle='#7ef5d2aa';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(lx,ly);ctx.stroke();
    ctx.fillStyle='#081117e8';ctx.fillRect(lx-31,ly-12,62,26);ctx.textAlign='center';ctx.fillStyle='#b4cbd4';ctx.font='8px Arial';ctx.fillText(finNames[i],lx,ly-3);
    ctx.fillStyle='#7ef5d2';ctx.font='11px Consolas';const angle=THREE.MathUtils.lerp(sample.a.fin_angles[i],sample.b.fin_angles[i],sample.alpha)*deg;ctx.fillText(`${fmt(angle,1)}°`,lx,ly+10);
  });ctx.restore();
}

function updateTelemetry() {
  const sample=sampleAt(state.time),f=sample?.a;if(!f)return;
  text('clock',clock(state.time));text('hudAlt',`${fmt(f.position[2],2)} m`);text('hudVz',`${fmt(f.velocity[2],2)} m/s`);text('hudPad',`${fmt(f.pad_distance,2)} m`);
  text('thrust',fmt(f.thrust_n,1));text('throttle',fmt(f.throttle*100,1));text('rpm',fmt(f.rotor_rpm,0));$('thrustBar').style.width=`${Math.min(100,f.thrust_n/(state.metadata?.physics_parameters?.edf.max_thrust??48)*100)}%`;
  finNames.forEach((_,i)=>{text(`fc${i}`,fmt(f.fin_commands[i]*deg,2));text(`fa${i}`,fmt(THREE.MathUtils.lerp(f.fin_angles[i],sample.b.fin_angles[i],sample.alpha)*deg,2));text(`fcr${i}`,fmt(f.fin_command_rates[i]*deg,1));text(`far${i}`,fmt(f.fin_rates[i]*deg,1));});
  const rotationLimits=f.rotation?.soft_limits_deg_s??[90,90,180];
  f.gyro.forEach((v,i)=>{text(`g${i}`,fmt(v*deg,1));$('g'+i).style.color=Math.abs(v*deg)>rotationLimits[i]?'#ffc06a':'';});
  rotationFields.forEach(([key,,precision])=>[0,1,2].forEach(i=>text(`rotation_${key}_${i}`,fmt(f.rotation?.[key]?.[i],precision))));
  text('rotationNote',`Soft limits: ${rotationLimits.map(v=>fmt(v,0)).join(' / ')} °/s. ${f.rotation?'Travel counts turns and reversals; excess counts rotation above each limit.':'Cumulative rotation was not recorded in this older replay.'}`);
  const rateNote=state.metadata?.policy.controller==='pid'?'PID uses attitude feedback and fin mixing; its internal mix commands are not calibrated body-rate setpoints.':'PPO commands fin angles and throttle directly; no body-rate setpoint is generated.';
  text('rateNote',`${rateNote} Target rates above describe servo target motion.${f.observed_gyro?' Sensor P/Q/R: '+f.observed_gyro.map(v=>fmt(v*deg,1)).join(' / ')+' °/s.':''}`);
  text('soc',f.battery?fmt(f.battery.soc*100,1):'OFF');$('socBar').style.width=`${(f.battery?.soc??0)*100}%`;
  text('batteryState',!f.battery?'IDEAL BUS':f.battery.cutoff?'CUTOFF':f.battery.current_limited?'CURRENT LIMIT':'DISCHARGING');
  batteryFields.forEach(([key,,unit,d])=>text(`b_${key}`,fmt(f.battery?.[key],d)));
  text('propulsiveDv',fmt(f.propulsive_delta_v_m_s,2));
  text('contactState',f.control_phase==='POST_TOUCHDOWN_DISARM'?'MOTOR OFF / SETTLING':['AIRBORNE','CONTACT DWELL','LANDED','CRASHED'][f.contact]??'—');text('impact',fmt(f.impact_speed,3));text('contactLoad',fmt(f.contact_force_n,1));
  $('timeline').max=Math.max(.001,state.frames.at(-1).t);$('timeline').value=state.time;text('elapsed',`${fmt(state.time,2)} / ${fmt(state.frames.at(-1).t,2)} s`);
  $('play').textContent=state.playing?'Ⅱ':'▶';text('mode',state.recording?'RECORDING':state.live&&state.busy?'LIVE TELEMETRY':'RECORDED REPLAY');
}
function drawPlots() {
  const canvas=$('plots'),w=canvas.clientWidth,h=canvas.clientHeight,dpr=Math.min(devicePixelRatio,2);
  if(canvas.width!==Math.round(w*dpr)){canvas.width=w*dpr;canvas.height=h*dpr;}
  const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
  const n=state.frames.length,end=state.frames.at(-1)?.t??1,left=58,plotW=w-left-18;
  const specs=[['ALT / m',f=>f.position[2],colors[0]],['THRUST / N',f=>f.thrust_n,colors[1]],['GYRO / °/s',f=>Math.hypot(...f.gyro)*deg,colors[2]]];
  specs.forEach(([name,value,color],row)=>{const top=12+row*60,range=Math.max(1,...state.frames.map(value));ctx.fillStyle='#82939f';ctx.font='8px Arial';ctx.fillText(name,10,top+9);ctx.fillText(fmt(range,0),10,top+25);ctx.strokeStyle='#253039';ctx.beginPath();ctx.moveTo(left,top+40);ctx.lineTo(w-16,top+40);ctx.stroke();
    if(n){ctx.strokeStyle=color;ctx.lineWidth=1.3;ctx.beginPath();state.frames.forEach((f,i)=>{const x=left+f.t/end*plotW,y=top+40-value(f)/range*35;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();ctx.strokeStyle='#d0e0eb88';ctx.beginPath();const x=left+state.time/end*plotW;ctx.moveTo(x,top);ctx.lineTo(x,top+40);ctx.stroke();}
  });
  drawRotationPlots();
}
function drawRotationPlots(){
  const canvas=$('rotationPlots'),w=canvas.clientWidth,h=canvas.clientHeight,dpr=Math.min(devicePixelRatio,2);
  if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=w*dpr;canvas.height=h*dpr;}
  const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
  const limits=state.frames[0]?.rotation?.soft_limits_deg_s??[90,90,180],end=state.frames.at(-1)?.t??1,left=40,right=w-6;
  ['ROLL','PITCH','YAW'].forEach((name,axis)=>{
    const center=27+axis*54,range=Math.max(limits[axis]*1.15,...state.frames.map(f=>Math.abs(f.gyro[axis]*deg))),scale=20/range;
    ctx.font='8px Arial';ctx.fillStyle=colors[axis];ctx.fillText(name,0,center-12);ctx.fillStyle='#82939f';ctx.fillText('±'+fmt(range,0),0,center+3);
    ctx.strokeStyle='#ffc06a88';ctx.setLineDash([3,3]);for(const sign of [-1,1]){ctx.beginPath();ctx.moveTo(left,center-sign*limits[axis]*scale);ctx.lineTo(right,center-sign*limits[axis]*scale);ctx.stroke();}ctx.setLineDash([]);
    ctx.strokeStyle=colors[axis];ctx.beginPath();state.frames.forEach((f,i)=>{const x=left+f.t/end*(right-left),y=center-f.gyro[axis]*deg*scale;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
    ctx.strokeStyle='#d0e0eb88';ctx.beginPath();const x=left+state.time/end*(right-left);ctx.moveTo(x,center-22);ctx.lineTo(x,center+22);ctx.stroke();
  });
}
function updateTrajectory(){trajectory.geometry.dispose();trajectory.geometry=new THREE.BufferGeometry().setFromPoints(state.frames.map(f=>new THREE.Vector3(...f.position)));}
function updateEvents(){
  const events=[];let previous=-1;
  let phase;
  for(const f of state.frames){if(f.contact!==previous){events.push(`${clock(f.t)}  ${['AIRBORNE','CONTACT DETECTED / DWELL','LANDED','CRASHED'][f.contact]}`);previous=f.contact;}if(f.control_phase==='POST_TOUCHDOWN_DISARM'&&phase!==f.control_phase)events.push(`${clock(f.t)}  MOTOR OFF / SETTLING CHECK`);phase=f.control_phase;}
  if(state.result)events.push(`${clock(state.result.duration_s)}  ${state.result.outcome}${state.result.outcome==='LANDED'?(state.result.success?' / SUCCESS CRITERIA MET':' / OUTSIDE SUCCESS CRITERIA'):''}`);
  $('events').replaceChildren(...events.map(e=>{const d=document.createElement('div');d.textContent=e;return d;}));
}

async function refreshHistory(){const missions=await api('/api/missions');const current=$('history').value;$('history').replaceChildren(new Option('Select a recorded mission',''),...missions.map(m=>new Option(`${m.request?.name??m.id} · ${m.summary?.success?'SUCCESS':m.phase??m.state}`,m.id)));$('history').value=current;return missions;}
async function selectMission(id){
  state.id=id;state.frames=[];state.metadata=null;state.time=0;state.result=null;state.playing=false;state.live=true;orbitInitialized=false;
  const mission=await poll();if(mission)fillMissionForm(mission.request);$('history').value=id;
  const url=new URL(location.href);url.searchParams.set('mission',id);history.replaceState({},'',url);
}
async function poll(){
  if(!state.id)return;const id=state.id;
  const [m,data]=await Promise.all([api(`/api/missions/${id}`),api(`/api/missions/${id}/frames?after=${state.frames.length}`)]);if(id!==state.id)return;
  state.metadata=m.metadata;state.frames.push(...data.frames);state.result=m.summary;state.busy=['starting','running'].includes(m.state);
  if(data.frames.length){if(state.live)state.time=state.frames.at(-1).t;updateTrajectory();drawPlots();}
  updateEvents();
  text('missionTitle',m.request.name);text('flightStatus',m.summary?.success?'LANDED / PASS':m.summary?.outcome==='LANDED'?'LANDED / FAIL':m.summary?.outcome??m.state.toUpperCase());
  $('flightStatus').style.color=m.state==='failed'||(m.summary&&!m.summary.success)?'#ffa78e':'';
  $('run').disabled=state.busy||state.recording||state.training;$('stop').disabled=!state.busy;$('export').disabled=state.busy||state.frames.length<2||state.recording;
  const profile=m.request.hardware_profile==='planned_8s'?'8S PLANNED':'6S LEGACY';text('footerProfile',`${profile} · 3.104 kg${m.metadata?.physics_dt?' · '+fmt(1/m.metadata.physics_dt,0)+' Hz PHYSICS':''}`);
  const recordedPolicy=m.metadata?.policy;
  text('notice',`${profile} · ${m.request.battery.enabled?'LiPo coupled to EDF':'Ideal voltage, battery disabled'}${m.metadata?.hinge_layout==='radial_span_v1'?' · Radial hinges':' · ARCHIVE: OLD HINGE AXES'}${recordedPolicy?.step?' · Recorded PPO '+fmt(recordedPolicy.step/1e6,2)+'M / '+recordedPolicy.action_mode:''}${recordedPolicy?.diagnostic_checkpoint_override?' · Experimental replay; full-task qualification pending':''} · Hardware calibration pending${m.metadata?.initial_conditions_outside_training?' · Outside full training envelope':''}`);
  message(m.error??(state.busy?`${m.phase} · ${m.frames??0} samples received`:m.summary?`${m.summary.outcome} · impact ${fmt(m.summary.impact_speed,3)} m/s · pad error ${fmt(m.summary.pad_distance,3)} m`:'Mission loaded'),!!m.error);
  $('jsonDownload').hidden=!state.frames.length;$('jsonDownload').href=`/api/missions/${id}/download`;
  $('videoDownload').hidden=!m.video;$('videoDownload').href=`/api/missions/${id}/video`;
  if(state.frames.length)updateTelemetry();
  return m;
}
function fillMissionForm(request){
  for(const key of ['name','controller','hardware_profile','seed','duration_s'])$(key).value=request[key];
  for(const key of ['position','velocity','attitude_deg','angular_rate_deg_s'])request[key].forEach((v,i)=>{$(`${key}_${i}`).value=v;});
  $('initial_motor_fraction').value=request.initial_motor_fraction*100;
  const selected=Array.isArray(request.disturbance)?request.disturbance:[request.disturbance];
  document.querySelectorAll('input[name="disturbance"]').forEach(input=>{input.checked=selected.includes(input.value);});
  $('battery_enabled').checked=request.battery.enabled;
  for(const key of ['capacity_ah','c_rating','max_current_a'])$(key).value=request.battery[key];
  $('initial_soc').value=request.battery.initial_soc*100;$('cell_resistance_ohm').value=request.battery.cell_resistance_ohm*1000;
  text('packLabel',request.hardware_profile==='planned_8s'?'8S / ESTIMATED':'6S / ESTIMATED');
  planner.setWaypoints(request.waypoints??[]);
}
function missionRequest(){
  const result={};for(const key of ['name','controller','hardware_profile'])result[key]=$(key).value;
  result.disturbance=[...document.querySelectorAll('input[name="disturbance"]:checked')].map(input=>input.value);
  for(const key of ['seed','duration_s'])result[key]=Number($(key).value);
  for(const key of ['position','velocity','attitude_deg','angular_rate_deg_s'])result[key]=[0,1,2].map(i=>Number($(`${key}_${i}`).value));
  result.initial_motor_fraction=Number($('initial_motor_fraction').value)/100;
  result.waypoints=planner.getWaypoints();
  result.battery={enabled:$('battery_enabled').checked};for(const key of ['capacity_ah','c_rating','max_current_a'])result.battery[key]=Number($(key).value);
  result.battery.initial_soc=Number($('initial_soc').value)/100;result.battery.cell_resistance_ohm=Number($('cell_resistance_ohm').value)/1000;return result;
}
$('missionForm').addEventListener('submit',async e=>{e.preventDefault();$('run').disabled=true;message('Launching Isaac Sim…');try{const m=await api('/api/missions',missionRequest());await refreshHistory();await selectMission(m.id);}catch(error){message(error.message,true);$('run').disabled=false;}});
$('stop').onclick=async()=>{try{await api(`/api/missions/${state.id}/stop`,{});message('Stop requested; Isaac will finish the current control interval.');$('stop').disabled=true;}catch(e){message(e.message,true);}};
$('history').onchange=()=>{if($('history').value)selectMission($('history').value).catch(e=>message(e.message,true));};
$('play').onclick=()=>{if(!state.frames.length)return;state.live=false;if(state.time>=state.frames.at(-1).t)state.time=0;state.playing=!state.playing;};
$('timeline').oninput=()=>{state.live=false;state.playing=false;state.time=Number($('timeline').value);drawPlots();};
$('live').onclick=()=>{state.live=true;state.playing=false;state.time=state.frames.at(-1)?.t??0;};
$('hardware_profile').onchange=()=>text('packLabel',$('hardware_profile').value==='planned_8s'?'8S / ESTIMATED':'6S / ESTIMATED');
$('hardwareButton').onclick=()=>$('hardwareDialog').showModal();$('closeHardware').onclick=()=>$('hardwareDialog').close();

// Video uses the same four camera renders and exact telemetry as the replay.
const captureCanvas=document.createElement('canvas');captureCanvas.width=1600;captureCanvas.height=1000;
function captureFrame(){
  renderViews(1160,620);const ctx=captureCanvas.getContext('2d'),sample=sampleAt(state.time),f=sample?.a;if(!f)return;
  ctx.fillStyle='#080d12';ctx.fillRect(0,0,1600,1000);ctx.fillStyle='#eef3f5';ctx.font='22px Arial';ctx.fillText('EDF / MISSION CONTROL',26,40);ctx.font='27px Consolas';ctx.fillText(clock(state.time),1250,40);
  ctx.drawImage($('scene'),20,68,1160,620);
  ctx.save();ctx.translate(20,68);drawBottomLabels(ctx,1160,620,sampleAt(state.time));ctx.restore();
  ctx.fillStyle='#a2b6c4';ctx.font='11px Arial';const cameraLabels=[[34,90,'CAM 01 / ORBIT'],[800,90,'CAM 02 / GROUND'],[800,297,'CAM 03 / OVERHEAD'],[800,504,'CAM 04 / FINS BOTTOM']];cameraLabels.forEach(([x,y,label])=>ctx.fillText(label,x,y));
  ctx.fillStyle='#7ef5d2';ctx.font='18px Consolas';ctx.fillText(`ALT ${fmt(f.position[2],2)} m     VZ ${fmt(f.velocity[2],2)} m/s     PAD ${fmt(f.pad_distance,2)} m`,40,660);
  drawPlots();ctx.drawImage($('plots'),20,708,1160,190);
  let y=95;const line=(label,value,color='#e8f2f5')=>{ctx.fillStyle='#8296a3';ctx.font='11px Arial';ctx.fillText(label,1210,y);ctx.fillStyle=color;ctx.font='19px Consolas';ctx.fillText(value,1210,y+23);y+=56;};
  line('THRUST / THROTTLE',`${fmt(f.thrust_n)} N / ${fmt(f.throttle*100)} %`,'#7ef5d2');line('ROTOR',`${fmt(f.rotor_rpm,0)} rpm`);
  ctx.fillStyle='#8296a3';ctx.font='11px Arial';ctx.fillText('FIN       CMD °        ACT °',1210,y);y+=26;
  finNames.forEach((n,i)=>{ctx.font='16px Consolas';ctx.fillStyle='#dfeaf0';ctx.fillText(`${n.padEnd(6)} ${fmt(f.fin_commands[i]*deg,2).padStart(7)}   ${fmt(THREE.MathUtils.lerp(f.fin_angles[i],sample.b.fin_angles[i],sample.alpha)*deg,2).padStart(7)}`,1210,y);y+=25;});y+=15;
  ctx.fillStyle='#8296a3';ctx.font='11px Arial';ctx.fillText('FIN RATE °/s   TARGET     ACTUAL',1210,y);y+=24;
  finNames.forEach((n,i)=>{ctx.font='16px Consolas';ctx.fillStyle='#dfeaf0';ctx.fillText(`${n.padEnd(6)} ${fmt(f.fin_command_rates[i]*deg,1).padStart(7)}   ${fmt(f.fin_rates[i]*deg,1).padStart(7)}`,1210,y);y+=25;});y+=15;
  line('GYRO P / Q / R · °/s',f.gyro.map(v=>fmt(v*deg,1)).join(' / '));
  if(f.rotation){
    line('PEAK P / Q / R · °/s',f.rotation.peak_rate_deg_s.map(v=>fmt(v,0)).join(' / '));
    line('TRAVEL P / Q / R · degrees',f.rotation.angular_travel_deg.map(v=>fmt(v,0)).join(' / '));
    line('ABOVE SOFT LIMITS · seconds',f.rotation.time_above_limit_s.map(v=>fmt(v,2)).join(' / '));
  }
  line('LIPO / SOC',f.battery?`${state.metadata.battery_model.cells}S / ${fmt(f.battery.soc*100,1)} %`:'DISABLED');
  line('BUS / CURRENT',f.battery?`${fmt(f.battery.voltage_v,2)} V / ${fmt(f.battery.current_a,1)} A`:'IDEAL BUS');
  line('ENERGY / TEMPERATURE',f.battery?`${fmt(f.battery.energy_wh,2)} Wh / ${fmt(f.battery.temperature_c,1)} °C`:'—');
  line('PROPULSIVE DELTA-V',`${fmt(f.propulsive_delta_v_m_s,2)} m/s`);
  const ended=state.time>=state.frames.at(-1).t;line('FLIGHT STATE',ended&&state.result?`${state.result.outcome}${state.result.success?' / PASS':' / FAIL'}`:f.control_phase==='POST_TOUCHDOWN_DISARM'?'MOTOR OFF / SETTLING':['AIRBORNE','CONTACT DWELL','LANDED','CRASHED'][f.contact]);
  line('IMPACT / PAD ERROR',`${fmt(f.impact_speed,3)} m/s / ${fmt(f.pad_distance,3)} m`);
  ctx.fillStyle='#a9bac6';ctx.font='12px Arial';ctx.fillText(`${state.metadata?.request.name??'Landing'} · ${state.metadata?.policy.controller??''} · ${state.metadata?.hardware_profile??''} · Isaac PhysX / actual CAD / WebGL cameras`,26,935);
  ctx.fillStyle='#d1b580';ctx.font='11px Arial';ctx.fillText('Planned parts; pack and aerodynamic parameters estimated. No body-rate command: controller outputs fin angles and throttle.',26,960);
  return captureCanvas;
}
async function recordVideo(){
  if(state.recording||state.frames.length<2)return;
  if(!window.MediaRecorder||!MediaRecorder.isTypeSupported('video/webm;codecs=vp9'))throw new Error('This browser needs WebM VP9 recording support. Use Chrome or Edge.');
  const mid=state.id;state.recording=true;state.playing=false;state.live=false;state.time=0;$('export').disabled=true;$('run').disabled=true;$('history').disabled=true;
  const chunks=[],stream=captureCanvas.captureStream(30),recorder=new MediaRecorder(stream,{mimeType:'video/webm;codecs=vp9',videoBitsPerSecond:6500000});
  recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data);};
  const ended=new Promise((resolve,reject)=>{recorder.onstop=resolve;recorder.onerror=reject;});
  try{captureFrame();recorder.start(1000);const start=performance.now(),duration=state.frames.at(-1).t;
    await new Promise(resolve=>{function frame(now){state.time=Math.min(duration,Math.max(0,(now-start)/1000-.5));captureFrame();updateTelemetry();message(`Recording cameras + telemetry · ${fmt(state.time,1)} / ${fmt(duration,1)} s`);if((now-start)/1000<duration+1.2)requestAnimationFrame(frame);else resolve();}requestAnimationFrame(frame);});
    recorder.stop();await ended;const blob=new Blob(chunks,{type:'video/webm'});const response=await fetch(`/api/missions/${mid}/video`,{method:'POST',headers:{'X-Mission-Control':'local','Content-Type':'video/webm'},body:blob});if(!response.ok)throw new Error('Video save failed');
    $('videoDownload').href=`/api/missions/${mid}/video`;$('videoDownload').hidden=false;message('Video saved with synchronized cameras and telemetry.');return await response.json();
  }finally{if(recorder.state==='recording')recorder.stop();stream.getTracks().forEach(t=>t.stop());state.recording=false;$('export').disabled=false;$('run').disabled=state.busy||state.training;$('history').disabled=false;}
}
$('export').onclick=()=>recordVideo().catch(e=>message(e.message,true));
window.missionControl={state,seek(t){state.live=false;state.playing=false;state.time=t;renderViews();updateTelemetry();drawPlots();},recordVideo,captureFrame,selectMission,geometry};
let last=performance.now(),lastUi=0;
function animate(now){const elapsed=(now-last)/1000;last=now;if(state.playing&&!state.recording&&state.frames.length){state.time=Math.min(state.frames.at(-1).t,state.time+elapsed*Number($('speed').value));if(state.time>=state.frames.at(-1).t)state.playing=false;}if(!state.recording)renderViews();if(now-lastUi>100){updateTelemetry();drawPlots();lastUi=now;}requestAnimationFrame(animate);}
requestAnimationFrame(animate);
try{
  config=await api('/api/config');updateTraining(config);text('hardwareStatus',config.hardware.status);
  $('controller').replaceChildren(...Object.entries(config.policies).map(([key,name])=>new Option(name,key)));$('controller').value=config.defaults.controller;
  for(const part of config.hardware.parts){const el=document.createElement('div');el.className='hardware-part';const heading=document.createElement('h3');heading.textContent=part.part;const body=document.createElement('div');const name=document.createElement('strong');name.textContent=part.name;const spec=document.createElement('p');spec.textContent=part.spec;const basis=document.createElement('p');basis.textContent=part.basis;body.append(name,spec,basis);if(part.source){const a=document.createElement('a');a.href=part.source;a.target='_blank';a.rel='noreferrer';a.textContent='MANUFACTURER SOURCE ↗';body.append(a);}el.append(heading,body);$('hardwareParts').append(el);}
  const missions=await refreshHistory(),requested=new URLSearchParams(location.search).get('mission');const selected=requested??config.active??missions.find(m=>m.hinge_layout==='radial_span_v1'&&m.summary?.success)?.id??missions.find(m=>m.hinge_layout==='radial_span_v1'&&m.state==='complete')?.id??missions.find(m=>m.state==='complete')?.id;
  if(selected){await selectMission(selected);state.live=false;state.time=0;}
}catch(e){text('connection','SERVICE ERROR');message(e.message,true);}
setInterval(()=>{if(!state.recording)poll().catch(e=>message(e.message,true));},1200);
setInterval(async()=>{try{updateTraining(await api('/api/config'));}catch{}},5000);
