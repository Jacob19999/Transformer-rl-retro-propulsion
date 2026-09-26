import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { createMissionPlanner, samplePlannerSpline } from './planner.js';
import { attitude, drawAdi, drawWebcast, fitCanvas, series } from './instruments.js';
import { createCharts } from './charts.js';
import { createModels } from './models.js';
import { createChecklist } from './checklist.js';
import { createRotorAnimation } from './rotor.js';
import { collectPlans, planAt, createGuidancePanel } from './guidance.js';

const $ = id => document.getElementById(id);
const deg = 180 / Math.PI;
const finNames = ['FWD', 'RIGHT', 'AFT', 'LEFT'];
const linkNames = ['FwdFin', 'RightFin', 'AftFin', 'LeftFin'];
const contactNames = ['AIRBORNE', 'CONTACT DWELL', 'LANDED', 'CRASHED'];
const state = { id: null, frames: [], metadata: null, time: 0, playing: false, live: true, busy: false, recording: false, result: null, training:false, requestError:null, milestones: [], mission: null, connected: false };
let config, previousPosition = new THREE.Vector3(), orbitInitialized = false;
const fmt = (n, d = 1) => Number.isFinite(n) ? n.toFixed(d) : '—';
const clock = t => `T+ ${String(Math.floor(t / 60)).padStart(2, '0')}:${(t % 60).toFixed(2).padStart(5, '0')}`;
const text = (id, value) => { const el = $(id); if (el.textContent !== value) el.textContent = value; };
function message(value, error = false) { for (const id of ['runMessage', 'launchMessage']) { text(id, value); $(id).style.color = error ? '#ff8f8f' : ''; } }
// Pages share one state and one render loop; only the visible page is drawn,
// and only when something it shows has changed (see animate()).
const pages = ['flight', 'plan', 'telemetry', 'models'];
const pageShown = {};
let page = 'flight', sceneDirty = true, uiVersion = 0;
const invalidate = () => { sceneDirty = true; uiVersion++; };
function showPage(name) {
  page = pages.includes(name) ? name : 'flight';
  invalidate(); pageShown[page]?.();
  for (const key of pages) $(`page-${key}`).hidden = key !== page;
  document.querySelectorAll('[data-page]').forEach(tab => { const active = tab.dataset.page === page; tab.setAttribute('aria-selected', active); tab.tabIndex = active ? 0 : -1; });
  if (location.hash !== `#${page}`) history.replaceState({}, '', `${location.pathname}${location.search}#${page}`);
}
document.querySelectorAll('[data-page]').forEach(tab => tab.onclick = () => showPage(tab.dataset.page));
window.addEventListener('hashchange', () => showPage(location.hash.slice(1)));
showPage(location.hash.slice(1));
function trainingSummary(m){
  if(m?.step==null)return 'PPO TRAINING · Starting simulator and restoring checkpoint';
  const stage=`Stage ${m.stage+1}/${m.stages??'—'}${m.stage_name?' '+m.stage_name.replaceAll('_',' '):''}`;
  const evaluation=m.full_success==null?'Full-task evaluation pending':`Full-task evaluation ${fmt(m.full_success*100,1)}% at ${fmt(m.full_eval_step/1e6,2)}M`;
  if(m.task==='waypoint_flight')return `WAYPOINT FLIGHT PPO · ${fmt(m.step/1e6,2)}M transitions · ${stage} · Stage success ${fmt(m.stage_success*100,1)}% · Mean peak yaw ${fmt(m.peak_yaw,0)}°/s · ${fmt(m.sps,0)} steps/s · ${evaluation}`;
  return `PPO TRAINING · ${fmt(m.step/1e6,2)}M transitions · ${stage} · Recent stage success ${fmt(m.stage_success*100,1)}% · ${evaluation}${m.success_energy_wh==null?'':' · Successful full-task landings: '+fmt(m.success_energy_wh,2)+' Wh / '+fmt(m.success_delta_v,1)+' m/s Δv'}`;
}
function updateTraining(c){
  const choice=$('controller').value;
  const choices=Object.entries(c.policies);
  if(JSON.stringify([...$('controller').options].map(o=>[o.value,o.text]))!==JSON.stringify(choices)){
    $('controller').replaceChildren(...choices.map(([key,name])=>new Option(name,key)));
    $('controller').value=choice in c.policies?choice:c.defaults.controller;
    syncVaneModel();
  }
  state.connected=true;state.training=!!c.training;state.trainingMetrics=c.training_metrics;
  text('connection',state.training?'TRAINER ACTIVE / REPLAY READY':'ISAAC SERVICE ONLINE');
  $('trainingStatus').hidden=!state.training;text('trainingStatus',trainingSummary(c.training_metrics));
  $('run').disabled=state.busy||state.recording||state.training;
  renderBoard();uiVersion++;
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
$('finRows').innerHTML = finNames.map((n,i)=>`<tr><td>${n}</td><td class="defl"><div class="dbar"><i id="fd${i}" style="background:${series[i]}"></i><b id="fdc${i}"></b></div></td><td id="fc${i}">—</td><td id="fa${i}">—</td></tr>`).join('');
$('finRateRows').innerHTML = finNames.map((n,i)=>`<tr><td>${n}</td><td id="fcr${i}">—</td><td id="far${i}">—</td></tr>`).join('');
$('gyro').innerHTML = ['P · ROLL','Q · PITCH','R · YAW'].map((n,i)=>`<div class="gyro-row"><span>${n}</span><div class="gbar"><i id="gb${i}" style="background:${series[i]}"></i><span class="limit" style="left:25%"></span><span class="limit" style="left:75%"></span></div><b id="g${i}">—</b></div>`).join('');
const rotationFields=[['peak_rate_deg_s','PEAK °/s',1],['angular_travel_deg','TRAVEL °',0],['excess_rotation_deg','EXCESS °',0],['time_above_limit_s','OVER / s',2]];
$('rotationRows').innerHTML=rotationFields.map(([key,title])=>`<tr><td>${title}</td>${[0,1,2].map(i=>`<td id="rotation_${key}_${i}">—</td>`).join('')}</tr>`).join('');
const batteryFields = [['voltage_v','BUS VOLTAGE','V',2],['current_a','CURRENT','A',1],['power_w','POWER','W',0],['energy_wh','USED ENERGY','Wh',2],['temperature_c','PACK TEMP','°C',1],['ocv_v','OPEN CIRCUIT','V',2]];
$('batteryMetrics').innerHTML = batteryFields.map(([key,name,unit])=>`<div><span>${name}</span><b id="b_${key}">—</b><small>${unit}</small></div>`).join('');
$('batteryMetrics').insertAdjacentHTML('beforeend','<div><span>PROPULSIVE ΔV</span><b id="propulsiveDv">—</b><small>m/s</small></div>');
const seekTo=t=>{state.live=false;state.playing=false;state.time=THREE.MathUtils.clamp(t,0,state.frames.at(-1)?.t??0);};
const guidancePanel=createGuidancePanel($('optimizationPanel'),{onSeek:seekTo});
const charts=createCharts($('charts'),{onSeek:seekTo});
const flightCharts=createCharts($('flightCharts'),{onSeek:seekTo,titles:['ALTITUDE','VELOCITY','THRUST','BODY RATES · FRD']});
const models=createModels($('models'),api);
// Live details follow the next-run form; mass and fin travel come from the loaded replay's physics.
const checklist=createChecklist($('checklist'),{onChange:renderBoard,context:()=>{
  const physics=state.metadata?.physics_parameters,controller=$('controller').value;
  return {cells:$('hardware_profile').value==='planned_8s'?8:6,soc:Number($('initial_soc').value),controller:config?.policies?.[controller]??controller,
    mass:physics?.vehicle?.total_mass??3.104,finLimit:(physics?.vehicle?.fins?.max_deflection??.262)*deg};
}});
$('missionForm').addEventListener('input',()=>checklist.update());$('missionForm').addEventListener('change',()=>checklist.update());
// The models panel polls training logs, so it only refreshes while shown.
pageShown.models=()=>models.refresh().catch(()=>{});

// No preserveDrawingBuffer: captureFrame() copies the canvas in the same task
// as its render, while the buffer is still valid, so frames skip the extra copy.
const renderer = new THREE.WebGLRenderer({ canvas: $('scene'), antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
renderer.setClearColor('#05080b');
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.45;
const scene = new THREE.Scene();
scene.background = new THREE.Color('#070b0f');
scene.fog = new THREE.Fog('#070b0f', 160, 520);
scene.add(new THREE.HemisphereLight(0xdfeaf2, 0x2c3236, 2.8));
const sun = new THREE.DirectionalLight(0xf2f6f8, 3.2); sun.position.set(4,-6,12); scene.add(sun);
const fill = new THREE.DirectionalLight(0x8aa4b8, 1.8); fill.position.set(-3,4,3); scene.add(fill);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(300,300), new THREE.MeshStandardMaterial({color:0x0c0f12, roughness:.98}));
floor.position.z = -.006; scene.add(floor);
const grid = new THREE.GridHelper(100,100,0x3a444c,0x1c2227); grid.rotation.x = Math.PI/2; grid.position.z=.001; scene.add(grid);
const pad = new THREE.Mesh(new THREE.CircleGeometry(1.25,80), new THREE.MeshStandardMaterial({color:0x1f2327,roughness:.95})); pad.position.z=.004;scene.add(pad);
for (const radius of [.5, 1.15]) { const ring = new THREE.Mesh(new THREE.RingGeometry(radius-.012,radius,96),new THREE.MeshBasicMaterial({color:radius<1?0xf4f6f7:0x6d777f,side:THREE.DoubleSide}));ring.position.z=.007;scene.add(ring); }
for (const angle of [0,Math.PI/2]) { const line = new THREE.Mesh(new THREE.PlaneGeometry(.55,.018),new THREE.MeshBasicMaterial({color:0xd6dde1})); line.rotation.z=angle;line.position.z=.008;scene.add(line); }
const groundObjects = scene.children.filter(object=>object.isMesh||object===grid);
const trajectory = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({color:0xf4f6f7,transparent:true,opacity:.55})); scene.add(trajectory);
const plannedRoute=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0x3987e5,transparent:true,opacity:.75}));scene.add(plannedRoute);
// Convex guidance: the optimized trajectory in force at the replay time (the
// SOCP is re-solved every 0.5 s, so the drawn plan changes as the flight runs).
const guidancePath=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0x2fbf71,transparent:true,opacity:.9}));scene.add(guidancePath);
const guidanceMarkers=new THREE.Group();scene.add(guidanceMarkers);
const guidanceMarker=(color,wireframe=false)=>{const mesh=new THREE.Mesh(new THREE.SphereGeometry(1,12,8),new THREE.MeshBasicMaterial({color,wireframe,depthTest:false}));mesh.renderOrder=5;guidanceMarkers.add(mesh);return mesh;};
const vehicleMarker=guidanceMarker(0xf4f6f7),referenceMarker=guidanceMarker(0x6ab7ff,true),endpointMarker=guidanceMarker(0x48dba2,true);
let guidancePlanShown=null;
function updateGuidancePath(){
  const plan=planAt(state.plans??[],state.time);
  if(plan===guidancePlanShown)return;guidancePlanShown=plan;setLinePoints(guidancePath,plan?.positions??[]);
}
function setLinePoints(line,points){
  line.geometry.dispose();line.geometry=new THREE.BufferGeometry();
  line.geometry.setAttribute('position',new THREE.Float32BufferAttribute(points.flat(),3));sceneDirty=true;
}
function updatePlannedRoute(){
  // Replay cameras show the recorded plan; editing a new draft only changes
  // the planner canvases, never the reference displayed in exported footage.
  const initial=state.frames[0]?.position??readPlannerInitial().position;
  const waypoints=state.frames.length?(state.metadata?.request?.waypoints??[]):planner.getWaypoints();
  setLinePoints(plannedRoute,waypoints.length?samplePlannerSpline(initial,waypoints):[]);
}
const thrustVector = new THREE.ArrowHelper(new THREE.Vector3(0,0,1),new THREE.Vector3(),.4,0xfab219,.05,.025);scene.add(thrustVector);
const cameras = Array.from({length:4},()=> {const c = new THREE.PerspectiveCamera(40,1,.008,300);c.up.set(0,0,1);return c;});
cameras[3] = new THREE.OrthographicCamera(-.18,.18,.10,-.10,.001,5);
const finLabels = document.createElement('canvas');finLabels.id='finLabels';$('views').append(finLabels);
document.querySelector('.fin-tag b').textContent='FINS / BOTTOM';
const controls = new OrbitControls(cameras[0],$('scene')); controls.enableDamping = true; controls.dampingFactor=.1;controls.minDistance=.3;controls.maxDistance=40;controls.enablePan=false;
let cameraMode='vehicle',overviewPlan=null,overviewSize='';
function setCameraMode(mode){
  cameraMode=mode;overviewPlan=null;orbitInitialized=false;
  $('cameraVehicle').setAttribute('aria-pressed',mode==='vehicle');$('cameraPlan').setAttribute('aria-pressed',mode==='plan');
  document.querySelector('.main-tag b').textContent=mode==='plan'?'PLAN OVERVIEW / DRAG TO ROTATE':'ORBIT / DRAG TO ROTATE';
  controls.maxDistance=mode==='plan'?600:40;invalidate();
}
$('cameraVehicle').onclick=()=>setCameraMode('vehicle');$('cameraPlan').onclick=()=>setCameraMode('plan');
// Drags and damping dispatch 'change' until the orbit settles; each one earns a render.
controls.addEventListener('change',()=>{sceneDirty=true;});
// Detailed CAD/Blender render model (export_visual_model.py), one node per
// rigid link in the same local frames as the physics USD in geometry.json.
// Version the model together with the rotor animation so browser caches do
// not reuse the older export, whose fan was baked into the Body mesh.
const model = await new GLTFLoader().loadAsync('/static/drone_visual.glb?v=edf-rotor-1');
const geometry = await (await fetch('/static/geometry.json')).json();
const links = Object.fromEntries(['Body',...linkNames].map(name=>[name,model.scene.getObjectByName(name)]));
const rotor = model.scene.getObjectByName('EDFRotor');
const rotorAnimation = createRotorAnimation(rotor);
$('rpm').title = 'Recorded rotor RPM. Fan blade animation is slowed 200× for visibility.';
// Fins take their UI series colour so CAM 04 labels match the hardware. CAD
// tessellation has open shells (both faces drawn), and the studio reflections
// are scoped to the drone so the glossy prints read without relighting the pad.
linkNames.forEach((name,i)=>{links[name].material=new THREE.MeshStandardMaterial({color:series[i],roughness:.5});});
const droneEnvironment=new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(),.04).texture;
model.scene.traverse(object=>{if(object.isMesh)Object.assign(object.material,{side:THREE.DoubleSide,envMap:droneEnvironment,envMapIntensity:.45});});
scene.add(model.scene);
for (const link of geometry.links) { const obj=links[link.name]; if(obj){obj.position.fromArray(link.neutral_position);obj.quaternion.set(link.neutral_quaternion[1],link.neutral_quaternion[2],link.neutral_quaternion[3],link.neutral_quaternion[0]);} }

// Per-frame scratch objects: the render path allocates nothing.
const scratch={v:new THREE.Vector3(),q:new THREE.Quaternion(),pos:new THREE.Vector3(),move:new THREE.Vector3(),a:new THREE.Vector3(),b:new THREE.Vector3()};
function pose(obj, aPos, aQuat, bPos, bQuat, alpha) {
  obj.position.fromArray(aPos).lerp(scratch.v.fromArray(bPos),alpha);
  obj.quaternion.set(aQuat[1],aQuat[2],aQuat[3],aQuat[0]).slerp(scratch.q.set(bQuat[1],bQuat[2],bQuat[3],bQuat[0]),alpha);
}
function sampleAt(t) {
  const frames=state.frames;if(!frames.length)return null;
  let lo=0,hi=frames.length-1;
  while(lo<hi){const m=Math.ceil((lo+hi)/2);if(frames[m].t<=t)lo=m;else hi=m-1;}
  const a=frames[lo],b=frames[Math.min(lo+1,frames.length-1)];
  return {a,b,index:lo,alpha:a===b?0:THREE.MathUtils.clamp((t-a.t)/(b.t-a.t),0,1)};
}
function renderViews(width=$('views').clientWidth,height=$('views').clientHeight) {
  if(renderer.domElement.width!==Math.round(width*renderer.getPixelRatio())||renderer.domElement.height!==Math.round(height*renderer.getPixelRatio()))renderer.setSize(width,height,false);
  const sample=sampleAt(state.time);updateGuidancePath();
  rotorAnimation.update(sample);
  if(sample){const {a,b,alpha}=sample;pose(links.Body,a.position,a.quaternion,b.position,b.quaternion,alpha);linkNames.forEach((name,i)=>pose(links[name],a.fin_positions[i],a.fin_quaternions[i],b.fin_positions[i],b.fin_quaternions[i],alpha));}
  const pos=scratch.pos.copy(links.Body.position);
  if(cameraMode==='plan'&&guidancePlanShown){
    if(!orbitInitialized||overviewPlan!==guidancePlanShown||overviewSize!==`${width}:${height}`){
      const bounds=new THREE.Box3().setFromPoints([...guidancePlanShown.positions.map(p=>new THREE.Vector3(...p)),pos.clone(),new THREE.Vector3()]);
      const center=bounds.getCenter(new THREE.Vector3()),radius=Math.max(1,bounds.getSize(new THREE.Vector3()).length()/2);
      const aspect=Math.max(.2,width*.66/height),halfFov=Math.atan(Math.tan(cameras[0].fov/2/deg)*Math.min(1,aspect));
      const distance=radius/Math.sin(halfFov)*1.18;
      const direction=orbitInitialized?scratch.v.copy(cameras[0].position).sub(controls.target).normalize():scratch.v.set(1,-1,.65).normalize();
      controls.target.copy(center);cameras[0].position.copy(center).add(direction.multiplyScalar(distance));
      cameras[0].far=Math.max(300,distance+radius*3);overviewPlan=guidancePlanShown;overviewSize=`${width}:${height}`;orbitInitialized=true;
    }
    previousPosition.copy(pos);controls.update();
  }else{
    if(!orbitInitialized){controls.target.copy(pos);cameras[0].position.copy(pos).add(scratch.v.set(.85,-1.05,.43));orbitInitialized=true;previousPosition.copy(pos);}
    const movement=scratch.move.copy(pos).sub(previousPosition);cameras[0].position.add(movement);controls.target.add(movement);previousPosition.copy(pos);controls.update();
  }
  const guidance=sample?.a.guidance;
  guidancePath.material.color.set(guidance?.solver?.mode==='soft_terminal'?0xfab219:0x48dba2);
  guidancePath.material.opacity=['TERMINAL_DESCENT','LANDED','HOLD'].includes(guidance?.phase) ? .3 : .9;
  trajectory.geometry.setDrawRange(0,sample?sample.index+1:0);
  const markerSize=Math.max(.04,cameras[0].position.distanceTo(controls.target)*.003);
  vehicleMarker.position.copy(pos);vehicleMarker.scale.setScalar(markerSize);
  referenceMarker.visible=!!guidance?.reference_position;
  if(referenceMarker.visible){referenceMarker.position.fromArray(guidance.reference_position);referenceMarker.scale.setScalar(markerSize*1.7);}
  endpointMarker.visible=!!guidancePlanShown;
  if(endpointMarker.visible){endpointMarker.position.fromArray(guidancePlanShown.positions.at(-1));endpointMarker.scale.setScalar(markerSize*1.7);endpointMarker.material.color.copy(guidancePath.material.color);}
  cameras[1].position.set(3,-5,.25);cameras[1].lookAt(pos);cameras[1].fov=THREE.MathUtils.clamp(2*Math.atan(.7/cameras[1].position.distanceTo(pos))*deg,4,45);
  cameras[2].position.copy(pos).add(scratch.v.set(0,0,1.6));cameras[2].up.set(0,1,0);cameras[2].lookAt(pos);
  const q=links.Body.quaternion;
  // Body-fixed underside view: +X (forward) is up, camera looks up the EDF.
  // Recorded fin-link quaternions retain the real radial hinge deflections.
  cameras[3].position.copy(pos).add(scratch.v.set(0,0,-.55).applyQuaternion(q));
  cameras[3].up.set(1,0,0).applyQuaternion(q);
  cameras[3].lookAt(scratch.a.copy(pos).add(scratch.v.set(0,0,-.13).applyQuaternion(q)));
  thrustVector.position.copy(pos);thrustVector.setDirection(scratch.v.set(0,0,1).applyQuaternion(q));thrustVector.setLength(.015+(sample?.a.thrust_n??0)/90,.045,.022);
  const split=Math.floor(width*.66),right=width-split,third=height/3;
  const rects=[[0,0,split-1,height],[split+1,2*third,right-1,third-1],[split+1,third,right-1,third-1],[split+1,0,right-1,third-1]];
  renderer.setScissorTest(true);
  rects.forEach(([x,y,w,h],i)=>{renderer.setViewport(x,y,w,h);renderer.setScissor(x,y,w,h);cameras[i].aspect=w/h;if(i===3){cameras[i].left=-.1*w/h;cameras[i].right=.1*w/h;}cameras[i].updateProjectionMatrix();links.Body.visible=i!==3;trajectory.visible=i<3;plannedRoute.visible=i<3;guidancePath.visible=i<3;guidanceMarkers.visible=i===0&&cameraMode==='plan'&&!!guidance;thrustVector.visible=i===0;groundObjects.forEach(object=>{object.visible=i!==3;});renderer.render(scene,cameras[i]);});
  links.Body.visible=true;groundObjects.forEach(object=>{object.visible=true;});renderer.setScissorTest(false);
  drawBottomLabels(fitCanvas(finLabels,width,height),width,height,sample);
}

function drawBottomLabels(ctx,width,height,sample){
  if(!sample)return;const split=Math.floor(width*.66)+1,right=width-split,third=height/3;
  const project=v=>{v.project(cameras[3]);return [split+(v.x+1)*right/2,2*third+(1-v.y)*third/2];};
  const radial=[[1,0,0],[0,-1,0],[-1,0,0],[0,1,0]];
  ctx.save();ctx.beginPath();ctx.rect(split,2*third,right,third);ctx.clip();
  if(state.metadata?.hinge_layout!=='radial_span_v1'){
    ctx.fillStyle='#3a1414';ctx.fillRect(split,2*third,right,22);ctx.fillStyle='#ffc9c9';
    ctx.font='10px Bahnschrift, Arial';ctx.textAlign='center';ctx.fillText('OLD HINGE RECORDING · RUN AGAIN',split+right/2,2*third+15);
  }
  linkNames.forEach((name,i)=>{
    const anchor=scratch.a.copy(links[name].position).add(scratch.v.set(0,0,-.025).applyQuaternion(links[name].quaternion));
    const label=scratch.b.copy(links.Body.position).add(scratch.v.set(radial[i][0]*.077,radial[i][1]*.077,-.13).applyQuaternion(links.Body.quaternion));
    const [ax,ay]=project(anchor),[lx,ly]=project(label);
    ctx.strokeStyle=series[i];ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(lx,ly);ctx.stroke();
    ctx.fillStyle='rgba(0,0,0,.82)';ctx.fillRect(lx-31,ly-12,62,26);ctx.fillStyle=series[i];ctx.fillRect(lx-31,ly-12,2,26);ctx.textAlign='center';ctx.fillStyle='#b9c2c8';ctx.font='8px Bahnschrift, Arial';ctx.fillText(finNames[i],lx,ly-3);
    ctx.fillStyle='#f4f6f7';ctx.font='11px Bahnschrift, Consolas';const angle=THREE.MathUtils.lerp(sample.a.fin_angles[i],sample.b.fin_angles[i],sample.alpha)*deg;ctx.fillText(`${fmt(angle,1)}°`,lx,ly+10);
  });ctx.restore();
}

// Discrete mission events derived from recorded frames: shared by the webcast
// timeline, chart markers and the event log so they can never disagree.
function computeMilestones(){
  const result=[],frames=state.frames;if(!frames.length)return result;
  result.push({t:frames[0].t,label:'START'});
  let contact=frames[0].contact,waypoint=frames[0].mission?.waypoint_index,phase=frames[0].control_phase,ready=!!frames[0].mission?.ready_to_land;
  let guidancePhase=frames[0].guidance?.phase,fallback=false;
  const guidanceLabels={SPOOL_UP:'SPOOL UP',ROUTE:'ROUTE',POWERED_DESCENT:'PDG',TERMINAL_DESCENT:'TERMINAL',HOLD:'HOLD'};
  for(const f of frames){
    // Convex guidance phase changes and soft-terminal (no safe plan) fallbacks.
    const g=f.guidance;
    if(g?.phase&&g.phase!==guidancePhase&&guidanceLabels[g.phase])result.push({t:f.t,label:guidanceLabels[g.phase],detail:`CONVEX GUIDANCE · ${g.phase.replaceAll('_',' ')}`,tone:g.phase==='HOLD'?'critical':undefined});
    guidancePhase=g?.phase??guidancePhase;
    const soft=g?.solver?.mode==='soft_terminal';
    if(soft&&!fallback)result.push({t:f.t,label:'FALLBACK',detail:'NO SAFE PLAN · SOFT-TERMINAL MAXIMUM-BRAKING PLAN',tone:'warning'});
    if(g?.solver)fallback=soft;
    const index=f.mission?.waypoint_index;
    if(index!=null&&waypoint!=null&&index>waypoint){for(let k=waypoint;k<index;k++)result.push({t:f.t,label:`WP ${k+1}`,detail:`WAYPOINT ${k+1} CAPTURED`});}
    waypoint=index??waypoint;
    // The final capture already marks the moment the route completes.
    if(f.mission?.ready_to_land&&!ready&&result.at(-1)?.t!==f.t)result.push({t:f.t,label:'LAND',detail:'ROUTE COMPLETE · LANDING PHASE'});
    ready=!!f.mission?.ready_to_land||ready;
    if(f.contact!==contact){const tone=f.contact===2?'good':f.contact===3?'critical':undefined;result.push({t:f.t,label:['AIRBORNE','CONTACT','LANDED','CRASHED'][f.contact],detail:['AIRBORNE','CONTACT DETECTED / DWELL','LANDED','CRASHED'][f.contact],tone});contact=f.contact;}
    if(f.control_phase==='POST_TOUCHDOWN_DISARM'&&phase!==f.control_phase)result.push({t:f.t,label:'MOTOR OFF',detail:'MOTOR OFF / SETTLING CHECK'});
    phase=f.control_phase;
  }
  if(state.result){
    const r=state.result,tone=r.success?'good':'critical';
    const detail=`${r.outcome}${r.outcome==='LANDED'?(r.success?' / SUCCESS CRITERIA MET':' / OUTSIDE SUCCESS CRITERIA'):''}`;
    const t=Math.min(r.duration_s??frames.at(-1).t,frames.at(-1).t);
    const last=result.at(-1);
    if(last&&last.label===r.outcome){last.tone=tone;last.detail=detail;}else result.push({t,label:r.success?'PASS':r.outcome,detail,tone});
  }
  return result;
}

function chartContext(){
  const physics=state.metadata?.physics_parameters;
  const limits=state.frames[0]?.rotation?.soft_limits_deg_s??[90,90,180];
  return {maxThrust:physics?.edf?.max_thrust??48,softLimits:[...new Set(limits)],
    finLimit:(physics?.vehicle?.fins?.max_deflection??.262)*deg,maxCurrent:state.metadata?.request?.battery?.max_current_a};
}
function webcastMaxima(){
  let speed=5,altitude=10;
  for(const f of state.frames){speed=Math.max(speed,Math.hypot(...f.velocity));altitude=Math.max(altitude,f.position[2]);}
  return {speed:Math.ceil(speed*1.1),altitude:Math.ceil(altitude*1.1),thrust:state.metadata?.physics_parameters?.edf?.max_thrust??48};
}
let maxima=webcastMaxima(),dataVersion=0;
function dataChanged(){
  dataVersion++;invalidate();
  rotorAnimation.setFrames(state.frames);
  state.plans=collectPlans(state.frames);guidancePlanShown=undefined;
  guidancePanel.setData(state.frames,state.plans);
  $('cameraPlan').hidden=!state.plans.length;
  if(!state.plans.length&&cameraMode==='plan')setCameraMode('vehicle');
  state.milestones=computeMilestones();maxima=webcastMaxima();
  const events=state.milestones.filter(m=>m.label!=='START'),context=chartContext();
  charts.setData(state.frames,context,events);flightCharts.setData(state.frames,context,events);
  updateTrajectory();updatePlannedRoute();updateEvents();checklist.update();
}

// GO / NO-GO board: every state carries a glyph and a word, never colour alone.
const glyphs={good:'✓',warning:'!',critical:'✕',idle:'–'};
let boardHtml='';
function renderBoard(){
  const f=sampleAt(state.time)?.a,items=[];
  items.push(['ISAAC SIM',!state.connected?['critical','OFFLINE']:state.training?['warning','TRAINER OWNS GPU']:state.busy?['good','MISSION RUNNING']:['good','READY']]);
  const m=state.trainingMetrics;
  items.push(['PPO TRAINER',state.training?['good',m?.step!=null?`${m.task==='waypoint_flight'?'WAYPOINT':'LANDING'} · ${fmt(m.step/1e6,1)}M · S${m.stage+1}/${m.stages??'—'}`:'STARTING']:['idle','IDLE']]);
  const controller=$('controller').value,policy=config?.policies?.[controller]??controller;
  items.push(['NEXT CONTROLLER',controller==='pid'?['good','PID']:controller==='convex'?['good','CONVEX SOCP']:[/EXPERIMENTAL/i.test(policy)?'warning':'good',controller==='ppo_mission'?'PPO · EXPERIMENTAL':policy.toUpperCase()]]);
  // A rotor spun up in flight takes its angular momentum (~0.78 N m s at hover)
  // from the body; momentum-bounded vanes hold ~0.3 N m, so the body spins.
  // Spool up on the pad, or start in the air with the rotor already turning.
  const airborne=Number($('position_2')?.value??0)>1,cold=Number($('initial_motor_fraction').value)<50;
  if(airborne&&cold&&$('vane_model').value==='momentum')items.push(['ROTOR START',['warning','COLD IN AIR · SPIN-UP YAWS THE BODY']]);
  const preflight=checklist.summary();
  items.push(['PRE-FLIGHT',preflight.done===preflight.total?['good',`COMPLETE · ${preflight.total}/${preflight.total}`]:preflight.done?['warning',`HOLD · ${preflight.done}/${preflight.total}`]:['idle','NOT STARTED']]);
  items.push(['TELEMETRY',!state.frames.length?['idle','NO DATA']:state.busy&&state.live?['good',`LIVE · ${state.frames.length} SAMPLES`]:['idle',`REPLAY · ${state.frames.length} SAMPLES`]]);
  if(f){
    const limits=f.rotation?.soft_limits_deg_s??[90,90,180],over=f.gyro.some((v,i)=>Math.abs(v*deg)>limits[i]);
    items.push(['VEHICLE',f.contact===3?['critical','CRASHED']:f.contact===2?['good','LANDED']:over?['warning','RATE LIMIT EXCEEDED']:['good',contactNames[f.contact]??'—']]);
    items.push(['POWER',!f.battery?['idle','IDEAL BUS']:f.battery.cutoff?['critical','CUTOFF']:f.battery.current_limited?['warning','CURRENT LIMIT']:['good',`${fmt(f.battery.soc*100,0)}% · ${fmt(f.battery.voltage_v,1)} V`]]);
  }else{items.push(['VEHICLE',['idle','NO DATA']]);items.push(['POWER',['idle','NO DATA']]);}
  const html=items.map(([name,[tone,value]])=>`<div class="go-item ${tone}"><i class="glyph" aria-hidden="true">${glyphs[tone]}</i><div><span>${name}</span><b>${value}</b></div></div>`).join('');
  if(html!==boardHtml){$('goBoard').innerHTML=html;boardHtml=html;}
}

function setBar(el,value,limit){
  const v=Math.max(-1,Math.min(1,value/limit));
  el.style.left=`${50+Math.min(0,v)*50}%`;el.style.width=`${Math.abs(v)*50}%`;
}
function updateTelemetry() {
  const sample=sampleAt(state.time),f=sample?.a;renderBoard();guidancePanel.update(f,state.time);if(!f)return;
  const lerp=(a,b)=>THREE.MathUtils.lerp(a,b,sample.alpha);
  text('clock',clock(state.time));text('telemetryTime',clock(f.t));
  text('hudAlt',fmt(f.position[2],2));text('hudVz',fmt(f.velocity[2],2));text('hudVh',fmt(Math.hypot(f.velocity[0],f.velocity[1]),2));text('hudPad',fmt(f.pad_distance,2));
  const att=attitude(f.quaternion);drawAdi($('adi'),att);
  text('attRoll',fmt(att.roll,1));text('attPitch',fmt(att.pitch,1));text('attYaw',fmt((att.yaw+360)%360,0));text('attTilt',fmt(att.tilt,1));
  const mission=f.mission,g=f.guidance,guidancePhase=g?.phase?.replaceAll('_',' ');
  const guidanceLine=g?`CONVEX · ${guidancePhase}${g.time_to_go_s!=null?' · gate in '+fmt(g.time_to_go_s,1)+' s':''}${g.solver?` · plan #${g.plan_id} ${g.solver.mode==='soft_terminal'?'SOFT-TERMINAL FALLBACK':(g.solver.status??'status unavailable')} · ${fmt(g.solver.solve_ms,0)} ms / ${g.solver.solves} SOCPs`:''}`:null;
  const phaseLine=mission?mission.ready_to_land?`LAND · ${mission.waypoint_count} waypoints completed · Soft contact required`:`${mission.phase} · Waypoint ${mission.waypoint_index+1}/${mission.waypoint_count} · Cross-track ${fmt(mission.cross_track_error_m,2)} m${mission.phase==='HOVER'?' · Hold '+fmt(mission.hold_elapsed_s,1)+' / '+fmt(mission.waypoints[mission.waypoint_index].hold_s,1)+' s':''}`:guidanceLine??'Waypoint telemetry was not recorded in this replay.';
  text('waypointStatus',mission&&g?`${phaseLine} · CONVEX ${guidancePhase}`:phaseLine);text('waypointBadge',mission?`${Math.min(mission.waypoint_index,mission.waypoint_count)}/${mission.waypoint_count} CAPTURED`:'');
  $('phaseTag').hidden=!mission&&!g;if(mission)text('phaseTag',mission.ready_to_land?'LANDING PHASE':`${mission.phase} · WP ${mission.waypoint_index+1}/${mission.waypoint_count}`);else if(g)text('phaseTag',`CONVEX · ${guidancePhase}`);
  const physics=state.metadata?.physics_parameters,maxThrust=physics?.edf?.max_thrust??48,mass=physics?.vehicle?.total_mass??3.104;
  text('thrust',fmt(f.thrust_n,1));text('thrustWeight',`T/W ${fmt(f.thrust_n/(mass*9.81),2)}`);text('throttle',fmt(f.throttle*100,1));text('rpm',fmt(f.rotor_rpm,0));$('thrustBar').style.width=`${Math.min(100,f.thrust_n/maxThrust*100)}%`;
  const finLimit=(physics?.vehicle?.fins?.max_deflection??.262)*deg;text('finLimitLabel',fmt(finLimit,0));
  finNames.forEach((_,i)=>{const actual=lerp(f.fin_angles[i],sample.b.fin_angles[i])*deg,command=f.fin_commands[i]*deg;
    text(`fc${i}`,fmt(command,2));text(`fa${i}`,fmt(actual,2));text(`fcr${i}`,fmt(f.fin_command_rates[i]*deg,1));text(`far${i}`,fmt(f.fin_rates[i]*deg,1));
    setBar($('fd'+i),actual,finLimit);$('fdc'+i).style.left=`calc(${50+Math.max(-1,Math.min(1,command/finLimit))*50}% - 1px)`;});
  const rotationLimits=f.rotation?.soft_limits_deg_s??[90,90,180];
  f.gyro.forEach((v,i)=>{const value=v*deg,scale=rotationLimits[i]*2;text(`g${i}`,fmt(value,1));$('g'+i).classList.toggle('over',Math.abs(value)>rotationLimits[i]);
    setBar($('gb'+i),value,scale);});
  rotationFields.forEach(([key,,precision])=>[0,1,2].forEach(i=>text(`rotation_${key}_${i}`,fmt(f.rotation?.[key]?.[i],precision))));
  text('rotationNote',`Soft limits: ${rotationLimits.map(v=>fmt(v,0)).join(' / ')} °/s; gyro bars span ±2× each limit. ${f.rotation?'Travel counts turns and reversals; excess counts rotation above each limit.':'Cumulative rotation was not recorded in this older replay.'}`);
  const recordedController=state.metadata?.policy.controller;
  const rateNote=recordedController==='pid'?'PID uses attitude feedback and fin mixing; its internal mix commands are not calibrated body-rate setpoints.':recordedController==='convex'?'Convex guidance plans a thrust-vector trajectory (SOCP); a geometric attitude loop turns the thrust direction into fin efforts. No body-rate setpoint is generated.':'PPO commands fin angles and throttle directly; no body-rate setpoint is generated.';
  text('rateNote',`${rateNote}${f.observed_gyro?' Sensor P/Q/R: '+f.observed_gyro.map(v=>fmt(v*deg,1)).join(' / ')+' °/s.':''}`);
  text('soc',f.battery?fmt(f.battery.soc*100,1):'OFF');$('socBar').style.width=`${(f.battery?.soc??0)*100}%`;
  text('batteryState',!f.battery?'IDEAL BUS':f.battery.cutoff?'CUTOFF':f.battery.current_limited?'CURRENT LIMIT':'DISCHARGING');
  batteryFields.forEach(([key,,unit,d])=>text(`b_${key}`,fmt(f.battery?.[key],d)));
  text('propulsiveDv',fmt(f.propulsive_delta_v_m_s,2));
  text('contactState',f.control_phase==='POST_TOUCHDOWN_DISARM'?'MOTOR OFF / SETTLING':contactNames[f.contact]??'—');text('impact',fmt(f.impact_speed,3));text('contactLoad',fmt(f.contact_force_n,1));
  $('timeline').max=Math.max(.001,state.frames.at(-1).t);$('timeline').value=state.time;text('elapsed',`${fmt(state.time,2)} / ${fmt(state.frames.at(-1).t,2)} s`);
  $('play').textContent=state.playing?'Ⅱ':'▶';text('mode',state.recording?'RECORDING':state.live&&state.busy?'LIVE TELEMETRY':'RECORDED REPLAY');
}
function drawWebcastBand(canvas=$('webcast'),width,height){
  drawWebcast(canvas,{frame:sampleAt(state.time)?.a,time:state.time,end:state.frames.at(-1)?.t??1,milestones:state.milestones,maxima,width,height});
}
function drawPlots(){
  charts.draw(state.time);flightCharts.draw(state.time);
  if(page==='flight')drawWebcastBand();
  if(page==='telemetry')drawRotationPlots();
}
// The traces only change with the data, so they are drawn once into a cached
// layer; each UI tick just copies it and moves the time cursor.
const rotationCache={canvas:document.createElement('canvas'),key:''};
function drawRotationPlots(){
  const canvas=$('rotationPlots'),w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h)return;
  const end=state.frames.at(-1)?.t??1,left=40,right=w-6,key=`${dataVersion}:${w}:${h}`;
  if(rotationCache.key!==key){
    rotationCache.key=key;const ctx=fitCanvas(rotationCache.canvas,w,h);
    const limits=state.frames[0]?.rotation?.soft_limits_deg_s??[90,90,180];
    ['ROLL','PITCH','YAW'].forEach((name,axis)=>{
      let range=limits[axis]*1.15;for(const f of state.frames)range=Math.max(range,Math.abs(f.gyro[axis]*deg));
      const center=27+axis*54,scale=20/range;
      ctx.font='8px Bahnschrift, Arial';ctx.fillStyle='#b9c2c8';ctx.fillText(name,0,center-12);ctx.fillStyle='#7c878f';ctx.fillText('±'+fmt(range,0),0,center+3);
      ctx.strokeStyle='rgba(250,178,25,.55)';ctx.setLineDash([3,3]);for(const sign of [-1,1]){ctx.beginPath();ctx.moveTo(left,center-sign*limits[axis]*scale);ctx.lineTo(right,center-sign*limits[axis]*scale);ctx.stroke();}ctx.setLineDash([]);
      ctx.strokeStyle=series[axis];ctx.lineWidth=1.4;ctx.beginPath();state.frames.forEach((f,i)=>{const x=left+f.t/end*(right-left),y=center-f.gyro[axis]*deg*scale;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
    });
  }
  const ctx=fitCanvas(canvas,w,h);ctx.drawImage(rotationCache.canvas,0,0,w,h);
  const x=left+state.time/end*(right-left);ctx.strokeStyle='rgba(244,246,247,.7)';ctx.lineWidth=1;ctx.beginPath();
  for(let axis=0;axis<3;axis++){const center=27+axis*54;ctx.moveTo(x,center-22);ctx.lineTo(x,center+22);}ctx.stroke();
}
function updateTrajectory(){setLinePoints(trajectory,state.frames.map(f=>f.position));}
function updateEvents(){
  const rows=state.milestones.filter(m=>m.label!=='START'||state.frames.length).map(m=>{const d=document.createElement('div');d.textContent=`${clock(m.t)}  ${m.detail??m.label}`;if(m.tone)d.className='tone-'+m.tone;return d;});
  if(!rows.length){$('events').textContent='Awaiting simulation.';return;}
  $('events').replaceChildren(...rows);
}

async function refreshHistory(){const missions=await api('/api/missions');const current=$('history').value;$('history').replaceChildren(new Option('Select a recorded mission',''),...missions.map(m=>new Option(`${m.request?.name??m.id} · ${m.summary?.success?'SUCCESS':m.phase??m.state}`,m.id)));$('history').value=current;return missions;}
async function selectMission(id){
  state.requestError=null;state.id=id;state.frames=[];state.metadata=null;state.time=0;state.result=null;state.playing=false;state.live=true;state.settled=false;orbitInitialized=false;dataChanged();
  const mission=await poll();if(mission)fillMissionForm(mission.request);$('history').value=id;
  const url=new URL(location.href);url.searchParams.set('mission',id);url.hash=page;history.replaceState({},'',url);
}
async function poll(){
  if(!state.id)return;const id=state.id;
  const [m,data]=await Promise.all([api(`/api/missions/${id}`),api(`/api/missions/${id}/frames?after=${state.frames.length}`)]);if(id!==state.id)return;
  const newMetadata=!state.metadata&&m.metadata,newResult=!state.result&&m.summary;
  state.metadata=m.metadata;state.frames.push(...data.frames);state.result=m.summary;state.busy=['starting','running'].includes(m.state);
  // A finished mission's record no longer changes: stop polling it once drained.
  state.settled=!state.busy&&!data.frames.length;
  if(data.frames.length||newMetadata||newResult){if(state.live&&data.frames.length)state.time=state.frames.at(-1).t;dataChanged();drawPlots();}
  text('missionTitle',m.request.name);
  const failed=m.state==='failed'||(m.summary&&!m.summary.success);
  text('flightStatus',m.summary?.success?'✓ LANDED / PASS':m.summary?.outcome==='LANDED'?'✕ LANDED / FAIL':m.summary?.outcome?(failed?'✕ ':'')+m.summary.outcome:m.state.toUpperCase());
  $('flightStatus').className=`status${failed?' fail':m.summary?.success?' pass':''}`;
  $('run').disabled=state.busy||state.recording||state.training;$('stop').disabled=!state.busy;$('export').disabled=state.busy||state.frames.length<2||state.recording;
  const profile=m.request.hardware_profile==='planned_8s'?'8S PLANNED':'6S LEGACY';const vanes=m.request.vane_model==='legacy'?'LEGACY VANES':'MOMENTUM VANES';text('footerProfile',`${profile} · ${vanes} · 3.104 kg${m.metadata?.physics_dt?' · '+fmt(1/m.metadata.physics_dt,0)+' Hz PHYSICS':''}`);
  const recordedPolicy=m.metadata?.policy;
  text('notice',`${profile} · ${m.request.battery.enabled?'LiPo coupled to EDF':'Ideal voltage, battery disabled'}${m.metadata?.hinge_layout==='radial_span_v1'?' · Radial hinges':' · ARCHIVE: OLD HINGE AXES'}${recordedPolicy?.step?' · Recorded PPO '+fmt(recordedPolicy.step/1e6,2)+'M / '+recordedPolicy.action_mode:''}${recordedPolicy?.diagnostic_checkpoint_override||m.metadata?.experimental_policy?' · Experimental replay; full-task qualification pending':''} · Hardware calibration pending${m.metadata?.initial_conditions_outside_training?' · Outside checkpoint training bounds':''}`);
  message(state.requestError??m.error??(state.busy?`${m.phase} · ${m.frames??0} samples received`:m.summary?`${m.summary.outcome} · impact ${fmt(m.summary.impact_speed,3)} m/s · pad error ${fmt(m.summary.pad_distance,3)} m`:'Mission loaded'),!!(state.requestError||m.error));
  $('jsonDownload').hidden=!state.frames.length;$('jsonDownload').href=`/api/missions/${id}/download`;
  $('videoDownload').hidden=!m.video;$('videoDownload').href=`/api/missions/${id}/video`;
  if(state.frames.length)updateTelemetry();
  return m;
}
// Vane physics each controller flies (mission_control/models.py): PID is the
// legacy-vane reference, PPO policies replay their momentum-bounded training
// plant, and only the convex controller offers the choice.
let convexVaneModel='momentum';
function syncVaneModel(){
  const controller=$('controller').value,select=$('vane_model');
  const fixed=controller==='pid'?'legacy':controller==='convex'?null:'momentum';
  select.value=fixed??convexVaneModel;select.disabled=!!fixed;
}
$('vane_model').onchange=()=>{if($('controller').value==='convex')convexVaneModel=$('vane_model').value;};
function fillMissionForm(request){
  for(const key of ['name','controller','hardware_profile','seed','duration_s'])$(key).value=request[key];
  if(request.controller==='convex')convexVaneModel=request.vane_model??'momentum';
  // Replays can name a retired policy; keep the next run on an available one.
  if(!$('controller').value)$('controller').value=config?.defaults?.controller??'pid';
  syncVaneModel();
  for(const key of ['position','velocity','attitude_deg','angular_rate_deg_s'])request[key].forEach((v,i)=>{$(`${key}_${i}`).value=v;});
  $('initial_motor_fraction').value=request.initial_motor_fraction*100;
  const selected=Array.isArray(request.disturbance)?request.disturbance:[request.disturbance];
  document.querySelectorAll('input[name="disturbance"]').forEach(input=>{input.checked=selected.includes(input.value);});
  $('battery_enabled').checked=request.battery.enabled;
  for(const key of ['capacity_ah','c_rating','max_current_a'])$(key).value=request.battery[key];
  $('initial_soc').value=request.battery.initial_soc*100;$('cell_resistance_ohm').value=request.battery.cell_resistance_ohm*1000;
  text('packLabel',request.hardware_profile==='planned_8s'?'8S / ESTIMATED':'6S / ESTIMATED');
  planner.setWaypoints(request.waypoints??[]);checklist.update();
}
function missionRequest(){
  const result={};for(const key of ['name','controller','hardware_profile','vane_model'])result[key]=$(key).value;
  result.disturbance=[...document.querySelectorAll('input[name="disturbance"]:checked')].map(input=>input.value);
  for(const key of ['seed','duration_s'])result[key]=Number($(key).value);
  for(const key of ['position','velocity','attitude_deg','angular_rate_deg_s'])result[key]=[0,1,2].map(i=>Number($(`${key}_${i}`).value));
  result.initial_motor_fraction=Number($('initial_motor_fraction').value)/100;
  result.waypoints=planner.getWaypoints();
  result.battery={enabled:$('battery_enabled').checked};for(const key of ['capacity_ah','c_rating','max_current_a'])result.battery[key]=Number($(key).value);
  result.battery.initial_soc=Number($('initial_soc').value)/100;result.battery.cell_resistance_ohm=Number($('cell_resistance_ohm').value)/1000;return result;
}
$('missionForm').addEventListener('submit',async e=>{e.preventDefault();state.requestError=null;$('run').disabled=true;message('Launching Isaac Sim…');try{const m=await api('/api/missions',missionRequest());showPage('flight');await refreshHistory();await selectMission(m.id);}catch(error){state.requestError=error.message;message(error.message,true);$('run').disabled=state.training||state.busy;}});
$('stop').onclick=async()=>{try{await api(`/api/missions/${state.id}/stop`,{});message('Stop requested; Isaac will finish the current control interval.');$('stop').disabled=true;}catch(e){message(e.message,true);}};
$('history').onchange=()=>{if($('history').value)selectMission($('history').value).catch(e=>message(e.message,true));};
$('play').onclick=()=>{if(!state.frames.length)return;state.live=false;if(state.time>=state.frames.at(-1).t)state.time=0;state.playing=!state.playing;};
$('timeline').oninput=()=>{state.live=false;state.playing=false;state.time=Number($('timeline').value);};
$('live').onclick=()=>{state.live=true;state.playing=false;state.time=state.frames.at(-1)?.t??0;};
$('controller').onchange=()=>{syncVaneModel();renderBoard();};
$('hardware_profile').onchange=()=>text('packLabel',$('hardware_profile').value==='planned_8s'?'8S / ESTIMATED':'6S / ESTIMATED');
$('hardwareButton').onclick=()=>$('hardwareDialog').showModal();$('closeHardware').onclick=()=>$('hardwareDialog').close();
document.addEventListener('keydown',e=>{
  if(e.target.closest('input,select,textarea,dialog')||e.ctrlKey||e.metaKey||e.altKey)return;
  if(/^[1-4]$/.test(e.key)){showPage(pages[Number(e.key)-1]);return;}
  if(!state.frames.length)return;
  if(e.code==='Space'){e.preventDefault();$('play').click();}
  else if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();state.live=false;state.playing=false;state.time=THREE.MathUtils.clamp(state.time+(e.key==='ArrowLeft'?-1:1)*(e.shiftKey?5:.5),0,state.frames.at(-1).t);}
});

// Video uses the same four camera renders and exact telemetry as the replay.
const captureCanvas=document.createElement('canvas');captureCanvas.width=1600;captureCanvas.height=1000;
const webcastCapture=document.createElement('canvas');
function captureFrame(){
  renderViews(1160,560);const ctx=captureCanvas.getContext('2d'),sample=sampleAt(state.time),f=sample?.a;if(!f)return;
  ctx.fillStyle='#000';ctx.fillRect(0,0,1600,1000);ctx.fillStyle='#f4f6f7';ctx.font='20px Bahnschrift, Arial';ctx.fillText('EDF / MISSION CONTROL',26,38);
  ctx.textAlign='center';ctx.font='30px Bahnschrift, Consolas';ctx.fillText(clock(state.time),600,40);ctx.textAlign='left';
  ctx.drawImage($('scene'),20,58,1160,560);
  ctx.save();ctx.translate(20,58);drawBottomLabels(ctx,1160,560,sampleAt(state.time));ctx.restore();
  ctx.fillStyle='#b9c2c8';ctx.font='11px Bahnschrift, Arial';const cameraLabels=[[34,80,'CAM 01 / ORBIT'],[800,80,'CAM 02 / GROUND'],[800,267,'CAM 03 / OVERHEAD'],[800,453,'CAM 04 / FINS BOTTOM']];cameraLabels.forEach(([x,y,label])=>ctx.fillText(label,x,y));
  drawWebcastBand(webcastCapture,1160,118);ctx.drawImage(webcastCapture,20,620,1160,118);
  flightCharts.draw(state.time);
  flightCharts.canvases().forEach((canvas,i)=>{const x=20+i*292,y=760;ctx.fillStyle='#b9c2c8';ctx.font='10px Bahnschrift, Arial';ctx.fillText(canvas.getAttribute('aria-label').split(' time history')[0],x,y);ctx.drawImage(canvas,x,y+6,284,150);});
  let y=85;const line=(label,value,color='#f4f6f7')=>{ctx.fillStyle='#7c878f';ctx.font='11px Bahnschrift, Arial';ctx.fillText(label,1210,y);ctx.fillStyle=color;ctx.font='19px Bahnschrift, Consolas';ctx.fillText(value,1210,y+23);y+=52;};
  line('THRUST / THROTTLE',`${fmt(f.thrust_n)} N / ${fmt(f.throttle*100)} %`);line('ROTOR',`${fmt(f.rotor_rpm,0)} rpm`);
  const att=attitude(f.quaternion);line('ROLL / PITCH / TILT · °',`${fmt(att.roll,1)} / ${fmt(att.pitch,1)} / ${fmt(att.tilt,1)}`);
  ctx.fillStyle='#7c878f';ctx.font='11px Bahnschrift, Arial';ctx.fillText('FIN       CMD °        ACT °',1210,y);y+=24;
  finNames.forEach((n,i)=>{ctx.font='15px Consolas';ctx.fillStyle='#f4f6f7';ctx.fillText(`${n.padEnd(6)} ${fmt(f.fin_commands[i]*deg,2).padStart(7)}   ${fmt(THREE.MathUtils.lerp(f.fin_angles[i],sample.b.fin_angles[i],sample.alpha)*deg,2).padStart(7)}`,1210,y);y+=23;});y+=12;
  line('GYRO P / Q / R · °/s',f.gyro.map(v=>fmt(v*deg,1)).join(' / '));
  if(f.rotation){
    line('PEAK P / Q / R · °/s',f.rotation.peak_rate_deg_s.map(v=>fmt(v,0)).join(' / '));
    line('ABOVE SOFT LIMITS · seconds',f.rotation.time_above_limit_s.map(v=>fmt(v,2)).join(' / '));
  }
  line('LIPO / SOC',f.battery?`${state.metadata.battery_model.cells}S / ${fmt(f.battery.soc*100,1)} %`:'DISABLED');
  line('BUS / CURRENT',f.battery?`${fmt(f.battery.voltage_v,2)} V / ${fmt(f.battery.current_a,1)} A`:'IDEAL BUS');
  line('ENERGY / PROPULSIVE ΔV',`${f.battery?fmt(f.battery.energy_wh,2)+' Wh':'—'} / ${fmt(f.propulsive_delta_v_m_s,2)} m/s`);
  const ended=state.time>=state.frames.at(-1).t;line('FLIGHT STATE',ended&&state.result?`${state.result.outcome}${state.result.success?' / PASS':' / FAIL'}`:f.control_phase==='POST_TOUCHDOWN_DISARM'?'MOTOR OFF / SETTLING':contactNames[f.contact]);
  line('IMPACT / PAD ERROR',`${fmt(f.impact_speed,3)} m/s / ${fmt(f.pad_distance,3)} m`);
  ctx.fillStyle='#b9c2c8';ctx.font='12px Bahnschrift, Arial';ctx.fillText(`${state.metadata?.request.name??'Landing'} · ${state.metadata?.policy.controller??''} · ${state.metadata?.hardware_profile??''} · Isaac PhysX / actual CAD / WebGL cameras`,26,960);
  ctx.fillStyle='#d6b36a';ctx.font='11px Bahnschrift, Arial';ctx.fillText('Planned parts; pack and aerodynamic parameters estimated. No body-rate command: controller outputs fin angles and throttle.',26,982);
  return captureCanvas;
}
async function recordVideo(){
  if(state.recording||state.frames.length<2)return;
  if(!window.MediaRecorder||!MediaRecorder.isTypeSupported('video/webm;codecs=vp9'))throw new Error('This browser needs WebM VP9 recording support. Use Chrome or Edge.');
  showPage('flight');await new Promise(requestAnimationFrame);
  const mid=state.id;state.recording=true;state.playing=false;state.live=false;state.time=0;$('export').disabled=true;$('run').disabled=true;$('history').disabled=true;
  const chunks=[],stream=captureCanvas.captureStream(30),recorder=new MediaRecorder(stream,{mimeType:'video/webm;codecs=vp9',videoBitsPerSecond:6500000});
  recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data);};
  const ended=new Promise((resolve,reject)=>{recorder.onstop=resolve;recorder.onerror=reject;});
  try{captureFrame();recorder.start(1000);const start=performance.now(),duration=state.frames.at(-1).t;
    await new Promise(resolve=>{function frame(now){state.time=Math.min(duration,Math.max(0,(now-start)/1000-.5));captureFrame();updateTelemetry();message(`Recording cameras + telemetry · ${fmt(state.time,1)} / ${fmt(duration,1)} s`);if((now-start)/1000<duration+1.2)requestAnimationFrame(frame);else resolve();}requestAnimationFrame(frame);});
    recorder.stop();await ended;const blob=new Blob(chunks,{type:'video/webm'});const response=await fetch(`/api/missions/${mid}/video`,{method:'POST',headers:{'X-Mission-Control':'local','Content-Type':'video/webm'},body:blob});if(!response.ok)throw new Error('Video save failed');
    $('videoDownload').href=`/api/missions/${mid}/video`;$('videoDownload').hidden=false;message('Video saved with synchronized cameras and telemetry.');return await response.json();
  }finally{if(recorder.state==='recording')recorder.stop();stream.getTracks().forEach(t=>t.stop());state.recording=false;invalidate();$('export').disabled=false;$('run').disabled=state.busy||state.training;$('history').disabled=false;}
}
$('export').onclick=()=>recordVideo().catch(e=>message(e.message,true));
window.missionControl={state,seek(t){seekTo(t);renderViews();updateTelemetry();drawPlots();},recordVideo,captureFrame,selectMission,geometry,rotor};
// Render on demand: the four cameras redraw only when replay time, data, the
// orbit camera or the layout changed, and the 2D instruments (≤12 Hz) only
// when anything they show changed. A paused replay costs no GPU or DOM work.
let last=performance.now(),lastUi=0,sceneKey='',uiKey='';
const resized=new ResizeObserver(invalidate);resized.observe(document.body);resized.observe($('views'));
function animate(now){
  const elapsed=(now-last)/1000;last=now;
  if(state.playing&&!state.recording&&state.frames.length){state.time=Math.min(state.frames.at(-1).t,state.time+elapsed*Number($('speed').value));if(state.time>=state.frames.at(-1).t)state.playing=false;}
  if(!state.recording&&page==='flight'){
    const key=`${state.time}|${dataVersion}`;
    if(sceneDirty||key!==sceneKey){sceneDirty=false;sceneKey=key;renderViews();}
  }
  if(now-lastUi>80){
    const key=`${state.time}|${uiVersion}|${state.playing}|${state.live}|${charts.hoverTime()}|${flightCharts.hoverTime()}`;
    if(key!==uiKey){lastUi=now;uiKey=key;try{updateTelemetry();drawPlots();}catch(error){console.error(error);}}
  }
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
try{
  config=await api('/api/config');updateTraining(config);text('hardwareStatus',config.hardware.status);
  $('controller').replaceChildren(...Object.entries(config.policies).map(([key,name])=>new Option(name,key)));$('controller').value=config.defaults.controller;syncVaneModel();checklist.update();
  for(const part of config.hardware.parts){const el=document.createElement('div');el.className='hardware-part';const heading=document.createElement('h3');heading.textContent=part.part;const body=document.createElement('div');const name=document.createElement('strong');name.textContent=part.name;const spec=document.createElement('p');spec.textContent=part.spec;const basis=document.createElement('p');basis.textContent=part.basis;body.append(name,spec,basis);if(part.source){const a=document.createElement('a');a.href=part.source;a.target='_blank';a.rel='noreferrer';a.textContent='MANUFACTURER SOURCE ↗';body.append(a);}el.append(heading,body);$('hardwareParts').append(el);}
  if(page==='models')pageShown.models();
  const missions=await refreshHistory(),requested=new URLSearchParams(location.search).get('mission');const selected=requested??config.active??missions.find(m=>m.hinge_layout==='radial_span_v1'&&m.summary?.success)?.id??missions.find(m=>m.hinge_layout==='radial_span_v1'&&m.state==='complete')?.id??missions.find(m=>m.state==='complete')?.id;
  if(selected){await selectMission(selected);state.live=false;state.time=0;}
}catch(e){state.connected=false;text('connection','SERVICE ERROR');message(e.message,true);renderBoard();}
// Background tabs poll nothing; returning to the tab catches up at once.
const pollMission=()=>{if(!state.recording&&!state.settled)poll().catch(e=>message(e.message,true));};
const pollConfig=async()=>{try{updateTraining(await api('/api/config'));}catch{state.connected=false;renderBoard();}};
setInterval(()=>{if(!document.hidden)pollMission();},1200);
setInterval(()=>{if(!document.hidden)pollConfig();},5000);
setInterval(()=>{if(!document.hidden&&page==='models')pageShown.models();},15000);
document.addEventListener('visibilitychange',()=>{if(!document.hidden){pollMission();pollConfig();if(page==='models')pageShown.models();}});
