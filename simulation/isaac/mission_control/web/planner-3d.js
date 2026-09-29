import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {sampleRouteLegs,waypointColors,waypointLabel,waypointTypes,escapeHtml} from './flight-plan.js';
import {createRouteRibbon} from './route-line.js';

// Z-up editor: axis drags use a camera-facing plane containing the chosen axis.
// Projecting onto that axis keeps the other two coordinates unchanged.
export function axisDragPlane(axis,origin,direction){
  const normal=direction.clone().addScaledVector(axis,-direction.dot(axis));
  if(normal.lengthSq()<1e-6)return null; // An end-on axis has no usable screen displacement.
  return new THREE.Plane().setFromNormalAndCoplanarPoint(normal.normalize(),origin);
}
export function editableAxes(id,waypoints){
  if(id==='start')return [true,true,true];
  if(typeof id==='string')return [true,true,false];
  const w=waypoints[id];
  return !w||w.type==='land'?[false,false,false]:['takeoff','descent'].includes(w.type)?[false,false,true]:[true,true,true];
}
// Leg of the drawn route that ends at a selection: legs run start → each
// non-landing step → landing pad (sampleRouteLegs).
export function legOfSelection(id,waypoints){
  if(typeof id!=='number'||!waypoints[id])return -1;
  const w=waypoints[id];
  return w.type==='land'?waypoints.filter(x=>x.type!=='land').length:waypoints.slice(0,id).filter(x=>x.type!=='land').length;
}
// Which step types the plan can still take (one takeoff, one landing, 12 steps).
export function addableTypes(waypoints){
  return Object.fromEntries(Object.keys(waypointTypes).map(type=>[type,waypoints.length<12&&!(['takeoff','land'].includes(type)&&waypoints.some(w=>w.type===type))]));
}
export const snapValue=(value,step)=>step>0?Math.round(value/step)*step:value;

const PAD_COLOR='#48dba2',START_COLOR='#ffffff';
const coarse=()=>matchMedia('(pointer: coarse)').matches;

export function createPlanner3D(host,{read,move,select,context,add,remove}){
  host.innerHTML=`<div class="scene-viewport">
      <canvas tabindex="0" aria-label="3D waypoint editor. Select a marker and drag it, or use the arrow keys; Page Up and Page Down change altitude. Right-click or long-press for step details."></canvas>
      <div class="scene-top">
        <div class="scene-label">3D ROUTE <span>WORLD XYZ · Z UP · METRES</span></div>
        <div class="scene-views" role="group" aria-label="Camera"><button type="button" data-view="fit" title="Frame the whole route">Fit</button><button type="button" data-view="iso" title="Oblique view">3D</button><button type="button" data-view="top" title="Look down the Z axis">Top</button><button type="button" data-view="side" title="Look along +Y">Side</button><button type="button" class="expand-route" title="Full screen" aria-label="Full screen">⛶</button></div>
      </div>
      <div class="scene-dock" role="toolbar" aria-label="Route editing">
        <button type="button" data-mode="select" aria-pressed="true" title="Select and drag markers">Select</button>
        <button type="button" data-mode="add" aria-pressed="false" aria-haspopup="true" title="Place a new step in the scene">＋ Add</button>
        <span class="dock-divider" aria-hidden="true"></span>
        <button type="button" data-toggle="snap" aria-pressed="true" title="Round moves to 0.5 m">Snap 0.5</button>
        <button type="button" data-toggle="labels" aria-pressed="false" title="Show every step name (otherwise only the selected step)">Names</button>
        <button type="button" data-toggle="corridor" aria-pressed="true" title="Show the convex route corridor">Corridor</button>
        <button type="button" data-toggle="legend" aria-pressed="false" title="Show the colour key">Key</button>
      </div>
      <div class="scene-add-types" role="group" aria-label="Step type to place" hidden>${Object.entries(waypointTypes).map(([type,name])=>`<button type="button" data-place="${type}" style="--step-color:${waypointColors[type]}"><i></i>${name}</button>`).join('')}</div>
      <div class="scene-legend" hidden><span><i class="legend-start"></i>Start</span>${Object.entries(waypointTypes).map(([type,name])=>`<span><i style="background:${waypointColors[type]}"></i>${name}</span>`).join('')}<span><i class="legend-pad"></i>Pad</span><span><i class="legend-route"></i>Route · direction</span><span><i class="legend-corridor"></i>Corridor</span></div>
      <div class="scene-toast" role="status" aria-live="polite"></div>
      <div class="waypoint-context" role="dialog" aria-label="Waypoint actions" hidden></div>
    </div>
    <section class="scene-inspector" aria-label="Selected marker" aria-live="polite"></section>
    <p class="scene-hint"></p>`;
  const viewport=host.querySelector('.scene-viewport'),canvas=host.querySelector('canvas'),menu=host.querySelector('.waypoint-context');
  const inspector=host.querySelector('.scene-inspector'),hint=host.querySelector('.scene-hint'),toast=host.querySelector('.scene-toast');
  const addTypes=host.querySelector('.scene-add-types'),legend=host.querySelector('.scene-legend');
  const ui={mode:'select',placeType:null,snap:true,labels:false,corridor:true,hover:null};
  const renderer=new THREE.WebGLRenderer({canvas,antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor('#060b11');
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(45,1,.05,1500);camera.up.set(0,0,1);camera.position.set(18,-24,20);
  const controls=new OrbitControls(camera,canvas);controls.target.set(0,0,5);controls.minDistance=2;controls.maxDistance=600;
  controls.touches={ONE:THREE.TOUCH.ROTATE,TWO:THREE.TOUCH.DOLLY_PAN};
  // 1 m minor and 10 m major grid on the ground plane; axis tags mark +X / +Y.
  const minor=new THREE.GridHelper(200,200,0x121c26,0x121c26),major=new THREE.GridHelper(200,20,0x2f445a,0x223344);
  [minor,major].forEach(g=>{g.rotation.x=Math.PI/2;g.material.transparent=true;g.material.opacity=g===minor?.5:.9;g.material.depthWrite=false;scene.add(g);});
  const axes=new THREE.AxesHelper(4);axes.material.transparent=true;axes.material.opacity=.7;scene.add(axes);
  const textures=new Map();
  // Canvas textures are cached: drags redraw the scene every frame.
  function texture(key,paint,width,height){
    if(!textures.has(key)){const c=document.createElement('canvas');c.width=width;c.height=height;paint(c.getContext('2d'),c);const t=new THREE.CanvasTexture(c);t.colorSpace=THREE.SRGBColorSpace;textures.set(key,{t,aspect:width/height});}
    return textures.get(key);
  }
  function sprite(key,paint,width,height,scale,parent){
    const {t,aspect}=texture(key,paint,width,height);
    const s=new THREE.Sprite(new THREE.SpriteMaterial({map:t,depthTest:false,sizeAttenuation:false,transparent:true}));
    s.scale.set(scale*aspect,scale,1);s.renderOrder=12;s.userData.cachedMap=true;parent.add(s);return s;
  }
  const axisTag=(text,color,position)=>{const s=sprite(`axis:${text}`,x=>{x.fillStyle=color;x.font='bold 30px sans-serif';x.fillText(text,6,42);},64,64,.045,scene);s.position.set(...position);};
  axisTag('+X','#ff7070',[4.6,0,0]);axisTag('+Y','#7fe36b',[0,4.6,0]);
  let content=new THREE.Group(),handles=[],drag=null,press=null,frame=0;scene.add(content);
  const ribbon=createRouteRibbon({width:3.2,ground:true});scene.add(ribbon.group);
  const ghost=new THREE.Group();ghost.visible=false;scene.add(ghost);
  const ghostBall=new THREE.Mesh(new THREE.SphereGeometry(.3,16,12),new THREE.MeshBasicMaterial({color:0xffffff,transparent:true,opacity:.55,depthTest:false}));ghostBall.renderOrder=9;ghost.add(ghostBall);
  const ghostLine=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3(0,0,-1)]),new THREE.LineDashedMaterial({color:0xffffff,dashSize:.25,gapSize:.2,transparent:true,opacity:.6}));ghost.add(ghostLine);
  const gizmo=new THREE.Group(),axisHandles=[];scene.add(gizmo);
  const directions=[new THREE.Vector3(1,0,0),new THREE.Vector3(0,1,0),new THREE.Vector3(0,0,1)],axisColors=['#ff5d5d','#70df58','#5aa2ff'];
  directions.forEach((direction,i)=>{
    const arrow=new THREE.ArrowHelper(direction,new THREE.Vector3(),1,axisColors[i],.24,.14);
    // Wide invisible shafts make the arrows easy to grab, fingers included.
    // They start clear of the marker, whose body is the free (XY) drag handle.
    const shaft=new THREE.Mesh(new THREE.CylinderGeometry(.12,.12,.8,10),new THREE.MeshBasicMaterial({visible:false}));
    shaft.position.y=.7;
    arrow.add(shaft);[shaft,arrow.cone].forEach(o=>{o.userData.axis=i;axisHandles.push(o);});
    arrow.traverse(o=>{if(o.material){o.material.depthTest=false;o.material.depthWrite=false;}o.renderOrder=20;});
    const tag=sprite(`gizmo:${i}`,ctx=>{ctx.fillStyle=axisColors[i];ctx.font='bold 48px sans-serif';ctx.fillText('XYZ'[i],10,50);},64,64,.2,arrow);
    tag.material.sizeAttenuation=true;tag.position.y=1.2;tag.scale.set(.24,.24,1);tag.renderOrder=21;gizmo.add(arrow);
  });

  const positionOf=(id,{initial,waypoints,pads}=read())=>id==='start'?initial.position:typeof id==='string'?pads[Number(id.split(':')[1])]?.position:waypoints[id]?.position;
  const markerPosition=(id,state)=>{const p=positionOf(id,state);if(!p)return null;return typeof id==='number'&&state.waypoints[id].type==='land'?[p[0],p[1],.4]:typeof id==='string'&&id.startsWith('pad:')?[p[0],p[1],.08]:p;};
  const colorOf=(id,{waypoints})=>id==='start'?START_COLOR:typeof id==='string'?PAD_COLOR:waypointColors[waypoints[id]?.type]??'#fff';
  function titleOf(id,{waypoints,pads}){
    if(id==='start')return 'Start';
    if(typeof id==='string')return pads[Number(id.split(':')[1])]?.name??'Pad';
    return waypointLabel(waypoints[id],id);
  }
  function closeMenu(){menu.hidden=true;menu.replaceChildren();}
  function say(text){toast.textContent=text;toast.classList.toggle('shown',!!text);clearTimeout(say.timer);if(text)say.timer=setTimeout(()=>toast.classList.remove('shown'),2200);}

  function updateGizmo(){
    const state=read(),position=positionOf(state.selected,state);
    gizmo.visible=!!position&&ui.mode==='select';if(!gizmo.visible)return;
    gizmo.position.set(...position);
    const enabled=editableAxes(state.selected,state.waypoints);
    gizmo.children.forEach((arrow,i)=>{arrow.visible=enabled[i];});
    const pixels=coarse()?120:90;
    const scale=camera.position.distanceTo(gizmo.position)*2*Math.tan(THREE.MathUtils.degToRad(camera.fov/2))*pixels/Math.max(1,canvas.clientHeight);
    gizmo.scale.setScalar(scale);
  }
  const ray=new THREE.Raycaster(),pointer=new THREE.Vector2();
  function render(){
    frame=0;if(!canvas.clientWidth||!canvas.clientHeight)return;
    renderer.setSize(canvas.clientWidth,canvas.clientHeight,false);camera.aspect=canvas.clientWidth/canvas.clientHeight;camera.updateProjectionMatrix();
    ribbon.setResolution(canvas.clientWidth,canvas.clientHeight);updateGizmo();scalePicks();renderer.render(scene,camera);
  }
  const schedule=()=>{if(!frame)frame=requestAnimationFrame(render);};
  controls.addEventListener('change',schedule);

  const measure=document.createElement('canvas').getContext('2d');
  function nameTag(text,color,position,below){
    measure.font='600 26px sans-serif';const width=Math.min(720,Math.ceil(measure.measureText(text).width)+30);
    const s=sprite(`name:${text}:${color}`,x=>{x.fillStyle='#07111df0';x.beginPath();x.roundRect(1,1,width-2,44,10);x.fill();
      x.fillStyle=color;x.fillRect(1,8,4,30);x.font='600 26px sans-serif';x.fillStyle='#eaf2ff';x.fillText(text,14,31,width-24);},width,46,.034,content);
    s.position.set(...position);s.center.set(.5,below?1.9:-.75);return s;
  }
  // Numbered badge on every marker: readable at any zoom, and far less
  // clutter than a name per step. Names show for the selection (or all).
  function badge(text,color,position,selected){
    const s=sprite(`badge:${text}:${color}:${selected}`,x=>{x.beginPath();x.arc(32,32,27,0,Math.PI*2);x.fillStyle=selected?'#ffffff':color;x.fill();x.lineWidth=4;x.strokeStyle='#060b11';x.stroke();
      x.fillStyle='#071119';x.font='bold 28px sans-serif';x.textAlign='center';x.textBaseline='middle';x.fillText(text,32,34);},64,64,selected?.052:.042,content);
    s.position.set(...position);s.center.set(.5,-.35);return s;
  }
  function marker(position,color,id,title,number,selected,hovered){
    const radius=(id==='start'?.34:.28)*(selected?1.3:hovered?1.15:1);
    const mesh=new THREE.Mesh(id==='start'?new THREE.OctahedronGeometry(radius):new THREE.SphereGeometry(radius,20,14),new THREE.MeshBasicMaterial({color,depthTest:false}));
    mesh.position.set(...position);mesh.userData.id=id;mesh.renderOrder=5;content.add(mesh);handles.push(mesh);
    // Invisible pick sphere: markers stay easy to hit however far the camera is.
    const pick=new THREE.Mesh(new THREE.SphereGeometry(1,8,6),new THREE.MeshBasicMaterial({visible:false}));pick.userData.id=id;pick.userData.pick=true;pick.position.copy(mesh.position);content.add(pick);handles.push(pick);
    if(selected||hovered){const ring=new THREE.Mesh(new THREE.TorusGeometry(radius*1.9,.045,8,40),new THREE.MeshBasicMaterial({color:selected?0xffffff:color,transparent:true,opacity:selected?1:.6,depthTest:false}));ring.position.copy(mesh.position);ring.renderOrder=6;ring.onBeforeRender=()=>ring.quaternion.copy(camera.quaternion);content.add(ring);}
    const tagged=typeof id==='string'&&id.startsWith('pad:');
    // Badges and names are hit targets too: they are what people aim at.
    const tags=[number!=null&&badge(number,color,position,selected),(ui.labels||selected||hovered||tagged)&&nameTag(title,color,position,tagged)];
    tags.forEach(t=>{if(t){t.userData.id=id;handles.push(t);}});
  }
  // Dashed plumb line and ground shadow make each marker's altitude readable from any angle.
  function dropLine(position,color){
    if(position[2]<.2)return;
    const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...position),new THREE.Vector3(position[0],position[1],.01)]),new THREE.LineDashedMaterial({color,dashSize:.25,gapSize:.2,transparent:true,opacity:.5}));
    line.computeLineDistances();content.add(line);
    const shadow=new THREE.Mesh(new THREE.CircleGeometry(.22,20),new THREE.MeshBasicMaterial({color,transparent:true,opacity:.3,depthWrite:false}));shadow.position.set(position[0],position[1],.015);content.add(shadow);
  }
  function scalePicks(){
    const unit=2*Math.tan(THREE.MathUtils.degToRad(camera.fov/2))/Math.max(1,canvas.clientHeight),radius=coarse()?26:16;
    handles.forEach(h=>{if(h.userData.pick)h.scale.setScalar(Math.max(.35,camera.position.distanceTo(h.position)*unit*radius));});
  }
  function draw(){
    content.traverse(o=>{o.geometry?.dispose();if(!o.userData.cachedMap)o.material?.map?.dispose();o.material?.dispose();});scene.remove(content);content=new THREE.Group();scene.add(content);handles=[];
    const state=read(),{initial,waypoints,pads,corridor,convex,disturbances,selected}=state;
    if(disturbances?.selected.includes('wind')){
      const vector=new THREE.Vector3(...disturbances.settings.wind.steady_vector),speed=vector.length();
      if(speed>1e-6){
        const origin=new THREE.Vector3(...initial.position).add(new THREE.Vector3(0,0,1));
        content.add(new THREE.ArrowHelper(vector.normalize(),origin,Math.min(8,Math.max(1,speed)),0xe5ad48,.5,.25));
        nameTag(`WIND ${speed.toFixed(1)} m/s`,'#e5ad48',origin.toArray(),false);
      }
    }
    const target=pads[waypoints.at(-1)?.pad??0]?.position??[0,0,0];
    const route=waypoints.filter(w=>w.type!=='land'),legs=sampleRouteLegs(initial.position,waypoints,target,{convex});
    // Each leg blends from the step it leaves to the step it flies to; the
    // landing leg ends in the pad colour.
    const colors=legs.map((_,i)=>[i?waypointColors[route[i-1].type]:START_COLOR,route[i]?waypointColors[route[i].type]:PAD_COLOR]);
    const highlight=legOfSelection(selected,waypoints);
    ribbon.set(legs,colors,{highlight});
    if(convex&&ui.corridor&&waypoints.length)legs.forEach((leg,i)=>{
      const width=(route[i]??waypoints.find(w=>w.type==='land'))?.corridor_m??corridor,points=leg.map(p=>new THREE.Vector3(...p));
      if(!(width>0)||points[0].distanceTo(points.at(-1))<1e-6)return;
      const curve=new THREE.Curve();curve.getPoint=t=>{const f=t*(points.length-1),n=Math.min(points.length-2,Math.floor(f));return points[n].clone().lerp(points[n+1],f-n);};
      // Front faces only and a faint fill: overlapping double-sided tubes
      // stacked into opaque blue bands that hid the route itself.
      const tube=new THREE.Mesh(new THREE.TubeGeometry(curve,64,width,14,false),new THREE.MeshBasicMaterial({color:0x4a9af0,transparent:true,opacity:highlight===i?.12:.05,depthWrite:false}));content.add(tube);
    });
    dropLine(initial.position,START_COLOR);
    marker(initial.position,START_COLOR,'start',`START · ${initial.position[2].toFixed(1)} m`,'S',selected==='start',ui.hover==='start');
    waypoints.forEach((w,i)=>{
      const position=markerPosition(i,state);
      if(w.type!=='land')dropLine(position,waypointColors[w.type]);
      marker(position,waypointColors[w.type],i,w.type==='land'?waypointLabel(w,i):`${waypointLabel(w,i)} · ${w.position[2].toFixed(1)} m`,String(i+1),selected===i,ui.hover===i);
      // Capture sphere of the selected step: arrival happens inside it.
      if(selected===i&&w.type!=='land'){
        const sphere=new THREE.Mesh(new THREE.SphereGeometry(w.radius_m,24,16),new THREE.MeshBasicMaterial({color:waypointColors[w.type],wireframe:true,transparent:true,opacity:.22,depthWrite:false}));
        sphere.position.set(...w.position);content.add(sphere);
      }
    });
    pads.forEach((p,i)=>{
      const id=`pad:${i}`,on=selected===id;
      const fill=new THREE.Mesh(new THREE.CircleGeometry(1.25,48),new THREE.MeshBasicMaterial({color:0x48dba2,transparent:true,opacity:on?.28:.12,depthWrite:false}));fill.position.set(p.position[0],p.position[1],.012);content.add(fill);
      const disc=new THREE.Mesh(new THREE.RingGeometry(1.1,1.25,48),new THREE.MeshBasicMaterial({color:0x48dba2,side:THREE.DoubleSide}));disc.position.set(p.position[0],p.position[1],.02);content.add(disc);
      const cross=new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints([[-.45,0],[.45,0],[0,-.45],[0,.45]].map(([x,y])=>new THREE.Vector3(p.position[0]+x,p.position[1]+y,.025))),new THREE.LineBasicMaterial({color:0x48dba2}));content.add(cross);
      marker([p.position[0],p.position[1],.08],PAD_COLOR,id,p.name,null,on,ui.hover===id);
    });
    drawInspector(state);updateHint();schedule();
  }

  // ---- inspector: the selection's position with steppers and actions ----
  function drawInspector(state=read()){
    const {selected,waypoints}=state,position=positionOf(selected,state);
    inspector.classList.toggle('empty',!position);
    if(!position){
      inspector.innerHTML=`<p class="inspector-empty"><b>Nothing selected.</b> ${coarse()?'Tap':'Click'} a marker or a numbered badge to edit it, or use <b>＋ Add</b> to place a step.</p>`;return;
    }
    const enabled=editableAxes(selected,waypoints),color=colorOf(selected,state),step=ui.snap?.5:.1;
    const w=typeof selected==='number'?waypoints[selected]:null;
    const kind=selected==='start'?'START STATE':typeof selected==='string'?'LANDING PAD':`STEP ${selected+1} · ${waypointTypes[w.type].toUpperCase()}`;
    const note=w?.type==='land'?'Follows its pad: select the pad to move it.':w&&['takeoff','descent'].includes(w.type)?'Vertical leg: X/Y follow the previous point.':'';
    const count=waypoints.length;
    inspector.innerHTML=`<div class="inspector-head" style="--step-color:${color}"><i></i><div><small>${kind}</small><b>${escapeHtml(titleOf(selected,state))}</b></div><button type="button" data-inspect="close" aria-label="Clear selection" title="Clear selection (Esc)">×</button></div>
      <div class="inspector-axes">${['X','Y','Z'].map((name,a)=>`<div class="inspector-axis axis-${name.toLowerCase()}${enabled[a]?'':' locked'}"><span>${name}</span><button type="button" data-nudge="${a}" data-sign="-1" ${enabled[a]?'':'disabled'} aria-label="${name} minus ${step} m">−</button><output>${Number(position[a]).toFixed(1)}</output><button type="button" data-nudge="${a}" data-sign="1" ${enabled[a]?'':'disabled'} aria-label="${name} plus ${step} m">+</button></div>`).join('')}</div>
      ${note?`<p class="inspector-note">${note}</p>`:''}
      <div class="inspector-actions">${typeof selected==='number'?`<button type="button" data-inspect="prev" ${selected<=0?'disabled':''} aria-label="Previous step" title="Previous step ([)">‹</button><button type="button" data-inspect="next" ${selected>=count-1?'disabled':''} aria-label="Next step" title="Next step (])">›</button>`:''}
        <button type="button" data-inspect="details">${typeof selected==='number'?'Details':'Info'}</button>${typeof selected==='number'&&remove?'<button type="button" class="danger" data-inspect="remove" title="Remove step (Delete)">Remove</button>':''}</div>`;
  }
  function nudge(axis,sign){
    const state=read(),id=state.selected,position=positionOf(id,state);
    if(!position||!editableAxes(id,state.waypoints)[axis])return;
    const step=ui.snap?.5:.1,next=[...position];next[axis]=snapValue(position[axis]+sign*step,step);
    move(id,next,axis);
  }
  inspector.addEventListener('click',e=>{
    const button=e.target.closest('button');if(!button)return;
    if(button.dataset.nudge!=null){nudge(Number(button.dataset.nudge),Number(button.dataset.sign));return;}
    const state=read(),id=state.selected;
    ({close:()=>{select(null);draw();},prev:()=>{select(id-1);draw();},next:()=>{select(id+1);draw();},
      details:()=>openMenu(id,null,null),remove:()=>{closeMenu();remove?.(id);}})[button.dataset.inspect]?.();
  });

  // ---- modes, toggles, hint ----
  function setMode(mode,type=null){
    ui.mode=mode;ui.placeType=mode==='add'?type:null;
    host.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===mode)));
    const allowed=addableTypes(read().waypoints);
    addTypes.querySelectorAll('[data-place]').forEach(b=>{b.disabled=!allowed[b.dataset.place];b.setAttribute('aria-pressed',String(b.dataset.place===ui.placeType));});
    addTypes.hidden=mode!=='add';viewport.classList.toggle('placing',mode==='add'&&!!ui.placeType);
    ghost.visible=false;closeMenu();updateHint();schedule();
  }
  function placementHeight(){
    const {waypoints,selected}=read();
    if(typeof selected==='number'&&waypoints[selected]&&waypoints[selected].type!=='land')return waypoints[selected].position[2];
    return waypoints.filter(w=>w.type!=='land').at(-1)?.position[2]??3;
  }
  function updateHint(){
    const touch=coarse(),state=read();
    hint.innerHTML=ui.mode==='add'?(ui.placeType?`<b>${touch?'Tap':'Click'}</b> in the scene to place a <b>${waypointTypes[ui.placeType].toLowerCase()}</b> at ${placementHeight().toFixed(1)} m (the selected step's height)${ui.placeType==='land'?'; landing always uses its pad':''} · <b>Esc</b> or <b>Select</b> to stop`:'Choose the type of step to place.')
      :touch?`<b>Tap</b> a marker to select · <b>drag</b> it to move · <b>long-press</b> for details · one finger orbits, two fingers pan and zoom`
      :`<b>Drag</b> a marker to move it (arrows lock one axis) · <b>double-click</b> or <b>right-click</b> for details · <b>arrows / PgUp PgDn</b> nudge · <b>drag</b> orbit · <b>right-drag</b> pan · <b>scroll</b> zoom`;
    if(ui.mode==='select'&&!state.waypoints.length)hint.innerHTML+=' · Empty route: use <b>＋ Add</b> or load a sample plan.';
  }
  host.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>setMode(b.dataset.mode,b.dataset.mode==='add'?ui.placeType:null));
  addTypes.querySelectorAll('[data-place]').forEach(b=>b.onclick=()=>{setMode('add',b.dataset.place);canvas.focus({preventScroll:true});});
  host.querySelectorAll('[data-toggle]').forEach(b=>b.onclick=()=>{
    const key=b.dataset.toggle,on=b.getAttribute('aria-pressed')!=='true';b.setAttribute('aria-pressed',String(on));
    if(key==='legend')legend.hidden=!on;else ui[key]=on;
    try{localStorage.setItem(`missionControl.planner3d.${key}`,on?'1':'0');}catch{}
    draw();
  });
  // Per-viewer display preferences; the editor works the same without storage.
  host.querySelectorAll('[data-toggle]').forEach(b=>{
    let stored=null;try{stored=localStorage.getItem(`missionControl.planner3d.${b.dataset.toggle}`);}catch{}
    const on=stored==null?(b.dataset.toggle==='legend'?viewport.clientWidth>=760:b.getAttribute('aria-pressed')==='true'):stored==='1';
    b.setAttribute('aria-pressed',String(on));if(b.dataset.toggle==='legend')legend.hidden=!on;else ui[b.dataset.toggle]=on;
  });

  // ---- context menu (right-click, long-press, Details) ----
  function openMenu(id,position,at){
    closeMenu();
    context?.({id,position,menu,close:closeMenu});
    if(!menu.childElementCount)return;
    menu.hidden=false;
    // Narrow viewports get a bottom sheet instead of a popover at the pointer.
    const sheet=viewport.clientWidth<560||!at;menu.classList.toggle('sheet',sheet);
    if(sheet){menu.style.left=menu.style.top='';}
    else{const rect=viewport.getBoundingClientRect();menu.style.left=`${Math.max(0,Math.min(at[0]-rect.left,viewport.clientWidth-menu.offsetWidth))}px`;menu.style.top=`${Math.max(0,Math.min(at[1]-rect.top,viewport.clientHeight-menu.offsetHeight))}px`;}
    if(!coarse())menu.querySelector('input,select,button')?.focus();
    schedule();
  }
  function groundPoint(height){return ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0,0,1),-Math.max(1,height)),new THREE.Vector3());}
  function contextAt(e){
    cast(e);const hit=pick(),id=hit?.object.userData.id;
    const position=groundPoint(placementHeight());
    if(id==null&&!position)return;
    if(id!=null){select(id);draw();}
    openMenu(id,position?.toArray().map(v=>snapValue(v,ui.snap?.5:0)),[e.clientX,e.clientY]);
  }

  // ---- pointer interaction ----
  function cast(e){const rect=canvas.getBoundingClientRect();pointer.set((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);ray.setFromCamera(pointer,camera);}
  const pick=()=>{const hits=ray.intersectObjects(handles,false);return hits.find(h=>!h.object.userData.pick)??hits[0];};
  function startAxisDrag(e,axisIndex,id){
    const axis=directions[axisIndex],origin=new THREE.Vector3(...positionOf(id));
    const plane=axisDragPlane(axis,origin,camera.getWorldDirection(new THREE.Vector3()));
    const anchor=plane&&ray.ray.intersectPlane(plane,new THREE.Vector3());
    if(!anchor)return false;
    drag={id,origin,anchor,plane,axis,axisIndex,moved:false};return true;
  }
  // Dragging a marker body moves it in its horizontal plane (the plane of
  // constant altitude); vertical-only steps move along Z instead.
  function startBodyDrag(id){
    const state=read(),enabled=editableAxes(id,state.waypoints),origin=new THREE.Vector3(...positionOf(id,state));
    if(enabled[0]&&enabled[1]){
      const plane=new THREE.Plane(new THREE.Vector3(0,0,1),-origin.z),anchor=ray.ray.intersectPlane(plane,new THREE.Vector3());
      // A grazing view makes the horizontal plane unusable: fall back to the view-facing axes.
      if(anchor&&Math.abs(ray.ray.direction.z)>.12){drag={id,origin,anchor,plane,axis:null,moved:false};return;}
    }
    if(enabled[2])startAxisDrag(null,2,id);
  }
  canvas.addEventListener('pointerdown',e=>{
    if(e.button===2){press={right:true,x:e.clientX,y:e.clientY};return;}
    if(e.button!==0||!e.isPrimary)return;
    closeMenu();cast(e);canvas.focus({preventScroll:true});
    const state=read();
    // A direct hit on a marker beats the gizmo arrows that start inside it.
    const direct=ray.intersectObjects(handles,false).find(h=>!h.object.userData.pick);
    const axisHit=!direct&&gizmo.visible?ray.intersectObjects(axisHandles.filter(o=>o.parent.visible),false)[0]:null;
    const hit=axisHit?null:direct??pick();
    press={x:e.clientX,y:e.clientY,id:hit?.object.userData.id,t:performance.now(),pointerId:e.pointerId};
    if(e.pointerType!=='mouse')press.timer=setTimeout(()=>{if(press&&!press.moved){const p=press;press=null;drag=null;controls.enabled=true;contextAt({clientX:p.x,clientY:p.y});navigator.vibrate?.(12);}},550);
    if(axisHit)startAxisDrag(e,axisHit.object.userData.axis,state.selected);
    else if(hit&&ui.mode==='select'){if(state.selected!==press.id){select(press.id);draw();}startBodyDrag(press.id);}
    else return; // empty space (or add mode): orbit, and a tap acts on release
    controls.enabled=false;try{canvas.setPointerCapture(e.pointerId);}catch{}e.stopImmediatePropagation();e.preventDefault();
  },true);
  canvas.addEventListener('pointermove',e=>{
    if(press&&!press.right&&Math.hypot(e.clientX-press.x,e.clientY-press.y)>6){press.moved=true;clearTimeout(press.timer);}
    if(drag){
      if(!press?.moved&&!drag.moved)return;
      cast(e);const hit=ray.ray.intersectPlane(drag.plane,new THREE.Vector3());if(!hit)return;
      const step=ui.snap?.5:0;drag.moved=true;
      if(drag.axis){const next=drag.origin.clone().addScaledVector(drag.axis,hit.sub(drag.anchor).dot(drag.axis)).toArray();next[drag.axisIndex]=snapValue(next[drag.axisIndex],step);move(drag.id,next,drag.axisIndex);}
      else{
        const delta=hit.sub(drag.anchor);if(delta.length()>camera.position.distanceTo(drag.origin)*3)return;
        const next=drag.origin.clone().add(delta).toArray().map((v,i)=>i<2?snapValue(v,step):v);
        move(drag.id,next,0);move(drag.id,next,1);
      }
      return;
    }
    if(e.pointerType==='mouse'&&!e.buttons)hoverAt(e);
  });
  function hoverAt(e){
    cast(e);
    if(ui.mode==='add'&&ui.placeType){
      const p=groundPoint(placementHeight());ghost.visible=!!p&&ui.placeType!=='land';
      if(ghost.visible){const step=ui.snap?.5:0;ghost.position.set(snapValue(p.x,step),snapValue(p.y,step),p.z);ghostBall.material.color.set(waypointColors[ui.placeType]);ghostLine.scale.set(1,1,p.z);ghostLine.computeLineDistances();}
      canvas.style.cursor='crosshair';schedule();return;
    }
    const axisHit=gizmo.visible&&ray.intersectObjects(axisHandles.filter(o=>o.parent.visible),false)[0];
    const id=axisHit?null:pick()?.object.userData.id??null;
    canvas.style.cursor=axisHit?'grab':id!=null?'pointer':'';
    if(id!==ui.hover){ui.hover=id;draw();}
  }
  canvas.addEventListener('pointerleave',()=>{if(ui.hover!=null){ui.hover=null;draw();}if(ghost.visible){ghost.visible=false;schedule();}});
  function end(e){
    const p=press;press=null;clearTimeout(p?.timer);
    const wasDrag=drag?.moved;drag=null;controls.enabled=true;
    if(e&&canvas.hasPointerCapture?.(e.pointerId))canvas.releasePointerCapture(e.pointerId);
    if(!p||p.right||p.moved||wasDrag||e?.type!=='pointerup')return;
    // A tap (no movement): place in add mode, otherwise select or clear.
    cast(e);
    if(ui.mode==='add'){
      if(!ui.placeType){say('Choose a step type first.');return;}
      if(!addableTypes(read().waypoints)[ui.placeType]){say(`The plan cannot take another ${waypointTypes[ui.placeType].toLowerCase()} step.`);return;}
      const point=groundPoint(placementHeight());if(!point&&ui.placeType!=='land')return;
      const step=ui.snap?.5:0,position=point?[snapValue(point.x,step),snapValue(point.y,step),point.z]:null;
      add?.(ui.placeType,position);say(`${waypointTypes[ui.placeType]} placed.`);
      if(!addableTypes(read().waypoints)[ui.placeType])setMode('select');
      return;
    }
    if(p.id==null&&read().selected!=null){select(null);draw();}
  }
  canvas.addEventListener('pointerup',end);canvas.addEventListener('pointercancel',end);
  canvas.addEventListener('lostpointercapture',e=>{if(drag)end(e);});
  canvas.addEventListener('dblclick',e=>{cast(e);const id=pick()?.object.userData.id;if(id!=null)openMenu(id,null,[e.clientX,e.clientY]);});
  canvas.addEventListener('contextmenu',e=>{
    e.preventDefault();const p=press;press=null;
    if(p?.right&&Math.hypot(e.clientX-p.x,e.clientY-p.y)>5)return; // that was a pan
    if(e.pointerType&&e.pointerType!=='mouse')return;               // long-press handles touch
    end();contextAt(e);
  });
  document.addEventListener('pointerdown',e=>{if(!menu.hidden&&!menu.contains(e.target)&&e.target!==canvas)closeMenu();},true);
  canvas.addEventListener('keydown',e=>{
    const state=read(),id=state.selected;
    if(e.key==='Escape'){if(!menu.hidden)closeMenu();else if(ui.mode==='add')setMode('select');else if(id!=null){select(null);draw();}if(isExpanded())toggleExpanded(false);return;}
    if(e.key==='f'||e.key==='F'){fit();return;}
    if(e.key==='['||e.key===']'){const count=state.waypoints.length;if(!count)return;const next=typeof id==='number'?id+(e.key===']'?1:-1):e.key===']'?0:count-1;select((next+count)%count);draw();e.preventDefault();return;}
    if(id==null)return;
    const keys={ArrowRight:[0,1],ArrowLeft:[0,-1],ArrowUp:[1,1],ArrowDown:[1,-1],PageUp:[2,1],PageDown:[2,-1]};
    if(keys[e.key]){nudge(...keys[e.key]);e.preventDefault();return;}
    if((e.key==='Delete'||e.key==='Backspace')&&typeof id==='number'&&remove){remove(id);e.preventDefault();}
  });
  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&isExpanded()&&!host.contains(document.activeElement))toggleExpanded(false);});

  // ---- camera ----
  function fit(){
    const {initial,waypoints,pads}=read(),box=new THREE.Box3().setFromPoints([initial.position,...waypoints.map(w=>w.position),...pads.map(p=>p.position)].map(p=>new THREE.Vector3(...p)));
    const center=box.getCenter(new THREE.Vector3()),size=Math.max(8,box.getSize(new THREE.Vector3()).length());
    // Portrait viewports need more distance to fit the route's width.
    const aspect=Math.max(.45,Math.min(1,canvas.clientWidth/Math.max(1,canvas.clientHeight)));
    camera.up.set(0,0,1);controls.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(.85,-1.15,.8).multiplyScalar(size/aspect**.6));controls.update();schedule();
  }
  host.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>{
    fit();const view=button.dataset.view;if(view==='fit'||view==='iso')return;
    const distance=camera.position.distanceTo(controls.target);
    camera.up.set(0,view==='top'?1:0,view==='top'?0:1);camera.position.copy(controls.target).add(view==='top'?new THREE.Vector3(0,0,distance):new THREE.Vector3(0,-distance,0));controls.update();schedule();
  });
  // Full screen: the Fullscreen API where it exists (not on iPhone Safari),
  // otherwise a fixed overlay that fills the viewport.
  const isExpanded=()=>document.fullscreenElement===host||host.classList.contains('expanded');
  async function toggleExpanded(on=!isExpanded()){
    if(!on){if(document.fullscreenElement===host)await document.exitFullscreen();host.classList.remove('expanded');document.body.classList.remove('planner-expanded');}
    else{try{if(!host.requestFullscreen)throw new Error('unsupported');await host.requestFullscreen();}catch{host.classList.add('expanded');document.body.classList.add('planner-expanded');}}
    syncExpanded();
  }
  function syncExpanded(){const on=isExpanded(),b=host.querySelector('.expand-route');b.textContent=on?'✕':'⛶';b.title=b.ariaLabel=on?'Exit full screen':'Full screen';requestAnimationFrame(()=>{fit();render();});}
  host.querySelector('.expand-route').onclick=()=>toggleExpanded();
  document.addEventListener('fullscreenchange',syncExpanded);
  matchMedia('(pointer: coarse)').addEventListener?.('change',()=>{updateHint();draw();});
  new ResizeObserver(render).observe(viewport);
  setMode('select');draw();fit();
  return {draw,fit};
}
