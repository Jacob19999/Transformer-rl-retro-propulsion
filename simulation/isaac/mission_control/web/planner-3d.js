import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {sampleRouteLegs,waypointColors,waypointLabel,waypointTypes} from './flight-plan.js';

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
export function createPlanner3D(host,{read,move,select,context}){
  host.innerHTML=`<canvas tabindex="0" aria-label="3D waypoint editor: click a waypoint, drag its X Y Z arrows; right-click to edit or add a waypoint"></canvas>
    <div class="scene-label">3D ROUTE <span>WORLD XYZ · Z UP · METRES</span></div>
    <div class="scene-tools" role="toolbar" aria-label="3D view"><div class="scene-views" role="group" aria-label="Camera"><button type="button" class="fit-route" title="Frame the whole route">Fit</button><button type="button" data-view="iso" title="Oblique view">3D</button><button type="button" data-view="top" title="Look down the Z axis">Top</button><button type="button" data-view="side" title="Look along +Y">Side</button></div><button type="button" class="expand-route" title="Full screen">⛶ Full screen</button></div>
    <div class="scene-legend" aria-hidden="true"><span><i class="legend-start"></i>Start</span>${Object.entries(waypointTypes).map(([type,name])=>`<span><i style="background:${waypointColors[type]}"></i>${name}</span>`).join('')}<span><i class="legend-pad"></i>Pad</span><span><i class="legend-corridor"></i>Corridor</span></div>
    <div class="scene-instructions"><b>Click</b> select · <b>drag arrows</b> move · <b>right-click</b> add / edit · <b>drag</b> orbit · <b>right-drag</b> pan · <b>scroll</b> zoom</div>
    <div class="waypoint-context" role="dialog" aria-label="Waypoint actions" hidden></div>`;
  const canvas=host.querySelector('canvas');
  const renderer=new THREE.WebGLRenderer({canvas,antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setClearColor('#060b11');
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(45,1,.05,1500);camera.up.set(0,0,1);camera.position.set(18,-24,20);
  const controls=new OrbitControls(camera,canvas);controls.target.set(0,0,5);controls.minDistance=2;controls.maxDistance=600;
  // 1 m minor and 10 m major grid on the ground plane; axis tags mark +X / +Y.
  const minor=new THREE.GridHelper(200,200,0x16222e,0x16222e),major=new THREE.GridHelper(200,20,0x3a516a,0x2a3c4f);
  [minor,major].forEach(g=>{g.rotation.x=Math.PI/2;g.material.transparent=true;g.material.opacity=g===minor?.45:.9;scene.add(g);});
  const axes=new THREE.AxesHelper(5);scene.add(axes);
  const axisTag=(text,color,position)=>{const c=document.createElement('canvas');c.width=c.height=64;const x=c.getContext('2d');x.fillStyle=color;x.font='bold 30px sans-serif';x.fillText(text,6,42);const s=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),depthTest:false,sizeAttenuation:false}));s.position.set(...position);s.scale.set(.05,.05,1);scene.add(s);};
  axisTag('+X','#ff7070',[5.6,0,0]);axisTag('+Y','#7fe36b',[0,5.6,0]);
  let content=new THREE.Group(),handles=[],drag=null;scene.add(content);
  const gizmo=new THREE.Group(),axisHandles=[],menu=host.querySelector('.waypoint-context');scene.add(gizmo);
  const directions=[new THREE.Vector3(1,0,0),new THREE.Vector3(0,1,0),new THREE.Vector3(0,0,1)];
  directions.forEach((direction,i)=>{
    const arrow=new THREE.ArrowHelper(direction,new THREE.Vector3(),1,['#ff5555','#70df58','#559dff'][i],.22,.13);
    // Wide invisible shafts improve picking without introducing planar handles.
    const shaft=new THREE.Mesh(new THREE.CylinderGeometry(.065,.065,.8,12),new THREE.MeshBasicMaterial({visible:false}));
    shaft.position.y=.5;
    arrow.add(shaft);[shaft,arrow.cone].forEach(o=>{o.userData.axis=i;axisHandles.push(o);});
    arrow.traverse(o=>{if(o.material){o.material.depthTest=false;o.material.depthWrite=false;}o.renderOrder=20;});
    const c=document.createElement('canvas');c.width=c.height=64;const ctx=c.getContext('2d');ctx.fillStyle=['#ff5555','#70df58','#559dff'][i];ctx.font='bold 48px sans-serif';ctx.fillText('XYZ'[i],10,50);
    const tag=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),depthTest:false}));tag.position.y=1.15;tag.scale.set(.22,.22,1);tag.renderOrder=21;arrow.add(tag);gizmo.add(arrow);
  });
  function closeMenu(){menu.hidden=true;menu.replaceChildren();}
  function updateGizmo(){
    const {initial,waypoints,pads,selected}=read();
    const position=selected==='start'?initial.position:typeof selected==='string'?pads[Number(selected.split(':')[1])]?.position:waypoints[selected]?.position;
    gizmo.visible=!!position;if(!position)return;
    gizmo.position.set(...position);
    const enabled=editableAxes(selected,waypoints);
    gizmo.children.forEach((arrow,i)=>{arrow.visible=enabled[i];});
    const scale=camera.position.distanceTo(gizmo.position)*2*Math.tan(THREE.MathUtils.degToRad(camera.fov/2))*90/Math.max(1,canvas.clientHeight);
    gizmo.scale.setScalar(scale);
  }
  const ray=new THREE.Raycaster(),pointer=new THREE.Vector2();
  function render(){if(!canvas.clientWidth||!canvas.clientHeight)return;renderer.setSize(canvas.clientWidth,canvas.clientHeight,false);camera.aspect=canvas.clientWidth/canvas.clientHeight;camera.updateProjectionMatrix();updateGizmo();renderer.render(scene,camera);}
  controls.addEventListener('change',render);
  function label(text,position,color,below=false){
    const c=document.createElement('canvas'),x=c.getContext('2d');x.font='28px sans-serif';
    c.width=Math.min(720,Math.ceil(x.measureText(text).width)+24);c.height=48;
    x.fillStyle='#07111de6';x.fillRect(0,0,c.width,c.height);x.fillStyle=color;x.font='28px sans-serif';x.fillText(text,12,34,c.width-24);
    const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),depthTest:false,sizeAttenuation:false}));
    sprite.position.copy(position);sprite.center.set(.5,below?1.7:-.6);sprite.scale.set(.032*c.width/c.height,.032,1);content.add(sprite);
  }
  function handle(position,color,id,title,selected=false){
    const mesh=new THREE.Mesh(id==='start'?new THREE.OctahedronGeometry(selected?.42:.34):new THREE.SphereGeometry(selected?.36:.28,20,14),new THREE.MeshBasicMaterial({color,depthTest:false}));
    mesh.position.set(...position);mesh.userData.id=id;mesh.renderOrder=5;content.add(mesh);handles.push(mesh);
    if(selected){const ring=new THREE.Mesh(new THREE.TorusGeometry(.55,.05,8,40),new THREE.MeshBasicMaterial({color:0xffffff,depthTest:false}));ring.position.copy(mesh.position);ring.renderOrder=6;ring.onBeforeRender=()=>ring.quaternion.copy(camera.quaternion);content.add(ring);}
    label(title,mesh.position,color,typeof id==='string'&&id.startsWith('pad:'));
  }
  // Dashed plumb line and ground shadow make each marker's altitude readable from any angle.
  function dropLine(position,color){
    if(position[2]<.2)return;
    const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...position),new THREE.Vector3(position[0],position[1],.01)]),new THREE.LineDashedMaterial({color,dashSize:.25,gapSize:.2,transparent:true,opacity:.55}));
    line.computeLineDistances();content.add(line);
    const shadow=new THREE.Mesh(new THREE.CircleGeometry(.22,20),new THREE.MeshBasicMaterial({color,transparent:true,opacity:.35,depthWrite:false}));shadow.position.set(position[0],position[1],.015);content.add(shadow);
  }
  function draw(){
    content.traverse(o=>{o.geometry?.dispose();o.material?.map?.dispose();o.material?.dispose();});scene.remove(content);content=new THREE.Group();scene.add(content);handles=[];
    const {initial,waypoints,pads,corridor,convex,disturbances,selected}=read();
    if(disturbances?.selected.includes('wind')){
      const vector=new THREE.Vector3(...disturbances.settings.wind.steady_vector),speed=vector.length();
      if(speed>1e-6){
        const origin=new THREE.Vector3(...initial.position).add(new THREE.Vector3(0,0,1));
        const arrow=new THREE.ArrowHelper(vector.normalize(),origin,Math.min(8,Math.max(1,speed)),0xe5ad48,.5,.25);content.add(arrow);
        label(`WIND ${speed.toFixed(1)} m/s`,origin,'#e5ad48');
      }
    }
    const target=pads[waypoints.at(-1)?.pad??0]?.position??[0,0,0];
    const route=waypoints.filter(w=>w.type!=='land');
    sampleRouteLegs(initial.position,waypoints,target,{convex}).forEach((leg,i)=>{
      const points=leg.map(p=>new THREE.Vector3(...p));
      // Each leg takes the colour of the step it flies to; the landing leg is white.
      const color=new THREE.Color(route[i]?waypointColors[route[i].type]:waypointColors.land);
      content.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color,transparent:true,opacity:selected===i?1:.8})));
      const width=waypoints[i]?.corridor_m??corridor;
      if(convex&&waypoints.length&&width>0&&points[0].distanceTo(points.at(-1))>1e-6){
        const curve=new THREE.Curve();curve.getPoint=t=>{const f=t*(points.length-1),n=Math.min(points.length-2,Math.floor(f));return points[n].clone().lerp(points[n+1],f-n);};
        const tube=new THREE.Mesh(new THREE.TubeGeometry(curve,96,width,12,false),new THREE.MeshBasicMaterial({color:0x3987e5,transparent:true,opacity:selected===i?.2:.1,side:THREE.DoubleSide,depthWrite:false}));content.add(tube);
      }
    });
    dropLine(initial.position,'#ffffff');
    handle(initial.position,'#ffffff','start',`START · ${initial.position[2].toFixed(1)} m`,selected==='start');
    waypoints.forEach((w,i)=>{
      const position=w.type==='land'?[w.position[0],w.position[1],.4]:w.position;
      if(w.type!=='land')dropLine(position,waypointColors[w.type]);
      handle(position,waypointColors[w.type],i,w.type==='land'?waypointLabel(w,i):`${waypointLabel(w,i)} · ${w.position[2].toFixed(1)} m`,selected===i);
      // Capture sphere of the selected step: arrival happens inside it.
      if(selected===i&&w.type!=='land'){
        const sphere=new THREE.Mesh(new THREE.SphereGeometry(w.radius_m,24,16),new THREE.MeshBasicMaterial({color:waypointColors[w.type],wireframe:true,transparent:true,opacity:.25,depthWrite:false}));
        sphere.position.set(...w.position);content.add(sphere);
      }
    });
    pads.forEach((p,i)=>{
      const fill=new THREE.Mesh(new THREE.CircleGeometry(1.25,48),new THREE.MeshBasicMaterial({color:0x48dba2,transparent:true,opacity:selected===`pad:${i}`?.28:.12,depthWrite:false}));fill.position.set(p.position[0],p.position[1],.012);content.add(fill);
      const disc=new THREE.Mesh(new THREE.RingGeometry(1.1,1.25,48),new THREE.MeshBasicMaterial({color:0x48dba2,side:THREE.DoubleSide}));disc.position.set(...p.position);disc.position.z=.02;content.add(disc);
      const cross=new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints([[-.45,0],[.45,0],[0,-.45],[0,.45]].map(([x,y])=>new THREE.Vector3(p.position[0]+x,p.position[1]+y,.025))),new THREE.LineBasicMaterial({color:0x48dba2}));content.add(cross);
      handle([p.position[0],p.position[1],.08],'#48dba2',`pad:${i}`,p.name,selected===`pad:${i}`);
    });render();
  }
  function fit(){const {initial,waypoints,pads}=read(),box=new THREE.Box3().setFromPoints([initial.position,...waypoints.map(w=>w.position),...pads.map(p=>p.position)].map(p=>new THREE.Vector3(...p)));const center=box.getCenter(new THREE.Vector3()),size=Math.max(8,box.getSize(new THREE.Vector3()).length());controls.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(.85,-1.15,.8).multiplyScalar(size));controls.update();render();}
  function cast(e){const rect=canvas.getBoundingClientRect();pointer.set((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);ray.setFromCamera(pointer,camera);}
  canvas.addEventListener('pointerdown',e=>{
    if(e.button!==0)return;closeMenu();cast(e);
    const axisHit=gizmo.visible?ray.intersectObjects(axisHandles.filter(o=>o.parent.visible),false)[0]:null;
    if(axisHit){
      const axis=directions[axisHit.object.userData.axis],origin=gizmo.position.clone(),id=read().selected;
      const plane=axisDragPlane(axis,origin,camera.getWorldDirection(new THREE.Vector3()));
      const anchor=plane&&ray.ray.intersectPlane(plane,new THREE.Vector3());
      if(anchor){drag={id,origin,anchor,plane,axis};controls.enabled=false;canvas.setPointerCapture(e.pointerId);}
      e.stopImmediatePropagation();e.preventDefault();return;
    }
    const hit=ray.intersectObjects(handles)[0];
    if(hit){select(hit.object.userData.id);render();e.stopImmediatePropagation();e.preventDefault();}
    else{select(null);render();}
  },true);
  canvas.addEventListener('pointermove',e=>{if(!drag)return;cast(e);const hit=ray.ray.intersectPlane(drag.plane,new THREE.Vector3());if(!hit)return;const delta=hit.sub(drag.anchor).dot(drag.axis);move(drag.id,drag.origin.clone().addScaledVector(drag.axis,delta).toArray(),directions.indexOf(drag.axis));});
  const end=e=>{drag=null;controls.enabled=true;if(e&&canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId);};canvas.addEventListener('pointerup',end);canvas.addEventListener('pointercancel',end);canvas.addEventListener('lostpointercapture',end);
  let rightPress=null;
  canvas.addEventListener('pointerdown',e=>{if(e.button===2)rightPress=[e.clientX,e.clientY];},true);
  canvas.addEventListener('contextmenu',e=>{
    e.preventDefault();if(rightPress&&Math.hypot(e.clientX-rightPress[0],e.clientY-rightPress[1])>5){rightPress=null;return;}rightPress=null;end();cast(e);
    const hit=ray.intersectObjects(handles)[0],id=hit?.object.userData.id;
    // Empty-space placement uses the selected waypoint's height (ground projected
    // to the default 3 m when no waypoint is selected), then the normal task bounds.
    const {waypoints,selected}=read(),height=typeof selected==='number'?waypoints[selected]?.position[2]??3:3;
    const position=ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0,0,1),-Math.max(1,height)),new THREE.Vector3());
    closeMenu();if(id==null&&!position)return;
    if(id!=null)select(id);
    context?.({id,position:position?.toArray(),menu,close:closeMenu});
    if(!menu.childElementCount)return;menu.hidden=false;
    const rect=host.getBoundingClientRect();menu.style.left=`${Math.max(0,Math.min(e.clientX-rect.left,host.clientWidth-menu.offsetWidth))}px`;menu.style.top=`${Math.max(0,Math.min(e.clientY-rect.top,host.clientHeight-menu.offsetHeight))}px`;
    menu.querySelector('input,select,button')?.focus();render();
  });
  document.addEventListener('pointerdown',e=>{if(!menu.hidden&&!menu.contains(e.target))closeMenu();},true);
  document.addEventListener('keydown',e=>{if(e.key==='Escape'){closeMenu();end();}});
  host.querySelector('.fit-route').onclick=()=>{camera.up.set(0,0,1);fit();};
  host.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>{camera.up.set(0,0,1);fit();if(button.dataset.view==='iso')return;const distance=camera.position.distanceTo(controls.target);camera.up.set(0,button.dataset.view==='top'?1:0,button.dataset.view==='top'?0:1);camera.position.copy(controls.target).add(button.dataset.view==='top'?new THREE.Vector3(0,0,distance):new THREE.Vector3(0,-distance,0));controls.update();render();});
  host.querySelector('.expand-route').onclick=async()=>{if(document.fullscreenElement===host)await document.exitFullscreen();else await host.requestFullscreen();};
  document.addEventListener('fullscreenchange',()=>{host.querySelector('.expand-route').textContent=document.fullscreenElement===host?'✕ Exit full screen':'⛶ Full screen';render();});
  new ResizeObserver(render).observe(host);draw();fit();
  return {draw,fit};
}
