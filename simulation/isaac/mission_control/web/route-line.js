import * as THREE from 'three';
import {Line2} from 'three/addons/lines/Line2.js';
import {LineMaterial} from 'three/addons/lines/LineMaterial.js';
import {LineGeometry} from 'three/addons/lines/LineGeometry.js';

// Drawn route as screen-space ribbons: WebGL lines are always 1 px wide, which
// read as a hairline next to the markers. Each leg blends from the colour of
// the step it leaves to the step it flies to, chevrons mark the direction of
// travel, and an optional ground trace makes altitude readable from any angle.
// The geometry is the caller's (the sequencer's Catmull-Rom legs); only the
// drawing lives here.

export function routeLength(legs){
  let total=0;
  for(const leg of legs)for(let i=1;i<leg.length;i++)total+=Math.hypot(leg[i][0]-leg[i-1][0],leg[i][1]-leg[i-1][1],leg[i][2]-leg[i-1][2]);
  return total;
}

// Evenly spaced points along a polyline with their unit tangents; the first
// sits half a spacing in so chevrons never overlap a marker at a leg's end.
export function chevronStations(points,spacing){
  const out=[];if(points.length<2||!(spacing>0))return out;
  let next=spacing/2,travelled=0;
  for(let i=1;i<points.length;i++){
    const a=points[i-1],b=points[i],d=[b[0]-a[0],b[1]-a[1],b[2]-a[2]],length=Math.hypot(...d);
    if(length<1e-9)continue;
    while(next<=travelled+length){const f=(next-travelled)/length;out.push({point:a.map((v,k)=>v+d[k]*f),tangent:d.map(v=>v/length)});next+=spacing;}
    travelled+=length;
  }
  return out;
}

const chevronShape=(()=>{const s=new THREE.Shape();s.moveTo(0,.5);s.lineTo(.42,-.35);s.lineTo(0,-.1);s.lineTo(-.42,-.35);s.closePath();return s;})();

export function createRouteRibbon({width=3,opacity=1,dashed=false,chevrons=true,ground=false,depthTest=true}={}){
  const group=new THREE.Group(),materials=new Set(),resolution=new THREE.Vector2(1,1);
  const up=new THREE.Vector3(0,0,1),forward=new THREE.Vector3(),side=new THREE.Vector3(),basis=new THREE.Matrix4();
  function clear(){group.traverse(o=>{if(o!==group){o.geometry?.dispose();o.material?.dispose();}});group.clear();materials.clear();}
  function material(w,alpha){
    const m=new LineMaterial({linewidth:w,vertexColors:true,transparent:alpha<1,opacity:alpha,dashed,dashSize:.6,gapSize:.35,depthTest,worldUnits:false});
    m.resolution.copy(resolution);materials.add(m);return m;
  }
  // legs: arrays of [x,y,z]; colors: one [from,to] pair (any THREE.Color input) per leg.
  function set(legs,colors,{highlight=-1,scale=null}={}){
    clear();
    const usable=legs.map((leg,i)=>({leg,i})).filter(({leg})=>leg.length>1);
    if(!usable.length)return;
    const total=routeLength(legs),size=scale??THREE.MathUtils.clamp(total/70,.18,1.1);
    const spacing=THREE.MathUtils.clamp(total/28,1.2,9)*Math.max(1,size/.5);
    for(const {leg,i} of usable){
      const [from,to]=colors[i].map(c=>new THREE.Color(c)),n=leg.length,positions=[],vertexColors=[];
      leg.forEach((p,j)=>{positions.push(...p);const c=from.clone().lerp(to,j/(n-1));vertexColors.push(c.r,c.g,c.b);});
      const geometry=new LineGeometry();geometry.setPositions(positions);geometry.setColors(vertexColors);
      const line=new Line2(geometry,material(i===highlight?width*1.7:width,i===highlight||highlight<0?opacity:opacity*.7));
      if(dashed)line.computeLineDistances();
      line.renderOrder=2;group.add(line);
      if(ground&&leg.some(p=>p[2]>.25)){
        const trace=new THREE.Line(new THREE.BufferGeometry().setFromPoints(leg.map(p=>new THREE.Vector3(p[0],p[1],.03))),
          new THREE.LineDashedMaterial({color:to,dashSize:.35,gapSize:.3,transparent:true,opacity:.35,depthWrite:false}));
        trace.computeLineDistances();group.add(trace);
      }
      if(chevrons)for(const {point,tangent} of chevronStations(leg,spacing)){
        // Flat arrowhead in the plane of the tangent and the horizontal
        // normal, so it reads in plan and oblique views alike.
        forward.set(...tangent);side.crossVectors(forward,up);
        if(side.lengthSq()<1e-6)side.set(1,0,0);side.normalize();
        const normal=new THREE.Vector3().crossVectors(side,forward).normalize();
        basis.makeBasis(side,forward,normal);
        const mesh=new THREE.Mesh(new THREE.ShapeGeometry(chevronShape),new THREE.MeshBasicMaterial({color:to.clone().lerp(new THREE.Color('#ffffff'),.35),side:THREE.DoubleSide,transparent:true,opacity:Math.min(1,opacity+.1),depthTest}));
        mesh.quaternion.setFromRotationMatrix(basis);mesh.position.set(...point);mesh.scale.setScalar(size*(i===highlight?1.35:1));mesh.renderOrder=3;group.add(mesh);
      }
    }
  }
  function setResolution(w,h){resolution.set(w,h);materials.forEach(m=>m.resolution.set(w,h));}
  return {group,set,setResolution,dispose:clear};
}
