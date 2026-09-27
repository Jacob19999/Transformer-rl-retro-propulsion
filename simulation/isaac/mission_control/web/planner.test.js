import test from 'node:test';
import assert from 'node:assert/strict';
import {convexCaptureStatus,normalizeWaypoints,parseFlightPlan,serializeFlightPlan,waypointLabel,escapeHtml,sampleRouteLegs} from './flight-plan.js';
import {samplePlannerSpline} from './planner.js';
import * as THREE from 'three';
import {axisDragPlane,editableAxes} from './planner-3d.js';

test('gizmo drag planes preserve the selected axis for oblique cameras',()=>{
  const origin=new THREE.Vector3(3,4,5),camera=new THREE.Vector3(18,-24,20);
  const direction=origin.clone().sub(camera).normalize();
  for(const axis of [new THREE.Vector3(1,0,0),new THREE.Vector3(0,1,0),new THREE.Vector3(0,0,1)]){
    const plane=axisDragPlane(axis,origin,direction);
    const target=origin.clone().addScaledVector(axis,2.5);
    const ray=new THREE.Ray(camera,target.clone().sub(camera).normalize());
    const anchor=new THREE.Ray(camera,direction).intersectPlane(plane,new THREE.Vector3());
    const hit=ray.intersectPlane(plane,new THREE.Vector3());
    const moved=origin.clone().addScaledVector(axis,hit.sub(anchor).dot(axis));
    assert.ok(moved.distanceTo(target)<1e-10);
  }
  assert.equal(axisDragPlane(new THREE.Vector3(0,0,1),origin,new THREE.Vector3(0,0,-1)),null);
});

test('gizmo respects vertical mission legs and pad-bound landing positions',()=>{
  const route=['hover','flypass','takeoff','descent','land'].map(type=>({type}));
  assert.deepEqual(editableAxes(0,route),[true,true,true]);
  assert.deepEqual(editableAxes(1,route),[true,true,true]);
  assert.deepEqual(editableAxes(2,route),[false,false,true]);
  assert.deepEqual(editableAxes(3,route),[false,false,true]);
  assert.deepEqual(editableAxes(4,route),[false,false,false]);
  assert.deepEqual(editableAxes('pad:0',route),[true,true,false]);
  assert.deepEqual(editableAxes('start',route),[true,true,true]);
});

test('portable plans round-trip initial conditions and all waypoint parameters',()=>{
  const initial={position:[0,0,.34],velocity:[0,0,0],attitude_deg:[0,0,0],angular_rate_deg_s:[0,0,0]};
  const waypoints=normalizeWaypoints([{type:'takeoff',name:'Climb',position:[0,0,5],speed_m_s:1},
    {type:'hover',position:[0,0,5],hold_s:4},{type:'descent',position:[0,0,2],speed_m_s:.7},
    {type:'land',position:[0,0,0],speed_m_s:.2}]);
  assert.deepEqual(parseFlightPlan(JSON.stringify({version:1,initial,waypoints})),{initial,waypoints});
  assert.equal(waypointLabel(waypoints[0],0),'1 · Climb');
});
test('invalid plan files fail before mutating the draft',()=>{
  assert.throws(()=>parseFlightPlan('{'));
  assert.throws(()=>parseFlightPlan('{"version":2}'));
  assert.throws(()=>normalizeWaypoints([{type:'unknown',position:[0,0,5]}]));
  assert.throws(()=>normalizeWaypoints([{type:'land',position:[0,0,0],speed_m_s:2}]));
  assert.equal(escapeHtml('<img src=x onerror="x">'),'&lt;img src=x onerror=&quot;x&quot;&gt;');
});
test('vertical legs are straight and explicit landing does not duplicate the pad',()=>{
  const route=[{type:'takeoff',position:[2,1,6]},{type:'hover',position:[5,3,8]},{type:'land',position:[0,0,0]}];
  const points=samplePlannerSpline([2,1,.34],route);
  assert.equal(points.length,3*49);
  for(const point of points.slice(0,49)){assert.equal(point[0],2);assert.equal(point[1],1);assert.ok(point[2]>=.34);}
  assert.deepEqual(points.at(-1),[0,0,0]);
});

test('version 2 preserves the complete mission including selected pads and optimizer',()=>{
  const mission={name:'East pad',duration_s:600,fast_live:true,pads:[{name:'East',position:[8,2,0]}],
    convex_settings:{guidance:{route_corridor_m:2,solver_workers:4}},
    waypoints:[{type:'land',position:[8,2,0],pad:0,speed_m_s:.2,corridor_m:1.3,approach_speed_m_s:1.5}]};
  assert.deepEqual(parseFlightPlan(serializeFlightPlan(mission)),{mission});
  const [w]=normalizeWaypoints(mission.waypoints);
  assert.equal(w.pad,0);assert.equal(w.corridor_m,1.3);assert.equal(w.approach_speed_m_s,1.5);
  assert.deepEqual(sampleRouteLegs([0,0,5],[w]).at(-1).at(-1),[8,2,0]);
});

test('convex display clamps corridor floor and direct plans target the selected pad',()=>{
  const route=[{type:'hover',position:[0,0,6]},{type:'hover',position:[15,0,4.5]}];
  const legs=sampleRouteLegs([0,0,50],route,[20,0,0],{convex:true});
  assert.ok(legs[1].every(p=>p[2]>=4.5));
  const direct=sampleRouteLegs([0,0,5],[],[20,0,0],{convex:true});
  assert.deepEqual(direct[0][24],[10,0,2.5]);
});


test('convex capture diagnostics distinguish distance, speed and continuous dwell',()=>{
  const frame={position:[0,0,7],velocity:[0,0,0],guidance:{},mission:{waypoint_index:0,waypoints:[{type:'hover',position:[0,0,5],radius_m:1}]}};
  assert.match(convexCaptureStatus(frame),/^Outside capture radius/);
  frame.position=[0,0,5];frame.velocity=[.5,0,0];
  assert.match(convexCaptureStatus(frame),/^Slowing for capture/);
  frame.velocity=[0,0,0];
  assert.match(convexCaptureStatus(frame),/^Hold counting/);
  frame.mission.ready_to_land=true;assert.equal(convexCaptureStatus(frame),'');
  frame.mission.ready_to_land=false;frame.guidance=null;assert.equal(convexCaptureStatus(frame),'');
});
