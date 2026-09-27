"""Advanced plans preserve their semantics from API through navigation and SOCP."""
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from mission_control.models import validate_mission
from tvc_env.envs.waypoints import WaypointMission
from tvc_env.sim.contacts import ContactStateMachine


def flight_plan():
    return dict(position=[0,0,.34], velocity=[0,0,0], waypoints=[
        dict(type='takeoff', name='Climb', position=[0,0,5], speed_m_s=1),
        dict(type='hover', position=[0,0,5], hold_s=3),
        dict(type='flypass', position=[4,0,5], speed_m_s=2),
        dict(type='hover', position=[8,0,5], hold_s=1),
        dict(type='descent', position=[8,0,2], speed_m_s=.8),
        dict(type='land', position=[0,0,0], speed_m_s=.2)])


def test_api_preserves_flight_plan_parameters():
    request=validate_mission(flight_plan())
    assert [w['type'] for w in request['waypoints']]==['takeoff','hover','flypass','hover','descent','land']
    assert request['waypoints'][0]['name']=='Climb'
    assert request['waypoints'][1]['hold_s']==3
    assert request['waypoints'][-1]['speed_m_s']==.2
    assert validate_mission(request)==request


@pytest.mark.parametrize('index,key,value,message', [
    (0,'position',[1,0,5],'vertical'), (0,'position',[0,0,.1],'finite'),
    (2,'type','takeoff','first'), (4,'position',[8,0,6],'below'),
    (5,'pad',4,'landing pad'), (2,'type','land','final'),
    (5,'speed_m_s',.7,'speed'), (2,'speed_m_s',5,'speed'),
    (2,'speed_m_s',float('nan'),'finite'), (1,'hold_s',0,'Hover'),
    (0,'name','a'*41,'name')])
def test_invalid_plans_rejected(index,key,value,message):
    plan=flight_plan();plan['waypoints'][index][key]=value
    with pytest.raises(ValueError,match=message):validate_mission(plan)


def test_stationary_flythrough_is_rejected_before_it_can_stall_navigation():
    with pytest.raises(ValueError,match='Fly-through must differ'):
        validate_mission(dict(position=[0,0,5],waypoints=[dict(type='flypass',position=[0,0,5])]))


def test_ground_launch_keeps_contact_and_crash_detection():
    sm=ContactStateMachine(2,dwell_frames=2)
    contact=torch.ones(2,dtype=torch.bool);force=torch.ones(2)*30
    enabled=torch.tensor([False,True]);safe=torch.zeros(2,dtype=torch.bool)
    for _ in range(20):state=sm.update(contact,safe,force,landing_enabled=enabled)
    assert state.tolist()==[1,2]  # launch contact is visible, ordinary landing completes
    state=sm.update(contact,torch.tensor([True,False]),force,landing_enabled=enabled)
    assert state.tolist()==[3,2]  # crash is never suppressed during launch
    sm.reset()
    for _ in range(3):sm.update(contact,safe,force,landing_enabled=~contact)
    sm.update(~contact,safe,force,landing_enabled=contact)  # liftoff
    assert sm.state.tolist()==[0,0]
    for _ in range(2):sm.update(contact,safe,force,landing_enabled=contact)
    assert sm.state.tolist()==[2,2]


def test_liftoff_arms_landing_and_clears_launch_impact_measurement():
    from tvc_env.envs.direct_rl_env import TVCDirectRLEnv
    false=lambda *args: torch.tensor([False])
    nav=SimpleNamespace(launch_pending=torch.tensor([True]),origins=torch.zeros(1,3))
    env=SimpleNamespace(_navigation=nav,_contact_sm=ContactStateMachine(1),
        _touchdown_speed=torch.tensor([.734]),_airborne_frames=torch.tensor([20]),
        _body_iface=SimpleNamespace(get_root_position=lambda:torch.tensor([[0.,0.,.9]]),
            get_root_quaternion_wxyz=lambda:torch.tensor([[1.,0.,0.,0.]])),
        _crash_detector=SimpleNamespace(check_impact_speed=false,check_tilt_at_contact=false,
            check_angular_rate_at_contact=false,check_excessive_tilt=false))
    TVCDirectRLEnv._update_contact_state(env,torch.zeros(1),torch.tensor([False]),torch.zeros(1),torch.zeros(1,3))
    assert not nav.launch_pending.item()
    assert env._touchdown_speed.item()==0


def test_stationary_hold_leg_does_not_draw_a_spline_excursion():
    from tvc_env.controllers.convex_guidance import catmull_rom_leg
    from tvc_env.envs.waypoints import catmull_rom
    points=[[0.,0.,.34],[0.,0.,5.],[0.,0.,5.],[4.,0.,5.]]
    np.testing.assert_allclose(catmull_rom_leg(points,1),np.tile(points[1],(49,1)))
    curve=catmull_rom(*(torch.tensor([p]) for p in points))
    np.testing.assert_allclose(curve[0].numpy(),np.tile(points[1],(49,1)))


@pytest.mark.parametrize('kind,start,goal', [('takeoff',[0,0,.34],[0,0,5]),('descent',[0,0,5],[0,0,2])])
def test_vertical_waypoints_require_stable_capture_and_keep_type(kind,start,goal):
    cfg={'task':{'navigation':{'waypoints':[dict(type=kind,position=goal,radius_m=.2,name='Vertical')]}}}
    nav=WaypointMission(1,'cpu',cfg,torch.zeros(1,3),torch.zeros(1,3))
    start=torch.tensor([start],dtype=torch.float32);goal=torch.tensor([goal],dtype=torch.float32)
    nav.reset(torch.tensor([0]),start)
    assert nav.record()['phase']==kind.upper()
    assert nav.record()['waypoints'][0]['name']=='Vertical'
    assert nav.launch_pending.item()==(kind=='takeoff')
    assert nav.curves[0,:,0].abs().max()==0
    nav.advance(start,goal,torch.tensor([[0,0,1.]]),.1,torch.tensor([True]))
    assert not nav.ready_to_land.item()
    nav.advance(goal,goal,torch.zeros(1,3),.1,torch.tensor([True]))
    assert nav.ready_to_land.item()


def test_convex_leg_caps_and_flythrough_velocity_are_constraints():
    from test_convex_guidance import planner, WEIGHT, GATE
    from tvc_env.controllers.convex_guidance import RouteWaypoint
    guidance=planner()
    route=[RouteWaypoint((0,0,5),'takeoff',.5,1),
           RouteWaypoint((4,0,5),'flypass',.5,1),
           RouteWaypoint((8,0,5),'hover',.5,1,1),
           RouteWaypoint((8,0,2),'descent',.5,.7)]
    plan=guidance.plan([0,0,1],[0,0,0],WEIGHT,GATE,route)
    assert plan.mode=='optimal'
    node=0
    for segment in plan.segments:
        first=node;node+=segment.nodes
        if segment.speed is not None:
            assert np.linalg.norm(plan.velocity[first+1:node+1],axis=1).max()<=segment.speed+1e-4
        if segment.kind=='flypass':
            np.testing.assert_allclose(plan.velocity[node],[1,0,0],atol=1e-5)
        elif segment.kind in ('takeoff','hover','descent'):
            np.testing.assert_allclose(plan.velocity[node],0,atol=1e-5)
            np.testing.assert_allclose(plan.position[node],segment.target,atol=1e-5)


def test_adapter_preserves_advanced_types_and_only_subtracts_hover_dwell():
    from tvc_env.controllers.convex_adapter import ConvexGuidanceController
    request=validate_mission(flight_plan())
    route=ConvexGuidanceController._route(request['waypoints'][:-1],2.)
    assert route[0].kind=='takeoff' and route[0].hold_s==0
    assert route[1].hold_s==3 and route[-1].kind=='descent'
    active=ConvexGuidanceController._route(request['waypoints'][1:-1],2.)
    assert active[0].hold_s==1


def test_selected_pad_corridors_and_settings_survive_validation_and_navigation():
    from mission_control.models import landing_pad
    from tvc_env.controllers.convex_adapter import ConvexGuidanceController
    request = flight_plan()
    request.update(duration_s=600, pads=[dict(name='Home', position=[0,0,0]),
                                       dict(name='East', position=[8,0,0])],
                   convex_settings={'guidance': {'max_speed_m_s': 5, 'route_corridor_mode': 'strict'}})
    request['waypoints'][0]['corridor_m'] = .7
    request['waypoints'][-1].update(pad=1, approach_speed_m_s=1.5, corridor_m=2)
    spec = validate_mission(request)
    assert spec['waypoints'][-1]['position'] == [8,0,0]
    assert landing_pad(spec)['position'] == [8,0,0]
    assert validate_mission(spec) == spec
    nav = WaypointMission(1, 'cpu', {'task': {'navigation': {'waypoints': spec['waypoints'][:-1]}}},
                          torch.zeros(1,3), torch.tensor([[8.,0.,0.]]))
    nav.reset(torch.tensor([0]), torch.tensor([[0.,0.,.34]]))
    route = ConvexGuidanceController._route(nav.record()['waypoints'], 0)
    assert route[0].corridor_m == .7
    assert nav.positions[0, -1].tolist() == [8,0,0]


def test_convex_profiles_validate_and_round_trip(tmp_path, monkeypatch):
    from mission_control import convex_parameters as cp
    monkeypatch.setattr(cp, 'PROFILES', tmp_path)
    settings = {'guidance': {'max_speed_m_s': 5, 'solver_workers': 2}, 'tracking': {'kp_xy': 1.2}}
    cp.save_profile('Test route', settings)
    assert cp.list_profiles()[0]['settings'] == settings
    assert cp.resolve(settings)['tracking']['kp_xy'] == 1.2
    for invalid in ({'guidance': {'solver_workers': 2.5}}, {'guidance': {'unknown': 1}},
                    {'guidance': {'max_tilt_deg': 30}}, {'tracking': {'kp_xy': True}}):
        with pytest.raises(ValueError): cp.validate_settings(invalid)
    with pytest.raises(ValueError): cp.save_profile('../escape', {})


def test_multiple_pads_and_extended_duration_bounds():
    for change in ({'duration_s': 601}, {'fast_live': 1}, {'pads': []},
                   {'pads': [dict(position=[0,0,1])]},
                   {'pads': [dict(position=[0,0,0]), dict(position=[1,0,0])]}):
        with pytest.raises(ValueError): validate_mission(change)


def test_landing_leg_speed_and_width_reach_socp():
    from test_convex_guidance import planner, WEIGHT, GATE
    guidance = planner()
    path = [np.linspace([0,0,5], GATE.position, 49)]
    plan = guidance.plan([0,0,5], [0,0,0], WEIGHT, GATE, path=path,
                         landing_speed_m_s=.6, landing_corridor_m=.4)
    assert plan.mode == 'optimal'
    assert plan.segments[-1].corridor == .4
    assert np.linalg.norm(plan.velocity, axis=1).max() <= .6001
    guidance.close()


def test_strict_corridor_rejects_emergency_fallback():
    from test_convex_guidance import planner, WEIGHT, GATE
    guidance = planner()
    guidance.corridor_mode = 'strict'
    path = [np.linspace([0,0,5], GATE.position, 49)]
    # Starting far outside this narrow authored tube is explicitly infeasible.
    plan = guidance.plan([10,0,5], [0,0,0], WEIGHT, GATE, path=path, landing_corridor_m=.2)
    assert plan is None
    guidance.close()


def test_strict_corridor_accepts_feasible_plan_without_slack():
    from test_convex_guidance import planner, WEIGHT, GATE
    guidance = planner()
    guidance.corridor_mode = 'strict'
    path = [np.linspace([0,0,5], GATE.position, 49)]
    plan = guidance.plan([0,0,5], [0,0,0], WEIGHT, GATE, path=path,
                         landing_corridor_m=.2, landing_speed_m_s=.6)
    assert plan is not None and plan.mode == 'optimal'
    assert plan.corridor_excess_m < 1e-5
    assert np.linalg.norm(plan.position[:, :2], axis=1).max() < .2001
    guidance.close()


def test_parallel_time_candidates_match_serial_solution():
    from test_convex_guidance import planner, WEIGHT, GATE
    serial, parallel = planner(), planner()
    serial.workers, parallel.workers = 1, 4
    a = serial.plan([0,0,5], [0,0,0], WEIGHT, GATE)
    b = parallel.plan([0,0,5], [0,0,0], WEIGHT, GATE)
    assert a.mode == b.mode == 'optimal'
    np.testing.assert_allclose(a.position, b.position, atol=1e-5)
    assert a.solves == b.solves
    serial.close(); parallel.close()
