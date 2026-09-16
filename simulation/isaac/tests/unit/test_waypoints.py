import math
from types import SimpleNamespace

import pytest
import torch

from tvc_env.envs.waypoints import WaypointMission, catmull_rom, segment_distance
from tvc_env.envs.rewards import compute_landing_success_reward, compute_premature_landing_penalty
from tvc_env.sim.reset_logic import sample_spawn_state
from tvc_env.common.frames import isaac_position_to_frd
from tvc_env.common.quaternions import rotate_vector, inverse


def mission(kind='hover', hold=2.):
    cfg={'task':{'navigation':{'enabled':True,'waypoints':[dict(position=[1,0,5],type=kind,hold_s=hold,radius_m=.5)],'shaping_gamma':.999}}}
    nav=WaypointMission(2,'cpu',cfg,torch.zeros(2,3),torch.zeros(2,3))
    start=torch.tensor([[0.,0,5]]*2)
    nav.reset(torch.arange(2),start)
    return nav,start


def test_hover_requires_continuous_slow_hold_then_switches_to_landing():
    nav,start=mission()
    goal=torch.tensor([[1.,0,5]]*2); active=torch.tensor([True,True]); zero=torch.zeros(2,3)
    nav.advance(start,goal,zero,1.,active)
    assert not nav.ready_to_land.any()
    nav.advance(goal,goal,torch.tensor([[1.,0,0],[0,0,0]]),.5,active)
    assert nav.hold_elapsed.tolist()==[0.,1.5]
    nav.advance(goal,goal,zero,.5,active)
    assert nav.ready_to_land.tolist()==[False,True]
    assert nav.step_completed.tolist()==[0.,1.]
    assert nav.observation(torch.tensor([[1.,0,0,0]]*2)).shape==(2,15)


def test_curriculum_can_isolate_waypoint_kind_and_hover_braking_reference():
    cfg={'task':{'spawn':{'waypoint_count_range':[1,1], 'waypoint_kind':'hover',
                         'waypoint_hover_hold_s':.5, 'waypoint_radius_m':1.5,
                         'waypoint_speed_m_s':2.},
                 'navigation':{'enabled':True, 'shaping_gamma':.999,
                               'hover_braking_time_s':2.,
                               'observation_distance_scale_m':25.}}}
    nav=WaypointMission(2,'cpu',cfg,torch.zeros(2,3),torch.zeros(2,3))
    start=torch.tensor([[0.,0,10.]]*2)
    nav.reset(torch.arange(2),start)
    assert nav.count.tolist()==[1,1]
    assert nav.kinds[:,0].tolist()==[1,1]
    assert nav.holds[:,0].tolist()==pytest.approx([.5,.5])
    assert nav.radii[:,0].tolist()==pytest.approx([1.5,1.5])
    before=nav.positions[:,0].clone(); before[:,0]-=2
    after=nav.positions[:,0].clone(); after[:,0]-=1
    nav.advance(before,after,torch.tensor([[.5,0,0.]]*2),.1,torch.ones(2,dtype=torch.bool))
    # At 1 m range and a 2 s braking horizon, desired speed is 0.5 m/s,
    # so velocity mismatch contributes zero and only cross-track can remain.
    assert torch.all(nav.step_path_cost < .051)


def test_flypass_catches_swept_sphere_and_does_not_accept_reverse_pass():
    nav,start=mission('flypass')
    after=torch.tensor([[2.,0,5],[-1.,0,5]])
    before=torch.tensor([[0.,0,5],[2.,0,5]])
    nav.advance(before,after,(after-before)/.1,.1,torch.tensor([True,True]))
    assert nav.ready_to_land.tolist()==[True,False]
    assert segment_distance(torch.tensor([[1.,0,0]]),torch.tensor([[0.,0,0]]),torch.tensor([[2.,0,0]])).item()==0


def test_absorbing_potential_and_partial_reset_preserve_other_episode():
    nav,start=mission()
    phi=nav.last_potential.clone()
    nav.finish_reward(start,torch.tensor([True,False]))
    assert nav.step_progress[0].item()==pytest.approx(-phi[0].item())
    assert nav.step_progress[1].item()==pytest.approx(-.001*phi[1].item(),abs=1e-6)
    nav.hold_elapsed[1]=1.5
    nav.reset(torch.tensor([0]),start)
    assert nav.hold_elapsed[1].item()==1.5


def test_landing_before_waypoints_never_receives_success():
    state=SimpleNamespace(contact_state=torch.tensor([2,2]),position=torch.zeros(2,3),
        touchdown_speed=torch.zeros(2),mission_ready_to_land=torch.tensor([False,True]))
    assert compute_landing_success_reward(state,{}).tolist()==[0.,1.]
    assert compute_premature_landing_penalty(state,{}).tolist()==[1.,0.]


def test_spline_interpolates_goals_and_is_not_a_piecewise_straight_line():
    p=[torch.tensor([v],dtype=torch.float32) for v in ([0,0,4],[1,0,5],[2,2,6],[3,0,7])]
    curve=catmull_rom(*p)
    assert torch.allclose(curve[:,0],p[1]) and torch.allclose(curve[:,-1],p[2])
    assert not torch.allclose(curve[:,24],(p[1]+p[2])/2)


def test_body_rate_sampling_respects_frame_even_inverted():
    cfg={'spawn':{'position_range':[[0,0,50]]*2,'velocity_range':[[0,0,0]]*2,
        'attitude_range':[[math.pi,0,1.2]]*2,'angular_velocity_range':[[2.,3.,4.]]*2}}
    _,q,_,world=sample_spawn_state(cfg,torch.arange(4),'cpu')
    body=isaac_position_to_frd(rotate_vector(inverse(q),world))
    assert torch.allclose(body,torch.tensor([[2.,3.,4.]]*4),atol=1e-5)
