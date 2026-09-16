"""Explicit spline/waypoint task, without a controller or action overrides.

The policy observes the active goal, path geometry and waypoint semantics.
Hover requires a continuous timed position/speed hold; fly-through uses a
swept segment arrival test. Final success still requires real soft contact.
"""
from __future__ import annotations

import torch
from tvc_env.common.frames import isaac_position_to_frd
from tvc_env.common.quaternions import inverse, normalize, rotate_vector

LAND, HOVER, FLYPASS = 0, 1, 2
EXTRA_OBSERVATIONS = 15
MAX_WAYPOINTS = 12


def catmull_rom(p0, p1, p2, p3, samples=49):
    """Uniform cubic interpolating spline, including both leg endpoints."""
    t = torch.linspace(0, 1, samples, device=p1.device, dtype=p1.dtype)[None,:,None]
    a, b, c, d = (p[:,None,:] for p in (p0,p1,p2,p3))
    return .5*((2*b)+(-a+c)*t+(2*a-5*b+4*c-d)*t*t+(-a+3*b-3*c+d)*t*t*t)


def segment_distance(point, a, b):
    delta = b-a
    alpha = ((point-a)*delta).sum(-1)/delta.square().sum(-1).clamp(min=1e-12)
    closest = a + alpha.clamp(0,1)[:,None]*delta
    return (point-closest).norm(dim=-1)


class WaypointMission:
    def __init__(self, num_envs, device, config, origins, landing_target):
        self.config, self.origins, self.landing_target = config, origins, landing_target
        self.device, self.n = device, num_envs
        self.ids = torch.arange(num_envs, device=device)
        self.positions = torch.zeros(num_envs, MAX_WAYPOINTS+1, 3, device=device)
        self.kinds = torch.zeros(num_envs, MAX_WAYPOINTS+1, dtype=torch.long, device=device)
        self.radii = torch.ones(num_envs, MAX_WAYPOINTS+1, device=device)
        self.holds = torch.zeros_like(self.radii)
        self.speeds = torch.ones_like(self.radii)*2
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.index = torch.zeros_like(self.count)
        self.hold_elapsed = torch.zeros(num_envs, device=device)
        self.start = torch.zeros(num_envs,3, device=device)
        self.curves = torch.zeros(num_envs,49,3, device=device)
        self.path_error = torch.zeros_like(self.start)
        self.tangent = torch.zeros_like(self.start)
        self.goal = landing_target.clone()
        self.step_completed = torch.zeros(num_envs, device=device)
        self.step_path_cost = torch.zeros(num_envs, device=device)
        self.step_progress = torch.zeros(num_envs, device=device)
        self.last_potential = torch.zeros(num_envs, device=device)

    @property
    def ready_to_land(self):
        return self.index >= self.count

    def reset(self, env_ids, position):
        task = self.config.get('task', {})
        nav = task.get('navigation', {})
        spawn = task.get('spawn', {})
        m = len(env_ids)
        self.start[env_ids] = position[env_ids]
        self.index[env_ids] = 0
        self.hold_elapsed[env_ids] = 0
        self.step_completed[env_ids] = self.step_path_cost[env_ids] = self.step_progress[env_ids] = 0
        self.positions[env_ids] = self.landing_target[env_ids,None,:]
        self.kinds[env_ids] = LAND
        self.holds[env_ids] = 0
        explicit = nav.get('waypoints')
        if explicit is not None:
            if len(explicit) > MAX_WAYPOINTS:
                raise ValueError(f'At most {MAX_WAYPOINTS} waypoints are supported')
            self.count[env_ids] = len(explicit)
            for j, wp in enumerate(explicit):
                self.positions[env_ids,j] = torch.tensor(wp['position'], device=self.device)+self.origins[env_ids]
                self.kinds[env_ids,j] = HOVER if wp['type']=='hover' else FLYPASS
                self.radii[env_ids,j] = wp.get('radius_m',1.)
                self.holds[env_ids,j] = wp.get('hold_s',2.) if wp['type']=='hover' else 0.
                self.speeds[env_ids,j] = wp.get('speed_m_s',2.)
        else:
            bounds = spawn.get('waypoint_count_range', nav.get('count_range',[0,3]))
            self.count[env_ids] = torch.randint(int(bounds[0]), int(bounds[1])+1, (m,), device=self.device)
            spread = float(spawn.get('waypoint_spread_m', nav.get('spread_m',5.)))
            kind_mode = str(spawn.get('waypoint_kind', nav.get('kind','mixed'))).lower()
            radius = float(spawn.get('waypoint_radius_m', nav.get('radius_m',1.)))
            hold_s = float(spawn.get('waypoint_hover_hold_s', nav.get('hover_hold_s',2.)))
            speed_m_s = float(spawn.get('waypoint_speed_m_s', nav.get('speed_m_s',2.)))
            for j in range(MAX_WAYPOINTS):
                fraction = (j+1)/(self.count[env_ids].float()+1)
                endpoint = self.landing_target[env_ids].clone(); endpoint[:,2] += 4.
                goal = torch.lerp(position[env_ids], endpoint, fraction[:,None])
                goal[:,:2] += (torch.rand(m,2,device=self.device)*2-1)*spread
                goal[:,2] = torch.maximum(goal[:,2], self.origins[env_ids,2]+3.)
                valid = j < self.count[env_ids]
                self.positions[env_ids,j] = torch.where(valid[:,None], goal, self.landing_target[env_ids])
                if kind_mode == 'hover':
                    kind = torch.full((m,), HOVER, device=self.device, dtype=torch.long)
                elif kind_mode == 'flypass':
                    kind = torch.full((m,), FLYPASS, device=self.device, dtype=torch.long)
                else:
                    kind = torch.where(torch.rand(m,device=self.device)<.5,HOVER,FLYPASS)
                self.kinds[env_ids,j] = torch.where(valid, kind, LAND)
                self.radii[env_ids,j] = radius
                self.holds[env_ids,j] = hold_s
                self.speeds[env_ids,j] = speed_m_s
        self._refresh_curves(env_ids)
        self.update_features(position)
        self.last_potential[env_ids] = self.potential(position)[env_ids]

    def _refresh_curves(self, env_ids):
        i = self.index[env_ids]
        p1 = torch.where((i==0)[:,None], self.start[env_ids], self.positions[env_ids,(i-1).clamp(min=0)])
        p0 = torch.where((i<2)[:,None], self.start[env_ids], self.positions[env_ids,(i-2).clamp(min=0)])
        p2 = self.positions[env_ids,i]
        p3 = self.positions[env_ids,(i+1).clamp(max=MAX_WAYPOINTS)]
        self.curves[env_ids] = catmull_rom(p0,p1,p2,p3)

    def update_features(self, position):
        segments = self.curves[:,1:]-self.curves[:,:-1]
        alpha = (((position[:,None,:]-self.curves[:,:-1])*segments).sum(-1)
                 /segments.square().sum(-1).clamp(min=1e-9)).clamp(0,1)
        projections = self.curves[:,:-1]+alpha[:,:,None]*segments
        nearest = (projections-position[:,None,:]).square().sum(-1).argmin(-1)
        nearest_point = projections[self.ids,nearest]
        tangent = segments[self.ids,nearest]
        self.tangent = tangent/tangent.norm(dim=-1,keepdim=True).clamp(min=1e-6)
        self.path_error = nearest_point-position
        kind = self.kinds[self.ids,self.index]
        # The lookahead is a public path reference, not an action controller.
        # It avoids demanding a stop at each fly-through point.
        lengths = segments.norm(dim=-1)
        arc = torch.cat((torch.zeros_like(lengths[:,:1]),lengths.cumsum(-1)),dim=-1)
        distance = arc[self.ids,nearest]+alpha[self.ids,nearest]*lengths[self.ids,nearest]
        lookahead_arc = torch.minimum(distance+self.speeds[self.ids,self.index].clamp(min=2),arc[:,-1])
        lookahead_index = torch.searchsorted(arc.contiguous(),lookahead_arc[:,None].contiguous()).squeeze(-1).clamp(1,48)-1
        fraction = ((lookahead_arc-arc[self.ids,lookahead_index]) / lengths[self.ids,lookahead_index].clamp(min=1e-9)).clamp(0,1)
        lookahead = self.curves[self.ids,lookahead_index]+fraction[:,None]*segments[self.ids,lookahead_index]
        self.goal = torch.where((kind==FLYPASS)[:,None],lookahead,self.positions[self.ids,self.index])
        self.path_error = torch.where(self.ready_to_land[:,None],torch.zeros_like(self.path_error),self.path_error)
        self.tangent = torch.where(self.ready_to_land[:,None],torch.zeros_like(self.tangent),self.tangent)

    def potential(self, position):
        current = self.positions[self.ids,self.index]
        remaining = (current-position).norm(dim=-1)
        lengths = (self.positions[:,1:]-self.positions[:,:-1]).norm(dim=-1)
        legs = torch.arange(MAX_WAYPOINTS,device=self.device)[None,:]
        remaining += (lengths*((legs>=self.index[:,None]) & (legs<self.count[:,None]))).sum(-1)
        return -remaining

    def advance(self, before, after, velocity, dt, active):
        goal = self.positions[self.ids,self.index]
        kind = self.kinds[self.ids,self.index]
        radius = self.radii[self.ids,self.index]
        # Reset the dwell timer whenever the position/speed condition breaks.
        stable = ((after-goal).norm(dim=-1)<=radius) & (velocity.norm(dim=-1)<=.4)
        self.hold_elapsed = torch.where(active & (kind==HOVER) & stable, self.hold_elapsed+dt,
                                       torch.where(active,torch.zeros_like(self.hold_elapsed),self.hold_elapsed))
        hover_complete = (kind==HOVER) & (self.hold_elapsed>=self.holds[self.ids,self.index])
        swept_arrival = segment_distance(goal,before,after)<=radius
        moving_forward = ((after-before)*self.tangent).sum(-1)>0
        fly_complete = (kind==FLYPASS) & swept_arrival & moving_forward
        complete = active & (hover_complete|fly_complete) & ~self.ready_to_land
        self.step_completed = complete.float()
        self.update_features(after)
        cross_error = self.path_error.norm(dim=-1)
        # User-facing reference speed is an explicit task preference. Hover
        # capture requires zero speed; fly-through keeps positive tangent speed.
        cruise_speed = self.speeds[self.ids,self.index]
        distance_to_goal = (after-goal).norm(dim=-1)
        braking_time = float(self.config.get('task', {}).get('navigation', {}).get(
            'hover_braking_time_s', 2.0))
        # A hover waypoint publishes a smooth speed reference that reaches
        # zero at the capture point.  The previous 3 -> 0 m/s discontinuity
        # occurred only after the vehicle was already inside the dwell gate,
        # making stable capture needlessly sparse.  This remains a reward/path
        # reference: it never modifies the policy action.
        hover_speed = torch.minimum(cruise_speed, distance_to_goal/max(braking_time, 1e-3))
        hover_speed = torch.where(stable, torch.zeros_like(hover_speed), hover_speed)
        desired_speed = torch.where(kind==HOVER, hover_speed, cruise_speed)
        velocity_error = (velocity-self.tangent*desired_speed[:,None]).norm(dim=-1)
        self.step_path_cost = .5*(cross_error/(1+cross_error)+velocity_error/(1+velocity_error))*dt*active*~self.ready_to_land
        self.index = self.index + complete.long()
        self.hold_elapsed[complete] = 0
        ids = complete.nonzero(as_tuple=False).squeeze(-1)
        if len(ids):
            self._refresh_curves(ids)
        self.update_features(after)

    def finish_reward(self, position, terminated):
        gamma = float(self.config['task']['navigation'].get('shaping_gamma',.999))
        next_potential = self.potential(position)
        self.step_progress = gamma*torch.where(terminated,torch.zeros_like(next_potential),next_potential)-self.last_potential
        self.last_potential = next_potential

    def observation(self, quaternion):
        q = inverse(normalize(quaternion))
        def body(vector):
            return isaac_position_to_frd(rotate_vector(q,vector))
        kind = self.kinds[self.ids,self.index]
        onehot = torch.nn.functional.one_hot(kind,3).float()
        next_goal = self.positions[self.ids,(self.index+1).clamp(max=MAX_WAYPOINTS)]
        # 3 kind + 1 speed + 1 radius + 1 remaining hold + 3 path error +
        # 3 tangent + 3 next goal relative to active goal = 15 extra channels.
        distance_scale = float(self.config.get('task', {}).get('navigation', {}).get(
            'observation_distance_scale_m', 25.0))
        return torch.cat((onehot, self.speeds[self.ids,self.index,None]/10,
            self.radii[self.ids,self.index,None]/10,
            (self.holds[self.ids,self.index]-self.hold_elapsed).clamp(min=0)[:,None]/10,
            body(self.path_error)/distance_scale,body(self.tangent),
            body(next_goal-self.goal)/distance_scale),dim=-1)

    def record(self, env_id=0):
        i, count = int(self.index[env_id]), int(self.count[env_id])
        return dict(waypoint_index=i, waypoint_count=count, ready_to_land=i>=count,
                    phase=('LAND','HOVER','FLYPASS')[int(self.kinds[env_id,i])],
                    target_position=(self.goal[env_id]-self.origins[env_id]).tolist(),
                    hold_elapsed_s=float(self.hold_elapsed[env_id]),
                    cross_track_error_m=float(self.path_error[env_id].norm()),
                    waypoints=[dict(position=(self.positions[env_id,j]-self.origins[env_id]).tolist(),
                        type='hover' if int(self.kinds[env_id,j])==HOVER else 'flypass',
                        hold_s=float(self.holds[env_id,j]),radius_m=float(self.radii[env_id,j]),
                        speed_m_s=float(self.speeds[env_id,j])) for j in range(count)])
