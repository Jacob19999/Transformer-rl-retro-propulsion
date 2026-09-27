"""One real Isaac mission, streaming replay poses and telemetry to disk."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from runner_safety import WallClockWatchdog, force_process_exit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mission_control.models import validate_mission, battery_config, hardware_overrides, disturbance_config, policy_paths, training_envelope_violations, vane_model_overrides
from mission_control.models import landing_pad
from mission_control.convex_parameters import resolve as resolve_convex_settings

FLIGHT_CONTRACT = 'waypoint_flight_v1'
PAD_RADIUS_M = .5  # the mission success criterion's pad radius


CONVEX_ALTITUDE_FAIL_STOP_M = 105.0


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
    for attempt in range(40):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            # Windows refuses the replace while a reader (the mission service
            # polling status.json) has the target open; that aborted mission
            # aea16ce345d4 at t = 37 s. Readers hold it for milliseconds.
            if attempt == 39:
                raise
            time.sleep(0.05)


def flight_overrides(request, saved):
    """A waypoint_flight checkpoint flies its own training plant and task.

    Only the mission's initial state, battery, disturbances and duration
    replace training values. The throttle-rate integrator and yaw damper in
    the saved task config are the flight computer the policy was trained with.
    """
    from tvc_env.envs.task_registry import deep_merge
    if request['hardware_profile'] != 'planned_8s' or not request['battery']['enabled']:
        raise ValueError('waypoint_flight policies fly the planned 8S plant and observe its coupled battery')
    trained = copy.deepcopy(saved['task_config'])
    config = {key: trained[key] for key in ('env', 'edf', 'servo', 'battery', 'physics', 'dynamics', 'task')}
    config['env'].update(num_envs=1, replicate_physics=False, reset_on_crash=False, gizmos_enabled=False)
    spawn = config['task']['spawn']
    spawn.update(position_range=[request['position']] * 2, velocity_range=[request['velocity']] * 2,
                 attitude_range=[[math.radians(x) for x in request['attitude_deg']]] * 2,
                 angular_velocity_range=[[0., 0., 0.]] * 2,  # body rates are written after reset
                 initial_motor_omega_fraction=request['initial_motor_fraction'], initial_motor_omega_jitter=0.,
                 initial_soc_range=[request['battery']['initial_soc']] * 2)
    if 'wind' in request['disturbance']:
        spawn.pop('wind_speed_range', None)  # the selected wind disturbance defines the wind
    else:
        spawn['wind_speed_range'] = [0., 0.]
    config['task']['episode_length_s'] = request['duration_s']
    battery = config['battery']
    battery.update({key: request['battery'][key] for key in
                    ('capacity_ah', 'c_rating', 'initial_soc', 'cell_resistance_ohm', 'max_current_a')})
    battery['max_current_a'] = min(battery['max_current_a'], battery['capacity_ah'] * battery['c_rating'], 120.)
    return deep_merge(config, disturbance_config(request))


def flight_route(request, max_waypoints):
    """Mission waypoints, then a landing on the pad at the origin."""
    if len(request['waypoints']) > max_waypoints - 1:
        raise ValueError(f'waypoint_flight policies observe up to {max_waypoints - 1} waypoints before the landing')
    route = [dict(position=w['position'], type=w['type'], radius_m=w['radius_m'],
                  **({'hold_s': w['hold_s']} if w['type'] == 'hover' else {})) for w in request['waypoints']]
    return route + [dict(position=[0., 0., 0.], type='land', radius_m=PAD_RADIUS_M)]


def flight_record(env):
    """waypoint_flight mission state in the navigation record the UI reads.

    The final LAND waypoint is the landing phase, not a route waypoint.
    """
    from tvc_env.envs.waypoint_flight import segment_distance
    task = env._flight
    record = task.record()
    route = [w for w in record['waypoints'] if w['type'] != 'land']
    index = int(task.index[0])
    previous = task.start[:1] if index == 0 else task.positions[:1, index - 1]
    position = env._body_iface.get_root_position()[:1]
    cross_track = float(segment_distance(position, previous, task.positions[:1, index])[0])
    return dict(waypoint_index=min(index, len(route)), waypoint_count=len(route),
                ready_to_land=record['phase'] == 'LAND' or index >= len(route),
                phase=record['phase'], target_position=record['target_position'],
                hold_elapsed_s=record['hold_elapsed_s'], cross_track_error_m=cross_track,
                outcome=record['outcome'], waypoints=route)


def imu_record(measurement, vec):
    """World-frame pose and velocity the flight computer measured, for the UI.

    Velocity is measured in body FRD; it is rotated to world with the measured
    attitude, which is how the flight computer itself would resolve it.
    """
    if measurement is None:
        return None
    from tvc_env.common.frames import frd_velocity_to_isaac
    from tvc_env.common.quaternions import normalize, rotate_vector
    q = normalize(measurement['quaternion_wxyz'][:1])
    velocity = rotate_vector(q, frd_velocity_to_isaac(measurement['linear_vel_frd'][:1]))
    return dict(position=vec(measurement['position']), quaternion=vec(q),
                velocity=vec(velocity), gyro=vec(measurement['angular_vel_frd']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=None,
                        help='Local diagnostic checkpoint override; does not publish or qualify a policy')
    parser.add_argument('--action-mode', choices=['stochastic','deterministic','mean'], default=None)
    parser.add_argument('--convex-settings', type=Path, default=None,
                        help='Local diagnostic YAML deep-merged over configs/controllers/convex_guidance.yaml; '
                             'recorded in metadata, never used by the mission service')
    args = parser.parse_args()
    request = validate_mission(json.loads(args.request.read_text()))
    if args.checkpoint is not None and request['controller'] == 'convex':
        raise ValueError('Checkpoint diagnostics require a PPO mission')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status = dict(state='starting', phase='Starting Isaac Sim', frames=0, sim_time_s=0., name=request['name'])
    def update(**values):
        status.update(values, wall_time_s=round(time.time() - started, 2))
        atomic_json(output / 'status.json', status)
    update()
    # 480 Hz physics and up to 600 simulated seconds can exceed the old
    # fixed 8-minute watchdog. Keep an explicit finite allowance per request.
    watchdog = WallClockWatchdog(max(480, request['duration_s'] * 45), label='Mission control simulation')
    watchdog.start()
    env = app = convex = None
    try:
        from isaac_launcher import launch_simulation_app, close_simulation_app
        app = launch_simulation_app(headless=True)
        import torch
        from tvc_env.envs.base_env import BaseEnvConfig
        from tvc_env.envs.direct_rl_env import TVCDirectRLEnv
        from tvc_env.controllers.ppo_model import ActorCritic
        from tvc_env.common.frames import isaac_position_to_frd, frd_velocity_to_isaac
        from tvc_env.common.quaternions import normalize, inverse, rotate_vector, tilt_angle
        from tvc_env.common.constants import ContactState

        angles = [math.radians(x) for x in request['attitude_deg']]
        spawn = dict(position_range=[request['position']] * 2, velocity_range=[request['velocity']] * 2,
                     attitude_range=[angles] * 2, initial_motor_omega_fraction=request['initial_motor_fraction'])
        overrides = hardware_overrides(request)
        overrides.update(disturbance_config(request))
        overrides.update(env=dict(reset_on_crash=False, gizmos_enabled=False),
                         task=dict(spawn=spawn, episode_length_s=request['duration_s']),
                         battery=battery_config(request))
        policies = policy_paths()
        saved = None
        flight = False
        if request['controller'] in policies:
            checkpoint = ROOT / (args.checkpoint or policies[request['controller']])
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            flight = saved.get('observation_contract') == FLIGHT_CONTRACT
        if flight:
            overrides = flight_overrides(request, saved)
            obs_dim = saved['obs_dim']
        elif saved is not None:
            obs_dim = saved['model']['actor.0.weight'].shape[1]
            if saved.get('args', {}).get('residual_pid') or saved.get('args', {}).get('landing_guidance'):
                raise ValueError('This mission runner requires a direct-action PPO checkpoint')
            overrides['env']['observe_battery'] = obs_dim in (28,43)
            if request.get('waypoints') and obs_dim != 43:
                raise ValueError('Waypoint missions require the experimental 43-observation mission PPO policy')
            if saved.get('task_config', {}).get('dynamics', {}).get('coupled_jet', {}).get('enabled'):
                # A qualified policy must replay its actual training plant.
                # User-selected initial conditions/battery/disturbances remain
                # explicit overrides; never silently run new weights on legacy
                # damping or independent unlimited vane-force equations.
                from tvc_env.envs.task_registry import deep_merge
                trained = saved['task_config']
                plant = {key: trained[key] for key in ('dynamics','physics','env','task')}
                overrides = deep_merge(plant, overrides)
                overrides['env'].update(num_envs=1, replicate_physics=False)
                overrides['task']['spawn']['curriculum'] = {'enabled':False}
            overrides['task']['navigation'] = dict(enabled=obs_dim==43,waypoints=request.get('waypoints',[]))
        elif request['controller'] == 'convex':
            overrides['task']['target_position'] = landing_pad(request)['position']
            # The mission sequencer (WaypointMission) referees the route; its
            # observation contract needs the battery channels.
            if request.get('waypoints'):
                overrides['env']['observe_battery'] = True
                overrides['task']['navigation'] = dict(enabled=True, waypoints=[
                    w for w in request['waypoints'] if w['type'] != 'land'])
            # The landing task's 30 m altitude fail-stop is a training geofence;
            # planned missions start up to 100 m, so guard just above that.
            overrides['task']['termination'] = dict(max_altitude_error=CONVEX_ALTITUDE_FAIL_STOP_M)
        if saved is None:
            # Convex missions fly the momentum-bounded vanes; PPO policies
            # always replay their own training plant (above).
            from tvc_env.envs.task_registry import deep_merge
            overrides = deep_merge(overrides, vane_model_overrides(request))
        torch.manual_seed(request['seed'])
        if flight:
            config = BaseEnvConfig(task_name='waypoint_flight', sim_root=ROOT, overrides=overrides)
        else:
            config = BaseEnvConfig(task_name='landing', sim_root=ROOT,
                                   env_config_path=ROOT / 'configs/env/single_env_debug.yaml',
                                   overrides=overrides)
        update(phase='Building physics scene')
        env = TVCDirectRLEnv(config)
        dt = config.physics_dt * config.decimation
        device = env.device
        model = decode = convex = None
        policy_meta = dict(controller=request['controller'])
        if saved is not None:
            model = (ActorCritic(obs_dim, saved['act_dim']) if flight else ActorCritic(obs_dim)).to(device)
            model.load_state_dict(saved['model'])
            model.eval()
            policy_meta.update(checkpoint=str(checkpoint), step=saved['step'],
                               diagnostic_checkpoint_override=args.checkpoint is not None,
                               checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                               body_frame_position_error=saved['args'].get('body_frame_position_error', False),
                               observation_dim=obs_dim,
                               # waypoint_flight evaluations and deployment use the actor mean.
                               action_mode=args.action_mode or ('deterministic' if flight else 'stochastic'))
        if flight:
            from tvc_env.envs import waypoint_flight as wf
            contract = saved['action_contract']
            max_angle = float(env._servo_model.max_command_angle)
            if contract['throttle'] == 'rate':
                decode = lambda raw: wf.policy_to_env_rate_action(raw, max_angle)
            else:
                decode = lambda raw: wf.policy_to_env_action(raw, max_angle, contract['throttle_center'],
                                                             contract['throttle_span'])
            env._flight.set_explicit_missions([flight_route(request, wf.MAX_WAYPOINTS)])
            policy_meta.update(observation_contract=FLIGHT_CONTRACT, action_contract=contract,
                               curriculum_stage=(saved.get('curriculum') or {}).get('stage_index'))
        elif request['controller'] == 'convex':
            import dataclasses
            import clarabel
            import yaml
            from tvc_env.controllers.convex_adapter import ConvexGuidanceController, VehicleModel
            convex_settings = resolve_convex_settings(request['convex_settings'])
            if args.convex_settings is not None:
                from tvc_env.envs.task_registry import deep_merge
                convex_settings = deep_merge(convex_settings, yaml.safe_load(args.convex_settings.read_text()))
            electrical = config.config['battery']
            landing = next((w for w in request['waypoints'] if w['type'] == 'land'), None)
            if landing:
                convex_settings['guidance']['touchdown_speed_m_s'] = landing['speed_m_s']
                convex_settings['guidance']['terminal_max_descent_m_s'] = landing['speed_m_s'] * 1.2
            # Thrust-stand-style calibration of this plant's vanes (hardware:
            # measure it); the controller scales its vane efforts by it.
            roll_authority, yaw_authority = env.vane_authority()
            vehicle = VehicleModel(
                mass_kg=float(env._vehicle_mass[0]), hover_throttle=env.nominal_hover_throttle(),
                reference_voltage_v=float(electrical['reference_voltage_v']) if env._battery_model is not None else 1.0,
                shaft_power_at_max_w=float(electrical['shaft_power_at_max_w']),
                motor_efficiency=float(electrical['motor_efficiency']),
                auxiliary_power_w=float(electrical['auxiliary_power_w']),
                rotor_inertia=float(env._edf_model.rotor_inertia), omega_max=float(env._edf_model.omega_max),
                vane_authority_nm_per_rad=roll_authority, yaw_authority_nm_per_rad=yaw_authority,
                **env.attitude_model())
            touchdown = float(config.config['task']['descent_reward']['touchdown_root_height'])
            servo_deadband = float(env._servo_model.deadband) if env._servo_model.apply_deadband else 0.
            convex = ConvexGuidanceController(convex_settings, vehicle, env._target_position[0].tolist(), touchdown,
                                              dt, max_command_angle=float(env._servo_model.max_command_angle),
                                              servo_deadband_rad=servo_deadband,
                                              landing_corridor_m=landing.get('corridor_m') if landing else None,
                                              landing_speed_m_s=landing.get('approach_speed_m_s') if landing else None)
            policy_meta.update(
                guidance='Convex SOCP powered-descent guidance (Acikmese & Ploen 2007 lossless convexification), '
                         're-solved in closed loop; geometric attitude tracking',
                parameters=convex_settings, solver=f'Clarabel {clarabel.__version__}',
                vehicle_model=dict(dataclasses.asdict(vehicle), full_thrust_n=vehicle.full_thrust_n,
                                   servo_deadband_rad=servo_deadband),
                touchdown_root_height_m=touchdown, altitude_fail_stop_m=CONVEX_ALTITUDE_FAIL_STOP_M,
                diagnostic_settings_override=args.convex_settings is not None)
        obs = env.reset(seed=request['seed'])[0]['policy']
        # User input specifies gyro rates in body FRD; PhysX root writer takes world rates.
        rates = torch.tensor([[math.radians(x) for x in request['angular_rate_deg_s']]], device=device)
        initial_quat = env._body_iface.get_root_quaternion_wxyz()
        world_rates = rotate_vector(initial_quat, frd_velocity_to_isaac(rates))
        env._body_iface.set_root_state(env._body_iface.get_root_position(), initial_quat,
                                      torch.tensor([request['velocity']], device=device), world_rates)
        obs = env._get_observations()['policy']
        if convex:
            convex.reset()
        metadata = dict(schema_version=2, request=request, policy=policy_meta, dt=dt,
                        live_execution=dict(fast_live=request['fast_live'], telemetry_stride=1,
                                            headless_render_wall_interval_s=.25 if request['fast_live'] else None),
                        physics_dt=config.physics_dt, decimation=config.decimation,
                        hinge_layout='radial_span_v1',
                        asset_sha256=hashlib.sha256((ROOT / 'assets/usd/drone_v2_physics.usd').read_bytes()).hexdigest(),
                        battery_model=config.config['battery'],
                        hardware_profile=request['hardware_profile'],
                        disturbances=config.config['disturbances'],
                        fin_link_names=env._art_map.fin_link_names,
                        physics_parameters=env._resolved_hardware,
                        dynamics=config.config.get('dynamics', {}),
                        physics=config.config.get('physics', {}),
                        vane_model=('momentum' if config.config.get('dynamics', {}).get('coupled_jet', {}).get('enabled')
                                    else 'legacy'),
                        vane_authority_nm_per_rad=dict(zip(('roll_pitch', 'yaw'), env.vane_authority())),
                        coordinate_frame='World XYZ, Z up; body rates FRD; quaternion wxyz',
                        source='Isaac Sim / PhysX', model_asset='drone_visual.glb',
                        battery_calibration='Estimated 1-RC LiPo model; voltage and power coupled to EDF',
                        body_rate_command=None,
                        terminal_procedure='After LANDED: zero fin/throttle commands for 2 seconds; record actual spool-down and verify final settling. This procedure is outside the PPO episode.',
                        telemetry_notes='Fin command rate is servo target motion per control interval; body-rate targets are not commanded by these policies.',
                        initial_conditions_outside_training=bool(training_envelope_violations(request, saved)) if saved else False,
                        training_envelope_violations=training_envelope_violations(request, saved) if saved else [],
                        experimental_policy=request['controller']=='ppo_mission',
                        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for folder in ('apps', 'tvc_env', 'configs') for p in (ROOT / folder).rglob('*')
                                       if p.suffix in ('.py', '.yaml')})
        atomic_json(output / 'metadata.json', metadata)
        last_target = env._reset_manager.servo_state.clone()
        frames = 0
        cumulative_delta_v = 0.
        latest = None
        control_phase = 'CONTROLLER'
        landing_frame = None
        settled_after_shutdown = None
        last_pump = time.perf_counter()
        def pump_viewer():
            # Browser cameras use recorded poses, not Kit rendering. Keep all
            # PhysX substeps, control calls and telemetry; only avoid redundant
            # headless redraws. Pump Kit at least four times per wall second.
            nonlocal last_pump
            now = time.perf_counter()
            if not request['fast_live'] or now - last_pump >= .25:
                env.render()
                last_pump = now
        with (output / 'frames.jsonl').open('w', encoding='utf-8', buffering=1) as stream:
            def capture(t, action, rate):
                nonlocal frames, latest
                state = env._build_vehicle_state()
                def vec(x):
                    return x[0].detach().cpu().tolist()
                art = env._drone.data
                ids = env._art_map.fin_body_indices
                debug = env._last_dynamics_debug if hasattr(env, '_last_dynamics_debug') else {}
                raw = float(env._edf_model.compute_thrust(state.motor_omega)[0])
                applied = float(debug['edf_applied_thrust_N'][0]) if debug else raw
                if flight and env._pending_actions is not None:
                    # Record what the flight computer commanded: integrated
                    # throttle duty and vane targets after the yaw damper.
                    action = env._pending_actions
                battery = {k: v[0].item() for k, v in env._battery_model.telemetry().items()} if env._battery_model else None
                frame = dict(t=t, control_phase=control_phase, position=vec(state.position), quaternion=vec(state.quaternion_wxyz),
                             velocity=vec(state.linear_vel_world), gyro=vec(state.angular_vel_frd),
                             # The state the controller acted on: what the IMU and
                             # position sensor reported, sensor noise included.
                             # Equal to the PhysX state when noise is off.
                             imu=imu_record(env.sensor_measurement, vec),
                             fin_angles=vec(state.fin_angles), fin_rates=vec(state.fin_rates),
                             fin_commands=vec(action[:, :4]), fin_command_rates=vec(rate),
                             fin_positions=vec(art.body_pos_w[:, ids]), fin_quaternions=vec(art.body_quat_w[:, ids]),
                             throttle=float(action[0, 4]), rotor_rpm=float(state.motor_omega[0]) * 60 / (2 * math.pi),
                             thrust_n=applied, raw_thrust_n=raw, battery=battery,
                             contact=int(state.contact_state[0]), impact_speed=float(env._touchdown_speed[0]),
                             pad_distance=float((state.position[0, :2] - env._target_position[0, :2]).norm()),
                             contact_force_n=float(env._landing_contact_force_step[0]),
                             propulsive_delta_v_m_s=cumulative_delta_v,
                             rotation=env._rotation.record(),
                             mission=(flight_record(env) if flight else
                                      env._navigation.record() if env._navigation else None),
                             # Convex guidance: phase, plan reference and solver
                             # statistics; the planned path only on replan frames.
                             guidance=(convex.last_telemetry or None) if convex and control_phase == 'CONTROLLER' else None,
                             body_rate_command=None)
                stream.write(json.dumps(frame, allow_nan=False) + '\n')
                frames += 1
                latest = frame
            initial = torch.zeros(1, 5, device=device)
            if flight:
                initial[0, 4] = env._throttle_state[0]  # the duty holding the spawned rotor
            capture(0., initial, torch.zeros(1, 4, device=device))
            simulation_started = time.perf_counter()
            update(state='running', phase='Simulating', frames=frames)
            cancelled = False
            terminated = truncated = torch.tensor([False], device=device)
            with torch.no_grad():
                for step in range(math.ceil(request['duration_s'] / dt)):
                    if (output / 'STOP').exists():
                        cancelled = True
                        break
                    if model:
                        po = obs.clone()
                        if policy_meta['body_frame_position_error']:
                            po[:, :3] = isaac_position_to_frd(rotate_vector(inverse(normalize(obs[:, 3:7])), obs[:, :3]))
                        raw = model.act(po, policy_meta['action_mode'])
                        if decode is not None:
                            action = decode(raw)
                        else:
                            action = torch.cat([raw[:, :4] * env._servo_model.max_command_angle, (raw[:, 4:] + 1) / 2], dim=-1)
                    else:
                        # Mission sequencer state: obs[:, :3] refers to its
                        # active goal; the controller gets the remaining route.
                        nav = env._navigation
                        record = nav.record() if nav else None
                        origin = env._env_origins[0].tolist()
                        waypoints = [dict(w, position=[p + o for p, o in zip(w['position'], origin)])
                                     for w in record['waypoints']] if record else []
                        route = waypoints[record['waypoint_index']:] if record else []
                        # The drawn route (the sequencer's Catmull-Rom curve through
                        # the start, the waypoints and the pad) is the route corridor.
                        path_points = ([nav.start[0].tolist()] + [w['position'] for w in waypoints]
                                       + [env._target_position[0].tolist()]) if record else None
                        action = convex.compute_action(
                            obs, reference_position=(nav.goal[0] if nav else env._target_position[0]).tolist(),
                            route=route, hold_elapsed_s=record['hold_elapsed_s'] if record else 0.,
                            bus_voltage_v=float(env._battery_model.voltage_v[0]) if env._battery_model is not None else None,
                            path_points=path_points, path_index=record['waypoint_index'] if record else 0)
                    obs_dict, _, terminated, truncated, info = env.step(action)
                    cumulative_delta_v += float(info['propulsive_delta_v_step'][0])
                    obs = obs_dict['policy']
                    servo = env._reset_manager.servo_state
                    capture((step + 1) * dt, action, (servo - last_target) / dt)
                    last_target = servo.clone()
                    if step % 10 == 0:
                        update(frames=frames, sim_time_s=latest['t'])
                    pump_viewer()
                    if bool((terminated | truncated)[0]):
                        break
                if latest['contact'] == int(ContactState.LANDED) and not cancelled:
                    # A finite PPO episode ends at physical LANDED. The flight
                    # executive then disarms explicitly; no controller alters
                    # the learned approach or generates a fake landing event.
                    landing_frame = latest
                    control_phase = 'POST_TOUCHDOWN_DISARM'
                    update(phase='Motor shutdown and settling', frames=frames)
                    start_t = latest['t']
                    stable_samples = []
                    unsafe_shutdown = False
                    for tail_step in range(math.ceil(2. / dt)):
                        if (output / 'STOP').exists():
                            cancelled = True
                            break
                        action = torch.zeros(1, 5, device=device)
                        if flight:
                            # A zero rate command would hold throttle; the
                            # disarm cuts the flight computer's duty directly.
                            env._throttle_state.zero_()
                        obs_dict, _, _, _, info = env.step(action)
                        cumulative_delta_v += float(info['propulsive_delta_v_step'][0])
                        obs = obs_dict['policy']
                        servo = env._reset_manager.servo_state
                        capture(start_t + (tail_step + 1) * dt, action, (servo - last_target) / dt)
                        last_target = servo.clone()
                        attitude = float(tilt_angle(env._body_iface.get_root_quaternion_wxyz())[0])
                        unsafe_shutdown |= bool(env._unsafe_contact_step[0]) or attitude > .5
                        stable_samples.append(latest['contact_force_n'] >= 1.
                                              and math.sqrt(sum(v*v for v in latest['velocity'])) < .05
                                              and math.sqrt(sum(v*v for v in latest['gyro'])) < .15
                                              and attitude < .2 and latest['pad_distance'] <= .5)
                        pump_viewer()
                    window = math.ceil(.5 / dt)
                    settled_after_shutdown = (not cancelled and not unsafe_shutdown
                                              and len(stable_samples) >= window
                                              and all(stable_samples[-window:]))
        landed = latest['contact'] == int(ContactState.LANDED)
        landing_event_success = bool(landing_frame and landing_frame['impact_speed'] <= .25
                                     and landing_frame['pad_distance'] <= .5
                                     and (landing_frame['mission'] is None or landing_frame['mission']['ready_to_land']))
        success = landing_event_success and latest['impact_speed'] <= .25 and latest['pad_distance'] <= .5 and settled_after_shutdown is True
        premature = landed and latest['mission'] is not None and not latest['mission']['ready_to_land']
        failure = 'CRASHED'
        if flight and latest['mission']['outcome'] not in ('RUNNING', 'SUCCESS', 'CRASH'):
            failure = latest['mission']['outcome']  # e.g. GEOFENCE, SPIN, TILT
        outcome = ('CANCELLED' if cancelled else 'PREMATURE_LANDING' if premature else 'POST_LANDING_FAILURE' if landed and not settled_after_shutdown
                   else 'LANDED' if landed else failure if bool(terminated[0]) else 'TIMEOUT')
        summary = dict(outcome=outcome, success=success, duration_s=latest['t'], frames=frames,
                       simulation_wall_time_s=round(time.perf_counter() - simulation_started, 3),
                       real_time_factor=round(latest['t'] / max(time.perf_counter() - simulation_started, 1e-6), 4),
                       landing_duration_s=landing_frame['t'] if landing_frame else None,
                       landing_event_success=landing_event_success,
                       settled_after_shutdown=settled_after_shutdown,
                       flight_energy_wh=(landing_frame['battery']['energy_wh'] if landing_frame and landing_frame['battery'] else None),
                       propulsive_delta_v_m_s=cumulative_delta_v,
                       flight_rotation=env._rotation.record(),
                       mission=flight_record(env) if flight else env._navigation.record() if env._navigation else None,
                       impact_speed=latest['impact_speed'], pad_distance=latest['pad_distance'], battery=latest['battery'])
        atomic_json(output / 'summary.json', summary)
        update(state='cancelled' if cancelled else 'complete', phase=outcome, frames=frames,
               sim_time_s=latest['t'], summary=summary)
        return 0
    except Exception as exc:
        update(state='failed', phase='Simulation failed', error=str(exc))
        raise
    finally:
        watchdog.reset(30, label='Mission cleanup')
        if convex is not None:
            convex.guidance.close()
        if env is not None:
            env.close()
        if app is not None:
            close_simulation_app(app)
        watchdog.stop()


if __name__ == '__main__':
    force_process_exit(main())
