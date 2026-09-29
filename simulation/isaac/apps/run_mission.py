"""One real Isaac mission, streaming replay poses and telemetry to disk."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from runner_safety import WallClockWatchdog, force_process_exit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mission_control.models import validate_mission, battery_config, hardware_overrides, disturbance_config, vane_model_overrides
from mission_control.models import landing_pad
from mission_control.convex_parameters import resolve as resolve_convex_settings


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
    parser.add_argument('--convex-settings', type=Path, default=None,
                        help='Local diagnostic YAML deep-merged over configs/controllers/convex_guidance.yaml; '
                             'recorded in metadata, never used by the mission service')
    args = parser.parse_args()
    request = validate_mission(json.loads(args.request.read_text()))
    if request['controller'] != 'convex':
        raise ValueError('Only convex guidance is available')
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
        # CPU PhysX runs the same scene description; Kit must not reserve a
        # GPU physics pipeline for it.
        app = launch_simulation_app(headless=True, **(dict(device='cpu') if request['cpu_physics'] else {}))
        import torch
        from tvc_env.envs.base_env import BaseEnvConfig
        from tvc_env.envs.direct_rl_env import TVCDirectRLEnv
        from tvc_env.common.frames import frd_velocity_to_isaac
        from tvc_env.common.quaternions import rotate_vector, tilt_angle
        from tvc_env.common.constants import ContactState

        angles = [math.radians(x) for x in request['attitude_deg']]
        spawn = dict(position_range=[request['position']] * 2, velocity_range=[request['velocity']] * 2,
                     attitude_range=[angles] * 2, initial_motor_omega_fraction=request['initial_motor_fraction'])
        overrides = hardware_overrides(request)
        overrides.update(disturbance_config(request))
        overrides.update(env=dict(reset_on_crash=False, gizmos_enabled=False),
                         task=dict(spawn=spawn, episode_length_s=request['duration_s']),
                         battery=battery_config(request))
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
        # Convex missions fly the momentum-bounded vanes.
        from tvc_env.envs.task_registry import deep_merge
        overrides = deep_merge(overrides, vane_model_overrides(request))
        if request['cpu_physics']:
            # One articulation: the GPU pipeline spends a live mission's wall
            # time on kernel launches and host syncs at every 480 Hz substep.
            # CPU PhysX keeps the step size, TGS solver, iteration counts and
            # contact offsets; only the device changes (results are not
            # bit-identical to GPU PhysX, notably contact generation).
            from tvc_env.envs.task_registry import deep_merge
            overrides = deep_merge(overrides, dict(physics=dict(device='cpu')))
        torch.manual_seed(request['seed'])
        config = BaseEnvConfig(task_name='landing', sim_root=ROOT,
                               env_config_path=ROOT / 'configs/env/single_env_debug.yaml',
                               overrides=overrides)
        update(phase='Building physics scene')
        env = TVCDirectRLEnv(config)
        dt = config.physics_dt * config.decimation
        device = env.device
        convex = None
        policy_meta = dict(controller=request['controller'])
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
        # The simulated IMU was power-cycled by reset() before this override; without re-seeding it its
        # first substep sees the velocity jump as a huge acceleration and the navigation starts wrong.
        env._reset_imu(torch.arange(config.num_envs, device=device))
        obs = env._get_observations()['policy']
        if convex:
            convex.reset()
        metadata = dict(schema_version=2, request=request, policy=policy_meta, dt=dt,
                        live_execution=dict(fast_live=request['fast_live'], telemetry_stride=1,
                                            physics_device=str(device),
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
                        terminal_procedure='After LANDED: zero fin/throttle commands for 2 seconds; record actual spool-down and verify final settling. This procedure is outside the simulated episode.',
                        telemetry_notes='Fin command rate is servo target motion per control interval; body-rate targets are not commanded by the controller.',
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
                # IMU overlay still uses this; batched host transfer below covers
                # the PhysX/actuator fields without per-field syncs.
                def vec(x):
                    return x[0].detach().cpu().tolist()
                art = env._drone.data
                ids = env._art_map.fin_body_indices
                debug = env._last_dynamics_debug if hasattr(env, '_last_dynamics_debug') else {}
                raw = env._edf_model.compute_thrust(state.motor_omega)[0]
                battery = env._battery_model.telemetry() if env._battery_model else {}
                # Env 0's telemetry in one device-to-host copy instead of one
                # sync per field. float64 holds every float32/integer value
                # exactly; each field gets its own Python type back (battery
                # cutoff flags stay booleans), so frames.jsonl is unchanged.
                fields = dict(position=state.position[0], quaternion=state.quaternion_wxyz[0],
                              velocity=state.linear_vel_world[0], gyro=state.angular_vel_frd[0],
                              fin_angles=state.fin_angles[0], fin_rates=state.fin_rates[0],
                              fin_commands=action[0, :4], fin_command_rates=rate[0],
                              fin_positions=art.body_pos_w[0, ids], fin_quaternions=art.body_quat_w[0, ids],
                              throttle=action[0, 4], motor_omega=state.motor_omega[0],
                              thrust_n=debug['edf_applied_thrust_N'][0] if debug else raw, raw_thrust_n=raw,
                              contact=state.contact_state[0], impact_speed=env._touchdown_speed[0],
                              pad_distance=(state.position[0, :2] - env._target_position[0, :2]).norm(),
                              contact_force_n=env._landing_contact_force_step[0],
                              **{f'battery.{k}': v[0] for k, v in battery.items()})
                flat = torch.cat([v.detach().reshape(-1).to(device, torch.float64) for v in fields.values()]).cpu().tolist()
                host = {}
                for key, value in fields.items():
                    count = value.numel()
                    cast = (bool if value.dtype == torch.bool else float if value.is_floating_point() else int)
                    chunk, flat = [cast(x) for x in flat[:count]], flat[count:]
                    host[key] = (chunk[0] if value.dim() == 0 else
                                 [chunk[i:i + value.shape[-1]] for i in range(0, count, value.shape[-1])]
                                 if value.dim() == 2 else chunk)
                frame = dict(t=t, control_phase=control_phase, position=host['position'], quaternion=host['quaternion'],
                             velocity=host['velocity'], gyro=host['gyro'],
                             # The state the controller acted on: what the IMU and
                             # position sensor reported, sensor noise included.
                             # Equal to the PhysX state when noise is off.
                             imu=imu_record(env.sensor_measurement, vec),
                             # Rangefinder / flow / baro readings and the fusion filter's health.
                             fusion=env._fusion.record() if env._fusion is not None else None,
                             fin_angles=host['fin_angles'], fin_rates=host['fin_rates'],
                             fin_commands=host['fin_commands'], fin_command_rates=host['fin_command_rates'],
                             fin_positions=host['fin_positions'], fin_quaternions=host['fin_quaternions'],
                             throttle=host['throttle'], rotor_rpm=host['motor_omega'] * 60 / (2 * math.pi),
                             thrust_n=host['thrust_n'], raw_thrust_n=host['raw_thrust_n'],
                             battery={k: host[f'battery.{k}'] for k in battery} if env._battery_model else None,
                             contact=int(host['contact']), impact_speed=host['impact_speed'],
                             pad_distance=host['pad_distance'],
                             contact_force_n=host['contact_force_n'],
                             propulsive_delta_v_m_s=cumulative_delta_v,
                             rotation=env._rotation.record(),
                             mission=env._navigation.record() if env._navigation else None,
                             # Convex guidance: phase, plan reference and solver
                             # statistics; the planned path only on replan frames.
                             guidance=(convex.last_telemetry or None) if convex and control_phase == 'CONTROLLER' else None,
                             body_rate_command=None)
                stream.write(json.dumps(frame, allow_nan=False) + '\n')
                frames += 1
                latest = frame
            initial = torch.zeros(1, 5, device=device)
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
                    # The episode ends at physical LANDED. The flight executive
                    # then disarms explicitly; no controller alters the approach
                    # or generates a fake landing event.
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
                       mission=env._navigation.record() if env._navigation else None,
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
