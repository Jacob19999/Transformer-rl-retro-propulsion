# Data Model: Isaac Sim Vectorized Drone Simulation Environment

**Branch**: `001-isaac-sim-env` | **Date**: 2026-03-10

---

## Entities

### 1. DroneAsset

Represents the physical drone in the simulation world, constructed programmatically from YAML configuration.

**Fields**:
- `body_prim_path` (string): USD scene graph path for the root rigid body (e.g., `/World/envs/env_0/Drone`)
- `fin_prim_paths` (list[string], len=4): USD paths for each fin articulation prim
- `total_mass` (float, kg): Composite mass summed from YAML primitives
- `center_of_mass` (vector3, m): Composite CoM in body frame, computed from YAML primitives
- `inertia_tensor` (matrix3x3, kg·m²): Composite inertia about CoM, computed via parallel-axis theorem
- `fin_hinge_axes` (list[vector3], len=4): Rotation axis for each fin joint (body frame)
- `fin_positions` (list[vector3], len=4): Fin attachment point positions (body frame, m)
- `fin_deflection_limit` (float, rad): Symmetric joint limit applied to all fins (default: 0.2618 rad = 15°)
- `thrust_offset` (vector3, m): EDF thrust application point in body frame

**Validation rules**:
- `total_mass` > 0
- `fin_deflection_limit` ∈ (0, π/4] (no more than 45°)
- All `fin_positions` must be distinct
- `inertia_tensor` must be positive-definite

**State transitions**: Created once at environment initialization; rebuilt when YAML config changes.

---

### 2. DynamicsState (per environment instance)

Tracks the complete physical state of one drone instance across simulation steps.

**Fields**:
- `position` (vector3, m): World-frame position [x, y, z] (NED convention)
- `quaternion` (vector4): Attitude quaternion [qx, qy, qz, qw] (scalar-last)
- `linear_velocity_body` (vector3, m/s): Velocity expressed in body frame (FRD)
- `angular_velocity_body` (vector3, rad/s): Angular rate in body frame (FRD)
- `thrust_actual` (float, N): Current thrust output (after first-order lag)
- `thrust_cmd` (float, N): Current thrust command (normalized [-1,1] → [0, T_max])
- `fin_deflections_actual` (vector4, rad): Current fin angles (after servo lag)
- `fin_deflections_cmd` (vector4, rad): Current fin deflection commands

**State transitions**:
- Reset: Sampled from randomized initial condition distributions
- Step: Updated by PhysX + custom force application each dt = 1/120 s

---

### 3. EnvironmentInstance

A self-contained simulation unit containing one drone and one arena.

**Fields**:
- `env_id` (int): Unique index in the vectorized batch [0, num_envs)
- `episode_step` (int): Current step count within episode
- `episode_time` (float, s): Elapsed simulation time in current episode
- `is_done` (bool): Whether the episode has terminated (crashed or landed successfully)
- `dynamics_state` (DynamicsState): Current physical state
- `reward_last` (float): Reward computed at last step

**Relationships**: Many EnvironmentInstances share one ArenaScene and one DroneAsset definition.

---

### 4. ArenaScene

The static world environment (landing pad, ground plane, walls/boundaries).

**Fields**:
- `dimensions` (vector2, m): Arena footprint [width, depth] = [10.0, 10.0]
- `landing_pad_position` (vector3, m): Center of landing pad in world frame (default: [0, 0, 0])
- `landing_pad_radius` (float, m): Radius of circular landing pad (default: 0.5 m)
- `ground_friction` (float): PhysX friction coefficient for landing pad surface (0.5)
- `wall_height` (float, m): Arena boundary wall height (default: 0.0 — open arena initially)

---

### 5. ObservationVector

The 20-dimensional observation returned to the control policy each step.

**Fields** (indices):
- `[0:3]` `e_p_body` (vector3): Position error relative to target, expressed in body frame (m)
- `[3:6]` `v_body` (vector3): Velocity in body frame (m/s)
- `[6:9]` `g_body` (vector3): Gravity direction in body frame (unit vector, FRD +z = down)
- `[9:12]` `omega` (vector3): Angular velocity in body frame (rad/s)
- `[12]` `twr` (float): Thrust-to-weight ratio (dimensionless)
- `[13:16]` `wind_ema` (vector3): Exponential moving average of wind in body frame (m/s); zero-initialized in Isaac Sim (no wind model initially)
- `[16]` `h_agl` (float): Altitude above ground level (m, clamped ≥ 0)
- `[17]` `speed` (float): Linear speed magnitude (m/s)
- `[18]` `ang_speed` (float): Angular speed magnitude (rad/s)
- `[19]` `time_frac` (float): Episode time fraction t/t_max, ∈ [0, 1]

**Validation rules**: All values finite; `h_agl` ≥ 0; `twr` ≥ 0; `time_frac` ∈ [0, 1].

---

### 6. ActionVector

The 5-dimensional control input applied each step.

**Fields**:
- `[0]` `thrust_cmd` (float, normalized [-1, 1]): Maps to [0, T_max] N (unidirectional thrust)
- `[1]` `delta_1_cmd` (float, normalized [-1, 1]): Fin 1 (right) deflection → [-δ_max, +δ_max] rad
- `[2]` `delta_2_cmd` (float, normalized [-1, 1]): Fin 2 (left) deflection → [-δ_max, +δ_max] rad
- `[3]` `delta_3_cmd` (float, normalized [-1, 1]): Fin 3 (forward) deflection → [-δ_max, +δ_max] rad
- `[4]` `delta_4_cmd` (float, normalized [-1, 1]): Fin 4 (aft) deflection → [-δ_max, +δ_max] rad

**Validation rules**: All values ∈ [-1, 1]; clipped before application.

---

### 7. IsaacEnvConfig

Configuration aggregating all parameters for the Isaac Sim environment, loaded from YAML.

**Fields**:
- `num_envs` (int): Number of parallel environment instances (1 for single-env; 128–1024 for training)
- `episode_length_steps` (int): Maximum steps per episode (default: 600 = 5 s at 1/120s)
- `physics_dt` (float, s): Simulation timestep = 1/120 ≈ 0.00833 (maps to `SimulationCfg.dt`)
- `decimation` (int): Physics steps per policy step = 1 (maps to `DirectRLEnvCfg.decimation`)
- `num_physics_substeps` (int): PhysX TGS substeps per physics step (default: 4, maps to `SimulationCfg.num_substeps`)
- `spawn_altitude_range` (vector2, m): [min, max] altitude at reset (default: [5.0, 10.0])
- `spawn_velocity_magnitude_range` (vector2, m/s): [min, max] |v| at reset (default: [0.0, 5.0])
- `vehicle_config_path` (string): Path to `default_vehicle.yaml`
- `domain_randomization_config_path` (string): Path to `domain_randomization.yaml`
- `target_position` (vector3, m): Landing target in world frame (default: [0, 0, 0])
- `max_episode_time` (float, s): Episode time limit (default: 5.0)

---

## State Transition Diagram

```
[init]
  │
  ▼
[IDLE] ─── reset() ──► [SPAWNING]
                              │
                              ▼ PhysX initializes state
                         [RUNNING]
                              │
                   step() × N │
                              ▼
                         [RUNNING] ──── is_done=True ──► [TERMINAL]
                                                                │
                                                                ▼
                                                           reset() ──► [SPAWNING]
```
