# Tasks: Phase 1 Isaac Sim EDF TVC Simulation Environment

**Input**: Design documents from `/specs/007-isaac-sim-env/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Included — the specification explicitly requires a 13-step incremental validation ladder (tests 00-12) and 6 unit test files as core deliverables (Constitution Principle III: Test-Driven Validation).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions
- All paths relative to repository root unless noted

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create project directory structure, package scaffolding, and build configuration

- [x] T001 Create full directory structure under `simulation/isaac/` per plan.md: `apps/`, `assets/usd/`, `assets/metadata/`, `configs/physics/`, `configs/vehicle/`, `configs/env/`, `configs/tasks/`, `configs/reward/`, `configs/disturbances/`, `configs/params/`, `configs/debug/`, `tvc_env/common/`, `tvc_env/asset/`, `tvc_env/dynamics/`, `tvc_env/sim/`, `tvc_env/envs/`, `tvc_env/tasks/`, `tvc_env/controllers/`, `tvc_env/telemetry/`, `tests/unit/`, `tests/sim/`, `tests/goldens/fin_force_curves/`, `tests/goldens/reaction_torque_curves/`, `tests/goldens/touchdown_cases/`
- [x] T002 Create Python package `__init__.py` files for `simulation/isaac/tvc_env/` and all subpackages: `common/`, `asset/`, `dynamics/`, `sim/`, `envs/`, `tasks/`, `controllers/`, `telemetry/`
- [x] T003 Create `simulation/isaac/pyproject.toml` with editable install config for tvc_env package, declaring dependencies: Isaac Lab 2.3.2, PyTorch >= 2.0, NumPy >= 1.24, PyYAML, pytest

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core frame/math utilities that MUST be complete before ANY user story can be implemented

**CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 [P] Implement FRD-to-Isaac and Isaac-to-FRD frame conversion functions in `simulation/isaac/tvc_env/common/frames.py` — single canonical conversion boundary per research decision R5, including position, velocity, and force vector transforms between body-FRD (x=fwd, y=right, z=down) and Isaac/world (x=right, y=up, z=back or as defined by Isaac convention)
- [x] T005 [P] Implement quaternion utilities in `simulation/isaac/tvc_env/common/quaternions.py` — (w,x,y,z) convention per Isaac Lab 2.3.2, including: multiply, inverse, rotate_vector, to_rotation_matrix, from_euler, to_euler, normalize, identity, and convention converters `isaac_to_body_quat(wxyz→xyzw)` / `body_to_isaac_quat(xyzw→wxyz)` for constitution boundary
- [x] T006 [P] Implement rotation and translation utilities in `simulation/isaac/tvc_env/common/transforms.py` — quat_apply, quat_conjugate, transform_points, local_to_world, world_to_local, compute_heading, axis_angle_to_quat
- [x] T007 [P] Implement physical constants and enumerations in `simulation/isaac/tvc_env/common/constants.py` — gravity vector, air density, ContactState enum (AIRBORNE=0, GROUND_CONTACT_CANDIDATE=1, LANDED=2, CRASHED=3), DispatchMode enum (PER_LINK_FORCE, COLLAPSED_BODY_WRENCH)
- [x] T008 [P] Implement typed data structures in `simulation/isaac/tvc_env/common/datatypes.py` — FinForceResult (force_vector, normal_force, tangential_force, thrust_loss), EDFOutput (thrust_force, static_reaction_torque, dynamic_spool_torque, gyro_precession_torque), VehicleState dataclass
- [x] T009 [P] Write unit tests for frame conversions in `simulation/isaac/tests/unit/test_frames.py` — round-trip FRD↔Isaac identity, known vector transforms, batch tensor operations
- [x] T010 [P] Write unit tests for quaternion operations in `simulation/isaac/tests/unit/test_quaternions.py` — multiply, rotate_vector, convention conversion round-trip, edge cases (identity, 180° rotation)

**Checkpoint**: Foundation ready — user story implementation can now begin

---

## Phase 3: User Story 1 — Asset Loading and Validation (Priority: P1) MVP

**Goal**: Load the EDF drone USD asset and validate all structural requirements (links, joints, axes, mass, hierarchy), failing fast with diagnostics on any violation.

**Independent Test**: Load USD asset, run asset validation suite, confirm pass/fail for each structural check.

**Dependencies**: Phase 2 complete

### Implementation for User Story 1

- [x] T011 [P] [US1] Create vehicle configuration YAML at `simulation/isaac/configs/vehicle/edf_drone_v2.yaml` per config_schema contract — total_mass, body_com_offset, inertia_tensor (Ixx/Iyy/Izz), fins (count, area, max_deflection, cop_offset), with source labels on every parameter
- [x] T012 [P] [US1] Create asset metadata YAML at `simulation/isaac/assets/metadata/edf_drone_v2.asset.yaml` — body_link_name, fin_link_names[4], fin_joint_names[4], hinge_axes[4], joint_limits, fin_cop_positions[4], fin_chord_directions[4], fin_normal_directions[4], edf_thrust_axis, rotor_spin_axis, landing_contact_regions
- [x] T013 [US1] Implement USD loading and prim access in `simulation/isaac/tvc_env/asset/usd_loader.py` — load USD scene, resolve prim paths, extract ArticulationRootAPI, RigidBodyAPI, MassAPI from prims, return structured asset data using YAML metadata for name mapping
- [x] T014 [P] [US1] Implement link/joint name-to-index mapping in `simulation/isaac/tvc_env/asset/articulation_map.py` — map fin_link_names and fin_joint_names from metadata to Isaac Lab articulation body/joint indices, provide lookup methods by name or position (+X, +Y, -X, -Y)
- [x] T015 [P] [US1] Implement hinge axis extraction in `simulation/isaac/tvc_env/asset/hinge_axis_extractor.py` — extract revolute joint axes from USD PhysicsRevoluteJoint prims, validate axes are unit vectors along cardinal directions, compare against metadata YAML
- [x] T016 [US1] Implement mass property extraction and validation in `simulation/isaac/tvc_env/asset/mass_properties.py` — extract mass, COM offset, inertia from USD MassAPI, compare against `edf_drone_v2.yaml` with 1% tolerance per constitution, log warnings for mismatches
- [x] T017 [US1] Implement fail-fast structural validation in `simulation/isaac/tvc_env/asset/asset_validator.py` — validate: body link exists, 4 fin links exist, 4 revolute joints with defined axes, joint limits match config, mass properties valid, link hierarchy consistent; raise descriptive errors on any failure
- [x] T018 [US1] Write simulation test in `simulation/isaac/tests/sim/test_00_asset_validation.py` — test valid asset passes all checks, test missing fin link causes diagnostic failure, test undefined joint axis causes diagnostic failure
- [x] T019 [US1] Implement test runner script at `simulation/isaac/apps/run_single_test.py` — bootstrap Isaac Sim runtime, accept `--test` argument to select test module from `tests/sim/`, accept `--physics` for solver config override, run selected test and report results

**Checkpoint**: Asset loads and validates correctly — downstream stories can build on a proven asset foundation

---

## Phase 4: User Story 2 — Frame-Correct Fin Articulation and Force Application (Priority: P1)

**Goal**: Command individual fin deflections along correct hinge axes with correct sign convention, apply forces at fin COP, and verify body reaction through articulation physics.

**Independent Test**: Command known fin angles, verify axis/sign/limits. Apply synthetic unit forces at fin COPs, check body reaction r x F direction and magnitude.

**Dependencies**: US1 complete (asset loaded and validated)

### Implementation for User Story 2

- [x] T020 [US2] Implement fin spatial layout and COP position computation in `simulation/isaac/tvc_env/dynamics/fin_geometry.py` — compute fin COP positions in body frame from metadata, provide fin-local-to-body transforms, fin ordering validation (+X, +Y, -X, -Y)
- [x] T021 [US2] Implement InteractiveScene setup and environment cloning in `simulation/isaac/tvc_env/sim/scene_builder.py` — create InteractiveSceneCfg with configurable num_envs/env_spacing/replicate_physics, spawn drone articulation and ground plane, clone environments, filter inter-env collisions per research R3
- [x] T022 [US2] Implement articulation state read/write interface in `simulation/isaac/tvc_env/sim/body_interface.py` — read root_state_w (position, quaternion wxyz, linear/angular velocity), read/write joint positions and velocities, provide body-frame velocity computation using frames.py conversion
- [x] T023 [US2] Implement per-link force application at COP in `simulation/isaac/tvc_env/sim/link_force_interface.py` — wrap `Articulation.set_external_force_and_torque()` with `positions` parameter per research R2, accept force vectors and COP offsets per fin link body_id, handle write_data_to_sim() sequencing
- [x] T024 [P] [US2] Write unit test for fin geometry in `simulation/isaac/tests/unit/test_fin_geometry.py` — verify COP positions for each fin, verify fin-local-to-body transforms, verify fin ordering consistency
- [x] T025 [US2] Write simulation test in `simulation/isaac/tests/sim/test_01_joint_axes.py` — for each fin joint, command positive deflection with gravity disabled, verify rotation occurs around the correct hinge axis in the expected direction
- [x] T026 [US2] Write simulation test in `simulation/isaac/tests/sim/test_02_single_fin_articulation.py` — command each fin to known angles, verify actual joint position matches command within tolerance, verify joint limits are respected (clamping at max_deflection)
- [x] T027 [US2] Write simulation test in `simulation/isaac/tests/sim/test_03_unit_force_on_fin.py` — apply unit force at each fin COP with gravity disabled, simulate for N steps, verify body reaction direction matches expected r x F cross product, verify sign correctness

**Checkpoint**: Fin articulation proven correct in axis direction, sign, limits, and force-at-COP body reaction

---

## Phase 5: User Story 3 — Per-Fin Aerodynamic Force Modeling (Priority: P1)

**Goal**: Implement subsonic semi-empirical vane aero model with physically plausible force behavior: near-zero normal force at zero deflection, linear at small angles, saturation at large angles.

**Independent Test**: Sweep single fin through deflection range, record normal/tangential force components, compare curves against expected characteristics.

**Dependencies**: US2 complete (fin articulation and force application proven)

### Implementation for User Story 3

- [x] T028 [US3] Implement semi-empirical jet-vane aero model in `simulation/isaac/tvc_env/dynamics/fin_aero.py` — compute per-fin normal force F_n = q*S*C_N(α) with saturation (C_N = C_N_α*α*(1 - k_sat*α²)), tangential force F_t = q*S*C_D(α) with drag-vs-angle² model, dynamic pressure from exhaust speed, aspect ratio and duct confinement corrections, vectorized for (num_envs, 4) fins per research R7
- [x] T029 [US3] Implement per-fin force computation pipeline in `simulation/isaac/tvc_env/dynamics/fin_force_dispatch.py` — orchestrate: get actual fin angles from servo state, compute aero forces via fin_aero.py, transform force vectors from fin-local to body frame using fin_geometry.py, output per-fin force vectors at COP positions ready for dispatch
- [x] T030 [US3] Implement force dispatch mode switching in `simulation/isaac/tvc_env/sim/wrench_dispatch.py` — abstract dispatch layer per research R10: `per_link_force` mode applies forces via link_force_interface.py to each fin link at COP, `collapsed_body_wrench` mode sums all forces into net body wrench, mode selected from env config `dispatch_mode` field
- [x] T031 [P] [US3] Write unit test for aero model in `simulation/isaac/tests/unit/test_fin_aero.py` — verify near-zero normal force at zero deflection, verify approximately linear response at small angles, verify saturation at large angles, verify drag increases with deflection magnitude, verify vectorized computation across 4 fins
- [x] T032 [US3] Write simulation test in `simulation/isaac/tests/sim/test_04_fin_force_sweep.py` — sweep single fin from -max to +max deflection, record normal and tangential force at each angle, verify force curve qualitative characteristics match expected behavior, save curves to `tests/goldens/fin_force_curves/`
- [x] T033 [US3] Write simulation test in `simulation/isaac/tests/sim/test_05_four_fin_superposition.py` — command four fins in known patterns (roll-only, pitch-only, yaw-only, symmetric), verify resultant body forces and torques match expected directions and relative magnitudes

**Checkpoint**: Fin aero model produces physically plausible forces, four-fin superposition gives correct roll/pitch/yaw moments

---

## Phase 6: User Story 4 — EDF Propulsion with Realistic Dynamics (Priority: P1)

**Goal**: Implement EDF thrust with spool dynamics, static/dynamic reaction torques, and gyroscopic precession. Implement MG996R servo actuator model.

**Independent Test**: Command motor step inputs, verify thrust lag timing. Impose body angular velocity with spinning rotor, check precession direction and magnitude.

**Dependencies**: US2 complete (scene and body interface available). Can proceed in parallel with US3.

### Implementation for User Story 4

- [x] T034 [P] [US4] Create servo parameter config at `simulation/isaac/configs/params/servo_mg996r.yaml` per config_schema contract — mass, stall_torque, transit_time_60deg, max_angular_velocity (derived), tau_servo (estimate), deadband (estimate), max_command_angle (measured), all with source labels
- [x] T035 [P] [US4] Create EDF parameter config at `simulation/isaac/configs/params/edf_90mm.yaml` per config_schema contract — max_thrust (estimate), diameter (measured), k_T/k_Q (to-be-calibrated: null), rotor_inertia (estimate), tau_motor (estimate), omega_max/d_omega_max (to-be-calibrated: null), all with source labels
- [x] T036 [P] [US4] Implement MG996R servo dynamics model in `simulation/isaac/tvc_env/dynamics/actuator_servo.py` — first-order lag (ẋ = (x_cmd - x) / τ_servo), angular rate limiting to ±ω_max_servo, clamping to ±max_command_angle, optional deadband, vectorized state update for (num_envs, 4) servos per data-model ServoActuator entity
- [x] T037 [P] [US4] Implement EDF thrust and spool dynamics in `simulation/isaac/tvc_env/dynamics/propulsion_edf.py` — throttle-to-RPM mapping (ω_target = throttle * ω_max), first-order motor lag with τ_motor, RPM rate limiting to ±dω_max, thrust = k_T * ω², vectorized for num_envs per data-model EDFPropulsion entity
- [x] T038 [US4] Implement rotor reaction torque computation in `simulation/isaac/tvc_env/dynamics/rotor_reaction.py` — static reaction torque (opposes spin, Q = k_Q * ω²), dynamic spool torque (I_rotor * dω/dt), gyroscopic precession (ω_body × H_rotor where H_rotor = I_rotor * ω_rotor * spin_axis), all outputs as separate vec3 tensors for logging per FR-018
- [x] T039 [P] [US4] Write unit test for rotor torque computations in `simulation/isaac/tests/unit/test_rotor_reaction.py` — verify static torque opposes spin direction, verify dynamic spool torque sign during accel/decel, verify gyro precession direction follows ω × H, verify magnitudes against hand calculations
- [x] T040 [US4] Write simulation test in `simulation/isaac/tests/sim/test_06_edf_spool_and_reaction.py` — command motor step from 0% to 100%, measure thrust response over time, verify first-order lag within 10% of configured τ_motor, verify static reaction torque opposes spin, log all torque components separately, save curves to `tests/goldens/reaction_torque_curves/`
- [x] T041 [US4] Write simulation test in `simulation/isaac/tests/sim/test_07_gyro_precession.py` — spin rotor to steady speed, impose body angular velocity around each axis, measure precession torque, verify direction follows ω_body × H_rotor, verify magnitude is proportional to I_rotor * ω_rotor * ω_body

**Checkpoint**: All actuator dynamics proven — EDF spool lag, reaction torques, gyro precession, servo lag/rate-limit all verified

---

## Phase 7: User Story 5 — Contact State Machine for Landing/Crash Detection (Priority: P2)

**Goal**: Implement 4-state contact state machine (AIRBORNE → GROUND_CONTACT_CANDIDATE → LANDED | CRASHED) with dwell-interval thresholds for correct landing/crash classification.

**Independent Test**: Run scripted drop tests at various speeds and angles, verify state transitions and final classifications.

**Dependencies**: US2 complete (scene builder, body interface, sensor access)

### Implementation for User Story 5

- [x] T042 [US5] Implement contact and IMU sensor access in `simulation/isaac/tvc_env/sim/sensor_interface.py` — read contact forces from PhysX contact reporter, detect ground contact per landing-gear contact regions from asset metadata, provide contact normal vectors and impact velocities
- [x] T043 [US5] Implement 4-state contact state machine in `simulation/isaac/tvc_env/sim/contacts.py` — vectorized state tensor [num_envs], dwell counter tracking, transition logic per data-model ContactStateMachine: AIRBORNE→CANDIDATE on contact, CANDIDATE→LANDED when all dwell criteria met for N frames, CANDIDATE→AIRBORNE on bounce, CANDIDATE/AIRBORNE→CRASHED on crash triggers, all thresholds from task YAML config
- [x] T044 [US5] Implement crash detection criteria in `simulation/isaac/tvc_env/sim/crash_logic.py` — impact speed threshold, excessive tilt on contact, excessive angular rate on contact, unsafe body contact (non-landing-gear collision), tip-over after initial contact, all thresholds configurable, vectorized evaluation for num_envs
- [x] T045 [P] [US5] Write unit test for crash logic in `simulation/isaac/tests/unit/test_crash_logic.py` — verify each crash criterion triggers independently, verify below-threshold does not trigger, verify vectorized evaluation across multiple envs
- [x] T046 [US5] Write simulation test in `simulation/isaac/tests/sim/test_09_contact_landed_crash.py` — scripted soft touchdown (verify AIRBORNE→CANDIDATE→LANDED), scripted bounce (verify CANDIDATE→AIRBORNE without false LANDED), scripted hard impact (verify →CRASHED), scripted tip-over after contact (verify →CRASHED), save cases to `tests/goldens/touchdown_cases/`

**Checkpoint**: Contact state machine correctly classifies soft landing, bounce, hard impact, and tip-over scenarios

---

## Phase 8: User Story 6 — Task-Configurable Hover and Landing Modes (Priority: P2)

**Goal**: Support hover and landing tasks through configuration only — same physics core, different reward profiles, success criteria, termination logic, and spawn ranges selected by task YAML.

**Independent Test**: Run environment with hover config then landing config; verify physics identical but rewards/termination/success differ.

**Dependencies**: US3 + US4 + US5 complete (full physics pipeline + contacts needed for env step loop)

### Implementation for User Story 6

- [x] T047 [P] [US6] Create hover task config at `simulation/isaac/configs/tasks/hover.yaml` per config_schema contract — target_position, episode_length_s, spawn ranges, reward weights (alive_bonus, position_error, attitude_error, angular_velocity, control_effort, control_rate, hover_stability, drift_penalty, contact_penalty), success criteria (max_position_error, max_tilt, max_angular_rate, dwell_time), termination conditions
- [x] T048 [P] [US6] Create landing task config at `simulation/isaac/configs/tasks/landing.yaml` per config_schema contract — target_position [0,0,0], episode_length_s, spawn ranges, reward weights (alive_bonus, position_error, attitude_error, crash_penalty, touchdown_softness, landing_success, pad_accuracy, vertical_speed_shaping), success criteria (state: LANDED, max_pad_distance), termination conditions
- [x] T049 [P] [US6] Create reward config YAMLs: `simulation/isaac/configs/reward/common_terms.yaml` (shared term defaults), `simulation/isaac/configs/reward/hover_reward.yaml` (hover weights), `simulation/isaac/configs/reward/landing_reward.yaml` (landing weights)
- [x] T050 [US6] Implement reward term registry in `simulation/isaac/tvc_env/envs/reward_registry.py` — map string term names to reward functions per research R9, registry pattern: `{"alive_bonus": fn, "position_error": fn, ...}`, each function signature `fn(env_state, config) → Tensor`, register shared + task-specific terms
- [x] T051 [US6] Implement reward term functions in `simulation/isaac/tvc_env/envs/rewards.py` — shared terms: alive_bonus, position_error, attitude_error, angular_velocity, control_effort, control_rate, saturation, crash_penalty; hover-only: hover_stability, drift_penalty, contact_penalty; landing-only: touchdown_softness, landing_success, pad_accuracy, vertical_speed_shaping; all vectorized (num_envs,) output
- [x] T052 [US6] Implement observation vector assembly in `simulation/isaac/tvc_env/envs/observations.py` — assemble 24-dim (or 27-dim with wind) observation tensor per observation_space contract: position_error[3], attitude_quat_wxyz[4], linear_vel_body_frd[3], angular_vel_body_frd[3], height[1], fin_angles[4], fin_rates[4], motor_rpm_normalized[1], contact_state[1], optional wind_estimate[3]
- [x] T053 [US6] Implement termination condition checks in `simulation/isaac/tvc_env/envs/terminations.py` — max tilt exceeded, max altitude error exceeded, crash state reached, episode timeout; all configurable from task YAML, vectorized boolean tensor output
- [x] T054 [US6] Implement success condition checks in `simulation/isaac/tvc_env/envs/success_criteria.py` — hover success: position error + tilt + angular rate within tolerance for dwell_time seconds; landing success: contact state == LANDED and pad distance within tolerance; vectorized boolean tensor output
- [x] T055 [US6] Implement task name-to-config resolver in `simulation/isaac/tvc_env/envs/task_registry.py` — resolve task name string ("hover"/"landing") to task YAML path, load and merge task config with deep-merge loader (base → env → task → disturbance → CLI overrides per config_schema contract)
- [x] T056 [US6] Implement episode reset with randomized initial conditions in `simulation/isaac/tvc_env/sim/reset_logic.py` — sample position, velocity, attitude from spawn ranges in task config, set root state via body_interface, reset servo/EDF actuator states, reset contact state machine, vectorized per-env reset
- [x] T057 [US6] Implement shared DirectRLEnv base in `simulation/isaac/tvc_env/envs/base_env.py` — base class with config loading, YAML deep-merge, common initialization (scene, asset, actuators, contacts), shared step infrastructure, config validation (reject null to-be-calibrated values before training)
- [x] T058 [US6] Implement DirectRLEnv in `simulation/isaac/tvc_env/envs/direct_rl_env.py` — subclass DirectRLEnv from Isaac Lab, implement `_setup_scene()` (scene_builder), `_pre_physics_step(actions)` (clamp + store actions per action_space contract), `_apply_action()` (servo dynamics → fin aero → force dispatch → wrench application, called decimation times), `_get_observations()` (via observations.py), `_get_rewards()` (via reward_registry), `_get_dones()` (via terminations.py), define action_space Box(5,) and observation_space Box(24,)
- [x] T059 [P] [US6] Implement hover task config adapter in `simulation/isaac/tvc_env/tasks/hover_task.py` — load hover.yaml, configure hover-specific reward terms/weights, success criteria, termination conditions, spawn ranges
- [x] T060 [P] [US6] Implement landing task config adapter in `simulation/isaac/tvc_env/tasks/landing_task.py` — load landing.yaml, configure landing-specific reward terms/weights, success criteria (LANDED state + pad distance), termination conditions, spawn ranges

**Checkpoint**: Environment runs with both hover and landing tasks via config-only switching, same physics core for both

---

## Phase 9: User Story 7 — Single-Environment Debug Visualization (Priority: P2)

**Goal**: Render all debug gizmos in single-env mode: body axes, COM marker, thrust vector, per-fin force arrows, contact normals, and HUD telemetry.

**Independent Test**: Launch single-env debug run, visually confirm all gizmos render correctly and HUD values update in real time.

**Dependencies**: US6 complete (environment running)

### Implementation for User Story 7

- [X] T061 [P] [US7] Create gizmo config YAML at `simulation/isaac/configs/debug/gizmos.yaml` — enable/disable flags and styling for each gizmo type (body_axes, com_marker, target_marker, thrust_vector, fin_force_arrows, total_aero_force, reaction_torque, contact_normals, hover_tolerance_volume), color/scale/opacity settings
- [X] T062 [P] [US7] Create single-env debug config at `simulation/isaac/configs/env/single_env_debug.yaml` per config_schema contract — num_envs: 1, env_spacing: 4.0, dispatch_mode: per_link_force, gizmos_enabled: true, decimation: 4, physics_dt: 0.00833
- [X] T063 [US7] Implement debug visualization manager in `simulation/isaac/tvc_env/sim/gizmos.py` — dual API per research R6: VisualizationMarkers for 3D shapes (FrameMarkerCfg for body axes, Arrow USD for thrust/force/torque vectors, SphereCfg for COM, CylinderCfg for target/pad), debug_draw for contact normal lines; update all markers each step from current env state; auto-disable when num_envs > 1; HUD overlay with altitude error, position error, tilt, body-rate magnitude, motor RPM, per-fin angles/forces, total reward, task mode, vehicle state
- [X] T064 [US7] Implement single-env wrapper in `simulation/isaac/tvc_env/envs/single_env.py` — subclass or wrap direct_rl_env with gizmos enabled, initialize gizmo manager, call gizmo update in post-step, keyboard/gamepad input handling for manual control
- [X] T065 [US7] Implement single-env debug app in `simulation/isaac/apps/run_single_env_debug.py` — accept `--task`, `--env-config`, `--disturbance` args, launch Isaac Sim viewport, instantiate single_env, run interactive loop with gizmos and manual control

**Checkpoint**: All gizmos visible in single-env mode, HUD telemetry updates each step

---

## Phase 10: User Story 8 — 128-Environment Vectorized Training (Priority: P2)

**Goal**: Launch 128 parallel environments with correct reset/observe/step/terminate across all instances, no tensor shape mismatches or NaN values, gizmos disabled for performance.

**Independent Test**: Run 128-env smoke test with random actions for 1000 steps, verify tensor shapes and value ranges.

**Dependencies**: US6 complete (DirectRLEnv implementation)

### Implementation for User Story 8

- [X] T066 [P] [US8] Create physics config YAMLs: `simulation/isaac/configs/physics/physx_single.yaml` (single-env PhysX), `simulation/isaac/configs/physics/physx_train.yaml` (GPU pipeline for training), `simulation/isaac/configs/physics/solver_high_fidelity.yaml` (high-fidelity solver for validation)
- [X] T067 [P] [US8] Create training env config at `simulation/isaac/configs/env/train_128.yaml` per config_schema contract — num_envs: 128, env_spacing: 4.0, dispatch_mode: per_link_force, gizmos_enabled: false, decimation: 4, physics_dt: 0.00833
- [X] T068 [US8] Implement per-reset domain randomization in `simulation/isaac/tvc_env/envs/domain_randomization.py` — randomize spawn position/velocity/attitude from task config ranges, optional mass/inertia perturbation, optional servo parameter variation, all randomization seeded for reproducibility per constitution Principle IV
- [X] T069 [US8] Implement 128-env smoke test app at `simulation/isaac/apps/run_smoke_128.py` — accept `--task`, `--env-config`, `--steps` args, instantiate vectorized env with 128 envs, run random actions for N steps, report tensor shape validation, NaN check, reset count, and performance metrics (steps/sec)
- [X] T070 [US8] Write simulation test in `simulation/isaac/tests/sim/test_11_rl_api_128env_smoke.py` — initialize 128 envs, reset all, step with random actions for 1000 steps, assert observation tensor shape (128, 24), reward tensor shape (128,), no NaN in observations or rewards, no tensor shape errors, verify independent per-env auto-reset on termination

**Checkpoint**: 128 environments run stably for 1000+ steps with zero NaN and zero shape errors

---

## Phase 11: User Story 9 — Wind and Disturbance Injection (Priority: P3)

**Goal**: Inject constant wind, gust pulses, COM offsets, and sensor noise through configuration. Vehicle responds physically with correct frame behavior.

**Independent Test**: Inject known constant wind, verify vehicle drifts in expected direction. Rotate vehicle and confirm frame-correct response.

**Dependencies**: US6 complete (environment running). Wind/COM dynamics modules can be developed in parallel with US7/US8.

### Implementation for User Story 9

- [X] T071 [P] [US9] Create disturbance config YAMLs: `simulation/isaac/configs/disturbances/nominal.yaml` (all disabled), `simulation/isaac/configs/disturbances/wind.yaml` (steady wind + gusts + body drag per config_schema), `simulation/isaac/configs/disturbances/sensor_noise.yaml` (observation noise std), `simulation/isaac/configs/disturbances/com_shift.yaml` (COM offset range)
- [X] T072 [P] [US9] Create wind drag coefficient config at `simulation/isaac/configs/params/wind_model.yaml` — body drag cd, reference area, exhaust speed at nominal throttle, air density
- [X] T073 [P] [US9] Implement wind and drag model in `simulation/isaac/tvc_env/dynamics/wind_model.py` — steady wind vector (world frame), gust event generation (magnitude, duration, random interval), body drag force (F_drag = 0.5 * ρ * cd * A * |v_rel|² * v_rel_hat where v_rel = v_body - v_wind), transform wind to body frame for drag computation, vectorized for num_envs
- [X] T074 [P] [US9] Implement center-of-mass offset model in `simulation/isaac/tvc_env/dynamics/com_model.py` — sample COM offset from configured range on reset, apply as force application point offset, vectorized for num_envs
- [X] T075 [US9] Implement state derivative utilities in `simulation/isaac/tvc_env/dynamics/state_deriv_helpers.py` — helper functions for computing derived quantities (relative airspeed, dynamic pressure, exhaust velocity at current throttle), used by wind_model and fin_aero
- [X] T076 [US9] Write simulation test in `simulation/isaac/tests/sim/test_08_wind_disturbance.py` — apply constant wind to hovering vehicle, verify drift direction matches wind vector, rotate vehicle 90° and verify drag still opposes relative airflow correctly, trigger gust event and verify transient force magnitude and duration

**Checkpoint**: Wind and disturbance framework works with correct frame behavior, gust events trigger on schedule

---

## Phase 12: User Story 10 — PID Hover Validation (Priority: P3)

**Goal**: Run PID controller with all modeled forces, actuator lags, and rotor dynamics enabled. Achieve bounded hover: position error < 0.5m, tilt < 15°, no NaN, no ground contact, for 10+ seconds.

**Independent Test**: Run PID hover scenario, check all stability metrics remain bounded for full test duration.

**Dependencies**: US6 + US9 complete (full environment with disturbances)

### Implementation for User Story 10

- [X] T077 [US10] Implement controller base interface in `simulation/isaac/tvc_env/controllers/base.py` — abstract base class defining `compute_action(obs) → action_tensor` method, common controller configuration loading, action space bounds validation
- [X] T078 [P] [US10] Implement PID controller adapter in `simulation/isaac/tvc_env/controllers/pid_adapter.py` — map PID output (altitude error → throttle, attitude error → roll/pitch/yaw commands) to 5-dim action vector per action_space contract, configurable gains, anti-windup
- [X] T079 [P] [US10] Implement PID fin mixing logic in `simulation/isaac/tvc_env/controllers/pid_fin_mixer.py` — convert roll/pitch/yaw rate commands to 4 individual fin deflection angles using fin geometry (fin positions at +X, +Y, -X, -Y), sign-correct mixing matrix based on fin hinge axes and COP positions
- [X] T080 [P] [US10] Implement PPO action adapter in `simulation/isaac/tvc_env/controllers/ppo_adapter.py` — interpret raw 5-dim network output as fin angles[4] + throttle[1], apply action scaling/clipping, pass through to environment action space
- [X] T081 [P] [US10] Implement GTrXL-PPO action adapter in `simulation/isaac/tvc_env/controllers/gtrxl_adapter.py` — interpret raw 5-dim network output as fin angles[4] + throttle[1], handle sequence context for transformer policy, apply action scaling/clipping
- [X] T082 [US10] Implement PID hover evaluation app at `simulation/isaac/apps/run_eval_pid.py` — accept `--task`, `--env-config`, `--disturbance`, `--duration` args, instantiate single env with PID controller, run for configured duration, log telemetry, report position error, tilt, angular rate statistics
- [X] T083 [P] [US10] Implement PPO training entrypoint at `simulation/isaac/apps/run_train_ppo.py` — accept `--task`, `--env-config`, `--disturbance`, `--seed`, `--total-steps` args, instantiate vectorized env, configure PPO with ppo_adapter, run training loop with timestamped output directory under `runs/`
- [X] T084 [P] [US10] Implement GTrXL-PPO training entrypoint at `simulation/isaac/apps/run_train_gtrxl.py` — accept `--task`, `--env-config`, `--disturbance`, `--seed`, `--total-steps` args, instantiate vectorized env, configure GTrXL-PPO with gtrxl_adapter, run training loop with timestamped output directory under `runs/`
- [X] T085 [US10] Write simulation test in `simulation/isaac/tests/sim/test_10_pid_hover_smoke.py` — run PID hover for 10+ seconds with all physics effects enabled, assert position error < 0.5m, tilt < 15° (0.26 rad), angular rate < 1.0 rad/s, no NaN in any state variable, no ground contact
- [X] T086 [US10] Write simulation test in `simulation/isaac/tests/sim/test_12_steady_hover_all_forces.py` — run PID hover with wind disturbance enabled, log all torque contributions (fin, static reaction, dynamic spool, gyro precession, wind drag) separately per FR-018, verify all torque magnitudes are physically reasonable relative to each other, verify no sign-error-induced divergence

**Checkpoint**: PID achieves bounded hover — full environment validated end-to-end before RL training

---

## Phase 13: Polish & Cross-Cutting Concerns

**Purpose**: Telemetry infrastructure, documentation, and final validation

- [X] T087 [P] Implement per-step telemetry logger in `simulation/isaac/tvc_env/telemetry/logger.py` — log observation vector, action vector, reward, all torque contributions, contact state, episode metrics per step to structured format (CSV or HDF5)
- [X] T088 [P] Implement aggregate episode metrics in `simulation/isaac/tvc_env/telemetry/metrics.py` — compute per-episode: mean/max position error, mean/max tilt, total reward, episode length, success/crash/timeout outcome, landing accuracy
- [X] T089 [P] Implement diagnostic plot generation in `simulation/isaac/tvc_env/telemetry/plots.py` — generate fin force curves, thrust response, torque comparison, trajectory, and state history plots from telemetry data
- [X] T090 [P] Implement episode data export in `simulation/isaac/tvc_env/telemetry/episode_export.py` — export episode telemetry to JSON/CSV for analysis, include metadata (task, config, seed, git hash)
- [X] T091 Create HIL validation env config at `simulation/isaac/configs/env/hil_validation.yaml` — HIL-oriented settings for hardware-in-the-loop validation scenarios
- [X] T092 Populate golden test data directories: `simulation/isaac/tests/goldens/fin_force_curves/` (reference aero curves from T032), `simulation/isaac/tests/goldens/reaction_torque_curves/` (reference torque curves from T040), `simulation/isaac/tests/goldens/touchdown_cases/` (reference contact cases from T046)
- [X] T093 Run complete validation ladder per `specs/007-isaac-sim-env/quickstart.md` — execute all 13 sim tests (test_00 through test_12) in order, run all 6 unit test files, confirm all pass, document results

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion — BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Foundational (Phase 2) — MVP entry point
- **US2 (Phase 4)**: Depends on US1 (loaded/validated asset)
- **US3 (Phase 5)**: Depends on US2 (fin articulation + force application proven)
- **US4 (Phase 6)**: Depends on US2 (scene + body interface). **Can run in parallel with US3** (independent physics models)
- **US5 (Phase 7)**: Depends on US2 (scene + sensor interface). **Can run in parallel with US3 and US4**
- **US6 (Phase 8)**: Depends on US3 + US4 + US5 (full physics pipeline + contacts)
- **US7 (Phase 9)**: Depends on US6 (environment running)
- **US8 (Phase 10)**: Depends on US6 (environment running). **Can run in parallel with US7**
- **US9 (Phase 11)**: Dynamics modules (T073-T075) can start after Phase 2. Sim test (T076) depends on US6.
- **US10 (Phase 12)**: Depends on US6 + US9 (full env with disturbances)
- **Polish (Phase 13)**: Depends on all user stories complete

### User Story Dependency Graph

```
Setup → Foundational → US1 → US2 ──┬── US3 ──┐
                                    ├── US4 ──┤
                                    └── US5 ──┴── US6 ──┬── US7 ──┐
                                                        ├── US8 ──┤
                                                        └── US9 ──┴── US10 → Polish
```

### Within Each User Story

- Config YAMLs before implementation modules that consume them
- Pure math/dynamics modules before sim integration modules
- Core implementation before tests
- Sim tests depend on all implementation within the story being complete

### Parallel Opportunities

**After Phase 2 (Foundational)**:
- All common/ unit tests (T009, T010) can run in parallel

**After US2 (Phase 4)**:
- US3, US4, and US5 can proceed in parallel (3 independent work streams)
- Within US4: T034, T035, T036, T037 can all run in parallel (independent files)

**After US6 (Phase 8)**:
- US7, US8, and US9 dynamics can proceed in parallel (3 independent work streams)
- Within US10: T078-T081 controller adapters can all run in parallel

**Within Polish (Phase 13)**:
- T087-T090 telemetry modules can all run in parallel

---

## Parallel Example: User Stories 3, 4, 5 (after US2 complete)

```bash
# Stream A: US3 — Per-Fin Aero Model
Task: "Implement fin_aero.py in simulation/isaac/tvc_env/dynamics/fin_aero.py"
Task: "Implement fin_force_dispatch.py in simulation/isaac/tvc_env/dynamics/fin_force_dispatch.py"
Task: "Write test_fin_aero.py in simulation/isaac/tests/unit/test_fin_aero.py"

# Stream B: US4 — EDF Propulsion (parallel with Stream A)
Task: "Create servo_mg996r.yaml in simulation/isaac/configs/params/servo_mg996r.yaml"
Task: "Create edf_90mm.yaml in simulation/isaac/configs/params/edf_90mm.yaml"
Task: "Implement actuator_servo.py in simulation/isaac/tvc_env/dynamics/actuator_servo.py"
Task: "Implement propulsion_edf.py in simulation/isaac/tvc_env/dynamics/propulsion_edf.py"

# Stream C: US5 — Contact State Machine (parallel with Streams A and B)
Task: "Implement sensor_interface.py in simulation/isaac/tvc_env/sim/sensor_interface.py"
Task: "Implement contacts.py in simulation/isaac/tvc_env/sim/contacts.py"
Task: "Write test_crash_logic.py in simulation/isaac/tests/unit/test_crash_logic.py"
```

## Parallel Example: User Stories 7, 8 (after US6 complete)

```bash
# Stream A: US7 — Debug Visualization
Task: "Implement gizmos.py in simulation/isaac/tvc_env/sim/gizmos.py"
Task: "Implement single_env.py in simulation/isaac/tvc_env/envs/single_env.py"

# Stream B: US8 — 128-Env Vectorized Training (parallel with Stream A)
Task: "Implement domain_randomization.py in simulation/isaac/tvc_env/envs/domain_randomization.py"
Task: "Write test_11_rl_api_128env_smoke.py in simulation/isaac/tests/sim/test_11_rl_api_128env_smoke.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (common/ utilities)
3. Complete Phase 3: User Story 1 (Asset Loading & Validation)
4. **STOP and VALIDATE**: Load USD, run asset validation, confirm all structural checks pass
5. This proves the foundation is solid before investing in physics models

### Core Physics Increment (US1 → US4)

1. Complete US1 → US2 → US3 + US4 (parallel)
2. **VALIDATE**: Run validation ladder tests 00-07
3. All physics models proven before building the environment layer

### Full Environment Increment (US5 → US8)

1. Complete US5 → US6 → US7 + US8 (parallel)
2. **VALIDATE**: Run validation ladder tests 09-11
3. Environment runs in both single-env debug and 128-env vectorized modes

### Research-Ready Increment (US9 → US10)

1. Complete US9 → US10
2. **VALIDATE**: Run full validation ladder tests 00-12
3. PID hover proven, RL training can begin

### Incremental Delivery

Each increment adds independently testable value:
1. Setup + Foundational + US1 → Asset pipeline proven
2. + US2 → Articulation and frame correctness proven
3. + US3 + US4 → Full physics models proven
4. + US5 + US6 → Complete environment with tasks
5. + US7 + US8 → Debug and training modes
6. + US9 + US10 → Disturbances and PID validation
7. + Polish → Telemetry, documentation, final validation

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks in same phase
- [Story] label maps task to specific user story for traceability
- All file paths under `simulation/isaac/` — the tvc_env package root
- YAML configs must include source labels (measured/datasheet/estimate/to-be-calibrated/derived) on every parameter
- `to-be-calibrated` null values must be replaced before training runs (validated by base_env.py)
- Quaternion convention: (w,x,y,z) internally per Isaac Lab 2.3.2; conversion at boundary
- Body frame: FRD (x=forward, y=right, z=down) — all controller/aero computations
- Force dispatch default: per_link_force (forces at fin COP on fin links)
- Gizmos auto-disabled when num_envs > 1
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
