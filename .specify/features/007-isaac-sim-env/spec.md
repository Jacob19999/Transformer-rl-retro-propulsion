# Feature Specification: Phase 1 Isaac Sim EDF Thrust-Vectoring Simulation Environment

**Feature Branch**: `007-isaac-sim-env`
**Created**: 2026-03-22
**Status**: Draft
**Input**: User description: "Sim Environment Implementation Plan Details — Phase 1 simulation environment for the EDF retro-propulsion / thrust-vectoring project, built from scratch in Isaac Sim 5.1 + Isaac Lab 2.3.2, validated through an incremental test ladder, supporting hover and landing tasks with 128-env vectorized training."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Asset Loading and Validation (Priority: P1)

A researcher loads the EDF drone USD asset into Isaac Sim and the environment automatically validates all structural requirements: body and fin links exist, revolute joints are correctly configured, hinge axes are defined, mass/inertia values are valid, and the link hierarchy is consistent. The environment fails fast with clear diagnostics if any structural requirement is violated.

**Why this priority**: Every downstream capability depends on a correctly loaded and validated vehicle asset. Without this, no physics, control, or training work can proceed.

**Independent Test**: Can be fully tested by loading the USD asset and running the asset validation suite, confirming pass/fail status for each structural check.

**Acceptance Scenarios**:

1. **Given** a valid USD vehicle asset, **When** the environment initializes, **Then** all links, joints, hinge axes, mass properties, and hierarchy pass validation without errors.
2. **Given** a USD asset with a missing fin link, **When** the environment initializes, **Then** it fails immediately with a diagnostic message identifying the missing link.
3. **Given** a USD asset with an undefined joint axis, **When** the environment initializes, **Then** it fails immediately with a diagnostic message identifying the malformed joint.

---

### User Story 2 - Frame-Correct Fin Articulation and Force Application (Priority: P1)

A researcher commands individual fin deflections and observes that each fin moves along the correct hinge axis with the correct sign convention. When aerodynamic forces are computed per-fin, they are applied at the fin center of pressure on the fin link, and the resulting body reaction emerges through articulation physics. All coordinate conversions between the internal body-FRD frame and the Isaac/world frame pass through a single conversion boundary.

**Why this priority**: Frame and sign correctness is the single largest historical source of simulation bugs. Proving this early prevents compounding errors throughout all subsequent validation.

**Independent Test**: Can be tested by commanding known fin angles and verifying axis direction, sign, and joint limits, then applying synthetic unit forces at fin COPs and checking the observed body reaction direction and magnitude.

**Acceptance Scenarios**:

1. **Given** a loaded vehicle with gravity disabled, **When** a single fin is commanded to a positive deflection, **Then** the fin rotates around the correct hinge axis in the expected positive direction.
2. **Given** a loaded vehicle, **When** a known synthetic force is applied at a fin center of pressure, **Then** the observed body reaction matches the expected r × F direction and sign.
3. **Given** any coordinate conversion in the codebase, **When** the conversion is traced, **Then** it passes through the single canonical conversion module and no sign flips exist elsewhere.

---

### User Story 3 - Per-Fin Aerodynamic Force Modeling (Priority: P1)

A researcher sweeps a single fin through its range of deflection angles under simulated exhaust flow and observes physically plausible force behavior: near-zero normal force at zero deflection, approximately linear normal force at small angles, nonlinear saturation at large angles, and drag increasing with deflection magnitude. The model uses a subsonic EDF-appropriate semi-empirical approach rather than supersonic rocket coefficients.

**Why this priority**: The fin aerodynamic model is the core force-generation mechanism for thrust-vectoring control. Without correct force behavior, no controller can stabilize the vehicle.

**Independent Test**: Can be tested by sweeping one fin from minimum to maximum deflection, recording normal and tangential force components, and comparing the resulting curves against expected qualitative characteristics.

**Acceptance Scenarios**:

1. **Given** a fin at zero deflection with exhaust flow, **When** force is computed, **Then** the normal (control) force is near zero and a small tangential drag may be present.
2. **Given** a fin swept from 0° to maximum deflection, **When** force curves are plotted, **Then** normal force is approximately linear for small/moderate angles and saturates at large angles.
3. **Given** all four fins commanded in known patterns (roll-only, pitch-only, yaw-only, symmetric), **When** the summed wrench is computed, **Then** the resultant body forces and torques match expected directions and relative magnitudes.

---

### User Story 4 - EDF Propulsion with Realistic Dynamics (Priority: P1)

A researcher steps the motor command and observes realistic thrust response including spool lag, static rotor reaction torque during steady rotation, dynamic spool torque during acceleration/deceleration, and gyroscopic precession when the body rotates with a spinning rotor. All torque terms are logged separately for magnitude comparison.

**Why this priority**: Omitting propulsion dynamics makes hover and landing results unrealistically optimistic, undermining the sim-to-real transfer goal of the project.

**Independent Test**: Can be tested by commanding motor step inputs and verifying thrust lag timing, then imposing body angular motion with a spinning rotor and checking precession direction and magnitude.

**Acceptance Scenarios**:

1. **Given** a step motor command, **When** thrust is measured over time, **Then** the response exhibits first-order lag consistent with the configured time constant.
2. **Given** a rotor at steady speed, **When** static reaction torque is measured, **Then** it opposes the rotor spin direction with magnitude consistent with the configured torque coefficient.
3. **Given** a spinning rotor and an imposed body angular velocity, **When** gyroscopic precession torque is measured, **Then** the precession direction follows ω_body × H_rotor and the magnitude is logged relative to fin-generated torques.

---

### User Story 5 - Contact State Machine for Landing/Crash Detection (Priority: P2)

A researcher runs scripted touchdown scenarios and the environment correctly classifies each outcome using a state machine with states: AIRBORNE, GROUND_CONTACT_CANDIDATE, LANDED, and CRASHED. A soft touchdown that meets all dwell criteria is declared LANDED; a hard impact or tip-over is declared CRASHED; a bounce is held in GROUND_CONTACT_CANDIDATE and does not falsely declare LANDED.

**Why this priority**: Correct landing/crash detection is essential for the landing task reward and termination logic, which is the primary research task.

**Independent Test**: Can be tested by running scripted drop tests at various speeds and angles and verifying the state machine transitions and final classifications.

**Acceptance Scenarios**:

1. **Given** a vehicle descending at low speed and near-vertical attitude, **When** it contacts the ground and remains within thresholds for the dwell interval, **Then** the state machine transitions AIRBORNE → GROUND_CONTACT_CANDIDATE → LANDED.
2. **Given** a vehicle contacting the ground then bouncing, **When** contact is lost during the dwell interval, **Then** the state machine returns to AIRBORNE without declaring LANDED.
3. **Given** a vehicle impacting at high speed or excessive tilt, **When** it contacts the ground, **Then** the state machine transitions to CRASHED.
4. **Given** a vehicle that tips over after initial contact, **When** tilt exceeds the crash threshold, **Then** the state machine transitions to CRASHED.

---

### User Story 6 - Task-Configurable Hover and Landing Modes (Priority: P2)

A researcher selects either "hover" or "landing" task through configuration without modifying environment code. Each task uses the same physics core but applies different reward profiles, success criteria, termination logic, spawn ranges, and disturbance settings. Reward terms are composable and selected by task configuration from a shared registry.

**Why this priority**: Task configurability proves that the environment is general-purpose and avoids the code-forking anti-pattern that plagued earlier iterations.

**Independent Test**: Can be tested by running the environment with hover config and landing config in sequence, verifying that physics behavior is identical but reward signals, termination conditions, and success criteria differ appropriately.

**Acceptance Scenarios**:

1. **Given** the hover task configuration, **When** the vehicle maintains position within tolerance for the dwell interval, **Then** the task reports success and the hover-specific reward terms (stability bonus, drift penalty) are active.
2. **Given** the landing task configuration, **When** the vehicle touches down softly within the landing region, **Then** the task reports success and the landing-specific reward terms (touchdown softness, pad accuracy) are active.
3. **Given** a switch from hover to landing configuration, **When** the environment reloads, **Then** all physics modules remain identical and only task-layer settings change.

---

### User Story 7 - Single-Environment Debug Visualization (Priority: P2)

A researcher runs a single-environment debug session and sees all required visual gizmos: body axes, COM marker, target/pad marker, thrust vector, per-fin force arrows, total aerodynamic force arrow, reaction torque arrow, and contact normals. A heads-up display shows altitude error, position error, tilt, body-rate magnitude, motor RPM, per-fin angles and forces, total reward, task mode, and vehicle state.

**Why this priority**: Debug visualization is essential for diagnosing physics and control issues. Without gizmos, subtle sign or magnitude errors can go undetected for weeks.

**Independent Test**: Can be tested by launching a single-env debug run and visually confirming all gizmos render correctly and HUD values update in real time.

**Acceptance Scenarios**:

1. **Given** a single-env debug run, **When** the vehicle is in flight, **Then** all required gizmos (body axes, thrust vector, per-fin force arrows, contact normals) are visible and correctly oriented.
2. **Given** a single-env debug run, **When** forces change due to fin deflection, **Then** the force arrow gizmos update in real time reflecting the new force vectors.
3. **Given** a single-env debug run, **When** the HUD is displayed, **Then** all required telemetry values are present and update each step.

---

### User Story 8 - 128-Environment Vectorized Training (Priority: P2)

A researcher launches the environment in vectorized mode with 128 parallel environments. The environment correctly resets, observes, steps, and terminates across all instances without tensor shape mismatches or NaN values. Gizmos are disabled in vectorized mode for performance.

**Why this priority**: Vectorized training is the pathway to RL policy learning (PPO/GTrXL-PPO), which is the ultimate research objective.

**Independent Test**: Can be tested by running a 128-env smoke test that resets all environments, takes random actions, collects observations, and verifies tensor shapes and value ranges.

**Acceptance Scenarios**:

1. **Given** 128 environments initialized, **When** all environments are reset, **Then** each receives a valid initial observation with correct tensor shape.
2. **Given** 128 environments running, **When** random actions are stepped for 1000 steps, **Then** no NaN values appear in observations or rewards and no tensor shape errors occur.
3. **Given** some environments reaching termination, **When** they auto-reset, **Then** they resume with valid states without affecting other running environments.

---

### User Story 9 - Wind and Disturbance Injection (Priority: P3)

A researcher enables the wind/disturbance framework through configuration and injects constant wind, gust pulses, or randomized lateral gusts. The vehicle responds physically to the disturbance, and the response direction is correct regardless of vehicle orientation. Center-of-mass offsets and sensor noise are available as optional toggles.

**Why this priority**: Disturbance robustness is explicitly part of the research proposal but can be validated after the core physics and task systems are proven.

**Independent Test**: Can be tested by injecting a known constant wind vector and verifying the vehicle drifts in the expected direction, then rotating the vehicle and confirming the response remains frame-correct.

**Acceptance Scenarios**:

1. **Given** a constant wind vector applied to an upright hovering vehicle, **When** the wind is enabled, **Then** the vehicle drifts in the wind direction at a rate consistent with the configured drag coefficient and reference area.
2. **Given** a wind vector and a rotated vehicle, **When** force is computed, **Then** no frame sign errors occur and the drag opposes relative airflow in the correct world-frame direction.
3. **Given** a gust event configuration, **When** the gust triggers during flight, **Then** a transient force disturbance is applied for the configured duration and magnitude.

---

### User Story 10 - PID Hover Validation (Priority: P3)

A researcher runs a PID controller in the simulation with all modeled forces, actuator lags, and rotor dynamics enabled. The PID achieves bounded hover: position error stays within tolerance, tilt stays within tolerance, angular velocity stays within tolerance, and no ground contact occurs, all persisting for a dwell interval. This validates the integrated environment before RL training.

**Why this priority**: PID hover is the integration test that proves the entire environment works correctly end-to-end before committing to RL training campaigns.

**Independent Test**: Can be tested by running the PID hover scenario and checking that all stability metrics remain bounded for the full test duration.

**Acceptance Scenarios**:

1. **Given** a PID controller with all physics effects enabled, **When** the hover test runs for the configured duration, **Then** position error remains within the configured tolerance.
2. **Given** the hover test running, **When** no NaN values appear in any state variable, **Then** the simulation is considered numerically stable.
3. **Given** the hover test with wind disturbance enabled, **When** the PID attempts to reject the disturbance, **Then** the vehicle oscillation remains bounded and no sign-error-induced divergence occurs.

---

### Edge Cases

- What happens when a servo command exceeds the maximum deflection angle? The servo model must clamp to the configured joint limit.
- What happens when motor RPM change rate exceeds the configured slew limit? The spool dynamics must rate-limit the change.
- What happens when all four fins are commanded to maximum deflection simultaneously? The environment must remain stable without numerical blowup.
- What happens when the vehicle is inverted (>90° tilt)? The environment must either handle inverted flight gracefully or trigger a crash termination.
- What happens when wind speed exceeds thrust capability? The vehicle should drift and potentially crash rather than behaving unphysically.
- What happens when the simulation runs with a very small timestep? Force computations must remain numerically stable.
- What happens during a vectorized reset if one environment crashes while others are mid-flight? The crashing environment must reset independently without corrupting other environments.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST load and validate a USD vehicle asset containing a main body, four fin links, four revolute joints, colliders, mass/inertia properties, and a consistent link hierarchy, failing fast with diagnostics on any structural violation.
- **FR-002**: System MUST perform all controller and aerodynamic computations in a body-fixed FRD frame (x=forward, y=right, z=down) with all frame conversions isolated in a single canonical module.
- **FR-003**: System MUST extract and validate from USD or companion metadata: body link name, fin link/joint names, hinge axes, joint limits, fin COP positions, fin chord/normal directions, EDF thrust/spin axes, and landing-gear contact regions.
- **FR-004**: System MUST compute per-fin aerodynamic forces using a subsonic semi-empirical model with normal and tangential force components, applying forces at the fin center of pressure on the fin link and letting body torque emerge through articulation physics.
- **FR-005**: System MUST model servo dynamics including first-order lag, angular rate limiting, maximum deflection limits, and optional deadband/backlash approximation using MG996R-class starting parameters.
- **FR-006**: System MUST model EDF propulsion including thrust along the EDF axis, motor spool dynamics with first-order lag and RPM rate limiting, static rotor reaction torque, dynamic spool torque, and gyroscopic precession.
- **FR-007**: System MUST implement a contact state machine with AIRBORNE, GROUND_CONTACT_CANDIDATE, LANDED, and CRASHED states, using dwell-interval thresholds for vertical speed, lateral speed, tilt, and angular rate to distinguish outcomes.
- **FR-008**: System MUST expose a controller-agnostic observation vector including position error, attitude, linear/angular velocity, height, fin angles/rates, motor RPM, contact state, and optional disturbance state.
- **FR-009**: System MUST accept a 5-dimensional action vector (4 fin target angles + 1 throttle/RPM target) that is interpreted by the environment without controller-specific assumptions.
- **FR-010**: System MUST support hover and landing tasks through configuration only, using the same physics core with task-specific reward profiles, success criteria, termination logic, spawn ranges, and disturbance settings.
- **FR-011**: System MUST provide composable reward terms selected by task configuration from a shared registry, with shared terms (alive bonus, position/attitude/rate penalties, control effort, crash penalty) and task-specific terms (hover stability, landing softness, pad accuracy).
- **FR-012**: System MUST render debug gizmos in single-env mode showing body axes, COM marker, target marker, thrust vector, per-fin force arrows, total aerodynamic force, reaction torque, and contact normals, along with a HUD displaying all required telemetry values.
- **FR-013**: System MUST support vectorized execution of 128 parallel environments through Isaac Lab scene replication with correct reset, observe, step, and terminate operations and no tensor shape or NaN issues.
- **FR-014**: System MUST support a wind/disturbance framework with constant wind, gust pulses, randomized lateral gusts, center-of-mass offsets, and optional sensor noise, all configurable through YAML files.
- **FR-015**: System MUST use a two-mode force dispatch architecture: per-link force at COP (default for validation) and collapsed body wrench (optional fallback for training throughput), with both modes isolated behind a dispatch abstraction layer.
- **FR-016**: System MUST separate configuration parameters into measured values, datasheet values, engineering estimates, and to-be-calibrated values, with clear labeling in all parameter files.
- **FR-017**: System MUST use the same physics modules for single-env debug, PID evaluation, hover test, landing test, and vectorized RL training, with only wrappers, reward profiles, and visualization settings changing between modes.
- **FR-018**: System MUST log all torque contributions (fin-generated, static rotor reaction, dynamic spool, gyro precession, wind-induced) separately during validation runs for magnitude comparison.

### Key Entities

- **Vehicle Asset**: USD-defined articulated rigid body with main body link, four fin links, four revolute joints, colliders, and mass/inertia properties. Accompanied by metadata defining hinge axes, COPs, and thrust geometry.
- **Fin**: Articulated vane in the EDF exhaust stream. Has hinge axis, center of pressure, chord direction, normal direction, and joint limits. Generates normal and tangential aerodynamic forces.
- **EDF Propulsion Unit**: Thrust source with configurable thrust coefficient, motor inertia, spool time constant, RPM limits, and torque coefficients. Produces thrust, static reaction torque, dynamic spool torque, and gyroscopic precession.
- **Servo Actuator**: MG996R-class actuator driving fin deflection. Modeled with first-order lag, rate limit, deflection limit, and optional deadband.
- **Task**: Configuration-driven operating mode (hover or landing) defining reward profile, success criteria, termination conditions, spawn ranges, and disturbance settings.
- **Contact State Machine**: Four-state machine (AIRBORNE, GROUND_CONTACT_CANDIDATE, LANDED, CRASHED) tracking vehicle-ground interaction through dwell-interval thresholds.
- **Disturbance Model**: Configurable wind and perturbation framework including steady wind vector, gust events, COM offset, and sensor noise.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Vehicle asset loads and passes all structural validation checks (links, joints, axes, mass, hierarchy) with zero errors on a compliant asset.
- **SC-002**: Single-fin deflection tests confirm correct hinge axis direction and sign for all four fins, with zero sign or axis errors.
- **SC-003**: Synthetic unit-force tests confirm correct r × F body reaction direction for all four fin COP positions, with zero sign inversions.
- **SC-004**: Fin force sweep curves exhibit near-zero normal force at zero deflection, approximately linear behavior at small angles, and saturation at large angles for all four fins.
- **SC-005**: Four-fin superposition tests confirm correct roll-only, pitch-only, and yaw-only wrench behavior matching expected moment directions.
- **SC-006**: Motor step response exhibits first-order lag within 10% of the configured time constant.
- **SC-007**: Gyroscopic precession direction matches ω × H for all tested body rotation axes.
- **SC-008**: Contact state machine correctly classifies soft landing, bounce-and-recover, hard impact, and tip-over scenarios in scripted touchdown tests with zero misclassifications.
- **SC-009**: A PID controller achieves bounded hover for at least 10 seconds with all modeled forces and lags enabled, maintaining position error within 0.5m and tilt within 15°.
- **SC-010**: 128-environment vectorized execution runs for 1000 steps with random actions, producing zero NaN values and zero tensor shape errors.
- **SC-011**: Switching between hover and landing task configurations requires zero code changes — only configuration file selection.
- **SC-012**: Single-env debug mode renders all required gizmos and HUD telemetry values, confirmed by visual inspection during the validation ladder.

## Assumptions

- The EDF drone USD asset (v2) is available or will be authored as part of the implementation, following the structural requirements defined in this spec.
- Isaac Sim 5.1 and Isaac Lab 2.3.2 are the target platform versions; no migration to newer versions during Phase 1.
- PhysX is the physics backend; Newton is explicitly out of scope for Phase 1.
- MG996R servo parameters are starting estimates to be refined with hardware bench data.
- EDF parameters (48N thrust, 90mm diameter) are engineering estimates from the current hardware target.
- The subsonic semi-empirical fin aerodynamic model is a starting approximation that will be calibrated with bench test data in subsequent phases.
- PID controller used for hover validation is a basic tuned controller, not the final flight controller.
- 128 environments is the target vectorization count; actual GPU memory may constrain this.
- Wind model uses simplified body-drag assumptions; full CFD-level wind modeling is out of scope.

## Scope Boundaries

### In Scope

- 6-DOF articulated rigid-body vehicle in Isaac Sim
- Per-fin aerodynamic force model (subsonic semi-empirical)
- EDF propulsion with spool dynamics, reaction torques, and gyro precession
- Servo actuator modeling (MG996R-class)
- Contact state machine for landing/crash detection
- Hover and landing task configurations
- Composable reward system
- Single-env debug visualization with gizmos and HUD
- 128-env vectorized training readiness
- Wind and disturbance framework
- Incremental validation test ladder (tests 00-12)
- PID hover smoke test
- Controller-agnostic observation and action spaces
- PID, PPO, and GTrXL-PPO adapter interfaces

### Out of Scope

- Full-scale vehicle fidelity
- Supersonic rocket exhaust or combustion chemistry
- Untethered hardware flight testing
- Transonic reentry aerothermodynamics
- Newton physics backend
- Simulink/HIL bridge implementation
- Jetson deployment pipeline
- Final GTrXL-PPO benchmarking campaign
- Real-time co-simulation synchronization
- Full CFD wind modeling
- Vane-to-vane interference or duct wall interaction at large deflection

