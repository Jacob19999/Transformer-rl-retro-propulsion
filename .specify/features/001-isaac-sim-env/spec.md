# Feature Specification: Isaac Sim Vectorized Drone Simulation Environment

**Feature Branch**: `001-isaac-sim-env`
**Created**: 2026-03-10
**Status**: Draft
**Input**: User description: "Define a vectorized simulation environment in NVIDIA Isaac Sim for training control policies on an EDF drone emulating TVC retro-propulsive landings."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Single-Environment Drone Simulation (Priority: P1)

A researcher launches a simulation containing one EDF drone in a bounded arena with a landing pad. The drone spawns at a randomized altitude and velocity, and the researcher visually confirms that the drone behaves physically (falls under gravity, contacts the ground, does not clip through surfaces). The researcher can send zero-valued control inputs and observe the drone's passive response.

**Why this priority**: A single functioning simulation is the foundation for all subsequent work. Without a stable single environment, parallelism and training are meaningless.

**Independent Test**: Can be fully tested by launching the simulation, spawning a drone, applying zero actions, and visually confirming the drone falls under gravity and settles on the ground without crashing or clipping.

**Acceptance Scenarios**:

1. **Given** the simulation is launched with a single environment, **When** a drone spawns at a randomized altitude between 5-10m with velocity between 0-5 m/s, **Then** the drone appears in the arena and responds to gravity immediately.
2. **Given** the drone is airborne with zero control inputs, **When** the simulation runs for 10 seconds, **Then** the drone descends, contacts the landing pad, and comes to rest without penetrating the surface.
3. **Given** the drone has landed, **When** the environment is reset, **Then** a new drone spawns at fresh randomized initial conditions without errors or visual artifacts.

---

### User Story 2 - Drone Asset with Controllable Fins (Priority: P1)

A researcher loads the EDF drone asset into the simulation. The drone consists of a cylindrical body with four control fins attached via rotational joints. Each fin can be deflected within its angular limits. The researcher can command fin deflections and observe the corresponding motion in the simulation.

**Why this priority**: The drone asset with articulated fins is essential for any control policy training; without controllable fins, no thrust-vectoring policy can be developed.

**Independent Test**: Can be tested by loading the drone, commanding individual fin deflections to their limits, and verifying each fin rotates correctly and stops at the limit.

**Acceptance Scenarios**:

1. **Given** the drone asset is loaded, **When** a fin deflection command of +15 degrees is sent to fin 1, **Then** fin 1 rotates to +15 degrees and does not exceed this limit.
2. **Given** all four fins are at neutral (0 degrees), **When** deflection commands of -15 degrees are sent to all fins simultaneously, **Then** all four fins rotate to -15 degrees.
3. **Given** a fin is at its maximum deflection, **When** a command beyond the limit is sent, **Then** the fin remains at its maximum deflection without simulation errors.

---

### User Story 3 - Parallel Environment Rollouts (Priority: P2)

A researcher configures the simulation to run multiple identical environments simultaneously (128 to 1024 instances). Each environment contains an independent drone that can be controlled and reset independently. The researcher confirms that all environments run in parallel and that throughput scales with the number of environments.

**Why this priority**: Parallelism is required for efficient reinforcement learning training but is not needed for initial validation or PID tuning. If a single environment is sufficient for early-stage controller development, parallelism can be deferred.

**Independent Test**: Can be tested by launching 128 parallel environments, applying different random actions to each, and verifying that all drones respond independently and no cross-environment interference occurs.

**Acceptance Scenarios**:

1. **Given** the simulation is configured for 128 parallel environments, **When** the simulation launches, **Then** 128 independent drone instances are created and all begin simulating simultaneously.
2. **Given** 128 environments are running, **When** environment 42 is reset while others continue, **Then** only environment 42 resets; all other environments are unaffected.
3. **Given** the simulation is configured for 1024 parallel environments, **When** a training step is executed across all environments, **Then** all 1024 environments complete the step and return observations within the same simulation frame.

---

### User Story 4 - Configuration-Driven Drone Generation (Priority: P2)

A researcher uses existing YAML configuration files (which define drone geometry, mass primitives, and physical properties) to automatically generate the simulation drone asset. Changes to the YAML config are reflected in the generated asset without manual editing.

**Why this priority**: Reusing existing configuration ensures consistency between the custom Python simulation and the new simulation environment, reducing integration errors.

**Independent Test**: Can be tested by modifying a YAML parameter (e.g., drone body length) and regenerating the asset, then verifying the change appears in the simulation.

**Acceptance Scenarios**:

1. **Given** a valid YAML configuration file with drone geometry primitives, **When** the asset generation process runs, **Then** a drone asset is produced matching the specified dimensions and properties.
2. **Given** the YAML config specifies fin deflection limits of +/-15 degrees, **When** the asset is generated, **Then** the resulting drone fins have joint limits matching the config values.

---

### Edge Cases

- What happens when the drone is spawned at the arena boundary (e.g., at x=5m, y=5m) and drifts outside the arena?
- How does the simulation handle a drone that flips upside down and contacts the ground with its top?
- What happens when all 1024 environments are reset simultaneously in the same frame?
- How does the system behave if the YAML configuration contains invalid or missing values (e.g., negative mass, zero-length body)?
- What happens when fin deflection commands change direction rapidly (oscillation at simulation frequency)?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST simulate a rigid-body drone with a fixed cylindrical body and four independently controllable fins.
- **FR-002**: Each fin MUST rotate about a single axis with deflection limits of +/-15 degrees.
- **FR-003**: System MUST simulate gravity at 9.81 m/s^2 directed downward at all times.
- **FR-004**: System MUST simulate ground contact and collisions, preventing the drone from penetrating surfaces.
- **FR-005**: The simulation arena MUST be a bounded 10m x 10m area with a designated landing pad surface.
- **FR-006**: The landing pad surface MUST have a friction coefficient of 0.5.
- **FR-007**: System MUST spawn the drone at randomized initial conditions: altitude between 5-10m, velocity magnitude between 0-5 m/s.
- **FR-008**: System MUST run at a fixed simulation timestep of 1/120 second (approximately 8.33 ms).
- **FR-009**: System MUST support environment reset, returning the drone to new randomized initial conditions without restarting the simulation.
- **FR-010**: System MUST support running 128 to 1024 parallel environments simultaneously for batch data collection.
- **FR-011**: Each parallel environment MUST be independent: actions in one environment MUST NOT affect any other environment.
- **FR-012**: System MUST accept a 5-dimensional control input per environment (1 thrust command + 4 fin deflection commands), normalized to the range [-1, 1].
- **FR-013**: System MUST return a 20-dimensional observation vector per environment, consistent with the existing observation pipeline format.
- **FR-014**: System MUST generate the drone asset from existing YAML configuration files that define geometry, mass primitives, and physical properties.
- **FR-015**: The simulation MUST accept existing YAML configuration files without modification to the config format.

### Key Entities

- **Drone**: Rigid-body vehicle with a cylindrical body, four articulated fins, thrust source, and mass/inertia properties derived from configuration.
- **Fin**: Rotational control surface attached to the drone body; characterized by angular position, deflection limits, and aerodynamic effect.
- **Arena**: Bounded simulation space (10m x 10m) containing the landing pad and defining spatial boundaries.
- **Landing Pad**: Ground surface within the arena where the drone targets its landing; characterized by friction properties and position.
- **Environment Instance**: A self-contained simulation containing one drone and one arena, capable of independent reset and control.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A single environment completes 1000 consecutive reset-simulate-reset cycles without errors or crashes.
- **SC-002**: A drone under zero control inputs descends from spawn altitude and settles on the landing pad within 10 seconds of simulation time, confirmed by visual inspection.
- **SC-003**: Fin deflection commands produce visible, physically plausible changes in drone orientation within 0.5 seconds of simulation time.
- **SC-004**: 128 parallel environments collectively produce at least 10x the data throughput (steps per second) of a single environment.
- **SC-005**: The observation vector produced by the simulation matches the 20-dimensional format of the existing observation pipeline, verified by dimensional and range checks.
- **SC-006**: Drone assets generated from YAML configuration match the specified geometry within 1% tolerance on all dimensional parameters.
- **SC-007**: Environment reset completes in under 100 ms per environment on average, enabling rapid training iteration.

## Assumptions

- The existing YAML configuration files (`default_vehicle.yaml`) contain sufficient geometric and physical data to generate the drone asset without additional manual specification.
- The 5-dimensional action space and 20-dimensional observation space from the existing custom simulation are the target interface for this environment.
- The FRD (Forward-Right-Down) body frame convention and scalar-last quaternion convention from the existing simulation carry over to the new environment.
- The drone's EDF thrust model (first-order lag, ground effect) can be approximated or replicated within the physics simulation's native force application mechanisms.
- The servo model (first-order actuator lag) for fin actuation can be approximated within the simulation's joint drive system.
- A single-environment mode is sufficient for initial PID controller validation; parallelism is primarily needed for reinforcement learning training.
- The indoor arena does not require wind or atmospheric modeling for the initial version; these can be added later.

## Dependencies

- Existing YAML configuration files in `simulation/configs/` (specifically `default_vehicle.yaml` for drone geometry and mass primitives).
- Existing observation pipeline specification (`simulation/training/observation.py`) defining the 20-dimensional observation format.
- Existing reward function specification (`simulation/training/reward.py`) for eventual training integration.
- Access to a workstation with a compatible GPU for running the physics simulation with parallel environments.
