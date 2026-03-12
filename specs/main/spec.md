# Feature Specification: Isaac Sim Mass Properties, Thrust Test Environment & Environmental Forces

**Feature Branch**: `002-isaac-mass-thrust-env`
**Created**: 2026-03-11
**Status**: Draft
**Input**: User description: "Implement mass properties and env variables for the Isaac Sim env. Validate vehicle setup (mass props from USDC scene), create thrust test env, validate environmental forces (wind disturbances). Build on Feature 001 stable env with fins articulation."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Mass Property Validation (Priority: P1)

A researcher runs a validation script that reads the USDC scene's rigid-body mass, center-of-mass, and inertia tensor, compares them against the YAML config (`default_vehicle.yaml`), and reports any discrepancies exceeding 1%. The script exits with a clear pass/fail status and human-readable diff.

**Why this priority**: Mass property correctness is the foundation for all Isaac Sim physics. Incorrect mass/inertia produces physically wrong behavior that invalidates all subsequent thrust, control, and training work.

**Independent Test**: Run the validation script against the current `drone.usdc` and `default_vehicle.yaml`. Verify it reports pass/fail with specific values.

**Acceptance Scenarios**:

1. **Given** a USDC scene with correct mass properties matching YAML, **When** the validation script runs, **Then** it reports PASS with all values within 1% tolerance.
2. **Given** a USDC scene with intentionally wrong mass (e.g., 5.0 kg instead of 3.13 kg), **When** the validation script runs, **Then** it reports FAIL with the specific discrepancy and expected vs. actual values.
3. **Given** the YAML config specifies explicit mass properties (total_mass, CoM, inertia tensor), **When** the validation script reads both sources, **Then** it compares all 10 scalar values (1 mass + 3 CoM + 6 unique inertia components).

---

### User Story 2 - Thrust Application Test Environment (Priority: P1)

A researcher launches a diagnostic environment where the drone starts on the ground (altitude ~0.4 m), applies full thrust, and observes the drone lift off and ascend. This validates that thrust forces are correctly applied in the Isaac Sim physics pipeline and that the drone's mass properties produce the expected acceleration.

**Why this priority**: Without confirmed thrust application, no control policy can be trained. This is the first test that the drone is a "flyable" vehicle in Isaac Sim, not just a falling rigid body.

**Independent Test**: Run the thrust test diagnostic, command T_cmd=1.0 (full thrust), and verify the drone ascends from ground level. Expected acceleration: (T_max / mass - g) ≈ (45/3.13 - 9.81) ≈ 4.56 m/s².

**Acceptance Scenarios**:

1. **Given** the drone starts on the ground at ~0.4 m altitude, **When** full thrust (T_cmd=1.0) is commanded for 2 seconds, **Then** the drone ascends and reaches an altitude > 5 m.
2. **Given** the drone starts on the ground, **When** thrust equal to hover (T_cmd ≈ weight/T_max ≈ 0.68) is commanded, **Then** the drone maintains approximate hover (altitude change < 0.5 m over 2 seconds).
3. **Given** the drone is ascending under full thrust, **When** thrust is cut to zero, **Then** the drone decelerates, stops ascending, and falls back under gravity.

---

### User Story 3 - Environmental Force Application (Priority: P2)

A researcher configures wind disturbance parameters in the Isaac Sim environment via `default_environment.yaml` and observes measurable lateral drift when wind is applied to the drone. The wind model produces forces consistent with the custom simulation's `WindModel` output for equivalent parameters.

**Why this priority**: Environmental forces are required for domain randomization during training to produce robust policies. However, thrust and mass validation (US1, US2) must work first before adding wind disturbances.

**Independent Test**: Run a diagnostic with constant wind (e.g., 5 m/s in +X), zero thrust, and observe lateral acceleration consistent with aerodynamic drag forces.

**Acceptance Scenarios**:

1. **Given** wind is configured at 5 m/s in the X direction via YAML, **When** the drone is in free fall, **Then** it drifts laterally in +X with measurable velocity within 2 seconds.
2. **Given** wind is disabled (0 m/s), **When** the drone is in free fall, **Then** no lateral drift occurs (lateral velocity < 0.01 m/s).
3. **Given** wind gust parameters are configured, **When** a gust event triggers during an episode, **Then** the observation vector [13:16] reflects non-zero wind estimates.

---

### Edge Cases

- What happens when USDC mass properties are set to zero or negative values?
- How does thrust behave when the drone is in contact with the ground (ground reaction force vs. thrust)?
- What happens when wind force exceeds thrust authority (drone blown sideways uncontrollably)?
- How does the validation script handle a USDC scene missing MassAPI schemas?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a validation script that compares USDC scene mass properties against YAML config values.
- **FR-002**: The validation script MUST check total mass, center-of-mass (3 components), and inertia tensor (6 unique components) within 1% tolerance.
- **FR-003**: The validation script MUST exit with non-zero status on failure and print human-readable discrepancy report.
- **FR-004**: System MUST support a thrust test diagnostic where the drone starts on the ground and lifts off under commanded thrust.
- **FR-005**: Thrust application MUST produce acceleration consistent with F=ma (within 5% of expected value given mass and thrust magnitude).
- **FR-006**: System MUST support configurable wind disturbance forces applied to the drone in Isaac Sim.
- **FR-007**: Wind parameters MUST be loaded from `default_environment.yaml`, not hard-coded.
- **FR-008**: The observation vector elements [13:16] MUST reflect actual wind estimates when wind is enabled.
- **FR-009**: All new scripts and diagnostics MUST accept YAML config paths as command-line arguments.
- **FR-010**: The validation script MUST be runnable without launching the full Isaac Sim simulation (USD-only inspection).

### Key Entities

- **MassPropertyValidator**: Reads USDC rigid-body mass/inertia and YAML explicit mass properties, compares within tolerance.
- **ThrustTestDiagnostic**: Ground-start environment with configurable thrust commands for liftoff validation.
- **IsaacWindModel**: Adapter that applies wind forces to the Isaac Sim drone, sourcing parameters from environment YAML config.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Mass property validation script passes on the current `drone.usdc` with all 10 values within 1% of YAML config.
- **SC-002**: Drone lifts off from ground under full thrust and reaches 5 m altitude within 2 seconds of simulation time.
- **SC-003**: Hover thrust (T_cmd ≈ 0.68) maintains altitude within ±0.5 m over 2 seconds.
- **SC-004**: Wind disturbance of 5 m/s produces measurable lateral velocity (>0.1 m/s) within 1 second.
- **SC-005**: All Isaac Sim diagnostics complete without errors on RTX 5070 with single-env config.

## Assumptions

- Feature 001 (`001-isaac-sim-env`) is complete and stable: drone falls under gravity, fins articulate, ground contact works.
- The hand-authored `drone.usdc` has MassAPI schemas applied to the root rigid body.
- YAML `mass_properties.use_explicit: true` path is used (not primitive computation).
- Wind forces can be applied via `robot.set_external_force_and_torque()` in the same manner as thrust and fin forces.
- The custom simulation's `WindModel` can be instantiated independently of `EnvironmentModel` for force computation.

## Dependencies

- Feature 001 complete (stable Isaac Sim env with fins articulation)
- `default_vehicle.yaml` with explicit mass properties section
- `default_environment.yaml` with wind configuration
- `pxr` (OpenUSD) Python bindings for USDC inspection
- Isaac Sim / IsaacLab runtime for thrust and wind diagnostics
