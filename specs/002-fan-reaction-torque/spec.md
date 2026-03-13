# Feature Specification: EDF Fan Reaction Torque — Steady-State Anti-Torque & RPM-Ramp Yaw Coupling

**Feature Branch**: `002-fan-reaction-torque`
**Created**: 2026-03-12
**Status**: Draft
**Input**: User description: "Extend the existing user stories to include reaction torque at constant RPM (EDF anti-torque: body yaw torque opposite to fan rotation, present whenever the fan runs) and RPM-ramp reaction torque (transient yaw torque from dω/dt ≠ 0 during thrust ramps). Add Isaac Sim diagnostic tests for thrust ramps and constant RPM."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Steady-State Anti-Torque Validation (Priority: P1)

A researcher runs a diagnostic where the drone is placed in a gravity-free environment with unconstrained yaw. Full hover thrust is commanded at a constant level. Because the EDF fan imparts angular momentum to the airflow, the body experiences a reaction torque (anti-torque) in the opposite direction to the fan rotation. The researcher observes a monotonically growing yaw angle, confirms the yaw rate is proportional to thrust magnitude, and validates the torque coefficient against the YAML-configured value.

**Why this priority**: The anti-torque is physically present in every flight regime whenever the motor runs. Ignoring it makes every simulation flight trajectory unrealistic: the vehicle would not yaw when it should. This corrupts training data for any control policy that must reject yaw disturbances.

**Independent Test**: Run `diag_reaction_torque` with `--mode constant` at 50% thrust for 3 seconds. Measure yaw rate; confirm it is non-zero and matches `k_torque * T / I_zz` within 10%.

**Acceptance Scenarios**:

1. **Given** the drone is spawned with unconstrained yaw and zero initial angular velocity, **When** hover thrust (T_cmd ≈ 0.68) is commanded for 3 seconds, **Then** the yaw angle grows monotonically and the steady yaw rate satisfies `ω_yaw = k_torque × T / I_zz` within 10% tolerance.
2. **Given** the drone is commanded at two different constant thrust levels (30% and 70%), **When** each is held for 3 seconds, **Then** the ratio of measured yaw rates matches the ratio of thrust values within 10%.
3. **Given** thrust is commanded at zero (T_cmd = 0), **When** the diagnostic runs for 3 seconds, **Then** yaw rate remains < 0.01 rad/s (no reaction torque without fan spin).
4. **Given** the YAML config specifies `propulsion.torque_coefficient_k_Q`, **When** the diagnostic reads it, **Then** the computed anti-torque matches the YAML value without hard-coded overrides.

---

### User Story 2 - RPM-Ramp Transient Yaw Torque Validation (Priority: P1)

A researcher runs a diagnostic where the drone starts from rest with no thrust, then commands a rapid thrust ramp from zero to full over ~1 second, then back to zero. During the ramp-up, the rotor's angular momentum is increasing (dω/dt > 0), creating an additional reaction torque on the body frame beyond the steady-state anti-torque. The researcher observes a transient yaw spike during the ramp that exceeds the steady-state yaw rate, then subsides once RPM is constant.

**Why this priority**: RPM ramps are unavoidable during takeoff, landing, and any thrust-modulation maneuver. Unmodelled transient yaw coupling causes the vehicle to overshoot yaw setpoints during thrust changes, which destabilizes any attitude controller trained without this effect.

**Independent Test**: Run `diag_reaction_torque` with `--mode ramp` ramping from 0 to full thrust over 1 second. Plot yaw rate vs. time and confirm: (a) peak yaw rate during ramp exceeds steady-state yaw rate at full thrust, (b) yaw rate reduces to steady-state level once thrust plateaus.

**Acceptance Scenarios**:

1. **Given** the drone starts with zero thrust and unconstrained yaw, **When** thrust is ramped linearly from 0 to 100% over 1 second, **Then** the peak yaw rate during the ramp exceeds the steady-state yaw rate at 100% thrust by a measurable margin (> 10%).
2. **Given** the same ramp is applied, **When** thrust reaches 100% and remains constant, **Then** the yaw rate decays from its ramp-peak and stabilizes to the steady-state anti-torque value within 2 seconds.
3. **Given** a slower ramp (0 to 100% over 3 seconds), **When** compared to a fast ramp (0 to 100% over 0.5 seconds), **Then** the fast ramp produces a larger yaw rate spike, consistent with higher dω/dt and thus higher I_rotor × dω/dt torque.
4. **Given** `propulsion.rotor_inertia_I_rotor` is configured in YAML, **When** the ramp diagnostic runs, **Then** the transient yaw torque matches `I_rotor × dω_fan/dt` to within 10%.

---

### User Story 3 - Anti-Torque Coupled with Thrust Test Liftoff (Priority: P2)

A researcher reruns the existing thrust-test liftoff scenario (from the mass-properties/thrust feature) with the anti-torque model active. The drone starts on the ground, applies full thrust, lifts off, and the researcher confirms that the drone rotates in yaw during ascent. This validates that the anti-torque is present end-to-end in the full Isaac Sim physics pipeline during a realistic liftoff event, not just in a controlled zero-gravity diagnostic.

**Why this priority**: Isolated unit tests in zero-gravity may pass while the full pipeline still ignores the torque due to integration order or force/torque accumulation bugs. An end-to-end liftoff with observable yaw rotation confirms the torque contribution is correctly applied in the full simulation loop.

**Independent Test**: Run `diag_thrust_test` with anti-torque enabled. After 2 seconds at full thrust, confirm (a) altitude > 5 m (thrust still lifts off), (b) yaw angle > 5° (anti-torque is active).

**Acceptance Scenarios**:

1. **Given** the drone is on the ground with zero initial yaw, **When** full thrust (T_cmd = 1.0) is applied for 2 seconds, **Then** the drone reaches altitude > 5 m AND yaw angle > 5°.
2. **Given** the liftoff scenario runs twice — once with `propulsion.anti_torque_enabled: true` and once with `false`, **When** results are compared, **Then** the enabled case shows measurable yaw divergence (> 5°) while the disabled case shows yaw < 0.5°.
3. **Given** the diagnostic outputs a data log, **When** the log is post-processed, **Then** the yaw rate time-series correlates with thrust magnitude (higher thrust → higher yaw rate) throughout the flight.

---

### Edge Cases

- What happens when `k_torque` is set to zero in YAML? Anti-torque should be absent and the drone should not yaw.
- What happens when `I_rotor` is set to zero? RPM-ramp torque should be absent; only steady-state anti-torque remains.
- What if the drone contacts the ground while anti-torque is active? Ground friction must prevent yaw spin; the body should not rotate.
- What happens at very high thrust ramp rates (step input)? The transient torque spike could be numerically large; the simulation should not diverge.
- What is the behaviour when fan rotation direction is reversed (e.g., a counter-rotating EDF configuration)? The sign of the anti-torque should flip.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The physics model MUST apply a yaw torque on the vehicle body equal to `τ_anti = -k_Q × T` whenever thrust T > 0, where `k_Q` is the torque coefficient and the sign opposes fan rotation direction.
- **FR-002**: The physics model MUST apply an additional transient yaw torque equal to `-I_rotor × dω_fan/dt` during any period when rotor angular velocity is changing, where `I_rotor` is the rotor moment of inertia.
- **FR-003**: Both `k_Q` (torque coefficient) and `I_rotor` (rotor inertia) MUST be loaded from the vehicle YAML config and MUST NOT be hard-coded in simulation scripts.
- **FR-004**: The Isaac Sim environment MUST accept and apply the anti-torque and ramp-torque contributions through the same force/torque application path used for thrust and fin forces.
- **FR-005**: A diagnostic script MUST provide a `--mode constant` test where constant thrust is held and yaw rate is measured, and a `--mode ramp` test where thrust is ramped and the transient yaw spike is measured.
- **FR-006**: The diagnostic script MUST accept command-line arguments for thrust magnitude, ramp duration, test duration, and YAML config path.
- **FR-007**: The diagnostic script MUST output a structured log (yaw angle, yaw rate, thrust, anti-torque magnitude) at each simulation timestep for post-processing validation.
- **FR-008**: The `diag_reaction_torque` script MUST support a `--mode liftoff` test (normal gravity, ground start) with a `--disable-anti-torque` flag so the yaw response can be observed and compared against the no-torque baseline. `diag_thrust_test` is NOT modified and remains a pure thrust/altitude diagnostic.
- **FR-009**: The custom Python simulation's `ThrustModel` (non-Isaac path) MUST also apply the same anti-torque and ramp-torque contributions to ensure consistency between the custom sim and Isaac Sim.
- **FR-010**: A unit test MUST validate that the computed anti-torque value matches `k_Q × T` for a set of known thrust inputs without launching Isaac Sim.

### Key Entities

- **AntiTorqueModel**: Computes the steady-state yaw reaction torque from constant fan spin. Inputs: current thrust T, config parameter k_Q. Output: scalar yaw torque.
- **RPMRampTorqueModel**: Computes the transient yaw torque from rotor angular acceleration. Inputs: current thrust T, previous thrust T_prev, timestep dt, config parameter I_rotor. Output: scalar yaw torque (zero when thrust is constant).
- **ReactionTorqueDiagnostic**: Isaac Sim diagnostic script for measuring yaw response under constant thrust and thrust ramps. Supports both zero-gravity (isolated torque test) and normal-gravity (liftoff) modes.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: At hover thrust (T ≈ 0.68 × T_max), the measured steady-state yaw rate falls within 10% of the analytically predicted value `k_Q × T / I_zz` using YAML-configured parameters.
- **SC-002**: During a 1-second linear thrust ramp from 0 to full, the peak yaw rate exceeds the steady-state yaw rate at full thrust by at least 10%, confirming the transient component is present.
- **SC-003**: The anti-torque is absent (yaw rate < 0.01 rad/s) when `T_cmd = 0`, confirming the torque is correctly gated on fan spin.
- **SC-004**: The liftoff diagnostic with anti-torque enabled reaches altitude > 5 m in 2 seconds AND accumulates > 5° of yaw, validating end-to-end pipeline integration.
- **SC-005**: With anti-torque disabled via flag, the liftoff diagnostic shows < 0.5° of yaw, confirming the flag correctly suppresses the contribution.
- **SC-006**: The unit test for anti-torque computation passes for at least 5 thrust levels spanning 0% to 100% of T_max, without launching Isaac Sim.
- **SC-007**: All diagnostics complete without errors on RTX 5070 with single-env Isaac Sim config.

## Assumptions

- The rotor spins in a fixed direction (no counter-rotation); sign convention is that anti-torque acts in the negative yaw direction relative to the body frame's +z axis.
- Fan angular velocity ω_fan is derived from thrust: `ω_fan = sqrt(T / k_T)` where `k_T` is the thrust coefficient, consistent with actuator-disk theory. The exact mapping can be approximated or parameterized in YAML.
- The thrust model already includes a 1st-order lag (from `ThrustModel`), so `dω_fan/dt` is implicitly captured by the lag dynamics; the ramp torque should be computed from the actual (lagged) thrust derivative, not the commanded value.
- Anti-torque forces are applied via `robot.set_external_force_and_torque()` in Isaac Sim, consistent with the pattern used for wind and fin forces.
- The existing `default_vehicle.yaml` will be extended with a `propulsion.torque_coefficient_k_Q` and `propulsion.rotor_inertia_I_rotor` field.
- Feature 001 (Isaac Sim stable env) and the mass/thrust validation feature are complete before this feature is integrated.

## Dependencies

- Feature 001 (stable Isaac Sim env with fins articulation and gravity)
- Feature 002 mass/thrust/wind spec (thrust application confirmed working)
- `default_vehicle.yaml` extended with `propulsion.torque_coefficient_k_Q` and `propulsion.rotor_inertia_I_rotor`
- Isaac Sim / IsaacLab runtime for reaction torque diagnostics
- Custom simulation's `ThrustModel` for non-Isaac consistency validation
