# Feature Specification: Refactor Drone Fin Physics Layer

**Feature Branch**: `006-refactor-fin-physics`
**Created**: 2026-03-21
**Status**: Draft
**Input**: User description: "Refactoring of the drone physics layer — unify fin joint state, fix sign conventions, apply aero forces at fin links, author real mass properties, simplify colliders, add fin-debug telemetry"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Consistent Fin Behaviour Across Command and Simulation (Priority: P1)

As a simulation developer, I need the fin deflection that drives aerodynamic force computation to be the same value that PhysX is actually applying to the articulated joint, so that the forces I see in telemetry match the physical motion I observe in the viewport.

**Why this priority**: This is the foundational correctness issue. If commanded state and physical state diverge, every downstream result (aero forces, reward signal, trained policy) is unreliable. Nothing else matters until the single-source-of-truth property holds.

**Independent Test**: Command a single fin to a known deflection (e.g. +15 deg), read back the joint position from the articulation, and confirm the aero force computation uses that same measured value — not a separate analytic state.

**Acceptance Scenarios**:

1. **Given** a running Isaac Sim environment with the drone articulation, **When** a fin target of +0.26 rad (~15 deg) is commanded for fin 0, **Then** the joint position read back from `robot.data.joint_pos` for that fin matches the target within the drive settling tolerance, and the aero force is computed from that measured joint position.
2. **Given** a step where the fin drive has not yet settled to the target (transient), **When** the aero force is computed, **Then** it uses the current measured joint position (lagging the target), not the commanded target.
3. **Given** all four fins commanded to zero deflection, **When** telemetry is inspected, **Then** measured joint positions, aero forces, and visual fin orientations all read zero (within floating-point tolerance).

---

### User Story 2 - Unified Sign Convention Without Manual Remapping (Priority: P1)

As a simulation developer, I need a positive fin command to produce a consistent positive deflection at the USD joint, in the aero model, and in the visual mesh — without any swap-and-negate mapping layer in the runtime task code.

**Why this priority**: The existing index-swap and sign-negation mapping (`joint_source_indices`, `joint_signs`, `FIN_JOINT_VISUAL_SIGN`) is a maintenance hazard and a frequent source of bugs. Eliminating it requires aligning the USD hinge axes and fin local frames at authoring time, which is a prerequisite for trustworthy aero force application (Story 3).

**Independent Test**: After the convention fix, remove or disable the `FinMapping` remap logic in the task. Command each fin individually to a positive deflection and confirm: (a) the joint rotates in the expected positive direction, (b) the aero force direction is physically correct, and (c) the visual mesh rotates consistently.

**Acceptance Scenarios**:

1. **Given** the updated USD asset with corrected hinge axes, **When** fin 0 is commanded to +0.26 rad, **Then** the joint rotates in the direction that produces positive lift according to the fin's defined lift direction — with no runtime sign correction applied.
2. **Given** the four canonical fin names `[RightFin, LeftFin, FwdFin, AftFin]`, **When** the task resolves joint IDs, **Then** the mapping from canonical index to joint index is identity (0→0, 1→1, 2→2, 3→3) and all joint signs are +1.
3. **Given** a training run using the refactored environment, **When** the policy commands positive deflection on all fins, **Then** the resulting pitch/roll/yaw moments match the expected sign from the vehicle configuration — without any additional sign fixups.

---

### User Story 3 - Aerodynamic Forces Applied at Fin Links (Priority: P2)

As a simulation developer, I need each fin's aerodynamic force to be applied at the fin link (or its center of pressure) rather than aggregated into a single body torque, so that PhysX correctly propagates the articulated load and I can debug per-fin contributions independently.

**Why this priority**: Applying forces at the correct point lets PhysX compute the resulting moments from the actual moment arm, which improves fidelity and makes the simulation easier to debug and validate against hardware data.

**Independent Test**: Command one fin to a large deflection while the others are at zero. Inspect the external force array and confirm only that fin's link has a non-zero force. Observe the drone's attitude response and verify the moment direction matches the expected cross-product of fin position and force.

**Acceptance Scenarios**:

1. **Given** fin 0 commanded to +0.26 rad and fins 1–3 at zero, **When** external forces are applied, **Then** only the link corresponding to fin 0 has a non-zero aerodynamic force vector; the body link has only thrust.
2. **Given** symmetric fin deflections producing a pure pitch moment, **When** the drone is in free flight, **Then** the resulting angular acceleration is about the pitch axis only (roll and yaw rates remain near zero), confirming correct moment-arm geometry.
3. **Given** the force application point on a fin link, **When** compared to the analytic moment arm from the vehicle config, **Then** the resulting torque (force × arm) matches the expected value within 5%.

---

### User Story 4 - Explicit Mass Properties from Hardware Data (Priority: P2)

As a simulation developer, I need the drone's mass, center of mass, and inertia tensor to be authored explicitly in the USD asset from measured or CAD-derived hardware values, so the simulation matches the real vehicle and does not depend on collider geometry for mass properties.

**Why this priority**: Collider-derived inertia is a rough approximation that diverges from the real drone. Accurate mass properties are essential for sim-to-real transfer, especially for attitude dynamics and control tuning.

**Independent Test**: Run the existing `validate_mass_props` script against the updated USD and confirm mass, CoM, and principal inertias match the YAML config values within tolerance.

**Acceptance Scenarios**:

1. **Given** the updated USD asset, **When** `validate_mass_props` is run, **Then** total mass matches `default_vehicle.yaml` within 1%, CoM position matches within 1 mm, and principal inertias match within 5%.
2. **Given** the runtime Isaac environment loads the updated USD, **When** `robot.data.default_mass` and `robot.data.default_inertia` are read, **Then** they match the authored values (not collider-derived estimates).
3. **Given** a gravity-drop diagnostic, **When** the drone free-falls and lands, **Then** the angular response on contact is consistent with the authored inertia tensor (no unexpected tumbling from incorrect inertia).

---

### User Story 5 - Simplified Fin Colliders (Priority: P3)

As a simulation developer, I need fin collision geometry to use simple shapes (box or convex hull) instead of convex decomposition, so that the PhysX solver is more stable and the simulation runs faster.

**Why this priority**: Convex decomposition on small, thin fin meshes can produce degenerate or overly complex collision shapes that hurt solver stability and performance. Simple colliders are recommended by NVIDIA for small articulated parts.

**Independent Test**: Replace fin colliders with box approximations, run the fin wiggle diagnostic, and confirm fins articulate without collision instability or interpenetration artifacts.

**Acceptance Scenarios**:

1. **Given** the updated USD with simplified fin colliders, **When** the fin wiggle diagnostic (`diag_fin_wiggle`) runs for 100 episodes, **Then** no PhysX warnings or solver failures occur.
2. **Given** a training run with 256 environments, **When** fin colliders are simplified, **Then** step throughput improves or remains the same (no regression) compared to convex decomposition.
3. **Given** fins at maximum deflection, **When** inspected in the viewport, **Then** the simplified collider envelopes the fin mesh without significant gaps or excess volume.

---

### User Story 6 - Verified Runtime Units for Joint Targets (Priority: P2)

As a simulation developer, I need the unit convention for joint position targets to be verified and documented, so that the rad-to-deg conversion before `set_joint_position_target()` is confirmed correct (or fixed) and the auto-detection heuristic in the read path is eliminated.

**Why this priority**: A unit mismatch between command and readback silently corrupts the aero model. The current auto-detection heuristic (threshold at 3.5) is fragile and can misclassify small deflections.

**Independent Test**: Command a known deflection, read back the joint position without any auto-conversion, and confirm the raw value matches expectations for the IsaacLab API's documented unit convention.

**Acceptance Scenarios**:

1. **Given** the IsaacLab joint API documentation, **When** the unit convention is verified, **Then** the code either always converts to the correct unit or operates natively in that unit — with no runtime heuristic.
2. **Given** a fin commanded to 0.10 rad (~5.7 deg), **When** the joint position is read back, **Then** the value is in the expected unit without auto-detection, and the aero model receives the correct radians value.
3. **Given** the verified convention, **When** documented in a code comment or conventions file, **Then** future developers can confirm the unit expectation without reverse-engineering the heuristic.

---

### User Story 7 - Per-Fin Debug Telemetry (Priority: P3)

As a simulation developer, I need a dedicated telemetry path that logs per-fin data (commanded angle, actual joint angle, link pose, flow velocity, angle of attack, applied force, joint wrench) so that I can diagnose fin behaviour without adding ad-hoc print statements.

**Why this priority**: The current debug output is limited to one-time joint resolution prints. Ongoing per-step fin telemetry is needed for validating the refactored physics layer and for future hardware correlation.

**Independent Test**: Enable fin telemetry, run a single episode, and confirm the telemetry log contains all specified fields for all four fins at each step.

**Acceptance Scenarios**:

1. **Given** fin telemetry is enabled (via config flag or environment variable), **When** one episode runs, **Then** a structured log or tensor buffer contains per-fin: commanded angle, measured joint angle, link world pose, local exhaust velocity, angle of attack, applied aero force (3-vector), and incoming joint wrench.
2. **Given** fin telemetry is disabled (default), **When** a training run executes, **Then** there is no measurable performance overhead from the telemetry system.
3. **Given** the telemetry output for a known single-fin deflection, **When** cross-checked against manual calculation, **Then** all logged quantities are mutually consistent (e.g., AoA matches deflection angle, force direction matches lift direction).

---

### Edge Cases

- What happens when a fin drive cannot reach its target within one physics step (very stiff drive with large step)? The aero model must use the actual (lagging) position, not the target.
- What happens when a fin hits its joint limit? The aero force must reflect the clamped position, and no solver instability should occur.
- What happens when fin mass is set to near-zero (current 1e-5 kg) but explicit inertia is authored? The mass/inertia ratio across links must remain within PhysX stability bounds.
- What happens if the USD asset is regenerated from Blender with different mesh orientation? The sign convention must be enforced by the postprocess script, not assumed from mesh authoring.
- What happens during domain randomization that perturbs fin parameters? The single-source-of-truth property must hold even with randomized drive stiffness or aero coefficients.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST compute fin aerodynamic forces using the measured joint position from the PhysX articulation, not from a separate analytic state variable.
- **FR-002**: The system MUST apply aerodynamic forces at each fin's link body (or its defined center of pressure), not aggregated at the main body.
- **FR-003**: The USD asset's hinge axes and fin local frames MUST be authored so that a positive joint command produces positive deflection in the controller convention — eliminating all runtime index-swap and sign-negate mappings.
- **FR-004**: The USD postprocess script MUST author explicit mass, center of mass, and diagonal inertia for the body and fin links from YAML-configured hardware values.
- **FR-005**: The USD postprocess script MUST use simple collision approximations (box or convex hull) for fin links instead of convex decomposition.
- **FR-006**: The runtime task MUST use a single, verified unit convention for joint targets and joint readback — no auto-detection heuristic.
- **FR-007**: The system MUST provide an opt-in per-fin telemetry mode that logs commanded angle, measured joint angle, link pose, local flow velocity, angle of attack, applied force, and incoming joint wrench per physics step.
- **FR-008**: The `FinMapping` runtime remap (index swap and sign correction) MUST be removed or reduced to identity once USD conventions are fixed.
- **FR-009**: The system MUST preserve compatibility with the existing RL training pipeline (observation space, action space, reward structure).
- **FR-010**: The system MUST validate mass properties at USD build time (via `validate_mass_props`) and warn if authored values deviate from YAML config.

### Key Entities

- **Fin Link**: An articulated rigid body attached to the drone body via a revolute joint. Has mass, inertia, collision geometry, and receives external aerodynamic forces. Four instances: Right, Left, Forward, Aft.
- **Fin Joint**: A revolute (hinge) joint connecting a fin link to the drone body. Has position target, drive stiffness/damping, joint limits, and returns measured position/velocity.
- **Drone Body**: The main rigid body of the articulation. Receives thrust force and environmental forces. Has authored mass properties.
- **Fin Telemetry Record**: A per-step, per-fin data structure containing commanded angle, measured angle, link pose, flow velocity, angle of attack, applied force, and joint wrench.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: After refactoring, the runtime task code contains zero sign-correction or index-swap operations for fin joint mapping (the `FinMapping` remap is identity or removed).
- **SC-002**: Per-fin aero forces, when summed and cross-producted with their moment arms, reproduce the expected pitch/roll/yaw moments within 5% of the analytic single-body calculation for a given set of fin deflections.
- **SC-003**: The `validate_mass_props` script passes with total mass within 1%, CoM within 1 mm, and principal inertias within 5% of YAML-configured values.
- **SC-004**: The fin wiggle diagnostic (`diag_fin_wiggle`) runs 100 episodes with zero PhysX solver warnings or instability events using simplified colliders.
- **SC-005**: A trained RL policy on the refactored environment achieves equivalent or better landing success rate compared to the pre-refactor baseline (no regression from physics changes).
- **SC-006**: All existing unit tests and integration tests pass without modification (or with documented, justified updates).
- **SC-007**: Fin telemetry, when enabled, produces a complete per-fin record at every physics step, and all logged quantities are mutually consistent within floating-point tolerance.

## Assumptions

- The IsaacLab articulation API supports applying external forces to individual link bodies (confirmed by current code using `set_external_force_and_torque` with per-body IDs).
- The USD postprocess script is the single authoring point for physics properties — no manual USD edits are needed after postprocessing.
- Hardware-measured mass properties (mass, CoM, inertia) are available in `default_vehicle.yaml` or can be derived from the existing mass primitives configuration.
- The IsaacLab joint position target API uses degrees (to be verified as part of this feature; the spec treats this as an open question to resolve, not an assumption).
- Fin telemetry will be stored in GPU tensor buffers and optionally flushed to disk, rather than using Python print statements.
- The existing `fin_aero.py` module's force computation logic is correct and only needs its input source changed (from analytic state to measured joint state).
