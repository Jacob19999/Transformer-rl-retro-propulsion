<!--
Sync Impact Report
- Version change: 1.1.0 → 1.2.0
- Modified principles:
  - I. Physics Fidelity — expanded: force-at-position requirement, subsonic EDF
    modeling boundary, single physics core mandate added
  - II. Configuration-Driven Design — expanded: parameter provenance
    categorization (measured/datasheet/estimate/to-be-calibrated) added
  - III. Test-Driven Validation — expanded: ordered validation ladder
    requirement before RL training added
  - V. Sim-to-Real Integrity — expanded: single conversion boundary for
    frame/quaternion transforms added
- Added sections:
  - None (all changes expand existing sections)
- Removed sections:
  - None
- Modified sections:
  - Technical Constraints: PhysX-first backend policy, Newton behind
    abstraction, wrench dispatch adapter targeting modern composable API,
    Isaac Lab version pinned to 2.3.2
  - Development Workflow: validation ladder enforcement before RL training,
    Feature 001 reference updated to Feature 007 as the active Isaac Sim
    milestone
- Templates requiring updates:
  - .specify/templates/plan-template.md — ✅ No updates needed
  - .specify/templates/spec-template.md — ✅ No updates needed
  - .specify/templates/tasks-template.md — ✅ No updates needed
- Follow-up TODOs:
  - TODO(MASS_MIGRATION_DECISION): Decide whether mass properties are the
    source of truth in USDC or YAML. Current guidance: YAML is authoritative;
    USDC is validated against YAML. Revisit if USDC becomes the design-time
    source of truth. (Carried forward from v1.1.0)
-->

# GTrXL-PPO Retro-Propulsion Constitution

## Core Principles

### I. Physics Fidelity

All simulation models MUST faithfully represent the physical dynamics of
the EDF drone testbed. Each sub-model (thrust, aero, fins, servos, mass
properties) MUST be parameterized from measured or datasheet values, not
arbitrary tuning constants. Approximations (e.g., linearized aero,
1st-order actuator lag) MUST be documented with their validity envelope.
RK4 integration at dt=0.005s is the minimum acceptable timestep for the
custom simulation; any change MUST demonstrate equivalent or better
numerical stability.

Mass properties (total mass, center-of-mass offset, inertia tensor)
derived from YAML config primitives MUST be validated against the
corresponding Isaac Sim USDC scene rigid-body physics settings via a
dedicated validation script. Any discrepancy exceeding 1% MUST be
resolved before simulation runs are used for training or benchmarking.
YAML config remains the authoritative source of truth; the USDC scene
MUST be regenerated or patched to match.

Aerodynamic forces MUST be applied at the physical point of action (e.g.,
fin center of pressure on the fin link), not synthesized as direct body
torques. Body torque MUST emerge from articulation geometry and physics
simulation. A collapsed-body-wrench mode MAY be supported as a
performance fallback, but MUST NOT be the primary validation path.

The same physics modules MUST power all execution modes — single-env
debug, PID evaluation, hover test, landing test, and vectorized RL
environments. Only wrappers, reward profiles, and visualization settings
MAY differ. Physics code MUST NOT fork across modes.

The EDF testbed is a subscale, subsonic proxy for powered-descent
control and disturbance rejection. Simulation models MUST NOT import
supersonic or full-scale rocket coefficient values without explicit
subsonic adaptation and documented justification. Thrust-vectoring
references are used for force decomposition geometry and moment-from-
force structure, not for Mach-dependent coefficient transfer. All
modeling limitations MUST be documented alongside the models they apply
to.

**Rationale**: The entire project hinges on sim-to-real transfer. A
simulation that diverges from hardware physics — including incorrect mass
or inertia in the Isaac Sim scene, forces applied at wrong locations, or
coefficient values from an inapplicable flow regime — produces policies
that fail on the real drone.

### II. Configuration-Driven Design

All physics parameters, hyperparameters, reward weights, and environment
settings MUST reside in YAML config files under `simulation/configs/`.
No magic numbers in source code. `config_loader.py` deep-merge semantics
MUST be preserved. Domain randomization ranges MUST be specified in
`domain_randomization.yaml`, not hard-coded in environment reset logic.

USDC scene physics attributes (mass, inertia, center-of-mass) MUST NOT
be set manually without a corresponding YAML config value to validate
against. A script (`validate_usd_mass_props.py` or equivalent) MUST be
maintained that reads both the YAML config and the USDC scene and asserts
equivalence within tolerance. This script MUST be run as part of any
workflow that modifies drone geometry or mass configuration.

Config files for actuators and physical components MUST categorize each
parameter value with its provenance: **measured** (from bench testing),
**datasheet** (from manufacturer specs), **engineering estimate** (from
analytical derivation or informed approximation), or **to-be-calibrated**
(placeholder awaiting hardware data). This categorization enables
systematic calibration campaigns and prevents silent use of unvalidated
constants in training.

**Rationale**: Separating configuration from code enables systematic
sweeps, reproducible experiments, and clear audit trails for parameter
changes without code diffs. Extending this discipline to USDC physics
attributes prevents silent divergence between the Python simulation and
the Isaac Sim environment. Parameter provenance tracking ensures the
team knows which values are trustworthy and which require future
calibration.

### III. Test-Driven Validation

Every physics sub-model and training component MUST have corresponding
pytest tests. Tests MUST use dedicated lightweight configs
(`test_vehicle.yaml`, `test_environment.yaml`) and MUST NOT modify
`default_*.yaml`. New simulation features MUST include at least one
test verifying expected physical behavior (e.g., conservation laws,
known analytical solutions, boundary conditions).

For Isaac Sim features, validation MUST include at minimum:
- A thrust application test confirming the drone lifts off from the
  ground when commanded thrust exceeds vehicle weight.
- A fin articulation test confirming all four fins deflect within limits.
- An environmental force test confirming wind disturbances produce
  measurable state changes consistent with expected dynamics.

Isaac Sim environment changes MUST follow an ordered validation ladder
before large-scale RL training begins. The ladder progresses from
isolated subsystem checks (asset validation, joint axes, single-fin
articulation) through integrated system tests (force superposition,
propulsion, contacts) to closed-loop validation (PID hover, vectorized
API smoke test, all-forces hover). Each ladder rung MUST pass before
proceeding to the next. RL training on an environment that has not
completed the validation ladder is not permitted.

**Rationale**: In a research project with complex interacting subsystems,
tests are the primary defense against silent regressions that corrupt
experimental results. The ordered validation ladder isolates failure
sources systematically, preventing hours of debugging a training failure
that originates from a sign error in a single fin joint.

### IV. Reproducibility

All training runs MUST accept a `--seed` argument that fully determines
the random state. Checkpoints MUST be saved at regular intervals (every
500K steps minimum). TensorBoard logs and model artifacts MUST be stored
under `runs/` with timestamped or descriptively named subdirectories.
`VecNormalize` statistics MUST be saved alongside model checkpoints.

**Rationale**: Research conclusions require reproducible experiments.
Without seed control and checkpoint discipline, results cannot be
verified or compared across runs.

### V. Sim-to-Real Integrity

The body-frame convention (FRD, thrust along +z) and unit conventions
(radians internally, scalar-last quaternions `[qx, qy, qz, qw]`) MUST
be consistent across all modules — including Isaac Sim environments.
Any coordinate transform MUST be explicit and tested. Domain
randomization MUST be applied per-episode at env reset to build robust
policies. Observation noise injection MUST match expected sensor
characteristics of the physical testbed.

All frame and quaternion conversions between body-fixed (FRD) and
simulator-native frames (Isaac Sim world: +X forward, +Z up,
scalar-first `(w, x, y, z)`) MUST pass through a single dedicated
conversion module (e.g., `common/frames.py`, `common/quaternions.py`,
`common/transforms.py`). Conversion logic MUST NOT be scattered across
files. This single-boundary rule prevents the class of bugs where a sign
flip is correct in one file but contradicted in another.

Environmental forces (wind gusts, atmospheric disturbances) MUST be
applicable to Isaac Sim environments via the existing `WindModel` and
`AtmosphereModel` abstractions, or via equivalent Isaac Sim force APIs
that produce consistent effects. Environmental force parameters MUST be
configurable via `default_environment.yaml`; hard-coded force values in
scene or script files are not permitted.

**Rationale**: Convention mismatches between simulation and hardware are
the most common and dangerous source of sim-to-real failure. Strict
consistency and a single conversion boundary prevent sign errors and
frame confusion. Validating environmental forces in Isaac Sim ensures
the training distribution matches expected real-world disturbances.

## Technical Constraints

- **Language**: Python 3.10+
- **RL Framework**: Stable-Baselines3 (PPO) for the MLP baseline;
  custom GTrXL-PPO for the target architecture
- **Simulation Stack (active)**:
  - Custom 6-DOF rigid-body plant (`simulation/`) — baseline for physics
    validation and PID tuning
  - NVIDIA Isaac Sim 5.1 / Isaac Lab 2.3.2 — active parallel environment
    for vectorized training; Feature 007 delivers the validated Phase 1
    Isaac Sim environment as the foundation for all subsequent work
- **Physics Backend**: PhysX is the Phase 1 baseline. Newton MUST NOT be
  adopted as the Phase 1 baseline. If explored later, Newton MUST be
  introduced behind a backend abstraction layer only after the PhysX
  environment is fully validated
- **Force Application API**: Isaac Lab's composable wrench / forces-and-
  torques path is the target API. A wrench dispatch adapter layer MUST
  isolate force application from specific Isaac Lab API versions to
  enable future migration without physics code changes
- **Version Migration**: Phase 1 MUST NOT move to Isaac Lab 3.0 / Isaac
  Sim 6.0. Newer releases are monitored but introduce unnecessary churn
  before the first validated environment is established
- **Gymnasium**: All environments MUST implement the Gymnasium API
  (`reset`, `step`, `observation_space`, `action_space`)
- **State dimensions**: 18-dim state, 20-dim observation, 5-dim action
  — changes to these dimensions MUST update `ObservationPipeline`,
  reward function, and all downstream consumers
- **Integration**: RK4 with quaternion re-normalization every 10 steps
  (custom sim); Isaac Sim uses its own GPU-accelerated integrator at
  1/120 s timestep
- **Vectorized training**: `SubprocVecEnv` + `VecNormalize`; best model
  tracked by landing success rate evaluated every 100K steps
- **Mass property source of truth**: YAML config files; USDC scenes are
  derived artifacts that MUST be validated against YAML

## Development Workflow

- Feature branches MUST branch from `main` and be merged via pull
  request
- All tests MUST pass before merge (`pytest` at repo root)
- Config changes MUST be reviewed for physical plausibility
- Training experiments SHOULD be logged with hyperparameters, seed, and
  commit hash for traceability
- Diagnostic scripts (`diag_single_ep`, `diag_inertia`, `diag_yaw`)
  SHOULD be run after significant dynamics changes in the custom sim
- For Isaac Sim changes, the following validation sequence MUST be
  followed before merge:
  1. Run mass property validation script to confirm USDC ↔ YAML
     agreement
  2. Run thrust application diagnostic to confirm drone lifts off
     correctly from the ground under commanded thrust
  3. Run fin articulation diagnostic to confirm all four fins deflect
  4. Run environmental force diagnostic to confirm wind disturbances
     produce physically plausible state changes
- For Isaac Sim environment changes, the full validation ladder (asset →
  joints → articulation → force → superposition → propulsion → gyro →
  wind → contacts → PID hover → vectorized API → all-forces hover) MUST
  complete before the environment is used for RL training runs
- Feature 007 (`007-isaac-sim-env`) delivers the validated Phase 1
  Isaac Sim environment; subsequent features build on this foundation
  without breaking its acceptance scenarios

## Governance

This constitution is the authoritative reference for project standards.
All code reviews and pull requests MUST verify compliance with these
principles. Amendments require:

1. A written proposal describing the change and its rationale
2. Update to this constitution file with version increment
3. Verification that no existing code violates the amended principle
   (or a migration plan if it does)

Versioning follows semantic versioning:
- **MAJOR**: Principle removal, redefinition, or backward-incompatible
  governance change
- **MINOR**: New principle added or existing principle materially
  expanded
- **PATCH**: Clarification, wording fix, or non-semantic refinement

Runtime development guidance is maintained in `CLAUDE.md` at the
repository root.

**Version**: 1.2.0 | **Ratified**: 2026-03-10 | **Last Amended**: 2026-03-22
