<!--
Sync Impact Report
- Version change: 1.0.0 → 1.1.0
- Modified principles:
  - I. Physics Fidelity — expanded: mass property validation from USDC scene added
  - II. Configuration-Driven Design — expanded: USDC ↔ YAML mass property reconciliation requirement added
  - V. Sim-to-Real Integrity — expanded: environmental forces (wind disturbances) added
- Added sections:
  - Technical Constraints: Isaac Sim status promoted from "future milestone" to active milestone
  - Development Workflow: Isaac Sim thrust test env and mass validation script requirements added
- Removed sections: None
- Templates requiring updates:
  - .specify/templates/plan-template.md — ✅ No updates needed (Constitution Check section is generic)
  - .specify/templates/spec-template.md — ✅ No updates needed (requirements structure compatible)
  - .specify/templates/tasks-template.md — ✅ No updates needed (phase structure compatible)
- Follow-up TODOs:
  - TODO(MASS_MIGRATION_DECISION): Decide whether mass properties are the source of truth in USDC or YAML.
    Current guidance: YAML is authoritative; USDC is validated against YAML. Revisit if USDC becomes
    the design-time source of truth.
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

**Rationale**: The entire project hinges on sim-to-real transfer. A
simulation that diverges from hardware physics — including incorrect mass
or inertia in the Isaac Sim scene — produces policies that fail on the
real drone.

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

**Rationale**: Separating configuration from code enables systematic
sweeps, reproducible experiments, and clear audit trails for parameter
changes without code diffs. Extending this discipline to USDC physics
attributes prevents silent divergence between the Python simulation and
the Isaac Sim environment.

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

**Rationale**: In a research project with complex interacting subsystems,
tests are the primary defense against silent regressions that corrupt
experimental results.

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

Environmental forces (wind gusts, atmospheric disturbances) MUST be
applicable to Isaac Sim environments via the existing `WindModel` and
`AtmosphereModel` abstractions, or via equivalent Isaac Sim force APIs
that produce consistent effects. Environmental force parameters MUST be
configurable via `default_environment.yaml`; hard-coded force values in
scene or script files are not permitted.

**Rationale**: Convention mismatches between simulation and hardware are
the most common and dangerous source of sim-to-real failure. Strict
consistency prevents sign errors and frame confusion. Validating
environmental forces in Isaac Sim ensures the training distribution
matches expected real-world disturbances.

## Technical Constraints

- **Language**: Python 3.10+
- **RL Framework**: Stable-Baselines3 (PPO) for the MLP baseline;
  custom GTrXL-PPO for the target architecture
- **Simulation Stack (active)**:
  - Custom 6-DOF rigid-body plant (`simulation/`) — baseline for physics
    validation and PID tuning
  - NVIDIA Isaac Sim / IsaacLab — active parallel environment for
    vectorized training; Feature 001 delivers a stable single-env with
    fins articulation as the foundation for all subsequent Isaac Sim work
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
- Feature 001 (`001-isaac-sim-env`) MUST remain the stable baseline for
  Isaac Sim work; subsequent features build on this foundation without
  breaking its acceptance scenarios

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

**Version**: 1.1.0 | **Ratified**: 2026-03-10 | **Last Amended**: 2026-03-11
