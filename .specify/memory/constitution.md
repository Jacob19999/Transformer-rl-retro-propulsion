<!--
Sync Impact Report
- Version change: N/A (initial) → 1.0.0
- Added principles:
  - I. Physics Fidelity
  - II. Configuration-Driven Design
  - III. Test-Driven Validation
  - IV. Reproducibility
  - V. Sim-to-Real Integrity
- Added sections:
  - Technical Constraints
  - Development Workflow
  - Governance
- Removed sections: None
- Templates requiring updates:
  - .specify/templates/plan-template.md — ✅ No updates needed (Constitution Check section is generic)
  - .specify/templates/spec-template.md — ✅ No updates needed (requirements structure compatible)
  - .specify/templates/tasks-template.md — ✅ No updates needed (phase structure compatible)
- Follow-up TODOs: None
-->

# GTrXL-PPO Retro-Propulsion Constitution

## Core Principles

### I. Physics Fidelity

All simulation models MUST faithfully represent the physical dynamics of
the EDF drone testbed. Each sub-model (thrust, aero, fins, servos, mass
properties) MUST be parameterized from measured or datasheet values, not
arbitrary tuning constants. Approximations (e.g., linearized aero,
1st-order actuator lag) MUST be documented with their validity envelope.
RK4 integration at dt=0.005s is the minimum acceptable timestep; any
change MUST demonstrate equivalent or better numerical stability.

**Rationale**: The entire project hinges on sim-to-real transfer. A
simulation that diverges from hardware physics produces policies that
fail on the real drone.

### II. Configuration-Driven Design

All physics parameters, hyperparameters, reward weights, and environment
settings MUST reside in YAML config files under `simulation/configs/`.
No magic numbers in source code. `config_loader.py` deep-merge semantics
MUST be preserved. Domain randomization ranges MUST be specified in
`domain_randomization.yaml`, not hard-coded in environment reset logic.

**Rationale**: Separating configuration from code enables systematic
sweeps, reproducible experiments, and clear audit trails for parameter
changes without code diffs.

### III. Test-Driven Validation

Every physics sub-model and training component MUST have corresponding
pytest tests. Tests MUST use dedicated lightweight configs
(`test_vehicle.yaml`, `test_environment.yaml`) and MUST NOT modify
`default_*.yaml`. New simulation features MUST include at least one
test verifying expected physical behavior (e.g., conservation laws,
known analytical solutions, boundary conditions).

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
be consistent across all modules. Any coordinate transform MUST be
explicit and tested. Domain randomization MUST be applied per-episode at
env reset to build robust policies. Observation noise injection MUST
match expected sensor characteristics of the physical testbed.

**Rationale**: Convention mismatches between simulation and hardware are
the most common and dangerous source of sim-to-real failure. Strict
consistency prevents sign errors and frame confusion that would crash
the real drone.

## Technical Constraints

- **Language**: Python 3.10+
- **RL Framework**: Stable-Baselines3 (PPO) for the MLP baseline;
  custom GTrXL-PPO for the target architecture
- **Simulation**: Custom 6-DOF rigid-body plant (not Isaac Sim at this
  stage; Isaac Sim is a future milestone)
- **Gymnasium**: All environments MUST implement the Gymnasium API
  (`reset`, `step`, `observation_space`, `action_space`)
- **State dimensions**: 18-dim state, 20-dim observation, 5-dim action
  — changes to these dimensions MUST update `ObservationPipeline`,
  reward function, and all downstream consumers
- **Integration**: RK4 with quaternion re-normalization every 10 steps
- **Vectorized training**: `SubprocVecEnv` + `VecNormalize`; best model
  tracked by landing success rate evaluated every 100K steps

## Development Workflow

- Feature branches MUST branch from `main` and be merged via pull
  request
- All tests MUST pass before merge (`pytest` at repo root)
- Config changes MUST be reviewed for physical plausibility
- Training experiments SHOULD be logged with hyperparameters, seed, and
  commit hash for traceability
- Diagnostic scripts (`diag_single_ep`, `diag_inertia`, `diag_yaw`)
  SHOULD be run after significant dynamics changes to verify behavior

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

**Version**: 1.0.0 | **Ratified**: 2026-03-10 | **Last Amended**: 2026-03-10
