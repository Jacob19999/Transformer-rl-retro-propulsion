# Implementation Plan: Refactor Drone Fin Physics Layer

**Branch**: `006-refactor-fin-physics` | **Date**: 2026-03-21 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/006-refactor-fin-physics/spec.md`

## Summary

Refactor the Isaac Sim drone physics layer to: (1) use measured PhysX joint state as the single source of truth for fin aerodynamic computations, (2) eliminate the runtime sign/index remapping hack by fixing USD hinge axes at authoring time, (3) author explicit mass properties (mass, CoM, inertia) from YAML config instead of relying on collider-derived values, (4) simplify fin colliders to box shapes, (5) replace the fragile unit auto-detection heuristic with unconditional deg↔rad conversion, and (6) add opt-in per-fin debug telemetry via GPU tensor ring buffers.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: NVIDIA Isaac Sim v5.1.0 / IsaacLab, PyTorch, USD (pxr), Stable-Baselines3
**Storage**: YAML config files (authoritative), USD/USDC scene files (derived), PyTorch `.pt` files (telemetry)
**Testing**: pytest (with `-m isaac` marker for Isaac-specific tests)
**Target Platform**: Linux/Windows with NVIDIA GPU (RTX 5070+)
**Project Type**: Research simulation library with Gymnasium-compatible RL environments
**Performance Goals**: 256+ parallel environments at ≥120 Hz physics step rate; no throughput regression from refactoring
**Constraints**: Sim-to-real fidelity — mass/inertia must match hardware within 1%/5%; fin physics must be physically correct
**Scale/Scope**: 4 fin links per drone, up to 1024 parallel environments, 7 source files modified + 1 new file

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Pre-Design Gate (Phase 0)

| Principle | Status | Evidence |
| --------- | ------ | -------- |
| I. Physics Fidelity | PASS | Feature directly improves fidelity: explicit mass properties from measured values, correct force application points, single-source-of-truth joint state |
| II. Configuration-Driven Design | PASS | All mass properties sourced from YAML `mass_properties` section; no new magic numbers. USD is derived artifact validated against YAML |
| III. Test-Driven Validation | PASS | Existing diagnostics (fin wiggle, thrust test, wind, mass validation) serve as acceptance tests; existing pytest suite must pass |
| IV. Reproducibility | PASS | No changes to seed handling, checkpointing, or logging infrastructure |
| V. Sim-to-Real Integrity | PASS | FRD convention preserved; unit convention verified and documented; sign convention fixed at authoring time to match hardware |
| Development Workflow | PASS | Validation sequence defined in quickstart.md follows the mandated order: mass props → thrust → fins → wind → tests |

**Gate result**: ALL PASS — proceed to Phase 0.

### Post-Design Gate (Phase 1)

| Principle | Status | Evidence |
| --------- | ------ | -------- |
| I. Physics Fidelity | PASS | Inertia tensor eigendecomposed and authored to USD; `validate_mass_props` confirms 1% mass / 5% inertia tolerance |
| II. Configuration-Driven Design | PASS | `use_explicit: true` flag in YAML controls authoring path; no new hardcoded values except `_FIN_MASS_KG = 1e-5` (unchanged) |
| III. Test-Driven Validation | PASS | Existing test suite + diagnostics cover all functional requirements. Telemetry adds self-consistency checks |
| IV. Reproducibility | PASS | Unchanged |
| V. Sim-to-Real Integrity | PASS | Hinge axes authored to match controller convention; deg↔rad verified unconditionally; force application at physical fin link |
| Development Workflow | PASS | quickstart.md documents full validation sequence |

**Gate result**: ALL PASS — no violations to track.

## Project Structure

### Documentation (this feature)

```text
specs/006-refactor-fin-physics/
├── plan.md              # This file
├── research.md          # Phase 0: joint units, sign conventions, mass pipeline, colliders, telemetry
├── data-model.md        # Phase 1: entity definitions, state transitions, data flows
├── quickstart.md        # Phase 1: verification commands, key files
├── checklists/
│   └── requirements.md  # Spec quality checklist
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
simulation/isaac/
├── usd/
│   └── postprocess_usd.py       # MODIFY: explicit mass/CoM/inertia; fix hinge axes; box fin colliders
├── tasks/
│   └── edf_landing_task.py      # MODIFY: remove FinMapping remap; unconditional deg→rad; telemetry hooks
├── fin_mapping.py               # MODIFY: set defaults to identity or deprecate
├── fin_aero.py                  # VERIFY: no changes needed (input source already measured state)
├── fin_telemetry.py             # NEW: GPU tensor ring buffer for per-fin debug data
├── conventions.py               # MODIFY: remove FIN_JOINT_VISUAL_SIGN; document unit convention
├── envs/
│   └── edf_isaac_env.py         # VERIFY: Gymnasium API preserved (FR-009)
├── scripts/
│   ├── validate_mass_props.py   # MODIFY: ensure full inertia tensor validation
│   ├── diag_fin_wiggle.py       # VERIFY: runs without PhysX warnings after collider change
│   └── calibrate_fin_mapping.py # MODIFY: update or remove if mapping is now identity
└── configs/
    ├── fin_mapping.yaml         # MODIFY: update to identity mapping
    └── isaac_env_base.yaml      # MODIFY: add debug.fin_telemetry flag

simulation/configs/
└── default_vehicle.yaml         # VERIFY: mass_properties section complete and accurate

simulation/isaac/usd/
└── parts_registry.py            # VERIFY: load_fin_specs() hinge axis handling

simulation/tests/
├── test_isaac_env.py            # MODIFY: update if sign convention affects test expectations
└── test_drone_builder.py        # VERIFY: passes after USD changes
```

**Structure Decision**: This feature modifies the existing Isaac Sim simulation layer in-place. No new directories are needed. One new module (`fin_telemetry.py`) is added to `simulation/isaac/`. All changes stay within the existing project structure.

## Complexity Tracking

No constitution violations — this section is intentionally empty.
