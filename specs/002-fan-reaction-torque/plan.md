# Implementation Plan: EDF Fan Reaction Torque

**Branch**: `002-fan-reaction-torque` | **Date**: 2026-03-12 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/002-fan-reaction-torque/spec.md`

## Summary

Implement two missing yaw torque effects from EDF fan rotation: (1) steady-state anti-torque `τ = -k_torque × ω²` present whenever thrust > 0, and (2) port the existing RPM-ramp reaction torque `τ = -I_fan × dω/dt` from the custom Python sim to Isaac Sim. Both effects are applied about the body yaw axis (FRD +Z). The config parameter `k_torque: 1.0e-8` already exists in `default_vehicle.yaml` but is unused — this feature activates it. A new diagnostic script validates both effects in zero-gravity and liftoff scenarios.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: Isaac Sim v5.1.0 / IsaacLab, PyTorch (GPU tensors), pxr (OpenUSD), NumPy, PyYAML
**Storage**: YAML config files (`simulation/configs/`)
**Testing**: pytest (unit tests without Isaac Sim), Isaac Sim diagnostics (integration)
**Target Platform**: Windows 11 / Linux, NVIDIA RTX 5070 GPU
**Project Type**: Research simulation (physics sub-model + Isaac Sim integration + diagnostic CLI)
**Performance Goals**: Single-env diagnostic at 120 Hz physics step; reaction torque computation adds negligible overhead (<0.1% of step time)
**Constraints**: All physics params from YAML (no magic numbers); body-frame FRD convention; scalar-last quaternions
**Scale/Scope**: 3 modified files, 2 new files, 1 YAML extension, 1 unit test file

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| **I. Physics Fidelity** | ✅ PASS | Anti-torque is parameterized from `k_torque` (datasheet-derived), not arbitrary tuning. RPM-ramp uses `I_fan` (measured rotor inertia). Both effects documented with validity envelope in research.md. |
| **II. Configuration-Driven** | ✅ PASS | `k_torque` already in YAML; new `anti_torque.enabled` toggle added. No magic numbers in source code. |
| **III. Test-Driven Validation** | ✅ PASS | Unit test (`test_reaction_torque.py`) validates math without Isaac Sim. Isaac Sim diagnostic (`diag_reaction_torque.py`) validates end-to-end. |
| **IV. Reproducibility** | ✅ PASS | Diagnostic is deterministic (single env, fixed config). No training changes in this feature. |
| **V. Sim-to-Real Integrity** | ✅ PASS | Same physics (`k_torque × ω²`) in both custom sim and Isaac Sim. Sign convention verified against existing gyro precession code (research.md RQ-6). FRD body frame preserved. |

**Post-Phase 1 re-check**: All gates still pass. Design adds `anti_torque.enabled` toggle to YAML (Principle II), unit tests for math (Principle III), and consistent sign convention with gyro precession (Principle V).

## Project Structure

### Documentation (this feature)

```text
specs/002-fan-reaction-torque/
├── plan.md              # This file
├── research.md          # Phase 0: 7 research questions resolved
├── data-model.md        # Phase 1: entity definitions and state transitions
├── quickstart.md        # Phase 1: validation commands
├── contracts/
│   └── cli-contracts.md # Phase 1: diagnostic CLI interface
└── tasks.md             # Phase 2 output (via /speckit.tasks)
```

### Source Code (repository root)

```text
simulation/
├── configs/
│   └── default_vehicle.yaml              # MODIFY: activate k_torque, add anti_torque.enabled
├── dynamics/
│   └── thrust_model.py                   # MODIFY: load k_torque, add steady_state_anti_torque()
├── isaac/
│   ├── tasks/
│   │   └── edf_landing_task.py           # MODIFY: add anti-torque + ramp-torque to _apply_action()
│   ├── scripts/
│   │   ├── diag_reaction_torque.py       # NEW: diagnostic with --mode constant/ramp/liftoff
│   │   └── diag_thrust_test.py           # MODIFY: add --disable-anti-torque flag
│   └── configs/
│       └── isaac_env_gyro_test.yaml      # REUSE: zero-gravity config for constant/ramp modes
└── tests/
    └── test_reaction_torque.py           # NEW: unit tests for anti-torque math
```

**Structure Decision**: Follows existing project layout. New files placed alongside existing diagnostics and tests. No new directories needed.

## File Change Summary

| File | Action | Scope |
|------|--------|-------|
| `simulation/dynamics/thrust_model.py` | Modify | Add `k_torque` + `anti_torque_enabled` to `ThrustModelConfig`; add `steady_state_anti_torque()` method; include in `outputs()` return |
| `simulation/isaac/tasks/edf_landing_task.py` | Modify | Add anti-torque + ramp-torque blocks in `_apply_action()` after gyro precession (line ~552); load config at init |
| `simulation/configs/default_vehicle.yaml` | Modify | Add `anti_torque.enabled: true` under `edf` section |
| `simulation/isaac/scripts/diag_reaction_torque.py` | New | Diagnostic CLI with 3 modes, structured CSV output, pass/fail criteria |
| `simulation/isaac/scripts/diag_thrust_test.py` | Modify | Add `--disable-anti-torque` flag |
| `simulation/tests/test_reaction_torque.py` | New | 7 unit test cases for anti-torque + ramp-torque math |

## Complexity Tracking

No constitution violations. No complexity justification needed.
