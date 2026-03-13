# Implementation Plan: Isaac Sim PID Controller Tuning, Evaluation & Episode Logging

**Branch**: `003-pid-controller` | **Date**: 2026-03-12 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/003-pid-controller/spec.md`

## Summary

Implement an Isaac-Sim-native PID tuning and evaluation workflow that reuses the existing `PIDController` exactly as-is in terms of control law, axes, and observation semantics. The work splits into four concrete pieces: (1) expose PID debug telemetry without changing control outputs, (2) add runtime force toggles and richer rollout info in the Isaac environment stack, (3) build a Ziegler-Nichols-based Isaac tuning script for single- and multi-env runs, and (4) build a best-gains validation script with per-step trace logging for action/state/reward/PID internals.

The key design constraint is convention fidelity: the controller assumes inertial `NED`, body `FRD`, `h_agl` positive upward, `v_b[2] > 0` descending, forward target error mapped to negative pitch command, and yaw damping applied through the differential fin pattern `[-d, +d, +d, -d]`. The Isaac integration must preserve those exact semantics instead of normalizing to a different robotics frame.

## Technical Context

**Language/Version**: Python 3.10+  
**Primary Dependencies**: Isaac Sim v5.1.0 / IsaacLab, PyTorch, NumPy, PyYAML, Gymnasium  
**Storage**: YAML configs in `simulation/configs/`, run artifacts under `runs/`  
**Testing**: `pytest` unit tests + Isaac-marked integration tests + Isaac script smoke tests  
**Target Platform**: Windows 11 / Linux, NVIDIA RTX 5070 GPU  
**Project Type**: Research simulation / controller-tuning tooling  
**Performance Goals**: single-env debug path must stay interactive; multi-env path must reuse existing vectorized Isaac env without per-env Python object churn  
**Constraints**: preserve existing PID law; preserve `NED`/`FRD` conventions; preserve 20-D observation contract; do not introduce a second Isaac-only controller implementation  
**Scale/Scope**: 3 modified core files, 2 new Isaac scripts, 1 new Isaac integration test file, 1 existing controller test file extended

## Constitution Check

*GATE: Must pass before implementation begins. Re-check after foundational design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| **I. Physics Fidelity** | ✅ PASS | Controller semantics are inherited from the existing simulation stack; no axis remapping or alternate control law is introduced. |
| **II. Configuration-Driven** | ✅ PASS | PID gains stay YAML-driven via `simulation/configs/pid.yaml`; Isaac scripts accept config paths and runtime toggles instead of hard-coded branches. |
| **III. Test-Driven Validation** | ✅ PASS | Plan includes controller parity tests, Isaac integration tests, and explicit evaluation scripts for tuned gains. |
| **IV. Reproducibility** | ✅ PASS | Run artifacts include saved best PID config, candidate scores, episode summaries, and per-step traces. |
| **V. Sim-to-Real Integrity** | ✅ PASS | The same `PIDController` used in the custom simulation remains the source of truth for Isaac runs, preserving control structure and sign conventions. |

**Post-Phase 1 re-check**: Still passes. The design adds telemetry and wrappers around the existing controller rather than rewriting it.

## Project Structure

### Documentation (this feature)

```text
specs/003-pid-controller/
├── spec.md
├── plan.md
└── tasks.md
```

### Source Code (repository root)

```text
simulation/
├── training/
│   └── controllers/
│       └── pid_controller.py              # MODIFY: expose debug telemetry without changing actions
├── isaac/
│   ├── tasks/
│   │   └── edf_landing_task.py            # MODIFY: runtime force toggles + richer rollout info
│   ├── envs/
│   │   └── edf_isaac_env.py               # MODIFY: pass runtime overrides, surface info cleanly
│   └── scripts/
│       ├── tune_pid_isaac.py              # NEW: Ziegler-Nichols tuning + single/multi-env evaluation
│       └── test_pid_isaac.py              # NEW: best-gains validation + episode trace logging
└── tests/
    ├── test_pid_controller.py             # MODIFY: telemetry parity / convention tests
    └── test_pid_isaac.py                  # NEW: Isaac integration smoke tests for tuning/logging/toggles
```

**Structure Decision**: Isaac-only orchestration lives in `simulation/isaac/scripts/`; controller behavior remains centralized in `simulation/training/controllers/pid_controller.py`; env/task changes are limited to runtime toggles and info plumbing needed by both tuning and evaluation scripts.

## Architecture Decisions

### 1. Reuse the Existing PID Law

`PIDController.get_action()` remains authoritative. Telemetry is added by extracting the internal loop computations into a shared path so the controller can optionally return a debug snapshot alongside the action. This avoids duplicated logic between control and logging.

### 2. Add Runtime Force Toggles at the Isaac Task Layer

Force toggles belong in `edf_landing_task.py`, not in the tuning script, because the actual physics contributions are applied there. The wrapper and scripts only pass requested overrides such as:

- `disable_wind`
- `disable_gyro`
- `disable_anti_torque`
- `disable_gravity`

This keeps single-env and multi-env behavior consistent.

### 3. Single Evaluator, Two Entry Points

Both `tune_pid_isaac.py` and `test_pid_isaac.py` should use the same rollout/evaluation logic:

- instantiate `EDFIsaacEnv`
- apply runtime toggles
- reset / step vectorized envs
- run `PIDController`
- collect per-step and per-episode metrics

The difference is orchestration:

- `tune_pid_isaac.py`: generate and score candidate gains
- `test_pid_isaac.py`: load a saved best PID config and emit detailed traces

### 4. Explicit Run Artifacts

Each run should write machine-readable artifacts under `runs/`:

- `best_pid.yaml`
- `candidate_scores.csv`
- `episode_summary.csv`
- `trace_epXXXX_envYY.csv`
- `run_metadata.json`

This is required to make debugging and regression checks practical.

## File Change Summary

| File | Action | Scope |
|------|--------|-------|
| `simulation/training/controllers/pid_controller.py` | Modify | Add debug telemetry path that returns loop errors, estimates, pre-clipped commands, and final action while preserving `get_action()` behavior |
| `simulation/isaac/tasks/edf_landing_task.py` | Modify | Add runtime force-toggle hooks and expose task-level quantities needed for trace logging |
| `simulation/isaac/envs/edf_isaac_env.py` | Modify | Accept runtime overrides, surface richer `info`, and keep vectorized episode bookkeeping usable by PID scripts |
| `simulation/isaac/scripts/tune_pid_isaac.py` | New | Ziegler-Nichols tuning flow, baseline-vs-tuned comparison, single-env and multi-env candidate evaluation, reward-trend monitoring |
| `simulation/isaac/scripts/test_pid_isaac.py` | New | Load best PID YAML, run evaluation episodes, emit per-step traces and per-episode summaries |
| `simulation/tests/test_pid_controller.py` | Modify | Assert telemetry parity and preserve axis/sign conventions |
| `simulation/tests/test_pid_isaac.py` | New | Isaac-marked smoke tests for force toggles, vectorized evaluator bookkeeping, and trace-log schema |

## Complexity Tracking

No constitution violations. Complexity is justified by one constraint only: logging and tuning must not fork the PID control law. The telemetry extraction is the smallest design that preserves correctness and debuggability at the same time.

## Implementation Strategy

### MVP First

1. Add PID telemetry extraction in `pid_controller.py`
2. Add runtime force toggles and richer rollout info in `edf_landing_task.py` / `edf_isaac_env.py`
3. Build `test_pid_isaac.py` first so best-gain evaluation and trace logging work on a fixed PID config
4. Build `tune_pid_isaac.py` on top of the same evaluator for single-env runs
5. Extend the evaluator to multi-env candidate scoring

### Why This Order

The trace/evaluation path is the fastest way to validate conventions and logging before any tuning logic is trusted. If the logger says the controller is commanding the wrong signed fin pair, Ziegler-Nichols tuning will only optimize the wrong behavior faster.

### Full Delivery

1. Foundational telemetry + runtime toggles
2. Single-env evaluation and trace logging
3. Single-env Ziegler-Nichols tuning
4. Multi-env batch evaluation/tuning
5. Reward-trend warnings, run artifacts, and polish

## Notes

- `PIDController` currently contains thrust rate limiting and yaw low-pass filtering; these are part of the controller and should remain active during normal tuning.
- Yaw in the current controller is derivative-only damping (`yaw_Kd`), not a full PID loop. Tuning must respect that instead of fabricating yaw `Kp/Ki`.
- The observation contract is already defined in `simulation/training/observation.py`; Isaac code should log and validate against those exact indices.
- Existing task internals already expose toggles like `_gyro_enabled`, `_anti_torque_enabled`, and `_wind_model`; the feature should formalize those into runtime overrides rather than inventing YAML variants for every ablation case.
