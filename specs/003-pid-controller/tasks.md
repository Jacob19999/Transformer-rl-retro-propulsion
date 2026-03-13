# Tasks: Isaac Sim PID Controller Tuning, Evaluation & Episode Logging

**Input**: Design documents from `specs/003-pid-controller/`  
**Prerequisites**: `plan.md` (required), `spec.md` (required)

**Tests**: Unit tests for controller telemetry plus Isaac-marked integration tests for force toggles, vectorized rollout bookkeeping, and trace-log schema.

**Organization**: Tasks are grouped by user story, but the blocking foundational phase must complete first because every story depends on the same controller telemetry and Isaac runtime overrides.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (for example `US1`)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Script and Test Scaffolds)

**Purpose**: Create the two Isaac PID entry points and the Isaac integration test module so later phases land on stable files.

- [ ] T001 Create `simulation/isaac/scripts/tune_pid_isaac.py` scaffold with argparse for `--config`, `--pid-config`, `--method`, `--episodes`, `--seed`, `--output-dir`, `--log-dir`, `--disable-wind`, `--disable-gyro`, `--disable-anti-torque`, `--disable-gravity`, `--monitor-window`, and `--monitor-direction`
- [ ] T002 [P] Create `simulation/isaac/scripts/test_pid_isaac.py` scaffold with argparse for `--config`, `--pid-config`, `--episodes`, `--seed`, `--log-dir`, and the same runtime force-toggle flags
- [ ] T003 [P] Create `simulation/tests/test_pid_isaac.py` scaffold marked with `pytest.mark.isaac` for PID/Isaac integration smoke tests

**Checkpoint**: The feature has stable script/test entry points; later phases only fill in logic.

---

## Phase 2: Foundational (Blocking Runtime and Telemetry Plumbing)

**Purpose**: Expose PID internals for logging and add runtime force toggles at the Isaac env/task layer. No tuning or trace logging should start before this phase is done.

**CRITICAL**: Phases 3–6 all depend on this phase.

- [ ] T004 Refactor `simulation/training/controllers/pid_controller.py` so the action computation flows through one shared internal path that can return both the final action and a debug payload, while keeping `get_action()` output unchanged
- [ ] T005 [P] Extend `simulation/tests/test_pid_controller.py` with telemetry-parity tests: `get_action()` must match the debug path exactly, forward target error must still map to negative pitch / negative `Fin_1` + `Fin_2`, right target error must still map to roll on `Fin_3` + `Fin_4`, and yaw damping must preserve the `[-d,+d,+d,-d]` pattern
- [ ] T006 Add runtime force-toggle support in `simulation/isaac/tasks/edf_landing_task.py` for wind, gyro precession, anti-torque, and optional gravity disable; toggles must be reset-safe and must not require YAML edits
- [ ] T007 Extend `simulation/isaac/envs/edf_isaac_env.py` so callers can pass runtime overrides into the task and so `step()` returns richer info needed by PID scripts (`episode_step`, success/failure flags, altitude, termination reason, and task-side rollout diagnostics)
- [ ] T008 Add Isaac integration smoke tests in `simulation/tests/test_pid_isaac.py` that instantiate `EDFIsaacEnv`, apply runtime toggles, and assert the requested task flags/models actually change before rollout begins

**Checkpoint**: PID internals are inspectable, Isaac runtime toggles are plumbed end-to-end, and scripts can trust the env/task API.

---

## Phase 3: User Story 1 — Single-Environment Isaac PID Tuning (Priority: P1) MVP

**Goal**: Tune the existing PID against Isaac Sim in single-env mode using Ziegler-Nichols-style candidate generation and baseline-vs-tuned evaluation.

**Independent Test**: Run `python -m simulation.isaac.scripts.tune_pid_isaac --config simulation/isaac/configs/isaac_env_single.yaml --pid-config simulation/configs/pid.yaml --method ziegler-nichols --episodes 20 --disable-wind --disable-gyro` and verify tuned gains plus a saved best PID YAML are produced.

### Implementation for User Story 1

- [ ] T009 [US1] In `simulation/isaac/scripts/tune_pid_isaac.py`, implement deterministic env construction, seed handling, baseline PID loading from `simulation/configs/pid.yaml`, and run-directory creation under `runs/`
- [ ] T010 [US1] Implement a reusable single-env rollout evaluator in `simulation/isaac/scripts/tune_pid_isaac.py` that runs `PIDController`, accumulates per-episode reward / success / CEP / termination metrics, and optionally captures controller debug telemetry
- [ ] T011 [US1] Implement Ziegler-Nichols-style candidate generation in `simulation/isaac/scripts/tune_pid_isaac.py` for `outer_loop.altitude`, `outer_loop.lateral_x`, `outer_loop.lateral_y`, `inner_loop.roll`, `inner_loop.pitch`, and derivative-only yaw damping via `inner_loop.yaw_Kd`
- [ ] T012 [US1] Implement baseline-vs-tuned ranking, `best_pid.yaml` writing, and `candidate_scores.csv` output in `simulation/isaac/scripts/tune_pid_isaac.py`
- [ ] T013 [US1] Add single-env Isaac smoke tests in `simulation/tests/test_pid_isaac.py` that run a short evaluation with disabled wind / gyro and assert finite episode summaries plus a writable best-PID artifact

**Checkpoint**: Single-env Isaac tuning completes end-to-end, writes a best-gains YAML, and reports baseline-vs-tuned scores.

---

## Phase 4: User Story 2 — Multi-Environment Isaac PID Batch Tuning (Priority: P1)

**Goal**: Reuse the same evaluator and controller in multi-env Isaac runs so multiple episodes/candidates can be scored efficiently without cross-env state leakage.

**Independent Test**: Run `python -m simulation.isaac.scripts.tune_pid_isaac --config simulation/isaac/configs/isaac_env_128.yaml --pid-config simulation/configs/pid.yaml --method ziegler-nichols --episodes 64 --disable-wind --disable-gyro` and verify a ranked candidate table is returned.

### Implementation for User Story 2

- [ ] T014 [US2] Extend `simulation/isaac/scripts/tune_pid_isaac.py` so the evaluator handles batched observations/actions/dones from `simulation/isaac/envs/edf_isaac_env.py` without per-env controller-state leakage
- [ ] T015 [US2] Implement vectorized episode aggregation in `simulation/isaac/scripts/tune_pid_isaac.py`: track per-env completion, completed episode counts, candidate-level reward stats, and candidate-level success / CEP summaries
- [ ] T016 [US2] Add progress reporting in `simulation/isaac/scripts/tune_pid_isaac.py` that prints candidate id, env count, completed episodes, current candidate score, and best gains so far during multi-env runs
- [ ] T017 [US2] Add multi-env Isaac smoke tests in `simulation/tests/test_pid_isaac.py` that verify completed envs do not corrupt unfinished env state and that vectorized summaries are consistent with per-env done counts

**Checkpoint**: Multi-env PID tuning works on the existing vectorized Isaac env and produces consistent candidate summaries.

---

## Phase 5: User Story 3 — Best-Gain Validation with Episode Trace Logging (Priority: P1)

**Goal**: Evaluate the best saved PID gains in Isaac Sim and emit per-step logs for action, state, reward, and PID internals.

**Independent Test**: Run `python -m simulation.isaac.scripts.test_pid_isaac --config simulation/isaac/configs/isaac_env_single.yaml --pid-config runs/pid_isaac/best_pid.yaml --episodes 5 --log-dir runs/pid_isaac_eval` and verify step traces plus episode summaries are written.

### Implementation for User Story 3

- [ ] T018 [US3] Implement the evaluation loop in `simulation/isaac/scripts/test_pid_isaac.py`: load a saved PID YAML, run single-env or multi-env episodes, and report success rate, mean total reward, mean CEP, and failure counts
- [ ] T019 [US3] Implement per-step trace logging in `simulation/isaac/scripts/test_pid_isaac.py` with at least `episode_id`, `env_id`, `step`, `time_s`, observation groups, action vector, step reward, cumulative reward, controller errors, desired roll/pitch, yaw damping term, pre-clipped loop outputs, and final fin commands
- [ ] T020 [US3] Implement `episode_summary.csv`, `run_metadata.json`, and `trace_epXXXX_envYY.csv` artifact writing in `simulation/isaac/scripts/test_pid_isaac.py`, recording active force toggles and config paths in metadata
- [ ] T021 [US3] Add log-schema assertions in `simulation/tests/test_pid_isaac.py` to verify required trace columns and episode-summary fields are present after a short Isaac evaluation run

**Checkpoint**: Best-gain evaluation produces machine-readable step traces and episode summaries rich enough to debug sign, saturation, and reward issues.

---

## Phase 6: User Story 4 — Reward Trend Monitoring & Force Ablation Controls (Priority: P2)

**Goal**: Long tuning runs warn when the monitored metric is not moving in the requested direction, and all force ablations are visible in run metadata.

**Independent Test**: Run `python -m simulation.isaac.scripts.tune_pid_isaac --config simulation/isaac/configs/isaac_env_single.yaml --pid-config simulation/configs/pid.yaml --monitor-window 10 --monitor-direction decreasing --disable-wind --disable-gyro` and verify warnings appear when the rolling metric stalls.

### Implementation for User Story 4

- [ ] T022 [US4] Implement rolling-metric monitoring in `simulation/isaac/scripts/tune_pid_isaac.py` with `--monitor-window` and `--monitor-direction {decreasing,increasing}`; print a warning when the monitored window violates the configured direction
- [ ] T023 [US4] Record active runtime force toggles in both `simulation/isaac/scripts/tune_pid_isaac.py` and `simulation/isaac/scripts/test_pid_isaac.py` metadata and summary outputs so ablation runs are unambiguous after the fact
- [ ] T024 [US4] Extend `simulation/tests/test_pid_isaac.py` with toggle-and-monitor smoke coverage: verify requested force toggles are reflected in run metadata and that the monitor code can emit a warning on a synthetic non-improving metric sequence

**Checkpoint**: Long runs can self-report stalled progress, and ablation runs are fully auditable from their artifacts.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, command discoverability, and targeted validation.

- [ ] T025 Update `simulation/isaac/issac_commands.txt` with `tune_pid_isaac` and `test_pid_isaac` command examples for single-env and multi-env runs
- [ ] T026 [P] Update `CLAUDE.md` commands section with the new Isaac PID tuning / evaluation commands
- [ ] T027 Run targeted tests: `pytest simulation/tests/test_pid_controller.py`, plus Isaac-marked PID tests, and one smoke command for each new script

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies; can start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1; **BLOCKS all user stories**
- **US1 (Phase 3)**: Depends on Phase 2
- **US2 (Phase 4)**: Depends on US1 evaluator work in Phase 3
- **US3 (Phase 5)**: Depends on Phase 2; may share evaluator logic with US1 but should be completed after Phase 3 stabilizes
- **US4 (Phase 6)**: Depends on Phases 3 and 5 because it monitors the tuning flow and records ablation metadata in artifacts
- **Phase 7 (Polish)**: Depends on all prior phases

### User Story Dependencies

- **User Story 1 (P1)**: First deliverable; provides the evaluator and best-PID artifact generation used by later stories
- **User Story 2 (P1)**: Builds directly on US1's evaluator and candidate-scoring code
- **User Story 3 (P1)**: Reuses the same controller telemetry and runtime-toggle plumbing from Phase 2, plus the saved PID artifacts from US1
- **User Story 4 (P2)**: Extends US1/US3 behavior with warnings and auditable metadata

### Within Each User Story

- In Phase 2, controller telemetry (T004) must land before telemetry tests (T005)
- In Phase 2, task/runtime toggles (T006) must land before env wrapper plumbing (T007) and integration smoke tests (T008)
- In US1, rollout evaluation (T010) should land before tuning / ranking outputs (T011-T012)
- In US3, trace logging (T019) should land before log-schema assertions (T021)

### Parallel Opportunities

- **Phase 1**: T001 ∥ T002 ∥ T003
- **Phase 2**: T005 can run after T004 while T006 proceeds independently; T008 follows T006+T007
- **US1**: T009 ∥ early scaffolding for T013; T011 follows T010
- **US3**: T019 ∥ T020 once T018 defines the evaluation loop
- **Polish**: T025 ∥ T026

---

## Parallel Example: Foundational Phase

```text
# After script scaffolds exist:
Task T004: "Refactor PIDController to expose debug telemetry"
Task T006: "Add runtime force toggles to edf_landing_task.py"

# After T004 completes:
Task T005: "Add telemetry parity / sign-convention tests"

# After T006 completes:
Task T007: "Plumb overrides through EDFIsaacEnv"

# After T006 + T007 complete:
Task T008: "Add Isaac smoke tests for runtime toggles"
```

---

## Implementation Strategy

### MVP First (Foundational + US1)

1. Complete Phase 1 setup (T001-T003)
2. Complete Phase 2 foundational telemetry / toggle work (T004-T008)
3. Complete US1 single-env tuning (T009-T013)
4. **STOP and VALIDATE**: run the single-env tuning command and inspect the saved best-PID artifact

### Incremental Delivery

1. **MVP** → Foundational + US1: single-env Isaac PID tuning works
2. **+US2** → multi-env candidate evaluation and ranking
3. **+US3** → best-gain evaluation with detailed trace logging
4. **+US4** → reward warnings and auditable ablation metadata
5. **+Polish** → docs and targeted validation

---

## Notes

- `tasks.md` is used here, not `task.md`, to match the existing repo convention in `specs/001-*` and `specs/002-*`
- `PIDController` must keep its current frame and sign conventions: inertial `NED`, body `FRD`, altitude via `h_agl`, and yaw damping via `[-d,+d,+d,-d]`
- The tuning flow must treat `thrust_rate_limit` and yaw low-pass filtering as part of the current controller unless an explicit ablation flag is added later
- Multi-env candidate evaluation should reuse the existing vectorized `EDFIsaacEnv` rather than spinning up one Python env per candidate
