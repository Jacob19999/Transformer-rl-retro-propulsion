# Tasks: Isaac Sim Mass Properties, Thrust Test & Environmental Forces

**Input**: Design documents from `specs/main/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Test tasks included — constitution v1.1.0 mandates Isaac Sim validation tests (thrust liftoff, fin articulation, environmental force).

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: Project structure and shared utilities for all user stories

- [x] T001 Create `simulation/isaac/wind/` directory with `__init__.py`
- [x] T002 [P] Add `zup_to_frd()` inverse coordinate conversion function to `simulation/isaac/usd/parts_registry.py` (inverse of existing `frd_to_zup()`)
- [x] T003 [P] Add `reconstruct_inertia_tensor()` helper to `simulation/isaac/usd/parts_registry.py` that converts USD diagonal inertia + principal axes quaternion into a full 3x3 inertia tensor in FRD body frame

**Checkpoint**: Shared utilities ready — user story work can begin

---

## Phase 2: User Story 1 — Mass Property Validation (Priority: P1)

**Goal**: Validation script that compares USDC scene mass/CoM/inertia against YAML config within 1% tolerance, runnable without Isaac Sim.

**Independent Test**: `python -m simulation.isaac.scripts.validate_mass_props --usd simulation/isaac/usd/drone.usdc --config simulation/configs/default_vehicle.yaml` exits 0 with all 10 values passing.

### Implementation for User Story 1

- [x] T004 [US1] Create `MassPropertyReport` and `Discrepancy` dataclasses in `simulation/isaac/scripts/validate_mass_props.py` per data-model.md entity definitions
- [x] T005 [US1] Implement `read_usd_mass_props()` function in `simulation/isaac/scripts/validate_mass_props.py` that opens USDC via `pxr`, reads MassAPI from `/Drone/Body`, and returns mass, CoM (Z-up), diagonal inertia, and principal axes quaternion
- [x] T006 [US1] Implement `compare_mass_properties()` function in `simulation/isaac/scripts/validate_mass_props.py` that loads YAML via `load_explicit_mass_props()` from `parts_registry.py`, converts USD CoM to FRD via `zup_to_frd()`, reconstructs full inertia tensor via `reconstruct_inertia_tensor()`, and compares all 10 scalar values (1 mass + 3 CoM + 6 unique inertia) within tolerance (depends on T004, T005)
- [x] T007 [US1] Implement `format_report()` function in `simulation/isaac/scripts/validate_mass_props.py` that prints human-readable table (field, YAML value, USD value, error %, status) per CLI contract output format
- [x] T008 [US1] Implement CLI `__main__` entry point in `simulation/isaac/scripts/validate_mass_props.py` with argparse: `--usd`, `--config`, `--tolerance`, `--json`, `--quiet` flags; exit 0 on pass, 1 on fail, 2 on error (depends on T006, T007)
- [x] T009 [US1] Create `simulation/tests/test_mass_validation.py` with pytest tests: (1) pass case with matching values, (2) fail case with intentionally wrong mass, (3) tolerance edge case at exactly 1%, (4) missing MassAPI schema raises appropriate error

**Checkpoint**: Mass property validation fully functional and tested. Run: `python -m simulation.isaac.scripts.validate_mass_props` → PASS.

---

## Phase 3: User Story 2 — Thrust Application Test Environment (Priority: P1)

**Goal**: Diagnostic script confirming the drone lifts off from the ground under commanded thrust, validating F=ma consistency with mass properties.

**Independent Test**: `python -m simulation.isaac.scripts.diag_thrust_test --thrust 1.0` → drone ascends past 5 m in 2 seconds.

### Implementation for User Story 2

- [x] T010 [US2] Create `simulation/isaac/scripts/diag_thrust_test.py` with CLI entry point: `--config`, `--thrust`, `--duration`, `--spawn-alt`, `--episodes` arguments per CLI contract
- [x] T011 [US2] Implement ground-start spawn logic in `diag_thrust_test.py`: override env reset to spawn at `--spawn-alt` (default 0.4 m) with zero velocity, use existing `EDFIsaacEnv` wrapper
- [x] T012 [US2] Implement thrust command sequence in `diag_thrust_test.py`: apply constant `--thrust` value for `--duration` seconds, log altitude (obs[16]) and vertical velocity every 30 steps to stdout
- [x] T013 [US2] Implement pass/fail assertions in `diag_thrust_test.py`: (1) full thrust T_cmd=1.0 → altitude > 5 m within 2 s (SC-002), (2) hover thrust T_cmd≈0.68 → altitude change < 0.5 m over 2 s (SC-003), (3) thrust cut → drone descends; exit 0 on pass, 1 on fail
- [x] T014 [US2] Add multi-phase test mode to `diag_thrust_test.py`: single episode runs three phases sequentially (full thrust 2 s → hover 2 s → cut 1 s), logging altitude profile and computed acceleration vs expected ((T/m - g) for full thrust phase)

**Checkpoint**: Thrust application validated. Drone lifts off, hovers, and falls back. F=ma consistency confirmed.

---

## Phase 4: User Story 3 — Environmental Force Application (Priority: P2)

**Goal**: Configurable wind disturbance forces in Isaac Sim with drag-based force model, YAML-driven parameters, and observation vector integration.

**Independent Test**: `python -m simulation.isaac.scripts.diag_wind --wind-x 5.0` → drone drifts laterally in +X; obs[13:16] shows non-zero wind.

### Implementation for User Story 3

- [x] T015 [US3] Create `IsaacWindModel` class in `simulation/isaac/wind/isaac_wind_model.py` with `__init__(config, num_envs, device)` that loads wind parameters from environment YAML `wind` section; initialize GPU tensors for wind state (mean_wind, gust state, wind_ema) per data-model.md `IsaacWindState`
- [x] T016 [US3] Implement `IsaacWindModel.reset(env_ids)` in `simulation/isaac/wind/isaac_wind_model.py`: sample mean wind vector from `[mean_vector_range_lo, mean_vector_range_hi]` uniform per env_id; sample gust event per `gust_prob`; set gust onset, duration, magnitude; zero wind_ema for reset envs
- [x] T017 [US3] Implement `IsaacWindModel.step(dt)` in `simulation/isaac/wind/isaac_wind_model.py`: compute current wind vector (mean + gust if active), update wind_ema with exponential filter (tau=0.5 s), return wind_vector_world tensor (num_envs, 3)
- [x] T018 [US3] Implement `IsaacWindModel.compute_drag_force(wind_vector, body_velocity)` in `simulation/isaac/wind/isaac_wind_model.py`: compute relative wind `v_rel = wind - body_vel` (world frame), compute drag `F = 0.5 * rho * |v_rel|^2 * Cd * A * unit(v_rel)` using composite drag coefficient and projected areas from vehicle YAML
- [x] T019 [US3] Add `isaac_wind` section to `simulation/configs/default_environment.yaml` with `enabled: false`, `drag_coefficient: 1.0`, `projected_area: [0.01, 0.01, 0.02]` defaults; add Isaac-specific overrides for dt (1/120 s instead of 0.005 s)
- [x] T020 [US3] Integrate `IsaacWindModel` into `EdfLandingTask.__init__()` in `simulation/isaac/tasks/edf_landing_task.py`: load environment YAML, instantiate `IsaacWindModel` if `isaac_wind.enabled: true`, store as `self._wind_model` (None if disabled)
- [x] T021 [US3] Integrate wind force into `EdfLandingTask._apply_action()` in `simulation/isaac/tasks/edf_landing_task.py`: if `self._wind_model` is not None, call `step(dt)` to get wind vectors, call `compute_drag_force()` with body velocity from `robot.data.root_lin_vel_w`, add resulting force to `forces` tensor before `set_external_force_and_torque()`
- [x] T022 [US3] Integrate wind into `EdfLandingTask._reset_idx()` in `simulation/isaac/tasks/edf_landing_task.py`: call `self._wind_model.reset(env_ids)` for wind state re-sampling on episode reset
- [x] T023 [US3] Update `EdfLandingTask._get_observations()` in `simulation/isaac/tasks/edf_landing_task.py`: replace hardcoded `self._wind_ema` zeros (line ~273) with `self._wind_model.wind_ema` when wind model is active; keep zeros when wind is disabled for backward compatibility
- [x] T024 [US3] Update `EDFIsaacEnv.__init__()` in `simulation/isaac/envs/edf_isaac_env.py` to load `default_environment.yaml` path from Isaac env YAML config and pass it through to `EdfLandingTaskCfg` so the task can instantiate `IsaacWindModel`
- [x] T025 [US3] Create `simulation/isaac/scripts/diag_wind.py` with CLI entry point per contract: `--config`, `--wind-x`, `--wind-y`, `--wind-z`, `--duration`, `--episodes`; override wind config to use constant specified wind vector with `isaac_wind.enabled: true`; log lateral position and velocity every 30 steps; assert lateral velocity > 0.1 m/s within 1 second (SC-004)
- [x] T026 [US3] Add wind force tests to `simulation/tests/test_isaac_env.py`: (1) wind enabled with 5 m/s → lateral velocity > 0.1 m/s within 1 s, (2) wind disabled → lateral velocity < 0.01 m/s, (3) obs[13:16] non-zero when wind active, (4) gust event produces transient wind spike

**Checkpoint**: Wind disturbances fully integrated. Drone drifts under wind, observations reflect wind state, all existing tests still pass.

---

## Phase 5: Polish & Cross-Cutting Concerns

**Purpose**: Integration verification and documentation

- [x] T027 Update `CLAUDE.md` commands section with new diagnostic commands: `validate_mass_props`, `diag_thrust_test`, `diag_wind`
- [x] T028 Run existing Feature 001 diagnostics (`diag_isaac_single`, `diag_fin_wiggle`, `test_fins`) to confirm no regression from wind integration changes
- [x] T029 Run full pytest suite (`pytest` at repo root) to verify all tests pass including new `test_mass_validation.py` and updated `test_isaac_env.py`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **User Story 1 (Phase 2)**: Depends on T002, T003 (coordinate + inertia helpers)
- **User Story 2 (Phase 3)**: No dependency on US1; only needs Feature 001 baseline env
- **User Story 3 (Phase 4)**: No dependency on US1 or US2; needs T001 (wind directory) from Setup
- **Polish (Phase 5)**: Depends on all user stories complete

### User Story Dependencies

- **US1 (Mass Validation, P1)**: Independent — headless script, no Isaac Sim launch
- **US2 (Thrust Test, P1)**: Independent — uses existing env, no code modifications
- **US3 (Wind Forces, P2)**: Independent in testing, but modifies `edf_landing_task.py` shared with US2's env. Execute after US2 validation to avoid mid-test code changes.

### Within Each User Story

- Entity/dataclass definitions before logic
- Core functions before CLI entry points
- Integration before diagnostics
- Tests after implementation (validation-style, not TDD)

### Parallel Opportunities

- T002, T003 can run in parallel (different functions in same file, but independent)
- US1 and US2 can proceed in parallel after Setup (different files entirely)
- T015, T016, T017, T018 are sequential within US3 (same class, building up)
- T027, T028, T029 in Polish are independent and can run in parallel

---

## Parallel Example: Setup Phase

```bash
# Launch both utility tasks together (different functions):
Task T002: "Add zup_to_frd() to parts_registry.py"
Task T003: "Add reconstruct_inertia_tensor() to parts_registry.py"
```

## Parallel Example: US1 + US2

```bash
# After Setup complete, both P1 stories can start simultaneously:
# Developer A: US1 (mass validation — pure pxr/YAML, no sim)
Task T004-T009: Mass Property Validation pipeline

# Developer B: US2 (thrust test — Isaac Sim diagnostic)
Task T010-T014: Thrust Application Test pipeline
```

---

## Implementation Strategy

### MVP First (User Story 1 + User Story 2)

1. Complete Phase 1: Setup (T001-T003)
2. Complete Phase 2: US1 — Mass Property Validation (T004-T009)
3. Complete Phase 3: US2 — Thrust Application Test (T010-T014)
4. **STOP and VALIDATE**: Both P1 stories independently pass
5. Confirm drone is physically correct (mass) and flyable (thrust)

### Incremental Delivery

1. Setup → shared utilities ready
2. US1 → Mass validated → `validate_mass_props` passes
3. US2 → Thrust validated → drone lifts off, hovers, falls
4. US3 → Wind integrated → environmental forces working
5. Polish → all diagnostics documented, regression confirmed clean

### Single Developer Strategy

1. Setup (T001-T003) — 1 session
2. US1 (T004-T009) — 1 session (no Isaac Sim needed, fast iteration)
3. US2 (T010-T014) — 1 session (requires Isaac Sim)
4. US3 (T015-T026) — 2 sessions (largest scope, modifies core task)
5. Polish (T027-T029) — 1 session

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story
- US1 requires only `pxr` (no GPU/Isaac Sim) — can develop on any machine
- US2 and US3 require Isaac Sim runtime — develop on GPU workstation
- Wind integration (US3) is backward compatible: `isaac_wind.enabled: false` preserves Feature 001 behavior
- All existing Isaac Sim configs default to wind disabled
