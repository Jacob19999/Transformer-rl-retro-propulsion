# Tasks: EDF Fan Reaction Torque — Steady-State Anti-Torque & RPM-Ramp Yaw Coupling

**Input**: Design documents from `specs/002-fan-reaction-torque/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/cli-contracts.md, quickstart.md

**Tests**: Unit tests included (FR-010 in spec explicitly requires them).

**Organization**: Tasks grouped by user story. US1 and US2 are both P1 but US2 builds on US1's Isaac Sim infrastructure (shared config loading, shared diagnostic scaffold).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Config Extensions)

**Purpose**: Extend YAML configs with anti-torque toggle. `k_torque: 1.0e-8` already exists in both configs — only the new `anti_torque.enabled` key is added.

- [x] T001 Add `anti_torque: { enabled: true }` section under `edf` in `simulation/configs/default_vehicle.yaml`
- [x] T002 [P] Add `anti_torque: { enabled: true }` section under `edf` in `simulation/configs/test_vehicle.yaml`

**Checkpoint**: Both YAML configs parseable with new key. Existing tests still pass (`pytest`).

---

## Phase 2: Foundational (Custom Sim ThrustModel — Blocking)

**Purpose**: Activate `k_torque` in the custom Python simulation's `ThrustModel`. This is the reference implementation that Isaac Sim must match. RPM-ramp torque (`motor_reaction_torque()`) already works — only steady-state anti-torque is new here.

**CRITICAL**: No Isaac Sim user story work can begin until this phase is complete.

- [x] T003 Extend `ThrustModelConfig` dataclass with `k_torque: float` and `anti_torque_enabled: bool` fields, and load both from config dict in `from_edf_config()` in `simulation/dynamics/thrust_model.py`
- [x] T004 Add `steady_state_anti_torque(self, *, T: float) -> np.ndarray` method to `ThrustModel` that returns `[0, 0, -k_torque * omega_fan²]` using `omega_from_thrust(T)` in `simulation/dynamics/thrust_model.py`
- [x] T005 Include steady-state anti-torque in `ThrustModel.outputs()` return, gated on `self.config.anti_torque_enabled`, adding to the existing `tau_offset + tau_reaction` sum in `simulation/dynamics/thrust_model.py`
- [x] T006 [P] Create unit tests for anti-torque and ramp-torque math in `simulation/tests/test_reaction_torque.py`: (1) `test_anti_torque_at_hover` — T=30.5N → τ_z ≈ −0.670 N·m, (2) `test_anti_torque_at_zero` — T=0 → τ=[0,0,0], (3) `test_anti_torque_proportional` — 3 thrust levels, verify τ ratio matches ω² ratio, (4) `test_anti_torque_disabled` — enabled=false → τ=[0,0,0], (5) `test_ramp_torque_sign` — T_dot>0 → τ_z<0, (6) `test_ramp_torque_at_constant_thrust` — T_dot=0 → τ_ramp=[0,0,0], (7) `test_combined_torque` — both terms non-zero simultaneously

**Checkpoint**: `pytest simulation/tests/test_reaction_torque.py` passes. Existing tests still pass. Custom sim now applies steady-state anti-torque.

---

## Phase 3: User Story 1 — Steady-State Anti-Torque Validation (Priority: P1) MVP

**Goal**: Drone yaws monotonically under constant thrust in zero-gravity Isaac Sim env, validating that `τ_anti = -k_torque × ω²` is applied correctly through the force/torque pipeline.

**Independent Test**: Run `python -m simulation.isaac.scripts.diag_reaction_torque --mode constant --thrust 0.68 --duration 3.0` and confirm yaw rate matches `k_torque × ω² / I_zz` within 10%.

### Implementation for User Story 1

- [x] T007 [US1] Load `anti_torque` config (enabled flag + `k_torque`) at `__init__()` alongside existing gyro config loading (~line 290) in `simulation/isaac/tasks/edf_landing_task.py`
- [x] T008 [US1] Add steady-state anti-torque computation block in `_apply_action()` after gyro precession block (~line 552): compute `ω_fan = sqrt(T_actual / k_thrust)`, `τ_anti_b = [0, 0, -k_torque × ω²]`, rotate body→world, accumulate into torques tensor, gated on `_anti_torque_enabled` in `simulation/isaac/tasks/edf_landing_task.py`
- [x] T009 [US1] Create `simulation/isaac/scripts/diag_reaction_torque.py` scaffold: argparse CLI with `--mode {constant,ramp,liftoff}`, `--thrust`, `--ramp-duration`, `--duration`, `--config`, `--vehicle-config`, `--disable-anti-torque`, `--output` arguments per contracts/cli-contracts.md
- [x] T010 [US1] Implement `--mode constant` test logic: spawn drone in zero-gravity, apply constant thrust for `--duration`, log per-step CSV (step, time, altitude, yaw_deg, yaw_rate_dps, thrust_N, tau_anti_Nm, tau_ramp_Nm) in `simulation/isaac/scripts/diag_reaction_torque.py`
- [x] T011 [US1] Add constant-mode pass/fail: compute expected yaw rate from `k_torque × ω² / I_zz`, compare measured yaw rate at t>1s, PASS if within 10%, print result and exit with code 0/1 in `simulation/isaac/scripts/diag_reaction_torque.py`

**Checkpoint**: `diag_reaction_torque --mode constant` reports PASS. Drone yaws monotonically at predicted rate. SC-001 and SC-003 validated.

---

## Phase 4: User Story 2 — RPM-Ramp Transient Torque Validation (Priority: P1)

**Goal**: During a thrust ramp, transient yaw **acceleration / torque** exceeds the anti-torque-only value at matched thrust, confirming `τ_ramp = -I_fan × dω/dt` is applied in Isaac Sim alongside the steady-state anti-torque.

**Independent Test**: Run `python -m simulation.isaac.scripts.diag_reaction_torque --mode ramp --ramp-duration 1.0 --duration 3.0` and confirm end-of-ramp yaw acceleration exceeds the anti-torque-only yaw acceleration at the same thrust by >10%.

### Implementation for User Story 2

- [x] T012 [US2] Add RPM-ramp reaction torque computation block in `_apply_action()` after anti-torque block: compute `T_dot = (T_cmd_clipped - T_actual) / tau_motor`, `dω/dt = T_dot / (2 × k_thrust × ω_safe)`, `τ_ramp_b = [0, 0, -I_fan × dω/dt]`, rotate body→world, accumulate into torques tensor, gated on `_anti_torque_enabled` in `simulation/isaac/tasks/edf_landing_task.py`
- [x] T013 [US2] Implement `--mode ramp` test logic: ramp thrust from 0 to 100% over `--ramp-duration`, then hold constant for remaining `--duration`, track yaw rate time-series for peak detection in `simulation/isaac/scripts/diag_reaction_torque.py`
- [x] T014 [US2] Add ramp-mode pass/fail: compare total yaw acceleration at ramp end against the anti-torque-only yaw acceleration at matched thrust, PASS if total > 110%, print comparison and exit with code 0/1 in `simulation/isaac/scripts/diag_reaction_torque.py`

**Checkpoint**: `diag_reaction_torque --mode ramp` reports PASS. End-of-ramp yaw acceleration exceeds the anti-torque-only reference by >10%. SC-002 validated.

---

## Phase 5: User Story 3 — End-to-End Liftoff Validation (Priority: P2)

**Goal**: Drone lifts off from ground under full thrust AND rotates in yaw, confirming anti-torque is active in the full physics pipeline (gravity + ground contact + thrust + reaction torque).

**Independent Test**: Run `python -m simulation.isaac.scripts.diag_reaction_torque --mode liftoff --duration 2.0` and confirm altitude > 5m AND yaw > 5°.

### Implementation for User Story 3

- [x] T015 [US3] Implement `--mode liftoff` test logic: use `isaac_env_single.yaml` (normal gravity), spawn on ground at ~0.4m, apply full thrust for `--duration`, log altitude + yaw alongside existing CSV fields in `simulation/isaac/scripts/diag_reaction_torque.py`
- [x] T016 [US3] Add liftoff-mode pass/fail: PASS if altitude > 5m AND yaw > 5° after `--duration` seconds; when `--disable-anti-torque` is set (already in T009 argparse), PASS if yaw < 0.5° (baseline confirmation) in `simulation/isaac/scripts/diag_reaction_torque.py`

**Checkpoint**: `diag_reaction_torque --mode liftoff` reports PASS with both altitude and yaw criteria. `diag_reaction_torque --mode liftoff --disable-anti-torque` shows < 0.5° yaw. SC-004, SC-005 validated. `diag_thrust_test.py` is NOT modified — it remains a pure thrust/altitude diagnostic.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, regression check, and full validation sequence.

- [x] T018 Update `CLAUDE.md` commands section with `diag_reaction_torque` invocations (all 3 modes + disable flag) following existing diagnostic command documentation pattern
- [ ] T019 Run full test suite (`pytest`) to verify no regressions in existing tests
- [ ] T020 Run quickstart.md validation sequence end-to-end: unit test → constant mode → ramp mode → liftoff mode → A/B comparison

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 (YAML config must have new keys) — **BLOCKS all user stories**
- **US1 (Phase 3)**: Depends on Phase 2 (ThrustModel must have anti-torque method for reference)
- **US2 (Phase 4)**: Depends on US1 Phase 3 tasks T007-T009 (shared Isaac Sim config loading + diagnostic scaffold)
- **US3 (Phase 5)**: Depends on US1 + US2 complete (both torque effects must be in Isaac Sim)
- **Polish (Phase 6)**: Depends on all user stories complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Phase 2 — no dependencies on other stories
- **User Story 2 (P1)**: Depends on US1's T007 (config loading), T008 (insertion point established), T009 (diagnostic scaffold) — builds on same files
- **User Story 3 (P2)**: Depends on both US1 and US2 being in Isaac Sim — validates the combined effect

### Within Each User Story

- Config loading (T007) before torque computation (T008)
- Diagnostic scaffold (T009) before mode implementation (T010)
- Mode implementation before pass/fail criteria

### Parallel Opportunities

- **Phase 1**: T001 ∥ T002 (different YAML files)
- **Phase 2**: T006 ∥ T003→T004→T005 (test file vs. thrust_model.py)
- **Phase 3**: T007 ∥ T009 (different files: edf_landing_task.py vs. diag_reaction_torque.py)
- **Phase 5**: T015 → T016 (sequential: liftoff logic then pass/fail, both in diag_reaction_torque.py)

---

## Parallel Example: User Story 1

```text
# After Phase 2 checkpoint, launch in parallel:
Task T007: "Load anti_torque config at __init__() in edf_landing_task.py"
Task T009: "Create diag_reaction_torque.py scaffold with argparse CLI"

# After T007 completes:
Task T008: "Add steady-state anti-torque in _apply_action()"

# After T009 completes:
Task T010: "Implement --mode constant test logic"

# After T008 + T010 complete:
Task T011: "Add constant-mode pass/fail criteria"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Config extensions (T001-T002)
2. Complete Phase 2: Custom sim ThrustModel (T003-T006)
3. Complete Phase 3: US1 — steady-state anti-torque in Isaac Sim + constant diagnostic (T007-T011)
4. **STOP and VALIDATE**: Run `diag_reaction_torque --mode constant` — confirm PASS
5. This alone delivers the most critical physics correction (steady-state anti-torque present in every flight)

### Incremental Delivery

1. **MVP** → Phase 1+2+3: Steady-state anti-torque works end-to-end (SC-001, SC-003, SC-006)
2. **+US2** → Phase 4: RPM-ramp torque also in Isaac Sim (SC-002)
3. **+US3** → Phase 5: Full pipeline liftoff validation (SC-004, SC-005)
4. **+Polish** → Phase 6: Docs, regression, full quickstart (SC-007)

---

## Notes

- `k_torque: 1.0e-8` already exists in YAML — do NOT add a duplicate. Only add `anti_torque.enabled`.
- `I_fan: 3.0e-5` is already loaded for gyro precession — reuse the same `self._I_fan` field in Isaac Sim.
- RPM-ramp torque already works in custom sim (`motor_reaction_torque()`) — do NOT reimplement. Only port the pattern to Isaac Sim tensors.
- Sign convention: anti-torque is `−k_torque × ω²` on body Z (negative = opposes fan spin). Same sign as existing `−I_fan × dω/dt`.
- Research notes `k_torque` may be ~2× the power-derived estimate. Use config value; flag for future hardware calibration.
- The diagnostic reuses `isaac_env_gyro_test.yaml` (zero gravity) for constant/ramp modes and `isaac_env_single.yaml` for liftoff mode.
- `diag_thrust_test.py` is NOT modified by this feature — it remains a pure thrust/altitude diagnostic with no yaw torque effects.
- Isaac Sim implementation note: keep the live-asset path already in `edf_landing_task.py` — body CoM/inertia and fin anchors come from the loaded USD/PhysX data, not the legacy YAML geometry fields.
