# Tasks: Isaac Sim Mass Properties, Thrust Test, Environmental Forces & Gyro Precession

**Input**: Design documents from `specs/main/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Test tasks included — constitution v1.1.0 mandates Isaac Sim validation tests (thrust liftoff, fin articulation, environmental force, precession response).

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
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

## Phase 5: User Story 4 — Gyro Precession Modeling (Priority: P2, toggleable)

**Goal**: Gyroscopic precession torque from spinning rotor injected into Isaac Sim physics pipeline, matching custom sim fidelity. Invisible rotor prim added to USDC for mass validation. Config toggle (on by default).

**Independent Test**:
- `python -m simulation.isaac.scripts.diag_gyro_precession --yaw-torque 0.5 --duration 2.0` → pitch rate develops proportional to `I_fan · ω_fan · yaw_rate / I_pitch`; exits 0.
- `python -m simulation.isaac.scripts.diag_gyro_precession --disable-precession` → no pitch response from yaw torque; exits 0.
- `python -m simulation.isaac.scripts.validate_mass_props` → rotor prim section reports PASS.

### Implementation for User Story 4

- [x] T030 [US4] Add `gyro_precession.enabled: true` config toggle to `simulation/configs/default_vehicle.yaml` under the `edf:` section (below `I_fan` line)
- [x] T031 [P] [US4] Add `ROTOR_PRIM = "/Drone/Body/Rotor"` path constant to `simulation/isaac/usd/parts_registry.py` alongside existing `BODY_PRIM` constant
- [ ] T032 [US4] Add invisible Rotor cylinder prim `/Drone/Body/Rotor` to `simulation/isaac/usd/drone.usdc` in Isaac Sim v5.1.0: create cylinder (radius=0.040 m, height=0.04 m), set `visibility = invisible`, apply `UsdPhysics.MassAPI` with `mass=0.0` (mass already in Body composite), position at `(0, 0, -0.06)` in Z-up world frame (= FRD `[0, 0, 0.06]` via `frd_to_zup()`) — **MANUAL SETUP in USD scene editor**
- [x] T033 [US4] Add `_rotate_body_to_world()` module-level helper function to `simulation/isaac/tasks/edf_landing_task.py` (near the existing `_rotate_world_to_body()` function): takes `quat_w (N,4)` scalar-last and `v_body (N,3)`, returns `v_world (N,3)` using formula `v + 2*qw*(q_vec × v) + 2*(q_vec × (q_vec × v))` — body-to-world is the un-conjugated form of the existing world-to-body helper
- [x] T034 [US4] Add `_I_FAN` module-level constant to `simulation/isaac/tasks/edf_landing_task.py` (value `3.0e-5`, loaded pattern matching existing `_K_THRUST = 4.55e-7`) and load `gyro_precession.enabled` from vehicle YAML in `EdfLandingTask.__init__()`, store as `self._gyro_enabled: bool` (default `True` if key missing)
- [x] T035 [US4] Implement gyro precession torque in `EdfLandingTask._apply_action()` in `simulation/isaac/tasks/edf_landing_task.py`: after fin force loop and before `set_external_force_and_torque()` call, add guarded block: if `self._gyro_enabled`: compute `omega_fan = (thrust_actual / _K_THRUST).clamp(min=0).sqrt()`, build `h_fan_b[:, 2] = _I_FAN * omega_fan`, compute `tau_gyro_b = -torch.linalg.cross(omega_b, h_fan_b)` where `omega_b = self.robot.data.root_ang_vel_b`, rotate to world via `_rotate_body_to_world(self.robot.data.root_quat_w, tau_gyro_b)`, add to `torques[:, 0, :]`
- [x] T036 [US4] Extend `simulation/isaac/scripts/validate_mass_props.py` with `read_usd_rotor_prim()` function: open USDC, find `/Drone/Body/Rotor` prim, read `UsdGeom.Cylinder` radius and height extents, read position from `xformOp:translate`; add `validate_rotor_prim()` that compares against YAML `edf_motor_rotor` primitive dimensions within tolerance and appends rotor section to `MassPropertyReport` output (per CLI contract extended format)
- [x] T037 [P] [US4] Create `simulation/isaac/configs/isaac_env_gyro_test.yaml` as a copy of `isaac_env_single.yaml` with `sim.gravity: [0.0, 0.0, 0.0]` override for zero-g precession diagnostic
- [x] T038 [US4] Create `simulation/isaac/scripts/diag_gyro_precession.py` with CLI per contract: `--config` (default `isaac_env_gyro_test.yaml`), `--torque-axis` (`pitch` or `roll`, default `pitch`), `--torque-mag` (default 0.5 N·m), `--duration` (default 2.0 s), `--spawn-alt` (default 5.0 m), `--no-gravity` (flag, default true), `--disable-precession` (flag); spawns drone at altitude with hover thrust, applies constant external **pitch** torque (body Y-axis) via `set_external_force_and_torque` — pitch rate `q` → expected roll precession `τ_x = −q·L`; NOTE: yaw torque is NOT used (yaw rate is parallel to spin axis, zero cross product); logs `[time, pitch_rate, roll_rate]` every 30 steps, asserts `roll_rate > 0.1 °/s` at t>0.5 s when precession enabled, asserts `roll_rate < 0.05 °/s` when disabled; prints ratio vs analytical prediction `I_fan·ω_fan·q/I_roll`
- [x] T039 [P] [US4] Create `simulation/tests/test_gyro_precession.py` with pure-Python unit tests (no Isaac Sim): (1) verify `tau_gyro = -cross([p,q,r], [0,0,I_fan*omega_fan])` matches expected values for known omega_b and omega_fan; (2) zero thrust → zero omega_fan → zero tau_gyro; (3) `_rotate_body_to_world` is inverse of `_rotate_world_to_body` for identity and 90° rotations; (4) `gyro_precession.enabled: false` loads correctly and skips torque block

**Checkpoint**: Gyro precession torque active in Isaac Sim. Yaw input → measurable pitch response. Rotor prim in USDC validates geometry. All existing tests still pass.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Integration verification and documentation

- [x] T027 Update `CLAUDE.md` commands section with new diagnostic commands: `validate_mass_props`, `diag_thrust_test`, `diag_wind`
- [x] T040 Update `CLAUDE.md` commands section with `diag_gyro_precession` command and `--disable-precession` flag
- [x] T028 Run existing Feature 001 diagnostics (`diag_isaac_single`, `diag_fin_wiggle`, `test_fins`) to confirm no regression from wind integration changes
- [ ] T041 Run `diag_gyro_precession --disable-precession` and `diag_gyro_precession` back-to-back to confirm A/B comparison output; verify no regression on Feature 001 diagnostics — **requires Isaac Sim runtime**
- [x] T029 Run full pytest suite (`pytest` at repo root) to verify all tests pass including new `test_mass_validation.py` and updated `test_isaac_env.py`
- [x] T042 Run full pytest suite (`pytest` at repo root) after US4 implementation: `test_gyro_precession.py` — 16/16 passed; pre-existing PID test failure confirmed unrelated to US4

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **User Story 1 (Phase 2)**: Depends on T002, T003 (coordinate + inertia helpers)
- **User Story 2 (Phase 3)**: No dependency on US1; only needs Feature 001 baseline env
- **User Story 3 (Phase 4)**: No dependency on US1 or US2; needs T001 (wind directory) from Setup
- **User Story 4 (Phase 5)**: No dependency on US1, US2, US3; modifies `edf_landing_task.py` — execute after US3 validation complete to avoid mid-integration conflicts
- **Polish (Phase 6)**: Depends on all user stories complete

### User Story Dependencies

- **US1 (Mass Validation, P1)**: Independent — headless script, no Isaac Sim launch
- **US2 (Thrust Test, P1)**: Independent — uses existing env, no code modifications
- **US3 (Wind Forces, P2)**: Independent in testing, but modifies `edf_landing_task.py` shared with US2's env
- **US4 (Gyro Precession, P2)**: Independent in testing; modifies `edf_landing_task.py` (different section from US3) and `validate_mass_props.py` (extension only). Execute after US3 to avoid file conflicts.

### Within Each User Story

- Entity/dataclass definitions before logic
- Core functions before CLI entry points
- Config changes before task integration
- USD scene changes before validation script extension
- Integration before diagnostics
- Tests after implementation (validation-style, not TDD)

### Parallel Opportunities

- T002, T003 can run in parallel (different functions in same file, but independent)
- T031, T033, T037, T039 in US4 are all parallel (different files: parts_registry.py, drone.usdc, yaml config, test file)
- T030, T031, T037, T039 can start in parallel (all different files with no inter-dependency)
- T033, T034 require T030 complete (config value needed) — sequential within US4
- T035 requires T033 and T034 complete — sequential
- US1 and US2 can proceed in parallel after Setup (different files entirely)
- T027, T028, T029 in Polish are independent and can run in parallel

---

## Parallel Example: US4 Setup Tasks

```bash
# All four can start simultaneously:
Task T030: "Add gyro_precession.enabled to default_vehicle.yaml"
Task T031: "Add ROTOR_PRIM constant to parts_registry.py"
Task T037: "Create isaac_env_gyro_test.yaml with gravity=0"
Task T039: "Create test_gyro_precession.py unit tests"
```

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
5. US4 → Gyro precession → physics fidelity matches custom sim; rotor prim in USDC
6. Polish → all diagnostics documented, regression confirmed clean

### Single Developer Strategy (US4 focus)

1. T030, T031, T037, T039 in parallel — config + constants + yaml + tests (~30 min)
2. T032 — USD scene edit in Isaac Sim v5.1.0 (~20 min)
3. T033 — add `_rotate_body_to_world()` helper (~10 min)
4. T034 — load config + store `self._gyro_enabled` (~10 min)
5. T035 — implement torque in `_apply_action()` (~20 min)
6. T036 — extend `validate_mass_props.py` for rotor prim (~30 min)
7. T038 — create `diag_gyro_precession.py` diagnostic (~45 min)
8. T040-T042 — polish + validation (~20 min)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story
- US1 requires only `pxr` (no GPU/Isaac Sim) — can develop on any machine
- US2, US3, US4 require Isaac Sim v5.1.0 runtime — develop on GPU workstation
- Wind integration (US3) is backward compatible: `isaac_wind.enabled: false` preserves Feature 001 behavior
- Gyro precession (US4) is backward compatible: `gyro_precession.enabled: false` disables torque injection
- `_I_FAN = 3.0e-5` (rotating fan blades only, ~60g at r≈35mm); motor stator mass already in Body composite inertia
- Rotor prim mass=0 — mass is already included in Body's 3.13 kg total; prim exists purely for geometry validation
- All existing Isaac Sim configs default precession to `enabled: true` (on by default per constitution)
