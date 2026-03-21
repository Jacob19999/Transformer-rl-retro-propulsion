# Tasks: Refactor Drone Fin Physics Layer

**Input**: Design documents from `/specs/006-refactor-fin-physics/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, quickstart.md

**Tests**: Not explicitly requested in the feature specification. Test tasks are omitted. Existing pytest suite and diagnostic scripts serve as validation (per constitution III).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: Verify existing config and conventions are ready for the refactoring

- [x] T001 Verify `simulation/configs/default_vehicle.yaml` has complete `mass_properties` section with `use_explicit: true`, `total_mass`, `center_of_mass` (FRD), and full 3×3 `inertia_tensor`
- [x] T002 [P] Document the verified IsaacLab joint unit convention (degrees for targets and readback) as a comment block in `simulation/isaac/conventions.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: USD postprocess changes that ALL user stories depend on — hinge axis fix, mass properties authoring, and collider simplification are authored together in a single USD regeneration pass

**CRITICAL**: No user story work in `edf_landing_task.py` can begin until this phase is complete, because the USD asset must be regenerated first.

- [x] T003 Add inertia tensor eigendecomposition utility to `simulation/isaac/usd/postprocess_usd.py` — implement a function that takes a 3×3 symmetric inertia matrix, computes eigenvalues (diagonal inertia) and eigenvectors (principal axes quaternion), and returns `(Gf.Vec3f, Gf.Quatf)` suitable for USD MassAPI. Include FRD→Z-up coordinate conversion for the CoM vector.
- [x] T004 Modify `_add_root_physics()` in `simulation/isaac/usd/postprocess_usd.py` to author explicit CoM and inertia from YAML `mass_properties` section when `use_explicit: true`. Read `center_of_mass` (convert FRD→Z-up), decompose `inertia_tensor` via T003 utility, author via `GetCenterOfMassAttr()`, `GetDiagonalInertiaAttr()`, and `GetPrincipalAxesAttr()`. Remove the `_clear_authored_inertia()` call. Keep `_body_local_bbox_center_z()` as fallback when `use_explicit` is false.
- [x] T005 [P] Modify `_add_collision_apis()` in `simulation/isaac/usd/postprocess_usd.py` to use `"convexHull"` approximation for fin mesh prims instead of `"convexDecomposition"`. Body mesh prims keep `"convexDecomposition"` unchanged.
- [x] T006 [P] Fix fin hinge axes in `_create_fin_joints()` in `simulation/isaac/usd/postprocess_usd.py` — apply 180° localRot flip to both joint local frames for all fins to negate effective hinge direction (FRD↔Z-up convention mismatch). This eliminates the need for runtime sign correction. Documented with code comment referencing RQ-2.
- [ ] T007 Regenerate the USD asset by running `postprocess_usd.py` with the changes from T003–T006 and verify the output USD contains: (a) explicit mass/CoM/inertia on `/Drone/Body`, (b) corrected hinge axes on all four fin joints, (c) simplified fin colliders. Run `validate_mass_props` to confirm YAML↔USD agreement within tolerance.

**Checkpoint**: USD asset is regenerated with correct conventions. All downstream task code changes can now begin.

---

## Phase 3: User Story 1 — Consistent Fin Behaviour (Priority: P1) — MVP

**Goal**: Fin aerodynamic forces are computed from measured PhysX joint state (single source of truth), not from a separate analytic variable.

**Independent Test**: Command a single fin to +15 deg, read back joint position, confirm aero force is computed from that measured value. Run `diag_fin_wiggle` to verify end-to-end consistency.

### Implementation for User Story 1

- [x] T008 [US1] Replace the auto-detection heuristic in `_read_fin_joint_pos_rad()` in `simulation/isaac/tasks/edf_landing_task.py` with unconditional `torch.deg2rad()` conversion. Remove the `if max_abs > 3.5` branch entirely. Add a code comment citing research.md RQ-1 decision.
- [x] T009 [US1] Verify that `fin_deflections_actual` in `simulation/isaac/tasks/edf_landing_task.py` is populated exclusively from `_read_fin_joint_pos_rad()` (measured joint state) and that `compute_fin_forces_body()` in `simulation/isaac/fin_aero.py` receives this measured value. Remove any code path that computes aero forces from commanded state. Confirm the existing line `self.fin_deflections_actual.copy_(self._read_fin_joint_pos_rad())` is the sole write site.
- [x] T010 [US1] Verify that `simulation/isaac/fin_aero.py` (`compute_fin_forces_body()`) does not independently source fin deflection from anywhere other than its `delta_rad` argument. No changes expected — this is a verification and documentation task.

**Checkpoint**: Aero forces now always use measured joint state. Run `diag_fin_wiggle` and `pytest -m isaac` to confirm no regressions.

---

## Phase 4: User Story 2 — Unified Sign Convention (Priority: P1)

**Goal**: Positive fin command → positive joint deflection → correct aero force direction, with zero runtime sign/index remapping.

**Independent Test**: Command each fin individually to +0.26 rad. Confirm joint rotates in expected positive direction, aero force matches expected lift direction, visual mesh is consistent. `FinMapping` is identity.

**Depends on**: Phase 2 (USD hinge axes fixed), Phase 3 (measured joint state verified)

### Implementation for User Story 2

- [x] T011 [US2] Update `default_fin_mapping()` in `simulation/isaac/fin_mapping.py` to return identity mapping: `joint_source_indices=(0, 1, 2, 3)` and `joint_signs=(1.0, 1.0, 1.0, 1.0)`.
- [x] T012 [P] [US2] Update `simulation/configs/fin_mapping.yaml` to set `joint_source_indices: [0, 1, 2, 3]` and `joint_signs: [1.0, 1.0, 1.0, 1.0]`. Update `pitch_weights`, `roll_weights`, `yaw_weights` if the index reordering changes which fin contributes to which axis.
- [x] T013 [P] [US2] Remove `FIN_JOINT_VISUAL_SIGN` from `simulation/isaac/conventions.py`. Add a comment documenting that visual and physical sign conventions are now aligned at USD authoring time (ref: research.md RQ-2). Remove any import of `FIN_JOINT_VISUAL_SIGN` in other files.
- [x] T014 [US2] Simplify the fin command path in `_apply_action()` in `simulation/isaac/tasks/edf_landing_task.py` — with identity mapping, the `index_select` and sign multiply operations become no-ops. Either remove them or guard with an assertion that mapping is identity. Keep the rad→deg conversion for `set_joint_position_target()`.
- [ ] T015 [US2] Verify joint name resolution order in `_ensure_fin_joint_ids()` in `simulation/isaac/tasks/edf_landing_task.py`. Confirm that the ordered lookup `["RightFin_Joint", "LeftFin_Joint", "FwdFin_Joint", "AftFin_Joint"]` returns indices `[0, 1, 2, 3]` matching the canonical controller order. If not, adjust the lookup names or the USD joint naming in `postprocess_usd.py`.
- [ ] T016 [US2] Remove or update `simulation/isaac/scripts/calibrate_fin_mapping.py` — if the mapping is now always identity and enforced by USD conventions, the calibration script should either be removed or updated to serve as a verification-only tool that confirms identity mapping holds.

**Checkpoint**: `FinMapping` is identity. Run `diag_fin_wiggle` — each fin should deflect in the correct direction without sign hacks. Run `pytest -m isaac` to confirm tests pass.

---

## Phase 5: User Story 3 — Aerodynamic Forces at Fin Links (Priority: P2)

**Goal**: Per-fin aero forces are applied at the correct fin link body, and the resulting moments match expected values within 5%.

**Independent Test**: Command one fin to large deflection, others to zero. Inspect external force array — only that fin's link has non-zero force. Check moment direction.

**Depends on**: Phase 4 (sign convention correct — force directions must be right)

### Implementation for User Story 3

- [ ] T017 [US3] Verify the force application code in `_apply_action()` / `_compute_forces()` in `simulation/isaac/tasks/edf_landing_task.py` (lines ~796–812) correctly applies per-fin forces after the sign convention fix. Confirm `fin_forces_world` vectors point in physically correct directions for each fin given a positive deflection. No code change expected if the convention fix (Phase 2) propagated correctly through the lift/drag direction vectors.
- [ ] T018 [US3] Verify that `_fin_lift` and `_fin_drag` direction tensors in `simulation/isaac/tasks/edf_landing_task.py` are loaded from YAML `fins.fins_config[i].lift_direction` and that these directions are consistent with the new hinge axis convention. If the hinge axis was negated in Phase 2, check whether the lift direction vector also needs updating in `default_vehicle.yaml`.
- [ ] T019 [US3] Add a moment-arm validation assertion (debug-mode only) in `simulation/isaac/tasks/edf_landing_task.py` that computes `cross(fin_position, fin_force)` for each fin and compares the resulting torque direction against the expected pitch/roll/yaw axis. Log a warning if discrepancy exceeds 5%. This can be a one-time validation at init or a periodic check controlled by a debug flag.

**Checkpoint**: Per-fin forces produce correct moments. Symmetric deflections yield pure pitch or pure roll as expected.

---

## Phase 6: User Story 4 — Explicit Mass Properties (Priority: P2)

**Goal**: USD asset has authored mass, CoM, and inertia from YAML hardware values. `validate_mass_props` passes within tolerance.

**Independent Test**: Run `python -m simulation.isaac.scripts.validate_mass_props` — should PASS with mass ≤1%, CoM ≤1mm, inertia ≤5%.

**Depends on**: Phase 2 (mass properties authored in USD)

### Implementation for User Story 4

- [x] T020 [US4] Update `simulation/isaac/scripts/validate_mass_props.py` to validate the full inertia tensor (not just mass and CoM). Ensure the script reads `DiagonalInertiaAttr` and `PrincipalAxesAttr` from USD, reconstructs the 3×3 tensor, and compares element-wise against YAML `inertia_tensor` within 5% tolerance. The `reconstruct_inertia_tensor()` function in `parts_registry.py` already handles this — confirm it's being called.
- [x] T021 [US4] Remove or update the warning message in `simulation/isaac/tasks/edf_landing_task.py` (lines ~323–327) that says "ignoring YAML fin positions and explicit CoM/inertia. Using Isaac Sim asset data instead." Now that the asset data IS the YAML data (authored by postprocess_usd), this warning is misleading. Replace with an info log confirming mass properties match, or remove entirely.
- [ ] T022 [US4] Verify runtime mass property reading in `edf_landing_task.py` — confirm `self._mass`, `self._body_com_default_frd`, and `self._body_inertia_default` now reflect the authored YAML values (not collider-derived). Run `validate_mass_props` to confirm end-to-end.

**Checkpoint**: `validate_mass_props` passes. Runtime reads correct values.

---

## Phase 7: User Story 6 — Verified Runtime Units (Priority: P2)

**Goal**: Single verified unit convention for joint targets (deg) and readback (deg→rad), no auto-detection heuristic.

**Independent Test**: Command fin to 0.10 rad (~5.7 deg), read back, confirm aero receives correct radians without heuristic.

**Depends on**: Phase 3 (T008 already implements the unconditional conversion)

### Implementation for User Story 6

- [x] T023 [US6] Confirm T008 (Phase 3) fully addresses this story. The unconditional `deg2rad` on read and existing `rad2deg` on write should satisfy FR-006. Verify the write path in `_apply_action()` still has `fin_target_deg = fin_target_rad * (180.0 / math.pi)` before `set_joint_position_target()`.
- [x] T024 [US6] Add a unit convention documentation block to `simulation/isaac/conventions.py` stating: "IsaacLab joint API uses degrees for `set_joint_position_target()` and `robot.data.joint_pos`. Internal task code uses radians. Conversion: unconditional `rad2deg` on write, unconditional `deg2rad` on read. No heuristic." Reference research.md RQ-1.

**Checkpoint**: No auto-detection heuristic in codebase. Convention documented.

---

## Phase 8: User Story 5 — Simplified Fin Colliders (Priority: P3)

**Goal**: Fin collision geometry uses box (or convex hull) instead of convex decomposition. No PhysX solver warnings.

**Independent Test**: Run `diag_fin_wiggle --episodes 100` — zero warnings or solver failures.

**Depends on**: Phase 2 (T005 already implements the collider change in USD)

### Implementation for User Story 5

- [ ] T025 [US5] Confirm T005 (Phase 2) fully addresses the USD-side change. Verify the regenerated USD has simplified fin colliders by inspecting the USD stage or running a validation check.
- [ ] T026 [US5] Run `diag_fin_wiggle --episodes 100` and confirm zero PhysX solver warnings. If warnings appear, adjust collider geometry (e.g., switch from convex hull to box, or add collision filtering between fins and body).

**Checkpoint**: `diag_fin_wiggle` runs clean. No collider-related solver issues.

---

## Phase 9: User Story 7 — Per-Fin Debug Telemetry (Priority: P3)

**Goal**: Opt-in telemetry logs per-fin data (commanded angle, measured angle, link pose, flow velocity, AoA, force, wrench) per step.

**Independent Test**: Enable telemetry, run single episode, confirm all fields present for all 4 fins at each step.

**Depends on**: Phase 3 (measured joint state), Phase 4 (correct conventions)

### Implementation for User Story 7

- [x] T027 [P] [US7] Create `simulation/isaac/fin_telemetry.py` — implement `FinTelemetryBuffer` class with: (a) `__init__(num_envs, num_fins=4, max_steps, device)` allocating GPU tensor buffers per data-model.md schema, (b) `record(step, cmd_angle, meas_angle, link_pos, link_quat, exhaust_vel, aoa, aero_force, joint_wrench)` writing to ring buffer, (c) `flush(path)` saving buffer to `.pt` file, (d) `reset()` clearing buffer. All tensors on GPU; no CPU transfer during recording.
- [x] T028 [P] [US7] Add `debug.fin_telemetry` (bool, default false) and `debug.fin_telemetry_save` (bool, default false) config keys to `simulation/isaac/configs/isaac_env_base.yaml`.
- [x] T029 [US7] Integrate `FinTelemetryBuffer` into `simulation/isaac/tasks/edf_landing_task.py` — conditionally instantiate at init if `debug.fin_telemetry` is true. In the physics step, after computing aero forces, call `record()` with all required fields. Read `body_incoming_joint_wrench_b` from `self.robot.data` for the joint wrench field. On episode reset, call `flush()` if `debug.fin_telemetry_save` is true. Ensure zero overhead when telemetry is disabled (no buffer allocation, no conditional branches in hot path — use a no-op sentinel or skip the call entirely).
- [ ] T030 [US7] Verify telemetry correctness — enable telemetry, run `diag_fin_wiggle` for 1 episode with a known single-fin deflection. Load the saved `.pt` file and confirm: (a) all 8 fields are present with correct shapes, (b) `meas_angle` matches `cmd_angle` within settling tolerance, (c) `aoa` is consistent with `meas_angle`, (d) `aero_force` direction matches expected lift direction.

**Checkpoint**: Telemetry produces complete, consistent per-fin records. Disabled by default with zero overhead.

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Final validation across all stories, compatibility check, cleanup

- [ ] T031 Run full pytest suite (`pytest`) and confirm all tests pass. Fix any test failures caused by the refactoring with documented justification for each change.
- [ ] T032 Run the full constitution-mandated validation sequence from quickstart.md: (1) `validate_mass_props`, (2) `diag_fin_wiggle --episodes 100`, (3) `diag_thrust_test --thrust 1.0 --duration 2.0 --spawn-alt 0.4`, (4) `diag_wind --wind-x 5.0 --duration 3.0`, (5) `pytest -m isaac`.
- [ ] T033 [P] Verify Gymnasium API compatibility (FR-009) — confirm `observation_space`, `action_space`, `reset()`, and `step()` signatures and shapes in `simulation/isaac/envs/edf_isaac_env.py` are unchanged.
- [x] T034 [P] Remove dead code — delete any unused imports of `FIN_JOINT_VISUAL_SIGN`, unused `FinMapping` remap logic, stale comments referencing the old sign hack, and the old `_clear_authored_inertia()` function if no longer called.
- [x] T035 Update `simulation/tests/test_isaac_env.py` if any assertions reference the old sign convention or the old auto-detection heuristic. Ensure test expectations match the new identity mapping and unconditional unit conversion.

**Checkpoint**: All tests green. All diagnostics pass. Codebase is clean.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — BLOCKS all user stories
- **Phase 3 (US1 — Single source of truth)**: Depends on Phase 2
- **Phase 4 (US2 — Sign convention)**: Depends on Phase 2 + Phase 3
- **Phase 5 (US3 — Forces at fin links)**: Depends on Phase 4
- **Phase 6 (US4 — Mass properties)**: Depends on Phase 2 only (parallel with Phase 3/4)
- **Phase 7 (US6 — Verified units)**: Depends on Phase 3 (T008 is the core change)
- **Phase 8 (US5 — Simplified colliders)**: Depends on Phase 2 only (parallel with Phase 3/4)
- **Phase 9 (US7 — Telemetry)**: Depends on Phase 3 + Phase 4
- **Phase 10 (Polish)**: Depends on all prior phases

### User Story Dependencies

```
Phase 1 (Setup)
    │
    ▼
Phase 2 (Foundational: USD changes) ──────────────────────────┐
    │                                                          │
    ├──▶ Phase 3 (US1: Single source of truth)                 │
    │        │                                                 │
    │        ├──▶ Phase 4 (US2: Sign convention)               │
    │        │        │                                        │
    │        │        └──▶ Phase 5 (US3: Forces at links)      │
    │        │                                                 │
    │        ├──▶ Phase 7 (US6: Verified units)                │
    │        │                                                 │
    │        └──▶ Phase 9 (US7: Telemetry)                     │
    │                                                          │
    ├──▶ Phase 6 (US4: Mass properties) ◄──────────────────────┘
    │
    └──▶ Phase 8 (US5: Simplified colliders)

    All ──▶ Phase 10 (Polish)
```

### Parallel Opportunities

- **T001 + T002**: Setup tasks can run in parallel
- **T005 + T006**: Collider simplification and hinge axis fix are in different functions — can run in parallel
- **T012 + T013**: `fin_mapping.yaml` and `conventions.py` are independent files — can run in parallel
- **Phase 6 (US4) + Phase 3 (US1)**: Mass properties and single-source-of-truth are independent work streams after Phase 2
- **Phase 8 (US5) + Phase 3 (US1)**: Collider verification and task code changes are independent after Phase 2
- **T027 + T028**: Telemetry module and config are independent files — can run in parallel
- **T033 + T034**: API verification and dead code removal are independent

---

## Parallel Example: After Phase 2 Completion

```
Stream A (critical path):      Stream B (independent):
  Phase 3 (US1)                  Phase 6 (US4: mass props)
    ↓                            Phase 8 (US5: colliders)
  Phase 4 (US2)
    ↓
  Phase 5 (US3)
  Phase 7 (US6)
  Phase 9 (US7)
    ↓
  Phase 10 (Polish)
```

---

## Implementation Strategy

### MVP First (User Stories 1 + 2 Only)

1. Complete Phase 1: Setup — verify YAML config, document conventions
2. Complete Phase 2: Foundational — regenerate USD with all fixes
3. Complete Phase 3: US1 — single source of truth for joint state
4. Complete Phase 4: US2 — identity sign mapping
5. **STOP and VALIDATE**: Run `diag_fin_wiggle`, `pytest -m isaac`
6. This delivers the core correctness fix (FR-001, FR-003, FR-006, FR-008)

### Incremental Delivery

1. **MVP**: Phases 1–4 → Core fin physics correctness
2. **+US3**: Phase 5 → Verified force application (builds confidence)
3. **+US4**: Phase 6 → Explicit mass properties (sim-to-real readiness)
4. **+US6**: Phase 7 → Documented unit convention (maintenance win)
5. **+US5**: Phase 8 → Simplified colliders (stability/performance)
6. **+US7**: Phase 9 → Debug telemetry (ongoing development support)
7. **Polish**: Phase 10 → Full validation and cleanup

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each phase completion
- Stop at any checkpoint to validate story independently
- The foundational phase (Phase 2) is the largest single phase because USD changes must happen atomically before any runtime code changes
