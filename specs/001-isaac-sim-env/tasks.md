# Tasks: Isaac Sim Vectorized Drone Simulation Environment

**Input**: Design documents from `/specs/001-isaac-sim-env/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

**Platform**: Windows 11 Pro · RTX 5070 (12 GB VRAM) · Isaac Sim 5.1.0 / IsaacLab 2.3 / Python 3.11

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Directory structure, config files, and module scaffolding. No user story work until this is complete.

- [X] T001 Create `simulation/isaac/` package tree: `usd/`, `tasks/`, `envs/`, `scripts/`, `configs/` with `__init__.py` files per plan.md structure
- [X] T002 [P] Create `simulation/isaac/configs/isaac_env_single.yaml` — single-env config (`num_envs: 1`, `physics_dt: 0.00833`, `decimation: 1`, `num_physics_substeps: 4`, spawn ranges, vehicle/DR config paths)
- [X] T003 [P] Create `simulation/isaac/configs/isaac_env_128.yaml` — 128-env training config (same as single, `num_envs: 128`)
- [X] T004 [P] Create `simulation/isaac/configs/isaac_env_1024.yaml` — 1024-env training config (`num_envs: 1024`; note VRAM constraint on RTX 5070)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: USD asset builder and config loader bridge — MUST complete before any IsaacLab env work can begin.

**⚠️ CRITICAL**: Phases 3–7 all depend on a valid `drone.usd` being producible from YAML.

- [X] T005 Implement `simulation/isaac/usd/drone_builder.py` — main entry point: loads `default_vehicle.yaml` via existing `config_loader.py`, orchestrates body + fin USD construction, writes `simulation/isaac/usd/drone.usd`
- [X] T006 Implement body geometry emitter inside `drone_builder.py`: iterate YAML `primitives`, emit `UsdGeom.Cylinder` / `UsdGeom.Cube` / `UsdGeom.Sphere` children under `/Drone/Body`; apply `UsdPhysics.CollisionAPI` to each; apply `UsdPhysics.RigidBodyAPI` + `UsdPhysics.MassAPI` (composite mass, CoM, diagonalInertia) to `/Drone` root per `usd-asset-schema.md`
- [X] T007 Implement fin geometry emitter inside `drone_builder.py`: for each of 4 fins in `fins.fins_config`, create `/Drone/Fin_N` Xform + `/Drone/Fin_N/Geom` as `UsdGeom.Cube` scaled to `(0.0325, 0.0275, 0.0039)` half-extents, translated `+0.0325` along local chord axis from hinge; add `UsdPhysics.MassAPI(mass=0.003)` per `usd-asset-schema.md`
- [X] T008 Implement joint + drive emitter inside `drone_builder.py`: create `UsdPhysics.RevoluteJoint` on each `/Drone/Fin_N` with correct `physics:axis`, `lowerLimit=-15`, `upperLimit=+15`, `body0=/Drone`, `localPos0=<hinge_pos>`; add `UsdPhysics.DriveAPI` with `stiffness=25.0`, `damping=0.05`, `targetPosition=0.0` per `usd-asset-schema.md`
- [X] T009 Add coordinate frame transform in `drone_builder.py`: apply 90° rotation about X at `/Drone` root to align FRD body +Z with Isaac Sim Y-up world frame
- [X] T010 [P] Add CLI entrypoint to `drone_builder.py`: `--config` and `--output` args; validate output USD opens in Isaac Sim without errors
- [X] T011 [P] Add pytest test `simulation/tests/test_drone_builder.py::test_usd_round_trip`: load generated USD via `pxr`, verify body mass within 1% of YAML aggregate, verify 4 revolute joints present with ±15° limits, verify fin cube half-extents within 1% of (0.0325, 0.0275, 0.0039)

**Checkpoint**: `drone.usd` generated, round-trip test passes — IsaacLab env work can now begin

---

## Phase 3: User Story 1 — Single-Environment Drone Simulation (Priority: P1) 🎯 MVP

**Goal**: A single drone in a 10×10m arena falls under gravity, lands on the pad, and resets cleanly 1000× without error.

**Independent Test**: Run `python -m simulation.isaac.scripts.diag_isaac_single --config simulation/isaac/configs/isaac_env_single.yaml`; observe drone descend and settle on landing pad within 10 s sim-time. Run `pytest simulation/tests/test_isaac_env.py::test_single_env_reset_stability`.

### Implementation for User Story 1

- [X] T012 [US1] Implement `simulation/isaac/tasks/edf_landing_task.py` — `EdfLandingTaskCfg` dataclass inheriting `DirectRLEnvCfg`: set `observation_space=20`, `action_space=5`, `decimation=1`; nested `EdfSceneCfg(InteractiveSceneCfg)` with `TerrainImporterCfg(terrain_type="plane", prim_path="/World/ground")` and `ArticulationCfg(prim_path="{ENV_REGEX_NS}/Drone", spawn=UsdFileCfg(usd_path="simulation/isaac/usd/drone.usd"))` per `usd-asset-schema.md`
- [X] T013 [US1] Add `SimulationCfg(dt=1/120, num_substeps=4, improve_determinism=True)` to `EdfLandingTaskCfg`; configure landing pad `RigidObjectCfg` with `PhysicsMaterialCfg(static_friction=0.5, dynamic_friction=0.5, restitution=0.1)` per contracts/usd-asset-schema.md
- [X] T014 [US1] Implement `EdfLandingTask(DirectRLEnv)` class in `edf_landing_task.py`: override `_setup_scene()` to instantiate `InteractiveScene(self.cfg.scene)`; reference drone articulation as `self.robot = self.scene["robot"]`
- [X] T015 [US1] Implement `_reset_idx()` in `EdfLandingTask`: sample random altitude [5, 10] m and velocity magnitude [0, 5] m/s uniformly per `data-model.md` spawn ranges; call `self.robot.write_root_pose_to_sim()` and `self.robot.write_root_velocity_to_sim()` for given env indices; reset episode step counter and lag state tensors
- [X] T016 [US1] Implement physics lag state tensors in `EdfLandingTask.__init__()`: `self.thrust_actual` shape `(num_envs,)` for EDF first-order lag; `self.fin_deflections_actual` shape `(num_envs, 4)` for servo lag; both initialized to zero on CUDA device
- [X] T017 [US1] Implement `_pre_physics_step()` in `EdfLandingTask`: (1) unpack 5-dim action tensor → thrust_cmd scalar + 4 fin cmds; (2) update `thrust_actual` via discrete first-order lag: `thrust_actual += (dt/tau_motor) * (T_cmd_phys - thrust_actual)` where `tau_motor=0.10`; (3) update `fin_deflections_actual` via `tau_servo=0.04`; (4) compute EDF force vector `[0, 0, thrust_actual]` in body frame; (5) compute NACA0012 fin forces using `Cl_alpha`, exhaust velocity, and current deflections per `default_vehicle.yaml`; (6) call `self.robot.set_external_force_and_torque(forces, torques, body_ids=[0])` then `self.robot.write_data_to_sim()`
- [X] T018 [US1] Implement `_get_observations()` in `EdfLandingTask`: read `self.robot.data.root_pos_w`, `.root_quat_w`, `.root_lin_vel_b`, `.root_ang_vel_b`; compute 20-dim observation matching layout in `observation.py` (`e_p_body`, `v_body`, `g_body`, `omega`, `twr`, `wind_ema` [zeros], `h_agl`, `speed`, `ang_speed`, `time_frac`); return as `(num_envs, 20)` float32 tensor
- [X] T019 [US1] Implement `_get_rewards()` in `EdfLandingTask`: port existing `RewardFunction` logic from `simulation/training/reward.py` — shaped shaping terms + terminal landing success/crash bonus; read reward weights from existing `reward.yaml`
- [X] T020 [US1] Implement `_get_dones()` in `EdfLandingTask`: `terminated` = crash condition (`h_agl < 0` AND `|v| > threshold`) OR successful landing; `truncated` = step count ≥ `episode_length_steps`
- [X] T021 [US1] Implement `simulation/isaac/envs/edf_isaac_env.py` — `EDFIsaacEnv` Gymnasium wrapper: wraps `EdfLandingTask`; converts PyTorch tensors to numpy; exposes `observation_space = Box(-inf, inf, (20,), float32)` and `action_space = Box(-1, 1, (5,), float32)` per `contracts/gymnasium-env-interface.md`; implements auto-reset on done per SB3 VecEnv contract
- [X] T022 [US1] Implement `simulation/isaac/scripts/diag_isaac_single.py`: launch single env, step with zero actions for 600 steps, print h_agl each 60 steps, assert drone contacts ground within episode; serves as visual inspection entry point per quickstart.md Step 2
- [X] T023 [US1] Add pytest tests in `simulation/tests/test_isaac_env.py`:
  - `test_single_env_reset_stability`: 1000 reset-step-reset cycles headless, assert no exception
  - `test_observation_dimensions`: obs shape `(20,)`, dtype float32, all finite after reset
  - `test_gravity_fall`: zero actions, h_agl decreases monotonically until contact within 600 steps

**Checkpoint**: Single env launches, drone falls under gravity, lands, resets 1000× clean — US1 complete and independently verified

---

## Phase 4: User Story 2 — Drone Asset with Controllable Fins (Priority: P1)

**Goal**: Each fin deflects to ±15° on command and holds at limit when over-commanded; servo lag is observable.

**Independent Test**: Run `python -m simulation.isaac.scripts.test_fins --config simulation/isaac/configs/isaac_env_single.yaml`; confirm each fin hits ±15° and clamps. Run `pytest simulation/tests/test_isaac_env.py::test_fin_deflection_limits`.

### Implementation for User Story 2

- [X] T024 [US2] Implement `simulation/isaac/scripts/test_fins.py`: command each fin sequentially to +1.0, 0.0, -1.0 normalized, read back `robot.data.joint_pos`, print deflection in degrees; assert within 1% of ±15°
- [X] T025 [US2] Add `test_fin_deflection_limits` to `simulation/tests/test_isaac_env.py`: command `action=[0, 1, 0, 0, 0]` (fin 1 to max), step 60 frames, assert `fin_deflections_actual[0,0]` ≤ `0.2618 + ε` (15° in rad); repeat for -1.0 command and other fins
- [X] T026 [US2] Add `test_action_scaling` to `simulation/tests/test_isaac_env.py`: command `action=[1, 0, 0, 0, 0]`, step until lag settles (>5× tau_motor steps), assert `thrust_actual` ≈ `T_max=45.0 N` within 5%
- [X] T027 [US2] Add `test_fin_lag_response` to `simulation/tests/test_isaac_env.py`: step fin command step-change at t=0, verify `fin_deflections_actual` reaches 63% of target within `tau_servo/dt = 0.04*120 ≈ 5` steps (first-order lag characteristic)

**Checkpoint**: Fin articulation verified programmatically and visually — US2 complete

---

## Phase 5: User Story 3 — Parallel Environment Rollouts (Priority: P2)

**Goal**: 128 environments run simultaneously; resetting one does not affect others; throughput ≥10× single env.

**Independent Test**: Run `pytest simulation/tests/test_isaac_env.py::test_parallel_independence`; run throughput benchmark and confirm ≥10× speedup at 128 envs.

### Implementation for User Story 3

- [X] T028 [US3] Verify `EdfLandingTask` launches with `num_envs=128` using `isaac_env_128.yaml`; confirm `(128, 20)` obs tensor and `(128,)` reward tensor returned each step
- [X] T029 [US3] Add `test_parallel_independence` to `simulation/tests/test_isaac_env.py`: launch 128 envs; step 10 frames; force-reset env index 42 via `_reset_idx([42])`; step 10 more frames; assert obs[42] episode_step == 0 while obs[41] episode_step == 20
- [X] T030 [US3] Implement throughput benchmark script `simulation/isaac/scripts/benchmark_envs.py`: run 1000 steps at `num_envs` ∈ [1, 128, 512, 1024], log wall-clock steps/s to CSV; assert 128-env throughput ≥ 10× single-env per SC-004
- [X] T031 [US3] Add VRAM guard to `edf_isaac_env.py`: if `num_envs > 512` and GPU VRAM < 16 GB, emit warning (RTX 5070 = 12 GB); do not block but log advisory

**Checkpoint**: Parallel envs verified independent; throughput benchmark logged — US3 complete

---

## Phase 6: User Story 4 — Configuration-Driven Drone Generation (Priority: P2)

**Goal**: Modifying a YAML parameter regenerates the USD asset with the change reflected within 1% tolerance.

**Independent Test**: Modify `edf_duct.radius` in YAML from 0.045 → 0.050, run `drone_builder.py`, open USD, verify duct cylinder radius = 0.050 m within 1%.

### Implementation for User Story 4

- [X] T032 [US4] Add `test_yaml_to_usd_parameter_propagation` to `simulation/tests/test_drone_builder.py`: override `edf_duct.radius` to 0.060 in a temp YAML copy; regenerate USD; read USD via `pxr`; assert `UsdGeom.Cylinder(duct_prim).GetRadiusAttr().Get() == 0.060` within 1%
- [X] T033 [US4] Add `test_fin_joint_limits_from_yaml` to `simulation/tests/test_drone_builder.py`: set `fins.max_deflection = 0.2618` (15°) in test YAML; regenerate USD; assert all 4 `RevoluteJoint` `lowerLimit == -15.0` and `upperLimit == 15.0` (in degrees)
- [X] T034 [US4] Add `test_composite_mass_from_yaml` to `simulation/tests/test_drone_builder.py`: sum all primitive masses from YAML (expected ≈ 3.1 kg); assert `UsdPhysics.MassAPI(drone_prim).GetMassAttr().Get()` within 1% of computed sum

**Checkpoint**: Full YAML → USD → physics parameter chain verified — US4 complete

---

## Phase 7: Training Integration (Priority: P2, depends on US3)

**Goal**: SB3 PPO training loop connects to Isaac Sim env, logs to TensorBoard under `runs/`, saves checkpoints.

**Independent Test**: Run `python -m simulation.training.scripts.train_isaac_ppo --config simulation/isaac/configs/isaac_env_128.yaml --seed 0` for 10K steps; confirm `runs/` directory created with TensorBoard event file and checkpoint.

### Implementation for Phase 7

- [X] T035 Implement `simulation/training/scripts/train_isaac_ppo.py`: instantiate `EDFIsaacEnv(config)`; wrap with SB3 `VecNormalize`; configure PPO with same hyperparameters as `ppo_mlp.yaml`; add `--seed` arg that sets numpy/torch/Isaac Sim seeds; run training loop
- [X] T036 Add TensorBoard callback to `train_isaac_ppo.py`: log episode reward mean, landing success rate, steps/s; save to `runs/isaac_ppo_<seed>_<commit_hash>_<timestamp>/`
- [X] T037 Add checkpoint saving to `train_isaac_ppo.py`: save PPO model every 500K steps to `runs/.../checkpoints/`; save `VecNormalize` stats alongside each checkpoint per Constitution §IV
- [X] T038 [P] Add `simulation/isaac/configs/isaac_env_training.yaml` — optimized training config: `num_envs: 256` (safe for RTX 5070 12 GB), full DR enabled, all reward weights from `reward.yaml`

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Hardening, documentation, and integration validation across all stories.

- [X] T039 [P] Update `CLAUDE.md` with Isaac Sim training commands: `python -m simulation.training.scripts.train_isaac_ppo`, `python -m simulation.isaac.usd.drone_builder`, `python -m simulation.isaac.scripts.diag_isaac_single`
- [X] T040 [P] Add `simulation/isaac/scripts/diag_yaw_isaac.py`: equivalent of existing `diag_yaw` for Isaac Sim env — step with pure yaw torque input, verify angular response
- [X] T041 Run quickstart.md validation: execute Steps 1–5 in order on clean checkout; update quickstart.md if any command path changed
- [X] T042 [P] Add `pytest.ini` test marker `isaac` to `simulation/tests/test_isaac_env.py` and `test_drone_builder.py`; update `pytest.ini` to skip `isaac`-marked tests if Isaac Sim not available (for CI without GPU)
- [X] T043 Fix research.md Decision 8 stale API reference: replace `set_root_state_tensor()` (IsaacLab 1.x) with `write_root_pose_to_sim()` / `write_root_velocity_to_sim()` (IsaacLab 2.x) throughout research.md

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — BLOCKS all Isaac env work (T005–T011)
- **Phase 3 (US1 — Single Env)**: Depends on Phase 2 (drone.usd exists)
- **Phase 4 (US2 — Fins)**: Depends on Phase 3 (env and lag state implemented)
- **Phase 5 (US3 — Parallel)**: Depends on Phase 3 (single env working)
- **Phase 6 (US4 — Config-driven)**: Depends on Phase 2 only (USD builder); can run in parallel with Phase 3/4/5
- **Phase 7 (Training)**: Depends on Phase 5 (parallel env working)
- **Phase 8 (Polish)**: Depends on all prior phases

### Parallel Opportunities


| Parallelizable group               | Tasks                                                 |
| ---------------------------------- | ----------------------------------------------------- |
| Config file creation               | T002, T003, T004                                      |
| USD builder — body + fins + joints | T006, T007, T008 (after T005)                         |
| USD CLI + round-trip test          | T010, T011 (after T008)                               |
| US1 env internals                  | T016 (lag tensors), T012+T013 (scene cfg) can overlap |
| US2 test additions                 | T024, T025, T026, T027                                |
| US3 parallel tests + benchmark     | T029, T030 (after T028)                               |
| US4 YAML propagation tests         | T032, T033, T034                                      |
| Training config + TensorBoard      | T036, T037, T038 (after T035)                         |
| Polish                             | T039, T040, T042, T043                                |


---

## Implementation Strategy

### MVP (User Stories 1 + 2 only — Phases 1–4)

1. Phase 1: Setup (T001–T004)
2. Phase 2: USD builder (T005–T011) — **stop and run `test_drone_builder.py` before proceeding**
3. Phase 3: Single env (T012–T023) — **stop and run `diag_isaac_single.py` visual check**
4. Phase 4: Fin validation (T024–T027)
5. **STOP and VALIDATE**: PID tuning can begin on the single env MVP

### Full Delivery (all phases)

Add Phase 5 (parallel), Phase 6 (config round-trip), Phase 7 (training), Phase 8 (polish) after MVP is validated.

### VRAM Note (RTX 5070, 12 GB)

- Safe for 128 envs: confirmed
- 512 envs: likely fine, benchmark to verify
- 1024 envs: may exceed 12 GB; use `isaac_env_training.yaml` with `num_envs: 256` as default training config (T038)

---

## Notes

- `[P]` = parallelizable with other `[P]` tasks at same phase (different files, no shared state)
- `[USN]` maps task to spec.md user story for traceability
- All IsaacLab imports use `isaaclab.`* namespace (not `omni.isaac.lab.*`)
- External forces applied in **body local frame** via `set_external_force_and_torque` + `write_data_to_sim()`
- Coordinate frame: FRD body ↔ Y-up Isaac Sim via 90° X-rotation at `/Drone` root
- Commit after each checkpoint; include seed + git hash in run directory names

