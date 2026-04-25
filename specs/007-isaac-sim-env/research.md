# Research: Phase 1 Isaac Sim EDF TVC Simulation Environment

**Branch**: `007-isaac-sim-env` | **Date**: 2026-03-22

## R1: Isaac Lab Environment Architecture Pattern

**Decision**: Use `DirectRLEnv` (not `ManagerBasedRLEnv`) for the TVC environment.

**Rationale**: DirectRLEnv provides fine-grained control over reward/observation/reset logic by implementing them directly in the task class. This is the recommended pattern for custom physics environments where the step logic requires explicit force computation and application. ManagerBasedRLEnv is designed for composable manipulation/locomotion tasks with standard MDP building blocks, which does not fit the custom per-fin force application model.

**Key abstract methods to implement**:

| Method | Purpose |
|--------|---------|
| `_setup_scene(self)` | Create articulation, spawn ground, clone envs, register assets |
| `_pre_physics_step(self, actions)` | Pre-process actions before physics step (called once per RL step) |
| `_apply_action(self)` | Apply actions to simulation (called `decimation` times per RL step) |
| `_get_observations(self)` | Compute and return observation dict |
| `_get_rewards(self)` | Compute and return per-env reward tensor |
| `_get_dones(self)` | Return `(terminated, timed_out)` boolean tensors |

**Config class**: Subclass `DirectRLEnvCfg` with fields: `decimation`, `episode_length_s`, `action_space`, `observation_space`, `state_space`, `sim` (SimulationCfg), `scene` (InteractiveSceneCfg).

**Alternatives considered**:
- ManagerBasedRLEnv: Too modular for custom force-based physics; would require fighting the framework.
- Raw IsaacGymEnvs pattern: Deprecated; Isaac Lab is the successor.

---

## R2: Force/Wrench Application API

**Decision**: Use `Articulation.set_external_force_and_torque()` with the `positions` parameter for per-fin force application at COP.

**Rationale**: This is the current API in Isaac Lab 2.x for applying external wrenches. The `positions` parameter (available since early 2025) allows specifying force application points offset from body CoM, which is exactly what per-fin COP force application requires.

**Method signature**:
```python
def set_external_force_and_torque(
    self,
    forces: torch.Tensor,          # (num_envs, num_bodies, 3)
    torques: torch.Tensor,         # (num_envs, num_bodies, 3)
    positions: torch.Tensor | None,  # (num_envs, num_bodies, 3) — offset from body CoM
    body_ids: Sequence[int] | slice | None,
    env_ids: Sequence[int] | None,
)
```

**Critical usage notes**:
- Forces/torques are in the **bodies' local frame** by default
- This method only fills internal buffers — `write_data_to_sim()` must be called before `sim.step()`
- The `positions` parameter enables force-at-COP without manually computing r × F torques

**Design implication for wrench_dispatch.py**: The dispatch layer wraps this API call, translating the internal standardized force format to the Isaac Lab call. For `per_link_force` mode, forces are applied to each fin link's body index at the COP position. For `collapsed_body_wrench` mode, all forces are summed and applied as a net body wrench.

**Alternatives considered**:
- Direct PhysX API calls via `root_physx_view`: Bypasses Isaac Lab abstraction; not recommended.
- Older `set_external_force_and_torque()` without positions: Would require manual torque computation for off-center forces; more error-prone.

---

## R3: Scene Replication and Vectorization

**Decision**: Use `InteractiveSceneCfg` with `num_envs=128, env_spacing=4.0, replicate_physics=True`.

**Rationale**: `InteractiveSceneCfg` is the standard scene config class. `replicate_physics=True` enables homogeneous cloning optimization where the physics engine parses only `env_0` and replicates it, which is correct since all 128 environments run the same drone configuration.

**Configuration pattern**:
```python
@configclass
class TVCSceneCfg(InteractiveSceneCfg):
    num_envs = 128
    env_spacing = 4.0
    replicate_physics = True
    drone: ArticulationCfg = ArticulationCfg(...)
    ground: GroundPlaneCfg = GroundPlaneCfg(...)
```

**Cloning workflow in `_setup_scene()`**:
1. Define drone articulation and ground plane
2. `self.scene.clone_environments(copy_from_source=False)`
3. `self.scene.filter_collisions(global_prim_paths=[])` — prevents inter-env collisions
4. Register assets: `self.scene.articulations["drone"] = self.drone`

**Alternatives considered**:
- Manual USD Cloner APIs: Unnecessary; InteractiveSceneCfg handles cloning automatically.
- `replicate_physics=False`: Only needed for heterogeneous environments.

---

## R4: USD Articulation and Joint Conventions

**Decision**: Follow OpenUSD PhysX schema: `ArticulationRootAPI` on root Xform, `RigidBodyAPI` + `MassAPI` on each link, `PhysicsRevoluteJoint` for fin hinges.

**Rationale**: Isaac Sim only vectorizes Revolute and Prismatic joints in articulations. The USD physics schema is the standard interchange format for Isaac Sim.

**Joint definition structure**:
- `physics:axis` = "X", "Y", or "Z" (local frame cardinal axis)
- `physics:lowerLimit` / `physics:upperLimit` in **degrees**
- `physics:body0` = parent link, `physics:body1` = child link (fin)
- `physics:localPos0/1` and `physics:localRot0/1` for joint frame placement

**Mass properties via `UsdPhysics.MassAPI`**:
- `physics:mass` in kg (takes precedence over density)
- `physics:centerOfMass` as local-frame offset
- `physics:diagonalInertia` as principal-axis diagonal
- `physics:principalAxes` as quaternion orientation (identity if aligned)

**Design implication for asset_validator.py**: Validator must check that all 4 fin joints are RevoluteJoint type, have defined axes, have valid limits, and that all links have RigidBodyAPI and MassAPI applied.

**Alternatives considered**:
- D6 joints: Not supported as articulation joints in Isaac Sim.
- Explicit joint axis vectors: Not supported; must use localRot to orient cardinal axes.

---

## R5: Quaternion Convention

**Decision**: Use **(w, x, y, z)** quaternion ordering throughout, matching Isaac Lab 2.3.2.

**Rationale**: Isaac Lab 2.x consistently uses wxyz. All tensor outputs from `root_state_w`, `body_state_w`, etc. are in (w,x,y,z). All `isaaclab.utils.math` quaternion functions expect this order. The switch to (x,y,z,w) happened in Isaac Lab 3.0 — not applicable here.

**Interaction with constitution**: The constitution specifies "scalar-last quaternions [qx, qy, qz, qw]" for the custom simulation. The Isaac Lab environment uses scalar-first (w,x,y,z). The frame conversion module (`common/quaternions.py`) must handle this translation at the boundary between the Isaac Lab env and any external consumers that expect the constitution's convention.

**Design implication**: `common/quaternions.py` must provide:
- `isaac_to_body_quat(q_wxyz) → q_xyzw` for external output
- `body_to_isaac_quat(q_xyzw) → q_wxyz` for internal use
- All internal Isaac Lab code uses wxyz exclusively

**Alternatives considered**:
- Force xyzw everywhere: Would fight Isaac Lab's API at every call site.
- Dual convention with implicit conversion: Too error-prone; explicit boundary is safer.

---

## R6: Debug Drawing and Gizmo APIs

**Decision**: Use dual API approach: `isaaclab.markers.VisualizationMarkers` for 3D shape markers (arrows, frames, cones) and `isaacsim.util.debug_draw` for lightweight lines/points.

**Rationale**: VisualizationMarkers provides GPU-instanced rendering of complex shapes (arrows for thrust/force vectors, frame markers for coordinate axes), while debug_draw provides minimal-overhead line rendering for trajectories and contact normals.

**Usage mapping**:

| Gizmo | API | Marker Type |
|-------|-----|-------------|
| Body axes | VisualizationMarkers | FrameMarkerCfg |
| COM marker | VisualizationMarkers | SphereCfg |
| Target/pad marker | VisualizationMarkers | CylinderCfg |
| Thrust vector | VisualizationMarkers | Arrow USD |
| Per-fin force arrows | VisualizationMarkers | Arrow USD |
| Total aero force | VisualizationMarkers | Arrow USD |
| Reaction torque arrow | VisualizationMarkers | Arrow USD |
| Contact normals | debug_draw | draw_lines |
| Hover tolerance volume | VisualizationMarkers | WireframeCfg |

**Performance consideration**: All gizmo rendering is disabled when `num_envs > 1` (training mode). Gizmos are only active in single-env debug mode, controlled by `debug/gizmos.yaml`.

**Alternatives considered**:
- Raw USD geometry prims: Heavy; not suitable for per-frame updates.
- Omniverse viewport overlays: 2D only; insufficient for 3D force visualization.

---

## R7: Fin Aerodynamic Model Formulation

**Decision**: Semi-empirical jet-vane model using thin-airfoil/flat-plate starting approximation with subsonic corrections.

**Rationale**: The Technical Plan explicitly specifies a semi-empirical approach rather than free-stream wing models or supersonic rocket-vane coefficients. The model decomposes forces into normal (control-producing) and tangential (thrust-loss) components.

**Initial model formulation**:
- Normal force: `F_n = q * S * C_N(α)` where `C_N(α) = C_N_α * α * (1 - k_sat * α²)` for finite-angle saturation
- Tangential force: `F_t = q * S * C_D(α)` where `C_D(α) = C_D0 + C_D_α² * α²`
- Dynamic pressure: `q = 0.5 * ρ * V_exhaust²`
- `α` = effective angle of attack = fin deflection angle relative to exhaust flow
- Correction terms: aspect ratio factor, duct confinement factor, empirical calibration multiplier

**Known simplifications** (Phase 1):
- No vane-to-vane interference
- Uniform duct velocity profile assumed
- No duct wall interaction at large deflection
- No separated-flow modeling

**Design implication**: All coefficient parameters stored in YAML with source labeling (estimate/calibrated). The model is stateless (pure function of inputs) for easy testing and vectorization.

**Alternatives considered**:
- XFOIL/panel method: Too heavyweight for real-time sim; Phase 2+ consideration.
- Direct CFD coupling: Out of scope; Phase 1 uses semi-empirical.
- Supersonic jet-vane coefficients: Physically inappropriate for subsonic EDF flow.

---

## R8: Contact State Machine Design

**Decision**: Four-state machine (AIRBORNE → GROUND_CONTACT_CANDIDATE → LANDED | CRASHED) with dwell-interval thresholds.

**Rationale**: The Technical Plan explicitly requires a state machine rather than one-frame checks to prevent false landing declarations after bounces.

**State transitions**:
- AIRBORNE → GROUND_CONTACT_CANDIDATE: Contact detected
- GROUND_CONTACT_CANDIDATE → LANDED: All dwell criteria met for required interval (vertical speed, lateral speed, tilt, angular rate all below thresholds)
- GROUND_CONTACT_CANDIDATE → AIRBORNE: Contact lost before dwell completes (bounce)
- GROUND_CONTACT_CANDIDATE → CRASHED: Any crash criterion triggered (impact speed, excessive tilt, excessive angular rate, unsafe body contact)
- AIRBORNE → CRASHED: Immediate crash criteria (impact speed exceeds threshold)

**Implementation**: Implemented as a vectorized state tensor `[num_envs]` using integer state encoding. Dwell counters track consecutive frames meeting landed criteria. All thresholds configurable via `tasks/hover.yaml` and `tasks/landing.yaml`.

**Alternatives considered**:
- Single-frame contact check: Explicitly rejected by Technical Plan due to bounce misclassification.
- More granular states (BOUNCING, SLIDING): Over-engineering for Phase 1.

---

## R9: Reward Composition Architecture

**Decision**: Registry-based composable rewards with term-name → function mapping and YAML-driven weight selection per task.

**Rationale**: The Technical Plan mandates a single reward framework with task-configurable weights, avoiding separate code paths per algorithm or task.

**Architecture**:
1. `reward_registry.py` maps string names to reward functions: `{"alive_bonus": alive_bonus_fn, "pos_error": pos_error_fn, ...}`
2. Each reward function signature: `fn(env_state, config) → torch.Tensor` (per-env scalar)
3. Task YAML selects terms and weights: `{term_name: weight, ...}`
4. `rewards.py` iterates the active terms, calls each function, multiplies by weight, sums

**Term inventory**:
- Shared: alive_bonus, position_error, attitude_error, angular_velocity, control_effort, control_rate, saturation, crash_penalty
- Hover-only: hover_stability, drift_penalty, contact_penalty
- Landing-only: touchdown_softness, landing_success, pad_accuracy, vertical_speed_shaping

**Alternatives considered**:
- Hard-coded reward functions per task: Violates DRY; constitution warns against reward duplication.
- Class-based reward terms: Over-engineering for a registry of pure functions.

---

## R10: Force Dispatch Architecture

**Decision**: Dual-mode dispatch behind `wrench_dispatch.py` abstraction, with `per_link_force` as default and `collapsed_body_wrench` as optional fallback.

**Rationale**: Phase 1 correctness must be proven in `per_link_force` mode (forces applied at fin COPs on fin links, body torque emerges from articulation physics). The collapsed mode sums all forces into a single body wrench for potential throughput gains during large-scale training.

**Implementation**:
- `link_force_interface.py`: Translates internal force representations to Isaac Lab `set_external_force_and_torque()` calls with `body_ids` targeting individual fin links and `positions` set to COP offsets
- `wrench_dispatch.py`: Selects dispatch mode from config, delegates to `link_force_interface.py` or collapses forces to body wrench
- Mode selection via `env/single_env_debug.yaml` (`dispatch_mode: per_link_force`) or `env/train_128.yaml` (configurable)

**Alternatives considered**:
- Only per_link_force: Simplest, but may limit training throughput.
- Only collapsed_body_wrench: Loses articulation-reaction fidelity in validation.
