# Research: Refactor Drone Fin Physics Layer

**Feature Branch**: `006-refactor-fin-physics`
**Date**: 2026-03-21

## RQ-1: IsaacLab Joint Position Units

**Decision**: IsaacLab `set_joint_position_target()` expects **degrees**; `robot.data.joint_pos` returns **degrees** (confirmed by USD authoring in degrees and the existing rad→deg conversion). The auto-detection heuristic (threshold at 3.5) will be replaced with an explicit, unconditional `deg2rad` conversion on the read path.

**Rationale**: USD joint limits are authored in degrees (`postprocess_usd.py` line 418: `joint_limit_deg = math.degrees(max_deflection_rad)`). The write path already converts rad→deg before `set_joint_position_target()` (line 777). The read path's heuristic is fragile — small deflections (< 3.5 deg) would be misclassified as radians. Since all USD joints are authored in degrees and Isaac Sim's PhysX joint API consistently uses the USD-authored unit, the safest approach is to always convert degrees→radians on read and radians→degrees on write.

**Alternatives considered**:
- Author USD joints in radians: Rejected — USD/PhysX convention is degrees for angular limits and drives.
- Keep the heuristic: Rejected — misclassifies deflections < 3.5 deg; fails silently.
- Use `joint_pos_target` instead of `joint_pos` for aero: Rejected — spec requires measured (actual) state, not target.

---

## RQ-2: Fin Sign/Index Remapping Root Cause

**Decision**: Fix the USD hinge axes at authoring time (in `postprocess_usd.py`) so that the joint rotation direction matches the controller's positive-deflection convention. Then set `FinMapping` to identity.

**Rationale**: The current remapping exists because:
1. **Index swap (1,0,3,2)**: The USD prim order (as resolved by `find_joints`) does not match the canonical controller order `[Right, Left, Fwd, Aft]`. The swap bridges this gap.
2. **Sign negation (-1,-1,-1,-1)**: All four hinge axes produce rotation in the opposite direction from what the controller expects as "positive deflection."

The fix requires two changes:
- **Ensure joint name resolution order matches canonical order**: The `_ensure_fin_joint_ids()` already attempts ordered lookup by `["RightFin_Joint", "LeftFin_Joint", "FwdFin_Joint", "AftFin_Joint"]`. If IsaacLab resolves these correctly, the index mapping is already identity. The swap may be compensating for a historical joint naming or resolution order issue.
- **Flip hinge axes in USD**: For each fin where positive joint rotation produces the wrong deflection direction, negate the hinge axis vector in the USD (e.g., change `[0, 1, 0]` to `[0, -1, 0]`). This is done in `postprocess_usd.py` by conditionally negating the axis based on the desired positive-deflection convention.

**Alternatives considered**:
- Keep the runtime remap permanently: Rejected — spec FR-003/FR-008 require elimination. The remap is a maintenance hazard and prevents direct introspection of joint state.
- Change fin prim names in Blender: Rejected — names already match physical positions; renaming would confuse CAD workflow.
- Change controller canonical order: Rejected — breaks backward compatibility with existing training runs and Python sim.

---

## RQ-3: Mass Properties Authoring Pipeline

**Decision**: Modify `postprocess_usd.py` to author explicit CoM and inertia from YAML `mass_properties` section (when `use_explicit: true`) instead of clearing inertia and using collider-derived values.

**Rationale**: The YAML already contains an `explicit mass_properties` section with `total_mass: 3.13`, `center_of_mass`, and full `inertia_tensor` (derived from primitives or CAD/measured values). The validation script (`validate_mass_props.py`) is already structured to compare these values against the USD. The only missing piece is authoring them into the USD.

Current flow:
```
YAML mass_properties → (ignored by postprocess_usd.py) → USD has mass only, CoM from bbox, inertia cleared
                                                        → Isaac Sim computes inertia from colliders
                                                        → Runtime reads collider-derived values
```

Target flow:
```
YAML mass_properties → postprocess_usd.py authors mass + CoM + inertia → USD has explicit values
                                                                        → Isaac Sim uses authored values
                                                                        → Runtime reads authored values
                                                                        → validate_mass_props confirms match
```

Changes needed in `_add_root_physics()`:
1. Accept the YAML config (not just `body_mass`)
2. Read `center_of_mass` from YAML, convert FRD→Z-up, author via `GetCenterOfMassAttr()`
3. Decompose `inertia_tensor` into diagonal + principal axes quaternion
4. Author via `GetDiagonalInertiaAttr()` and `GetPrincipalAxesAttr()`
5. Remove `_clear_authored_inertia()` call

For fin links: Keep current approach (negligible mass 1e-5 kg, zeroed inertia) — fins are aerodynamic surfaces, not structural mass contributors.

**Alternatives considered**:
- Keep collider-derived inertia: Rejected — spec FR-004 requires explicit authoring; collider geometry does not match real drone mass distribution.
- Author only diagonal inertia (assume aligned axes): Rejected — the YAML inertia tensor may have off-diagonal terms; full decomposition needed.
- Increase fin mass to realistic values: Deferred — physical fin mass (~5-10g) is negligible relative to 3.13 kg body; current 1e-5 kg prevents link mass ratio instability.

---

## RQ-4: Simplified Fin Colliders

**Decision**: Replace convex decomposition with **box colliders** for fin links. Keep convex decomposition for the body mesh only.

**Rationale**: NVIDIA recommends simple colliders for small articulated parts. Fins are thin, flat surfaces where convex decomposition can produce degenerate shapes. Box colliders:
- Are the simplest stable shape for thin rectangular fins
- Have deterministic collision behavior
- Are faster to compute
- Eliminate degenerate hull edge cases

The box dimensions will be derived from each fin mesh's bounding box (already computable from the USD stage).

**Alternatives considered**:
- Convex hull (single): Acceptable for fin shape but adds complexity over box with negligible fidelity benefit for thin flat surfaces.
- No fin colliders at all: Considered — fins don't need ground contact in the current simulation. However, removing colliders would prevent future contact detection scenarios and could cause PhysX to use default collision behavior. Use collision filtering instead to disable fin-ground contact if needed.
- Capsule: Poor fit for flat rectangular fins.

---

## RQ-5: Aerodynamic Force Application Architecture

**Decision**: The current architecture already applies forces at fin links (confirmed by codebase exploration). The refactoring should ensure this continues to work correctly after the sign convention fix, and should add center-of-pressure offset if the vehicle config specifies one.

**Rationale**: The exploration found that `edf_landing_task.py` lines 806-812 already apply per-fin forces to individual fin bodies via `set_external_force_and_torque(forces, torques, body_ids=ext_body_ids)` where `ext_body_ids = [body_id, *fin_body_ids]`. The code comment states: "Apply aerodynamic forces directly to each fin link. PhysX propagates the resulting reactions to the main body; no manual tau_fins needed."

The main change needed is ensuring the force direction is correct after the sign convention fix (RQ-2), since the lift/drag direction vectors are defined relative to the controller convention.

**Alternatives considered**: N/A — current architecture is already correct. Only needs validation after convention changes.

---

## RQ-6: Fin Telemetry Architecture

**Decision**: Implement telemetry as a GPU tensor ring buffer in the task, controlled by a config flag (`debug.fin_telemetry: true`). Optionally flush to disk as a `.pt` (PyTorch) file per episode.

**Rationale**:
- GPU tensor buffers avoid CPU-GPU transfer overhead during training.
- A ring buffer of configurable depth (default: 1 episode = max_steps) prevents unbounded memory growth.
- PyTorch `.pt` format is native to the existing codebase and supports efficient tensor serialization.
- Config flag ensures zero overhead when disabled (no buffer allocation, no compute).

**Data per fin per step** (7 fields × 4 fins):
- Commanded angle (1 scalar)
- Measured joint angle (1 scalar)
- Link world pose (7: pos[3] + quat[4])
- Local exhaust velocity (1 scalar, or 3-vector if directional)
- Angle of attack (1 scalar)
- Applied aero force (3-vector)
- Incoming joint wrench (6: force[3] + torque[3])

Total: ~20 floats × 4 fins = 80 floats/step. At 120 Hz for 10s episode = 1200 steps × 80 = 96K floats = 384 KB per env per episode — negligible.

**Alternatives considered**:
- Python logging/print: Rejected — too slow for per-step data, breaks GPU pipeline.
- TensorBoard logging: Deferred — useful for scalar summaries but too heavy for per-step tensor data.
- Structured JSON: Rejected — CPU-side serialization, poor performance.
