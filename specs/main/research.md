# Research: Isaac Sim Mass Properties, Thrust Test & Environmental Forces

**Feature**: 002-isaac-mass-thrust-env
**Date**: 2026-03-11

## Research Questions & Decisions

### RQ-1: How to read mass properties from a USDC scene without launching Isaac Sim?

**Decision**: Use `pxr` (OpenUSD) Python bindings to open the `.usdc` stage and read `UsdPhysics.MassAPI` attributes directly.

**Rationale**: The `pxr` library is already a dependency (used by `postprocess_usd.py` and `drone_builder.py`). It can open USD stages headlessly without an Isaac Sim runtime. This enables the validation script to run in CI or pre-commit hooks.

**Alternatives considered**:
- Isaac Sim ArticulationView queries — requires full sim launch, GPU, ~30s startup. Rejected: too heavy for validation.
- Manual USD XML parsing — fragile, no schema support. Rejected.

**Key APIs**:
- `Usd.Stage.Open(path)` — open stage
- `UsdPhysics.MassAPI(prim)` — access mass, CoM, inertia
- `prim.GetAttribute("physics:mass").Get()` — total mass
- `prim.GetAttribute("physics:centerOfMass").Get()` — CoM offset
- `prim.GetAttribute("physics:diagonalInertia").Get()` — diagonal inertia
- `prim.GetAttribute("physics:principalAxes").Get()` — orientation quaternion for inertia axes

**Note**: Isaac Sim USD may store inertia as diagonal + principal axes rotation, not full 3x3 tensor. The validation script must handle both representations and convert to compare against YAML's full 3x3 tensor.

### RQ-2: How does Isaac Sim represent mass properties in the USDC scene?

**Decision**: The hand-authored `drone.usdc` uses `UsdPhysics.MassAPI` on the Body rigid body with explicit mass. The MassAPI stores:
- `physics:mass` — scalar total mass (kg)
- `physics:centerOfMass` — GfVec3f offset from prim origin
- `physics:diagonalInertia` — GfVec3f principal moments
- `physics:principalAxes` — GfQuatf rotation of principal axes relative to prim frame

**Rationale**: This is the standard PhysX / USD Physics schema. PostprocessUSD already sets these values when processing Blender exports. The validation script must read them back.

**FRD ↔ Z-up conversion**: YAML stores CoM in FRD body frame. USD stores in Z-up prim-local frame. Conversion via `frd_to_zup()` from `parts_registry.py` is required before comparison.

### RQ-3: How to apply wind forces in Isaac Sim?

**Decision**: Use `robot.set_external_force_and_torque()` — the same API already used for thrust and fin aero forces in `_apply_action()`. Add wind force as an additional component to the `forces` tensor.

**Rationale**: This is the simplest approach with zero new API surface. Wind force is a body force proportional to dynamic pressure of the relative wind:
```
F_drag = 0.5 × ρ × |v_rel|² × Cd × A × direction
v_rel = v_wind - v_body (world frame)
```

**Implementation approach**:
1. Add a lightweight `IsaacWindModel` class that wraps the existing `WindModel` or directly samples wind vectors from config.
2. At each `_apply_action()` call, compute relative wind → aerodynamic drag force.
3. Add drag force to the `forces` tensor alongside thrust and fin forces.
4. Use the drone's projected areas from YAML (`_computed` section or explicit drag coefficients) for Cd × A.

**Alternatives considered**:
- Port full `DrydenFilter` to PyTorch GPU tensors — complex, deferred to training optimization stage. Rejected for MVP.
- Isaac Sim native wind field — not available in IsaacLab's DirectRLEnv API. Rejected.
- Simple constant wind only — accepted for MVP; Dryden turbulence can be added later.

### RQ-4: What drag model to use for wind forces?

**Decision**: Use the aerodynamic drag model from `aero_model.py` as reference. The drone's projected areas (from YAML primitives) and drag coefficients define directional drag.

**Key parameters from `default_vehicle.yaml`**:
- Each primitive defines `projected_drag_areas` (x, y, z) and `drag_coefficient`
- `aero_model.py` aggregates these into composite Cd×A per axis
- For Isaac Sim MVP: use a simplified single composite drag coefficient applied to total projected area per axis

**Rationale**: Matching the custom sim's drag model ensures consistency between the two simulation stacks, supporting sim-to-sim validation before sim-to-real transfer.

### RQ-5: Should mass properties migrate from YAML to USDC as source of truth?

**Decision**: **No** — YAML remains authoritative. USDC is a derived artifact.

**Rationale**:
- YAML is human-readable, version-controlled, and supports domain randomization ranges
- USDC is binary, harder to review in PRs
- The custom Python sim already reads from YAML; having a single source prevents divergence
- Constitution v1.1.0 codifies this decision

**Future consideration**: If a CAD pipeline (SolidWorks/Inventor → USDC) becomes the primary design tool, the constitution must be amended to make USDC authoritative and YAML derived.

### RQ-6: Thrust direction validation

**Decision**: Current thrust is applied as `forces[:, 0, 2] = thrust_actual` with `is_global=True` (world +Z). In Z-up world, +Z is up. For a retro-propulsive lander, thrust opposes gravity (pushes up). This is correct.

**Verification needed**: The thrust test diagnostic must confirm that positive thrust_cmd produces upward acceleration. If the drone descends under full thrust, the direction is wrong.

**Note on body-frame thrust**: Currently thrust is world-frame +Z (always up regardless of tilt). For realistic TVC, thrust should be body-frame +Z (down in FRD) → rotated to world frame. This simplification is acceptable for initial validation but must be addressed for training. The thrust test will reveal if this matters at hover/near-hover attitudes.
