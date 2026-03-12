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

---

## User Story 4: Gyro Precession Modeling — Research

**Isaac Sim version**: v5.1.0 (IsaacLab)

### RQ-7: How to compute gyroscopic precession torque in Isaac Sim?

**Decision**: Inject `τ_gyro = −ω_body × h_fan` as an external torque via `set_external_force_and_torque()`, computed in body frame then rotated to world frame (since `is_global=True`).

**Rationale**: PhysX (Isaac Sim's physics engine) handles the standard rigid body `ω × (I·ω)` term internally for the drone's body inertia. However, the spinning fan's angular momentum `h_fan` is *not* represented as a spinning rigid body in the simulation — the fan is lumped into the Body rigid body. Therefore, the gyroscopic coupling from the fan spin must be explicitly injected as an external torque.

**Physics reference** (vehicle.md §4.1):
```
I·ω̇ = τ_total − ω × (I·ω) − ω × h_fan
```
PhysX handles: `I·ω̇ + ω × (I·ω) = τ_external`
We must add: `τ_gyro = −ω × h_fan` to external torques.

**Implementation**:
```python
# In _apply_action(), after fin forces, before set_external_force_and_torque:
omega_b = self.robot.data.root_ang_vel_b          # (N, 3) body frame
omega_fan = (self.thrust_actual / _K_THRUST).clamp(min=0).sqrt()  # (N,)

# h_fan in body frame: fan spins about +Z in FRD (down = thrust axis)
h_fan_b = torch.zeros_like(omega_b)
h_fan_b[:, 2] = _I_FAN * omega_fan

# Precession torque in body frame
tau_gyro_b = -torch.linalg.cross(omega_b, h_fan_b)  # (N, 3)

# Rotate to world frame for is_global=True
tau_gyro_w = _rotate_body_to_world(self.robot.data.root_quat_w, tau_gyro_b)
torques[:, 0, :] += tau_gyro_w
```

**Axis coupling breakdown** (h_fan = [0, 0, L], L = I_fan·ω_fan):
```
τ_gyro = −[p, q, r] × [0, 0, L] = [−q·L,  p·L,  0]
```
- Pitch rate q → roll torque (τ_x = −q·L) ✓
- Roll rate p → pitch torque (τ_y = p·L) ✓
- Yaw rate r → **zero torque** (r is parallel to spin axis, cross product = 0)

**Implication for fin-commanded yaw**: When fins produce a yaw torque and the drone develops yaw rate `r`, that yaw rate itself produces NO precession (correct physics — spin axis parallel to yaw axis). The precession observed during a yaw maneuver comes from residual pitch/roll rates (p, q ≠ 0) induced by: (1) fin aerodynamic cross-coupling in the 4-fin geometry, (2) off-diagonal inertia terms accelerating pitch/roll from the yaw torque. The implementation correctly captures this because `root_ang_vel_b` is the actual real-time ω_body — whatever pitch/roll rates the fins have induced are already present in `omega_b` and drive the precession term each substep.

**Key API** (IsaacLab v5.1.0 / Isaac Sim v5.1.0):
- `self.robot.data.root_ang_vel_b` → body-frame angular velocity `(num_envs, 3)`
- `self.robot.data.root_quat_w` → world-frame quaternion `(num_envs, 4)`, scalar-last `[qx, qy, qz, qw]`
- Rotation helper: reuse existing `_rotate_world_to_body()` pattern (lines 689–711 of `edf_landing_task.py`) with inverse (conjugate) quaternion for body→world

**Alternatives considered**:
- Model rotor as separate spinning rigid body in PhysX → overly complex, adds joint DOFs, performance cost for N envs. Rejected.
- Use `is_global=False` (body frame) for `set_external_force_and_torque` → would require all existing forces (thrust, fin, wind) to also be in body frame. Rejected: disruptive refactor.
- Compute precession in world frame directly → requires rotating `omega_body` to world first, then h_fan to world, then cross product. Same result, more steps. The body-frame approach is cleaner because h_fan has a trivial form `[0, 0, I_fan * ω_fan]` in body frame.

### RQ-8: How should the rotor be represented for gyroscopic precession?

**Decision**: No USD prim is needed. `I_fan` is read directly from `edf.I_fan` in `default_vehicle.yaml` and stored as `self._I_fan` at task init. No `/Drone/Body/Rotor` cylinder is added to the USDC scene.

**Rationale**:
- Gyroscopic precession torque `τ_gyro = −ω × h_fan` is a **pure torque** applied to the rigid body's CoM. In rigid body dynamics, pure torques have the same effect regardless of where within the body the fan is located, so rotor position in the scene is irrelevant.
- `I_fan` (3.0e-5 kg·m²) represents only the rotating fan blades (~60g at r≈35mm). The full motor assembly's (0.35 kg) inertia is already in the Body's composite inertia tensor. This value is already present in `default_vehicle.yaml` under `edf.I_fan`.
- An invisible USD prim with `mass=0` would contribute nothing to physics and nothing to the torque computation — it was purely a validation artifact with no computational role.

**Alternatives considered**:
- Add `/Drone/Body/Rotor` invisible cylinder as a validation anchor → no computational value since torque is position-independent. Rejected.
- Hardcode `_I_FAN = 3.0e-5` as a module constant → works, but reading from config is cleaner and config-driven (constitution II). Rejected.
- Model rotor as a separate spinning rigid body with joint → adds physics DOFs, performance cost for N envs. Rejected.

### RQ-9: What is the correct `_rotate_body_to_world` implementation?

**Decision**: Implement as the conjugate of the existing `_rotate_world_to_body()` function. For unit quaternion `q`, the inverse rotation is the conjugate `q* = [-qx, -qy, -qz, qw]`.

**Implementation** (matching existing pattern at `edf_landing_task.py:689–711`):
```python
def _rotate_body_to_world(quat_w: torch.Tensor, v_body: torch.Tensor) -> torch.Tensor:
    """Rotate body-frame vector to world frame using quaternion.

    quat_w: (N, 4) scalar-last [qx, qy, qz, qw]
    v_body: (N, 3) body-frame vector
    Returns: (N, 3) world-frame vector
    """
    q_vec = quat_w[:, :3]
    qw    = quat_w[:, 3:4]
    # v_world = v + 2*qw*(q_vec × v) + 2*(q_vec × (q_vec × v))
    # Same formula as world-to-body but with original quat (not conjugate)
    t = 2.0 * torch.linalg.cross(q_vec, v_body)
    return v_body + qw * t + torch.linalg.cross(q_vec, t)
```

**Rationale**: The existing `_rotate_world_to_body` uses the conjugate quaternion (negates q_vec). For body-to-world, we use the original quaternion. This is a standard quaternion rotation formula, O(N) vectorized, no overhead.

### RQ-10: How to disable gravity for the precession diagnostic?

**Decision**: Use Isaac Sim's scene config `sim.gravity = (0.0, 0.0, 0.0)` or override at runtime via `self.sim.set_gravity((0, 0, 0))` if the API supports it. Alternatively, create a dedicated `isaac_env_gyro_test.yaml` config with gravity disabled.

**Rationale**: The precession diagnostic needs zero-g to isolate the gyroscopic effect from gravitational torques. A dedicated config file is the cleanest approach — consistent with the configuration-driven constitution.

**Implementation approach**:
1. Create `simulation/isaac/configs/isaac_env_gyro_test.yaml` with `sim.gravity: [0, 0, 0]`
2. Spawn drone at 5 m altitude with hover thrust
3. Apply external yaw torque pulse
4. Log roll/pitch/yaw rates over 2 seconds
5. Verify pitch response when precession enabled, no pitch response when disabled

**Alternative**: Use `PhysxScene.set_gravity()` at runtime → works but breaks config-driven principle. Rejected for primary approach; may be used as a CLI override `--no-gravity`.

### RQ-11: Fan angular momentum sign convention in Z-up world frame

**Decision**: In FRD body frame, the fan spins about +Z (downward, toward exhaust). In Z-up world frame at identity orientation, this maps to −Z. The `h_fan` vector in body frame is `[0, 0, +I_fan * ω_fan]` (positive Z, i.e., downward in FRD = along thrust axis).

**Verification**: The custom sim uses:
```python
h_fan = np.array([0.0, 0.0, self.I_fan * omega_fan])  # vehicle.py line 857
```
This is body-frame FRD, where +Z is down. The cross product `ω × h_fan` in body frame directly gives the precession torque. The sign is correct — positive fan spin about body +Z produces the expected precession coupling.

**Key check**: When the drone yaws (body ω_z > 0, i.e., clockwise viewed from above in FRD), the precession torque should produce a pitch or roll response. The direction depends on the fan spin direction (CW vs CCW viewed from above). For a typical EDF, the fan convention is documented in the config and determines the sign of `I_fan * ω_fan`.
