# Data Model: Refactor Drone Fin Physics Layer

**Feature Branch**: `006-refactor-fin-physics`
**Date**: 2026-03-21

## Entities

### Fin Link (USD/PhysX)

Represents one of four articulated control surfaces attached to the drone body.

| Attribute | Type | Source | Notes |
| --------- | ---- | ------ | ----- |
| prim_name | string | USD | One of: `RightFin`, `LeftFin`, `FwdFin`, `AftFin` |
| joint_name | string | USD | `{prim_name}_Joint` |
| mass | float (kg) | USD MassAPI | 1e-5 kg (negligible) |
| inertia | Vec3f | USD MassAPI | Zeroed (negligible link) |
| collision_shape | enum | USD CollisionAPI | `box` (was `convexDecomposition`) |
| hinge_axis | string | USD RevoluteJoint | `"X"` or `"Y"` — authored to match positive-deflection convention |
| joint_lower_limit | float (deg) | USD RevoluteJoint | From YAML `max_deflection` |
| joint_upper_limit | float (deg) | USD RevoluteJoint | From YAML `max_deflection` |
| drive_stiffness | float | USD AngularDrive | From YAML servo config |
| drive_damping | float | USD AngularDrive | From YAML servo config |

**Relationships**: Each Fin Link is a child of Drone Body via a revolute joint.

**State at runtime** (per env, per step):
- `joint_pos` — measured position (degrees from PhysX, converted to radians)
- `joint_vel` — measured velocity
- `link_pos_w` — world position of link origin
- `link_quat_w` — world orientation of link

---

### Drone Body (USD/PhysX)

The main rigid body of the articulation.

| Attribute | Type | Source | Notes |
| --------- | ---- | ------ | ----- |
| prim_path | string | USD | `/Drone/Body` |
| mass | float (kg) | USD MassAPI | From YAML `mass_properties.total_mass` minus fin masses |
| center_of_mass | Vec3f (m) | USD MassAPI | From YAML `mass_properties.center_of_mass`, converted FRD→Z-up |
| diagonal_inertia | Vec3f (kg·m²) | USD MassAPI | Decomposed from YAML `mass_properties.inertia_tensor` |
| principal_axes | Quatf | USD MassAPI | Decomposed from YAML `mass_properties.inertia_tensor` |
| collision_shape | enum | USD CollisionAPI | `convexDecomposition` (unchanged) |

**Relationships**: Parent of all four Fin Links. Receives thrust and environmental forces.

---

### Fin Telemetry Record (Runtime)

Per-fin, per-step diagnostic data. Stored in GPU tensor buffers when enabled.

| Field | Shape | Unit | Description |
| ----- | ----- | ---- | ----------- |
| cmd_angle | (N, 4) | rad | Commanded fin deflection |
| meas_angle | (N, 4) | rad | Measured joint position (from PhysX) |
| link_pos | (N, 4, 3) | m | Fin link world position |
| link_quat | (N, 4, 4) | — | Fin link world orientation (wxyz) |
| exhaust_vel | (N, 4) | m/s | Local exhaust stream velocity at fin |
| aoa | (N, 4) | rad | Angle of attack |
| aero_force | (N, 4, 3) | N | Applied aerodynamic force (world frame) |
| joint_wrench | (N, 4, 6) | N, N·m | Incoming joint wrench (force[3] + torque[3]) |

Where N = number of environments.

**Storage**: Ring buffer of depth `max_episode_steps`. Flushed to `.pt` file on episode end if `debug.fin_telemetry_save: true`.

---

### Fin Aero Parameters (Config → Runtime)

Per-fin aerodynamic parameters used by `compute_fin_forces_body()`.

| Field | Type | Source | Notes |
| ----- | ---- | ------ | ----- |
| lift_dir_body | Vec3 | YAML `fins.fins_config[i].lift_direction` | Unit vector, body frame |
| drag_dir_body | Vec3 | Derived | Perpendicular to lift in flow plane |
| position_body | Vec3 | YAML `fins.fins_config[i].position` | Moment arm for torque verification |
| chord | float (m) | YAML `fins.chord` | NACA0012 chord length |
| span | float (m) | YAML `fins.span` | Fin span |
| Cl_alpha | float (1/rad) | YAML `fins.Cl_alpha` | Lift curve slope |
| Cd_0 | float | YAML `fins.Cd_0` | Zero-lift drag coefficient |

**Relationships**: One set per fin, indexed consistently with Fin Link order after convention fix.

## State Transitions

### Fin Joint Command Flow (per step)

```
Policy action (normalized [-1,1])
    │
    ▼
Scale to radians (× max_deflection)
    │
    ▼
[REMOVED: index swap & sign negate]   ← Convention fix eliminates this
    │
    ▼
Convert rad → deg
    │
    ▼
set_joint_position_target(deg)  → PhysX drive → joint settles
    │
    ▼
Read joint_pos (deg) → convert deg → rad (unconditional)
    │
    ▼
compute_fin_forces_body(measured_rad)
    │
    ▼
Apply force at fin link body (world frame)
```

### Mass Properties Authoring Flow

```
YAML mass_properties (use_explicit: true)
    │
    ├─ total_mass → body_mass = total_mass - 4×fin_mass
    ├─ center_of_mass (FRD) → convert FRD→Z-up → MassAPI.centerOfMass
    └─ inertia_tensor (3×3) → eigendecompose → diagonal + principal_axes_quat
                                              → MassAPI.diagonalInertia + principalAxes
    │
    ▼
validate_mass_props.py: read USD ↔ compare YAML → PASS/FAIL
```
