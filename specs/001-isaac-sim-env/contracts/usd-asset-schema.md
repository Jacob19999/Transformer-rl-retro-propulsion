# Contract: USD Drone Asset Schema

**Feature**: Isaac Sim Vectorized Drone Simulation Environment
**Branch**: `001-isaac-sim-env` | **Date**: 2026-03-10

This contract defines the USD scene graph structure that `drone_builder.py` MUST produce from `default_vehicle.yaml`.

---

## USD Scene Hierarchy

```
/World
└── /envs
    └── /env_0              (Xform — one per parallel environment)
        └── /Drone          (Xform + RigidBodyAPI + MassAPI)
            ├── /Body       (Cylinder geom — EDF duct as single compound shape)
            ├── /Fin_1      (Mesh/Cube + RevoluteJoint + DriveAPI)
            ├── /Fin_2      (Mesh/Cube + RevoluteJoint + DriveAPI)
            ├── /Fin_3      (Mesh/Cube + RevoluteJoint + DriveAPI)
            └── /Fin_4      (Mesh/Cube + RevoluteJoint + DriveAPI)
```

---

## USD API Requirements

### Root Rigid Body (`/Drone`)

| USD API | Required Attributes |
|---------|---------------------|
| `UsdPhysics.RigidBodyAPI` | `physics:rigidBodyEnabled = true` |
| `UsdPhysics.MassAPI` | `physics:mass = <total_mass_kg>` |
| `UsdPhysics.MassAPI` | `physics:centerOfMass = <com_body_frame_m>` |
| `UsdPhysics.MassAPI` | `physics:diagonalInertia = <[Ixx, Iyy, Izz]>` |
| `UsdPhysics.CollisionAPI` | Applied to all geometry children |

Note: Full inertia tensor off-diagonal terms (Ixy, Ixz, Iyz) MUST be included if non-negligible (>5% of diagonal) via `physics:principalAxes`.

### Fin Joints (`/Drone/Fin_N`)

| USD API | Required Attributes |
|---------|---------------------|
| `UsdPhysics.RevoluteJoint` | `physics:axis = <hinge_axis>` |
| `UsdPhysics.RevoluteJoint` | `physics:lowerLimit = -15.0` (degrees) |
| `UsdPhysics.RevoluteJoint` | `physics:upperLimit = +15.0` (degrees) |
| `UsdPhysics.DriveAPI` | `physics:stiffness = <servo_stiffness>` |
| `UsdPhysics.DriveAPI` | `physics:damping = <servo_damping>` |
| `UsdPhysics.DriveAPI` | `physics:targetPosition = 0.0` (default neutral) |

The servo first-order lag (tau = 0.04 s) is approximated via PD drive parameters: `stiffness = 1/tau_servo`, `damping = 2*sqrt(stiffness*inertia_fin)`.

### Material & Friction

```
/World/Looks/GroundMaterial (UsdShade.Material)
  physxMaterial:staticFriction  = 0.5
  physxMaterial:dynamicFriction = 0.5
  physxMaterial:restitution     = 0.1
```

Landing pad surface MUST reference `/World/Looks/GroundMaterial`.

---

## Coordinate Frame Convention

- Body frame: **FRD (Forward-Right-Down)** with +Z as thrust direction
- USD / PhysX world frame: **Y-Up** (Isaac Sim default); a fixed 90° rotation about X is applied at the `/Drone` root to align FRD body +Z with PhysX world -Y (up direction in Y-Up)
- All positions in the YAML config (`position: [x, y, z]`) are in FRD body frame and must be transformed at USD construction time

---

## YAML → USD Primitive Mapping

| YAML shape | USD Prim Type | Dimension mapping |
|------------|---------------|-------------------|
| `cylinder` | `UsdGeom.Cylinder` | `radius` → `radius`; `height` → `height` |
| `box` | `UsdGeom.Cube` | `dimensions: [x,y,z]` → scaled unit cube |
| `sphere` | `UsdGeom.Sphere` | `radius` → `radius` |

All primitives are authored as children of `/Drone/Body` and contribute to the composite rigid body via compound collision shapes. Mass and inertia are overridden at the root level with the pre-computed composite values from `MassProperties`.

---

## Output File

The asset builder MUST produce:
- `simulation/isaac/usd/drone.usd` — reusable asset (not per-env)
- Asset is instantiated per environment via USD instance references
