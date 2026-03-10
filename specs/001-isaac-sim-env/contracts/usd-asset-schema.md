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
            ├── /Fin_1      (Xform + RevoluteJoint + DriveAPI)
            │   └── /Geom   (UsdGeom.Cube, scaled to chord × span × thickness)
            ├── /Fin_2      (Xform + RevoluteJoint + DriveAPI)
            │   └── /Geom   (UsdGeom.Cube, scaled to chord × span × thickness)
            ├── /Fin_3      (Xform + RevoluteJoint + DriveAPI)
            │   └── /Geom   (UsdGeom.Cube, scaled to chord × span × thickness)
            └── /Fin_4      (Xform + RevoluteJoint + DriveAPI)
                └── /Geom   (UsdGeom.Cube, scaled to chord × span × thickness)
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

### Fin Physical Dimensions (from `default_vehicle.yaml`)

All four fins are identical NACA0012 control surfaces:

| Property | Value | Source |
|----------|-------|--------|
| Chord (along exhaust flow, body +Z) | 0.065 m | `fins.chord` |
| Span (radial, perpendicular to flow) | 0.055 m | `fins.span` |
| Max thickness (at 30% chord, NACA0012) | 0.0078 m | `fins.max_thickness` = chord × 0.12 |
| Planform area | 0.003575 m² | `fins.planform_area` = chord × span |
| Airfoil profile | NACA0012 (symmetric) | `fins.airfoil_profile` |

**Geometry prim** (`/Drone/Fin_N/Geom`): `UsdGeom.Cube` with `xformOp:scale` set to half-extents:

```
size = 1.0  (unit cube, all scaling via xformOp:scale)
xformOp:scale = (chord/2, span/2, thickness/2)
             = (0.0325,  0.0275,  0.0039)   [metres, in fin-local frame]
```

The geometry origin is **offset from the hinge axis** so the hinge sits at the fin's leading edge:

```
xformOp:translate = (0.0, 0.0, chord/2) = (0.0, 0.0, 0.0325)
```

This places the leading edge at z=0 (the hinge point) and the trailing edge at z=+chord in the fin's local frame.

> **Note on NACA0012 fidelity**: A box approximation is used for the collision shape because PhysX convex hull collision from a true airfoil mesh is expensive to author and provides no additional physics accuracy for the force-injection workflow (aerodynamic forces are computed analytically in Python, not from PhysX contact geometry). If visual fidelity is required, the `/Geom` prim can be swapped for a proper NACA0012 mesh with a separate convex-hull collision approximant.

### Per-Fin Placement (body frame, from `fins.fins_config`)

| Fin | Name | Hinge position (body FRD, m) | Hinge axis (body frame) | Span direction |
|-----|------|------------------------------|-------------------------|----------------|
| 1 | `fin_1_right`   | [0, +0.055, 0.14] | [1, 0, 0] (body X) | +Y (starboard → outboard) |
| 2 | `fin_2_left`    | [0, −0.055, 0.14] | [1, 0, 0] (body X) | −Y (port → outboard)      |
| 3 | `fin_3_forward` | [+0.055, 0, 0.14] | [0, 1, 0] (body Y) | +X (forward → outboard)   |
| 4 | `fin_4_aft`     | [−0.055, 0, 0.14] | [0, 1, 0] (body Y) | −X (aft → outboard)       |

All fins are located at body-frame Z = 0.14 m (within the exhaust stream, aft of the EDF exit plane at Z ≈ 0.12 m).

### Fin Joints (`/Drone/Fin_N`)

| USD API | Required Attributes |
|---------|---------------------|
| `UsdPhysics.RevoluteJoint` | `physics:axis = <hinge_axis>` (from per-fin table above) |
| `UsdPhysics.RevoluteJoint` | `physics:lowerLimit = -15.0` (degrees) |
| `UsdPhysics.RevoluteJoint` | `physics:upperLimit = +15.0` (degrees) |
| `UsdPhysics.RevoluteJoint` | `physics:body0 = </Drone>` (parent rigid body) |
| `UsdPhysics.RevoluteJoint` | `physics:localPos0 = <hinge_position_body_frame>` |
| `UsdPhysics.MassAPI` | `physics:mass = 0.003` kg (≈3 g, estimated CFRP fin mass) |
| `UsdPhysics.DriveAPI` | `physics:stiffness = 25.0` (= 1/tau_servo = 1/0.04) |
| `UsdPhysics.DriveAPI` | `physics:damping = 0.05` (tuned; ≈ 2√(stiffness × I_fin)) |
| `UsdPhysics.DriveAPI` | `physics:targetPosition = 0.0` (default neutral) |

**Servo PD parameters**: The first-order lag (tau = 0.04 s) is approximated via a PD position drive. Estimated fin rotational inertia about hinge: I_fin ≈ (1/12) × m × span² = (1/12) × 0.003 × 0.055² ≈ 7.6×10⁻⁷ kg·m². Thus critical damping: `damping = 2 × √(25.0 × 7.6×10⁻⁷) ≈ 0.0087`. A slight over-damping factor of ~5× (`damping = 0.05`) is recommended to suppress oscillation at 1/120s steps.

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
