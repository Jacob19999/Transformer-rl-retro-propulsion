# Blender → IsaacLab Export Guide

How to model the drone in Blender and export it so `postprocess_usd.py` can
add physics APIs and produce a simulation-ready `drone.usd`.

---

## Hierarchy & Naming

Blender object names map directly to USD prim paths. The post-process script
**validates required names** and **warns about missing recommended ones**.

```
Drone  (Empty — root, do NOT rotate)
│
├── Body  (Empty — rigid body group, all fixed components go here)
│   ├── edf               (Mesh — combined duct + fan, single object)
│   │
│   │   ── Future body parts (add as children of Body) ──
│   ├── battery            (Mesh — 8S LiPo pack)
│   ├── esc                (Mesh — ESC board)
│   ├── jetson             (Mesh — compute module)
│   ├── frame_structure    (Mesh — carbon fibre frame + printed joints)
│   ├── imu                (Mesh — BNO085)
│   └── camera             (Mesh — optical flow)
│
├── Fin_1  (Mesh — right fin,    origin at hinge)   ← ARTICULATED
├── Fin_2  (Mesh — left fin,     origin at hinge)   ← ARTICULATED
├── Fin_3  (Mesh — forward fin,  origin at hinge)   ← ARTICULATED
├── Fin_4  (Mesh — aft fin,      origin at hinge)   ← ARTICULATED
│
└── Legs   (Mesh — single combined landing gear, rigid)
```

### Why this structure?

| Category | Parent | Reason |
|----------|--------|--------|
| **Fins** | `Drone` (direct child) | Need RevoluteJoint → must be separate articulation links |
| **Legs** | `Drone` (direct child) | Single rigid mesh now; if legs need individual articulation later, split into `Leg_1`…`Leg_4` and promote to direct children |
| **Body parts** | `Body` (child) | All rigidly attached — grouped under one Empty so they share the root rigid body |

### Naming contract

| Status | Blender Object | USD Prim Path | Notes |
|--------|----------------|---------------|-------|
| **Required** | `Drone` | `/Drone` | Root empty |
| **Required** | `Body` | `/Drone/Body` | Empty (rigid body group) |
| **Required** | `Fin_1` | `/Drone/Fin_1` | Right fin — origin at hinge |
| **Required** | `Fin_2` | `/Drone/Fin_2` | Left fin |
| **Required** | `Fin_3` | `/Drone/Fin_3` | Forward fin |
| **Required** | `Fin_4` | `/Drone/Fin_4` | Aft fin |
| Recommended | `edf` | `/Drone/Body/edf` | Combined duct + fan, child of Body |
| Recommended | `Legs` | `/Drone/Legs` | Single mesh, direct child of Drone |

**Required** = post-process script fails without them.
**Recommended** = script prints a warning if missing, but still proceeds.

---

## Coordinate System

| Axis | Meaning |
|------|---------|
| +X | Forward (drone nose direction) |
| -Y | Right (starboard) |
| +Z | Up |

This is Z-up right-handed — the Isaac Sim convention.

- Set Blender scene to **Metric, meters**
- **Do not rotate the `Drone` root**. All geometry is pre-oriented in Z-up world space.

---

## Fin Origin Placement (Critical)

Each fin's **Blender object origin must be at the hinge point**, not the mesh
center. The post-process script creates a revolute joint at this position.

**Required fin hinge positions (Z-up world frame):**

| Object | X (m) | Y (m) | Z (m) | Hinge Axis | Description |
|--------|--------|--------|--------|------------|-------------|
| `Fin_1` | 0.000 | -0.055 | -0.14 | X (pitch) | Right fin |
| `Fin_2` | 0.000 | +0.055 | -0.14 | X (pitch) | Left fin |
| `Fin_3` | -0.055 | 0.000 | -0.14 | Y (yaw) | Forward fin |
| `Fin_4` | +0.055 | 0.000 | -0.14 | Y (yaw) | Aft fin |

> Derived from `default_vehicle.yaml` → `fins.fins_config[i].position` via
> FRD→Z-up: `(x, y, z)_FRD → (x, -y, -z)_Zup`.

To set a custom origin in Blender:
1. Select the fin mesh → Edit Mode
2. Select a vertex at the hinge location → Shift+S → Cursor to Selected
3. Object Mode → Object → Set Origin → Origin to 3D Cursor

---

## Legs Placement

`Legs` is a single combined mesh — all four landing legs as one Blender object,
direct child of `Drone`. Origin can be at mesh center (no joint is created).

If individual leg articulation is needed later, split into `Leg_1`…`Leg_4`
as separate meshes, each a direct child of `Drone`, and add joint logic to
`postprocess_usd.py`.

---

## Adding Future Components

To add a new rigid body part (e.g. battery, ESC, Jetson):

1. Model it in Blender as a child of `Body`
2. Name it to match the YAML primitive name where possible
   (e.g. `battery` → `battery_8s_lipo` in YAML)
3. Re-export and re-run `postprocess_usd.py` — no script changes needed

To add a new articulated part (e.g. deployable legs):

1. Model as a direct child of `Drone` (not `Body`)
2. Add a spec dataclass to `parts_registry.py` (mirrors `FinSpec`)
3. Add joint creation logic to `postprocess_usd.py`

---

## Physics — Do NOT Add in Blender

The post-process script adds **all** USD physics APIs:
- `ArticulationRootAPI` + `RigidBodyAPI` + `MassAPI` on `/Drone`
- `RigidBodyAPI` + `MassAPI` on each fin
- `RevoluteJoint` + `DriveAPI` for each fin
- All mass properties from `default_vehicle.yaml`

Do **not** add Rigid Body or Constraint modifiers in Blender.

---

## Blender USD Export Settings

1. **File → Export → Universal Scene Description (.usd)**
2. Settings:
   - **Root Prim Path**: leave empty
   - **Up Axis**: Z
   - **Scale**: 1.0 (metres)
   - **Export Normals**: On
   - **Export Materials**: On (optional, visual only)
   - **Instancing**: Off
3. Save as `drone_blender.usd` (separate from the final `drone.usd`)

---

## Post-Processing

```bash
# Validate naming (no output — use during modelling)
python -m simulation.isaac.usd.postprocess_usd \
    --input  simulation/isaac/usd/drone_blender.usd \
    --validate-only

# Add physics and write final drone.usd
python -m simulation.isaac.usd.postprocess_usd \
    --input  simulation/isaac/usd/drone_blender.usd \
    --output simulation/isaac/usd/drone.usd \
    --config simulation/configs/default_vehicle.yaml
```

---

## Quick Checklist

- [ ] Scene units: **Metric / metres**
- [ ] Up axis: **Z**
- [ ] Root empty named **`Drone`** (no rotation applied)
- [ ] **`Body`** empty parented to `Drone`, with `edf` as child
- [ ] **`Fin_1`…`Fin_4`** meshes parented to `Drone`, origins at hinge points
- [ ] **`Legs`** single mesh parented to `Drone`
- [ ] No physics modifiers in Blender
- [ ] Exported with Scale=1.0, Up Axis=Z
- [ ] `--validate-only` passes before full post-process
