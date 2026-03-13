# Isaac Sim Scene Setup — EDF Drone USD

This document describes the required prim hierarchy, physics settings, and manual
Isaac Sim steps for the drone scene exported from Blender.

The automated path is `postprocess_usd.py` which applies all of these programmatically.
Use this guide when setting up or inspecting the scene manually in the Isaac Sim GUI.

---

## 1. Coordinate System


| Setting         | Value   |
| --------------- | ------- |
| Up axis         | **Z**   |
| Meters per unit | **1.0** |


Set via: **Edit → Stage → Up Axis / Meters Per Unit**

---

## 2. Expected Prim Hierarchy

```
/Drone                          ← Xform  [ArticulationRootAPI]
├── Body                        ← Xform  [RigidBodyAPI, MassAPI]
│   ├── edf                     ← Mesh   (visual only — no physics API)
│   └── Legs                    ← Mesh   (visual only — no physics API)
├── Fin_1                       ← Xform  [RigidBodyAPI, MassAPI]
│   └── RevoluteJoint           ← RevoluteJoint (Body → Fin_1)
├── Fin_2                       ← Xform  [RigidBodyAPI, MassAPI]
│   └── RevoluteJoint           ← RevoluteJoint (Body → Fin_2)
├── Fin_3                       ← Xform  [RigidBodyAPI, MassAPI]
│   └── RevoluteJoint           ← RevoluteJoint (Body → Fin_3)
└── Fin_4                       ← Xform  [RigidBodyAPI, MassAPI]
    └── RevoluteJoint           ← RevoluteJoint (Body → Fin_4)
```

> **Rule:** Only `Body` and `Fin_N` prims get physics APIs (`RigidBodyAPI`).
> `edf` and `Legs` are visual geometry — **do not apply RigidBodyAPI to them**.
> PhysX will error if a rigid body exists as a child of another rigid body in
> the same hierarchy without an XformStack reset.

---

## 3. /Drone — Articulation Root


| Property        | Value                                       |
| --------------- | ------------------------------------------- |
| API             | `ArticulationRootAPI`                       |
| Prim type       | `Xform`                                     |
| Physics enabled | via API only — no RigidBodyAPI on this prim |


**GUI:** Select `/Drone` → **Add → Physics → Articulation Root**

---

## 4. /Drone/Body — Rigid Body + Mass

### RigidBodyAPI


| Property           | Value |
| ------------------ | ----- |
| Rigid body enabled | true  |
| Kinematic          | false |


**GUI:** Select `/Drone/Body` → **Add → Physics → Rigid Body**

### MassAPI

Author total mass on `/Drone/Body`, then let Isaac Sim / PhysX compute CoM and
inertia from the collider geometry.


| Property         | Value (baseline) |
| ---------------- | ---------------- |
| Mass             | 3.13 kg          |
| Center of mass   | leave unset      |
| Diagonal inertia | leave unset      |
| Principal axes   | leave unset      |


> The CAD/Blender mesh and convex-decomposition colliders are now the source of
> truth for mass distribution. Legacy YAML CoM/inertia values are ignored by the
> Isaac pipeline.

**GUI:** Select `/Drone/Body` → **Add → Physics → Mass**

---

## 5. /Drone/Fin_N — Fin Rigid Bodies

Each of the 4 fins (`Fin_1` … `Fin_4`) needs:

### RigidBodyAPI


| Property           | Value |
| ------------------ | ----- |
| Rigid body enabled | true  |
| Kinematic          | false |


### MassAPI


| Property | Value                                  |
| -------- | -------------------------------------- |
| Mass     | 0.010 kg (from `fins.servo.weight_kg`) |


**GUI:** Select each fin → **Add → Physics → Rigid Body**, then **Add → Physics → Mass**

---

## 6. /Drone/Fin_N/RevoluteJoint — Fin Joints

One `RevoluteJoint` per fin, parented under the fin Xform. Each joint must have a
**unique per-fin name** (`Fin_N_Joint`) so IsaacLab can resolve them via
`find_joints(["Fin_1_Joint", "Fin_2_Joint", "Fin_3_Joint", "Fin_4_Joint"])`.

Full joint prim paths:

| Fin   | Joint prim path              |
| ----- | ---------------------------- |
| Fin_1 | `/Drone/Fin_1/Fin_1_Joint`   |
| Fin_2 | `/Drone/Fin_2/Fin_2_Joint`   |
| Fin_3 | `/Drone/Fin_3/Fin_3_Joint`   |
| Fin_4 | `/Drone/Fin_4/Fin_4_Joint`   |

Per-joint settings:

| Property           | Fin_1 / Fin_2                | Fin_3 / Fin_4                |
| ------------------ | ---------------------------- | ---------------------------- |
| Axis               | X                            | Y                            |
| Lower limit        | −20°                         | −20°                         |
| Upper limit        | +20°                         | +20°                         |
| Body 0             | `/Drone/Body`                | `/Drone/Body`                |
| Body 1             | `/Drone/Fin_1` / `Fin_2`     | `/Drone/Fin_3` / `Fin_4`     |
| Local pos 0 (Z-up) | see table below              | see table below              |
| Local pos 1        | (0, 0, 0)                    | (0, 0, 0)                    |


**Fin hinge positions**

Use the authored fin prim transforms as the source of truth. `postprocess_usd.py`
reads each fin origin relative to `/Drone/Body` and writes that value into the
joint `localPos0`. Legacy YAML fin positions are ignored.


**GUI:** Select fin prim → **Add → Physics → Joint → Revolute Joint**,
then set bodies, axis, limits, and local positions in the property panel.

### DriveAPI (angular)

Each joint also needs a `DriveAPI` on the `angular` token:


| Property        | Value |
| --------------- | ----- |
| Stiffness       | 20.0  |
| Damping         | 1.0   |
| Target position | 0.0°  |


**GUI:** Select the joint prim → **Add → Physics → Drive → Angular**

---

## 7. Common Errors


| Error                                                       | Cause                             | Fix                                                    |
| ----------------------------------------------------------- | --------------------------------- | ------------------------------------------------------ |
| `Rigid Body of (/Drone/Body/Legs) missing xformstack reset` | `Legs` has `RigidBodyAPI` applied | Remove `RigidBodyAPI` from `Legs` — it is visual only  |
| `Failed to open layer @drone_blender.usd@`                  | Input file has wrong extension    | Use the actual file extension (`.usdc` vs `.usd`)      |
| `MassAPI not applied`                                       | `postprocess_usd.py` was not run  | Run postprocess or add Mass API manually per section 4 |
| `prim /Drone/Body not found`                                | Hierarchy does not match          | Rename Blender objects to match expected names exactly |


---

## 8. Automated Setup

Instead of manual steps, run:

```bash
python -m simulation.isaac.usd.postprocess_usd \
    --input  simulation/isaac/usd/drone.usdc \
    --output simulation/isaac/usd/drone.usd \
    --config simulation/configs/default_vehicle.yaml
```

`validate_mass_props.py` is a legacy YAML-vs-USD comparison tool and is no longer
authoritative when PhysX computes CoM/inertia from colliders.

Legacy validation command:

```bash
python -m simulation.isaac.scripts.validate_mass_props \
    --usd simulation/isaac/usd/drone.usd \
    --config simulation/configs/default_vehicle.yaml \
    --tolerance 0.01
```

