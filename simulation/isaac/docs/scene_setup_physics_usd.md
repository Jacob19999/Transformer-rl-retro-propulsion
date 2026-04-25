# Physics USD Scene Setup — Isaac Sim / IsaacLab

> **Scope:** EDF Drone v2 TVC testbed
> **Runtime:** Isaac Sim 5.1.0 + IsaacLab
> **Working directory:** `simulation/isaac/`

---

## 1. USD Prim Hierarchy

The drone USD must follow this exact hierarchy. The **asset metadata YAML is the authoritative source** — all validation derives from it.

```
/Drone                            (defaultPrim, Xform)
└─ Body                           (RigidBody + ArticulationRootAPI)
   ├─ <collision/visual geometry>
   ├─ joint_FwdFin                (UsdPhysics.RevoluteJoint → /Drone/FwdFin)
   ├─ joint_RightFin
   ├─ joint_AftFin
   └─ joint_LeftFin
/Drone/FwdFin                     (RigidBody, sibling of Body — canonical +X forward)
/Drone/RightFin                   (canonical +Y right)
/Drone/AftFin                     (canonical -X aft)
/Drone/LeftFin                    (canonical -Y left)
```

**Rules:**

- `Body` must carry `UsdPhysics.ArticulationRootAPI` (not the root Xform).
- Fin links are **siblings** of `Body`, not children.
- Fin joints live as **children of `Body`**, pointing to their respective fin link.
- Joints must be `UsdPhysics.RevoluteJoint` — no fixed/prismatic joints for fins.

---

## 2. Asset Metadata YAML

**File:** `assets/metadata/edf_drone_v2.asset.yaml`
This is the single source of truth for USD structure expected at runtime.

Required keys:


| Key                             | Value                                                                                                      |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `body_link_name`                | `Body`                                                                                                     |
| `fin_link_names`                | `[FwdFin, RightFin, AftFin, LeftFin]`                                                                      |
| `fin_joint_names`               | `[joint_FwdFin, joint_RightFin, joint_AftFin, joint_LeftFin]`                                              |
| `hinge_axes` (4×3 unit vectors) | FwdFin: `[0,1,0]` / RightFin: `[-1,0,0]` / AftFin: `[0,-1,0]` / LeftFin: `[1,0,0]`                         |
| `joint_limits` (rad)            | `[-0.262, 0.262]` × 4 (±15°)                                                                               |
| `fin_cop_positions` (4×3, m)    | FwdFin: `[0.04,0,0.10]` / RightFin: `[0,0.04,0.10]` / AftFin: `[-0.04,0,0.10]` / LeftFin: `[0,-0.04,0.10]` |


---

## 3. Physical Properties (Vehicle Config)

**File:** `configs/vehicle/edf_drone_v2.yaml`


| Property           | Value                                 |
| ------------------ | ------------------------------------- |
| Total mass         | 3.1 kg                                |
| Inertia Ixx        | 0.0486 kg·m²                          |
| Inertia Iyy        | 0.0438 kg·m²                          |
| Inertia Izz        | 0.0202 kg·m²                          |
| Body COM offset    | `[0.0, 0.0, 0.01]` m (body-FRD frame) |
| Max fin deflection | ±0.262 rad (15°)                      |


USD `UsdPhysics.MassAPI` values on `Body` must match within **±1%** of these figures or validation will fail.

---

## 4. Isaac Lab ArticulationCfg

**Configured in:** `tvc_env/sim/scene_builder.py`

```python
ArticulationCfg(
    spawn=UsdFileCfg(usd_path="assets/usd/drone_v2_physics.usd"),
    prim_path="{ENV_REGEX_NS}/drone",          # templated — cloned per env
    articulation_root_prim_path="/Body",        # within defaultPrim

    rigid_props=RigidBodyPropertiesCfg(
        disable_gravity=False,                 # PhysX applies gravity; thrust/aero forces applied explicitly
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=100.0,             # m/s
        max_angular_velocity=100.0,            # rad/s
        max_depenetration_velocity=1.0,
    ),

    articulation_props=ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=4,
        solver_velocity_iteration_count=0,
    ),

    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=["joint_.*"],
            stiffness=400.0,                   # N/rad — servo stiffness
            damping=40.0,                      # N·s/rad — servo damping
        ),
    },

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 5.0),                  # 5 m above ground (Z-up stage)
        joint_pos={"joint_.*": 0.0},
    ),
)
```

---

## 5. Scene Builder Pipeline

**Entry point:** `tvc_env/sim/scene_builder.py → build_scene(config)`

```
SceneConfig
   │
   ├─ SimulationContext(SimulationCfg(physics_dt=1/120))
   │
   ├─ InteractiveSceneCfg
   │     └─ drone = ArticulationCfg (see §4)
   │
   ├─ InteractiveScene(cfg, num_envs, env_spacing, replicate_physics=True)
   │     ├─ Clones drone prim into /env_0/drone … /env_N/drone
   │     ├─ Enables GPU pipeline physics replication
   │     └─ Filters inter-environment collisions
   │
   ├─ sim.reset() + scene.reset()
   │
   └─ TVCSimScene  ←─ returned to caller
```

`**TVCSimScene.step()` each tick:**

```
scene.write_data_to_sim()   # flush joint position targets
sim.step(render=False)      # one physics substep (8.33 ms)
scene.update(physics_dt)    # refresh articulation buffers
```

---

## 6. SceneConfig Parameters

```python
@dataclass
class SceneConfig:
    num_envs: int = 1
    env_spacing: float = 4.0          # metres between env origins
    replicate_physics: bool = True     # GPU pipeline / physics cloning
    physics_dt: float = 1 / 120       # substep = 8.33 ms
    decimation: int = 4               # RL step = 4 substeps = 33 ms
    dispatch_mode: str = "per_link_force"
    drone_usd_path: str = "assets/usd/drone_v2_physics.usd"
```

---

## 6b. Coordinate Frame Convention

**Isaac Sim stage: Z-up** (confirmed in Preferences → Stage → Default Up Axis = Z)

```
World frame (Z-up, right-handed):   X = forward,  Y = left,   Z = up
Body FRD frame:                      X = forward,  Y = right,  Z = down

FRD ↔ World mapping (drone at neutral/hover heading):
  +X_body  =  +X_world  (forward)
  +Y_body  =  −Y_world  (right in body = −Y in world)
  +Z_body  =  −Z_world  (down in body = −Z in world)

Fin positions in world space (drone facing +X_world):
  RightFin (+Y_body) → −Y_world   (viewer's right when looking from front)
  LeftFin  (−Y_body) → +Y_world   (viewer's left  when looking from front)

Spawn height: pos=(0, 0, 5) → Z=5 m above ground  ✓
              NOT (0, 5, 0) which would be 5 m sideways in Y  ✗
```

---

## 7. Validation Pipeline

Run **offline** (no Isaac Sim headless needed) or **full** (with runtime):

### Offline checks (`asset_validator.py`)

- Metadata completeness: all required keys present
- Exactly 4 fins, 4 joints, 4 hinge axes, 4 COP positions
- Hinge axes are unit vectors (tolerance ±0.01)
- Joint limits within ±0.01 rad of `max_deflection` in vehicle config

### Full USD checks (stage required)

- Body prim exists at expected path
- Body carries `UsdPhysics.ArticulationRootAPI`
- All 4 fin prims exist as siblings of Body
- All 4 joint prims exist as children of Body
- Joints are `UsdPhysics.RevoluteJoint`

### Articulation checks (runtime articulation required)

- Fin link names present in `articulation.body_names`
- Fin joint names present in `articulation.joint_names`

### Mass properties check (`asset/mass_properties.py`)

- USD `UsdPhysics.MassAPI` mass vs config: ≤ 1% delta
- USD diagonal inertia vs config tensor: ≤ 1% delta per axis

---

## 8. Test Ladder (ordered)


| Test                              | What it validates                                  | Isaac required?                             |
| --------------------------------- | -------------------------------------------------- | ------------------------------------------- |
| `test_00_asset_validation`        | Metadata + USD prim structure                      | Partial (offline fixtures in `conftest.py`) |
| `test_01_joint_axes`              | Each fin rotates around its correct hinge axis     | Yes                                         |
| `test_02_single_fin_articulation` | Commanded angle matches actual within tolerance    | Yes                                         |
| `test_03_unit_force_on_fin`       | Unit force at COP produces correct torque          | Yes                                         |
| `test_04_fin_force_sweep`         | Force/deflection curve behavior                    | Yes                                         |
| `test_05_four_fin_superposition`  | Roll/pitch patterns produce correct net torques    | Yes                                         |
| `test_06_edf_spool_and_reaction`  | Motor spool time constant + static reaction torque | Yes                                         |


**Run a single test:**

```bash
python apps/run_single_test.py --test test_00_asset_validation
python apps/run_single_test.py --test test_01_joint_axes
```

---

## 9. Offline Test Fixtures (`conftest.py`)

Tests that don't need Isaac Sim use an in-memory USD stage built from metadata:

```python
@pytest.fixture
def usd_stage(asset_metadata):
    stage = Usd.Stage.CreateInMemory()
    # Creates /Drone (Xform, defaultPrim)
    # Creates /Drone/Body (with ArticulationRootAPI, MassAPI)
    # Creates /Drone/<FinName> × 4 (siblings)
    # Creates /Drone/Body/<joint_FinName> × 4 (RevoluteJoints)
    return stage
```

This lets `test_00_asset_validation` run without a headless Isaac Sim instance.

---

## 10. Common Mistakes to Avoid


| Mistake                                                          | Symptom                                    | Fix                                              |
| ---------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------ |
| `ArticulationRootAPI` on `/Drone` (root Xform) instead of `Body` | Articulation not found at runtime          | Apply API to `Body` prim only                    |
| Fin links as children of `Body`                                  | Joints form a tree → wrong DOF count       | Make fins **siblings** of Body                   |
| Joints as children of fins (not Body)                            | Joint parent/child reversed → wrong motion | Joints must be under `Body`                      |
| Gravity accidentally disabled                                    | Drone floats / flies up at any throttle    | `disable_gravity=False` in RigidBodyPropertiesCfg |
| Physics USD not checked offline                                  | Silent mass/inertia mismatch               | Always run `test_00` before runtime tests        |
| `prim_path` without `{ENV_REGEX_NS}`                             | Multi-env cloning fails                    | Use `"{ENV_REGEX_NS}/drone"` pattern             |
| Wrong USD file (geometry-only export)                            | No `ArticulationRootAPI` found             | Use `drone_v2_physics.usd`, not a mesh export    |


---

## 11. File Reference


| Purpose                        | Path                                      |
| ------------------------------ | ----------------------------------------- |
| Scene builder                  | `tvc_env/sim/scene_builder.py`            |
| USD loader                     | `tvc_env/asset/usd_loader.py`             |
| Asset validator                | `tvc_env/asset/asset_validator.py`        |
| Mass properties                | `tvc_env/asset/mass_properties.py`        |
| Fin geometry / COP             | `tvc_env/dynamics/fin_geometry.py`        |
| Asset metadata (authoritative) | `assets/metadata/edf_drone_v2.asset.yaml` |
| Vehicle physical properties    | `configs/vehicle/edf_drone_v2.yaml`       |
| Offline USD fixtures           | `tests/sim/conftest.py`                   |
| Test runner                    | `apps/run_single_test.py`                 |
| Commands reference             | `../Isaac_Commands.md`                    |


