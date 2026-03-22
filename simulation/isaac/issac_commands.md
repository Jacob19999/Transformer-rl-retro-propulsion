# Isaac / IsaacLab commands

Activate env first (PowerShell, repo root):

```
.\env_isaaclab\Scripts\activate
```

> **Debug tip**: add `--headless 2>&1 | Tee-Object diag.log` to any Isaac command to capture the full crash traceback without the GUI.

---

## Rebuild & Test sequence

Run these in order after any config or code change. Each step is independent — run from repo root with env activated.

### Step 0 — Pure Python tests (no Isaac Sim needed)

```
python -m pytest simulation/tests/ -q --tb=short
```

Expected: 117 passed, 2 skipped.

---

### Step 1 — Rebuild USD asset

Postprocess Blender export with updated YAML config (mass/CoM/inertia, fin joints, colliders):

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.usd.postprocess_usd --input simulation/isaac/usd/drone_v2.usdc --output simulation/isaac/usd/drone_v2_physics.usd --config simulation/configs/default_vehicle.yaml
```

Check the log for:
- `Mass authored: 3.13 kg`
- `CoM: (0.0, 0.0, 0.0)`
- `DiagonalInertia: (0.015, 0.015, 0.005)`
- `Fin colliders: convexHull` (×4)
- `Fin joints: localRot flip applied` (×4)

---

### Step 2 — Validate mass properties

Confirms USD ↔ YAML agreement within 1%:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.validate_mass_props --usd simulation/isaac/usd/drone_v2_physics.usd --config simulation/configs/default_vehicle.yaml --tolerance 0.01
```

Expected: PASS on mass (3.13 kg), CoM (0,0,0), diagonal inertia (0.015, 0.015, 0.005).

---

### Step 3 — Fin articulation smoke test

Verifies the four fin joints are found and moveable:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.test_fins --config simulation/isaac/configs/isaac_env_single.yaml
```

---

### Step 4 — Fin wiggle (sign convention + solver stability)

100 episodes, 3 sweeps — each fin should deflect in the correct direction, zero solver warnings:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_fin_wiggle --config simulation/isaac/configs/isaac_env_single.yaml --episodes 100 --sweeps 3
```

---

### Step 5 — Thrust test (liftoff validation)

Spawn at 0.4 m, full thrust — should lift off vertically with no lateral drift (centered CoM):

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_thrust_test --config simulation/isaac/configs/isaac_env_single.yaml --thrust 1.0 --duration 2.0 --spawn-alt 0.4
```

---

### Step 6 — Wind disturbance (no lateral bias)

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_wind --config simulation/isaac/configs/isaac_env_single.yaml --wind-x 5.0 --wind-y 0.0 --duration 3.0
```

---

### Step 7 — Isaac pytest

```
.\env_isaaclab\Scripts\python.exe -m pytest simulation/tests/ -q --tb=short -m isaac simulation/tests/test_isaac_env.py simulation/tests/test_drone_builder.py
```

---

## Launch Isaac Sim GUI

```
.\env_isaaclab\Scripts\isaacsim.exe
```

---

## Validate USD only (no physics run)

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.usd.postprocess_usd --input simulation/isaac/usd/drone_v2.usdc --validate-only
```

---

## Diagnostics — single env

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_isaac_single --config simulation/isaac/configs/isaac_env_single.yaml
```

128 envs:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_isaac_single --config simulation/isaac/configs/isaac_env_128.yaml
```

---

## Fin articulation

Fin wiggle — 128 envs:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_fin_wiggle --config simulation/isaac/configs/isaac_env_128.yaml
```

Thrust + fin wiggle (fixed altitude):

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_thrust_fin_wiggle --thrust 0.75 --max-deflection 0.1 --fixed-altitude
```

Thrust + fin, no wind/gyro/anti-torque:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_thrust_fin_wiggle --fixed-altitude --disable-wind --disable-gyro --disable-anti-torque --thrust 0.7 --max-deflection 1.0 --hold-secs 1.0
```

Thrust + fin + override inertia:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_thrust_fin_wiggle --thrust 0.75 --max-deflection 0.1 --fixed-altitude --override-inertia 0.1 0.1 0.1 --hold-secs 1.0
```

### Fin telemetry (per-fin debug recording)

Set `debug.fin_telemetry: true` and `debug.fin_telemetry_save: true` in `simulation/isaac/configs/isaac_env_single.yaml`, then run any diagnostic. Output saved to `runs/telemetry/`.

Load and inspect saved telemetry:

```python
import torch
data = torch.load("runs/telemetry/ep0.pt")
print(data.keys())               # cmd_angle, meas_angle, link_pos, aoa, aero_force, ...
print(data["meas_angle"].shape)  # (steps, num_envs, 4)
```

---

## Fin mapping calibration

Excites each fin, measures Δω per axis, writes `simulation/configs/fin_mapping.yaml`.

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.calibrate_fin_mapping --config simulation/isaac/configs/isaac_env_single.yaml
```

With force gizmos in viewport (GUI only — no `--headless`):

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.calibrate_fin_mapping --config simulation/isaac/configs/isaac_env_single.yaml --draw-forces
```

Colours: cyan = thrust, red/green/blue/yellow = fin aero (fins 0–3), orange = body torque.

---

## Thrust test — 128 envs

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_thrust_test --config simulation/isaac/configs/isaac_env_128.yaml
```

---

## Wind test — 128 envs

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_wind --config simulation/isaac/configs/isaac_env_128.yaml
```

---

## Yaw test

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_yaw_isaac --config simulation/isaac/configs/isaac_env_single.yaml
```

---

## Gyro precession (fin hold)

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_gyro_precession --mode fin_hold --torque-axis pitch --thrust 0.68 --fin-deflection 0.5 --duration 3.0
```

---

## Reaction torque

Constant anti-torque:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_reaction_torque --mode constant --thrust 0.68 --duration 3.0
```

RPM ramp:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_reaction_torque --mode ramp --ramp-duration 1.0 --duration 3.0
```

---

## PID tuning — baseline

Single env, 100 episodes:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.tune_pid_isaac --config simulation/isaac/configs/isaac_env_single.yaml --pid-config simulation/configs/pid.yaml --episodes 100
```

128 envs, 2048 episodes, with logs:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.tune_pid_isaac --config simulation/isaac/configs/isaac_env_128.yaml --pid-config simulation/configs/pid.yaml --episodes 2048 --seed 0 --output-dir runs/pid_isaac_128 --log-dir runs/pid_isaac_128/logs
```

---

## PID tuning — hover test

Single env, 5 m hover, no wind/gyro/anti-torque (50 ep):

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.tune_pid_isaac --config simulation/isaac/configs/isaac_env_single.yaml --pid-config simulation/configs/pid.yaml --test hover --disable-wind --disable-gyro --disable-anti-torque --hover-altitude 5.0 --hover-alt-tol 0.5 --episodes 50 --seed 0 --output-dir runs/pid_isaac_hover_5m --log-dir runs/pid_isaac_hover_5m/logs
```

128 envs, 2048 ep, no wind/gyro:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.tune_pid_isaac --config simulation/isaac/configs/isaac_env_128.yaml --pid-config simulation/configs/pid.yaml --test hover --hover-altitude 5.0 --hover-alt-tol 0.5 --episodes 2048 --seed 0 --disable-wind --disable-gyro --output-dir runs/pid_isaac_hover_grid_128 --log-dir runs/pid_isaac_hover_grid_128/logs
```

Z+roll+pitch only, no yaw/lateral, 10 ep:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.tune_pid_isaac --config simulation/isaac/configs/isaac_env_single.yaml --pid-config simulation/configs/pid.yaml --test hover --disable-wind --disable-gyro --disable-anti-torque --disable-yaw --disable-lateral-x --disable-lateral-y --hover-altitude 5.0 --hover-alt-tol 0.5 --episodes 10 --seed 0 --output-dir runs/pid_isaac_hover_zrp --log-dir runs/pid_isaac_hover_zrp/logs
```

---

## PID tuning — ZN (Ziegler–Nichols)

Roll loop:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.tune_pid_isaac --config simulation/isaac/configs/isaac_env_single.yaml --pid-config simulation/configs/pid.yaml --test hover --zn-loop roll --disable-wind --disable-gyro --disable-anti-torque --hover-altitude 5.0 --hover-alt-tol 0.5 --zn-kp-start 0.05 --zn-kp-stop 20.0 --zn-kp-steps 24 --zn-perturb-angle-deg 3.0 --zn-max-seconds 12.0 --episodes 8 --output-dir runs/pid_isaac_zn_roll --log-dir runs/pid_isaac_zn_roll/logs
```

Pitch loop:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.tune_pid_isaac --config simulation/isaac/configs/isaac_env_single.yaml --pid-config simulation/configs/pid.yaml --test hover --zn-loop pitch --disable-wind --disable-gyro --disable-anti-torque --hover-altitude 5.0 --hover-alt-tol 0.5 --zn-kp-start 0.05 --zn-kp-stop 20.0 --zn-kp-steps 24 --zn-perturb-angle-deg 3.0 --zn-max-seconds 12.0 --episodes 8 --output-dir runs/pid_isaac_zn_pitch --log-dir runs/pid_isaac_zn_pitch/logs
```

Altitude loop:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.tune_pid_isaac --config simulation/isaac/configs/isaac_env_single.yaml --pid-config simulation/configs/pid.yaml --test hover --zn-loop altitude --disable-wind --disable-gyro --disable-anti-torque --hover-altitude 5.0 --hover-alt-tol 0.5 --zn-kp-start 0.05 --zn-kp-stop 20.0 --zn-kp-steps 24 --zn-altitude-offset 0.3 --zn-max-seconds 12.0 --episodes 8 --output-dir runs/pid_isaac_zn_altitude --log-dir runs/pid_isaac_zn_altitude/logs
```

All loops + verify:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.tune_pid_isaac --config simulation/isaac/configs/isaac_env_single.yaml --pid-config simulation/configs/pid.yaml --test hover --zn-loop all --disable-wind --disable-gyro --disable-anti-torque --hover-altitude 5.0 --hover-alt-tol 0.5 --zn-kp-start 0.05 --zn-kp-stop 20.0 --zn-kp-steps 24 --zn-perturb-angle-deg 3.0 --zn-altitude-offset 0.3 --zn-max-seconds 12.0 --episodes 8 --zn-verify-episodes 16 --output-dir runs/pid_isaac_zn_all --log-dir runs/pid_isaac_zn_all/logs
```

---

## Omega / rotation test (zero-g)

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.tune_pid_isaac --test rotation --config simulation/isaac/configs/isaac_env_single.yaml --pid-config simulation/configs/pid.yaml --disable-gravity --disable-wind --disable-gyro --disable-anti-torque
```

---

## PPO training

256 envs (RTX 5070 safe):

```
.\env_isaaclab\Scripts\python.exe -m simulation.training.scripts.train_isaac_ppo --config simulation/isaac/configs/isaac_env_training.yaml --seed 0
```
