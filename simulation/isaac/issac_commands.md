# Isaac / IsaacLab commands

Activate env first (PowerShell, repo root):

```
.\env_isaaclab\Scripts\activate
```

Then run any command below. Use `.\env_isaaclab\Scripts\python.exe` if you're not in the activated env.

---

## Setup — launch Isaac Sim GUI

```
.\env_isaaclab\Scripts\isaacsim.exe
```

---

## Validate

Validate drone USD (no Isaac Sim):

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.usd.drone_builder validate --usd simulation/isaac/usd/drone.usdc
```

Mass properties vs YAML:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.validate_mass_props --usd simulation/isaac/usd/drone.usdc --config simulation/configs/default_vehicle.yaml --tolerance 0.01
```

---

## Post-process drone USD

Add physics APIs (articulation, rigid body, mass, fin joints) to a Blender-exported USD. Fin mass is set to zero; body CoM is (0, 0, z) with z from body bbox.

**Generic (Blender export → physics USD):**

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.usd.postprocess_usd --input simulation/isaac/usd/drone_blender.usd --output simulation/isaac/usd/drone.usd --config simulation/configs/default_vehicle.yaml
```

**drone_v2 (usdc → physics usdc):**

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.usd.postprocess_usd --input simulation/isaac/usd/drone_v2.usdc --output simulation/isaac/usd/drone_v2_physics.usdc --config simulation/configs/default_vehicle.yaml
```

**Validate-only (no output):**

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.usd.postprocess_usd --input simulation/isaac/usd/drone_v2.usdc --validate-only
```

---

## Diagnostics — single env

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_isaac_single --config simulation/isaac/configs/isaac_env_single.yaml
```

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_isaac_single --config simulation/isaac/configs/isaac_env_128.yaml
```

---

## Fin articulation

Test fins:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.test_fins --config simulation/isaac/configs/isaac_env_single.yaml
```

Fin wiggle (single / 128):

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_fin_wiggle --config simulation/isaac/configs/isaac_env_single.yaml
```

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_fin_wiggle --config simulation/isaac/configs/isaac_env_128.yaml
```

Thrust + fin wiggle (fixed altitude):

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_thrust_fin_wiggle --thrust 0.75 --max-deflection 0.1 --fixed-altitude
```

Thrust + fin, reduced physics (no wind/gyro/anti-torque):

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_thrust_fin_wiggle --fixed-altitude --disable-wind --disable-gyro --disable-anti-torque --thrust 0.7 --max-deflection 1.0 --hold-secs 1.0
```

Thrust + fin + override inertia:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_thrust_fin_wiggle --thrust 0.75 --max-deflection 0.1 --fixed-altitude --override-inertia 0.1 0.1 0.1 --hold-secs 1.0
```

---

## Thrust test

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_thrust_test --config simulation/isaac/configs/isaac_env_single.yaml
```

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_thrust_test --config simulation/isaac/configs/isaac_env_128.yaml
```

---

## Wind test

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.diag_wind --config simulation/isaac/configs/isaac_env_single.yaml
```

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

## PID tuning — baseline (episodes)

Single env, 100 episodes:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.tune_pid_isaac --config simulation/isaac/configs/isaac_env_single.yaml --pid-config simulation/configs/pid.yaml --episodes 100
```

128 envs, 100 episodes:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.tune_pid_isaac --config simulation/isaac/configs/isaac_env_128.yaml --pid-config simulation/configs/pid.yaml --episodes 100
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

128 envs, same:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.tune_pid_isaac --config simulation/isaac/configs/isaac_env_128.yaml --pid-config simulation/configs/pid.yaml --test hover --disable-wind --disable-gyro --disable-anti-torque --hover-altitude 5.0 --hover-alt-tol 0.5 --episodes 50 --seed 0 --output-dir runs/pid_isaac_hover_5m --log-dir runs/pid_isaac_hover_5m/logs
```

Single env, 1 ep, z+roll+pitch only (no yaw, no lateral-x):

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.tune_pid_isaac --config simulation/isaac/configs/isaac_env_single.yaml --pid-config simulation/configs/pid.yaml --test hover --disable-wind --disable-gyro --disable-anti-torque --disable-yaw --disable-lateral-x --hover-altitude 5.0 --hover-alt-tol 0.5 --episodes 1 --seed 0 --output-dir runs/pid_isaac_hover_5m --log-dir runs/pid_isaac_hover_5m/logs
```

Z+roll+pitch only, 10 ep (output to pid_isaac_hover_zrp):

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.tune_pid_isaac --config simulation/isaac/configs/isaac_env_single.yaml --pid-config simulation/configs/pid.yaml --test hover --disable-wind --disable-gyro --disable-anti-torque --disable-yaw --disable-lateral-x --disable-lateral-y --hover-altitude 5.0 --hover-alt-tol 0.5 --episodes 10 --seed 0 --output-dir runs/pid_isaac_hover_zrp --log-dir runs/pid_isaac_hover_zrp/logs
```

128 envs hover, 2048 ep, no wind/gyro:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.tune_pid_isaac --config simulation/isaac/configs/isaac_env_128.yaml --pid-config simulation/configs/pid.yaml --test hover --hover-altitude 5.0 --hover-alt-tol 0.5 --episodes 2048 --seed 0 --disable-wind --disable-gyro --output-dir runs/pid_isaac_hover_grid_128 --log-dir runs/pid_isaac_hover_grid_128/logs
```

Single env hover, no gyro/anti-torque, 10 ep:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.tune_pid_isaac --config simulation/isaac/configs/isaac_env_single.yaml --pid-config simulation/configs/pid.yaml --test hover --disable-gyro --disable-anti-torque --hover-altitude 5.0 --hover-alt-tol 0.5 --episodes 10 --seed 0 --output-dir runs/pid_isaac_hover_zrp --log-dir runs/pid_isaac_hover_zrp/logs
```

---

## PID tuning — ZN (Ziegler–Nichols) hover

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

All loops (roll + pitch + altitude), with verify:

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.tune_pid_isaac --config simulation/isaac/configs/isaac_env_single.yaml --pid-config simulation/configs/pid.yaml --test hover --zn-loop all --disable-wind --disable-gyro --disable-anti-torque --hover-altitude 5.0 --hover-alt-tol 0.5 --zn-kp-start 0.05 --zn-kp-stop 20.0 --zn-kp-steps 24 --zn-perturb-angle-deg 3.0 --zn-altitude-offset 0.3 --zn-max-seconds 12.0 --episodes 8 --zn-verify-episodes 16 --output-dir runs/pid_isaac_zn_all --log-dir runs/pid_isaac_zn_all/logs
```

Relay autotune (pitch):

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.tune_pid_isaac --method relay-autotune --zn-loop pitch --disable-wind --disable-gyro --disable-anti-torque
```

---

## Omega / rotation test (zero-g)

```
.\env_isaaclab\Scripts\python.exe -m simulation.isaac.scripts.tune_pid_isaac --test rotation --config simulation/isaac/configs/isaac_env_single.yaml --pid-config simulation/configs/pid.yaml --disable-gravity --disable-wind --disable-gyro --disable-anti-torque
```

