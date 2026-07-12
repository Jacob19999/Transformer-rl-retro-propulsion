# Isaac Sim — TVC Environment Commands

All commands are run from `simulation/isaac/` unless noted otherwise.

---

## 0. Prerequisites

```powershell
. .\activate-isaaclab.ps1
.\env_isaaclab\Scripts\isaacsim.exe
```

### Isaac Sim GUI startup

```powershell
# Interactive viewport (requires Isaac Sim 5.1.0 install)
python apps/run_single_env_debug.py --task hover

# Headless (no window) — for CI / SSH sessions
python apps/run_single_test.py --test test_00_asset_validation --headless
```

> These examples require a Python interpreter where `import isaacsim` works (i.e., Isaac Sim / IsaacLab's Python).
>
> - Linux/macOS: the bundled launcher is often `~/.local/share/ov/pkg/isaac-sim-5.1.0/python.sh`.
> - Windows PowerShell: activate IsaacLab (or your Isaac Sim) environment and use `python ...` (no `./python.sh`).
>
> Quick Windows PowerShell (from `simulation/isaac/`, no activation):
>
> ```powershell
> & "..\env_isaaclab\Scripts\python.exe" apps/run_single_test.py --test test_00_asset_validation --headless
> ```

---

## 1. Unit Tests (no Isaac Sim required)

Run offline — no GPU / simulator needed. All 81 tests should pass.

### Run all unit tests

```powershell
$env:PYTHONPATH = "."; python -m pytest tests/unit/ -v
```

> If `pip install -e .` succeeded, plain `python -m pytest tests/unit/ -v` also works.
> The `$env:PYTHONPATH = "."` line is the safe default when editable install is unavailable.

### Run individual unit test files


| Test file                           | What it covers                                                  | Count |
| ----------------------------------- | --------------------------------------------------------------- | ----- |
| `tests/unit/test_frames.py`         | FRD↔Isaac frame round-trips, known vector transforms, batch ops | 11    |
| `tests/unit/test_quaternions.py`    | (w,x,y,z) multiply, rotate, Euler, convention converters        | 17    |
| `tests/unit/test_fin_geometry.py`   | COP positions, hinge axes, fin-local-to-body transforms         | 14    |
| `tests/unit/test_fin_aero.py`       | Semi-empirical aero model: C_N linearity, C_D, saturation       | 11    |
| `tests/unit/test_rotor_reaction.py` | Static Q torque, spool torque sign, gyro precession direction   | 13    |
| `tests/unit/test_crash_logic.py`    | Each crash criterion isolated, below-threshold, vectorized      | 15    |


```powershell
$env:PYTHONPATH = "."; python -m pytest tests/unit/test_frames.py -v
$env:PYTHONPATH = "."; python -m pytest tests/unit/test_quaternions.py -v
$env:PYTHONPATH = "."; python -m pytest tests/unit/test_fin_geometry.py -v
$env:PYTHONPATH = "."; python -m pytest tests/unit/test_fin_aero.py -v
$env:PYTHONPATH = "."; python -m pytest tests/unit/test_rotor_reaction.py -v
$env:PYTHONPATH = "."; python -m pytest tests/unit/test_crash_logic.py -v
```

---

## 2. Debug Visualization (Isaac Sim GUI — interactive)

Opens the Isaac Sim viewport with all debug gizmos: force arrows, body axes, HUD telemetry.

```bash
python apps/run_single_env_debug.py [OPTIONS]
```


| Argument                   | Default                             | Description                                     |
| -------------------------- | ----------------------------------- | ----------------------------------------------- |
| `--task`                   | `hover`                             | Task to run: `hover`                            |
| `--env-config`             | `configs/env/single_env_debug.yaml` | Env config YAML path                            |
| `--disturbance`            | `configs/disturbances/nominal.yaml` | Disturbance config YAML path                    |
| `--override KEY=VALUE ...` | *(none)*                            | Runtime config overrides, e.g. `env.num_envs=1` |


**Examples:**

```bash
# Hover with no disturbance (default)
python apps/run_single_env_debug.py --task hover

# Landing task with wind
python apps/run_single_env_debug.py --task landing --disturbance configs/disturbances/wind.yaml

# Override specific param
python apps/run_single_env_debug.py --task hover --override env.episode_length_s=60
```

---

## 3. Simulation Validation Ladder (Isaac Sim required)

### Run a single sim test

```bash
python apps/run_single_test.py --test <TEST_NAME> [OPTIONS]
```


| Argument        | Default      | Description                                                           |
| --------------- | ------------ | --------------------------------------------------------------------- |
| `--test`        | *(required)* | Test module name from `tests/sim/` (without `.py`)                    |
| `--physics`     | `None`       | Path to PhysX solver config YAML override                             |
| `--headless`    | *(on)*       | Run without a viewport (default). Same as omitting `--no-headless`.   |
| `--no-headless` | —            | Open the Isaac Sim Kit viewport (interactive UI) while the test runs. |


### Run all sim tests via pytest

```bash
pytest tests/sim/ -v
```

### Run scripted visual validation scenarios

Use these when you want to visually confirm the qualitative behavior behind the
assertion-based tests with HUD + terminal state vectors and gizmo force arrows.

```powershell
# Single scenario: fin wiggle / sweep
python apps/run_visual_test.py --scenario fin_sweep

# Single scenario: EDF spool, reaction torque, and gyro coupling
python apps/run_visual_test.py --scenario edf_spool_gyro

# Single scenario: wind drop / displacement check
python apps/run_visual_test.py --scenario wind_drop

# Run the full visual suite (100-step episodes)
python apps/run_visual_test.py --scenario all --episode-steps 100

# Same suite, slower playback with viewport open
python apps/run_visual_test.py --scenario all --episode-steps 100 --step-sleep 0.10

# Headless run with terminal-only state-vector output
python apps/run_visual_test.py --scenario all --episode-steps 100 --headless

# Reduce terminal spam: print vectors every 5 steps
python apps/run_visual_test.py --scenario edf_spool_gyro --episode-steps 100 --print-every 5
```

### Validation ladder — tests in order (headless — default)

Append nothing, or pass `--headless` explicitly:


| Test      | Name                    | What it tests                                                                    | Headless command                                                        |
| --------- | ----------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `test_00` | Asset validation        | USD metadata, fin count, joint axes (offline portion runs without Isaac)         | `python apps/run_single_test.py --test test_00_asset_validation`        |
| `test_01` | Joint axes              | Each fin hinge rotates around correct axis at correct sign                       | `python apps/run_single_test.py --test test_01_joint_axes`              |
| `test_02` | Single fin articulation | Commanded fin angle matches joint position within tolerance                      | `python apps/run_single_test.py --test test_02_single_fin_articulation` |
| `test_03` | Unit force on fin       | Body reaction direction matches r × F for each COP                               | `python apps/run_single_test.py --test test_03_unit_force_on_fin`       |
| `test_04` | Fin force sweep         | F_n vs deflection angle: near-zero at zero, sign-correct, saturates              | `python apps/run_single_test.py --test test_04_fin_force_sweep`         |
| `test_05` | Four-fin superposition  | Coordinated roll-only / pitch-only patterns produce correct net torque           | `python apps/run_single_test.py --test test_05_four_fin_superposition`  |
| `test_06` | EDF spool & reaction    | Motor spool time constant within 10% of τ=0.15 s, static Q opposes spin          | `python apps/run_single_test.py --test test_06_edf_spool_and_reaction`  |
| `test_07` | Gyro precession         | τ_gyro = ω_body × H_rotor direction and magnitude                                | `python apps/run_single_test.py --test test_07_gyro_precession`         |
| `test_08` | Wind disturbance        | Drift aligns with wind, drag opposes airflow, gust is transient                  | `python apps/run_single_test.py --test test_08_wind_disturbance`        |
| `test_09` | Contact state machine   | Soft land → LANDED, bounce → AIRBORNE, hard impact → CRASHED, tip-over → CRASHED | `python apps/run_single_test.py --test test_09_contact_landed_crash`    |
| `test_10` | PID hover smoke         | 10 s hover: pos err < 0.5 m, tilt < 15°, ang rate < 1 rad/s, no NaN              | `python apps/run_single_test.py --test test_10_pid_hover_smoke`         |
| `test_11` | 128-env RL API smoke    | 128 envs × 1000 steps: shape (128,24), no NaN, independent resets                | `python apps/run_single_test.py --test test_11_rl_api_128env_smoke`     |
| `test_12` | Steady hover all forces | All torque contributions (fin/static Q/spool/gyro/wind) bounded under wind       | `python apps/run_single_test.py --test test_12_steady_hover_all_forces` |


### Validation ladder — same steps with UI (viewport open)

Add `--no-headless` so Kit opens a window (slower; use when you need to watch physics / scene):

```powershell
cd simulation/isaac/
python apps/run_single_test.py --test test_00_asset_validation --no-headless
python apps/run_single_test.py --test test_01_joint_axes --no-headless
python apps/run_single_test.py --test test_02_single_fin_articulation --no-headless
python apps/run_single_test.py --test test_03_unit_force_on_fin --no-headless
python apps/run_single_test.py --test test_04_fin_force_sweep --no-headless
python apps/run_single_test.py --test test_05_four_fin_superposition --no-headless
python apps/run_single_test.py --test test_06_edf_spool_and_reaction --no-headless
python apps/run_single_test.py --test test_07_gyro_precession --no-headless
python apps/run_single_test.py --test test_08_wind_disturbance --no-headless
python apps/run_single_test.py --test test_09_contact_landed_crash --no-headless
python apps/run_single_test.py --test test_10_pid_hover_smoke --no-headless
python apps/run_single_test.py --test test_11_rl_api_128env_smoke --no-headless
python apps/run_single_test.py --test test_12_steady_hover_all_forces --no-headless
```

**With high-fidelity solver:**

```bash
python apps/run_single_test.py --test test_06_edf_spool_and_reaction \
    --physics configs/physics/solver_high_fidelity.yaml
```

**Same with UI:**

```powershell
python apps/run_single_test.py --test test_06_edf_spool_and_reaction `
    --physics configs/physics/solver_high_fidelity.yaml --no-headless
```

---

## 4. PID Hover Evaluation

Runs PID controller for a configurable duration and reports stability statistics.

```bash
python apps/run_eval_pid.py [OPTIONS]
```


| Argument                       | Default                             | Description                                |
| ------------------------------ | ----------------------------------- | ------------------------------------------ |
| `--task`                       | `hover`                             | Task: `hover`                              |
| `--env-config`                 | `configs/env/single_env_debug.yaml` | Env config YAML path                       |
| `--disturbance`                | `None`                              | Disturbance config YAML (omit for nominal) |
| `--duration`                   | `30.0`                              | Evaluation duration in seconds             |
| `--headless` / `--no-headless` | headless                            | Whether to open the viewport               |


**Examples:**

```bash
# 30 s nominal hover
python apps/run_eval_pid.py --task hover --duration 30

# 60 s with wind disturbance
python apps/run_eval_pid.py --task hover \
    --disturbance configs/disturbances/wind.yaml --duration 60

# With viewport open
python apps/run_eval_pid.py --no-headless --duration 30
```

---

## 5. 128-Environment Smoke Test

Validates vectorized env performance and tensor correctness at training scale.

```bash
python apps/run_smoke_128.py [OPTIONS]
```


| Argument                   | Default                      | Description                               |
| -------------------------- | ---------------------------- | ----------------------------------------- |
| `--task`                   | `hover`                      | Task: `hover`                             |
| `--env-config`             | `configs/env/train_128.yaml` | Env config YAML (128 envs by default)     |
| `--steps`                  | `1000`                       | Number of RL steps to run                 |
| `--override KEY=VALUE ...` | *(none)*                     | Runtime overrides, e.g. `env.num_envs=64` |


**Examples:**

```bash
# Quick 1000-step smoke test
python apps/run_smoke_128.py --task hover

# 5000-step with custom env count
python apps/run_smoke_128.py --steps 5000 --override env.num_envs=64

# Landing task
python apps/run_smoke_128.py --task landing --steps 2000
```

---

## 6. PPO Training

Scaffolding for PPO training with the TVC environment. Integrate your RL library at the marked section in `apps/run_train_ppo.py`.

```bash
python apps/run_train_ppo.py [OPTIONS]
```


| Argument                       | Default                             | Description                                        |
| ------------------------------ | ----------------------------------- | -------------------------------------------------- |
| `--task`                       | `hover`                             | Task: `hover`                                      |
| `--env-config`                 | `configs/env/train_128.yaml`        | Env config YAML                                    |
| `--disturbance`                | `configs/disturbances/nominal.yaml` | Disturbance config                                 |
| `--seed`                       | `0`                                 | Random seed                                        |
| `--total-steps`                | `5000000`                           | Total environment steps                            |
| `--output-dir`                 | `runs`                              | Base output directory (timestamped subdir created) |
| `--headless` / `--no-headless` | headless                            | Viewport toggle                                    |


**Examples:**

```bash
python apps/run_train_ppo.py --task hover --seed 42

python apps/run_train_ppo.py --task landing \
    --disturbance configs/disturbances/wind.yaml \
    --total-steps 10000000 --seed 0
```

---

## 7. GTrXL Environment Compatibility (No Trainer Yet)

The repository does not yet implement sequence-aware GTrXL-PPO optimization.
The command below is an explicit random-action environment smoke only and
cannot produce a trained checkpoint. Without `--env-smoke-only`, it exits 2.

```bash
python apps/run_train_gtrxl.py --env-smoke-only [OPTIONS]
```


| Argument                       | Default                             | Description                             |
| ------------------------------ | ----------------------------------- | --------------------------------------- |
| `--task`                       | `hover`                             | Task: `hover`                           |
| `--env-config`                 | `configs/env/train_128.yaml`        | Env config YAML                         |
| `--disturbance`                | `configs/disturbances/nominal.yaml` | Disturbance config                      |
| `--seed`                       | `0`                                 | Random seed                             |
| `--output-dir`                 | `runs`                              | Base output directory                   |
| `--headless` / `--no-headless` | headless                            | Viewport toggle                         |
| `--env-smoke-only`             | false                               | Required acknowledgement: no training  |
| `--smoke-steps`                | `100`                               | Random-action policy steps              |


**Examples:**

```bash
python apps/run_train_gtrxl.py --env-smoke-only --task hover --smoke-steps 100 --seed 0
```

---

## 8. Config Reference

### Environment configs


| File                                | num_envs | Use case                     |
| ----------------------------------- | -------- | ---------------------------- |
| `configs/env/single_env_debug.yaml` | 1        | Debug / PID eval / GUI       |
| `configs/env/train_128.yaml`        | 128      | Training (GPU pipeline)      |
| `configs/env/hil_validation.yaml`   | 1        | Hardware-in-the-loop, 500 Hz |


### Disturbance configs


| File                                     | Contents                        |
| ---------------------------------------- | ------------------------------- |
| `configs/disturbances/nominal.yaml`      | All disturbances disabled       |
| `configs/disturbances/wind.yaml`         | Steady wind + gusts + body drag |
| `configs/disturbances/sensor_noise.yaml` | Observation noise std           |
| `configs/disturbances/com_shift.yaml`    | Per-episode COM offset range    |


### Physics configs (use with `--physics`)


| File                                        | Use case                            |
| ------------------------------------------- | ----------------------------------- |
| `configs/physics/physx_single.yaml`         | Single-env PhysX defaults           |
| `configs/physics/physx_train.yaml`          | GPU pipeline for 128-env training   |
| `configs/physics/solver_high_fidelity.yaml` | High-fidelity solver for validation |


---

## 9. Full Validation Run (sequential ladder)

Run all tests in order to validate a full environment build:

```powershell
# Step 1 — offline unit tests (no Isaac)
PYTHONPATH=. python -m pytest tests/unit/ -v

# Step 2 — sim tests 00–12 (requires Isaac Sim)
 $tests = @(
  'test_00_asset_validation',
  'test_01_joint_axes',
  'test_02_single_fin_articulation',
  'test_03_unit_force_on_fin',
  'test_04_fin_force_sweep',
  'test_05_four_fin_superposition',
  'test_06_edf_spool_and_reaction',
  'test_07_gyro_precession',
  'test_08_wind_disturbance',
  'test_09_contact_landed_crash',
  'test_10_pid_hover_smoke',
  'test_11_rl_api_128env_smoke',
  'test_12_steady_hover_all_forces'
 )

 foreach ($TEST in $tests) {
   Write-Host "=== $TEST ==="
   python apps/run_single_test.py --test $TEST
   if ($LASTEXITCODE -ne 0) { break }
 }
```

**Same ladder with UI (viewport opens for each step):** append `--no-headless` inside the loop:

```powershell
 foreach ($TEST in $tests) {
   Write-Host "=== $TEST (UI) ==="
   python apps/run_single_test.py --test $TEST --no-headless
   if ($LASTEXITCODE -ne 0) { break }
 }
```
