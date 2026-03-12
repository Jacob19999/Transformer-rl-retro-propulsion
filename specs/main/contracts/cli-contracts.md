# CLI Contracts: Feature 002

## validate_mass_props

**Module**: `simulation.isaac.scripts.validate_mass_props`

```
Usage: python -m simulation.isaac.scripts.validate_mass_props [OPTIONS]

Options:
  --usd PATH       Path to USDC scene (default: simulation/isaac/usd/drone.usdc)
  --config PATH    Path to vehicle YAML config (default: simulation/configs/default_vehicle.yaml)
  --tolerance FLOAT  Relative tolerance for comparison (default: 0.01 = 1%)
  --json           Output report as JSON instead of table
  --quiet          Only print PASS/FAIL, no detailed table

Exit codes:
  0  All values within tolerance (PASS)
  1  One or more values outside tolerance (FAIL)
  2  Error reading USD or YAML (file not found, schema missing, etc.)
```

**Output format (default)**:
```
Mass Property Validation Report
================================
Field           YAML        USD         Error    Status
total_mass      3.130 kg    3.130 kg    0.00%    ✓
com_x           0.0045 m    0.0045 m    0.00%    ✓
com_y           0.0008 m    0.0008 m    0.00%    ✓
com_z           0.0046 m    0.0046 m    0.00%    ✓
Ixx             0.01358     0.01358     0.00%    ✓
Iyy             0.01549     0.01549     0.00%    ✓
Izz             0.00480     0.00480     0.00%    ✓
Ixy            -0.00006    -0.00006     0.00%    ✓
Ixz             0.00034     0.00034     0.00%    ✓
Iyz             0.00006     0.00006     0.00%    ✓
================================
RESULT: PASS (10/10 within 1.0% tolerance)
```

## diag_thrust_test

**Module**: `simulation.isaac.scripts.diag_thrust_test`

```
Usage: python -m simulation.isaac.scripts.diag_thrust_test [OPTIONS]

Options:
  --config PATH      Isaac env YAML config (default: isaac_env_single.yaml)
  --thrust FLOAT     Normalized thrust command 0.0-1.0 (default: 1.0)
  --duration FLOAT   Test duration in seconds (default: 3.0)
  --spawn-alt FLOAT  Spawn altitude in meters (default: 0.4)
  --episodes INT     Number of episodes (default: 1)

Exit codes:
  0  Drone reached expected altitude
  1  Drone did not lift off or test failed
```

## diag_wind

**Module**: `simulation.isaac.scripts.diag_wind`

```
Usage: python -m simulation.isaac.scripts.diag_wind [OPTIONS]

Options:
  --config PATH      Isaac env YAML config (default: isaac_env_single.yaml)
  --wind-x FLOAT     Constant wind in X direction (m/s, default: 5.0)
  --wind-y FLOAT     Constant wind in Y direction (m/s, default: 0.0)
  --wind-z FLOAT     Constant wind in Z direction (m/s, default: 0.0)
  --duration FLOAT   Test duration in seconds (default: 3.0)
  --episodes INT     Number of episodes (default: 1)

Exit codes:
  0  Lateral drift detected (wind producing force)
  1  No measurable drift (wind not working)
```

## Internal API Contracts

### IsaacWindModel

```python
class IsaacWindModel:
    """GPU-batched wind model for Isaac Sim environments."""

    def __init__(self, config: dict, num_envs: int, device: torch.device):
        """Load wind config from environment YAML."""

    def reset(self, env_ids: torch.Tensor) -> None:
        """Sample new wind conditions for specified environments."""

    def step(self, dt: float) -> torch.Tensor:
        """Advance wind state by dt, return wind_vector_world (num_envs, 3)."""

    def compute_drag_force(
        self,
        wind_vector: torch.Tensor,    # (num_envs, 3)
        body_velocity: torch.Tensor,  # (num_envs, 3)
    ) -> torch.Tensor:
        """Compute aerodynamic drag force from relative wind. Returns (num_envs, 3)."""

    @property
    def wind_ema(self) -> torch.Tensor:
        """Exponential moving average of wind for observation vector. (num_envs, 3)."""
```

### validate_mass_props (library API)

```python
def compare_mass_properties(
    usd_path: str | Path,
    yaml_config: dict,
    tolerance: float = 0.01,
) -> MassPropertyReport:
    """Compare USDC scene mass props against YAML config.

    Returns MassPropertyReport with per-field comparison.
    Raises FileNotFoundError if USD path invalid.
    Raises KeyError if YAML missing mass_properties section.
    """
```

---

## User Story 4: Gyro Precession Contracts

## diag_gyro_precession

**Module**: `simulation.isaac.scripts.diag_gyro_precession`

```
Usage: python -m simulation.isaac.scripts.diag_gyro_precession [OPTIONS]

Options:
  --config PATH         Isaac env YAML config (default: isaac_env_gyro_test.yaml)
  --torque-axis STR     Axis for applied torque: 'pitch' or 'roll' (default: pitch)
  --torque-mag FLOAT    External torque magnitude in N·m (default: 0.5)
  --duration FLOAT      Test duration in seconds (default: 2.0)
  --spawn-alt FLOAT     Spawn altitude in meters (default: 5.0)
  --no-gravity          Disable gravity for isolated precession test (default: true)
  --disable-precession  Run with precession disabled for comparison

Physics note: --torque-axis 'yaw' is intentionally NOT supported. Yaw rate is parallel
to the fan spin axis, so cross(omega_z, h_fan_z) = 0. Precession only occurs from
pitch/roll rates coupling with h_fan. Use pitch or roll torque to observe precession.

Exit codes:
  0  Precession response detected (perpendicular rate > threshold when enabled)
  1  No precession response (test failed) or unexpected response when disabled
```

**Output format**:
```
Gyro Precession Diagnostic Report
===================================
Isaac Sim v5.1.0 | I_fan = 3.0e-5 kg·m² | k_thrust = 4.55e-7 N/(rad/s)²
Precession: ENABLED | Gravity: DISABLED | Spawn alt: 5.0 m

Phase 1: Hover stabilization (0.5 s)
  Thrust: 30.7 N (hover) | ω_fan: 8214 rad/s
  Altitude: 5.00 m (stable)

Phase 2: Yaw torque application (1.5 s)
  Applied yaw torque: 0.5 N·m about Z-axis (body +Z / spin axis)
  Note: yaw rate alone does NOT cause precession (parallel to spin axis).
  Pitch/roll rate from aerodynamic cross-coupling drives the precession term.
  Expected: residual pitch/roll rates → roll/pitch coupling via τ_gyro = −ω×h_fan

  Time(s)  Yaw(°/s)  Pitch(°/s)  Roll(°/s)
  0.0      0.00      0.00        0.00
  0.5      2.34      0.87        0.02
  1.0      4.12      1.54        0.05
  1.5      5.89      2.21        0.07

RESULT: PASS — pitch response detected (2.21 °/s at t=1.5s)
  Precession ratio (pitch_rate/yaw_rate): 0.375
  Expected ratio (I_fan·ω_fan/I_pitch): ~0.38
  Agreement: 1.3% — WITHIN TOLERANCE
===================================
```

## validate_mass_props (extended)

**Additional output** (appended to existing mass validation report):

```
Rotor Prim Validation
=====================
Prim path:  /Drone/Body/Rotor          ✓ (exists)
Radius:     YAML=0.040 m  USD=0.040 m  err=0.00%  ✓
Height:     YAML=0.040 m  USD=0.040 m  err=0.00%  ✓
Position:   YAML=(0,0,-0.06) USD=(0,0,-0.06)       ✓ (Z-up)
Mass:       0.000 kg (validation-only, not physics-active)
I_fan(cfg): 3.0e-5 kg·m² (from edf.I_fan — used for precession torque)
=====================
ROTOR: PASS
```

## Internal API Contracts (Gyro Precession)

### _rotate_body_to_world

```python
def _rotate_body_to_world(
    quat_w: torch.Tensor,   # (N, 4) scalar-last [qx, qy, qz, qw]
    v_body: torch.Tensor,   # (N, 3) body-frame vector
) -> torch.Tensor:
    """Rotate body-frame vector to world frame using quaternion.

    Returns: (N, 3) world-frame vector.
    Inverse of existing _rotate_world_to_body().
    """
```

### GyroPrecession computation (inline in _apply_action)

```python
# Added to EdfLandingTask._apply_action() after fin forces, before set_external_force_and_torque:
if self._gyro_enabled:
    omega_b = self.robot.data.root_ang_vel_b                    # (N, 3)
    omega_fan = (self.thrust_actual / _K_THRUST).clamp(min=0).sqrt()  # (N,)
    h_fan_b = torch.zeros((num_envs, 3), device=self.device)
    h_fan_b[:, 2] = _I_FAN * omega_fan
    tau_gyro_b = -torch.linalg.cross(omega_b, h_fan_b)         # (N, 3)
    tau_gyro_w = _rotate_body_to_world(self.robot.data.root_quat_w, tau_gyro_b)
    torques[:, 0, :] += tau_gyro_w
```
