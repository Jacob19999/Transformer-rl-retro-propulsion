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
