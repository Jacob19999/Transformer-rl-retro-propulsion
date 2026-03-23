# Contract: Action Space

**Feature**: 007-isaac-sim-env | **Date**: 2026-03-22

## Overview

The environment accepts a 5-dimensional action vector as a flat `torch.Tensor` of shape `(num_envs, 5)`. Actions are interpreted without controller-specific assumptions.

## Action Layout

| Index | Dim | Field | Unit | Range | Description |
|-------|-----|-------|------|-------|-------------|
| 0-3 | 4 | Fin target angles | rad | [-max_angle, max_angle] | Commanded deflection for each fin |
| 4 | 1 | Throttle target | — | [0, 1] | Normalized throttle command |

**Total dimension**: 5

## Conventions

- **Fin ordering**: Matches `fin_joint_names` from asset metadata (ordered +X, +Y, -X, -Y)
- **Positive fin deflection**: Follows the right-hand rule around each fin's hinge axis as defined in the USD joint
- **Throttle**: 0.0 = zero thrust, 1.0 = maximum thrust. Mapped internally to RPM target via `throttle * omega_max`

## Actuator Processing

Actions pass through actuator models before affecting the simulation:

1. **Fin angles** → `actuator_servo.py`:
   - Clamped to `[-max_command_angle, max_command_angle]`
   - First-order lag applied: `ẋ = (x_cmd - x) / τ_servo`
   - Rate-limited to `[-ω_max_servo, ω_max_servo]`
   - Deadband applied (optional)

2. **Throttle** → `propulsion_edf.py`:
   - Mapped to RPM target: `ω_target = throttle * ω_max`
   - Spool dynamics applied: first-order lag with `τ_motor`
   - Rate-limited to `[-dω_max, dω_max]`

## Controller Interpretation

Different controllers produce actions differently:

| Controller | Fin Angle Source | Throttle Source |
|------------|-----------------|-----------------|
| PID | `pid_fin_mixer.py` maps roll/pitch/yaw commands to 4 fin angles | PID altitude loop output |
| PPO | Direct network output (4 values) | Direct network output (1 value) |
| GTrXL-PPO | Direct network output (4 values) | Direct network output (1 value) |

The environment is agnostic to how actions are generated. All controllers produce the same 5D action vector.

## Gymnasium Space Definition

```python
action_space = gym.spaces.Box(
    low=np.array([-max_angle]*4 + [0.0]),
    high=np.array([max_angle]*4 + [1.0]),
    shape=(5,),
    dtype=np.float32,
)
```

## Consumers

- `_pre_physics_step(actions)` in DirectRLEnv receives raw actions
- `_apply_action()` processes through actuator models each substep
