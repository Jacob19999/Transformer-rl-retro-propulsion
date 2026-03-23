# Contract: Observation Space

**Feature**: 007-isaac-sim-env | **Date**: 2026-03-22

## Overview

The environment exposes a controller-agnostic observation vector as a flat `torch.Tensor` of shape `(num_envs, obs_dim)`. This contract defines the observation layout, value ranges, and frame conventions.

## Observation Layout

| Index | Dim | Field | Unit | Frame | Range |
|-------|-----|-------|------|-------|-------|
| 0-2 | 3 | Position error to target | m | world | [-100, 100] |
| 3-6 | 4 | Attitude quaternion | — | world (wxyz) | unit quaternion |
| 7-9 | 3 | Linear velocity | m/s | body-FRD | [-50, 50] |
| 10-12 | 3 | Angular velocity | rad/s | body-FRD | [-20, 20] |
| 13 | 1 | Height above ground | m | world | [0, 200] |
| 14-17 | 4 | Fin actual angles | rad | — | [-max_angle, max_angle] |
| 18-21 | 4 | Fin actual rates | rad/s | — | [-ω_max, ω_max] |
| 22 | 1 | Motor RPM (normalized) | — | — | [0, 1] |
| 23 | 1 | Contact state | encoded int | — | {0, 1, 2, 3} |

**Base dimension**: 24

### Optional extensions (configurable)

| Index | Dim | Field | Unit | Frame | Range |
|-------|-----|-------|------|-------|-------|
| 24-26 | 3 | Wind estimate | m/s | world | [-30, 30] |

**Extended dimension**: 27

## Conventions

- **Quaternion order**: (w, x, y, z) — Isaac Lab 2.3.2 convention
- **Body-frame velocities**: FRD convention (x=forward, y=right, z=down)
- **Position error**: `target_position - vehicle_position` in world frame
- **Fin ordering**: Consistent with `fin_link_names` from asset metadata (ordered +X, +Y, -X, -Y)
- **Contact state encoding**: 0=AIRBORNE, 1=GROUND_CONTACT_CANDIDATE, 2=LANDED, 3=CRASHED
- **Motor RPM normalization**: `current_rpm / max_rpm` so output is [0, 1]

## Gymnasium Space Definition

```python
observation_space = gym.spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(obs_dim,),
    dtype=np.float32,
)
```

The actual value ranges above are for documentation; the Gymnasium space uses unbounded Box for compatibility with RL libraries that handle normalization externally (e.g., VecNormalize).

## Consumers

- PPO policy network (via VecNormalize)
- GTrXL-PPO policy network (via custom normalization)
- PID controller adapter (reads specific indices directly)
- Telemetry logger (records full vector each step)
