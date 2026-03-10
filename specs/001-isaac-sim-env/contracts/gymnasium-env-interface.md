# Contract: Gymnasium Environment Interface

**Feature**: Isaac Sim Vectorized Drone Simulation Environment
**Branch**: `001-isaac-sim-env` | **Date**: 2026-03-10
**Target**: Isaac Sim 5.1.0 / IsaacLab 2.3 / Python 3.11

This contract defines the interface that `EDFIsaacEnv` must expose to be compatible with the existing training pipeline (SB3 PPO, `VecNormalize`, diagnostic scripts).

---

## Interface: `EDFIsaacEnv`

Inherits from / wraps Gymnasium `Env` or `VectorEnv` protocol.

### Properties

```
observation_space: gymnasium.spaces.Box
  dtype: float32
  shape: (20,)                         # single-env; (num_envs, 20) for vectorized
  low:   -inf (20,)
  high:  +inf (20,)

action_space: gymnasium.spaces.Box
  dtype: float32
  shape: (5,)                          # single-env; (num_envs, 5) for vectorized
  low:   -1.0 (5,)
  high:  +1.0 (5,)
```

### Methods

#### `reset(seed=None, options=None) → (obs, info)`

- Returns `obs: np.ndarray[float32]` shape `(num_envs, 20)` or `(20,)` for single-env
- Returns `info: dict` with at minimum:
  - `"episode_step": int` — always 0 after reset
  - `"env_ids": list[int]` — indices of environments that were reset
- Randomizes initial state from config ranges
- Resets episode timer and step counter

#### `step(action) → (obs, reward, terminated, truncated, info)`

- `action: np.ndarray[float32]` shape `(num_envs, 5)` or `(5,)` for single-env
- All action values clipped to `[-1, 1]` before application
- Returns:
  - `obs: np.ndarray[float32]` shape `(num_envs, 20)` or `(20,)`
  - `reward: np.ndarray[float32]` shape `(num_envs,)` or scalar
  - `terminated: np.ndarray[bool]` — episode ended by terminal condition (crash / landing)
  - `truncated: np.ndarray[bool]` — episode ended by step limit
  - `info: dict` with at minimum:
    - `"episode_step": np.ndarray[int]` current step per env
    - `"h_agl": np.ndarray[float32]` altitude per env
    - `"is_success": np.ndarray[bool]` landing success per env

#### `close() → None`

- Shuts down Isaac Sim; releases GPU resources

---

## Contract: Vectorized Reset Semantics

When `terminated[i]` or `truncated[i]` is True after `step()`, environment `i` is **automatically reset** by the next `step()` call (auto-reset pattern required by SB3's `VecEnv`). The returned `obs[i]` after a done episode reflects the **post-reset observation** of environment `i`, not the terminal observation.

---

## Contract: Action Scaling

| Action index | Field | Normalized range | Physical range |
|---|---|---|---|
| 0 | Thrust command | [0, 1] (half-range, unidirectional) | [0, 45.0] N |
| 1 | Fin 1 deflection | [-1, +1] | [-0.2618, +0.2618] rad (±15°) |
| 2 | Fin 2 deflection | [-1, +1] | [-0.2618, +0.2618] rad (±15°) |
| 3 | Fin 3 deflection | [-1, +1] | [-0.2618, +0.2618] rad (±15°) |
| 4 | Fin 4 deflection | [-1, +1] | [-0.2618, +0.2618] rad (±15°) |

Note: Thrust action `[-1, 0]` maps to 0 N (clamped); only positive half is physically meaningful.

---

## Contract: Observation Normalization

The Isaac Sim env returns **unnormalized** observations. `VecNormalize` from SB3 is applied externally (same as existing training pipeline). The env does NOT apply internal normalization.

---

## Contract: Seed Reproducibility

- `reset(seed=N)` MUST fully determine the subsequent random state for that episode
- All environments in a vectorized batch derive their per-env seed from the global seed via `seed + env_id` offset
- Isaac Sim scene randomization (mass perturbations from DR) MUST use the seeded RNG

---

## Contract: Config File Interface

The env accepts a single YAML config path at construction time. The config MUST contain:
- `num_envs`: int
- `vehicle_config_path`: str (path to `default_vehicle.yaml`)
- `spawn_altitude_range`: [float, float]
- `spawn_velocity_magnitude_range`: [float, float]
- `episode_length_steps`: int

Optional (with defaults):
- `num_physics_substeps`: int (default: 4)
- `domain_randomization_config_path`: str
