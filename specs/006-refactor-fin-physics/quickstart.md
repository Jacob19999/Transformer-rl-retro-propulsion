# Quickstart: Refactor Drone Fin Physics Layer

**Feature Branch**: `006-refactor-fin-physics`
**Date**: 2026-03-21

## Prerequisites

- Python 3.10+
- NVIDIA Isaac Sim v5.1.0 with IsaacLab
- Project dependencies: `pip install -r requirements.txt`
- Blender (if regenerating USD from CAD)

## Verification Commands

After implementing changes, run this validation sequence (per constitution Development Workflow):

```bash
# 1. Regenerate USD with explicit mass properties and fixed conventions
python -m simulation.isaac.usd.postprocess_usd \
  --input simulation/isaac/usd/drone_blender.usd \
  --output simulation/isaac/usd/drone.usd \
  --config simulation/configs/default_vehicle.yaml

# 2. Validate mass properties (YAML ↔ USD agreement)
python -m simulation.isaac.scripts.validate_mass_props

# 3. Run fin wiggle diagnostic (100 episodes, confirms articulation + collider stability)
python -m simulation.isaac.scripts.diag_fin_wiggle --episodes 100

# 4. Run thrust diagnostic (confirms drone lifts off correctly)
python -m simulation.isaac.scripts.diag_thrust_test --thrust 1.0 --duration 2.0 --spawn-alt 0.4

# 5. Run wind diagnostic (confirms environmental forces)
python -m simulation.isaac.scripts.diag_wind --wind-x 5.0 --duration 3.0

# 6. Run all tests
pytest

# 7. Run Isaac-specific tests
pytest -m isaac simulation/tests/test_isaac_env.py simulation/tests/test_drone_builder.py
```

## Key Files to Modify

| File | Change |
| ---- | ------ |
| `simulation/isaac/usd/postprocess_usd.py` | Author explicit mass/CoM/inertia; fix hinge axes; box colliders for fins |
| `simulation/isaac/tasks/edf_landing_task.py` | Remove FinMapping remap; unconditional deg→rad read; telemetry buffer |
| `simulation/isaac/fin_mapping.py` | Set defaults to identity or deprecate |
| `simulation/isaac/conventions.py` | Remove `FIN_JOINT_VISUAL_SIGN`; document verified unit convention |
| `simulation/configs/fin_mapping.yaml` | Update to identity mapping |
| `simulation/configs/default_vehicle.yaml` | Verify `mass_properties` section is complete |
| `simulation/isaac/scripts/validate_mass_props.py` | Ensure it validates inertia (not just mass + CoM) |

## Key Files to Create

| File | Purpose |
| ---- | ------- |
| `simulation/isaac/fin_telemetry.py` | Telemetry ring buffer and flush logic |

## Contracts

This feature modifies internal simulation modules only. There are no external-facing APIs, CLI contracts, or user-visible interface changes. The Gymnasium API contract (`reset`, `step`, `observation_space`, `action_space`) is explicitly preserved (FR-009).
