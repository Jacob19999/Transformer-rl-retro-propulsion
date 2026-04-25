# Golden: Touchdown Cases

Reference contact state trajectories from test_09 (contact/landed/crash state machine).

## Contents

- `soft_touchdown.json`   — Slow vertical descent → LANDED state (reference trajectory)
- `hard_impact.json`      — High-speed impact → CRASHED state
- `bounce_event.json`     — Light contact → bounce → AIRBORNE state
- `tip_over.json`         — Tilt-at-contact → CRASHED via excessive tilt check

## Format (JSON)

```json
{
  "case_name": "soft_touchdown",
  "description": "Gentle 0.5 m/s vertical descent, no tilt",
  "parameters": {
    "impact_speed_ms": 0.5,
    "tilt_rad": 0.02,
    "angular_rate_rad_s": 0.05
  },
  "expected_outcome": "LANDED",
  "contact_state_sequence": [0, 0, 0, 1, 1, 2],
  "dwell_steps_required": 5
}
```

## Generation

Run test_09 with `--golden-output tests/goldens/touchdown_cases/` after validating
the contact state machine against slow-motion drop test footage.

See `tvc_env/sim/contacts.py` for ContactState enum and `tvc_env/sim/crash_logic.py`
for crash detection thresholds.
