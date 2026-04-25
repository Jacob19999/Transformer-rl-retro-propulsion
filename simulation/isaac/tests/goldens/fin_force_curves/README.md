# Golden: Fin Force Curves

Reference aerodynamic force curves from test_04 (fin force sweep).

## Contents

- `fin_normal_force_sweep.csv` — C_N vs deflection angle (deg) at nominal exhaust speed
- `fin_drag_force_sweep.csv`   — C_D vs deflection angle (deg) at nominal exhaust speed

## Generation

Run test_04 with `--golden-output tests/goldens/fin_force_curves/` once the sim
environment is validated against hardware measurements. These curves then serve as
regression references to detect unintended aero model changes.

## Parameters (nominal)

- Exhaust speed: 40 m/s (max_exhaust_speed, source: estimate)
- Fin area: 0.002 m² (source: measured)
- C_N_alpha: 3.5 rad⁻¹ (source: estimate, to-be-calibrated)
- C_D_0: 0.02 (source: estimate)

## Usage

```python
import csv
with open("tests/goldens/fin_force_curves/fin_normal_force_sweep.csv") as f:
    reference = list(csv.DictReader(f))
```
