# Golden: Rotor Reaction Torque Curves

Reference torque curves from test_06 (EDF spool and reaction torque sweep).

## Contents

- `static_reaction_torque_vs_throttle.csv` — Q_static = k_Q * ω² vs throttle [0,1]
- `spool_torque_vs_throttle_rate.csv`       — Q_spool = I_rotor * α_rotor vs dω/dt
- `gyro_precession_vs_angular_rate.csv`     — |τ_gyro| vs |ω_body| at various RPM

## Generation

Run test_06 with `--golden-output tests/goldens/reaction_torque_curves/` once
motor parameters k_Q, I_rotor are calibrated from bench test data.

## Parameters (nominal)

- k_Q: null → to-be-calibrated (source: bench test)
- k_T: null → to-be-calibrated (source: bench test)
- omega_max: null → to-be-calibrated (source: bench test)
- I_rotor: estimated from fan geometry (to-be-measured)

See `configs/params/edf_90mm.yaml` for current parameter values.
