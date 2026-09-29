# IMU measurement model

Status: 2026-09-28. Opt-in; the default white-noise path is unchanged.

The environment used to corrupt attitude and body rate with independent white noise redrawn at every
control step. A fused-attitude IMU such as the WITMOTION WTGAHRS1 instead delivers a *filtered, delayed,
sample-and-held* signal whose error is slow and correlated. `tvc_env/dynamics/imu_model.py` simulates that
chain per environment.

## Signal chain

```
true q, v, w_body ──► specific force f = R^T (dv/dt − g)          (velocity difference per physics substep)
                      │
gyro  = M·w + turn-on bias + Gauss-Markov bias + random-walk bias + white noise + g-sensitivity·f
accel = M·f + turn-on bias + Gauss-Markov bias + random-walk bias + white noise      (M = scale + cross-axis)
                      │  saturate ─► first-order low-pass (bandwidth_hz) ─► quantize at sample ticks
                      ▼
sample & hold at sample_rate_hz ─► Mahony complementary filter (gyro + accel [+ heading]) ─► attitude q_est
                      ▼
transport latency (whole physics substeps) ─► flight computer: q_est, gyro
```

Every error term is redrawn per episode (one power cycle of one randomly drawn unit). Frames follow
`tvc_env.common.frames`: Isaac wxyz quaternions rotating body local axes into the Z-up world, sensor vectors in
body FRD, conversions only through `frames.py`.

What consumes it, when `disturbances.sensor_noise.imu` is enabled:

- the policy / convex controller observation: attitude quaternion and body rate (both the legacy 24-D
  observation and the 51-D waypoint-flight observation);
- the fixed yaw-rate damper, which used to read the true rate and now reads the held gyro;
- telemetry (`env.sensor_measurement`, the run_mission "IMU vs ACTUAL" overlay).

Position and velocity stay white noise: this IMU measures neither (assume motion capture / RTK).
`attitude_std` and `angular_velocity_std` are ignored while the IMU model is active.

## Enabling it

Any `BaseEnvConfig(..., disturbance_config_path=configs/disturbances/sensor_imu_wtgahrs1.yaml)` enables it. Nothing in `apps/` selects it yet. Wiring it into `run_mission.py` and the mission-control
UI (`validate_settings` rejects an `imu` block today) is the open follow-up.

`sensor_noise.imu` takes `profile: wtgahrs1` (from `configs/sensors/imu_wtgahrs1.yaml`) and any inline keys
override it. The profile is expanded and validated when the env config is built, so a typo fails before Isaac
starts and the recorded config carries the parameters actually used. Unknown keys are rejected.

## Conventions worth knowing

- `noise_density_*` is a datasheet **one-sided amplitude spectral density**. One sample at period dt has
  variance density² / (2·dt). Allan's white-noise coefficient N equals density/√2; the reference repo's `N`
  is the latter. Mixing the two overstates noise by √2 (a unit test caught this).
- `bias_instability_*` is the **peak Allan deviation of the bias**. For a first-order Gauss-Markov process that
  is 0.6174·σ at T ≈ 1.89·τ_c (analytic; confirmed by simulating an AR(1) process), so σ_GM = floor / 0.6174.
  The reference repo quotes the IEEE coefficient B = floor / 0.664 and a 0.4365 constant inside its own formula;
  that is a different parameterisation.
- After the 20 Hz low-pass the white gyro noise is ≈ density·√(π/2 · 20 Hz) (checked in the unit tests).

## Parameters and provenance

`configs/sensors/imu_wtgahrs1.yaml` tags every value. Summary of what is known versus assumed:

| Group | Source |
|---|---|
| 20 Hz bandwidth, 200 Hz max output rate, ±2000 dps / ±16 g ranges, gyro auto-calibration, 6-/9-axis algorithm | WTGAHRS1 manual v0707 §2.2.2, 2.3.3, 2.3.4, 2.4.3, 2.4.9 |
| Accel turn-on bias 2 mg | manual p.16 screenshot after calibration (X −0.0020 g, Y 0.0024 g) |
| Gyro stability 0.05 dps, accel stability 5 mg | existing preset ("datasheet v20-0615 §3.1") — **that datasheet is not in the repo and was not re-checked** |
| 16-bit resolution (0.061 dps, 0.488 mg per count) | assumed from the WIT serial protocol; the manual has no register table |
| Noise densities, correlation times, random walks, scale/misalignment, g-sensitivity, latency, filter gains, magnetometer errors | **placeholders** (typical MEMS magnitudes) |

Replace the placeholders from bench data (below). Do not describe results as WTGAHRS1-accurate until then.

## Validation

Unit tests (`tests/unit/test_imu_model.py`, `test_imu_allan.py`, `test_imu_observation.py`): white-noise sigma,
low-pass gain and −3 dB point, Gauss-Markov σ and correlation time, random-walk growth, per-episode bias and
per-env reset, saturation/quantization, hold-last-value, latency and delay-line flush, output aliasing, specific
force at rest / free fall / tilted, filter convergence and sign, gyro-only yaw hold, magnetometer pull, gyro bias →
linear yaw drift, accelerometer gate, config validation, and observation routing. The Allan fit recovers the
generating parameters (noise density 0.0199 vs 0.02, bias σ 0.0099 vs 0.01, τ_c 175 vs 200 s).

Isaac smoke (256 envs, zero action so the vehicle crashes and auto-resets often):

| Configuration | Result |
|---|---|
| Ideal chain (all errors off, no filter, kp = 0) | gyro error **0.000 °/s**, attitude error **0.02°**, across 176 auto-resets: frames, substep hook, reset hook and specific-force path are consistent with PhysX |
| Timing only (20 Hz low-pass, 200 Hz, 5 ms latency) | gyro error 2.5–3.1 °/s at a true rate of ~14 °/s rms; attitude 0.2–0.3° |
| Full WTGAHRS1 profile | gyro error ~2.2–4.3 °/s; attitude 2.3–3.0° mean (up to 9.8°) |

So most of the gyro error a controller sees is **lag on a fast-changing rate**, which the white-noise model could not
represent. No NaNs in any run. Not yet run: a convex-controller mission with the model enabled.

Cost: roughly +6–12 % wall time per policy step at 256 envs (run-to-run noise is of that order; the overhead is
small-kernel launches per physics substep).
Measure at training scale before enabling it in a long run; fusing the two channels or running the chain at the
sensor rate would cut it.

## The accelerometer aiding is level-seeking on this vehicle

A thrust-vectored vehicle's accelerometer always reads along the body axis, so any accelerometer aiding pulls the
tilt estimate toward level during sustained tilted flight; the magnitude gate cannot detect it. Mean attitude
error for a steady 10° tilt with the consistent acceleration g·tan θ (offline, 128 envs):

| kp, ki | 1 s | 2 s | 5 s | 10 s | 60 s hover: tilt rms / yaw rms |
|---|---|---|---|---|---|
| 1.0, 0.02 | 6.4° | 8.8° | 10.2° | 10.3° | 0.71° / 7.0° |
| 0.5, 0.01 | 4.1° | 6.5° | 9.5° | 10.4° | 0.71° / 7.0° |
| 0.25, 0.005 | 2.4° | 4.1° | 7.5° | 9.8° | 0.75° / 7.0° |
| **0.1, 0.002 (profile)** | 1.4° | 2.1° | 4.3° | 7.0° | 1.00° / 7.0° |
| 0.05, 0.001 | 1.1° | 1.4° | 2.6° | 4.6° | 1.67° / 7.0° |

No gain is good: low gains leave gyro bias uncorrected, high gains follow the thrust vector. The profile's 0.1 is
a compromise, not a measurement. WIT's onboard Kalman filter is proprietary and may behave differently; record the
sensor's fused angle on a tilted, accelerating rig or in flight against a reference to fit it. If the drone flies
long tilted legs, consider aiding the flight computer's own estimate with velocity or thrust instead of trusting the
onboard angle.

Gyro-only yaw drifts ≈ 0.1 °/s rms with the profile (7° in 60 s); magnetometer mode holds it near 3° but the
EDF's 120 A bus is expected to disturb the magnetometer, so the default is gyro.

## Bench characterisation

```powershell
python tools/imu_allan_variance.py static.csv --rate-hz 200 --gyro gx,gy,gz --gyro-unit dps --accel ax,ay,az --accel-unit g
```

Record ≥ 6 h static (EDF off, temperature settled); repeat with the EDF spinning. The tool fits white noise +
Gauss-Markov bias + random walk to the overlapping Allan variance and prints profile keys. Turn-on bias, scale
factor and misalignment are not observable from a static log; measure them with a multi-position calibration and
a turntable.

## Not modelled

Lever-arm (centripetal / tangential) acceleration at the mount point, EDF vibration and temperature drift,
magnetometer error correlated with motor current, WIT's proprietary filter, serial-protocol packet timing beyond
one fixed latency. Physics runs at 480 Hz; the sensor cannot report faster than that.

## References

- M. Nitsch, *IMU-Simulator* (BSD-3-Clause): structure of the error chain. Independent torch implementation; no code copied.
- A. D. Young, M. J. Ling, D. K. Arvind, "IMUSim", IPSN 2011 (GPL-3.0 code, **not used**): gyro acceleration
  sensitivity and complementary-filter orientation are concepts from that paper; please cite it if this is published.
- R. Mahony, T. Hamel, J.-M. Pflimlin, "Nonlinear complementary filters on the special orthogonal group", IEEE TAC 2008.
- IEEE Std 952-1997 (Allan variance for gyros).
