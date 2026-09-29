# IMU measurement model

Status: 2026-09-29. Opt-in; the default white-noise path is unchanged. Three hardware profiles
(WTGAHRS1, BNO085, VN-110E) with best-estimate parameters; **static accuracy validation is deferred**.

The environment used to corrupt attitude and body rate with independent white noise redrawn at every
control step. A fused-attitude IMU such as the WITMOTION WTGAHRS1 instead delivers a *filtered, delayed,
sample-and-held* signal whose error is slow and correlated. `tvc_env/dynamics/imu_model.py` simulates that
chain per environment.

## Signal chain

```
true q, v, w_body ──► specific force f = R^T (dv/dt − g)          (velocity difference per physics substep)
                      │
gyro  = M·w + turn-on bias + TCO·ΔT + Gauss-Markov bias + random-walk bias + white noise + g-sensitivity·f
accel = M·f + turn-on bias + TCO·ΔT + Gauss-Markov bias + random-walk bias + white noise      (M = scale + cross-axis)
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

Any `BaseEnvConfig(..., disturbance_config_path=configs/disturbances/sensor_imu_<name>.yaml)` enables it, with
`<name>` one of `wtgahrs1`, `bno085`, `vn110e`.

**Mission control:** on the Environment → Sensor noise tab each IMU hardware card has **Noise σ** (the datasheet
white-noise mapping) and **Physical chain**. Physical chain sets `disturbance_settings.sensor_noise.imu_profile`
(`""`, or a profile name checked against `configs/sensors/imu_*.yaml`); `disturbance_config` turns it into
`sensor_noise.imu = {enabled: true, profile: <name>}` and drops the request-only key, so `run_mission.py` needs no
change. While it is active the attitude and body-rate noise sliders are inert. Position and velocity keep their
white-noise settings: this IMU measures neither, and they stand for the external source (motion capture or RTK
today; optical-flow and barometer fusion later). The Flight page's IMU-vs-ACTUAL overlay shows the chain's output.

`sensor_noise.imu` takes `profile: wtgahrs1 | bno085 | vn110e` (from `configs/sensors/imu_<name>.yaml`) and any inline keys
override it. The profile is expanded and validated when the env config is built, so a typo fails before Isaac
starts and the recorded config carries the parameters actually used. Unknown keys are rejected.

## Strapdown navigation (opt-in, unaided)

`imu.nav.enabled: true` (mission control: **Position & velocity source → Inertial**) makes the flight computer dead-reckon
from the sensor's own registers: each physics substep it rotates the delayed accelerometer register by the delayed onboard
attitude estimate, subtracts nominal gravity, and integrates to velocity and position (trapezoidal). The solution starts from
the true position and velocity (`nav.initial_*_std`, default 0) and is otherwise unaided: no baro, optical flow or GNSS.
When on, the observation's position, height and body-frame velocity are that solution; `position_std` / `velocity_std` are
ignored, and the mission sequencer (waypoint capture, hover dwell) and the controller both fly on the estimate. Truth
(`pad_distance`, contact, landing outcome) stays physics truth, so a drifting run visibly misses the pad.

Drift mechanisms, all visible in the model: accelerometer scale factor and bias (1 % scale = 0.1 m/s² vertical at hover),
attitude error (tilt ε leaks g·ε into horizontal acceleration), noise, latency. Unaided, tilt and acceleration are not
separable, and an accelerometer-aided filter settles where the vehicle believes it is level and stationary while it
actually accelerates, so horizontal drift is largely set by the attitude filter, not the gyro or accelerometer grade.

Isaac hop (Hover 10 m in gusts, seed 2026, dead-reckoning only, position error vs truth, single flights):

| Sensor | 5 s | 10 s | 20 s | 40 s |
|---|---|---|---|---|
| error-free sensor (diagnostic) | 0.01 m | 0.01 m | 0.01 m | 0.01 m |
| VN-110E profile | 0.4 m | 1.1 m | 3.5 m | 11.6 m |
| WTGAHRS1 profile | 2.1 m | 7.3 m | 26 m | crashed at 29 s |

## Sensor fusion (opt-in): IMU + TFmini Plus + MTF-01P + barometer through an EKF3-style filter

`imu.fusion.enabled: true` with `profile: tfmini_mtf01p` (mission control: **Position & velocity source → Fused**) adds three
aiding sensors and replaces the flight computer's onboard-filter attitude with its own estimate, as ArduPilot does. Code:
`tvc_env/dynamics/nav_sensors.py` (sensors), `nav_ekf.py` (filter), `nav_fusion.py` (config, wiring), profile
`configs/sensors/fusion_tfmini_mtf01p.yaml` (per-value provenance).

| Sensor | Model | Source |
|---|---|---|
| Benewake TFmini Plus rangefinder | slant range to flat ground along body-down, 100 Hz, 10 ms latency, σ 3 cm, ±5 cm systematic (1 % beyond 5 m), 1 cm steps, valid 0.1–8 m and within 60° of nadir | datasheet A07; 8 m max is an estimate for a mid-grey pad (12 m at 90 %, 4 m at 10 % reflectivity) |
| MicoAir MTF-01P optical flow | apparent ground angular rate `f = (v_y/r − ω_x, −v_x/r − ω_y)`, 100 Hz, 15 ms latency, σ 0.05 rad/s, 5 % scale and 0.01 rad/s bias per episode, valid 0.08–12 m and up to 7 rad/s | product page (rate, 7 m/s at 1 m, >8 cm, 12 m ToF); noise, scale, bias are estimates |
| Barometer (WTGAHRS1) | height + Gauss-Markov drift σ 0.5 m (τ 300 s, zero at boot) + 15 cm noise, 20 Hz, 50 ms | datasheet "accuracy 1 m"; rest estimated |

The filter is a 16-state error-state EKF (attitude, velocity, position, gyro bias, accelerometer bias, terrain height) fed by
the IMU chain's raw gyro and accelerometer registers. It is **not** ArduPilot's code, but follows AP_NavEKF3: covariance
predicted every ~10 ms, process noise and defaults `EK3_GYRO_P_NSE 0.015`, `ACC_P_NSE 0.35`, `GBIAS_P_NSE 1e-3`,
`ABIAS_P_NSE 2e-2`, `TERR_GRAD 0.1`; flow line-of-sight model with `FLOW_M_NSE 0.25`, `FLOW_I_GATE 300`; rangefinder
`RNG_M_NSE 0.5`, `RNG_I_GATE 500`; baro `ALT_M_NSE 3`, `HGT_I_GATE 500`; late measurements are compared with the state stored at
their sample time. Accelerometer does not aid tilt directly, so the level-seeking trap of the onboard filter does not apply.
Defaults were read from the ArduPilot source (`AP_NavEKF3.cpp`, Copter build) on 2026-09-29.

Not modelled: compass/GNSS (heading is gyro-only and unobservable), flow-quality weighting, EKF3 lane switching, on-ground
constant-position fusion, delayed fusion horizon (the correction is applied to the current state), sensor lever arms other than
the shared down-axis mount offset (`mount_down_m 0.15`), rangefinder or flow on non-flat ground.

Isaac hop (Hover 10 m in gusts, seed 2026, fused navigation, single flights). Hover altitude is above the assumed 8 m rangefinder
limit, so height there is barometer-only and horizontal position comes from integrated flow velocity (no absolute horizontal
reference exists):

| IMU | Mission time | Position error 10 s / 20 s / 40 s | Touchdown | Pad distance |
|---|---|---|---|---|
| WTGAHRS1 | 45 s | 0.2 / 0.6 / 1.5 m | 0.23 m/s | 1.4 m |
| BNO085 | 44 s | 0.3 / 0.9 / 1.7 m | 0.14 m/s | 1.6 m |
| VN-110E | 43 s | 0.3 / 1.0 / 1.7 m | 0.14 m/s | 1.7 m |

All three land softly but outside the 0.5 m pad criterion, so mission control reports POST_LANDING_FAILURE: flow-only navigation
cannot hold absolute horizontal position. Unaided dead reckoning on the same hop reached 7–130 m. The three IMUs end up close
together because the aiding sensors, not the IMU grade, set the error, and every run shares the seed's flow scale error.
The filter adds roughly 30–40 % wall time in Isaac (real-time factor 0.75 → ~0.5 on CPU physics).

### Pad marker during the descent (opt-in)

Mission control **Position & velocity source → Fused + pad marker** (`imu_nav: fused_marker`, or `fusion.marker.enabled: true`)
adds a downward camera on an AprilTag-style marker at the landing target. It reports the marker's angular offsets in the
camera as ArduPilot's LANDING_TARGET message does, and the filter fuses them as a pad-relative position fix (tilt-compensated
through the attitude estimate), which is the only absolute horizontal reference in the stack. Model: 30 Hz, 60 ms latency,
60° field of view per axis, 640 px, 0.4 m tag needing ≥ 24 px (max range 9.2 m, min 0.3 m), centre noise 0.5 px, 0.23°
per-episode boresight error, 2 % missed detections. The flight computer only uses it between 0.3 and 8 m of estimated height
(`PLND_ALT_MIN/MAX`). No camera or tag is specified yet, so every number is an estimate; heading is not corrected from the tag.

Isaac hop, fused + marker, same seed: WTGAHRS1 lands 0.07 m from the pad at 0.17 m/s, VN-110E 0.07 m at 0.14 m/s (both
LANDED; without the marker 1.4–1.7 m and POST_LANDING_FAILURE). The marker is out of use through the 10 m hover (above 8 m)
and corrects the accumulated flow drift on the way down.

### CAM 05 · Down camera view

When a run used fused navigation, the Flight page adds **CAM 05** (bottom-left of the camera array, click its header to
collapse): the ground as the downward camera sees it, the optical-flow vectors (amber measured, white truth), the pad and
marker with the detector's state (tracking, visible but outside the altitude window, not detected, rejected, out of view with
an arrow), and a height ruler (truth, filter estimate, rangefinder) with the marker window shaded. Draw code:
`mission_control/web/downcam.js`; data: `frame.fusion` (`NavFusion.record()` carries the sensors' truth as well as their
readings). The ground texture is fixed to the world (1 m and 0.25 m cells) so altitude only changes how large it looks. It is
not included in recorded videos, and replays recorded before this telemetry existed do not show it.

## Truth angular rate (`rate_truth`)

`pose` (all three profiles) derives the gyro's true input from the change of the true attitude over each substep instead of
the engine's reported angular velocity. In free flight they agree. During pad contact PhysX corrects the pose without a
matching angular velocity: a gyro integrating the reported rate carried a fixed 0.8° pitch error from the first landing on
the pad, which the navigation turned into 6 m of horizontal drift at 10 s even for an error-free sensor. With `pose` the
error-free sensor tracks the truth to 1 cm over a 44 s hop and the mission lands. `reported` remains the default for
hand-built configs.

`run_mission.py` also re-seeds the IMU after it overrides the initial velocity and rates, otherwise the first substep read
the velocity jump as an acceleration spike.

## Conventions worth knowing

- `noise_density_*` is a datasheet **one-sided amplitude spectral density**. One sample at period dt has
  variance density² / (2·dt). Allan's white-noise coefficient N equals density/√2; the reference repo's `N`
  is the latter. Mixing the two overstates noise by √2 (a unit test caught this).
- `bias_instability_*` is the **peak Allan deviation of the bias**. For a first-order Gauss-Markov process that
  is 0.6174·σ at T ≈ 1.89·τ_c (analytic; confirmed by simulating an AR(1) process), so σ_GM = floor / 0.6174.
  The reference repo quotes the IEEE coefficient B = floor / 0.664 and a 0.4365 constant inside its own formula;
  that is a different parameterisation.
- After the 20 Hz low-pass the white gyro noise is ≈ density·√(π/2 · 20 Hz) (checked in the unit tests).

## Hardware profiles

Source documents are in `tools/`: `WITMotion IMU 2.pdf` (WTGAHRS1 **datasheet** v20-0615), `WITMotion IMU.pdf`
(WTGAHRS1 user manual v0707), `bst-bmi085-ds001.pdf`
(Bosch BMI085 data sheet rev 1.6, the sensor inside the BNO085), `adafruit-...-bno085.pdf` (breakout tutorial,
no accuracy figures) and `VN-110_DS.pdf`. Every value in `configs/sensors/imu_*.yaml` is tagged with where
it came from; `ESTIMATE`/`PLACEHOLDER` values are engineering guesses.

| | WTGAHRS1 | BNO085 (game rotation vector) | VN-110E |
|---|---|---|---|
| Class | hobby AHRS | hobby SiP with sensor hub | tactical |
| Output rate / bandwidth / latency | 200 Hz / 20 Hz / 5 ms | 100 Hz / 116 Hz / 5 ms | 400 Hz / 240 Hz / 2 ms |
| Gyro range | ±2000 dps | ±2000 dps | **±490 dps** |
| Gyro noise density (dps/√Hz) | 0.01 (placeholder) | 0.014 | 0.00139 |
| Gyro bias instability | 0.05 dps (preset) | 0.005 dps (estimate) | 0.55 °/hr |
| Gyro TCO | 0.02 dps/K (estimate) | 0.015 dps/K | 0.0002 dps/K (estimate) |
| Accel noise density (µg/√Hz) | 300 (placeholder) | 135 | 40 |
| Accel turn-on bias | 2 mg (manual p.16) | 20 mg (BMI085 zero-g offset) | 1.5 mg (from 0.05° static) |
| Accel TCO | 0.3 mg/K (estimate) | 0.2 mg/K | 0.005 mg/K (estimate) |
| Yaw | gyro (6-axis) | gyro (no magnetometer) | gyro |
| Filter stand-in (kp, ki, gate) | 0.1, 0.002, 0.3 g | 0.1, 0.002, 0.2 g | 0.05, 0.001, 0.1 g |

Points worth knowing:

- WTGAHRS1: the datasheet v20-0615 gives ±16 g / ±2000 dps, 16-bit frames (0.488 mg, 0.061 dps per count), gyro
  stability 0.05 dps, accelerometer stability 0.005 g and accuracy 0.01 g, and angle accuracy 0.05° (X/Y) / 1° (Z, after
  magnetic calibration). These are marketing-style figures with no test conditions and no noise density, and they do
  not agree with each other (0.01 g would be 0.6° of tilt against a 0.05° angle claim). The manual's post-calibration
  accelerometer reading (about 2 mg) is used for turn-on bias. The AK8963 magnetometer implies an MPU-9250-class chip;
  the WT901B sheet is not in the repo, so noise densities remain estimates. At the 9600 baud default 200 Hz output is
  impossible: acc+gyro+angle+mag is 44 bytes per sample, ~88 kbit/s, so 115200 baud is needed (76 % loaded, 95 % with the
  quaternion frame added).
- BNO085: the raw errors are the BMI085's. The sensor hub's continuous calibration and fusion are not simulated;
  the turn-on residuals stand in for the post-calibration state. CEVA's 0.5 °/min heading drift implies a residual
  gyro bias near 0.008 dps; the profile uses 0.05 dps (raw BMI085 offset is ±1 dps). CEVA's figures (2.5° dynamic,
  1.5° static, 3.1 dps gyro accuracy) are still the vendor excerpts recorded in the presets, not a datasheet in
  the repo. The 1.5° static figure is consistent with the BMI085's ±20 mg zero-g offset.
- VN-110E: the provided datasheet gives pitch/roll **0.05° RMS static** only. The **1.0° RMS dynamic** figure the
  presets use is not in it. Its ±490 dps gyro range saturates on fast rotations, and 240 Hz bandwidth means the
  chain adds almost no lag. The datasheet says it is individually calibrated for bias, scale, misalignment and
  temperature, so its systematic terms are small.
- Temperature: every profile draws a power-up offset within ±5 K of its calibration point and a +5 K
  self-heating rise (300 s). With the BMI085's 0.015 dps/K that is up to 0.15 dps of warm-up drift on the BNO085;
  for the other two the coefficients are estimates.
- The physics step is 480 Hz, so the VN-110E's 800 Hz IMU data is not reproduced; its attitude is modelled at 400 Hz.
- Mahony gains are stand-ins for three proprietary filters. Fit them to a scripted manoeuvre before
  reading anything into attitude error.

Status by group: manual/datasheet figures are cited per line; noise densities for the WTGAHRS1, bias
instabilities (except VN-110E), correlation times, latencies, the BNO085 hub bandwidth, thermal
warm-up shape, VN-110E g-sensitivity and scale factor, and all filter gains are **estimates**.
Do not describe results as sensor-accurate until they are replaced from bench data.

## Validation

Not yet done: **static accuracy validation of the three profiles** (simulated static attitude and Allan curves
against each datasheet headline, and a scripted-manoeuvre fit of the filter gains). `tests/unit/test_imu_profiles.py`
checks only that each profile loads, carries its datasheet figures, runs stably and that the temperature term works.

Unit tests (`tests/unit/test_imu_model.py`, `test_imu_profiles.py`, `test_imu_allan.py`, `test_imu_observation.py`): white-noise sigma,
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

Lever-arm (centripetal / tangential) acceleration at the mount point, EDF vibration, sensitivity versus temperature,
magnetometer error correlated with motor current, WIT's proprietary filter, serial-protocol packet timing beyond
one fixed latency. Physics runs at 480 Hz; the sensor cannot report faster than that.

## References

- M. Nitsch, *IMU-Simulator* (BSD-3-Clause): structure of the error chain. Independent torch implementation; no code copied.
- A. D. Young, M. J. Ling, D. K. Arvind, "IMUSim", IPSN 2011 (GPL-3.0 code, **not used**): gyro acceleration
  sensitivity and complementary-filter orientation are concepts from that paper; please cite it if this is published.
- R. Mahony, T. Hamel, J.-M. Pflimlin, "Nonlinear complementary filters on the special orthogonal group", IEEE TAC 2008.
- IEEE Std 952-1997 (Allan variance for gyros).
