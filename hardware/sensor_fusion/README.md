# Sensor-fusion portal: TFmini Plus + WTGAHRS1

Live web portal that fuses a Benewake **TFmini Plus** rangefinder with a WitMotion **WTGAHRS1** IMU/magnetometer/barometer, each on
its own CP2102 USB-UART adapter, with an EKF3-style filter on the raw IMU, plus IMU calibration and mount-orientation tools. Datasheets: `hardware/TF Mini Plus/`, `hardware/imu/witmotion/`.

## Wiring

| Sensor | Wire | Adapter |
|---|---|---|
| TFmini Plus | red +5 V, black GND, **green TXD** -> adapter RXD, white RXD -> adapter TXD | needs 5 V (up to 140 mA) |
| WTGAHRS1 | red VCC (3.3-5 V), black GND, **yellow TX** -> adapter RXD, green RX -> adapter TXD | |

COM numbers change between plugs, so the tools identify each sensor by what it sends (`59 59 ...` frames = TFmini, `55 5x ...`
packets = WitMotion), not by port name.

## Run

```
python hardware/sensor_fusion/init_sensors.py            # once: detect both, configure them (--check = measure only)
python hardware/sensor_fusion/portal.py                  # http://127.0.0.1:8002
python hardware/sensor_fusion/portal.py --simulate       # synthetic sensors, no hardware
python hardware/sensor_fusion/portal.py --log fusion.csv --no-bench-aiding --mag-yaw
python hardware/sensor_fusion/portal.py --simulate --sim-mount=-Z+Y+X   # try the wizards; Simulator pose freezes the vehicle
python -m pytest hardware/sensor_fusion -q
```

Ports: 8000 camera portal, 8001 BNO08x portal, **8002** this portal. A COM port can only be open in one program, so close
the Benewake GUI (and stop other portals) before starting.

`init_sensors.py` writes to the sensors only when they are not already configured:

- **TFmini Plus**: read-only firmware query, then frame-rate check; sets 100 Hz and saves only if it is not at 100 Hz.
- **WTGAHRS1**: outputs acc, gyro, angle, mag, pressure and quaternion at **100 Hz, 115200 baud** and saves that to flash
  (factory is 9600 baud / 10 Hz). The datasheet says rate and content changes take effect after **re-powering** the sensor;
  the baud change is immediate. No calibration command is ever sent.

## What the portal fuses

Two filters run behind the same inputs (page: *Filter*):

**EKF3-style (default)**, `ekf3.py`: a numpy port of the design in the simulator's `nav_ekf.py` (16-state error-state EKF, same
frames, same EKF3 process-noise defaults). It works on the *raw* calibrated gyro and accelerometer in the vehicle frame
(FRD body, NED world) and ignores the sensor's onboard attitude, like the flight computer does.

- **Predict** at the IMU rate: strapdown attitude / velocity / position, covariance every 10 ms.
- **TFmini range** (~100 Hz): slant range against height and tilt, with the TFmini lever arm. Rejected when the sensor flags it
  unreliable (strength < 100 or saturated), inside the 10 cm blind zone, beyond 8 m, tilted > 60 deg, or by a 5-sigma innovation
  gate; rejected for > 0.5 s -> the height restarts from the range. Noise from the manual's repeatability fit (section 2.4) + 2 cm floor.
- **Barometer** (20 Hz): learns its own offset while the range is good, so the height coasts on IMU + baro when the range is lost.
- **Bench aiding** (checkbox, on by default): gravity direction -> roll/pitch (skipped when |a| differs from g by > 30 %), and while
  the sensor is still a zero-velocity update plus the gyro reading as a bias measurement. This is *not* EKF3: without it tilt is
  observable only through the height (that is how EKF3 behaves without GPS or flow) and roll/pitch drift by degrees per minute.
  **Switch it off for flight-like tests**: in thrust-vectored flight the accelerometer always reads along body z and a hover looks
  "stationary".
- **Magnetic yaw** (checkbox, off): heading of the tilt-compensated calibrated field relative to the heading at power-on. Needs the
  magnetometer calibration; motors and steel will disturb it.
- Not modelled: horizontal position (no optical-flow sensor is connected, so N/E position and velocity drift and are shown as
  unaided), terrain, GPS.

**Simple vertical KF**, `fusion.py`: the earlier 4-state filter that trusts the sensor's onboard attitude; kept for comparison.

Rangefinder limits and baro noise come from `simulation/isaac/configs/sensors/fusion_tfmini_mtf01p.yaml` so the bench portal and
the simulator agree.

## Instruments on the page

- **Primary flight display, A320 style**: rectangular attitude area with the fixed roll arc and yellow sky pointer, pitch ladder
  (10 / 5 / 2.5 deg), yellow aircraft symbol and an RA readout (fused height, amber below 0.3 m); left tape = raw TFmini range
  (the "speed" slot, since a bench has no airspeed), right tape = fused height with the vertical-speed scale (+-2 m/s) beside
  it; heading scale along the bottom. Roll right = horizon rises on the right; nose up = horizon drops. The invert boxes and
  *Zero yaw* apply to it as well as to the 3-D view.
  **Heading** is the EKF yaw, relative to the power-on heading (there is no compass in the loop), unless the magnetometer has
  been calibrated: then the scale shows the absolute *magnetic* heading (cyan, no declination applied), computed from the
  tilt-compensated calibrated field and the EKF attitude, independent of the EKF's yaw origin. Motors and steel will pull it.
  The row field is `mag_hdg`.
- **Barometer**: pressure (hPa), absolute altitude, baro height (altitude minus the learned offset), the offset, the difference
  from the fused height (mean +- sigma over 10 s), the pressure noise over 3 s, and the noise the filter assumes. Two charts:
  pressure and baro height minus fused height. The sensor reports pressure in whole pascals and updates it slowly, so the pressure
  trace is stair-stepped; the absolute altitude wanders with weather, which is why only the difference is used.

## Response to rapid vertical motion

`response_check.py` (live portal, or a CSV from Export / `--log`) reports, over the moving parts: the time offset of the fused height
against the tilt-compensated raw range, the largest disagreement, and "freezes" (runs where in-band range readings were not used).
Simulator results with the sensor's ~20 Hz output filter and 20 ms of TFmini latency: the fused height lags the truth by about
10 ms (raw range: 20 ms) and errs by 1-5 cm at 1-5 m/s, with no rejections.

**Fixed 2026-09-30:** with bench aiding on, "no rotation and no acceleration" also describes a steady descent, so the zero-velocity update
held the vertical speed at 0 while the range kept falling; the range was rejected as outliers for 0.5 s (25-36 cm error at 0.3-0.5 m/s)
and the filter then reset. The zero-velocity update is now withheld while the range is changing, the estimated vertical speed is above
0.15 m/s, or the range innovations have grown; the gyro-bias half of the stationary aid is unaffected. Eight gate rejections in a row
that lie on a smooth line (real motion, not spikes) now restart the height within ~80 ms and carry the measured speed, instead of
waiting 0.5 s. `--legacy-stationary` switches both off for an A/B comparison.

**The filter assumes the IMU and the TFmini move together.** If the range changes while the IMU reports no motion (for example a bench
test where only the TFmini is moved), the filter treats the range jumps as outliers and keeps resetting; that is not a lag bug.
Test on hardware with both sensors fixed to the same rigid body.

## Calibration and mount orientation

All corrections are applied on the host, in `calibration.json` next to the scripts (`calibration_sim.json` with `--simulate`;
both git-ignored). Nothing is written to the sensor except the optional gyro auto-zero switch.

    corrected = (raw - offset) * scale          accelerometer, magnetometer
    corrected = raw - bias                      gyroscope
    vehicle   = R_trim @ R_mount @ corrected

| Procedure (page: *IMU calibration*) | How | Result |
|---|---|---|
| Gyro bias | still, 10 s | bias per axis; rejects motion |
| Accelerometer | 6 positions, each sensor axis up then down, 3 s each | offset + scale per axis (needs both faces of an axis) |
| Magnetometer | rotate through every orientation, *Finish* | hard-iron offset + per-axis scale from a sphere fit; needs >= 50 % direction coverage |
| Detect mount | hold level, capture; pitch nose up > 20 deg, capture | which sensor axes are forward / right / down (24 mounts) + level trim |
| Level trim | vehicle level and still | roll / pitch trim only |

Also on the page: pick the mount by hand from the 24 axis-aligned options, edit the trim angles, and give the TFmini's position
relative to the IMU (vehicle FRD, metres). Any change restarts the filter and is saved immediately. Samples in the first 0.5 s after
pressing a button are discarded. Suggested order: gyro bias, accelerometer 6-position, detect mount (or set it by hand), level
trim, then magnetometer if you want magnetic yaw.

**Sensor gyro auto-zero.** The WTGAHRS1 clamps its gyro output to 0 when it thinks it is still, which hides the real bias and can
snap while rotating slowly. The page can disable it (`FF AA 63 01 00`, saved to the sensor's flash; *Enable* restores it).

**Real-sensor finding.** A gyro-bias run on the real WTGAHRS1 returned exactly 0.0 dps with 0.0 noise over 952 samples: its gyro
auto-zero clamps the output to zero when still, so the bias run measures nothing. Disable auto-zero (button above) before
trusting the gyro-bias calibration or the estimated bias.

**Limits.** A tilt error and an accelerometer bias look the same to a still sensor, so the filter can only separate them after
the 6-position calibration; uncalibrated, expect a few tenths of a degree of tilt error at rest. Simulator results (biased sensor,
any mount): attitude within ~0.5 deg RMS, height within ~1-3 cm. On the real sensors only the static case has been checked.

## Files

| File | Role |
|---|---|
| `tfmini.py`, `witmotion.py` | frame/packet codecs, resyncing parsers, command builders |
| `common.py` | reader thread with reconnect, link statistics, diagnosis text |
| `detect.py` | identify sensors on the CP210x ports by content and baud |
| `init_sensors.py` | detect + configure both sensors |
| `ekf3.py` | EKF3-style 16-state filter on the raw calibrated IMU |
| `calibration.py`, `frames.py` | calibration model + fits + guided procedures; frames, 24 mounts, rotation helpers |
| `fusion.py`, `hub.py` | the simple vertical KF; joins both streams to a filter, applies calibration, keeps the rolling record |
| `sim.py` | wire-format simulator: any mount, gyro bias / accel offset+scale / mag hard-iron, still poses, "TFmini covered" every 20 s |
| `portal.html` | the page: Monitor tab (PFD, height, barometer, attitude, charts) and Setup & calibration tab; `#monitor` / `#setup` deep links |
| `portal.py` | Flask app serving `portal.html` (`/`, `/status`, `/samples`, `/samples.csv`, `POST /api/reset`, `/api/config`, `/api/calib/*`) |

## Not verified

On real hardware only the static case has been checked (level sensor, fixed range: aligned, still, height matches the TFmini,
gyro bias ~0). Not yet done on the real sensors: any calibration procedure, the mount wizard, dynamic behaviour (lifting, tilting,
covering the TFmini), magnetic yaw, and the 3-D view's sign conventions against a real tilt (use the invert boxes until they match).
The TFmini's chip temperature was reading 69 C (datasheet operating range -20..60 C); the page shows a banner above 60 C.
