# EDF Retro-Propulsion Testbed Parts List

## Scope

This is a practical BOM-style list for the physical thrust-vectoring (jet vane) testbed used by this project.  
It is aligned to the current simulation assumptions in `simulation/configs/default_vehicle.yaml`, with the user-confirmed servo change to **MG996R**.

## Core Propulsion and Control


| Subsystem         | Qty | Target spec                                | Selected / candidate part                                           | Key specs                                               |
| ----------------- | --- | ------------------------------------------ | ------------------------------------------------------------------- | ------------------------------------------------------- |
| EDF (fan + motor) | 1   | 90 mm, 12-blade, 8S class, high thrust     | FMS 90mm 12-blade metal EDF power system                            | 8S, approx 4.9 kgf static thrust (~48 N), metal housing |
| Main ESC          | 1   | 120A class, HV/8S capable                  | Hobbywing FlyFun 120A V5 (or BLHeli_32 120A equivalent if required) | 3-8S, 120A continuous, 150A peak                        |
| Flight controller | 1   | PX4/ArduPilot compatible, vibration robust | Pixhawk 6C (Cube Orange+ alternative)                               | Redundant IMU architecture, UAV-grade IO                |
| Jet vane servos   | 4   | Metal gear, high torque                    | **MG996R** (user-selected)                                          | 4.8-7.2V, ~11 kgf*cm @6V, ~0.14 s/60 deg @6V, ~55 g     |


## Power System


| Subsystem                | Qty | Target spec                                             | Selected / candidate part                          | Key specs                                              |
| ------------------------ | --- | ------------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------ |
| Main battery             | 1-2 | 8S LiPo, discharge margin for EDF + control electronics | 8S LiPo pack (capacity/C-rating per test duration) | Nominal 29.6V; select C-rating for >=120A burst margin |
| Power/current telemetry  | 1   | High-current HV current sensing                         | Mauch HS-200-HV                                    | Hall sensor, 0-200A, up to 14S                         |
| Servo rail regulator/BEC | 1   | Stable high-current 6V rail for 4x MG996R               | HV UBEC / power module (final part TBD)            | Design for multi-amp transients; avoid brownout        |
| Charger                  | 1   | 8S-capable charging workflow                            | ToolkitRC M8D (or equivalent)                      | Dual-channel, supports up to 8S lithium chemistry      |


## Structures, Mechanisms, and Test Instrumentation


| Subsystem                | Qty   | Target spec                                        | Selected / candidate part                         | Key specs                                       |
| ------------------------ | ----- | -------------------------------------------------- | ------------------------------------------------- | ----------------------------------------------- |
| Airframe + duct + mounts | 1 set | Supports ~3.13 kg test mass and EDF loads          | Custom CAD/CAM assembly                           | Includes fin hinges, servo mounts, FC/ESC trays |
| Jet vanes (fins)         | 4     | NACA0012-like vanes in exhaust stream              | Custom machined/printed vanes                     | Chord/spans per config (~65 mm / ~55 mm)        |
| Linkage hardware         | 1 set | Low backlash control linkages                      | M2/M3 rods, ball links, horns, standoffs          | Sized for MG996R torque and vibration           |
| Thrust stand sensor      | 1     | Static thrust calibration and repeatability checks | S-type load cell + HX711 (or instrumentation amp) | Use suitable capacity and overload margin       |


## MG996R Update Impact (Important)

The original config block models a small 9g servo. With MG996R, key differences are:

- significantly higher mass per servo (~55 g vs ~10 g assumed),
- higher torque,
- slower/noisier response under load than premium digital micro servos,
- much higher peak current draw (BEC and wiring must be sized accordingly).

For simulation fidelity, update `simulation/configs/default_vehicle.yaml` servo fields:

- `fins.servo.weight_kg`
- `fins.servo.dimensions_mm`
- `fins.servo.torque`
- `fins.servo.transit_time_60deg`
- `fins.servo.max_angular_velocity`
- optionally broaden `tau_servo_range` to reflect unit-to-unit variance.

## Open Decisions

- Confirm exact EDF SKU and motor kV (config currently cites 1750 kV; common 90mm FMS listing is 1500 kV).
- Select final 8S battery capacity/C-rating based on required hover/trajectory runtime.
- Select final HV UBEC/power module for robust 6V servo rail during simultaneous vane actuation.
- Decide whether ESC must be BLHeli_32 specifically or if FlyFun-class firmware is acceptable.

