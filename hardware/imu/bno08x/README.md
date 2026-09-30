# BNO08x on a CP2102 (UART-RVC) + live web portal

A CP2102 is a USB-**UART** bridge; it cannot talk I2C, so the Adafruit_BNO08x (I2C/SPI) library cannot be used with it.
The BNO08x also speaks UART: in **UART-RVC** mode it streams yaw/pitch/roll + acceleration at 100 Hz, 115200 baud.

## Wiring (UART-RVC)

| BNO08x breakout | CP2102 adapter | note |
|---|---|---|
| **P0** | **3V** | **selects UART-RVC** (P0/P1 both low = I2C = sensor is silent) |
| P1 | leave open / GND | must stay low |
| SDA | RXD | in UART mode SDA is the sensor's TX |
| SCL | (not connected) | sensor RX; unused, RVC is output-only |
| 3V | 3V3 | |
| GND | GND | |

Re-power the sensor after changing P0 (mode is read at boot).

## Run

```
python hardware/imu/bno08x/portal.py                 # http://127.0.0.1:8001, auto-detects the CP210x
python hardware/imu/bno08x/portal.py --port COM3 --log imu.csv
python hardware/imu/bno08x/portal.py --simulate      # no hardware
python -m pytest hardware/imu/bno08x/test_rvc.py -c hardware/imu/bno08x/test_rvc.py --rootdir hardware/imu/bno08x
```

The page shows a live 3D attitude view, yaw/pitch/roll and acceleration strip charts, frame rate, bad-checksum and
dropped-frame counters, and a banner explaining what is wrong when no frames arrive. The camera portal (port 8000)
uses the same CP210x VID/PID and cannot share a COM port with this one.

## Limits of RVC mode

No gyro, magnetometer or quaternion output, and no way to configure the sensor. For those, use the sensor's SH-2/SHTP
protocol (UART-SHTP needs P1 high and 3 Mbaud, which needs a CP2102N), or I2C from a Pi/ESP32.
