"""Detect and initialise the TFmini Plus and the WTGAHRS1 for sensor fusion.

    python hardware/sensor_fusion/init_sensors.py            # detect + configure both
    python hardware/sensor_fusion/init_sensors.py --check    # detect + measure only, write nothing

TFmini Plus: read-only firmware query and a frame-rate check; it is only reconfigured if it is not at 100 Hz.
WTGAHRS1:    output the six packets fusion uses (acc, gyro, angle, mag, pressure, quaternion) at 100 Hz, 115200 baud,
             and save that to the sensor's flash (it comes from the factory at 9600 baud / 10 Hz, and 20 Hz of the full
             packet set does not fit in 19200 baud). Gyro/accelerometer/magnetometer calibration is never touched.
"""
from __future__ import annotations

import argparse
import sys
import time

import serial

import tfmini
import witmotion
from detect import find_sensors, listen

TARGET_BAUD = witmotion.DEFAULT_BAUD
TARGET_HZ = witmotion.DEFAULT_HZ


def measure_tf(port: str, baud: int, seconds: float = 2.0) -> tuple[float, tfmini.TfParser]:
    p = tfmini.TfParser()
    p.feed(listen(port, baud, seconds), 0.0)
    return p.frames / seconds, p


def measure_wit(port: str, baud: int, seconds: float = 2.0) -> tuple[float, witmotion.WitParser]:
    p = witmotion.WitParser()
    p.feed(listen(port, baud, seconds), 0.0)
    return p.samples / seconds, p


def init_tfmini(port: str, baud: int, write: bool) -> bool:
    print(f"\nTFmini Plus on {port} @ {baud}")
    with serial.Serial(port, baud, timeout=0.05) as ser:
        ser.reset_input_buffer()
        ser.write(tfmini.CMD_VERSION)
        t0, buf = time.time(), bytearray()
        while time.time() - t0 < 0.5:
            buf += ser.read(512)
    reply = tfmini.find_reply(bytes(buf), 0x01)
    if reply and len(reply) >= 7:
        v = reply[3:6]
        print(f"  firmware: V{v[2]}.{v[1]}.{v[0]}  (raw {v.hex(' ')})")
    else:
        print("  firmware query: no valid reply (streaming frames still fine)")
    rate, p = measure_tf(port, baud)
    print(f"  measured {rate:.1f} Hz, {p.frames} frames, {p.crc_errors} bad checksums, "
          f"{p.invalid} flagged unreliable (out of range / weak signal)")
    if abs(rate - tfmini.NOMINAL_HZ) > 5 and write:
        print(f"  not at {tfmini.NOMINAL_HZ:.0f} Hz -> setting frame rate and saving")
        with serial.Serial(port, baud, timeout=0.05) as ser:
            ser.write(tfmini.cmd_frame_rate(int(tfmini.NOMINAL_HZ)))
            time.sleep(0.2)
            ser.write(tfmini.CMD_SAVE)
            time.sleep(0.2)
        rate, p = measure_tf(port, baud)
        print(f"  now {rate:.1f} Hz")
    ok = abs(rate - tfmini.NOMINAL_HZ) <= 5 and p.crc_errors == 0
    print(f"  -> {'OK' if ok else 'PROBLEM'}")
    return ok


def _send(ser: serial.Serial, *cmds: bytes, gap: float = 0.15) -> None:
    for c in cmds:
        ser.write(c)
        ser.flush()
        time.sleep(gap)


def init_wit(port: str, baud: int, write: bool) -> bool:
    print(f"\nWTGAHRS1 on {port} @ {baud}")
    rate, p = measure_wit(port, baud)
    types = sorted(p.packets)
    print(f"  currently {rate:.1f} Hz, packets {[hex(t) for t in types]}, {p.crc_errors} bad checksums")
    want_types = set(witmotion.DATA_TYPES)
    done = baud == TARGET_BAUD and abs(rate - TARGET_HZ) <= 5 and set(types) == want_types
    if done:
        print("  already configured")
    elif not write:
        print("  --check: leaving configuration unchanged")
        return False
    else:
        with serial.Serial(port, baud, timeout=0.05) as ser:
            print("  setting packets + rate, saving")
            _send(ser, witmotion.UNLOCK, witmotion.cmd_content(want_types), witmotion.cmd_rate(TARGET_HZ),
                  witmotion.SAVE, gap=0.2)
            if baud != TARGET_BAUD:
                print(f"  switching baud {baud} -> {TARGET_BAUD}")
                _send(ser, witmotion.UNLOCK, witmotion.cmd_baud(TARGET_BAUD), gap=0.3)
        # The sensor changes baud immediately; reopen at the new rate and save again so it survives a power cycle.
        with serial.Serial(port, TARGET_BAUD, timeout=0.05) as ser:
            time.sleep(0.3)
            _send(ser, witmotion.UNLOCK, witmotion.SAVE, gap=0.3)
    rate, p = measure_wit(port, TARGET_BAUD)
    types = sorted(p.packets)
    ok = abs(rate - TARGET_HZ) <= 5 and set(types) == want_types and p.crc_errors == 0
    print(f"  verified at {TARGET_BAUD}: {rate:.1f} Hz, packets {[hex(t) for t in types]}, {p.crc_errors} bad checksums")
    if not ok and baud != TARGET_BAUD:
        print(f"  no valid data at {TARGET_BAUD}: power-cycle the sensor, then rerun (it may have kept {baud})")
    print(f"  -> {'OK' if ok else 'PROBLEM'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="measure only; write no configuration to either sensor")
    args = ap.parse_args()

    print("Scanning CP210x adapters (identifying sensors by their traffic)...")
    found = find_sensors()
    ok = True
    for name, fn in (("tfmini", init_tfmini), ("witmotion", init_wit)):
        if name not in found:
            print(f"\n{name}: NOT FOUND")
            ok = False
            continue
        port, baud = found[name]
        ok &= fn(port, baud, write=not args.check)
    print("\nResult:", "both sensors ready" if ok else "see problems above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
