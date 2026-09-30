"""Smoke-test the CP2102 <-> ESP32-CAM serial link.

Wiring (see the RandomNerdTutorials ESP32-CAM diagram):
    ESP32 5V  <-> adapter 5V/VCC      ESP32 GND <-> adapter GND
    ESP32 U0R (RX) <-> adapter TXD    ESP32 U0T (TX) <-> adapter RXD
    ESP32 IO0 <-> GND                 (only while flashing / probing the bootloader)

Modes:
    bootloader (default)  IO0 grounded. Talks to the ROM bootloader via esptool and
                          reports chip type, MAC and flash size.
    monitor               IO0 floating (jumper removed). Press RESET on the board and
                          print whatever the running firmware sends over UART0.

The ESP32-CAM has no DTR/RTS auto-reset circuit, so esptool is run with --before no-reset:
put IO0 to GND and press RESET yourself before running bootloader mode.

Usage:
    python hardware/test_esp32_link.py                 # auto-detect CP210x, bootloader probe
    python hardware/test_esp32_link.py --port COM3
    python hardware/test_esp32_link.py --mode monitor --seconds 10
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import serial
from serial.tools import list_ports

CP210X_VID_PID = (0x10C4, 0xEA60)


def find_port(requested: str | None) -> str:
    ports = list(list_ports.comports())
    if requested:
        return requested
    for p in ports:
        if (p.vid, p.pid) == CP210X_VID_PID:
            return p.device
    listing = ", ".join(f"{p.device} ({p.description})" for p in ports) or "none"
    sys.exit(f"No CP210x adapter found. Ports seen: {listing}. Pass --port explicitly.")


def run_esptool(port: str, *args: str) -> tuple[int, str]:
    cmd = [sys.executable, "-m", "esptool", "--port", port, "--before", "no-reset",
           "--after", "no-reset", "--connect-attempts", "3", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return proc.returncode, proc.stdout + proc.stderr


def bootloader_probe(port: str) -> bool:
    print(f"[bootloader] probing {port} (IO0 must be on GND; press RESET if this fails)")
    rc, out = run_esptool(port, "flash-id")
    if rc != 0:
        print(out)
        print("FAIL: no bootloader response. Check: IO0 jumpered to GND, then tap RESET; "
              "TX/RX not swapped; 5V (not 3.3V) supplying the board; USB cable carries data.")
        return False
    for line in out.splitlines():
        if line.startswith(("Chip type", "Features", "MAC", "Detected flash size", "Manufacturer", "Device")):
            print("  " + line)
    print("PASS: ESP32 bootloader answers over the CP2102 link.")
    return True


def monitor(port: str, baud: int, seconds: float) -> bool:
    print(f"[monitor] {port} @ {baud} for {seconds:.0f}s (IO0 must be floating; press RESET now)")
    received = bytearray()
    with serial.Serial(port, baud, timeout=0.2) as ser:
        ser.reset_input_buffer()
        end = time.time() + seconds
        while time.time() < end:
            chunk = ser.read(256)
            if chunk:
                received += chunk
                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                sys.stdout.flush()
    if received:
        print(f"\nPASS: received {len(received)} bytes from the ESP32.")
        return True
    print("\nFAIL: nothing received. Remove the IO0-GND jumper, press RESET, and check that "
          "the board has firmware that prints to UART0 (a blank board stays silent).")
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port (default: auto-detect CP210x)")
    ap.add_argument("--mode", choices=("bootloader", "monitor"), default="bootloader")
    ap.add_argument("--baud", type=int, default=115200, help="monitor baud rate")
    ap.add_argument("--seconds", type=float, default=8.0, help="monitor duration")
    args = ap.parse_args()

    port = find_port(args.port)
    ok = bootloader_probe(port) if args.mode == "bootloader" else monitor(port, args.baud, args.seconds)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
