"""Build (PlatformIO) and flash the ESP32-CAM streaming firmware over the CP2102 adapter.

Before running: jumper IO0 to GND and tap RESET (the board has no DTR/RTS auto-reset).
After flashing: REMOVE the IO0-GND jumper (IO0 is the camera XCLK pin) and tap RESET.

    python hardware/flash_camera.py               # build + flash, auto-detect CP210x
    python hardware/flash_camera.py --no-build --port COM3

PlatformIO is expected at ~/.pio-venv (pip install platformio there) or on PATH.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from serial.tools import list_ports

PROJECT = Path(__file__).resolve().parent / "esp32_cam_stream"
BUILD = PROJECT / ".pio" / "build" / "esp32cam"
BOOT_APP0 = Path.home() / ".platformio/packages/framework-arduinoespressif32/tools/partitions/boot_app0.bin"


def find_pio() -> str:
    venv_pio = Path.home() / ".pio-venv/Scripts/pio.exe"
    if venv_pio.exists():
        return str(venv_pio)
    found = shutil.which("pio")
    if not found:
        sys.exit("PlatformIO not found. Run: python -m venv ~/.pio-venv && ~/.pio-venv/Scripts/pip install platformio")
    return found


def find_port() -> str:
    for p in list_ports.comports():
        if (p.vid, p.pid) == (0x10C4, 0xEA60):
            return p.device
    sys.exit("No CP210x adapter found; pass --port.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=460800, help="flash baud rate")
    ap.add_argument("--no-build", action="store_true", help="flash the existing build")
    args = ap.parse_args()

    if not args.no_build:
        subprocess.run([find_pio(), "run", "-d", str(PROJECT)], check=True)

    parts = [("0x1000", BUILD / "bootloader.bin"), ("0x8000", BUILD / "partitions.bin"),
             ("0xe000", BOOT_APP0), ("0x10000", BUILD / "firmware.bin")]
    for _, f in parts:
        if not f.exists():
            sys.exit(f"Missing {f}; build first.")

    port = args.port or find_port()
    print(f"Flashing {port} (IO0 must be on GND; tap RESET if it cannot connect)")
    cmd = [sys.executable, "-m", "esptool", "--port", port, "--baud", str(args.baud),
           "--before", "no-reset", "--after", "no-reset", "--connect-attempts", "3",
           "write-flash", "--flash-mode", "dio", "--flash-freq", "40m", "--flash-size", "4MB"]
    for offset, f in parts:
        cmd += [offset, str(f)]
    subprocess.run(cmd, check=True)
    print("\nFlashed. Now REMOVE the IO0-GND jumper, tap RESET, then run: python hardware/camera_portal.py")


if __name__ == "__main__":
    main()
