"""BNO08x UART-RVC ("Robot Vacuum Cleaner") driver: frame codec, resyncing parser, reader thread, simulator.

The BNO08x streams one 19-byte frame every 10 ms (100 Hz) at 115200 baud, output only, no commands:

    AA AA | index | yaw pitch roll (int16 LE, 0.01 deg) | ax ay az (int16 LE, 1 mg) | 3 reserved | checksum

checksum = sum(bytes[2:18]) mod 256 (index through the reserved bytes). Layout and scaling follow
Adafruit_CircuitPython_BNO08x_RVC. RVC mode has no gyro, magnetometer or quaternion output.
"""
from __future__ import annotations

import collections
import csv
import math
import random
import struct
import threading
import time
from dataclasses import dataclass

import serial
from serial.tools import list_ports

HEADER = b"\xaa\xaa"
FRAME_LEN = 19
BAUD = 115_200
NOMINAL_HZ = 100.0
DEG_PER_LSB = 0.01
MPS2_PER_LSB = 0.00980665  # 1 mg in m/s^2
BODY = struct.Struct("<Bhhhhhh3B")  # index, yaw, pitch, roll, ax, ay, az, 3 reserved (17 bytes with checksum)
CSV_FIELDS = ("pc_time_s", "n", "t_dev_s", "index", "yaw_deg", "pitch_deg", "roll_deg", "ax_mps2", "ay_mps2", "az_mps2")


@dataclass(frozen=True)
class Frame:
    index: int
    yaw: float  # deg
    pitch: float  # deg
    roll: float  # deg
    ax: float  # m/s^2 (includes gravity)
    ay: float
    az: float


def _clip16(v: float) -> int:
    return max(-32768, min(32767, int(round(v))))


def encode_frame(index: int, yaw: float, pitch: float, roll: float, ax: float, ay: float, az: float) -> bytes:
    """Build a valid RVC frame (used by the simulator and the tests)."""
    body = BODY.pack(index & 0xFF, _clip16(yaw / DEG_PER_LSB), _clip16(pitch / DEG_PER_LSB),
                     _clip16(roll / DEG_PER_LSB), _clip16(ax / MPS2_PER_LSB), _clip16(ay / MPS2_PER_LSB),
                     _clip16(az / MPS2_PER_LSB), 0, 0, 0)
    return HEADER + body + bytes([sum(body) & 0xFF])


class FrameParser:
    """Feed it arbitrary byte chunks; it hunts for AA AA, verifies the checksum and resyncs after garbage."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self.frames = 0
        self.crc_errors = 0  # header found but checksum wrong
        self.junk_bytes = 0  # bytes skipped while hunting
        self.dropped = 0  # frames missing according to the 8-bit index counter
        self._last_index: int | None = None

    def feed(self, data: bytes) -> list[Frame]:
        self._buf += data
        out: list[Frame] = []
        buf = self._buf
        while True:
            i = buf.find(HEADER)
            if i < 0:
                keep = 1 if buf.endswith(HEADER[:1]) else 0
                self.junk_bytes += len(buf) - keep
                del buf[:len(buf) - keep]
                break
            if i:
                self.junk_bytes += i
                del buf[:i]
            if len(buf) < FRAME_LEN:
                break
            if sum(buf[2:FRAME_LEN - 1]) & 0xFF != buf[FRAME_LEN - 1]:
                self.crc_errors += 1
                self.junk_bytes += 1
                del buf[:1]  # the AA AA may have been payload; retry from the next byte
                continue
            idx, yaw, pitch, roll, ax, ay, az, *_ = BODY.unpack(bytes(buf[2:FRAME_LEN - 1]))
            del buf[:FRAME_LEN]
            if self._last_index is not None:
                gap = (idx - self._last_index - 1) & 0xFF
                if 0 < gap < 128:  # a bigger "gap" is a repeat/reorder, not loss
                    self.dropped += gap
            self._last_index = idx
            self.frames += 1
            out.append(Frame(idx, yaw * DEG_PER_LSB, pitch * DEG_PER_LSB, roll * DEG_PER_LSB,
                             ax * MPS2_PER_LSB, ay * MPS2_PER_LSB, az * MPS2_PER_LSB))
        return out


# ---- byte sources ------------------------------------------------------------------------------
class SerialStream:
    def __init__(self, port: str, baud: int = BAUD):
        self.port = port
        self.ser = serial.Serial(port, baud, timeout=0.1)

    def read(self) -> bytes:
        return self.ser.read(max(1, self.ser.in_waiting))

    def close(self) -> None:
        self.ser.close()


class SimStream:
    """Emits real RVC byte frames at 100 Hz from a synthetic tumbling motion (no hardware needed)."""

    def __init__(self, hz: float = NOMINAL_HZ):
        self.dt = 1.0 / hz
        self.t0 = self._next = time.perf_counter()
        self.index = 0
        self.rng = random.Random(1)

    def read(self) -> bytes:
        time.sleep(max(0.0, self._next - time.perf_counter()) + 0.005)
        out = bytearray()
        now = time.perf_counter()
        while self._next <= now:
            t = self._next - self.t0
            self._next += self.dt
            yaw = ((30.0 * t + 180.0) % 360.0) - 180.0
            pitch = 35.0 * math.sin(0.7 * t)
            roll = 50.0 * math.sin(0.45 * t + 1.0)
            p, r, g = math.radians(pitch), math.radians(roll), 9.80665
            n = lambda: self.rng.gauss(0.0, 0.04)  # noqa: E731
            out += encode_frame(self.index, yaw, pitch, roll, -g * math.sin(p) + n(),
                                g * math.sin(r) * math.cos(p) + n(), g * math.cos(r) * math.cos(p) + n())
            self.index = (self.index + 1) & 0xFF
        return bytes(out)

    def close(self) -> None:
        pass


def find_port() -> str | None:
    for p in list_ports.comports():
        if (p.vid, p.pid) == (0x10C4, 0xEA60):  # Silicon Labs CP210x
            return p.device
    return None


# ---- reader thread -----------------------------------------------------------------------------
class RvcReader(threading.Thread):
    def __init__(self, port: str | None, simulate: bool = False, log_path: str | None = None, history: int = 6000):
        super().__init__(daemon=True)
        self.port_arg, self.simulate = port, simulate
        self.port: str | None = "simulator" if simulate else port
        self.parser = FrameParser()
        self.lock = threading.Lock()
        self.samples: collections.deque = collections.deque(maxlen=history)  # (n, t_dev, index, y, p, r, ax, ay, az)
        self.count = 0
        self.bytes_rx = 0
        self.error: str | None = None
        self.connected_at: float | None = None
        self.last_frame_t = 0.0
        self.hz = 0.0
        self._hz_t, self._hz_n = time.time(), 0
        self._t_dev = 0.0
        self._last_index: int | None = None
        self._csv_file = open(log_path, "w", newline="") if log_path else None
        self._csv = csv.writer(self._csv_file) if self._csv_file else None
        if self._csv:
            self._csv.writerow(CSV_FIELDS)

    def _open(self):
        if self.simulate:
            return SimStream()
        port = self.port_arg or find_port()
        if port is None:
            raise OSError("no CP210x adapter found (pass --port)")
        self.port = port
        return SerialStream(port)

    def run(self) -> None:
        while True:
            try:
                stream = self._open()
            except (OSError, serial.SerialException) as e:
                self.error = str(e)
                time.sleep(2.0)
                continue
            self.error, self.connected_at = None, time.time()
            self.parser = FrameParser()
            self._last_index = None
            try:
                while True:
                    data = stream.read()
                    if data:
                        self.bytes_rx += len(data)
                        for f in self.parser.feed(data):
                            self._on_frame(f)
            except (OSError, serial.SerialException) as e:
                self.error = f"{self.port}: {e}"
            finally:
                stream.close()
                self.connected_at = None
            time.sleep(1.0)

    def _on_frame(self, f: Frame) -> None:
        now = time.time()
        if self._last_index is None:
            self._t_dev = 0.0
        else:
            self._t_dev += ((f.index - self._last_index) & 0xFF or 256) / NOMINAL_HZ  # unwrap the 8-bit counter
        self._last_index = f.index
        self._hz_n += 1
        if now - self._hz_t >= 1.0:
            self.hz, self._hz_t, self._hz_n = self._hz_n / (now - self._hz_t), now, 0
        row = (self.count + 1, round(self._t_dev, 3), f.index, round(f.yaw, 2), round(f.pitch, 2), round(f.roll, 2),
               round(f.ax, 3), round(f.ay, 3), round(f.az, 3))
        with self.lock:
            self.samples.append(row)
            self.count += 1
            self.last_frame_t = now
        if self._csv:
            self._csv.writerow((f"{now:.4f}", row[0], f"{row[1]:.3f}", *row[2:]))
            if self.count % 100 == 0:
                self._csv_file.flush()

    # ---- views for the web layer ---------------------------------------------------------------
    def since(self, after: int, limit: int = 600) -> list:
        with self.lock:
            return [s for s in self.samples if s[0] > after][-limit:]

    def diagnosis(self) -> str:
        """One line telling the user what is wrong (empty when frames are flowing)."""
        if self.error:
            return self.error
        if self.connected_at is None:
            return "connecting..."
        live = self.last_frame_t and time.time() - self.last_frame_t < 1.0
        if live:
            return ""
        waited = time.time() - self.connected_at
        if waited < 3.0:
            return "waiting for data..."
        if self.bytes_rx == 0:
            return ("port open but the sensor is silent. The BNO08x powers up in I2C mode, which a CP2102 cannot "
                    "read: tie P0 to 3V (UART-RVC), wire sensor SDA -> adapter RXD, then re-power the sensor.")
        if self.parser.frames == 0:
            return "bytes arriving but no valid RVC frames: wrong mode (P0 high, P1 low) or wrong baud (needs 115200)."
        return "stream stalled (no frame for >1 s)"
