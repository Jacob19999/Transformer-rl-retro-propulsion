"""WitMotion WTGAHRS1 (WT901B-class AHRS, datasheet v20-0615): packet codec, grouping parser, config commands.

Every packet is 11 bytes ``55 <type> d0..d7 sum`` (sum = low byte of the first 10 bytes), little-endian int16:

    0x51 acceleration  a = raw/32768*16 g          | 0x52 angular rate  w = raw/32768*2000 deg/s
    0x53 angle         roll/pitch/yaw = raw/32768*180 deg (Z-Y-X, ENU: X right, Y forward, Z up)
    0x54 magnetometer  raw counts                  | 0x56 pressure Pa (int32) + barometric height cm (int32)
    0x59 quaternion    q0..q3 = raw/32768 (q0 = w) | 0x50 time, 0x57 lon/lat, 0x58 GPS speed (ignored here)

Packets arrive as a burst per output cycle; the parser groups a burst into one ``ImuSample``. Config commands are
``FF AA addr lo hi`` and need ``FF AA 69 88 B5`` first; save with ``FF AA 00 00 00``.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

G = 9.80665
PACKET_LEN = 11
DEFAULT_BAUD = 115_200
DEFAULT_HZ = 100.0
SCAN_BAUDS = (115200, 19200, 9600, 38400, 57600, 4800, 230400, 460800, 921600)
DATA_TYPES = (0x51, 0x52, 0x53, 0x54, 0x56, 0x59)  # the packets fusion uses


@dataclass(frozen=True)
class ImuSample:
    t: float  # PC arrival time of the first packet of the burst, s
    acc: tuple[float, float, float] | None  # m/s^2, body frame, includes gravity
    gyro: tuple[float, float, float] | None  # deg/s
    rpy: tuple[float, float, float] | None  # deg
    mag: tuple[int, int, int] | None  # raw counts
    pressure_pa: float | None
    baro_h_m: float | None  # barometric altitude (absolute, drifts with weather)
    quat: tuple[float, float, float, float] | None  # w, x, y, z
    temp_c: float | None


def encode_packet(ptype: int, data: bytes) -> bytes:
    body = bytes([0x55, ptype]) + data.ljust(8, b"\x00")[:8]
    return body + bytes([sum(body) & 0xFF])


def _i16(v: float) -> int:
    return max(-32768, min(32767, int(round(v))))


def encode_burst(acc_g, gyro_dps, rpy_deg, mag, pressure_pa, baro_cm, quat, temp_c=30.0) -> bytes:
    """One full output cycle in the sensor's packet order (used by the simulator and the tests)."""
    t100 = _i16(temp_c * 100)
    return (encode_packet(0x51, struct.pack("<4h", *(_i16(a / 16 * 32768) for a in acc_g), t100))
            + encode_packet(0x52, struct.pack("<4h", *(_i16(w / 2000 * 32768) for w in gyro_dps), t100))
            + encode_packet(0x53, struct.pack("<4h", *(_i16(a / 180 * 32768) for a in rpy_deg), 0x0107))
            + encode_packet(0x54, struct.pack("<4h", *mag, t100))
            + encode_packet(0x56, struct.pack("<ii", int(pressure_pa), int(baro_cm)))
            + encode_packet(0x59, struct.pack("<4h", *(_i16(q * 32768) for q in quat))))


class WitParser:
    def __init__(self) -> None:
        self._buf = bytearray()
        self._cyc: dict = {}
        self._cyc_t = 0.0
        self.samples = 0
        self.crc_errors = 0
        self.junk_bytes = 0
        self.packets: dict[int, int] = {}

    def feed(self, data: bytes, t: float) -> list[ImuSample]:
        self._buf += data
        buf, out = self._buf, []
        while len(buf) >= PACKET_LEN:
            if buf[0] != 0x55 or not 0x50 <= buf[1] <= 0x5A:
                self.junk_bytes += 1
                del buf[0]
                continue
            if sum(buf[:10]) & 0xFF != buf[10]:
                self.crc_errors += 1
                self.junk_bytes += 1
                del buf[0]
                continue
            ptype, d = buf[1], bytes(buf[2:10])
            del buf[:PACKET_LEN]
            self.packets[ptype] = self.packets.get(ptype, 0) + 1
            if ptype in self._cyc:  # type repeated: the previous burst is complete
                out.append(self._flush())
            if not self._cyc:
                self._cyc_t = t
            self._cyc[ptype] = d
        return out

    def _flush(self) -> ImuSample:
        c, self._cyc = self._cyc, {}
        self.samples += 1
        acc = gyro = rpy = mag = quat = None
        pressure = baro_h = temp = None
        if 0x51 in c:
            ax, ay, az, tr = struct.unpack("<4h", c[0x51])
            acc, temp = (ax / 32768 * 16 * G, ay / 32768 * 16 * G, az / 32768 * 16 * G), tr / 100.0
        if 0x52 in c:
            gyro = tuple(v / 32768 * 2000 for v in struct.unpack("<3h", c[0x52][:6]))
        if 0x53 in c:
            rpy = tuple(v / 32768 * 180 for v in struct.unpack("<3h", c[0x53][:6]))
        if 0x54 in c:
            mag = struct.unpack("<3h", c[0x54][:6])
        if 0x56 in c:
            p, h = struct.unpack("<ii", c[0x56])
            pressure, baro_h = float(p), h / 100.0
        if 0x59 in c:
            quat = tuple(v / 32768 for v in struct.unpack("<4h", c[0x59]))
        return ImuSample(self._cyc_t, acc, gyro, rpy, mag, pressure, baro_h, quat, temp)


# ---- configuration commands -----------------------------------------------------------------------
UNLOCK = bytes.fromhex("FFAA6988B5")
SAVE = bytes.fromhex("FFAA000000")
RATE_CODES = {0.2: 1, 0.5: 2, 1: 3, 2: 4, 5: 5, 10: 6, 20: 7, 50: 8, 100: 9, 125: 10, 200: 11}
BAUD_CODES = {4800: 1, 9600: 2, 19200: 3, 38400: 4, 57600: 5, 115200: 6, 230400: 7, 460800: 8, 921600: 9}


def config(addr: int, lo: int, hi: int = 0) -> bytes:
    return bytes([0xFF, 0xAA, addr, lo, hi])


def cmd_content(types) -> bytes:
    """RSW register: which packets the sensor outputs (0x50..0x57 in the low byte, 0x58..0x5A in the high byte)."""
    mask = sum(1 << (t - 0x50) for t in types)
    return config(0x02, mask & 0xFF, mask >> 8)


def cmd_rate(hz: float) -> bytes:
    return config(0x03, RATE_CODES[hz])


def cmd_baud(baud: int) -> bytes:
    return config(0x04, BAUD_CODES[baud])
