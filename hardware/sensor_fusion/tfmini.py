"""Benewake TFmini Plus (UART): frame codec, resyncing parser and command helpers.

Standard output frame, 100 Hz at 115200 8N1 by default (User Manual REV 01/04/2024, section 5.3):

    59 59 | dist_L dist_H (cm) | strength_L strength_H | temp_L temp_H | checksum

checksum = sum(bytes[0:8]) mod 256; chip temperature = raw / 8 - 256 degC. When strength < 100 or == 65535 the
sensor reports distance 0 (unreliable); 0-10 cm is a blind zone. Commands are ``5A len id payload checksum``.
"""
from __future__ import annotations

from dataclasses import dataclass

HEADER = b"\x59\x59"
FRAME_LEN = 9
BAUD = 115_200
NOMINAL_HZ = 100.0
MIN_STRENGTH = 100
SATURATED = 65535


@dataclass(frozen=True)
class Frame:
    t: float  # PC arrival time, s
    dist_m: float
    strength: int
    temp_c: float

    @property
    def valid(self) -> bool:
        return self.dist_m > 0.0 and MIN_STRENGTH <= self.strength < SATURATED


def encode_frame(dist_cm: int, strength: int, temp_c: float = 30.0) -> bytes:
    raw_t = int(round((temp_c + 256.0) * 8.0))
    body = HEADER + bytes([dist_cm & 0xFF, dist_cm >> 8 & 0xFF, strength & 0xFF, strength >> 8 & 0xFF,
                           raw_t & 0xFF, raw_t >> 8 & 0xFF])
    return body + bytes([sum(body) & 0xFF])


class TfParser:
    """Feed arbitrary byte chunks; hunts for 59 59, verifies the checksum and resyncs after garbage
    (including command replies, which start with 5A)."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self.frames = 0
        self.crc_errors = 0
        self.junk_bytes = 0
        self.invalid = 0  # checksum fine but the sensor flagged the reading unreliable

    def feed(self, data: bytes, t: float) -> list[Frame]:
        self._buf += data
        buf, out = self._buf, []
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
            if sum(buf[:FRAME_LEN - 1]) & 0xFF != buf[FRAME_LEN - 1]:
                self.crc_errors += 1
                self.junk_bytes += 1
                del buf[:1]
                continue
            dist = buf[2] | buf[3] << 8
            strength = buf[4] | buf[5] << 8
            temp = (buf[6] | buf[7] << 8) / 8.0 - 256.0
            del buf[:FRAME_LEN]
            f = Frame(t, dist / 100.0, strength, temp)
            self.frames += 1
            self.invalid += not f.valid
            out.append(f)
        return out


# ---- commands ------------------------------------------------------------------------------------
def command(cmd_id: int, payload: bytes = b"") -> bytes:
    body = bytes([0x5A, 4 + len(payload), cmd_id]) + payload
    return body + bytes([sum(body) & 0xFF])


CMD_VERSION = command(0x01)  # 5A 04 01 5F
CMD_SAVE = command(0x11)  # 5A 04 11 6F


def cmd_frame_rate(hz: int) -> bytes:
    return command(0x03, bytes([hz & 0xFF, hz >> 8 & 0xFF]))


def find_reply(data: bytes, cmd_id: int) -> bytes | None:
    """Return the first checksum-valid ``5A len id ...`` reply for ``cmd_id`` in a byte stream."""
    for i in range(len(data) - 3):
        if data[i] == 0x5A and data[i + 2] == cmd_id:
            n = data[i + 1]
            if 4 <= n <= 32 and i + n <= len(data) and sum(data[i:i + n - 1]) & 0xFF == data[i + n - 1]:
                return bytes(data[i:i + n])
    return None
