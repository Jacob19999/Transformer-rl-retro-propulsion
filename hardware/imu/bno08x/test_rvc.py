"""Unit tests for the RVC codec/parser: python -m pytest hardware/imu/bno08x/test_rvc.py -c /dev/null"""
import math

from rvc import FRAME_LEN, Frame, FrameParser, SimStream, encode_frame


def frame(i, **kw):
    v = dict(yaw=10.0, pitch=-5.5, roll=170.25, ax=0.5, ay=-1.0, az=9.8)
    v.update(kw)
    return encode_frame(i, **v)


def test_roundtrip_and_scaling():
    raw = frame(7)
    assert len(raw) == FRAME_LEN and raw[:2] == b"\xaa\xaa"
    (f,) = FrameParser().feed(raw)
    assert f.index == 7
    assert math.isclose(f.yaw, 10.0, abs_tol=0.01) and math.isclose(f.pitch, -5.5, abs_tol=0.01)
    assert math.isclose(f.roll, 170.25, abs_tol=0.01) and math.isclose(f.az, 9.8, abs_tol=0.01)


def test_known_frame_from_datasheet_layout():
    # index 1, yaw 90.00 deg (0x2328), pitch 0, roll 0, az 1000 mg (0x03E8): checksum over bytes 2..17
    body = bytes([1, 0x28, 0x23, 0, 0, 0, 0, 0, 0, 0, 0, 0xE8, 0x03, 0, 0, 0])
    raw = b"\xaa\xaa" + body + bytes([sum(body) & 0xFF])
    (f,) = FrameParser().feed(raw)
    assert f.yaw == 90.0 and math.isclose(f.az, 9.80665)


def test_split_across_chunks():
    p = FrameParser()
    raw = frame(0) + frame(1)
    got = []
    for i in range(0, len(raw), 5):
        got += p.feed(raw[i:i + 5])
    assert [f.index for f in got] == [0, 1] and p.crc_errors == 0


def test_resync_after_garbage_and_bad_checksum():
    p = FrameParser()
    bad = bytearray(frame(1))
    bad[-1] ^= 0xFF
    got = p.feed(b"\x00\x13\xaa" + frame(0) + bytes(bad) + b"\xaa\xaa\xaa" + frame(2))
    assert [f.index for f in got] == [0, 2]
    assert p.crc_errors >= 1 and p.frames == 2


def test_header_bytes_inside_payload():
    # yaw = 0xAAAA raw -> payload contains AA AA; frame must still parse
    raw = encode_frame(3, yaw=-218.46, pitch=0, roll=0, ax=0, ay=0, az=0)
    assert b"\xaa\xaa" in raw[2:]
    (f,) = FrameParser().feed(b"\xaa" + raw)
    assert f.index == 3 and math.isclose(f.yaw, -218.46, abs_tol=0.01)


def test_dropped_and_index_wrap():
    p = FrameParser()
    p.feed(frame(254) + frame(255) + frame(0))
    assert p.dropped == 0
    p.feed(frame(4))  # 1, 2, 3 missing
    assert p.dropped == 3


def test_simulator_output_parses_cleanly():
    s, p = SimStream(hz=500), FrameParser()
    frames = []
    while len(frames) < 50:
        frames += p.feed(s.read())
    assert p.crc_errors == 0 and p.dropped == 0
    assert all(isinstance(f, Frame) and 9.0 < math.hypot(f.ax, f.ay, f.az) < 10.6 for f in frames)
