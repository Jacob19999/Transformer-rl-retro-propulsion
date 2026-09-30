"""Work out which CP210x adapter carries which sensor by listening: COM numbers change between plugs.

    python hardware/sensor_fusion/detect.py
"""
from __future__ import annotations

import time

import serial

import tfmini
import witmotion
from common import cp210x_ports

LISTEN_S = 0.7
MIN_HITS = 5


def listen(port: str, baud: int, seconds: float = LISTEN_S) -> bytes:
    with serial.Serial(port, baud, timeout=0.05) as ser:
        ser.reset_input_buffer()
        t0, buf = time.time(), bytearray()
        while time.time() - t0 < seconds:
            buf += ser.read(4096)
    return bytes(buf)


def classify(data: bytes) -> str | None:
    tf = tfmini.TfParser()
    tf.feed(data, 0.0)
    wit = witmotion.WitParser()
    wit.feed(data, 0.0)
    wit_hits = sum(wit.packets.values())
    if tf.frames >= MIN_HITS and tf.frames > wit_hits:
        return "tfmini"
    if wit_hits >= MIN_HITS and not wit.crc_errors > wit_hits:
        return "witmotion"
    return None


def probe(port: str) -> tuple[str, int] | None:
    """(sensor, baud) for the first baud rate at which the port's traffic parses as a known sensor."""
    for baud in witmotion.SCAN_BAUDS:
        kind = classify(listen(port, baud))
        if kind:
            return kind, baud
    return None


def find_sensors(ports: list[str] | None = None, log=print) -> dict[str, tuple[str, int]]:
    """{'tfmini': (port, baud), 'witmotion': (port, baud)} for whatever answers."""
    found: dict[str, tuple[str, int]] = {}
    for port in ports or cp210x_ports():
        try:
            hit = probe(port)
        except serial.SerialException as e:
            log(f"{port}: cannot open ({e})")
            continue
        log(f"{port}: {f'{hit[0]} @ {hit[1]} baud' if hit else 'no known sensor traffic'}")
        if hit and hit[0] not in found:
            found[hit[0]] = (port, hit[1])
    return found


if __name__ == "__main__":
    find_sensors()
