"""Shared serial plumbing: a reader thread that owns one port, feeds a parser and keeps link statistics."""
from __future__ import annotations

import queue
import threading
import time
from typing import Callable

import serial
from serial.tools import list_ports

CP210X = (0x10C4, 0xEA60)  # Silicon Labs CP210x, the USB-UART bridge both sensors are wired through


def cp210x_ports() -> list[str]:
    return sorted(p.device for p in list_ports.comports() if (p.vid, p.pid) == CP210X)


class SerialStream:
    def __init__(self, port: str, baud: int):
        self.port, self.baud = port, baud
        self.ser = serial.Serial(port, baud, timeout=0.05)

    def read(self) -> bytes:
        return self.ser.read(max(1, self.ser.in_waiting))

    def write(self, data: bytes) -> None:
        self.ser.write(data)

    def close(self) -> None:
        self.ser.close()


class SensorReader(threading.Thread):
    """Opens a stream via ``open_stream() -> (stream, port_label, baud)``, feeds bytes to ``parser.feed(data, t)`` and
    calls ``on_item`` for every parsed item. Reconnects after errors. ``silent_hint`` / ``garbage_hint`` are the
    diagnosis texts for "port open but no bytes" and "bytes but no valid frames"."""

    STALL_REOPEN_S = 6.0

    def __init__(self, label: str, open_stream: Callable, make_parser: Callable, on_item: Callable,
                 silent_hint: str, garbage_hint: str, nominal_hz: float):
        super().__init__(daemon=True, name=f"reader-{label}")
        self.label, self.open_stream, self.make_parser, self.on_item = label, open_stream, make_parser, on_item
        self.silent_hint, self.garbage_hint, self.nominal_hz = silent_hint, garbage_hint, nominal_hz
        self.parser = make_parser()
        self._tx: queue.Queue = queue.Queue()
        self.port: str | None = None
        self.baud: int | None = None
        self.error: str | None = None
        self.connected_at: float | None = None
        self.bytes_rx = 0
        self.count = 0
        self.conn_bytes = 0  # since the last (re)connect: what the diagnosis is about
        self.conn_items = 0
        self.last_item_t = 0.0
        self.hz = 0.0
        self._hz_t, self._hz_n = time.time(), 0

    def run(self) -> None:
        while True:
            try:
                stream, self.port, self.baud = self.open_stream()
            except (OSError, serial.SerialException, RuntimeError) as e:
                self.error = str(e)
                time.sleep(2.0)
                continue
            self.error, self.connected_at = None, time.time()
            self.conn_bytes = self.conn_items = 0
            self.parser = self.make_parser()
            try:
                while True:
                    self._drain_tx(stream)
                    data = stream.read()
                    now = time.time()
                    if now - max(self.last_item_t, self.connected_at) > self.STALL_REOPEN_S:
                        # nothing parsed for a while: reopen, which lets the opener re-detect the baud rate
                        raise OSError(f"{self.port}: no valid {self.label} frames for {self.STALL_REOPEN_S:.0f} s")
                    if not data:
                        continue
                    self.bytes_rx += len(data)
                    self.conn_bytes += len(data)
                    for item in self.parser.feed(data, now):
                        self._tick(now)
                        self.on_item(item)
            except (OSError, serial.SerialException) as e:
                self.error = f"{self.port}: {e}"
            finally:
                stream.close()
                self.connected_at = None
            time.sleep(1.0)

    def send(self, *packets: bytes) -> None:
        """Queue bytes for the sensor; the reader thread writes them (200 ms apart) so the port has a single owner."""
        for p in packets:
            self._tx.put(p)

    def _drain_tx(self, stream) -> None:
        while True:
            try:
                data = self._tx.get_nowait()
            except queue.Empty:
                return
            stream.write(data)
            time.sleep(0.2)

    def _tick(self, now: float) -> None:
        self.count += 1
        self.conn_items += 1
        self.last_item_t = now
        self._hz_n += 1
        if now - self._hz_t >= 1.0:
            self.hz, self._hz_t, self._hz_n = self._hz_n / (now - self._hz_t), now, 0

    @property
    def age_s(self) -> float | None:
        return time.time() - self.last_item_t if self.last_item_t else None

    @property
    def live(self) -> bool:
        return self.age_s is not None and self.age_s < 1.0

    def diagnosis(self) -> str:
        """One line saying what is wrong (empty when frames are flowing)."""
        if self.error:
            return self.error
        if self.connected_at is None:
            return "connecting..."
        if self.live:
            return ""
        waited = time.time() - self.connected_at
        if waited < 3.0:
            return "waiting for data..."
        if self.conn_bytes == 0:
            return self.silent_hint
        if self.conn_items == 0:
            return self.garbage_hint
        return "stream stalled (no frame for >1 s)"
