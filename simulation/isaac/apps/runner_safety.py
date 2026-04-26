"""Safety helpers for Isaac Sim command-line runners."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import traceback


class WallClockWatchdog:
    """Force-exit the process if an Isaac runner stops making progress."""

    def __init__(self, timeout_s: float | None, label: str = "Isaac runner"):
        self.timeout_s = timeout_s
        self.label = label
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._deadline: float | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.timeout_s is None or self.timeout_s <= 0:
            return
        with self._lock:
            self._deadline = time.monotonic() + float(self.timeout_s)
        self._thread = threading.Thread(target=self._run, name="isaac-runner-watchdog", daemon=True)
        self._thread.start()

    def reset(self, timeout_s: float, label: str | None = None) -> None:
        """Set a new deadline for the running watchdog."""

        with self._lock:
            self.timeout_s = timeout_s
            if label is not None:
                self.label = label
            self._deadline = time.monotonic() + float(timeout_s)

    def stop(self) -> None:
        self._done.set()

    def _run(self) -> None:
        while not self._done.wait(timeout=1.0):
            with self._lock:
                deadline = self._deadline
                timeout_s = self.timeout_s
                label = self.label
            if deadline is None or time.monotonic() < deadline:
                continue
            print(
                f"\nERROR: {label} exceeded wall-clock timeout "
                f"({timeout_s:.1f}s); forcing process exit.",
                file=sys.stderr,
                flush=True,
            )
            for thread_id, frame in sys._current_frames().items():
                print(f"\n--- stack for thread {thread_id} ---", file=sys.stderr, flush=True)
                traceback.print_stack(frame, file=sys.stderr)
            _kill_process_tree_or_exit(124)


def force_process_exit(exit_code: int) -> None:
    """Flush streams and terminate without waiting for lingering Kit threads."""

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(exit_code)


def _kill_process_tree_or_exit(exit_code: int) -> None:
    """Terminate this process and descendants, falling back to os._exit."""

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(os.getpid()), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10.0,
                check=False,
            )
        except Exception:
            pass
    os._exit(exit_code)
