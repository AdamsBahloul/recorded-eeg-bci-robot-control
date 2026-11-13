"""Thread-safe deterministic controls for recorded-run signal replay."""
from __future__ import annotations

from threading import Condition
import time


class ReplayControl:
    """Pause, resume, stop, and pace a recorded replay without changing samples."""

    def __init__(self, monotonic=time.monotonic, sleep=time.sleep) -> None:
        self._condition = Condition()
        self._paused = False
        self._stopped = False
        self._monotonic = monotonic
        self._sleep = sleep

    @property
    def paused(self) -> bool:
        with self._condition:
            return self._paused

    @property
    def stopped(self) -> bool:
        with self._condition:
            return self._stopped

    def pause(self) -> None:
        with self._condition:
            self._paused = True
            self._condition.notify_all()

    def resume(self) -> None:
        with self._condition:
            self._paused = False
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._paused = False
            self._condition.notify_all()

    def wait_until_running(self) -> bool:
        """Block while paused and return false if replay was stopped."""
        with self._condition:
            while self._paused and not self._stopped:
                self._condition.wait()
            return not self._stopped

    def pace(self, signal_duration_seconds: float, speed: float) -> bool:
        """Pace one transmitted signal chunk; speed>1 is faster than recorded time."""
        if signal_duration_seconds < 0 or speed <= 0:
            raise ValueError("Signal duration must be >=0 and replay speed must be >0")
        remaining = signal_duration_seconds / speed
        while remaining > 0:
            if not self.wait_until_running():
                return False
            interval = min(0.05, remaining)
            start = self._monotonic()
            self._sleep(interval)
            elapsed = max(0.0, self._monotonic() - start)
            if not self.paused:
                remaining -= elapsed
        return self.wait_until_running()
