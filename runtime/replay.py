"""Deterministic sequential replay for the saved M1 epoch array."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
from typing import Callable, Iterator

import numpy as np

from bci.classification import EEGDataset


@dataclass(frozen=True)
class ReplaySample:
    """One EEG epoch in original archive order; labels are deliberately omitted."""

    epoch_id: str
    sample_index: int
    run_id: int
    run_epoch_index: int
    eeg: np.ndarray
    replay_elapsed_seconds: float
    timestamp_utc: str


class EpochReplay:
    """Replay M1 epochs with pause/resume/reset/step and optional real-time pacing."""

    def __init__(
        self,
        dataset: EEGDataset,
        delay_between_epochs_seconds: float = 0.05,
        speed: float = 1.0,
        deterministic_mode: bool = True,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if not np.isfinite(delay_between_epochs_seconds) or delay_between_epochs_seconds < 0:
            raise ValueError("Replay delay must be finite and non-negative")
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("Replay speed must be finite and greater than zero")
        self.dataset = dataset
        self.delay_between_epochs_seconds = float(delay_between_epochs_seconds)
        self.speed = float(speed)
        self.deterministic_mode = bool(deterministic_mode)
        self._sleep = sleep
        self._monotonic = monotonic
        self._utc_now = utc_now
        self._index = 0
        self._paused = False
        self._stopped = False
        self._run_indices = self._build_run_local_indices(dataset.run_ids)
        self._base_utc = utc_now()
        self._base_monotonic = monotonic()

    @staticmethod
    def _build_run_local_indices(run_ids: np.ndarray) -> list[int]:
        seen: dict[int, int] = {}
        local: list[int] = []
        for value in run_ids:
            run = int(value)
            local.append(seen.get(run, 0))
            seen[run] = seen.get(run, 0) + 1
        return local

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def complete(self) -> bool:
        return self._stopped or self._index >= len(self.dataset.X)

    @property
    def position(self) -> int:
        return self._index

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        if not self._stopped:
            self._paused = False

    def reset(self) -> None:
        self._index = 0
        self._paused = False
        self._stopped = False
        self._base_utc = self._utc_now()
        self._base_monotonic = self._monotonic()

    def stop(self) -> None:
        self._stopped = True
        self._paused = False

    def _sample_at(self, index: int) -> ReplaySample:
        run = int(self.dataset.run_ids[index])
        run_epoch_index = self._run_indices[index]
        interval = self.delay_between_epochs_seconds / self.speed
        elapsed = index * interval
        event_time = self._base_utc + timedelta(seconds=elapsed)
        return ReplaySample(
            epoch_id=f"S001-R{run:02d}-E{run_epoch_index:03d}",
            sample_index=index,
            run_id=run,
            run_epoch_index=run_epoch_index,
            eeg=self.dataset.X[index],
            replay_elapsed_seconds=elapsed,
            timestamp_utc=event_time.isoformat().replace("+00:00", "Z"),
        )

    def step(self) -> ReplaySample | None:
        """Advance exactly one epoch even while paused; useful for debugger stepping."""
        if self._stopped or self._index >= len(self.dataset.X):
            self._stopped = True
            return None
        sample = self._sample_at(self._index)
        self._index += 1
        if self._index >= len(self.dataset.X):
            self._stopped = True
        return sample

    def next_sample(self) -> ReplaySample | None:
        """Return the next epoch unless paused or finished."""
        if self._paused:
            return None
        return self.step()

    def iter_samples(self, on_wait: Callable[[], None] | None = None, poll_seconds: float = 0.05) -> Iterator[ReplaySample]:
        """Replay sequentially; on_wait runs during pacing for safety timeout checks."""
        if not np.isfinite(poll_seconds) or poll_seconds <= 0:
            raise ValueError("poll_seconds must be finite and > 0")
        previous_was_yielded = False
        while not self.complete:
            if self._paused:
                if self.deterministic_mode:
                    break
                self._sleep(min(poll_seconds, 0.05))
                if on_wait is not None:
                    on_wait()
                continue
            if previous_was_yielded and not self.deterministic_mode:
                remaining = self.delay_between_epochs_seconds / self.speed
                while remaining > 0 and not self.complete:
                    if self._paused:
                        self._sleep(min(poll_seconds, 0.05))
                        if on_wait is not None:
                            on_wait()
                        continue
                    duration = min(remaining, poll_seconds)
                    self._sleep(duration)
                    remaining -= duration
                    if on_wait is not None:
                        on_wait()
                    if self._stopped:
                        break
            sample = self.step()
            if sample is None:
                break
            previous_was_yielded = True
            yield sample

    def __iter__(self) -> Iterator[ReplaySample]:
        """Iterate samples using configured timing, without a safety callback."""
        return self.iter_samples()
