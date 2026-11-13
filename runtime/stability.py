"""Consecutive prediction stability filter with deterministic timing rules."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StabilityDecision:
    stable: bool
    count: int
    reason: str


class StabilityFilter:
    """Accept only a full streak of the same class within configured time bounds."""

    def __init__(self, required_consecutive: int = 3, window_seconds: float = 2.0, max_interval_seconds: float = 1.0) -> None:
        if required_consecutive < 1:
            raise ValueError("required_consecutive must be >= 1")
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be finite and > 0")
        if not math.isfinite(max_interval_seconds) or max_interval_seconds <= 0:
            raise ValueError("max_interval_seconds must be finite and > 0")
        self.required_consecutive = required_consecutive
        self.window_seconds = float(window_seconds)
        self.max_interval_seconds = float(max_interval_seconds)
        self.reset()

    def reset(self) -> None:
        self._candidate_class: int | None = None
        self._count = 0
        self._first_time: float | None = None
        self._last_time: float | None = None
        self._fired = False

    def update(self, predicted_class: int, valid_confidence: bool, now: float) -> StabilityDecision:
        """Update the streak, clearing it on invalid confidence, class change, or time gaps."""
        if not math.isfinite(now):
            self.reset()
            return StabilityDecision(False, 0, "invalid_monotonic_time")
        if not valid_confidence or type(predicted_class) is not int or predicted_class not in (0, 1):
            self.reset()
            return StabilityDecision(False, 0, "invalid_or_low_confidence_reset")

        reason = "stability_incomplete"
        reset_to_current = False
        if self._candidate_class is not None and self._last_time is not None:
            if now < self._last_time:
                self.reset()
                return StabilityDecision(False, 0, "monotonic_time_went_backwards")
            if now - self._last_time > self.max_interval_seconds:
                reason = "prediction_interval_exceeded_reset"
                reset_to_current = True
            elif self._first_time is not None and now - self._first_time > self.window_seconds:
                reason = "stability_window_expired_reset"
                reset_to_current = True
            elif predicted_class != self._candidate_class:
                reason = "class_changed_reset"
                reset_to_current = True
        else:
            reset_to_current = True

        if reset_to_current:
            self._candidate_class = predicted_class
            self._count = 1
            self._first_time = now
            self._fired = False
        else:
            self._count += 1
        self._last_time = now

        if self._count < self.required_consecutive:
            return StabilityDecision(False, self._count, reason)
        if self._fired:
            return StabilityDecision(True, self._count, "stable_streak_continues")
        self._fired = True
        return StabilityDecision(True, self._count, "stability_threshold_reached")
