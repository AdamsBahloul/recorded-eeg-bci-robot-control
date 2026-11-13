"""Temporal stability boundary and reset behavior."""
from __future__ import annotations

from runtime.stability import StabilityFilter


def test_requires_three_consecutive_matching_predictions() -> None:
    stability = StabilityFilter(required_consecutive=3, window_seconds=2, max_interval_seconds=1)
    assert not stability.update(0, True, 0.0).stable
    assert not stability.update(0, True, 0.2).stable
    assert stability.update(0, True, 0.4).stable


def test_class_changes_and_alternating_predictions_reset_streak() -> None:
    stability = StabilityFilter(3, 2, 1)
    for t, label in enumerate([0, 0, 1, 0, 0]):
        result = stability.update(label, True, t * 0.1)
        assert not result.stable
    assert result.count == 2


def test_invalid_confidence_resets_streak() -> None:
    stability = StabilityFilter(3, 2, 1)
    stability.update(0, True, 0.0)
    stability.update(0, True, 0.1)
    reset = stability.update(0, False, 0.2)
    assert not reset.stable and reset.count == 0
    assert stability.update(0, True, 0.3).count == 1


def test_gap_and_window_expiry_reset_with_current_prediction_as_first() -> None:
    interval_filter = StabilityFilter(3, 2.0, 0.2)
    interval_filter.update(1, True, 0.0)
    gap = interval_filter.update(1, True, 0.3)
    assert not gap.stable and gap.count == 1 and gap.reason == "prediction_interval_exceeded_reset"
    window_filter = StabilityFilter(3, 0.5, 2.0)
    window_filter.update(1, True, 0.0)
    window_filter.update(1, True, 0.3)
    expired = window_filter.update(1, True, 0.6)
    assert not expired.stable and expired.count == 1 and expired.reason == "stability_window_expired_reset"
