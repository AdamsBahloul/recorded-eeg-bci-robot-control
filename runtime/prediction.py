"""Typed prediction event produced from a held-out run fold model."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    """A prediction plus signed LDA score; score is never a probability."""

    epoch_id: str
    sample_index: int
    run_id: int
    predicted_class: int
    decision_score_class1: float
    source: str = "REPLAY"
