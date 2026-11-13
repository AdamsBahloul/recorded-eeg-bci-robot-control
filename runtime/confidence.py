"""Validation and confidence measurement for binary LDA decision scores."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from runtime.config import ConfidenceConfig
from runtime.prediction import Prediction


@dataclass(frozen=True)
class ConfidenceDecision:
    valid: bool
    accepted: bool
    confidence: float | None
    confidence_type: str
    decision_score: float | None
    reason: str


class ConfidenceGate:
    """Convert an LDA score to an absolute margin; never a probability."""

    confidence_type = "decision_score_margin"

    def __init__(self, config: ConfidenceConfig) -> None:
        if config.mode != self.confidence_type:
            raise ValueError("Only decision_score_margin confidence is supported")
        if config.threshold_direction != "greater_equal":
            raise ValueError("Decision-score margin threshold direction must be greater_equal")
        if not math.isfinite(config.threshold):
            raise ValueError("Confidence threshold must be finite")
        if config.enabled and config.threshold <= 0:
            raise ValueError("Enabled confidence threshold must be explicitly configured and positive (> 0)")
        if not config.enabled and config.threshold != 0:
            raise ValueError("Disabled confidence gate requires threshold=0")
        self.config = config

    def evaluate(self, prediction: Any) -> ConfidenceDecision:
        """Validate a prediction and threshold its absolute signed LDA margin."""
        if not isinstance(prediction, Prediction):
            return ConfidenceDecision(False, False, None, self.confidence_type, None, "malformed_prediction")
        try:
            if isinstance(prediction.decision_score_class1, bool):
                raise ValueError("boolean score")
            score = float(prediction.decision_score_class1)
        except (TypeError, ValueError, OverflowError):
            return ConfidenceDecision(False, False, None, self.confidence_type, None, "invalid_decision_score")
        if prediction.source != "REPLAY":
            return ConfidenceDecision(False, False, None, self.confidence_type, score, "unexpected_source")
        if type(prediction.predicted_class) is not int or prediction.predicted_class not in (0, 1):
            return ConfidenceDecision(False, False, None, self.confidence_type, score, "invalid_class")
        if not math.isfinite(score) or score == 0.0:
            return ConfidenceDecision(False, False, None, self.confidence_type, score, "non_finite_or_ambiguous_score")
        if (prediction.predicted_class == 1 and score < 0) or (prediction.predicted_class == 0 and score > 0):
            return ConfidenceDecision(False, False, None, self.confidence_type, score, "class_score_sign_mismatch")
        margin = abs(score)
        if not self.config.enabled:
            return ConfidenceDecision(True, False, margin, self.confidence_type, score, "active_commands_disabled")
        if margin < self.config.threshold:
            return ConfidenceDecision(True, False, margin, self.confidence_type, score, "below_decision_margin_threshold")
        return ConfidenceDecision(True, True, margin, self.confidence_type, score, "threshold_passed")
