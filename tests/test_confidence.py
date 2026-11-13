"""Decision-margin validation without probability claims."""
from __future__ import annotations

from dataclasses import replace
import math

import pytest

from runtime.confidence import ConfidenceGate
from runtime.config import ConfidenceConfig
from runtime.prediction import Prediction


def active_gate(threshold: float = 0.5) -> ConfidenceGate:
    return ConfidenceGate(ConfidenceConfig(enabled=True, threshold=threshold))


def test_confidence_is_absolute_decision_score_margin() -> None:
    result = active_gate().evaluate(Prediction("epoch-1", 0, 6, 1, 0.8))
    assert result.valid and result.accepted
    assert result.confidence == 0.8
    assert result.confidence_type == "decision_score_margin"
    assert result.reason == "threshold_passed"


def test_low_margin_is_rejected_and_no_probability_range_is_assumed() -> None:
    result = active_gate().evaluate(Prediction("epoch-2", 1, 6, 0, -0.3))
    assert result.valid and not result.accepted
    assert result.confidence == 0.3
    assert result.confidence > 0 and result.confidence_type != "probability"


@pytest.mark.parametrize("score", [0.0, math.nan, math.inf, -math.inf, "bad"])
def test_invalid_scores_are_rejected(score) -> None:
    result = active_gate().evaluate(Prediction("epoch-3", 2, 6, 0, score))
    assert not result.valid and not result.accepted


def test_score_direction_source_and_class_are_validated() -> None:
    gate = active_gate()
    assert gate.evaluate(Prediction("e", 0, 6, 0, 0.8)).reason == "class_score_sign_mismatch"
    assert gate.evaluate(Prediction("e", 0, 6, 1, 0.8, source="LIVE")).reason == "unexpected_source"
    assert gate.evaluate(Prediction("e", 0, 6, 2, 0.8)).reason == "invalid_class"
    assert not gate.evaluate(object()).valid


def test_invalid_confidence_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        ConfidenceGate(ConfidenceConfig(enabled=True, threshold=0.0))
    with pytest.raises(ValueError, match="greater_equal"):
        ConfidenceGate(replace(ConfidenceConfig(), threshold_direction="less_equal"))
