"""Transport-neutral command-event schema and strict payload validation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import math
from typing import Any, Mapping

from runtime.state_machine import ControlState


@dataclass(frozen=True)
class CommandEvent:
    command_id: str
    command_type: str
    timestamp: str
    source_epoch_id: str | None
    predicted_class: int | None
    decision_score: float | None
    confidence_measure: float | None
    confidence_type: str | None
    previous_state: str
    new_state: str
    validation_status: str
    reason: str
    sequence: int
    source: str = "REPLAY"

    def validate(self) -> None:
        if not self.command_id or not self.command_id.strip():
            raise ValueError("command_id must be non-empty")
        if self.source != "REPLAY":
            raise ValueError("M3 accepts only REPLAY commands")
        try:
            parsed = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamp must be ISO-8601") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        try:
            previous = ControlState(self.previous_state)
            new = ControlState(self.new_state)
        except ValueError as exc:
            raise ValueError("previous_state and new_state must be valid control states") from exc
        if self.command_type not in {"HOLD", "HANDS", "FEET"}:
            raise ValueError("command_type must be HOLD, HANDS, or FEET")
        if self.validation_status not in {"accepted", "rejected", "system_error"}:
            raise ValueError("validation_status must be accepted, rejected, or system_error")
        if self.sequence < 1:
            raise ValueError("sequence must be positive")
        if new in (ControlState.HOLD, ControlState.ERROR) and self.command_type != "HOLD":
            raise ValueError("HOLD and ERROR states cannot carry active commands")
        if new in (ControlState.HANDS, ControlState.FEET):
            expected_class = 0 if new == ControlState.HANDS else 1
            if self.command_type != new.value or self.validation_status != "accepted":
                raise ValueError("Active states require a matching accepted command")
            if self.predicted_class != expected_class:
                raise ValueError("Active command and predicted class do not match")
            if self.confidence_type != "decision_score_margin":
                raise ValueError("Active command confidence type must be decision_score_margin")
            if self.confidence_measure is None or not math.isfinite(self.confidence_measure) or self.confidence_measure <= 0:
                raise ValueError("Active command requires a finite positive decision margin")
            if self.decision_score is None or not math.isfinite(self.decision_score):
                raise ValueError("Active command requires a finite raw decision score")
        for name, value in (("decision_score", self.decision_score), ("confidence_measure", self.confidence_measure)):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite when supplied")
        if self.predicted_class is not None and self.predicted_class not in (0, 1):
            raise ValueError("predicted_class must be class 0 or 1")
        if not self.reason:
            raise ValueError("reason must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CommandEvent":
        expected = set(cls.__dataclass_fields__)
        missing = expected - set(payload)
        extra = set(payload) - expected
        if missing or extra:
            raise ValueError(f"Malformed command fields; missing={sorted(missing)}, extra={sorted(extra)}")
        event = cls(**dict(payload))
        event.validate()
        return event
