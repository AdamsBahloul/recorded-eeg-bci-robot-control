"""Reference command-history policy independent of runtime safety code."""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any


EXPECTED_FIELDS = {
    "command_id", "command_type", "timestamp", "source_epoch_id", "predicted_class",
    "decision_score", "confidence_measure", "confidence_type", "previous_state",
    "new_state", "validation_status", "reason", "sequence", "source",
}


class CommandSafetyOracle:
    """Small independent history checker used only by tests and judge reports."""

    def __init__(self, session_id: str, active_threshold: float | None = None, maximum_age_s: float = 2.0) -> None:
        self.session_id = session_id
        self.threshold = active_threshold
        self.maximum_age_s = maximum_age_s
        self.sequence = 0
        self.state = "HOLD"

    def inspect(self, message: dict[str, Any], now: datetime | None = None) -> tuple[bool, str]:
        try:
            if message.get("protocol_version") != 1 or message.get("message_type") != "command":
                return False, "protocol"
            if message.get("session_id") != self.session_id or message.get("device_id") != "raspberry-pi-3b-plus":
                return False, "identity"
            if type(message.get("valid_for_ms")) is not int or message["valid_for_ms"] < 1:
                return False, "expiry"
            event = message["command"]
            if not isinstance(event, dict) or set(event) != EXPECTED_FIELDS:
                return False, "schema"
            seq = event["sequence"]
            if type(seq) is not int or seq <= self.sequence:
                return False, "sequence"
            timestamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                return False, "timestamp_timezone"
            age = ((now or datetime.now(timezone.utc)) - timestamp).total_seconds()
            if age < -0.25 or age > min(self.maximum_age_s, message["valid_for_ms"] / 1000):
                return False, "stale"
            if event["source"] != "REPLAY" or event["previous_state"] != self.state:
                return False, "source_or_state"
            command, new_state = event["command_type"], event["new_state"]
            if command not in {"HOLD", "HANDS", "FEET"} or new_state not in {"HOLD", "HANDS", "FEET", "ERROR"}:
                return False, "command_enum"
            if new_state in {"HOLD", "ERROR"}:
                if command != "HOLD":
                    return False, "unsafe_fallback"
            else:
                if self.threshold is None or self.state != "HOLD":
                    return False, "active_disabled_or_transition"
                expected_class = 0 if command == "HANDS" else 1 if command == "FEET" else -1
                margin = event["confidence_measure"]
                score = event["decision_score"]
                if (event["validation_status"] != "accepted" or event["predicted_class"] != expected_class
                        or event["confidence_type"] != "decision_score_margin"
                        or isinstance(margin, bool) or not isinstance(margin, (int, float))
                        or not math.isfinite(margin) or margin < self.threshold
                        or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)):
                    return False, "margin_or_class"
            self.sequence = seq
            self.state = new_state
            return True, "accepted"
        except (KeyError, TypeError, ValueError, OverflowError):
            return False, "malformed"
