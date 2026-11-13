"""Independent Mac-side validation of commands received from the Pi."""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

from runtime.command import CommandEvent


class MacCommandValidator:
    """Fail-closed receiver guard; it does not run or trust EEG inference."""

    def __init__(self, session_id: str, max_age_seconds: float = 2.0, active_threshold: float | None = None) -> None:
        if not session_id or not math.isfinite(max_age_seconds) or max_age_seconds <= 0:
            raise ValueError("Valid session ID and positive command age are required")
        if active_threshold is not None and (not math.isfinite(active_threshold) or active_threshold <= 0):
            raise ValueError("An active-command threshold must be finite and positive")
        self.session_id = session_id
        self.max_age_seconds = max_age_seconds
        self.active_threshold = active_threshold
        self.last_sequence = 0
        self.seen_command_ids: set[str] = set()
        self.state = "HOLD"

    def validate(self, envelope: dict[str, Any], now: datetime | None = None) -> tuple[bool, str, CommandEvent | None]:
        """Validate identity, freshness, sequence, schema, and safe transition."""
        try:
            if envelope.get("protocol_version") != 1 or envelope.get("message_type") != "command":
                raise ValueError("unsupported_protocol")
            if envelope.get("session_id") != self.session_id:
                raise ValueError("unknown_session")
            if type(envelope.get("valid_for_ms")) is not int or envelope["valid_for_ms"] < 1:
                raise ValueError("invalid_expiry")
            event = CommandEvent.from_mapping(envelope["command"])
            if event.sequence <= self.last_sequence:
                raise ValueError("duplicate_or_out_of_order_sequence")
            if event.command_id in self.seen_command_ids:
                raise ValueError("duplicate_command_id")
            issued = datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))
            current = now or datetime.now(timezone.utc)
            age = (current - issued).total_seconds()
            allowed_age = min(self.max_age_seconds, envelope["valid_for_ms"] / 1000.0)
            if age < -0.25 or age > allowed_age:
                raise ValueError("stale_or_future_command")
            if event.previous_state != self.state:
                raise ValueError("state_transition_mismatch")
            if event.new_state in ("HANDS", "FEET"):
                if self.active_threshold is None:
                    raise ValueError("active_commands_disabled_on_mac")
                if self.state != "HOLD" or event.validation_status != "accepted":
                    raise ValueError("active_transition_not_authorized")
                if event.confidence_measure is None or event.confidence_measure < self.active_threshold:
                    raise ValueError("decision_margin_below_mac_threshold")
            elif event.new_state in ("HOLD", "ERROR"):
                if event.command_type != "HOLD":
                    raise ValueError("unsafe_fallback_command")
            self.last_sequence = event.sequence
            self.seen_command_ids.add(event.command_id)
            self.state = event.new_state
            return True, "accepted", event
        except (KeyError, TypeError, ValueError) as exc:
            return False, str(exc), None
