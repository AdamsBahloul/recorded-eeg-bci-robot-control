"""Confidence, stability, timeout, rate, and HOLD/ERROR safety controller."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import math
from typing import Any, Callable

from runtime.command import CommandEvent
from runtime.config import RuntimeConfig
from runtime.confidence import ConfidenceDecision, ConfidenceGate
from runtime.event_log import EventLog
from runtime.prediction import Prediction
from runtime.stability import StabilityFilter
from runtime.state_machine import ControlState, SafetyStateMachine


class SafetyController:
    """The only component allowed to create active HANDS or FEET command events."""

    def __init__(
        self,
        config: RuntimeConfig,
        event_log: EventLog,
        *,
        monotonic: Callable[[], float],
        utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        config.validate()
        self.config = config
        self.event_log = event_log
        self.monotonic = monotonic
        self.utc_now = utc_now
        self.machine = SafetyStateMachine()
        self.confidence = ConfidenceGate(config.confidence)
        self.stability = StabilityFilter(
            config.required_consecutive_predictions,
            config.stability_window_seconds,
            config.maximum_prediction_interval_seconds,
        )
        self._sequence = 0
        self._last_prediction_time: float | None = None
        self._last_valid_time: float | None = None
        self._active_commands: deque[float] = deque()
        self.command_events: list[CommandEvent] = []

    def start(self) -> CommandEvent:
        """Record a safe startup HOLD event."""
        self.event_log.write("startup", state=self.machine.state.value, source="REPLAY")
        return self._emit(
            target=ControlState.HOLD,
            prediction=None,
            confidence=None,
            status="accepted",
            reason="safe_startup",
        )

    @staticmethod
    def _iso(timestamp: datetime) -> str:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("UTC command timestamps must be timezone-aware")
        return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def _emit(
        self,
        *,
        target: ControlState,
        prediction: Any | None,
        confidence: ConfidenceDecision | None,
        status: str,
        reason: str,
        previous_state_override: ControlState | None = None,
    ) -> CommandEvent:
        previous = self.machine.state
        if target != previous:
            if target in (ControlState.HANDS, ControlState.FEET) and previous not in (ControlState.HOLD, target):
                raise RuntimeError("Active state transition must pass through HOLD")
            self.machine.transition(target)
            self.event_log.write(
                "state_transition",
                previous_state=previous.value,
                new_state=target.value,
                reason=reason,
                source_epoch_id=getattr(prediction, "epoch_id", None),
            )
        self._sequence += 1
        predicted_class = getattr(prediction, "predicted_class", None)
        command_type = target.value if target in (ControlState.HANDS, ControlState.FEET) else "HOLD"
        event = CommandEvent(
            command_id=f"m3-{self._sequence:06d}",
            command_type=command_type,
            timestamp=self._iso(self.utc_now()),
            source_epoch_id=getattr(prediction, "epoch_id", None),
            predicted_class=predicted_class,
            decision_score=confidence.decision_score if confidence else None,
            confidence_measure=confidence.confidence if confidence else None,
            confidence_type=confidence.confidence_type if confidence else None,
            previous_state=(previous_state_override or previous).value,
            new_state=target.value,
            validation_status=status,
            reason=reason,
            sequence=self._sequence,
            source="REPLAY",
        )
        event.validate()
        self.command_events.append(event)
        self.event_log.write_command(event)
        return event

    def _hold(
        self,
        reason: str,
        prediction: Any | None = None,
        confidence: ConfidenceDecision | None = None,
        status: str = "rejected",
        reset_stability: bool = True,
    ) -> CommandEvent:
        if reset_stability:
            self.stability.reset()
        self._last_valid_time = None
        target = ControlState.ERROR if self.machine.state == ControlState.ERROR else ControlState.HOLD
        return self._emit(target=target, prediction=prediction, confidence=confidence, status=status, reason=reason)

    def on_prediction(
        self,
        prediction: Any,
        *,
        now: float | None = None,
        prediction_age_seconds: float = 0.0,
    ) -> CommandEvent | None:
        """Validate and process one prediction; all rejected inputs force HOLD."""
        try:
            current = float(self.monotonic() if now is None else now)
        except (TypeError, ValueError, OverflowError):
            return self.internal_error("monotonic clock returned an invalid value")
        if not math.isfinite(current):
            return self.internal_error("monotonic clock returned a non-finite value")
        if self.machine.state == ControlState.ERROR:
            self.event_log.write("prediction_ignored", reason="system_in_error", source_epoch_id=getattr(prediction, "epoch_id", None))
            return None

        if self._last_prediction_time is not None and current < self._last_prediction_time:
            return self.internal_error("monotonic clock moved backwards")
        if (
            self._last_prediction_time is not None
            and current - self._last_prediction_time > self.config.maximum_prediction_interval_seconds
        ):
            self.stability.reset()
            self.event_log.write("prediction_interval_exceeded", seconds=current - self._last_prediction_time)
            if self.machine.state in (ControlState.HANDS, ControlState.FEET):
                self._hold("maximum_prediction_interval_exceeded", prediction)
        self._last_prediction_time = current

        try:
            decision = self.confidence.evaluate(prediction)
        except Exception as exc:
            return self.internal_error(f"confidence validation failed: {exc}")
        epoch_id = getattr(prediction, "epoch_id", None)
        self.event_log.write(
            "incoming_prediction",
            source="REPLAY",
            source_epoch_id=epoch_id,
            predicted_class=getattr(prediction, "predicted_class", None),
            raw_decision_score=decision.decision_score,
            confidence_measure=decision.confidence,
            confidence_type=decision.confidence_type,
            confidence_valid=decision.valid,
            confidence_accepted=decision.accepted,
            confidence_reason=decision.reason,
        )

        try:
            age = float(prediction_age_seconds)
        except (TypeError, ValueError, OverflowError):
            return self._hold("prediction_stale_or_invalid_age", prediction, decision)
        if not math.isfinite(age) or age < 0 or age > self.config.maximum_prediction_age_seconds:
            return self._hold("prediction_stale_or_invalid_age", prediction, decision)
        if not decision.valid or not decision.accepted:
            self.stability.update(getattr(prediction, "predicted_class", -1), False, current)
            return self._hold(decision.reason, prediction, decision)

        stable = self.stability.update(prediction.predicted_class, True, current)
        self.event_log.write(
            "stability_decision",
            source_epoch_id=prediction.epoch_id,
            stable=stable.stable,
            consecutive_count=stable.count,
            reason=stable.reason,
        )
        if not stable.stable:
            preserve_current_streak = stable.reason in {
                "stability_incomplete",
                "class_changed_reset",
                "prediction_interval_exceeded_reset",
                "stability_window_expired_reset",
            }
            return self._hold(stable.reason, prediction, decision, reset_stability=not preserve_current_streak)

        target = ControlState.HANDS if prediction.predicted_class == 0 else ControlState.FEET
        self._last_valid_time = current
        if self.machine.state == target:
            self.event_log.write("active_state_maintained", state=target.value, source_epoch_id=prediction.epoch_id)
            return None
        if self.machine.state != ControlState.HOLD:
            return self._hold("active_class_change_requires_hold", prediction, decision)
        while self._active_commands and current - self._active_commands[0] >= 60.0:
            self._active_commands.popleft()
        if self._active_commands and current - self._active_commands[-1] < self.config.command_cooldown_seconds:
            self.stability.reset()
            return self._hold("command_cooldown_active", prediction, decision)
        if len(self._active_commands) >= self.config.maximum_active_commands_per_minute:
            self.stability.reset()
            return self._hold("maximum_command_frequency_reached", prediction, decision)
        event = self._emit(
            target=target,
            prediction=prediction,
            confidence=decision,
            status="accepted",
            reason=stable.reason,
        )
        self._active_commands.append(current)
        self._last_valid_time = current
        return event

    def on_missing_input(self) -> CommandEvent:
        self.event_log.write("missing_prediction", state=self.machine.state.value)
        return self._hold("missing_input")

    def tick(self, now: float | None = None) -> CommandEvent | None:
        """Apply prediction timeout based on monotonic time."""
        try:
            current = float(self.monotonic() if now is None else now)
        except (TypeError, ValueError, OverflowError):
            return self.internal_error("monotonic clock returned an invalid value")
        if not math.isfinite(current):
            return self.internal_error("monotonic clock returned a non-finite value")
        if self.machine.state in (ControlState.HANDS, ControlState.FEET) and self._last_valid_time is not None:
            if current - self._last_valid_time >= self.config.prediction_timeout_seconds:
                self.event_log.write("prediction_timeout", elapsed_seconds=current - self._last_valid_time)
                return self._hold("prediction_timeout")
        return None

    def source_disconnected(self) -> CommandEvent:
        self.event_log.write("source_disconnected", state=self.machine.state.value)
        return self._hold("prediction_source_disconnected")

    def source_stopped(self) -> CommandEvent:
        self.event_log.write("source_stopped", state=self.machine.state.value)
        return self._hold("replay_source_stopped")

    def shutdown(self) -> CommandEvent | None:
        self.event_log.write("shutdown", state=self.machine.state.value)
        if self.machine.state == ControlState.ERROR:
            return None
        return self._hold("safe_shutdown", status="accepted")

    def internal_error(self, message: str) -> CommandEvent:
        self.stability.reset()
        self._last_valid_time = None
        previous = self.machine.state
        if self.machine.state != ControlState.ERROR:
            self.machine.transition(ControlState.ERROR)
            self.event_log.write("state_transition", previous_state=previous.value, new_state="ERROR", reason="internal_error")
        self.event_log.write("error", message=str(message))
        return self._emit(
            target=ControlState.ERROR,
            prediction=None,
            confidence=None,
            status="system_error",
            reason=f"internal_error: {message}",
            previous_state_override=previous,
        )

    def recover(self) -> CommandEvent:
        """Explicitly recover ERROR to HOLD; active commands remain disallowed."""
        previous, new = self.machine.recover()
        self.stability.reset()
        self._last_prediction_time = None
        self._last_valid_time = None
        self.event_log.write("recovery", previous_state=previous.value, new_state=new.value)
        return self._emit(
            target=ControlState.HOLD,
            prediction=None,
            confidence=None,
            status="accepted",
            reason="explicit_error_recovery",
            previous_state_override=previous,
        )
