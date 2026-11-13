"""Orchestration of recorded epoch replay, fold-model inference, and safety."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from runtime.event_log import EventLog
from runtime.prediction_provider import FoldModelPredictionProvider
from runtime.replay import EpochReplay
from runtime.safety import SafetyController
from runtime.state_machine import ControlState


@dataclass
class ReplayResult:
    prediction_records: list[dict[str, Any]]
    command_events: list[dict[str, Any]]
    event_records: list[dict[str, Any]]
    final_state: str


class NeuroBotReplayEngine:
    """Connect one replay source to a held-out model provider and safety controller."""

    def __init__(
        self,
        replay: EpochReplay,
        prediction_provider: FoldModelPredictionProvider,
        safety: SafetyController,
        event_log: EventLog,
        monotonic: Callable[[], float],
    ) -> None:
        self.replay = replay
        self.prediction_provider = prediction_provider
        self.safety = safety
        self.event_log = event_log
        self.monotonic = monotonic
        self.prediction_records: list[dict[str, Any]] = []

    def run(self) -> ReplayResult:
        """Replay all epochs, converting provider failures into safe ERROR state."""
        self.safety.start()
        try:
            for sample in self.replay.iter_samples(on_wait=self.safety.tick):
                self.safety.tick()
                before = len(self.event_log.records)
                try:
                    prediction = self.prediction_provider.predict(sample)
                except Exception as exc:
                    self.safety.internal_error(f"prediction provider failure: {exc}")
                    self.event_log.write("prediction_error", epoch_id=sample.epoch_id, message=str(exc))
                    break
                command = self.safety.on_prediction(prediction, now=self.monotonic())
                emitted = self.event_log.records[before:]
                confidence = next((record for record in emitted if record["event_type"] == "incoming_prediction"), {})
                stability = next((record for record in emitted if record["event_type"] == "stability_decision"), {})
                target_state = "HANDS" if prediction.predicted_class == 0 else "FEET"
                stable_accept = bool(stability.get("stable", False)) and bool(confidence.get("confidence_accepted", False))
                accepted = stable_accept and self.safety.machine.state == target_state
                reason = (
                    command.reason if command is not None
                    else stability.get("reason", confidence.get("confidence_reason", "active_state_maintained"))
                )
                self.prediction_records.append(
                    {
                        "sample_index": sample.sample_index,
                        "run_id": sample.run_id,
                        "run_epoch_index": sample.run_epoch_index,
                        "epoch_id": sample.epoch_id,
                        "replay_elapsed_seconds": sample.replay_elapsed_seconds,
                        "timestamp_utc": confidence.get("timestamp", sample.timestamp_utc),
                        "predicted_class": prediction.predicted_class,
                        "predicted_label": "hands" if prediction.predicted_class == 0 else "feet",
                        "decision_score_class1": prediction.decision_score_class1,
                        "confidence_measure": confidence.get("confidence_measure"),
                        "confidence_type": "decision_score_margin",
                        "confidence_accepted": bool(confidence.get("confidence_accepted", False)),
                        "stability_accepted": bool(stability.get("stable", False)),
                        "accepted": accepted,
                        "reason": reason,
                        "state_after": self.safety.machine.state.value,
                        "command_id": command.command_id if command else None,
                        "command_type": command.command_type if command else None,
                    }
                )
                self.event_log.write(
                    "replay_prediction_summary",
                    sample_index=sample.sample_index,
                    epoch_id=sample.epoch_id,
                    run_id=sample.run_id,
                    predicted_class=prediction.predicted_class,
                    confidence_accepted=bool(confidence.get("confidence_accepted", False)),
                    stability_accepted=bool(stability.get("stable", False)),
                    accepted=accepted,
                    state_after=self.safety.machine.state.value,
                    reason=reason,
                )
        except Exception as exc:
            self.safety.internal_error(f"replay engine failure: {exc}")
        finally:
            self.safety.source_stopped()
            if self.safety.machine.state != ControlState.ERROR:
                self.safety.shutdown()

        return ReplayResult(
            prediction_records=self.prediction_records,
            command_events=[event.to_dict() for event in self.safety.command_events],
            event_records=list(self.event_log.records),
            final_state=self.safety.machine.state.value,
        )
