"""Fail-safe state, timeout, cooldown, and prediction rejection tests."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from runtime.config import ConfidenceConfig, load_runtime_config
from runtime.event_log import EventLog
from runtime.prediction import Prediction
from runtime.safety import SafetyController
from runtime.state_machine import ControlState


ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def make_controller(
    tmp_path: Path,
    *,
    enabled: bool = True,
    threshold: float = 0.5,
    cooldown: float = 0.0,
    max_active: int = 10,
    required: int = 3,
) -> tuple[SafetyController, Clock, EventLog]:
    base = load_runtime_config(ROOT / "config/m3.toml", ROOT)
    config = replace(
        base,
        confidence=ConfidenceConfig(enabled=enabled, threshold=threshold if enabled else 0.0),
        command_cooldown_seconds=cooldown,
        maximum_active_commands_per_minute=max_active,
        required_consecutive_predictions=required,
        stability_window_seconds=2.0,
        maximum_prediction_interval_seconds=1.0,
        prediction_timeout_seconds=1.0,
    )
    clock = Clock()
    log = EventLog(tmp_path / "events.jsonl")
    return SafetyController(config, log, monotonic=clock,
                            utc_now=lambda: datetime(2026, 11, 12, tzinfo=timezone.utc)), clock, log


def feed(controller: SafetyController, label: int, score: float, start: float = 0.0, step: float = 0.1):
    result = None
    for index in range(3):
        result = controller.on_prediction(
            Prediction(f"S001-R06-E{index:03d}", index, 6, label, score), now=start + index * step
        )
    return result


def test_default_disabled_gate_never_enters_active_state(tmp_path) -> None:
    controller, _, _ = make_controller(tmp_path, enabled=False)
    controller.start()
    feed(controller, 0, -1.0)
    assert controller.machine.state == ControlState.HOLD
    assert all(event.command_type == "HOLD" for event in controller.command_events)


def test_stable_hands_and_feet_can_transition_only_after_three_predictions(tmp_path) -> None:
    controller, _, _ = make_controller(tmp_path)
    controller.start()
    first = controller.on_prediction(Prediction("e0", 0, 6, 0, -0.9), now=0.0)
    second = controller.on_prediction(Prediction("e1", 1, 6, 0, -0.9), now=0.1)
    third = controller.on_prediction(Prediction("e2", 2, 6, 0, -0.9), now=0.2)
    assert first.command_type == second.command_type == "HOLD"
    assert third.command_type == "HANDS"
    assert controller.machine.state == ControlState.HANDS
    next_commands = [
        controller.on_prediction(Prediction(f"f{i}", i + 3, 6, 1, 0.9), now=0.3 + 0.1 * i)
        for i in range(3)
    ]
    assert controller.machine.state == ControlState.FEET
    assert next_commands[-1].command_type == "FEET"


def test_unstable_switch_invalid_low_confidence_and_stale_input_return_hold(tmp_path) -> None:
    controller, _, _ = make_controller(tmp_path)
    controller.start()
    feed(controller, 0, -1.0)
    assert controller.machine.state == ControlState.HANDS
    changed = controller.on_prediction(Prediction("other", 3, 6, 1, 1.0), now=0.4)
    assert changed.command_type == "HOLD"
    low = controller.on_prediction(Prediction("low", 4, 6, 0, -0.1), now=0.5)
    assert low.command_type == "HOLD"
    stale = controller.on_prediction(Prediction("old", 5, 6, 0, -1.0), now=0.6, prediction_age_seconds=2.0)
    assert stale.command_type == "HOLD"
    assert controller.machine.state == ControlState.HOLD


def test_cooldown_and_command_rate_limit_reject_reactivation(tmp_path) -> None:
    controller, _, _ = make_controller(tmp_path, cooldown=2.0)
    controller.start()
    feed(controller, 0, -1.0)
    controller.on_prediction(Prediction("switch", 3, 6, 1, 1.0), now=0.3)
    controller.on_prediction(Prediction("f1", 4, 6, 1, 1.0), now=0.4)
    rejected = controller.on_prediction(Prediction("f2", 5, 6, 1, 1.0), now=0.5)
    assert rejected.command_type == "HOLD"
    assert rejected.reason == "command_cooldown_active"
    assert controller.machine.state == ControlState.HOLD

    rate_limited, _, _ = make_controller(tmp_path / "rate", max_active=1)
    rate_limited.start()
    feed(rate_limited, 0, -1.0)
    rate_limited.on_prediction(Prediction("switch", 3, 6, 1, 1.0), now=0.3)
    rate_limited.on_prediction(Prediction("f1", 4, 6, 1, 1.0), now=0.4)
    rejected_rate = rate_limited.on_prediction(Prediction("f2", 5, 6, 1, 1.0), now=0.5)
    assert rejected_rate.command_type == "HOLD"
    assert rejected_rate.reason == "maximum_command_frequency_reached"


def test_missing_disconnect_timeout_and_eof_force_hold(tmp_path) -> None:
    controller, _, _ = make_controller(tmp_path)
    controller.start()
    feed(controller, 0, -1.0)
    assert controller.machine.state == ControlState.HANDS
    assert controller.on_missing_input().command_type == "HOLD"
    feed(controller, 1, 1.0, start=1.0)
    assert controller.machine.state == ControlState.FEET
    assert controller.source_disconnected().command_type == "HOLD"
    feed(controller, 0, -1.0, start=2.0)
    assert controller.machine.state == ControlState.HANDS
    assert controller.tick(now=3.3).command_type == "HOLD"
    feed(controller, 1, 1.0, start=4.0)
    assert controller.machine.state == ControlState.FEET
    assert controller.source_stopped().command_type == "HOLD"
    assert controller.machine.state == ControlState.HOLD


def test_error_is_safe_and_recovery_is_explicit(tmp_path) -> None:
    controller, _, _ = make_controller(tmp_path)
    controller.start()
    failure = controller.internal_error("injected failure")
    assert failure.command_type == "HOLD" and controller.machine.state == ControlState.ERROR
    assert controller.on_prediction(Prediction("ignored", 0, 6, 0, -1.0), now=0.1) is None
    assert controller.machine.state == ControlState.ERROR
    recovery = controller.recover()
    assert recovery.command_type == "HOLD"
    assert recovery.previous_state == "ERROR" and recovery.new_state == "HOLD"
    assert controller.machine.state == ControlState.HOLD


def test_every_error_or_hold_command_is_non_active_and_logged(tmp_path) -> None:
    controller, _, log = make_controller(tmp_path, enabled=False)
    controller.start()
    controller.on_prediction(object(), now=0.1)
    controller.internal_error("test")
    assert all(event.command_type == "HOLD" for event in controller.command_events if event.new_state in {"HOLD", "ERROR"})
    assert any(row["event_type"] == "state_transition" for row in log.records)
    assert (tmp_path / "events.jsonl").is_file()
