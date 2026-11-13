"""Command schema validation and state-machine transition constraints."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from runtime.command import CommandEvent
from runtime.state_machine import ControlState, SafetyStateMachine


def hold_payload() -> dict:
    return {
        "command_id": "m3-000001", "command_type": "HOLD",
        "timestamp": datetime.now(timezone.utc).isoformat(), "source_epoch_id": None,
        "predicted_class": None, "decision_score": None, "confidence_measure": None,
        "confidence_type": None, "previous_state": "HOLD", "new_state": "HOLD",
        "validation_status": "rejected", "reason": "test", "sequence": 1, "source": "REPLAY",
    }


def test_hold_command_schema_round_trips() -> None:
    payload = hold_payload()
    parsed = CommandEvent.from_mapping(payload)
    assert parsed.to_dict() == payload


@pytest.mark.parametrize("mutation", [
    {"command_type": "MOVE_LEFT"},
    {"source": "LIVE"},
    {"timestamp": "not-a-date"},
    {"new_state": "HANDS"},
    {"sequence": 0},
    {"unexpected": True},
])
def test_malformed_or_unsafe_command_is_rejected(mutation) -> None:
    payload = hold_payload()
    payload.update(mutation)
    with pytest.raises(ValueError):
        CommandEvent.from_mapping(payload)


def test_state_machine_starts_hold_and_error_recovery_requires_hold() -> None:
    machine = SafetyStateMachine()
    assert machine.state == ControlState.HOLD
    machine.transition(ControlState.ERROR)
    with pytest.raises(RuntimeError, match="explicitly recovered"):
        machine.transition(ControlState.HANDS)
    machine.recover()
    assert machine.state == ControlState.HOLD
    machine.transition(ControlState.FEET)
    assert machine.state == ControlState.FEET
