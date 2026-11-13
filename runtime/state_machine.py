"""Explicit safe control-state machine."""
from __future__ import annotations

from enum import Enum


class ControlState(str, Enum):
    HOLD = "HOLD"
    HANDS = "HANDS"
    FEET = "FEET"
    ERROR = "ERROR"


class SafetyStateMachine:
    """Permit active states only from HOLD; ERROR recovers only through HOLD."""

    def __init__(self) -> None:
        self.state = ControlState.HOLD

    def transition(self, target: ControlState) -> tuple[ControlState, ControlState]:
        if self.state == ControlState.ERROR and target not in (ControlState.ERROR, ControlState.HOLD):
            raise RuntimeError("ERROR must be explicitly recovered to HOLD before any active state")
        if target in (ControlState.HANDS, ControlState.FEET) and self.state not in (ControlState.HOLD, target):
            raise RuntimeError("Active class changes must pass through HOLD")
        previous = self.state
        self.state = target
        return previous, target

    def recover(self) -> tuple[ControlState, ControlState]:
        if self.state != ControlState.ERROR:
            raise RuntimeError("Explicit recovery is only valid from ERROR")
        return self.transition(ControlState.HOLD)
