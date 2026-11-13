"""Append-only structured JSONL event log for decisions and commands."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from runtime.command import CommandEvent


class EventLog:
    """Write structured runtime events to a separate M3 output file."""

    def __init__(self, path: Path, level: str = "INFO") -> None:
        self.path = path
        self.level = level.upper()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []

    @staticmethod
    def timestamp_utc() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def write(self, event_type: str, **fields: Any) -> dict[str, Any]:
        record = {"timestamp": self.timestamp_utc(), "event_type": event_type, **fields}
        self.records.append(record)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        return record

    def write_command(self, event: CommandEvent) -> dict[str, Any]:
        payload = event.to_dict()
        return self.write("command_event", **payload)
