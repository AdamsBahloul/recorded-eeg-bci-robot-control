"""Versioned JSON envelopes for session control, commands, and acknowledgements."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any


PROTOCOL_VERSION = 1


def json_message(message_type: str, **fields: Any) -> bytes:
    """Serialize a protocol v1 JSON envelope."""
    return json.dumps(
        {"protocol_version": PROTOCOL_VERSION, "message_type": message_type, **fields},
        separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")


def parse_json_message(payload: bytes, expected_type: str | None = None) -> dict[str, Any]:
    """Parse and validate a protocol v1 JSON envelope."""
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Malformed protocol JSON") from exc
    if not isinstance(value, dict) or value.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported protocol version or message shape")
    if expected_type is not None and value.get("message_type") != expected_type:
        raise ValueError(f"Expected {expected_type} message")
    return value


def utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 form."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
