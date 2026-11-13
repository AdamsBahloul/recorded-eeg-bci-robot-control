"""Label-free representation of one recorded EEG run sent to the Pi."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import struct

import numpy as np


MAGIC = b"NBEEG1\0\0"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EEGRunMessage:
    """Raw EEG samples and unlabeled event boundaries for one replay run."""

    session_id: str
    source_id: str
    recording_id: str
    run_id: int
    sequence: int
    timestamp_utc: str
    sampling_frequency_hz: float
    channel_names: tuple[str, ...]
    event_samples: tuple[int, ...]
    signal_volts: np.ndarray
    source: str = "REPLAY"
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        data = np.asarray(self.signal_volts)
        if self.schema_version != SCHEMA_VERSION or self.source != "REPLAY":
            raise ValueError("Unsupported schema or non-REPLAY source")
        if not self.session_id or not self.source_id or not self.recording_id:
            raise ValueError("Session, source, and recording identifiers are required")
        if self.run_id not in (6, 10, 14) or self.sequence < 1:
            raise ValueError("Only participant 1 runs 6, 10, and 14 are permitted")
        if self.sampling_frequency_hz != 160.0:
            raise ValueError("Expected 160 Hz EEG")
        if len(self.channel_names) != 64 or len(set(self.channel_names)) != 64:
            raise ValueError("Expected 64 unique EEG channel names")
        if data.ndim != 2 or data.shape[0] != 64 or data.shape[1] < 1000:
            raise ValueError("Signal must have shape (64, samples) and contain a complete recording")
        if not np.issubdtype(data.dtype, np.number) or not np.isfinite(data).all():
            raise ValueError("Signal values must be finite numeric samples")
        if not 1 <= len(self.event_samples) <= 15 or tuple(sorted(set(self.event_samples))) != self.event_samples:
            raise ValueError("Expected 1 to 15 unique, sorted, unlabeled task-event sample positions")
        if self.event_samples[0] < 0 or self.event_samples[-1] >= data.shape[1]:
            raise ValueError("Event sample position is outside the signal")
        try:
            from datetime import datetime

            parsed = datetime.fromisoformat(self.timestamp_utc.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise ValueError("timestamp_utc must be ISO-8601") from exc
        if parsed.tzinfo is None:
            raise ValueError("timestamp_utc must include a timezone")

    def to_bytes(self) -> bytes:
        """Encode metadata plus little-endian float64 samples and checksum."""
        self.validate()
        data = np.asarray(self.signal_volts, dtype="<f8", order="C")
        header = {
            "schema_version": self.schema_version,
            "message_type": "eeg_run",
            "session_id": self.session_id,
            "source_id": self.source_id,
            "recording_id": self.recording_id,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "timestamp_utc": self.timestamp_utc,
            "source": self.source,
            "sampling_frequency_hz": self.sampling_frequency_hz,
            "channel_names": list(self.channel_names),
            "shape": list(data.shape),
            "dtype": "float64-le",
            "event_samples": list(self.event_samples),
            "payload_bytes": data.nbytes,
            "sha256": hashlib.sha256(data.tobytes()).hexdigest(),
        }
        header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return MAGIC + struct.pack("!I", len(header_bytes)) + header_bytes + data.tobytes()

    @classmethod
    def from_bytes(cls, message: bytes) -> "EEGRunMessage":
        """Decode and validate the bounded binary EEG message."""
        if len(message) < len(MAGIC) + 4 or not message.startswith(MAGIC):
            raise ValueError("Invalid EEG message magic or truncated header")
        header_size = struct.unpack("!I", message[len(MAGIC):len(MAGIC) + 4])[0]
        if header_size > 1_000_000 or len(message) < len(MAGIC) + 4 + header_size:
            raise ValueError("Invalid or oversized EEG message header")
        start = len(MAGIC) + 4
        header = json.loads(message[start:start + header_size].decode("utf-8"))
        payload = message[start + header_size:]
        if header.get("message_type") != "eeg_run" or header.get("dtype") != "float64-le":
            raise ValueError("Unsupported EEG message type or dtype")
        if len(payload) != header.get("payload_bytes") or hashlib.sha256(payload).hexdigest() != header.get("sha256"):
            raise ValueError("EEG message payload size or checksum mismatch")
        shape = tuple(header.get("shape", ()))
        if len(shape) != 2 or int(np.prod(shape)) * 8 != len(payload):
            raise ValueError("EEG message shape does not match payload")
        signal = np.frombuffer(payload, dtype="<f8").reshape(shape).copy()
        event_samples = tuple(int(value) for value in header["event_samples"])
        result = cls(
            session_id=header["session_id"], source_id=header["source_id"],
            recording_id=header["recording_id"], run_id=int(header["run_id"]),
            sequence=int(header["sequence"]), timestamp_utc=header["timestamp_utc"],
            sampling_frequency_hz=float(header["sampling_frequency_hz"]),
            channel_names=tuple(header["channel_names"]), event_samples=event_samples,
            signal_volts=signal, source=header["source"], schema_version=int(header["schema_version"]),
        )
        result.validate()
        return result
