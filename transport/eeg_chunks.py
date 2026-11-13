"""Chunked binary representation of one raw, unlabeled EEG run."""
from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any

import numpy as np

from embedded.signal_message import EEGRunMessage, SCHEMA_VERSION


CHUNK_MAGIC = b"NBCHNK1\0"


def _json_frame(kind: str, **fields: Any) -> bytes:
    return json.dumps({"schema_version": SCHEMA_VERSION, "message_type": kind, **fields},
                      separators=(",", ":"), sort_keys=True).encode("utf-8")


def run_start(message: EEGRunMessage, chunk_samples: int) -> bytes:
    """Encode run identity and unlabeled boundaries, without signal labels."""
    message.validate()
    if type(chunk_samples) is not int or chunk_samples < 1:
        raise ValueError("chunk_samples must be a positive integer")
    raw = np.asarray(message.signal_volts, dtype="<f8", order="C")
    return _json_frame(
        "eeg_run_start", session_id=message.session_id, source_id=message.source_id,
        recording_id=message.recording_id, run_id=message.run_id, sequence=message.sequence,
        timestamp_utc=message.timestamp_utc, source=message.source,
        sampling_frequency_hz=message.sampling_frequency_hz,
        channel_names=list(message.channel_names), event_samples=list(message.event_samples),
        signal_shape=list(raw.shape), dtype="float64-le", chunk_samples=chunk_samples,
        expected_chunks=math.ceil(raw.shape[1] / chunk_samples),
        signal_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
    )


def encode_chunk(message: EEGRunMessage, chunk_index: int, sample_offset: int, chunk_samples: int) -> bytes:
    """Encode one contiguous set of time samples across all EEG channels."""
    if type(chunk_index) is not int or chunk_index < 0 or type(sample_offset) is not int or sample_offset < 0:
        raise ValueError("Chunk index and sample offset must be non-negative integers")
    end = min(sample_offset + chunk_samples, message.signal_volts.shape[1])
    if sample_offset >= end:
        raise ValueError("Chunk has no signal samples")
    data = np.asarray(message.signal_volts[:, sample_offset:end], dtype="<f8", order="C")
    payload = data.tobytes()
    header = {
        "schema_version": SCHEMA_VERSION, "message_type": "eeg_signal_chunk",
        "session_id": message.session_id, "run_id": message.run_id,
        "sequence": message.sequence, "chunk_index": chunk_index,
        "sample_offset": sample_offset, "sample_count": end - sample_offset,
        "shape": list(data.shape), "dtype": "float64-le", "payload_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return CHUNK_MAGIC + struct.pack("!I", len(encoded)) + encoded + payload


def run_end(message: EEGRunMessage, chunk_count: int) -> bytes:
    """Encode completion marker and total signal integrity digest."""
    raw = np.asarray(message.signal_volts, dtype="<f8", order="C")
    return _json_frame("eeg_run_end", session_id=message.session_id, run_id=message.run_id,
                       sequence=message.sequence, total_samples=raw.shape[1],
                       total_chunks=chunk_count, signal_sha256=hashlib.sha256(raw.tobytes()).hexdigest())


class EEGRunAssembler:
    """Validate chunk ordering, sequence, shape, integrity, and assembly bounds."""

    def __init__(self, start_payload: bytes) -> None:
        try:
            header = json.loads(start_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Malformed EEG run-start header") from exc
        if header.get("schema_version") != SCHEMA_VERSION or header.get("message_type") != "eeg_run_start":
            raise ValueError("Unsupported EEG run-start schema")
        shape = tuple(header.get("signal_shape", ()))
        if (len(shape) != 2 or any(type(value) is not int for value in shape)
                or shape[0] != 64 or shape[1] < 1000 or shape[1] > 100_000):
            raise ValueError("EEG run dimensions outside supported bounds")
        if header.get("dtype") != "float64-le" or header.get("source") != "REPLAY":
            raise ValueError("Unsupported EEG dtype or source")
        chunk_samples = header.get("chunk_samples")
        if type(chunk_samples) is not int or not 1 <= chunk_samples <= 8192:
            raise ValueError("Invalid EEG chunk sample count")
        expected_chunks = header.get("expected_chunks")
        if (type(expected_chunks) is not int
                or expected_chunks != math.ceil(shape[1] / chunk_samples)):
            raise ValueError("Declared EEG chunk total is inconsistent")
        self.header = header
        self.signal = np.empty(shape, dtype="<f8")
        self.next_offset = 0
        self.next_chunk_index = 0

    def add_chunk(self, payload: bytes) -> None:
        """Validate a raw signal chunk and append it at the next contiguous offset."""
        if len(payload) < len(CHUNK_MAGIC) + 4 or not payload.startswith(CHUNK_MAGIC):
            raise ValueError("Invalid EEG chunk magic or truncated message")
        header_size = struct.unpack("!I", payload[len(CHUNK_MAGIC):len(CHUNK_MAGIC) + 4])[0]
        if header_size > 64_000:
            raise ValueError("Oversized EEG chunk metadata")
        offset = len(CHUNK_MAGIC) + 4
        header = json.loads(payload[offset:offset + header_size].decode("utf-8"))
        raw = payload[offset + header_size:]
        if header.get("schema_version") != SCHEMA_VERSION or header.get("message_type") != "eeg_signal_chunk":
            raise ValueError("Unsupported EEG signal chunk schema")
        for key in ("session_id", "run_id", "sequence"):
            if header.get(key) != self.header.get(key):
                raise ValueError(f"EEG chunk {key} mismatch")
        count = header.get("sample_count")
        if (type(header.get("chunk_index")) is not int or header["chunk_index"] != self.next_chunk_index
                or type(header.get("sample_offset")) is not int or header["sample_offset"] != self.next_offset):
            raise ValueError("Duplicate, missing, or out-of-order EEG chunk")
        if type(count) is not int or count < 1 or count > self.header["chunk_samples"]:
            raise ValueError("Invalid EEG chunk sample count")
        if tuple(header.get("shape", ())) != (64, count) or len(raw) != 64 * count * 8:
            raise ValueError("EEG chunk dimensions do not match its payload")
        if header.get("dtype") != "float64-le" or hashlib.sha256(raw).hexdigest() != header.get("sha256"):
            raise ValueError("EEG chunk checksum or dtype mismatch")
        if self.next_offset + count > self.signal.shape[1]:
            raise ValueError("EEG chunk exceeds declared run length")
        self.signal[:, self.next_offset:self.next_offset + count] = np.frombuffer(raw, dtype="<f8").reshape(64, count)
        self.next_offset += count
        self.next_chunk_index += 1

    def finish(self, end_payload: bytes) -> EEGRunMessage:
        """Require complete, uncorrupted samples and return a validated run."""
        try:
            end = json.loads(end_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Malformed EEG run-end message") from exc
        expected_chunks = self.header["expected_chunks"]
        if (end.get("message_type") != "eeg_run_end" or end.get("schema_version") != SCHEMA_VERSION
                or end.get("session_id") != self.header["session_id"]
                or end.get("run_id") != self.header["run_id"]
                or end.get("sequence") != self.header["sequence"]):
            raise ValueError("EEG run-end identity mismatch")
        if self.next_offset != self.signal.shape[1] or self.next_chunk_index != expected_chunks:
            raise ValueError("EEG run ended before all declared samples/chunks arrived")
        digest = hashlib.sha256(np.asarray(self.signal, dtype="<f8", order="C").tobytes()).hexdigest()
        if end.get("signal_sha256") != self.header.get("signal_sha256") or digest != end.get("signal_sha256"):
            raise ValueError("Assembled EEG run checksum mismatch")
        message = EEGRunMessage(
            session_id=self.header["session_id"], source_id=self.header["source_id"],
            recording_id=self.header["recording_id"], run_id=self.header["run_id"],
            sequence=self.header["sequence"], timestamp_utc=self.header["timestamp_utc"],
            sampling_frequency_hz=self.header["sampling_frequency_hz"],
            channel_names=tuple(self.header["channel_names"]),
            event_samples=tuple(self.header["event_samples"]), signal_volts=self.signal.copy(),
            source=self.header["source"], schema_version=self.header["schema_version"],
        )
        message.validate()
        return message
