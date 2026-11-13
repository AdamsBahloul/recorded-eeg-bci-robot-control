"""Mac TCP host for raw EEG replay and independently validated Pi commands."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import socket
from typing import Callable, Iterable, Any

from embedded.signal_message import EEGRunMessage
from bci.replay_control import ReplayControl
from runtime.command import CommandEvent
from transport.framing import recv_frame, send_frame
from transport.eeg_chunks import encode_chunk, run_end, run_start
from transport.protocol import json_message, parse_json_message


CommandHandler = Callable[[CommandEvent], dict[str, Any]]


def _send_ack(sock: socket.socket, **fields: Any) -> None:
    send_frame(sock, json_message("ack", **fields))


def serve_replay_session(
    sock: socket.socket,
    *,
    messages: Iterable[EEGRunMessage],
    session_id: str,
    command_validator,
    command_handler: CommandHandler,
    device_id: str = "raspberry-pi-3b-plus",
    event_sink: Callable[..., None] | None = None,
    chunk_samples: int = 512,
    replay_speed: float = 1000.0,
    replay_control: ReplayControl | None = None,
) -> dict[str, Any]:
    """Send raw runs and execute only commands accepted by the Mac validator."""
    hello = parse_json_message(recv_frame(sock), "hello")
    if hello.get("session_id") != session_id or hello.get("device_id") != device_id:
        raise ValueError("Unknown session or unauthorized Pi device identity")
    send_frame(sock, json_message("hello_ack", session_id=session_id, host_id="macbook-m4"))
    control = replay_control or ReplayControl()
    if chunk_samples < 1 or replay_speed <= 0:
        raise ValueError("chunk_samples and replay_speed must be positive")
    totals = {"runs_sent": 0, "commands_received": 0, "commands_accepted": 0, "commands_rejected": 0}
    command_history: list[dict[str, Any]] = []
    for message in messages:
        if message.session_id != session_id:
            raise ValueError("Replay EEG session does not match active connection")
        start_payload = run_start(message, chunk_samples)
        if event_sink is not None:
            event_sink(
                "eeg_signal_transmitted", session_id=session_id, recording_id=message.recording_id,
                run_id=message.run_id, sequence=message.sequence,
                sampling_frequency_hz=message.sampling_frequency_hz,
                channel_names=list(message.channel_names), signal_shape=list(message.signal_volts.shape),
                dtype=str(message.signal_volts.dtype), signal_sha256=hashlib.sha256(message.signal_volts.astype("<f8", copy=False).tobytes()).hexdigest(),
                event_boundary_count=len(message.event_samples), labels_transmitted=False,
                predictions_transmitted=False,
            )
        send_frame(sock, start_payload)
        offset = 0
        chunk_index = 0
        while offset < message.signal_volts.shape[1]:
            if not control.wait_until_running():
                raise InterruptedError("Recorded EEG replay stopped by operator")
            chunk_payload = encode_chunk(message, chunk_index, offset, chunk_samples)
            send_frame(sock, chunk_payload)
            chunk_count = min(chunk_samples, message.signal_volts.shape[1] - offset)
            if event_sink is not None:
                event_sink("eeg_chunk_transmitted", session_id=session_id, run_id=message.run_id,
                           run_sequence=message.sequence, chunk_index=chunk_index,
                           sample_offset=offset, sample_count=chunk_count,
                           sampling_frequency_hz=message.sampling_frequency_hz,
                           payload_bytes=len(chunk_payload))
            offset += chunk_count
            chunk_index += 1
            if not control.pace(chunk_count / message.sampling_frequency_hz, replay_speed):
                raise InterruptedError("Recorded EEG replay stopped by operator")
        send_frame(sock, run_end(message, chunk_index))
        totals.setdefault("chunks_sent", 0)
        totals["chunks_sent"] += chunk_index
        totals["runs_sent"] += 1
        while True:
            envelope = parse_json_message(recv_frame(sock))
            if envelope.get("message_type") == "command":
                totals["commands_received"] += 1
                command_payload = envelope.get("command")
                command_id = command_payload.get("command_id") if isinstance(command_payload, dict) else None
                if envelope.get("device_id") != device_id:
                    accepted, reason, event = False, "unauthorized_device_identity", None
                else:
                    accepted, reason, event = command_validator.validate(envelope)
                if accepted and event is not None:
                    try:
                        result = command_handler(event)
                    except Exception as exc:
                        result = {"accepted": False, "reason": f"simulation_error:{type(exc).__name__}"}
                    if result.get("accepted", True):
                        totals["commands_accepted"] += 1
                    else:
                        accepted, reason = False, str(result.get("reason", "simulation_rejected"))
                        totals["commands_rejected"] += 1
                else:
                    totals["commands_rejected"] += 1
                command_history.append({
                    "command_id": command_id,
                    "command_type": event.command_type if event else None,
                    "predicted_class": event.predicted_class if event else None,
                    "decision_score": event.decision_score if event else None,
                    "confidence_measure": event.confidence_measure if event else None,
                    "confidence_type": event.confidence_type if event else None,
                    "accepted": accepted,
                    "reason": reason,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                })
                if event_sink is not None:
                    event_sink("command_validation", command_id=command_id, accepted=accepted, reason=reason,
                               device_id=envelope.get("device_id"), command_type=command_payload.get("command_type") if isinstance(command_payload, dict) else None,
                               predicted_class=event.predicted_class if event else None,
                               decision_score=event.decision_score if event else None,
                               confidence_measure=event.confidence_measure if event else None,
                               confidence_type=event.confidence_type if event else None)
                _send_ack(sock, command_id=command_id, accepted=accepted, reason=reason)
                continue
            if envelope.get("message_type") == "run_ack":
                if envelope.get("run_id") != message.run_id:
                    raise ValueError("Pi run acknowledgement did not match transmitted recording")
                _send_ack(sock, run_id=message.run_id, accepted=True)
                heartbeat = parse_json_message(recv_frame(sock), "heartbeat")
                if heartbeat.get("session_id") != session_id or heartbeat.get("device_id") != device_id:
                    raise ValueError("Pi heartbeat identity/session mismatch")
                if heartbeat.get("run_id") != message.run_id or heartbeat.get("safety_state") not in {"HOLD", "HANDS", "FEET", "ERROR"}:
                    raise ValueError("Pi heartbeat run/state is invalid")
                if event_sink is not None:
                    event_sink("pi_heartbeat", **heartbeat)
                send_frame(sock, json_message("status_ack", run_id=message.run_id, accepted=True))
                break
            raise ValueError(f"Unexpected Pi message: {envelope.get('message_type')}")
    send_frame(sock, json_message("end_of_replay", session_id=session_id))
    final = parse_json_message(recv_frame(sock), "command")
    totals["commands_received"] += 1
    final_payload = final.get("command")
    final_command_id = final_payload.get("command_id") if isinstance(final_payload, dict) else None
    if final.get("device_id") != device_id:
        accepted, reason, event = False, "unauthorized_device_identity", None
    else:
        accepted, reason, event = command_validator.validate(final)
    if accepted and event is not None:
        try:
            result = command_handler(event)
        except Exception as exc:
            result = {"accepted": False, "reason": f"simulation_error:{type(exc).__name__}"}
        accepted = result.get("accepted", True)
        if not accepted:
            reason = str(result.get("reason", "simulation_rejected"))
    totals["commands_accepted" if accepted else "commands_rejected"] += 1
    _send_ack(sock, command_id=final_command_id, accepted=accepted, reason=reason)
    command_history.append({
        "command_id": event.command_id if event else None,
        "command_type": event.command_type if event else None,
        "predicted_class": event.predicted_class if event else None,
        "decision_score": event.decision_score if event else None,
        "confidence_measure": event.confidence_measure if event else None,
        "confidence_type": event.confidence_type if event else None,
        "accepted": accepted,
        "reason": reason,
    })
    if event_sink is not None:
        event_sink("command_validation", command_id=final_command_id, accepted=accepted, reason=reason,
                   device_id=final.get("device_id"), command_type=final_payload.get("command_type") if isinstance(final_payload, dict) else None,
                   predicted_class=event.predicted_class if event else None,
                   decision_score=event.decision_score if event else None,
                   confidence_measure=event.confidence_measure if event else None,
                   confidence_type=event.confidence_type if event else None)
    complete = parse_json_message(recv_frame(sock), "pi_session_complete")
    if complete.get("device_id") != device_id:
        raise ValueError("Pi completion identity mismatch")
    return {"counts": totals, "commands": command_history, "pi_counts": complete.get("counts", {})}
