"""TCP client loop for the Pi-side EEG inference and safety controller."""
from __future__ import annotations

from pathlib import Path
import json
import socket
import time
from typing import Any, Callable

from embedded.inference import PiInference
from embedded.signal_message import EEGRunMessage
from transport.eeg_chunks import EEGRunAssembler
from runtime.command import CommandEvent
from runtime.config import load_runtime_config
from runtime.event_log import EventLog
from runtime.prediction import Prediction
from runtime.safety import SafetyController
from transport.framing import recv_frame, send_frame
from transport.protocol import json_message, parse_json_message


class PiReplayController:
    """Process raw messages locally and return Pi-generated safety commands."""

    def __init__(self, inference: PiInference, safety: SafetyController) -> None:
        self.inference = inference
        self.safety = safety

    def process(
        self,
        message: EEGRunMessage,
        *,
        on_command: Callable[[CommandEvent], None] | None = None,
        prediction_interval_seconds: float = 0.0,
    ) -> tuple[list[CommandEvent], dict[str, Any]]:
        """Classify a run, pacing OOF-style decisions and streaming commands if requested."""
        try:
            predictions, timing = self.inference.process(message)
        except Exception as exc:
            event = self.safety.internal_error(f"Pi inference/preprocessing failure: {type(exc).__name__}")
            self.safety.event_log.write(
                "pi_processing_error", recording_id=message.recording_id, run_id=message.run_id,
                error_type=type(exc).__name__, message=str(exc),
            )
            if on_command is not None:
                on_command(event)
            return [event], {
                "preprocessing_seconds": 0.0,
                "feature_extraction_seconds": 0.0,
                "classification_seconds": 0.0,
                "total_seconds": 0.0,
                "epoch_count": 0.0,
                "generated_command_count": 1,
                "source_epoch_count": 0,
            }
        commands: list[CommandEvent] = []
        if prediction_interval_seconds < 0:
            raise ValueError("prediction_interval_seconds must be non-negative")
        for index, prediction in enumerate(predictions):
            if index and prediction_interval_seconds:
                time.sleep(prediction_interval_seconds)
            self.safety.event_log.write(
                "pi_inference_result", source="REPLAY", epoch_id=prediction.epoch_id,
                recording_id=prediction.recording_id, run_id=prediction.run_id,
                sample_index=prediction.sample_index, signal_timestamp_utc=prediction.timestamp_utc,
                predicted_class=prediction.predicted_class,
                raw_decision_score_class1=prediction.decision_score_class1,
                model_version=prediction.model_version,
            )
            result = self.safety.on_prediction(
                Prediction(
                    epoch_id=prediction.epoch_id,
                    sample_index=prediction.sample_index,
                    run_id=prediction.run_id,
                    predicted_class=prediction.predicted_class,
                    decision_score_class1=prediction.decision_score_class1,
                    source=prediction.source,
                ),
                prediction_age_seconds=0.0,
            )
            if result is not None:
                commands.append(result)
                if on_command is not None:
                    on_command(result)
        timing["generated_command_count"] = len(commands)
        timing["source_epoch_count"] = len(predictions)
        self.safety.event_log.write(
            "pi_run_processing_complete", source="REPLAY", recording_id=message.recording_id,
            run_id=message.run_id, sample_count=int(message.signal_volts.shape[1]),
            signal_shape=list(message.signal_volts.shape), **timing,
        )
        return commands, timing


def serve_connection(
    sock: socket.socket,
    *,
    inference: PiInference,
    config_path: Path,
    project_root: Path,
    session_id: str,
    device_id: str = "raspberry-pi-3b-plus",
) -> dict[str, Any]:
    """Serve one Mac session; only signal frames enter local inference."""
    config = load_runtime_config(config_path, project_root)
    event_log = EventLog(config.event_log_path, config.logging_level)
    safety = SafetyController(config, event_log, monotonic=time.monotonic)
    controller = PiReplayController(inference, safety)
    hello = parse_json_message(recv_frame(sock), "hello_ack")
    if hello.get("session_id") != session_id:
        raise ValueError("MacBook acknowledged the wrong session")
    safety.start()
    counts = {"runs_received": 0, "epochs_processed": 0, "commands_sent": 0}

    def send_command_event(event: CommandEvent) -> None:
        envelope = {
            "protocol_version": 1, "message_type": "command", "session_id": session_id,
            "device_id": device_id, "valid_for_ms": 2000, "command": event.to_dict(),
        }
        send_frame(sock, json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode())
        ack = parse_json_message(recv_frame(sock), "ack")
        if ack.get("command_id") != event.command_id:
            raise ValueError("MacBook rejected or mismatched Pi ERROR command acknowledgement")
        counts["commands_sent"] += 1

    while True:
        try:
            payload = recv_frame(sock)
        except (ConnectionError, OSError, TimeoutError) as exc:
            safety.source_disconnected()
            event_log.write("pi_mac_connection_lost", error_type=type(exc).__name__)
            return counts
        try:
            control = parse_json_message(payload)
        except ValueError:
            control = None
        if control and control.get("message_type") == "end_of_replay":
            break
        try:
            assembler = EEGRunAssembler(payload)
            for _ in range(assembler.header["expected_chunks"]):
                assembler.add_chunk(recv_frame(sock))
            signal = assembler.finish(recv_frame(sock))
            if signal.session_id != session_id:
                raise ValueError("EEG signal session mismatch")
        except Exception as exc:
            event = safety.internal_error(f"Pi EEG receiver failure: {type(exc).__name__}")
            event_log.write("pi_eeg_receiver_error", error_type=type(exc).__name__, message=str(exc))
            send_command_event(event)
            return counts
        counts["runs_received"] += 1
        commands, timing = controller.process(
            signal,
            on_command=send_command_event,
            prediction_interval_seconds=config.replay_delay_seconds / config.replay_speed,
        )
        counts["epochs_processed"] += int(timing["source_epoch_count"])
        send_frame(sock, json_message("run_ack", run_id=signal.run_id, timing=timing))
        run_ack = parse_json_message(recv_frame(sock), "ack")
        if run_ack.get("run_id") != signal.run_id:
            raise ValueError("MacBook run acknowledgement mismatch")
        send_frame(sock, json_message(
            "heartbeat", session_id=session_id, device_id=device_id,
            run_id=signal.run_id, epochs_processed=timing["source_epoch_count"],
            safety_state=safety.machine.state.value,
            model_version=inference.manifest["model_version"],
        ))
        status_ack = parse_json_message(recv_frame(sock), "status_ack")
        if status_ack.get("run_id") != signal.run_id:
            raise ValueError("MacBook status acknowledgement mismatch")
    if (safety.machine.state.value in {"HANDS", "FEET"}
            and config.post_active_display_seconds > 0):
        event_log.write(
            "active_display_hold_started",
            state=safety.machine.state.value,
            duration_seconds=config.post_active_display_seconds,
        )
        time.sleep(config.post_active_display_seconds)
    shutdown = safety.shutdown()
    if shutdown is not None:
        envelope = {
            "protocol_version": 1, "message_type": "command", "session_id": session_id,
            "device_id": device_id, "valid_for_ms": 2000, "command": shutdown.to_dict(),
        }
        send_frame(sock, json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode())
        ack = parse_json_message(recv_frame(sock), "ack")
        if ack.get("command_id") != shutdown.command_id:
            raise ValueError("MacBook shutdown acknowledgement mismatch")
        counts["commands_sent"] += 1
    send_frame(sock, json_message("pi_session_complete", device_id=device_id, counts=counts))
    return counts


def connect_and_serve(
    host: str, port: int, *, model_path: Path, manifest_path: Path, config_path: Path,
    project_root: Path, session_id: str, timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Connect to the Mac simulation host and run one recorded EEG session."""
    from embedded.inference import PiInference

    inference = PiInference(model_path, manifest_path)
    with socket.create_connection((host, port), timeout=timeout_seconds) as connection:
        connection.settimeout(None)
        send_frame(connection, json_message("hello", session_id=session_id, device_id="raspberry-pi-3b-plus"))
        return serve_connection(
            connection, inference=inference, config_path=config_path,
            project_root=project_root, session_id=session_id,
        )
