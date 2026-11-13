"""Protocol framing, independent validation, and loopback end-to-end tests."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import socket
from threading import Thread

import numpy as np
import pytest

from bci.replay_source import load_raw_run_message, load_sequence_message
from bci.replay_control import ReplayControl
from embedded.controller import serve_connection
from embedded.deployment import build_deployment_model
from embedded.inference import PiInference
from embedded.signal_processing import preprocess_run
from mac.command_validator import MacCommandValidator
from mac.replay_server import serve_replay_session
from runtime.command import CommandEvent
from simulation.controller import ArmSimulation
from validation.command_oracle import CommandSafetyOracle
from transport.framing import recv_frame, send_frame
from transport.eeg_chunks import EEGRunAssembler, encode_chunk, run_end, run_start
from transport.protocol import json_message, parse_json_message
from scripts.run_mac_simulation_server import parse_class_sequence, repeated_messages


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw/MNE-eegbci-data/files/eegmmidb/1.0.0/S001"


def hold_envelope(session_id: str, sequence: int, timestamp: datetime | None = None) -> dict:
    event = CommandEvent(
        command_id=f"test-{sequence}", command_type="HOLD",
        timestamp=(timestamp or datetime.now(timezone.utc)).isoformat().replace("+00:00", "Z"),
        source_epoch_id=None, predicted_class=None, decision_score=None,
        confidence_measure=None, confidence_type=None, previous_state="HOLD",
        new_state="HOLD", validation_status="accepted", reason="test_hold",
        sequence=sequence, source="REPLAY",
    )
    return {"protocol_version": 1, "message_type": "command", "session_id": session_id,
            "device_id": "raspberry-pi-3b-plus", "valid_for_ms": 2000, "command": event.to_dict()}


def test_length_prefixed_frames_round_trip() -> None:
    left, right = socket.socketpair()
    try:
        send_frame(left, b"neurobot")
        assert recv_frame(right) == b"neurobot"
    finally:
        left.close()
        right.close()


def test_mac_validator_accepts_then_rejects_duplicate() -> None:
    validator = MacCommandValidator("s1")
    envelope = hold_envelope("s1", 1)
    assert validator.validate(envelope)[0] is True
    accepted, reason, _ = validator.validate(envelope)
    assert accepted is False
    assert "sequence" in reason


def test_mac_validator_requires_configured_decision_margin_for_active_command() -> None:
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    event = CommandEvent(
        command_id="active-1", command_type="HANDS", timestamp=timestamp,
        source_epoch_id="S001R06-E001", predicted_class=0, decision_score=-2.0,
        confidence_measure=2.0, confidence_type="decision_score_margin",
        previous_state="HOLD", new_state="HANDS", validation_status="accepted",
        reason="threshold_passed", sequence=1,
    )
    envelope = {"protocol_version": 1, "message_type": "command", "session_id": "active-test",
                "device_id": "raspberry-pi-3b-plus", "valid_for_ms": 2000,
                "command": event.to_dict()}
    disabled = MacCommandValidator("active-test")
    assert disabled.validate(envelope)[1] == "active_commands_disabled_on_mac"
    enabled = MacCommandValidator("active-test", active_threshold=1.5)
    assert enabled.validate(envelope)[0] is True


def test_mac_validator_rejects_unknown_session_and_stale_command() -> None:
    validator = MacCommandValidator("s1")
    wrong = hold_envelope("other", 1)
    assert validator.validate(wrong)[1] == "unknown_session"
    stale = hold_envelope("s1", 1, datetime.now(timezone.utc) - timedelta(seconds=10))
    assert validator.validate(stale)[1] == "stale_or_future_command"


def test_independent_command_oracle_matches_default_hold_policy() -> None:
    oracle = CommandSafetyOracle("s1")
    assert oracle.inspect(hold_envelope("s1", 1)) == (True, "accepted")
    active_event = CommandEvent(
        command_id="active-1", command_type="HANDS",
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        source_epoch_id="e1", predicted_class=0, decision_score=-2.0,
        confidence_measure=2.0, confidence_type="decision_score_margin",
        previous_state="HOLD", new_state="HANDS", validation_status="accepted",
        reason="threshold_passed", sequence=2,
    )
    envelope = {"protocol_version": 1, "message_type": "command", "session_id": "s1",
                "device_id": "raspberry-pi-3b-plus", "valid_for_ms": 2000,
                "command": active_event.to_dict()}
    assert oracle.inspect(envelope)[0] is False


def test_protocol_rejects_wrong_version() -> None:
    with pytest.raises(ValueError, match="protocol version"):
        parse_json_message(b'{"protocol_version":99,"message_type":"hello"}')


def test_eeg_run_chunk_assembly_and_sequence_validation() -> None:
    message = load_raw_run_message(RAW_DIR / "S001R06.edf", 6, "chunks", 1)
    chunk_samples = 1024
    assembler = EEGRunAssembler(run_start(message, chunk_samples))
    chunk_count = 0
    for index, offset in enumerate(range(0, message.signal_volts.shape[1], chunk_samples)):
        assembler.add_chunk(encode_chunk(message, index, offset, chunk_samples))
        chunk_count += 1
    restored = assembler.finish(run_end(message, chunk_count))
    np.testing.assert_array_equal(restored.signal_volts, message.signal_volts)
    assert restored.event_samples == message.event_samples


def test_eeg_assembler_rejects_out_of_order_chunk() -> None:
    message = load_raw_run_message(RAW_DIR / "S001R06.edf", 6, "chunks-order", 1)
    assembler = EEGRunAssembler(run_start(message, 512))
    second = encode_chunk(message, 1, 512, 512)
    with pytest.raises(ValueError, match="out-of-order"):
        assembler.add_chunk(second)


def test_generated_sequence_sends_raw_signal_without_requested_class_labels(tmp_path: Path) -> None:
    message = load_sequence_message(
        RAW_DIR / "S001R06.edf", 6, "sequence-test", 1, [0, 1, 1, 0],
    )
    assert message.signal_volts.shape == (64, 4 * 8 * 160)
    assert len(message.event_samples) == 4
    assert message.source_id == "mac-sequence-replay"
    assert preprocess_run(message).shape == (4, 64, 481)
    model_dir = tmp_path / "model"
    build_deployment_model(
        ROOT / "data/processed/m1/S001_epochs.npz",
        ROOT / "data/processed/m1/S001_metadata.json",
        model_dir,
    )
    predictions, timing = PiInference(
        model_dir / "csp_lda.joblib", model_dir / "manifest.json",
    ).process(message)
    assert len(predictions) == 4
    assert all(prediction.predicted_class in (0, 1) for prediction in predictions)
    assert timing["epoch_count"] == 4

    wire_header = json.loads(run_start(message, 512).decode("utf-8"))
    assert not any("label" in key.lower() or "class" in key.lower() for key in wire_header)
    assert "event_samples" in wire_header
    assert "event_id" not in wire_header and "annotations" not in wire_header
    assert "hands" not in str(wire_header).lower() and "feet" not in str(wire_header).lower()


def test_class_sequence_parser_supports_repeat_notation_and_limits() -> None:
    assert parse_class_sequence("HANDS FEET*2, hands") == [0, 1, 1, 0]
    with pytest.raises(ValueError, match="HANDS or FEET"):
        parse_class_sequence("LEFT")
    with pytest.raises(ValueError, match="limited to 15"):
        parse_class_sequence("HANDS*16")


def test_infinite_replay_stream_assigns_unique_sequences_and_recording_ids() -> None:
    message = load_sequence_message(
        RAW_DIR / "S001R06.edf", 6, "infinite-test", 1, [0, 1, 0],
    )
    stream = repeated_messages([message])
    first, second, third = [next(stream) for _ in range(3)]
    assert [first.sequence, second.sequence, third.sequence] == [1, 2, 3]
    assert len({first.recording_id, second.recording_id, third.recording_id}) == 3


def test_replay_control_pause_resume_and_stop() -> None:
    control = ReplayControl()
    control.pause()
    result: list[bool] = []
    thread = Thread(target=lambda: result.append(control.wait_until_running()), daemon=True)
    thread.start()
    assert thread.is_alive()
    control.resume()
    thread.join(timeout=1)
    assert result == [True]
    control.stop()
    assert control.wait_until_running() is False


def test_loopback_mac_pi_integration_processes_edf_and_generated_eeg_and_generates_hold(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    build_deployment_model(
        ROOT / "data/processed/m1/S001_epochs.npz",
        ROOT / "data/processed/m1/S001_metadata.json",
        model_dir,
    )
    messages = [
        load_raw_run_message(RAW_DIR / f"S001R{run:02d}.edf", run, "loopback-session", i)
        for i, run in enumerate((6, 10, 14), start=1)
    ]
    messages.append(load_sequence_message(
        RAW_DIR / "S001R06.edf", 6, "loopback-session", 4, [0, 1, 0, 1],
    ))
    runtime_config = (ROOT / "config/m3.toml").read_text(encoding="utf-8")
    runtime_config = runtime_config.replace(
        'event_log = "data/processed/m3/events.jsonl"',
        f'event_log = "{tmp_path / "pi-events.jsonl"}"',
    )
    config_path = tmp_path / "m3.toml"
    config_path.write_text(runtime_config, encoding="utf-8")

    mac_sock, pi_sock = socket.socketpair()
    mac_sock.settimeout(10)
    pi_sock.settimeout(10)
    result: dict = {}
    errors: list[BaseException] = []
    validator = MacCommandValidator("loopback-session")
    arm = ArmSimulation(ROOT / "simulation/arm.xml")

    def mac_worker() -> None:
        try:
            result.update(serve_replay_session(
                mac_sock, messages=messages, session_id="loopback-session",
                command_validator=validator,
                command_handler=arm.apply_command,
            ))
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=mac_worker, daemon=True)
    thread.start()
    try:
        send_frame(pi_sock, json_message(
            "hello", session_id="loopback-session", device_id="raspberry-pi-3b-plus"
        ))
        counts = serve_connection(
            pi_sock,
            inference=__import__("embedded.inference", fromlist=["PiInference"]).PiInference(
                model_dir / "csp_lda.joblib", model_dir / "manifest.json"
            ),
            config_path=config_path,
            project_root=ROOT,
            session_id="loopback-session",
        )
    finally:
        pi_sock.close()
    thread.join(timeout=10)
    mac_sock.close()
    assert not thread.is_alive()
    assert not errors, errors
    assert counts["epochs_processed"] == 49
    assert counts["commands_sent"] == 50
    assert result["counts"]["runs_sent"] == 4
    assert result["counts"]["commands_accepted"] == 50
    assert all(command["accepted"] for command in result["commands"])
    assert arm.status()["state"] == "HOLD"
