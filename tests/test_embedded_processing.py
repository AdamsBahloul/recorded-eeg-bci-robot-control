"""Pi-side preprocessing and deployment compatibility tests."""
from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pytest

from bci.preprocessing import preprocess_raw
from bci.replay_source import load_raw_run_message
from embedded.deployment import build_deployment_model, load_deployment_model
from embedded.inference import PiInference
from embedded.controller import PiReplayController
from embedded.signal_message import EEGRunMessage
from embedded.signal_processing import preprocess_run
from runtime.config import load_runtime_config
from runtime.event_log import EventLog
from runtime.safety import SafetyController
from mac.command_validator import MacCommandValidator


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw/MNE-eegbci-data/files/eegmmidb/1.0.0/S001"


@pytest.fixture(scope="module")
def deployment_paths(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    directory = tmp_path_factory.mktemp("deployment")
    build_deployment_model(
        ROOT / "data/processed/m1/S001_epochs.npz",
        ROOT / "data/processed/m1/S001_metadata.json",
        directory,
    )
    return directory / "csp_lda.joblib", directory / "manifest.json"


@pytest.mark.parametrize("run_id", [6, 10, 14])
def test_pi_preprocessing_reproduces_m1_epoch_data(run_id: int) -> None:
    source = load_raw_run_message(RAW_DIR / f"S001R{run_id:02d}.edf", run_id, "parity-test", run_id)
    pi_epochs = preprocess_run(source)
    raw = mne.io.read_raw_edf(RAW_DIR / f"S001R{run_id:02d}.edf", preload=True, verbose="ERROR")
    reference, _ = preprocess_raw(raw)
    np.testing.assert_allclose(pi_epochs, reference.get_data(copy=True), rtol=1e-12, atol=1e-15)


def test_signal_message_contains_no_labels_or_predictions() -> None:
    source = load_raw_run_message(RAW_DIR / "S001R06.edf", 6, "schema-test", 1)
    restored = EEGRunMessage.from_bytes(source.to_bytes())
    assert restored.signal_volts.shape == (64, source.signal_volts.shape[1])
    assert len(restored.event_samples) == 15
    assert not hasattr(restored, "labels")
    assert not hasattr(restored, "predicted_class")


def test_signal_message_detects_corruption() -> None:
    source = load_raw_run_message(RAW_DIR / "S001R06.edf", 6, "checksum-test", 1)
    encoded = bytearray(source.to_bytes())
    encoded[-1] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        EEGRunMessage.from_bytes(bytes(encoded))


def test_deployment_model_manifest_round_trip(tmp_path: Path) -> None:
    manifest = build_deployment_model(
        ROOT / "data/processed/m1/S001_epochs.npz",
        ROOT / "data/processed/m1/S001_metadata.json",
        tmp_path,
    )
    model, loaded_manifest = load_deployment_model(tmp_path / "csp_lda.joblib", tmp_path / "manifest.json")
    assert json.loads(json.dumps(manifest)) == loaded_manifest
    assert model.named_steps["csp"].n_components == 4
    assert model.named_steps["lda"].solver == "svd"


def test_deployment_model_rejects_checksum_mismatch(tmp_path: Path) -> None:
    build_deployment_model(
        ROOT / "data/processed/m1/S001_epochs.npz",
        ROOT / "data/processed/m1/S001_metadata.json",
        tmp_path,
    )
    with (tmp_path / "csp_lda.joblib").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        load_deployment_model(tmp_path / "csp_lda.joblib", tmp_path / "manifest.json")


def test_pi_performs_local_preprocessing_and_classification(deployment_paths: tuple[Path, Path]) -> None:
    model_path, manifest_path = deployment_paths
    source = load_raw_run_message(RAW_DIR / "S001R06.edf", 6, "inference-test", 1)
    original_signal = source.signal_volts.copy()
    engine = PiInference(model_path, manifest_path)
    predictions, timings = engine.process(source)
    np.testing.assert_array_equal(source.signal_volts, original_signal)
    reference_model, _ = load_deployment_model(model_path, manifest_path)
    reference_epochs = preprocess_run(source)
    np.testing.assert_array_equal(
        np.asarray([prediction.predicted_class for prediction in predictions]),
        reference_model.predict(reference_epochs),
    )
    np.testing.assert_allclose(
        [prediction.decision_score_class1 for prediction in predictions],
        reference_model.decision_function(reference_epochs),
        rtol=1e-12, atol=1e-12,
    )
    assert len(predictions) == 15
    assert {prediction.predicted_class for prediction in predictions} <= {0, 1}
    assert all(prediction.model_version == "neurobot-csp-lda-deploy-v1" for prediction in predictions)
    assert timings["preprocessing_seconds"] > 0
    assert timings["feature_extraction_seconds"] >= 0
    assert timings["classification_seconds"] >= 0
    assert timings["total_seconds"] >= timings["preprocessing_seconds"]


def test_pi_inference_failure_generates_error_state(tmp_path: Path) -> None:
    config_text = (ROOT / "config/m3.toml").read_text(encoding="utf-8").replace(
        'event_log = "data/processed/m3/events.jsonl"',
        f'event_log = "{tmp_path / "events.jsonl"}"',
    )
    config_path = tmp_path / "m3.toml"
    config_path.write_text(config_text, encoding="utf-8")
    config = load_runtime_config(config_path, ROOT)
    safety = SafetyController(config, EventLog(tmp_path / "events.jsonl"), monotonic=lambda: 1.0)

    class FailingInference:
        def process(self, _message):
            raise RuntimeError("injected classifier fault")

    controller = PiReplayController(FailingInference(), safety)  # type: ignore[arg-type]
    message = load_raw_run_message(RAW_DIR / "S001R06.edf", 6, "error-test", 1)
    commands, timing = controller.process(message)
    assert len(commands) == 1
    assert commands[0].new_state == "ERROR"
    assert commands[0].previous_state == "HOLD"
    assert commands[0].command_type == "HOLD"
    assert timing["source_epoch_count"] == 0
    envelope = {
        "protocol_version": 1, "message_type": "command", "session_id": "error-test",
        "device_id": "raspberry-pi-3b-plus", "valid_for_ms": 2000,
        "command": commands[0].to_dict(),
    }
    assert MacCommandValidator("error-test").validate(envelope)[0] is True
