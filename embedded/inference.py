"""Local Pi preprocessing and CSP+LDA inference over a raw EEG run."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import numpy as np

from embedded.deployment import load_deployment_model
from embedded.signal_message import EEGRunMessage
from embedded.signal_processing import preprocess_run


@dataclass(frozen=True)
class LocalPrediction:
    epoch_id: str
    recording_id: str
    run_id: int
    sample_index: int
    timestamp_utc: str
    predicted_class: int
    decision_score_class1: float
    model_version: str
    source: str = "REPLAY"


class PiInference:
    """Validate a versioned model and run signal processing/classification locally."""

    def __init__(self, model_path: Path, manifest_path: Path, monotonic=time.perf_counter) -> None:
        self.model, self.manifest = load_deployment_model(model_path, manifest_path)
        self.monotonic = monotonic

    def process(self, message: EEGRunMessage) -> tuple[list[LocalPrediction], dict[str, float]]:
        """Process one label-free raw recording message into Pi-generated predictions."""
        message.validate()
        if list(message.channel_names) != self.manifest["channel_names"]:
            raise ValueError("EEG channel names/order do not match deployment model")
        if message.sampling_frequency_hz != self.manifest["sampling_frequency_hz"]:
            raise ValueError("EEG sampling frequency does not match deployment model")
        start = self.monotonic()
        epochs = preprocess_run(message)
        preprocessed = self.monotonic()
        features = self.model.named_steps["csp"].transform(epochs)
        featured = self.monotonic()
        lda = self.model.named_steps["lda"]
        predictions = lda.predict(features).astype(int)
        scores = np.asarray(lda.decision_function(features), dtype=float).reshape(-1)
        classified = self.monotonic()
        recording_start = datetime.fromisoformat(message.timestamp_utc.replace("Z", "+00:00"))
        event_times = [
            recording_start + timedelta(
                seconds=sample / message.sampling_frequency_hz + float(self.manifest["preprocessing"]["epoch_tmin_s"])
            )
            for sample in message.event_samples
        ]
        first_sample = (message.sequence - 1) * 15
        output = [
            LocalPrediction(
                epoch_id=f"{message.recording_id}-E{index + 1:03d}",
                recording_id=message.recording_id,
                run_id=message.run_id,
                sample_index=first_sample + index,
                timestamp_utc=event_times[index].astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                predicted_class=int(predictions[index]),
                decision_score_class1=float(scores[index]),
                model_version=self.manifest["model_version"],
            )
            for index in range(len(predictions))
        ]
        return output, {
            "preprocessing_seconds": preprocessed - start,
            "feature_extraction_seconds": featured - preprocessed,
            "classification_seconds": classified - featured,
            "total_seconds": classified - start,
            "epoch_count": float(len(output)),
        }
