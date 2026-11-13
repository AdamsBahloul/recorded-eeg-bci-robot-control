"""Prediction provider that selects the model for the epoch's held-out run."""
from __future__ import annotations

from typing import Mapping

import numpy as np
from sklearn.pipeline import Pipeline

from bci.classification import EEGDataset
from runtime.prediction import Prediction
from runtime.replay import ReplaySample


class FoldModelPredictionProvider:
    """Use each run's model trained only on the other two runs."""

    def __init__(self, models_by_validation_run: Mapping[int, Pipeline], model_version: str = "csp4-lda-svd-m2-oof") -> None:
        self._models = dict(models_by_validation_run)
        self.model_version = model_version
        if set(self._models) != {6, 10, 14}:
            raise ValueError("Provider requires exactly the held-out models for runs 6, 10, and 14")

    def predict(self, sample: ReplaySample) -> Prediction:
        """Predict one epoch; the provider receives features but never true labels."""
        if sample.run_id not in self._models:
            raise ValueError(f"No held-out fold model for run {sample.run_id}")
        if sample.eeg.ndim != 2 or not np.isfinite(sample.eeg).all():
            raise ValueError(f"Malformed EEG epoch {sample.epoch_id}")
        model = self._models[sample.run_id]
        x = sample.eeg[np.newaxis, ...]
        predicted = int(model.predict(x)[0])
        score = float(np.asarray(model.decision_function(x)).reshape(-1)[0])
        if predicted not in (0, 1) or not np.isfinite(score):
            raise ValueError(f"Invalid classifier result for {sample.epoch_id}")
        return Prediction(
            epoch_id=sample.epoch_id,
            sample_index=sample.sample_index,
            run_id=sample.run_id,
            predicted_class=predicted,
            decision_score_class1=score,
        )


def fit_heldout_prediction_provider(dataset: EEGDataset, config=None) -> FoldModelPredictionProvider:
    """Fit the three training-fold pipelines through the shared M2 implementation."""
    from bci.classification import ClassificationConfig, fit_run_heldout_models

    selected = config or ClassificationConfig()
    models, _ = fit_run_heldout_models(dataset, selected)
    return FoldModelPredictionProvider(models)


def verify_provider_against_m2(
    provider: FoldModelPredictionProvider,
    dataset: EEGDataset,
    predictions_path,
    *,
    atol: float = 1e-12,
) -> None:
    """Ensure replay inference equals the persisted M2 OOF labels and raw scores."""
    from pathlib import Path

    path = Path(predictions_path)
    if not path.is_file():
        raise FileNotFoundError(f"M2 OOF prediction artifact not found: {path}")
    with np.load(path, allow_pickle=False) as saved:
        expected_labels = np.asarray(saved["y_pred"])
        expected_scores = np.asarray(saved["decision_score_class1"])
        expected_runs = np.asarray(saved["run_id"])
        expected_indices = np.asarray(saved["sample_index"])
    if not (len(expected_labels) == len(expected_scores) == len(dataset.X)):
        raise ValueError("M2 OOF artifact length does not match M1 epochs")
    for index in range(len(dataset.X)):
        sample = ReplaySample(
            epoch_id=f"S001-R{int(dataset.run_ids[index]):02d}-E{index:03d}",
            sample_index=index,
            run_id=int(dataset.run_ids[index]),
            run_epoch_index=index,
            eeg=dataset.X[index],
            replay_elapsed_seconds=0.0,
            timestamp_utc="1970-01-01T00:00:00Z",
        )
        prediction = provider.predict(sample)
        if (
            int(expected_indices[index]) != index
            or int(expected_runs[index]) != prediction.run_id
            or int(expected_labels[index]) != prediction.predicted_class
            or not np.isclose(expected_scores[index], prediction.decision_score_class1, rtol=0.0, atol=atol)
        ):
            raise ValueError(f"M3 provider disagrees with M2 OOF result at sample {index}")
