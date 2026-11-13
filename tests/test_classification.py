"""M2 regression checks for CV folds, predictions, and metrics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bci.classification import (
    ClassificationConfig,
    generate_oof_predictions,
    load_m1_dataset,
    make_csp_lda_pipeline,
    make_run_folds,
    save_m2_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dataset():
    return load_m1_dataset(
        ROOT / "data/processed/m1/S001_epochs.npz",
        ROOT / "data/processed/m1/S001_metadata.json",
    )


@pytest.fixture(scope="module")
def result(dataset):
    return generate_oof_predictions(dataset, ClassificationConfig())


def test_m1_loading_and_label_mapping(dataset) -> None:
    assert dataset.X.shape == (45, 64, 481)
    np.testing.assert_array_equal(np.unique(dataset.y, return_counts=True)[0], [0, 1])
    np.testing.assert_array_equal(np.unique(dataset.y, return_counts=True)[1], [21, 24])


def test_pipeline_is_csp_then_lda_with_required_settings() -> None:
    pipeline = make_csp_lda_pipeline()
    assert [name for name, _ in pipeline.steps] == ["csp", "lda"]
    assert pipeline.named_steps["csp"].n_components == 4
    assert pipeline.named_steps["lda"].solver == "svd"


def test_run_grouped_folds_have_no_overlap_and_each_run_is_held_out_once(dataset) -> None:
    folds = make_run_folds(dataset)
    assert [fold.validation_run for fold in folds] == [6, 10, 14]
    for fold in folds:
        assert set(fold.training_runs).isdisjoint({fold.validation_run})
        assert len(fold.validation_indices) == 15
        assert len(fold.train_indices) == 30


def test_every_epoch_has_exactly_one_oof_prediction(result) -> None:
    np.testing.assert_array_equal(result["prediction_count"], np.ones(45, dtype=np.int8))
    assert result["y_pred"].shape == (45,)
    assert result["decision_score_class1"].shape == (45,)
    assert np.isfinite(result["decision_score_class1"]).all()


def test_reported_oof_metrics_are_reproduced(result) -> None:
    metrics = result["metrics"]
    assert metrics["confusion_matrix"] == [[20, 1], [1, 23]]
    assert metrics["accuracy"] == pytest.approx(43 / 45)
    assert metrics["balanced_accuracy"] == pytest.approx((20 / 21 + 23 / 24) / 2)


def test_m2_outputs_are_separate_and_complete(tmp_path, dataset, result) -> None:
    save_m2_artifacts(dataset, result, ClassificationConfig(), tmp_path)
    assert (tmp_path / "metrics.json").is_file()
    with np.load(tmp_path / "out_of_fold_predictions.npz") as predictions:
        assert predictions["prediction_count"].shape == (45,)
        np.testing.assert_array_equal(predictions["y_pred"], result["y_pred"])


def test_loader_reports_missing_artifacts(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="epochs file"):
        load_m1_dataset(tmp_path / "missing.npz", tmp_path / "missing.json")


def test_loader_rejects_malformed_epoch_archive(tmp_path) -> None:
    epoch_path = tmp_path / "epochs.npz"
    metadata_path = tmp_path / "metadata.json"
    epoch_path.write_bytes(b"not a numpy archive")
    metadata_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not read M1 artifacts"):
        load_m1_dataset(epoch_path, metadata_path)
