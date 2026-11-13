"""Independent CSP+LDA recomputation of M2 evaluation artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import Any

import mne
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
import sklearn


def _manual_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    matrix = [[0, 0], [0, 0]]
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual)][int(predicted)] += 1
    return matrix


def judge_m2(epochs_path: Path, metadata_path: Path, predictions_path: Path, metrics_path: Path | None = None) -> dict[str, Any]:
    """Independently load arrays, split by run, fit fold pipelines, and compare artifacts."""
    for path in (epochs_path, metadata_path, predictions_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("label_mapping", {}).get("T1", "").lower().find("hands") < 0:
        raise ValueError("Independent judge found no verified T1 hands mapping")
    if metadata.get("label_mapping", {}).get("T2", "").lower().find("feet") < 0:
        raise ValueError("Independent judge found no verified T2 feet mapping")
    with np.load(epochs_path, allow_pickle=False) as source:
        X = np.asarray(source["X"])
        names = np.asarray(source["y"]).astype(str)
        groups = np.asarray(source["run"], dtype=np.int64)
    y = np.where(names == "hands", 0, np.where(names == "feet", 1, -1)).astype(np.int64)
    if X.shape != (45, 64, 481) or y.shape != (45,) or set(np.unique(groups)) != {6, 10, 14} or np.any(y < 0):
        raise ValueError("Independent judge found invalid M1 sample dimensions or labels")

    split = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    prediction = np.full(45, -1, dtype=np.int64)
    score = np.full(45, np.nan, dtype=float)
    coverage = np.zeros(45, dtype=np.int8)
    validation_runs: list[int] = []
    training_runs: list[list[int]] = []
    with mne.use_log_level("ERROR"):
        for train, valid in split.split(X, y, groups=groups):
            train_groups = sorted(int(x) for x in np.unique(groups[train]))
            valid_groups = sorted(int(x) for x in np.unique(groups[valid]))
            if set(train_groups) & set(valid_groups) or len(valid_groups) != 1:
                raise AssertionError("Independent run-leakage judge found overlapping groups")
            model = make_pipeline(
                CSP(n_components=4, log=True, transform_into="average_power", component_order="mutual_info"),
                LinearDiscriminantAnalysis(solver="svd"),
            )
            model.fit(X[train], y[train])
            prediction[valid] = model.predict(X[valid]).astype(np.int64)
            score[valid] = model.decision_function(X[valid]).reshape(-1)
            coverage[valid] += 1
            validation_runs.append(valid_groups[0])
            training_runs.append(train_groups)
    with np.load(predictions_path, allow_pickle=False) as stored:
        saved_prediction = np.asarray(stored["y_pred"], dtype=np.int64)
        saved_truth = np.asarray(stored["y_true"], dtype=np.int64)
        saved_coverage = np.asarray(stored["prediction_count"], dtype=np.int8)
        saved_scores = np.asarray(stored["decision_score_class1"], dtype=float)
    if not np.all(coverage == 1):
        raise AssertionError("Every sample did not receive exactly one independent OOF prediction")
    if not np.array_equal(coverage, saved_coverage):
        raise AssertionError("Independent OOF coverage differs from production artifact")
    if not np.array_equal(y, saved_truth) or not np.array_equal(prediction, saved_prediction):
        raise AssertionError("Independent predictions/truth differ from saved production OOF artifact")
    if not np.allclose(score, saved_scores, rtol=1e-10, atol=1e-10):
        raise AssertionError("Independent decision scores differ from saved production OOF artifact")

    matrix = _manual_confusion(y, prediction)
    counts = [int(np.count_nonzero(y == cls)) for cls in (0, 1)]
    recalls = [matrix[cls][cls] / sum(matrix[cls]) for cls in (0, 1)]
    metrics = {
        "accuracy": float(np.mean(y == prediction)),
        "balanced_accuracy": float(np.mean(recalls)),
        "confusion_matrix_hands_feet": matrix,
        "class_counts_hands_feet": counts,
    }
    if metrics_path is not None:
        stored_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["out_of_fold_metrics"]
        if stored_metrics.get("confusion_matrix") != matrix:
            raise AssertionError("Independent confusion matrix differs from M2 metrics JSON")
        if not np.isclose(float(stored_metrics["accuracy"]), metrics["accuracy"], rtol=0, atol=1e-12):
            raise AssertionError("Independent accuracy differs from M2 metrics JSON")
        if not np.isclose(float(stored_metrics["balanced_accuracy"]), metrics["balanced_accuracy"], rtol=0, atol=1e-12):
            raise AssertionError("Independent balanced accuracy differs from M2 metrics JSON")
    return {
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "judge": "independent_m2_csp_lda_recomputation",
        "status": "PASS",
        "evaluation_type": "within-participant run-held-out CV; not independent test data",
        "method": {"CSP": 4, "LDA_solver": "svd", "splitter": "StratifiedGroupKFold", "groups": "recording run", "folds": 3, "seed": 42},
        "training_runs_by_fold": training_runs,
        "validation_runs": validation_runs,
        "prediction_count_min": int(coverage.min()),
        "prediction_count_max": int(coverage.max()),
        "metrics": metrics,
        "dependency_versions": {"python": sys.version.split()[0], "mne": mne.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "platform": platform.platform(),
    }
