#!/usr/bin/env python3
"""Run the recorded EEG M3 command engine with run-held-out fold models."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "data/cache"))
sys.path.insert(0, str(ROOT))

import numpy as np

from bci.classification import (
    ClassificationConfig,
    dependency_versions,
    generate_oof_predictions,
    load_m1_dataset,
)
from runtime.config import load_runtime_config
from runtime.engine import NeuroBotReplayEngine
from runtime.event_log import EventLog
from runtime.prediction_provider import FoldModelPredictionProvider, verify_provider_against_m2
from runtime.replay import EpochReplay
from runtime.safety import SafetyController
from runtime.visualization import save_m3_plots


def verify_restored_m2(metrics_path: Path, predictions_path: Path, dataset, recomputed: dict) -> dict:
    """Fail closed if M2 artifacts do not match the approved OOF methodology/results."""
    if not metrics_path.is_file():
        raise FileNotFoundError(f"M2 metrics not found; run scripts/evaluate_csp_lda.py first: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    saved = metrics.get("out_of_fold_metrics", {})
    expected_matrix = [[20, 1], [1, 23]]
    if saved.get("confusion_matrix") != expected_matrix or recomputed["metrics"]["confusion_matrix"] != expected_matrix:
        raise RuntimeError("M2 confusion matrix differs from the verified result; refusing M3 replay")
    for key, expected in (("accuracy", 43 / 45), ("balanced_accuracy", (20 / 21 + 23 / 24) / 2)):
        if not np.isclose(float(saved.get(key, np.nan)), expected, rtol=0, atol=1e-12):
            raise RuntimeError(f"Saved M2 {key} does not reproduce the approved baseline")
        if not np.isclose(recomputed["metrics"][key], expected, rtol=0, atol=1e-12):
            raise RuntimeError(f"Recomputed M2 {key} differs; refusing M3 replay")
    return metrics


def write_prediction_csv(path: Path, records: list[dict]) -> None:
    fields = [
        "sample_index", "run_id", "run_epoch_index", "epoch_id", "timestamp_utc",
        "replay_elapsed_seconds", "predicted_class", "predicted_label", "decision_score_class1",
        "confidence_measure", "confidence_type", "confidence_accepted", "stability_accepted",
        "accepted", "reason", "state_after", "command_id", "command_type",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/m3.toml")
    parser.add_argument("--deterministic", action="store_true", help="Skip wall-clock pacing for a fast deterministic replay")
    args = parser.parse_args()
    config = load_runtime_config(args.config, ROOT)
    if args.deterministic:
        config = replace(config, deterministic_mode=True)
    config.validate()

    dataset = load_m1_dataset(config.epochs_path, config.metadata_path)
    classifier_config = ClassificationConfig(
        csp_components=config.csp_components,
        lda_solver=config.lda_solver,
        n_splits=config.n_splits,
        shuffle=config.shuffle,
        random_seed=config.random_seed,
    )
    m2_recomputed = generate_oof_predictions(dataset, classifier_config)
    m2_metrics = verify_restored_m2(config.m2_metrics_path, config.m2_predictions_path, dataset, m2_recomputed)
    provider = FoldModelPredictionProvider(m2_recomputed["fold_models"])
    verify_provider_against_m2(provider, dataset, config.m2_predictions_path)

    config.output_directory.mkdir(parents=True, exist_ok=True)
    event_log = EventLog(config.event_log_path, config.logging_level)
    replay = EpochReplay(
        dataset,
        delay_between_epochs_seconds=config.replay_delay_seconds,
        speed=config.replay_speed,
        deterministic_mode=config.deterministic_mode,
    )
    safety = SafetyController(config, event_log, monotonic=time.monotonic)
    engine = NeuroBotReplayEngine(replay, provider, safety, event_log, monotonic=time.monotonic)
    result = engine.run()

    prediction_path = config.output_directory / "replay_predictions.csv"
    write_prediction_csv(prediction_path, result.prediction_records)
    plot_paths = save_m3_plots(
        result.prediction_records,
        result.command_events,
        config.output_directory,
        threshold=config.confidence.threshold,
        threshold_enabled=config.confidence.enabled,
    )
    active_events = [event for event in result.command_events if event["command_type"] in ("HANDS", "FEET")]
    summary = {
        "project": "NeuroBot",
        "source": "REPLAY (recorded EEG; not live acquisition)",
        "final_state": result.final_state,
        "epoch_count": len(result.prediction_records),
        "class_mapping": {"0": "imagined hands", "1": "imagined feet"},
        "active_commands_enabled": config.confidence.enabled,
        "confidence": {
            "mode": config.confidence.mode,
            "threshold": config.confidence.threshold,
            "threshold_direction": config.confidence.threshold_direction,
            "interpretation": "Absolute LDA decision-score margin; not a calibrated probability.",
            "threshold_validation": "Configuration validated; no probability calibration or threshold optimization performed.",
        },
        "model_configuration": {
            "pipeline": ["MNE CSP", "scikit-learn LDA"],
            "csp_components": config.csp_components,
            "lda_solver": config.lda_solver,
            "fold_group": config.group_by,
            "fold_seed": config.random_seed,
        },
        "cross_validation_reference": {
            "type": m2_metrics["evaluation_type"],
            "metrics_file": str(config.m2_metrics_path),
            "out_of_fold_metrics": m2_metrics["out_of_fold_metrics"],
            "replayed_models": "For each epoch, the model trained on the other two runs was selected; M2 OOF labels/scores were verified first.",
        },
        "command_event_counts": {
            "total": len(result.command_events),
            "HOLD": sum(event["command_type"] == "HOLD" for event in result.command_events),
            "HANDS": sum(event["command_type"] == "HANDS" for event in result.command_events),
            "FEET": sum(event["command_type"] == "FEET" for event in result.command_events),
            "active": len(active_events),
        },
        "accepted_predictions": sum(record["accepted"] for record in result.prediction_records),
        "rejected_predictions": sum(not record["accepted"] for record in result.prediction_records),
        "replay_configuration": {
            "delay_between_epochs_seconds": config.replay_delay_seconds,
            "speed": config.replay_speed,
            "deterministic_mode": config.deterministic_mode,
        },
        "stability_and_safety_configuration": {
            "required_consecutive_predictions": config.required_consecutive_predictions,
            "stability_window_seconds": config.stability_window_seconds,
            "maximum_prediction_interval_seconds": config.maximum_prediction_interval_seconds,
            "command_cooldown_seconds": config.command_cooldown_seconds,
            "maximum_active_commands_per_minute": config.maximum_active_commands_per_minute,
            "maximum_prediction_age_seconds": config.maximum_prediction_age_seconds,
            "prediction_timeout_seconds": config.prediction_timeout_seconds,
        },
        "artifacts": {
            "event_log": str(config.event_log_path),
            "prediction_csv": prediction_path.name,
            "plots": [path.name for path in plot_paths],
        },
        "dependency_versions": dependency_versions(),
        "physical_actuator_control": False,
    }
    (config.output_directory / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Processed {summary['epoch_count']} REPLAY epochs; final state={result.final_state}")
    print(f"Active HANDS/FEET commands enabled={config.confidence.enabled}; emitted={summary['command_event_counts']['active']}")
    print(f"Outputs: {config.output_directory}")
    return 0 if result.final_state != "ERROR" else 2


if __name__ == "__main__":
    raise SystemExit(main())
