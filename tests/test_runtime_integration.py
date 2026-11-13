"""M3 fold-model replay, default HOLD behavior, logs, and visualization outputs."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from runtime.config import load_runtime_config
from runtime.engine import NeuroBotReplayEngine
from runtime.event_log import EventLog
from runtime.prediction_provider import FoldModelPredictionProvider, verify_provider_against_m2
from runtime.replay import EpochReplay
from runtime.safety import SafetyController
from runtime.state_machine import ControlState
from runtime.visualization import save_m3_plots


ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def test_provider_matches_saved_m2_predictions_for_all_epochs(m1_dataset, heldout_models) -> None:
    provider = FoldModelPredictionProvider(heldout_models)
    artifact = ROOT / "data/processed/m2/out_of_fold_predictions.npz"
    verify_provider_against_m2(provider, m1_dataset, artifact)


def test_default_replay_holds_and_writes_all_m3_visualizations(tmp_path, m1_dataset, heldout_models) -> None:
    base = load_runtime_config(ROOT / "config/m3.toml", ROOT)
    config = replace(
        base,
        deterministic_mode=True,
        event_log_path=tmp_path / "events.jsonl",
        confidence=replace(base.confidence, enabled=False, threshold=0.0),
    )
    event_log = EventLog(config.event_log_path, config.logging_level)
    clock = Clock()
    safety = SafetyController(config, event_log, monotonic=clock)
    replay = EpochReplay(m1_dataset, deterministic_mode=True)
    engine = NeuroBotReplayEngine(replay, FoldModelPredictionProvider(heldout_models), safety, event_log, monotonic=clock)

    result = engine.run()

    assert len(result.prediction_records) == 45
    assert [record["sample_index"] for record in result.prediction_records] == list(range(45))
    assert all(not record["accepted"] for record in result.prediction_records)
    assert result.final_state == "HOLD"
    assert all(event["command_type"] == "HOLD" for event in result.command_events)
    assert event_log.path.is_file()
    log_rows = [json.loads(line) for line in event_log.path.read_text().splitlines()]
    assert any(row["event_type"] == "incoming_prediction" for row in log_rows)
    assert any(row["event_type"] == "source_stopped" for row in log_rows)

    plots = save_m3_plots(result.prediction_records, result.command_events, tmp_path / "plots", 0.0, False)
    assert len(plots) == 4
    assert all(path.is_file() and path.stat().st_size > 10_000 for path in plots)


def test_prediction_provider_failure_enters_error_with_only_hold(tmp_path, m1_dataset, heldout_models) -> None:
    base = load_runtime_config(ROOT / "config/m3.toml", ROOT)
    config = replace(base, deterministic_mode=True, event_log_path=tmp_path / "error.jsonl")

    class BrokenProvider:
        def predict(self, sample):
            raise RuntimeError("injected provider failure")

    log = EventLog(config.event_log_path)
    clock = Clock()
    safety = SafetyController(config, log, monotonic=clock)
    engine = NeuroBotReplayEngine(EpochReplay(m1_dataset, deterministic_mode=True), BrokenProvider(), safety, log, monotonic=clock)
    result = engine.run()

    assert result.final_state == "ERROR"
    assert all(event["command_type"] == "HOLD" for event in result.command_events)
    assert any(row["event_type"] == "error" for row in result.event_records)
