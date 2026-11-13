"""M3 TOML defaults and safety-critical configuration validation."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from runtime.config import ConfidenceConfig, load_runtime_config


ROOT = Path(__file__).resolve().parents[1]


def test_default_config_keeps_active_commands_disabled() -> None:
    config = load_runtime_config(ROOT / "config/m3.toml", ROOT)
    assert config.confidence.enabled is False
    assert config.confidence.threshold == 0.0
    assert config.confidence.mode == "decision_score_margin"
    assert config.group_by == "run" and config.random_seed == 42


def test_unvalidated_demo_override_is_separate_and_keeps_stability_filter() -> None:
    default = load_runtime_config(ROOT / "config/m3.toml", ROOT)
    demo = load_runtime_config(ROOT / "config/m3_unvalidated_simulation_demo.toml", ROOT)
    assert default.confidence.enabled is False and default.confidence.threshold == 0.0
    assert default.post_active_display_seconds == 0.0
    assert demo.confidence.enabled is True and demo.confidence.threshold == 0.01
    assert demo.post_active_display_seconds == 2.0
    assert demo.replay_delay_seconds == 0.75
    assert demo.required_consecutive_predictions == 3


def test_active_commands_require_positive_explicit_threshold() -> None:
    config = load_runtime_config(ROOT / "config/m3.toml", ROOT)
    with pytest.raises(ValueError, match="positive"):
        replace(config, confidence=ConfidenceConfig(enabled=True, threshold=0)).validate()
    replace(config, confidence=ConfidenceConfig(enabled=True, threshold=0.5)).validate()


def test_m3_rejects_probability_mode_unshuffled_or_wrong_fold_settings() -> None:
    config = load_runtime_config(ROOT / "config/m3.toml", ROOT)
    with pytest.raises(ValueError, match="decision_score_margin"):
        replace(config, confidence=replace(config.confidence, mode="probability")).validate()
    with pytest.raises(ValueError, match="shuffled three-fold"):
        replace(config, shuffle=False).validate()


def test_missing_config_is_reported(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="configuration file not found"):
        load_runtime_config(tmp_path / "missing.toml", ROOT)
