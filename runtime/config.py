"""Strict TOML configuration for replay and safety behavior."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class ConfidenceConfig:
    mode: str = "decision_score_margin"
    enabled: bool = False
    threshold: float = 0.0
    threshold_direction: str = "greater_equal"


@dataclass(frozen=True)
class RuntimeConfig:
    epochs_path: Path
    metadata_path: Path
    m2_predictions_path: Path
    m2_metrics_path: Path
    output_directory: Path
    csp_components: int
    lda_solver: str
    n_splits: int
    shuffle: bool
    random_seed: int
    group_by: str
    replay_delay_seconds: float
    replay_speed: float
    deterministic_mode: bool
    post_active_display_seconds: float
    confidence: ConfidenceConfig
    required_consecutive_predictions: int
    stability_window_seconds: float
    maximum_prediction_interval_seconds: float
    command_cooldown_seconds: float
    maximum_active_commands_per_minute: int
    maximum_prediction_age_seconds: float
    prediction_timeout_seconds: float
    logging_level: str
    event_log_path: Path

    def validate(self) -> None:
        """Reject invalid or unsafe settings before starting replay."""
        if self.csp_components != 4 or self.lda_solver != "svd":
            raise ValueError("M3 must use the verified M2 classifier settings: 4 CSP components and LDA SVD")
        if self.n_splits != 3 or not self.shuffle or self.group_by != "run" or self.random_seed != 42:
            raise ValueError("M3 must reuse shuffled three-fold run-grouped CV with random seed 42")
        if self.replay_delay_seconds < 0 or not math.isfinite(self.replay_delay_seconds):
            raise ValueError("replay.delay_between_epochs_seconds must be finite and >= 0")
        if self.replay_speed <= 0 or not math.isfinite(self.replay_speed):
            raise ValueError("replay.speed must be finite and > 0")
        if not isinstance(self.deterministic_mode, bool):
            raise ValueError("replay.deterministic_mode must be a boolean")
        if not math.isfinite(self.post_active_display_seconds) or self.post_active_display_seconds < 0:
            raise ValueError("replay.post_active_display_seconds must be finite and >= 0")
        if self.confidence.mode != "decision_score_margin":
            raise ValueError("Only decision_score_margin is supported; calibrated probabilities are not implemented")
        if self.confidence.threshold_direction != "greater_equal":
            raise ValueError("decision_score_margin can only accept the greater_equal threshold direction")
        if not math.isfinite(self.confidence.threshold):
            raise ValueError("confidence.threshold must be a finite number")
        if self.confidence.enabled and self.confidence.threshold <= 0:
            raise ValueError("Active commands require an explicitly enabled, positive decision-margin threshold")
        if not self.confidence.enabled and self.confidence.threshold != 0:
            raise ValueError("With confidence.enabled=false, threshold must remain 0 until explicitly configured")
        if self.required_consecutive_predictions < 1:
            raise ValueError("stability.required_consecutive_predictions must be >= 1")
        positive = {
            "stability.window_seconds": self.stability_window_seconds,
            "stability.maximum_prediction_interval_seconds": self.maximum_prediction_interval_seconds,
            "safety.maximum_prediction_age_seconds": self.maximum_prediction_age_seconds,
            "safety.prediction_timeout_seconds": self.prediction_timeout_seconds,
        }
        for key, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and > 0")
        if not math.isfinite(self.command_cooldown_seconds) or self.command_cooldown_seconds < 0:
            raise ValueError("safety.command_cooldown_seconds must be finite and >= 0")
        if self.maximum_active_commands_per_minute < 1:
            raise ValueError("safety.maximum_active_commands_per_minute must be >= 1")
        if self.logging_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("logging.level must be DEBUG, INFO, WARNING, or ERROR")


def _path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_runtime_config(path: Path, project_root: Path) -> RuntimeConfig:
    """Load an M3 TOML file and validate every safety-relevant field."""
    if not path.is_file():
        raise FileNotFoundError(f"M3 configuration file not found: {path}")
    try:
        raw: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
        paths = raw["paths"]
        classifier = raw["classifier"]
        replay = raw["replay"]
        confidence = raw["confidence"]
        stability = raw["stability"]
        safety = raw["safety"]
        logging = raw["logging"]
        for section, key in ((replay, "deterministic_mode"), (confidence, "enabled"), (classifier, "shuffle")):
            if type(section[key]) is not bool:
                raise ValueError(f"{key} must be a TOML boolean")
        for section, key in (
            (classifier, "csp_components"), (classifier, "n_splits"),
            (classifier, "random_seed"), (stability, "required_consecutive_predictions"),
            (safety, "maximum_active_commands_per_minute"),
        ):
            if type(section[key]) is not int:
                raise ValueError(f"{key} must be a TOML integer")
        if isinstance(confidence["threshold"], bool) or not isinstance(confidence["threshold"], (int, float)):
            raise ValueError("confidence.threshold must be a TOML number")
        config = RuntimeConfig(
            epochs_path=_path(project_root, paths["epochs"]),
            metadata_path=_path(project_root, paths["metadata"]),
            m2_predictions_path=_path(project_root, paths["m2_predictions"]),
            m2_metrics_path=_path(project_root, paths["m2_metrics"]),
            output_directory=_path(project_root, paths["output_directory"]),
            csp_components=int(classifier["csp_components"]),
            lda_solver=str(classifier["lda_solver"]),
            n_splits=int(classifier["n_splits"]),
            shuffle=bool(classifier["shuffle"]),
            random_seed=int(classifier["random_seed"]),
            group_by=str(classifier["group_by"]),
            replay_delay_seconds=float(replay["delay_between_epochs_seconds"]),
            replay_speed=float(replay["speed"]),
            deterministic_mode=bool(replay["deterministic_mode"]),
            post_active_display_seconds=float(replay.get("post_active_display_seconds", 0.0)),
            confidence=ConfidenceConfig(
                mode=str(confidence["mode"]),
                enabled=bool(confidence["enabled"]),
                threshold=float(confidence["threshold"]),
                threshold_direction=str(confidence["threshold_direction"]),
            ),
            required_consecutive_predictions=int(stability["required_consecutive_predictions"]),
            stability_window_seconds=float(stability["window_seconds"]),
            maximum_prediction_interval_seconds=float(stability["maximum_prediction_interval_seconds"]),
            command_cooldown_seconds=float(safety["command_cooldown_seconds"]),
            maximum_active_commands_per_minute=int(safety["maximum_active_commands_per_minute"]),
            maximum_prediction_age_seconds=float(safety["maximum_prediction_age_seconds"]),
            prediction_timeout_seconds=float(safety["prediction_timeout_seconds"]),
            logging_level=str(logging["level"]).upper(),
            event_log_path=_path(project_root, logging["event_log"]),
        )
    except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Invalid or incomplete M3 configuration: {exc}") from exc
    config.validate()
    return config
