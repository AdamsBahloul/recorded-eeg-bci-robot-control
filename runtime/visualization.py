"""Publication-quality diagnostic plots from an M3 replay result."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


STATE_LEVELS = {"HOLD": 0, "HANDS": 1, "FEET": 2, "ERROR": 3}


def save_m3_plots(predictions: list[dict[str, Any]], commands: list[dict[str, Any]], output_dir: Path, threshold: float, threshold_enabled: bool) -> list[Path]:
    """Generate state, confidence, acceptance, and command summary plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    x = np.asarray([row["replay_elapsed_seconds"] for row in predictions], dtype=float)
    states = [STATE_LEVELS[row["state_after"]] for row in predictions]
    state_path = output_dir / "state_timeline.png"
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax.step(x, states, where="post", color="#275d8c", linewidth=1.6)
    ax.set(yticks=list(STATE_LEVELS.values()), yticklabels=list(STATE_LEVELS.keys()),
           xlabel="Replay elapsed time (s)", ylabel="Safety state",
           title="NeuroBot state timeline (recorded-data REPLAY)")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(state_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    created.append(state_path)

    confidence_path = output_dir / "prediction_confidence.png"
    margins = np.asarray([row["confidence_measure"] if row["confidence_measure"] is not None else np.nan for row in predictions], dtype=float)
    class_values = np.asarray([row["predicted_class"] for row in predictions], dtype=int)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    colors = np.where(class_values == 0, "#2c7fb8", "#d95f0e")
    ax.scatter(x, class_values, c=colors, marker="o", s=28, label="Predicted class")
    ax.set_yticks([0, 1], ["Imagined hands (0)", "Imagined feet (1)"])
    ax.set_xlabel("Replay elapsed time (s)")
    ax.set_ylabel("Predicted class")
    ax.set_title("Prediction and decision-score margin (not a probability)")
    ax.grid(alpha=0.2)
    right = ax.twinx()
    right.plot(x, margins, color="#5e3c99", alpha=0.85, linewidth=1.2, label="Absolute LDA decision margin")
    right.set_ylabel("Decision-score margin (LDA score units)")
    if threshold_enabled:
        right.axhline(threshold, color="#b2182b", linestyle="--", label="Configured threshold")
    left_handles, left_labels = ax.get_legend_handles_labels()
    right_handles, right_labels = right.get_legend_handles_labels()
    ax.legend(left_handles + right_handles, left_labels + right_labels, loc="best")
    fig.savefig(confidence_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    created.append(confidence_path)

    acceptance_path = output_dir / "accepted_rejected_predictions.png"
    accepted = np.asarray([row["accepted"] for row in predictions], dtype=bool)
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax.scatter(x[~accepted], np.zeros(np.count_nonzero(~accepted)), color="#b2182b", marker="x", s=35, label="Rejected / HOLD")
    ax.scatter(x[accepted], np.ones(np.count_nonzero(accepted)), color="#1a9850", marker="o", s=35, label="Accepted active state")
    ax.set(yticks=[0, 1], yticklabels=["Rejected", "Accepted"], xlabel="Replay elapsed time (s)",
           title="Prediction acceptance after confidence and stability checks")
    ax.legend(loc="best")
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(acceptance_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    created.append(acceptance_path)

    summary_path = output_dir / "command_event_summary.png"
    command_counts = Counter(command["command_type"] for command in commands)
    command_counts.setdefault("HOLD", 0)
    command_counts.setdefault("HANDS", 0)
    command_counts.setdefault("FEET", 0)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    labels = ["HOLD", "HANDS", "FEET"]
    values = [command_counts[label] for label in labels]
    bars = ax.bar(labels, values, color=["#777777", "#2c7fb8", "#d95f0e"])
    ax.bar_label(bars, padding=3)
    ax.set(ylabel="Command event count", title="Command event summary (no robot actuation)")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(summary_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    created.append(summary_path)
    return created
