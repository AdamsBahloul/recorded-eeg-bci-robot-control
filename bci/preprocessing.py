"""Reproducible EEGBCI loading, preprocessing, and metadata utilities."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import mne
import numpy as np


DATASET_URL = "https://physionet.org/content/eegmmidb/1.0.0/"
ANNOTATION_LABELS = {"T1": "hands", "T2": "feet"}


@dataclass(frozen=True)
class PreprocessingConfig:
    """Parameters fixed for the first single-participant experiment."""

    l_freq_hz: float = 8.0
    h_freq_hz: float = 30.0
    epoch_tmin_s: float = 1.0
    epoch_tmax_s: float = 4.0
    baseline: None = None
    notch_hz: None = None
    filter_method: str = "FIR"
    filter_phase: str = "zero"
    filter_length: str = "auto"
    l_trans_bandwidth_hz: float = 2.0
    h_trans_bandwidth_hz: float = 2.0
    fir_window: str = "hamming"
    fir_design: str = "firwin"
    filter_pad: str = "reflect_limited"
    eeg_channels_only: bool = True
    excluded_annotations: tuple[str, ...] = ("T0",)
    reject_by_annotation: bool = True
    event_conversion: str = "mne.events_from_annotations default mapping; runtime IDs recorded"


def inspect_annotations(raw: mne.io.BaseRaw) -> dict[str, Any]:
    """Return raw annotation and MNE event ID/count information."""
    annotation_counts = dict(sorted(Counter(raw.annotations.description).items()))
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    event_counts = {
        description: int(np.count_nonzero(events[:, 2] == code))
        for description, code in event_id.items()
    }
    return {
        "annotation_counts": annotation_counts,
        "event_id": event_id,
        "event_counts": event_counts,
        "events": events,
    }


def verified_class_event_ids(raw: mne.io.BaseRaw) -> tuple[dict[str, int], dict[str, Any]]:
    """Resolve hands/feet labels only after required annotation labels exist."""
    inspected = inspect_annotations(raw)
    event_id = inspected["event_id"]
    descriptions = set(inspected["annotation_counts"])
    missing = set(("T0", "T1", "T2")) - descriptions
    if missing or not {"T1", "T2"}.issubset(event_id):
        raise ValueError(f"Expected EEGBCI T0/T1/T2 annotations; missing {sorted(missing)}")
    return {label: int(event_id[annotation]) for annotation, label in ANNOTATION_LABELS.items()}, inspected


def preprocess_raw(
    raw: mne.io.BaseRaw,
    config: PreprocessingConfig = PreprocessingConfig(),
) -> tuple[mne.Epochs, dict[str, Any]]:
    """Filter EEG and epoch verified hands/feet events from a single recording."""
    class_event_ids, inspected = verified_class_event_ids(raw)
    eeg = raw.copy()
    mne.datasets.eegbci.standardize(eeg)
    eeg.pick("eeg")
    eeg.filter(
        l_freq=config.l_freq_hz,
        h_freq=config.h_freq_hz,
        picks="eeg",
        method="fir",
        phase=config.filter_phase,
        filter_length=config.filter_length,
        l_trans_bandwidth=config.l_trans_bandwidth_hz,
        h_trans_bandwidth=config.h_trans_bandwidth_hz,
        fir_window=config.fir_window,
        fir_design=config.fir_design,
        pad=config.filter_pad,
        verbose="ERROR",
    )
    epochs = mne.Epochs(
        eeg,
        inspected["events"],
        event_id=class_event_ids,
        tmin=config.epoch_tmin_s,
        tmax=config.epoch_tmax_s,
        baseline=config.baseline,
        picks="eeg",
        preload=True,
        reject_by_annotation=config.reject_by_annotation,
        verbose="ERROR",
    )
    return epochs, {
        "event_id": inspected["event_id"],
        "annotation_counts": inspected["annotation_counts"],
        "event_counts": inspected["event_counts"],
        "class_event_ids": class_event_ids,
    }


def plot_recording(raw: mne.io.BaseRaw, destination: Path, run: int) -> None:
    """Save a small raw trace and power spectrum overview for one run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display = raw.copy()
    mne.datasets.eegbci.standardize(display)
    display.pick("eeg")
    names = [name for name in ("C3", "Cz", "C4") if name in display.ch_names]
    if not names:
        names = display.ch_names[:3]
    data = display.get_data(picks=names, start=0, stop=min(display.n_times, int(10 * display.info["sfreq"])))
    times = np.arange(data.shape[1]) / display.info["sfreq"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    for trace, name in zip(data, names):
        scale = max(float(np.std(trace)), 1e-12)
        axes[0].plot(times, trace / scale + len(names) - names.index(name), label=name)
    axes[0].set(title=f"Participant 1, run {run}: first 10 s (REPLAY source)", xlabel="Time (s)", ylabel="EEG (per-channel SD units)")
    axes[0].legend(loc="upper right")

    spectrum = display.compute_psd(method="welch", fmin=1, fmax=60, verbose="ERROR")
    psd = spectrum.get_data()
    mean_db = 10 * np.log10(np.maximum(psd.mean(axis=0), np.finfo(float).tiny))
    axes[1].plot(spectrum.freqs, mean_db)
    axes[1].axvline(50, color="tab:red", linestyle="--", linewidth=1, label="50 Hz check")
    axes[1].set(title="Mean EEG power spectrum", xlabel="Frequency (Hz)", ylabel="Power (dB, V²/Hz)", xlim=(1, 60))
    axes[1].legend(loc="upper right")
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def environment_versions() -> dict[str, str]:
    """Return versions for all installed distributions in the active venv."""
    from importlib.metadata import distributions

    result = {"python": __import__("sys").version.split()[0]}
    result.update({
        dist.metadata["Name"]: dist.version
        for dist in distributions()
        if dist.metadata.get("Name")
    })
    return result


def config_dict(config: PreprocessingConfig) -> dict[str, Any]:
    """Convert config to JSON-compatible values."""
    return asdict(config)


def write_metadata(metadata: dict[str, Any], destination: Path) -> None:
    """Write a UTF-8 JSON metadata record with stable formatting."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
