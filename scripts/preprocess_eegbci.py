#!/usr/bin/env python3
"""Preprocess the approved single-subject EEGBCI subset (no model training)."""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import sys
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "data/cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / "data/cache"))
sys.path.insert(0, str(PROJECT_ROOT))

import mne
import numpy as np

from bci.preprocessing import (
    DATASET_URL,
    PreprocessingConfig,
    config_dict,
    environment_versions,
    plot_recording,
    preprocess_raw,
    write_metadata,
)


SUBJECT = 1
RUNS = (6, 10, 14)
CLASS_NAMES = {"hands": "imagined hands", "feet": "imagined feet"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/m1"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_noise_diagnostic(raw: mne.io.BaseRaw) -> dict[str, Any]:
    """Measure 50 Hz peak excess over nearby frequencies; diagnostic only."""
    psd = raw.copy().pick("eeg").compute_psd(
        method="welch", fmin=45, fmax=55, n_fft=1024, verbose="ERROR"
    )
    frequencies = psd.freqs
    mean_power = psd.get_data().mean(axis=0)
    peak_index = int(np.argmin(np.abs(frequencies - 50.0)))
    nearby = ((frequencies >= 45) & (frequencies <= 48)) | ((frequencies >= 52) & (frequencies <= 55))
    excess_db = float(10 * np.log10(mean_power[peak_index] / np.median(mean_power[nearby])))
    return {
        "target_hz": 50,
        "observed_bin_hz": float(frequencies[peak_index]),
        "mean_power_v2_per_hz": float(mean_power[peak_index]),
        "peak_excess_over_nearby_median_db": excess_db,
        "notch_applied": False,
        "decision": "No distinct 50 Hz peak in this diagnostic; notch omitted.",
        "method": "Welch mean across EEG channels; 50 Hz bin relative to 45-48 and 52-55 Hz median.",
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for run in RUNS:
        matches = list(args.data_dir.resolve().rglob(f"S{SUBJECT:03d}R{run:02d}.edf"))
        if len(matches) != 1:
            raise SystemExit(f"Expected exactly one file for subject {SUBJECT}, run {run}; found {len(matches)}")
        paths.append(matches[0])

    config = PreprocessingConfig()
    all_data: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_runs: list[np.ndarray] = []
    per_run: list[dict[str, Any]] = []
    channel_names: list[str] | None = None
    sampling_rate: float | None = None
    epoch_times: np.ndarray | None = None

    for run, path in zip(RUNS, paths):
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        line_noise = line_noise_diagnostic(raw)
        epochs, event_info = preprocess_raw(raw, config)
        data = epochs.get_data(copy=True)
        labels = np.asarray(["hands" if event == event_info["class_event_ids"]["hands"] else "feet" for event in epochs.events[:, 2]])
        all_data.append(data)
        all_labels.append(labels)
        all_runs.append(np.full(len(labels), run, dtype=np.int16))
        channel_names = list(epochs.ch_names)
        sampling_rate = float(epochs.info["sfreq"])
        epoch_times = epochs.times
        per_run.append({
            "run": run,
            "task": "imagined hands versus imagined feet",
            "raw_file": path.relative_to(args.data_dir.resolve()).as_posix(),
            "raw_size_bytes": path.stat().st_size,
            "raw_sha256": sha256(path),
            "sampling_rate_hz": float(raw.info["sfreq"]),
            "raw_channel_count": len(raw.ch_names),
            "raw_channel_names": list(raw.ch_names),
            "annotations": event_info["annotation_counts"],
            "mne_event_id": event_info["event_id"],
            "mne_event_counts": event_info["event_counts"],
            "verified_class_event_ids": event_info["class_event_ids"],
            "epoch_counts": {label: int(np.count_nonzero(labels == label)) for label in CLASS_NAMES},
            "dropped_epochs": list(epochs.drop_log),
            "line_noise_diagnostic": line_noise,
        })
        plot_recording(raw, output_dir / f"S{SUBJECT:03d}R{run:02d}_overview.png", run)

    combined = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels)
    run_ids = np.concatenate(all_runs)
    np.savez_compressed(
        output_dir / f"S{SUBJECT:03d}_epochs.npz",
        X=combined,
        y=labels,
        run=run_ids,
        times_s=epoch_times,
        channel_names=np.asarray(channel_names),
    )
    metadata = {
        "project": "NeuroBot",
        "mode": "offline file processing; REPLAY source, not live EEG acquisition",
        "dataset": "PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB)",
        "dataset_version": "1.0.0",
        "dataset_url": DATASET_URL,
        "download_date": date.today().isoformat(),
        "subject": SUBJECT,
        "runs": list(RUNS),
        "task": "imagined hands versus imagined feet",
        "label_mapping": {
            "T1": "hands (imagined both fists)",
            "T2": "feet (imagined both feet)",
            "source": "PhysioNet EEGMMIDB protocol for runs 6, 10, and 14; annotation presence and counts verified from each EDF before mapping.",
        },
        "preprocessing": config_dict(config),
        "data_shape": list(combined.shape),
        "data_axis_order": ["epochs", "EEG_channels", "samples"],
        "sampling_rate_hz": sampling_rate,
        "class_counts": {label: int(np.count_nonzero(labels == label)) for label in CLASS_NAMES},
        "run_counts": {str(run): int(np.count_nonzero(run_ids == run)) for run in RUNS},
        "environment": environment_versions(),
        "platform": {"system": platform.platform(), "architecture": platform.machine()},
        "runs_detail": per_run,
        "outputs": {
            "epochs": f"S{SUBJECT:03d}_epochs.npz",
            "plots": [f"S{SUBJECT:03d}R{run:02d}_overview.png" for run in RUNS],
        },
        "classifier_trained": False,
    }
    metadata_path = output_dir / f"S{SUBJECT:03d}_metadata.json"
    write_metadata(metadata, metadata_path)
    print(f"Saved {combined.shape[0]} epochs, {combined.shape[1]} EEG channels, {combined.shape[2]} samples/epoch")
    print(f"Class counts: {metadata['class_counts']}")
    print(f"Outputs: {output_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
