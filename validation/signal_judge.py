"""Independent raw-signal payload and label-leakage checks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mne
import numpy as np

from bci.replay_source import load_raw_run_message
from transport.eeg_chunks import run_start


def judge_signal_run(edf_path: Path, run_id: int, session_id: str = "signal-judge") -> dict[str, Any]:
    """Compare serialized raw values/times with a separately read EDF."""
    candidate = load_raw_run_message(edf_path, run_id, session_id, 1)
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    mne.datasets.eegbci.standardize(raw)
    raw.pick("eeg")
    reference = raw.get_data(picks="eeg")
    events, ids = mne.events_from_annotations(raw, verbose="ERROR")
    task_codes = {ids["T1"], ids["T2"]}
    event_positions = tuple(sorted(int(sample - raw.first_samp) for sample, _, code in events if code in task_codes))
    if candidate.signal_volts.shape != (64, raw.n_times):
        raise AssertionError("Transferred raw signal dimensions differ from EDF")
    if not np.array_equal(candidate.channel_names, tuple(raw.ch_names)):
        raise AssertionError("Transferred channel order differs from EDF reference")
    if candidate.sampling_frequency_hz != raw.info["sfreq"]:
        raise AssertionError("Transferred sampling frequency differs from EDF reference")
    if not np.array_equal(candidate.event_samples, event_positions):
        raise AssertionError("Unlabeled event timing differs from EDF reference")
    if not np.array_equal(candidate.signal_volts, reference):
        raise AssertionError("Transferred samples differ from raw EDF values")
    header = json.loads(run_start(candidate, 512).decode("utf-8"))
    forbidden = {"labels", "class_labels", "ground_truth", "predicted_class", "decision_score", "prediction", "command"}
    if forbidden.intersection(header):
        raise AssertionError("Signal message includes class, prediction, score, or command data")
    return {
        "judge": "independent_raw_signal_and_label_leakage",
        "status": "PASS",
        "run_id": run_id,
        "shape": list(reference.shape),
        "channel_count": len(raw.ch_names),
        "sampling_frequency_hz": float(raw.info["sfreq"]),
        "unlabeled_event_boundary_count": len(event_positions),
        "sample_values_exact": True,
        "labels_transmitted": False,
    }
