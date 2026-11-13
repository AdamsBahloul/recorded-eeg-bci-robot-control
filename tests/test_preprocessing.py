"""Focused checks for deterministic EEG preprocessing helpers."""
from __future__ import annotations

import mne
import numpy as np
import pytest

from bci.preprocessing import plot_recording, preprocess_raw, verified_class_event_ids, write_metadata


def make_raw(include_all_events: bool = True) -> mne.io.RawArray:
    sfreq = 160.0
    names = ["C3.", "Cz.", "C4.", "EEG004"]
    info = mne.create_info(names + ["STI 014"], sfreq, ["eeg"] * 4 + ["stim"])
    rng = np.random.default_rng(42)
    signals = rng.standard_normal((5, int(50 * sfreq))) * 1e-6
    raw = mne.io.RawArray(signals, info, verbose="ERROR")
    descriptions = ["T0", "T1", "T0", "T2"] if include_all_events else ["T0", "T1", "T0"]
    onsets = [0.5, 2.0, 8.0, 12.0] if include_all_events else [0.5, 2.0, 8.0]
    raw.set_annotations(mne.Annotations(onset=onsets, duration=[0] * len(onsets), description=descriptions))
    return raw


def test_event_mapping_is_resolved_from_annotations_and_counts() -> None:
    mapping, inspected = verified_class_event_ids(make_raw())

    assert mapping == {"hands": 2, "feet": 3}
    assert inspected["annotation_counts"] == {"T0": 2, "T1": 1, "T2": 1}
    assert inspected["event_counts"] == {"T0": 2, "T1": 1, "T2": 1}


def test_missing_class_annotation_is_rejected() -> None:
    with pytest.raises(ValueError, match="Expected EEGBCI T0/T1/T2"):
        verified_class_event_ids(make_raw(include_all_events=False))


def test_preprocessing_selects_eeg_and_is_deterministic() -> None:
    raw = make_raw()
    first, event_info = preprocess_raw(raw)
    second, _ = preprocess_raw(raw)

    first_data = first.get_data()
    second_data = second.get_data()
    assert first.ch_names == ["C3", "Cz", "C4", "EEG004"]
    assert first_data.shape == (2, 4, 481)
    assert first_data.shape == second_data.shape
    np.testing.assert_array_equal(first_data, second_data)
    np.testing.assert_array_equal(first.events[:, 2], [event_info["class_event_ids"]["hands"], event_info["class_event_ids"]["feet"]])


def test_metadata_writer_creates_json(tmp_path) -> None:
    destination = tmp_path / "nested" / "metadata.json"
    write_metadata({"runs": [6, 10, 14], "classifier_trained": False}, destination)

    assert destination.read_text(encoding="utf-8").endswith("\n")
    assert '"classifier_trained": false' in destination.read_text(encoding="utf-8")


def test_plot_recording_creates_png(tmp_path) -> None:
    destination = tmp_path / "overview.png"
    plot_recording(make_raw(), destination, run=6)

    assert destination.is_file()
    assert destination.stat().st_size > 10_000
