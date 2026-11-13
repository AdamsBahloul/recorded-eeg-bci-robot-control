"""Mac-side extraction of label-free raw signal runs for Pi replay."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np

from embedded.signal_message import EEGRunMessage


def load_raw_run_message(edf_path: Path, run_id: int, session_id: str, sequence: int) -> EEGRunMessage:
    """Read one authorized EDF and emit raw samples plus unlabeled task times."""
    if run_id not in (6, 10, 14):
        raise ValueError("Only runs 6, 10, and 14 are allowed")
    if not edf_path.is_file():
        raise FileNotFoundError(edf_path)
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    mne.datasets.eegbci.standardize(raw)
    raw.pick("eeg")
    if raw.info["sfreq"] != 160.0 or len(raw.ch_names) != 64:
        raise ValueError("Raw EEG does not match the verified 64-channel, 160 Hz layout")
    events, event_ids = mne.events_from_annotations(raw, verbose="ERROR")
    if not {"T1", "T2"}.issubset(event_ids):
        raise ValueError("Raw recording lacks verified hands/feet task annotation timings")
    task_codes = {event_ids["T1"], event_ids["T2"]}
    # Strip T1/T2 identity: only event positions are transmitted to the Pi.
    event_samples = tuple(sorted(int(sample - raw.first_samp) for sample, _, code in events if code in task_codes))
    if len(event_samples) != 15:
        raise ValueError(f"Expected 15 unlabeled task event boundaries; got {len(event_samples)}")
    signal = raw.get_data(picks="eeg")
    message = EEGRunMessage(
        session_id=session_id,
        source_id="mac-replay",
        recording_id=f"S001R{run_id:02d}",
        run_id=run_id,
        sequence=sequence,
        timestamp_utc=(raw.info["meas_date"] or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        sampling_frequency_hz=float(raw.info["sfreq"]),
        channel_names=tuple(raw.ch_names),
        event_samples=event_samples,
        signal_volts=np.asarray(signal, dtype=np.float64),
    )
    message.validate()
    return message


def load_sequence_message(
    edf_path: Path,
    run_id: int,
    session_id: str,
    sequence: int,
    requested_classes: list[int],
) -> EEGRunMessage:
    """Compose a label-free signal sequence from raw trials in one approved EDF.

    The requested classes are used only on the Mac to select existing raw EEG
    trial windows. Their identities are deliberately absent from the message;
    the Pi receives the concatenated signal and unlabeled event boundaries.
    """
    if run_id not in (6, 10, 14):
        raise ValueError("Only runs 6, 10, and 14 are allowed")
    if not requested_classes or len(requested_classes) > 15:
        raise ValueError("A generated replay sequence must contain 1 to 15 trials")
    if any(type(label) is not int or label not in (0, 1) for label in requested_classes):
        raise ValueError("Generated replay classes must be hands=0 or feet=1")
    if not edf_path.is_file():
        raise FileNotFoundError(edf_path)

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    mne.datasets.eegbci.standardize(raw)
    raw.pick("eeg")
    if raw.info["sfreq"] != 160.0 or len(raw.ch_names) != 64:
        raise ValueError("Raw EEG does not match the verified 64-channel, 160 Hz layout")
    events, event_ids = mne.events_from_annotations(raw, verbose="ERROR")
    if not {"T1", "T2"}.issubset(event_ids):
        raise ValueError("Raw recording lacks verified hands/feet task annotation timings")

    # Resolve annotations locally only to choose signal windows; never serialize them.
    class_codes = {0: event_ids["T1"], 1: event_ids["T2"]}
    candidates = {
        label: [int(sample - raw.first_samp) for sample, _, code in events if code == event_code]
        for label, event_code in class_codes.items()
    }
    if any(not samples for samples in candidates.values()):
        raise ValueError("Raw recording must contain both verified hands and feet trials")

    sfreq = int(raw.info["sfreq"])
    samples_before_event = 2 * sfreq
    samples_after_event = 6 * sfreq
    segment_length = samples_before_event + samples_after_event
    raw_data = raw.get_data(picks="eeg")
    positions: list[int] = []
    segments: list[np.ndarray] = []
    next_candidate = {0: 0, 1: 0}
    for index, label in enumerate(requested_classes):
        options = candidates[label]
        chosen = options[next_candidate[label] % len(options)]
        next_candidate[label] += 1
        start = chosen - samples_before_event
        stop = chosen + samples_after_event
        if start < 0 or stop > raw_data.shape[1]:
            raise ValueError(f"Task trial at sample {chosen} lacks enough surrounding raw signal")
        segments.append(raw_data[:, start:stop])
        positions.append(index * segment_length + samples_before_event)

    signal = np.concatenate(segments, axis=1).astype(np.float64, copy=False)
    message = EEGRunMessage(
        session_id=session_id,
        source_id="mac-sequence-replay",
        recording_id=f"S001R{run_id:02d}-sequence-{sequence:03d}",
        run_id=run_id,
        sequence=sequence,
        timestamp_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        sampling_frequency_hz=float(raw.info["sfreq"]),
        channel_names=tuple(raw.ch_names),
        event_samples=tuple(positions),
        signal_volts=signal,
    )
    message.validate()
    return message
