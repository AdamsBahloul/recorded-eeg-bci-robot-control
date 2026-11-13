"""Pi-side reconstruction of the verified M1 filtering and epoching path."""
from __future__ import annotations

import mne
import numpy as np

from bci.preprocessing import PreprocessingConfig
from embedded.signal_message import EEGRunMessage


def preprocess_run(message: EEGRunMessage, config: PreprocessingConfig = PreprocessingConfig()) -> np.ndarray:
    """Filter the raw run and return unlabeled (epochs, channels, samples)."""
    message.validate()
    info = mne.create_info(list(message.channel_names), message.sampling_frequency_hz, ch_types="eeg")
    # MNE may modify RawArray storage while filtering; own a copy so callers can
    # safely retry, replay, checksum, or independently audit the original input.
    raw = mne.io.RawArray(np.array(message.signal_volts, dtype=np.float64, copy=True), info, verbose="ERROR")
    events = np.zeros((len(message.event_samples), 3), dtype=np.int64)
    events[:, 0] = message.event_samples
    events[:, 2] = 1  # A single class-neutral event code; no annotation identities are transmitted.
    raw.filter(
        l_freq=config.l_freq_hz, h_freq=config.h_freq_hz, picks="eeg", method="fir",
        phase=config.filter_phase, filter_length=config.filter_length,
        l_trans_bandwidth=config.l_trans_bandwidth_hz,
        h_trans_bandwidth=config.h_trans_bandwidth_hz, fir_window=config.fir_window,
        fir_design=config.fir_design, pad=config.filter_pad, verbose="ERROR",
    )
    epochs = mne.Epochs(
        raw, events, event_id={"task": 1}, tmin=config.epoch_tmin_s,
        tmax=config.epoch_tmax_s, baseline=config.baseline, picks="eeg",
        preload=True, reject_by_annotation=False, verbose="ERROR",
    )
    result = epochs.get_data(copy=True)
    expected_shape = (len(message.event_samples), 64, 481)
    if result.shape != expected_shape or not np.isfinite(result).all():
        raise ValueError(f"Pi preprocessing produced invalid epoch array {result.shape}")
    return result
