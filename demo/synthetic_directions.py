"""A deliberately synthetic four-direction EEG demo; not a BCI decoder."""
from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


ACTIONS = ("APPROACH_A", "LOWER", "GRASP", "LIFT", "MOVE_B", "RELEASE", "RETRACT", "HOME", "HOLD")
CHANNELS = ("Fp1", "Fp2", "F3", "F4", "C3", "Cz", "C4", "P3", "P4", "O1", "O2", "T7", "T8", "Pz")
SFREQ = 160.0
N_SAMPLES = 481
_PATTERNS = np.zeros((len(ACTIONS), len(CHANNELS)))
_ACTION_CHANNELS = ((2, 3, 5), (7, 8, 13), (4, 11, 2), (6, 12, 3), (5, 6, 9),
                    (2, 7, 10), (3, 8, 12), (4, 6, 13), (2, 5, 11))
for _row, _indices in enumerate(_ACTION_CHANNELS):
    _PATTERNS[_row, list(_indices)] = 1.0


def synthesize_epoch(direction: str, seed: int = 42) -> np.ndarray:
    """Generate an artificial direction-coded array (volts; channels x samples)."""
    if direction not in ACTIONS:
        raise ValueError(f"action must be one of {ACTIONS}")
    rng = np.random.default_rng(seed)
    times = np.arange(N_SAMPLES) / SFREQ
    label = ACTIONS.index(direction)
    carrier = np.sin(2 * np.pi * (7 + label) * times + rng.uniform(0, 2 * np.pi))
    envelope = np.sin(2 * np.pi * 0.7 * times) * 0.15 + 1.0
    spatial = _PATTERNS[label] * 0.8 + 0.2
    data = rng.normal(0, 0.12, (len(CHANNELS), N_SAMPLES))
    data += (spatial[:, None] * carrier[None, :] * envelope[None, :])
    data += 0.25 * rng.normal(size=(len(CHANNELS), 1)) * np.sin(2 * np.pi * 6 * times)
    return (data * 1e-6).astype(np.float64)


def make_synthetic_model(seed: int = 42) -> Pipeline:
    """Fit spectral-feature+LDA on generated examples only; no PhysioNet data."""
    rng = np.random.default_rng(seed)
    epochs, labels = [], []
    for class_id, direction in enumerate(ACTIONS):
        for _ in range(32):
            epochs.append(synthesize_epoch(direction, int(rng.integers(0, 2**32 - 1))))
            labels.append(class_id)
    model = Pipeline([
        ("synthetic_frequency_features", FunctionTransformer(_frequency_features, validate=False)),
        ("lda", LinearDiscriminantAnalysis(solver="svd")),
    ])
    model.fit(np.asarray(epochs), np.asarray(labels))
    return model


def _frequency_features(epochs: np.ndarray) -> np.ndarray:
    """Mean channel power near each artificial class carrier frequency."""
    spectrum = np.abs(np.fft.rfft(epochs, axis=-1)) ** 2
    frequencies = np.fft.rfftfreq(epochs.shape[-1], 1 / SFREQ)
    carriers = tuple(float(frequency) for frequency in range(7, 7 + len(ACTIONS)))
    features = []
    for carrier in carriers:
        bins = np.abs(frequencies - carrier) <= 0.7
        features.append(spectrum[..., bins].mean(axis=(1, 2)))
    return np.column_stack(features)


def predict_synthetic(model: Pipeline, signal: np.ndarray) -> tuple[str, np.ndarray]:
    if signal.shape != (len(CHANNELS), N_SAMPLES) or not np.isfinite(signal).all():
        raise ValueError(f"expected finite EEG array shaped ({len(CHANNELS)}, {N_SAMPLES})")
    prediction = int(model.predict(signal[None, ...])[0])
    return ACTIONS[prediction], model.decision_function(signal[None, ...])[0]
