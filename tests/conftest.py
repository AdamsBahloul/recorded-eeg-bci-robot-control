"""Make the repository modules importable in an uninstalled checkout."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def m1_dataset():
    from bci.classification import load_m1_dataset

    return load_m1_dataset(
        PROJECT_ROOT / "data/processed/m1/S001_epochs.npz",
        PROJECT_ROOT / "data/processed/m1/S001_metadata.json",
    )


@pytest.fixture(scope="session")
def heldout_models(m1_dataset):
    from bci.classification import ClassificationConfig, fit_run_heldout_models

    return fit_run_heldout_models(m1_dataset, ClassificationConfig())[0]
