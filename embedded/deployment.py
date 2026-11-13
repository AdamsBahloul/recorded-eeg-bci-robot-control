"""Build and validate the separately versioned full-data deployment model."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import platform

import joblib
import mne
import numpy as np
import sklearn
from sklearn.base import clone

from bci.classification import ClassificationConfig, load_m1_dataset, make_csp_lda_pipeline
from bci.preprocessing import PreprocessingConfig


MODEL_VERSION = "neurobot-csp-lda-deploy-v1"


def build_deployment_model(epochs_path: Path, metadata_path: Path, output_dir: Path) -> dict:
    """Fit on all permitted labeled M1 epochs for deployment, not evaluation."""
    dataset = load_m1_dataset(epochs_path, metadata_path)
    config = ClassificationConfig()
    model = clone(make_csp_lda_pipeline(config)).fit(dataset.X, dataset.y)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "csp_lda.joblib"
    joblib.dump(model, model_path, compress=3)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest = {
        "model_version": MODEL_VERSION,
        "model_file": model_path.name,
        "sha256": digest,
        "role": "deployment inference only; not an independent evaluation model",
        "training_data": {
            "subject": 1, "runs": [6, 10, 14], "epoch_count": 45,
            "epochs_file": epochs_path.name,
            "epochs_sha256": hashlib.sha256(epochs_path.read_bytes()).hexdigest(),
            "metadata_file": metadata_path.name,
            "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        },
        "class_mapping": {"hands": 0, "feet": 1},
        "input_shape": [64, 481],
        "channel_names": list(dataset.channel_names),
        "sampling_frequency_hz": 160.0,
        "preprocessing": asdict(PreprocessingConfig()),
        "classifier": {"pipeline": ["MNE CSP", "scikit-learn LDA"], "csp_components": 4, "lda_solver": "svd"},
        "software": {"python": platform.python_version(), "mne": mne.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "evaluation_reference": str(epochs_path),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_deployment_model(model_path: Path, manifest_path: Path):
    """Fail closed on checksum or software/input incompatibility."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != manifest.get("sha256"):
        raise ValueError("Deployment model checksum mismatch")
    if manifest.get("model_version") != MODEL_VERSION or manifest.get("class_mapping") != {"hands": 0, "feet": 1}:
        raise ValueError("Unsupported deployment model version or class mapping")
    expected = manifest.get("software", {})
    current = {"mne": mne.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__}
    if any(expected.get(key) != value for key, value in current.items()):
        raise ValueError("Deployment model dependency versions do not match this runtime")
    model = joblib.load(model_path)
    if model.named_steps["csp"].n_components != 4 or model.named_steps["lda"].solver != "svd":
        raise ValueError("Deployment model classifier settings are incompatible")
    if not np.array_equal(model.classes_, np.asarray([0, 1])):
        raise ValueError("Deployment model classes must be hands=0, feet=1")
    return model, manifest
