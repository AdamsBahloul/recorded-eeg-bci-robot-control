#!/usr/bin/env python3
"""Restore and run the run-held-out CSP + LDA baseline on existing M1 data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "data/cache"))
sys.path.insert(0, str(ROOT))

from bci.classification import (
    ClassificationConfig,
    generate_oof_predictions,
    load_m1_dataset,
    save_m2_artifacts,
)


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/m2.toml")
    parser.add_argument("--epochs", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    settings = tomllib.loads(args.config.read_text(encoding="utf-8"))
    data_settings = settings["dataset"]
    model_settings = settings["model"]
    cv_settings = settings["cross_validation"]
    output_settings = settings["output"]
    dataset = load_m1_dataset(
        resolve(args.epochs or data_settings["epochs"]),
        resolve(args.metadata or data_settings["metadata"]),
    )
    config = ClassificationConfig(
        csp_components=int(model_settings["csp_components"]),
        lda_solver=str(model_settings["lda_solver"]),
        n_splits=int(cv_settings["n_splits"]),
        shuffle=bool(cv_settings["shuffle"]),
        random_seed=int(cv_settings["random_seed"]),
    )
    result = generate_oof_predictions(dataset, config)
    output_dir = resolve(args.output_dir or output_settings["directory"])
    save_m2_artifacts(dataset, result, config, output_dir)
    print(f"M2 OOF metrics: {json.dumps(result['metrics'], sort_keys=True)}")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
