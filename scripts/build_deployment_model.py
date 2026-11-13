#!/usr/bin/env python3
"""Build the full-data deployment model (separate from M2 CV artifacts)."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/cache/matplotlib"))
sys.path.insert(0, str(ROOT))

from embedded.deployment import build_deployment_model


def main() -> int:
    manifest = build_deployment_model(
        ROOT / "data/processed/m1/S001_epochs.npz",
        ROOT / "data/processed/m1/S001_metadata.json",
        ROOT / "data/processed/deployment",
    )
    print(f"Built {manifest['model_version']} ({manifest['sha256']})")
    print("Role: deployment inference only; do not report training-set predictions as evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
