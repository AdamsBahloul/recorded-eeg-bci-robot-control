#!/usr/bin/env python3
"""Build a minimal source + model bundle for the Raspberry Pi runtime."""
from __future__ import annotations

from pathlib import Path
import shutil
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/processed/deployment/neurobot-pi-bundle.tar.gz"
PACKAGES = ("bci", "embedded", "runtime", "transport")
FILES = (
    "scripts/run_pi_controller.py",
    "config/pi.toml",
    "config/m3.toml",
    "config/m3_unvalidated_simulation_demo.toml",
    "pi/README.md",
    "pi/install.sh",
    "pi/inspect_device.sh",
    "pi/requirements-pi.txt",
    "pi/requirements-pi-lock.txt",
)
MODEL_FILES = (
    "csp_lda.joblib",
    "manifest.json",
)


def main() -> int:
    missing = [relative for relative in FILES if not (ROOT / relative).is_file()]
    missing.extend(
        f"data/processed/deployment/{name}"
        for name in MODEL_FILES
        if not (ROOT / "data/processed/deployment" / name).is_file()
    )
    if missing:
        raise FileNotFoundError(f"Cannot package Pi runtime; missing files: {', '.join(missing)}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="neurobot-pi-") as temporary:
        stage = Path(temporary) / "neurobot_starter"
        stage.mkdir()
        for package in PACKAGES:
            shutil.copytree(ROOT / package, stage / package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for relative in FILES:
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        model_target = stage / "models/deployment"
        model_target.mkdir(parents=True, exist_ok=True)
        for name in MODEL_FILES:
            shutil.copy2(ROOT / "data/processed/deployment" / name, model_target / name)
        with tarfile.open(OUTPUT, "w:gz") as archive:
            archive.add(stage, arcname="neurobot_starter")
    with tarfile.open(OUTPUT, "r:gz") as archive:
        names = archive.getnames()
        if any(name.startswith("neurobot_starter/data/raw/") for name in names):
            raise RuntimeError("Pi bundle must not contain raw dataset files")
        if any(name.startswith("neurobot_starter/data/processed/m1/") for name in names):
            raise RuntimeError("Pi bundle must not contain M1 training artifacts")
        if not all(f"neurobot_starter/models/deployment/{name}" in names for name in MODEL_FILES):
            raise RuntimeError("Pi bundle is missing the deployment model or manifest")
    print(f"Pi deployment bundle: {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")
    print("Contents: Pi inference/runtime source, safety configuration, and deployment model only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
