#!/usr/bin/env python3
"""Small environment smoke test; does not download data."""
from __future__ import annotations

import platform
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "data/cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / "data/cache"))


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")

    modules = ["numpy", "scipy", "mne", "matplotlib"]
    for name in modules:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "installed")
            print(f"[OK] {name}: {version}")
        except Exception as exc:  # report missing/broken imports clearly
            print(f"[MISSING/ERROR] {name}: {exc}")
            raise SystemExit(1) from exc

    print("Milestone 1 environment smoke test passed. MuJoCo is deferred.")


if __name__ == "__main__":
    main()
