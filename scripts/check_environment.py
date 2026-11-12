#!/usr/bin/env python3
"""Small environment smoke test; does not download data."""
from __future__ import annotations

import platform
import sys


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")

    modules = ["numpy", "scipy", "mne", "sklearn", "mujoco"]
    for name in modules:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "installed")
            print(f"[OK] {name}: {version}")
        except Exception as exc:  # report missing/broken imports clearly
            print(f"[MISSING/ERROR] {name}: {exc}")
            raise SystemExit(1) from exc

    import mujoco
    print(f"[OK] MuJoCo version: {mujoco.__version__}")
    print("Environment smoke test passed.")


if __name__ == "__main__":
    main()
