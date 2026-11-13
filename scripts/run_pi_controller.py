#!/usr/bin/env python3
"""Run local EEG inference and safety control on the Raspberry Pi."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "data/cache"))
sys.path.insert(0, str(ROOT))

from embedded.controller import connect_and_serve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="MacBook LAN address")
    parser.add_argument("--port", type=int)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--config", type=Path, default=Path("config/pi.toml"))
    parser.add_argument("--runtime-config", type=Path,
                        help="override the safety configuration named in the Pi TOML")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    root = args.project_root.resolve()
    model_path = (root / config["model"]["path"]).resolve()
    manifest_path = (root / config["model"]["manifest"]).resolve()
    runtime_path = (root / (args.runtime_config or config["runtime"]["config"])).resolve()
    port = args.port or int(config["network"]["port"])
    timeout = float(config["network"]["connect_timeout_seconds"])
    result = connect_and_serve(
        args.host, port, model_path=model_path, manifest_path=manifest_path,
        config_path=runtime_path, project_root=root, session_id=args.session_id,
        timeout_seconds=timeout,
    )
    print(f"Pi replay complete: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
