#!/usr/bin/env python3
"""Send raw recorded EEG to a Pi and validate its generated commands."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import tomllib

from bci.replay_source import load_raw_run_message
from mac.command_validator import MacCommandValidator
from mac.replay_server import serve_replay_session


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/mac.toml")
    parser.add_argument("--runs", nargs="+", type=int)
    parser.add_argument("--bind-host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/distributed")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    network = config["network"]
    data = config["data"]
    bind_host = args.bind_host or network["bind_host"]
    port = args.port or int(network["port"])
    runs = args.runs or data["runs"]
    if any(run not in (6, 10, 14) for run in runs):
        parser.error("only runs 6, 10, and 14 are allowed")
    raw_dir = (ROOT / data["raw_directory"]).resolve()
    messages = [
        load_raw_run_message(raw_dir / f"S001R{run:02d}.edf", run, args.session_id, sequence)
        for sequence, run in enumerate(runs, start=1)
    ]
    validator = MacCommandValidator(args.session_id, float(network["maximum_command_age_seconds"]))

    def hold_only_sink(event):
        if event.command_type == "HOLD" and event.new_state in ("HOLD", "ERROR"):
            return {"accepted": True, "reason": "safe_no_motion_state"}
        return {"accepted": False, "reason": "simulation_not_enabled_or_not_connected"}

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((bind_host, port))
        server.listen(1)
        print(f"Waiting for Raspberry Pi at {bind_host}:{port}; session={args.session_id}")
        connection, address = server.accept()
        with connection:
            connection.settimeout(float(network["client_timeout_seconds"]))
            result = serve_replay_session(
                connection, messages=messages, session_id=args.session_id,
                command_validator=validator, command_handler=hold_only_sink,
            )
            result["peer_address"] = str(address)
            result["simulation_enabled"] = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result["dataset"] = {"participant": 1, "runs": runs, "input": "raw EEG values; no labels or precomputed predictions transmitted"}
    (args.output_dir / f"{args.session_id}_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
