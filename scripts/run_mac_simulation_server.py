#!/usr/bin/env python3
"""Open the Mac MuJoCo arm and serve raw EEG replay to a Raspberry Pi."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
from threading import Thread
import time
import tomllib
import sys
from queue import Queue, Empty

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "data/cache"))
sys.path.insert(0, str(ROOT))

import mujoco
import mujoco.viewer

from bci.replay_source import load_raw_run_message, load_sequence_message
from mac.command_validator import MacCommandValidator
from mac.event_log import append_event
from mac.replay_server import serve_replay_session
from bci.replay_control import ReplayControl
from simulation.controller import ArmSimulation


def choose_runs() -> list[int]:
    """Offer the existing recordings before opening the simulation server."""
    choices = {"1": [6], "2": [10], "3": [14], "4": [6, 10, 14]}
    print("\nChoose recorded EEG to replay (participant 1):")
    print("  1) Run 6  — imagined hands/feet")
    print("  2) Run 10 — imagined hands/feet")
    print("  3) Run 14 — imagined hands/feet")
    print("  4) All three runs")
    while True:
        selected = input("Selection [4]: ").strip() or "4"
        if selected in choices:
            return choices[selected]
        print("Enter 1, 2, 3, or 4.")


def parse_class_sequence(text: str) -> list[int]:
    """Parse a requested hands/feet replay sequence without serializing labels."""
    import re

    classes: list[int] = []
    for token in (part for part in re.split(r"[,\s]+", text.strip().upper()) if part):
        match = re.fullmatch(r"(HANDS|FEET)(?:[X*](\d+))?", token)
        if not match:
            raise ValueError(f"Unknown class {token!r}; use HANDS or FEET")
        count = int(match.group(2) or 1)
        if not 1 <= count <= 15 or len(classes) + count > 15:
            raise ValueError("A replay sequence is limited to 15 trials")
        classes.extend([0 if match.group(1) == "HANDS" else 1] * count)
    if not classes:
        raise ValueError("Enter at least one HANDS or FEET trial")
    return classes


def choose_input_source() -> str:
    print("\nChoose EEG input for Raspberry Pi inference:")
    print("  1) Raw EDF recording")
    print("  2) Generate a hands/feet sequence from raw EDF trial windows")
    while True:
        selected = input("Selection [1]: ").strip() or "1"
        if selected in ("1", "2"):
            return "edf" if selected == "1" else "sequence"
        print("Enter 1 or 2.")


def choose_class_sequence() -> str:
    return input("Enter a class sequence, e.g. HANDS, FEET*2, HANDS: ").strip()


def repeated_messages(messages):
    """Yield replay copies with unique recording IDs and increasing sequences."""
    sequence = 1
    cycle = 1
    while True:
        for message in messages:
            yield replace(
                message,
                sequence=sequence,
                recording_id=f"{message.recording_id}-cycle-{cycle:06d}",
            )
            sequence += 1
        cycle += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/mac.toml")
    parser.add_argument("--m3-config", type=Path, default=ROOT / "config/m3.toml",
                        help="Pi safety configuration mirrored for independent Mac validation")
    parser.add_argument("--source", choices=("edf", "sequence"), help="input source; prompts when omitted")
    parser.add_argument("--runs", nargs="+", type=int)
    parser.add_argument("--edf-path", type=Path, help="one authorized raw EDF file named S001R06/10/14.edf")
    parser.add_argument("--sequence", help="HANDS/FEET replay sequence, e.g. 'HANDS FEET*2'")
    parser.add_argument("--sequence-run", type=int, choices=(6, 10, 14), default=6,
                        help="approved EDF run used as the sequence generator's raw signal source")
    parser.add_argument("--bind-host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--enable-simulation", action="store_true", help="permit validated HANDS/FEET pose changes")
    parser.add_argument("--speed", type=float, help="recorded-time replay multiplier; >1 runs faster")
    parser.add_argument("--chunk-samples", type=int)
    parser.add_argument("--interactive-controls", action="store_true", help="accept pause, resume, and stop from stdin")
    parser.add_argument("--repeat-forever", action="store_true",
                        help="replay the selected EEG input continuously until stopped")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/distributed")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    network, data_config = config["network"], config["data"]
    bind_host = args.bind_host or network["bind_host"]
    port = args.port or int(network["port"])
    source = args.source or (
        "edf" if args.edf_path is not None or args.runs is not None
        else "sequence" if args.sequence is not None
        else choose_input_source()
    )
    raw_dir = (ROOT / data_config["raw_directory"]).resolve()
    if source == "edf":
        if args.sequence:
            parser.error("--sequence can only be used with --source sequence")
        if args.edf_path is not None:
            import re

            if args.runs is not None:
                parser.error("--edf-path cannot be combined with --runs")
            match = re.fullmatch(r"S001R(06|10|14)\.edf", args.edf_path.name, flags=re.IGNORECASE)
            if not match:
                parser.error("EDF file must be participant 1 run 6, 10, or 14")
            runs = [int(match.group(1))]
            edf_paths = {runs[0]: args.edf_path.expanduser().resolve()}
        else:
            runs = args.runs if args.runs is not None else choose_runs()
            edf_paths = {run: raw_dir / f"S001R{run:02d}.edf" for run in runs}
        if not runs or any(run not in (6, 10, 14) for run in runs):
            parser.error("select one or more of runs 6, 10, and 14")
        messages = [
            load_raw_run_message(edf_paths[run], run, args.session_id, i)
            for i, run in enumerate(runs, start=1)
        ]
        source_description = "raw EDF replay; labels and Mac predictions are not transmitted"
    else:
        if args.runs is not None or args.edf_path is not None:
            parser.error("--runs and --edf-path can only be used with --source edf")
        try:
            requested_classes = parse_class_sequence(args.sequence or choose_class_sequence())
        except ValueError as exc:
            parser.error(str(exc))
        sequence_path = raw_dir / f"S001R{args.sequence_run:02d}.edf"
        messages = [load_sequence_message(
            sequence_path, args.sequence_run, args.session_id, 1, requested_classes,
        )]
        source_description = (
            f"label-free generated replay of {len(requested_classes)} raw trial windows "
            f"from run {args.sequence_run}; requested labels remain on the Mac"
        )
    if args.repeat_forever:
        messages = repeated_messages(messages)
        args.interactive_controls = True
        source_description += "; repeating until you type 'stop' or close the viewer"
    m3_path = args.m3_config if args.m3_config.is_absolute() else ROOT / args.m3_config
    m3_confidence = tomllib.loads(m3_path.read_text(encoding="utf-8"))["confidence"]
    active_threshold = float(m3_confidence["threshold"]) if m3_confidence["enabled"] else None
    if m3_path.name == "m3_unvalidated_simulation_demo.toml":
        print("WARNING: using the unvalidated threshold override for simulation only; never use with physical hardware.")
    simulation_enabled = bool(config["simulation"]["enabled"] and args.enable_simulation)
    print(f"Prepared input: {source_description}")
    print(
        "Active simulation commands: enabled only with a configured M3 decision-margin threshold, "
        f"Mac config gate, and --enable-simulation (currently {'available' if active_threshold is not None and simulation_enabled else 'disabled; Pi predictions will return as HOLD'})."
    )
    validator = MacCommandValidator(
        args.session_id,
        float(network["maximum_command_age_seconds"]),
        active_threshold=active_threshold if simulation_enabled else None,
    )
    arm = ArmSimulation(ROOT / "simulation/arm.xml")
    replay_control = ReplayControl()
    replay_speed = float(args.speed if args.speed is not None else config["replay"]["speed"])
    chunk_samples = int(args.chunk_samples if args.chunk_samples is not None else config["replay"]["chunk_samples"])
    if replay_speed <= 0 or chunk_samples < 1:
        parser.error("replay speed and chunk sample count must be positive")
    event_log_path = args.output_dir / f"{args.session_id}_events.jsonl"
    result: dict = {}
    failures: list[str] = []
    listener_holder: list[socket.socket] = []
    connection_holder: list[socket.socket] = []
    viewer_controls: Queue[str] = Queue()

    def serve() -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                listener_holder.append(server)
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((bind_host, port))
                server.listen(1)
                print(f"MuJoCo arm is ready; waiting for Pi at {bind_host}:{port}; session={args.session_id}")
                connection, _address = server.accept()
                connection_holder.append(connection)
                with connection:
                    connection.settimeout(float(network["client_timeout_seconds"]))
                    def apply_validated_command(event):
                        outcome = arm.apply_command(event)
                        append_event(
                            event_log_path, "simulation_command_execution",
                            command_id=event.command_id, command_type=event.command_type,
                            accepted=outcome["accepted"], outcome=outcome, arm_status=arm.status(),
                        )
                        if event.predicted_class in (0, 1):
                            decoded = "HANDS" if event.predicted_class == 0 else "FEET"
                            margin = event.confidence_measure
                            print(
                                f"Pi decoded {decoded} (LDA decision-score margin={margin:.4f}); "
                                f"simulation command={event.command_type}, accepted={outcome['accepted']} "
                                f"(state={arm.state})"
                            )
                        else:
                            print(f"{event.command_id} {event.command_type}: {outcome['reason']} (state={arm.state})")
                        return outcome

                    result.update(serve_replay_session(
                        connection, messages=messages, session_id=args.session_id,
                        command_validator=validator,
                        command_handler=lambda event: (
                            apply_validated_command(event)
                            if (event.command_type == "HOLD" or simulation_enabled)
                            else {"accepted": False, "reason": "simulation_disabled"}
                        ),
                        event_sink=lambda event_type, **fields: append_event(
                            event_log_path, event_type, **fields
                        ),
                        chunk_samples=chunk_samples, replay_speed=replay_speed,
                        replay_control=replay_control,
                    ))
        except InterruptedError:
            arm.force_hold("replay_stopped")
            result["stopped_by_operator"] = True
            append_event(event_log_path, "replay_stopped_by_operator", fallback_state="HOLD")
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
            arm.force_hold("communication_or_server_failure")
            append_event(event_log_path, "communication_or_server_failure", error=failures[-1], fallback_state="HOLD")

    if args.interactive_controls:
        def read_controls() -> None:
            print("Replay controls: type 'pause', 'resume', or 'stop'.")
            for line in sys.stdin:
                command = line.strip().lower()
                if command == "pause":
                    replay_control.pause()
                    arm.force_hold("replay_paused")
                    append_event(event_log_path, "replay_paused", fallback_state="HOLD")
                elif command == "resume":
                    replay_control.resume()
                    append_event(event_log_path, "replay_resumed")
                elif command == "stop":
                    replay_control.stop()
                    break

        Thread(target=read_controls, daemon=True).start()

    server_thread = Thread(target=serve, daemon=True)
    server_thread.start()
    print("Simulation controls: press R in the MuJoCo window to reset to safe home; press H to hold.")
    try:
        with mujoco.viewer.launch_passive(arm.model, arm.data, key_callback=lambda key: viewer_controls.put(chr(key).lower())) as viewer:
            while viewer.is_running() and server_thread.is_alive():
                while True:
                    try:
                        key = viewer_controls.get_nowait()
                    except Empty:
                        break
                    if key == "r":
                        arm.reset()
                        append_event(event_log_path, "simulation_reset", state="HOLD")
                    elif key == "h":
                        arm.force_hold("operator_hold")
                        append_event(event_log_path, "operator_hold", state="HOLD")
                step_started = time.perf_counter()
                with viewer.lock():
                    arm.step()
                viewer.sync()
                remaining = arm.model.opt.timestep - (time.perf_counter() - step_started)
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        replay_control.stop()
        arm.force_hold("operator_stop")
    if server_thread.is_alive():
        replay_control.stop()
        arm.force_hold("viewer_closed_or_server_stopped")
        for active_socket in connection_holder + listener_holder:
            try:
                active_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                active_socket.close()
            except OSError:
                pass
        if not failures:
            failures.append("simulation viewer closed before replay session completed")
    server_thread.join(timeout=1.0)
    if not server_thread.is_alive():
        result["simulation_final_state"] = arm.status()
        result["active_commands_enabled"] = active_threshold is not None and simulation_enabled
        result["failures"] = failures
        result["source"] = source_description
        result["hardware_note"] = "Pi hardware result only when a physical Pi connects"
        result["event_log"] = str(event_log_path)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"{args.session_id}_system_summary.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
