#!/usr/bin/env python3
"""One-window synthetic EEG + arm demo with an editable action sequence."""
from __future__ import annotations

from pathlib import Path
import re
import json
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import numpy as np

from bci.replay_source import load_raw_run_message
from demo.synthetic_directions import ACTIONS, CHANNELS, SFREQ, make_synthetic_model, predict_synthetic, synthesize_epoch

RAW_DIR = ROOT / "data/raw/MNE-eegbci-data/files/eegmmidb/1.0.0/S001"
MAX_SEQUENCE_ACTIONS = 24

PAPER = "#d8d2c2"
PANEL = "#e9e4d8"
INK = "#34352f"
MUTED = "#77756b"
RULE = "#aaa596"
ACCENT = "#8e382b"
TRACE_COLORS = ("#52675e", "#8e382b", "#526783", "#89784f", "#646254", "#a05d43", "#68716c")


def parse_sequence(text: str) -> list[str]:
    """Parse comma/space separated action names with optional xN repetition."""
    sequence: list[str] = []
    for token in (part for part in re.split(r"[,\s]+", text.strip().upper()) if part):
        match = re.fullmatch(r"([A-Z_]+)(?:[X*](\d+))?", token)
        if not match or match.group(1) not in ACTIONS:
            raise ValueError(f"Unknown action {token!r}. Choose from: {', '.join(ACTIONS)}")
        count = int(match.group(2) or 1)
        if not 1 <= count <= MAX_SEQUENCE_ACTIONS:
            raise ValueError(f"Repetition must be from 1 to {MAX_SEQUENCE_ACTIONS}")
        sequence.extend([match.group(1)] * count)
        if len(sequence) > MAX_SEQUENCE_ACTIONS:
            raise ValueError(f"Sequence is limited to {MAX_SEQUENCE_ACTIONS} actions")
    if not sequence:
        raise ValueError("Enter at least one action")
    return sequence


def _plot_signal(axis, signal: np.ndarray, sfreq: float, title: str) -> None:
    axis.clear()
    axis.set_facecolor(PANEL)
    times = np.arange(signal.shape[1]) / sfreq
    scale_uv = max(float(np.std(signal)) * 4e6, 1e-3)
    for index in range(signal.shape[0]):
        axis.plot(times, signal[index] * 1e6 + (signal.shape[0] - index) * scale_uv,
                  lw=0.9, color=TRACE_COLORS[index % len(TRACE_COLORS)], alpha=0.9)
    axis.set_title(title, loc="left", color=INK, fontsize=12, weight="bold", pad=12)
    axis.set_xlabel("TIME  /  seconds", color=MUTED, fontsize=9, labelpad=8)
    axis.set_ylabel("CHANNELS  /  µV offset", color=MUTED, fontsize=9, labelpad=8)
    axis.tick_params(colors=MUTED, labelsize=8)
    axis.grid(color=RULE, alpha=0.42, linewidth=0.6)
    for spine in axis.spines.values():
        spine.set_color(RULE)


def _animate_signal(axis, signal: np.ndarray, sfreq: float, title: str, status, stop_requested) -> bool:
    """Reveal one synthetic epoch progressively at its configured sample rate."""
    axis.clear()
    axis.set_facecolor(PANEL)
    times = np.arange(signal.shape[1]) / sfreq
    scale_uv = max(float(np.std(signal)) * 4e6, 1e-3)
    offsets = np.arange(signal.shape[0], 0, -1) * scale_uv
    lines = [axis.plot([], [], lw=0.95, color=TRACE_COLORS[i % len(TRACE_COLORS)], alpha=0.9)[0]
             for i in range(signal.shape[0])]
    axis.set_xlim(0, times[-1])
    low = float(np.min(signal) * 1e6)
    high = float(np.max(signal) * 1e6 + offsets[0])
    axis.set_ylim(low - scale_uv, high + scale_uv)
    axis.set_title(title, loc="left", color=INK, fontsize=12, weight="bold", pad=12)
    axis.set_xlabel("TIME  /  seconds", color=MUTED, fontsize=9, labelpad=8)
    axis.set_ylabel("CHANNELS  /  µV offset", color=MUTED, fontsize=9, labelpad=8)
    axis.tick_params(colors=MUTED, labelsize=8)
    axis.grid(color=RULE, alpha=0.42, linewidth=0.6)
    for spine in axis.spines.values():
        spine.set_color(RULE)
    frames = min(90, signal.shape[1])
    interval = signal.shape[1] / sfreq / frames
    for frame in range(frames):
        if stop_requested["value"]:
            return False
        end = min(signal.shape[1], int((frame + 1) * signal.shape[1] / frames))
        for channel, line in enumerate(lines):
            line.set_data(times[:end], signal[channel, :end] * 1e6 + offsets[channel])
        status.set_text(f"Drawing EEG in real time: {end}/{signal.shape[1]} samples • {end/sfreq:.2f}/{times[-1]:.2f} s")
        axis.figure.canvas.draw_idle()
        plt.pause(interval)
    return True


def main() -> int:
    model = make_synthetic_model()
    mjpython = ROOT / ".venv/bin/mjpython"
    if not mjpython.is_file():
        raise FileNotFoundError(f"MuJoCo launcher not found: {mjpython}; install requirements-simulation.txt first")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    viewer_process = subprocess.Popen([str(mjpython), str(ROOT / "scripts/run_synthetic_arm_viewer.py"), "--port", str(port)])
    listener.settimeout(20)
    try:
        viewer_connection, _ = listener.accept()
    except (TimeoutError, OSError):
        viewer_process.terminate()
        raise RuntimeError("3D MuJoCo viewer did not connect; check its startup output")
    finally:
        listener.close()
    viewer_connection.settimeout(5)
    ready = viewer_connection.makefile("rb").readline()
    if not ready:
        viewer_process.terminate()
        raise RuntimeError("3D MuJoCo viewer closed before becoming ready")
    fig = plt.figure(figsize=(11, 8), facecolor=PAPER)
    fig.text(0.075, 0.948, "NEUROBOT", color=INK, fontsize=10, weight="bold", family="DejaVu Sans")
    fig.text(0.925, 0.948, "RECORDED EEG  /  SIMULATION", color=MUTED, fontsize=8, ha="right")
    fig.add_artist(plt.Line2D([0.075, 0.925], [0.925, 0.925], transform=fig.transFigure, color=RULE, lw=0.8))
    for x, label, active in ((0.075, "SIGNAL", True), (0.19, "SEQUENCE", False), (0.34, "ARM SIMULATION", False)):
        fig.text(x, 0.89, label, color=ACCENT if active else MUTED, fontsize=8, weight="bold" if active else "normal")
    fig.text(0.925, 0.89, "01  /  DEMO", color=MUTED, fontsize=8, ha="right")
    eeg_ax = fig.add_axes([0.09, 0.405, 0.82, 0.435], facecolor=PANEL)
    eeg_ax.set_title("EEG SIGNAL  /  WAITING FOR SEQUENCE", loc="left", color=INK, fontsize=12, weight="bold", pad=12)
    status = fig.text(0.09, 0.365, "READY   ·   Enter a sequence, then start replay.", ha="left", color=INK, fontsize=10, weight="bold")
    warning = fig.text(0.09, 0.337, "ARTIFICIAL SIGNAL + SYNTHETIC MODEL  ·  Not real EEG decoding or robotic control.", ha="left", color=MUTED, fontsize=8)

    text_ax = fig.add_axes([0.09, 0.247, 0.82, 0.055], facecolor=PANEL)
    sequence_box = TextBox(text_ax, "TASK SEQUENCE  ", initial="APPROACH_A, LOWER, GRASP, LIFT, MOVE_B, LOWER, RELEASE, RETRACT, HOME", color=PANEL, hovercolor="#f1ede4")
    sequence_box.label.set_color(MUTED)
    sequence_box.text_disp.set_color(INK)
    for spine in text_ax.spines.values():
        spine.set_color(RULE)
    buttons = []
    for x, label, color, hover in ((0.24, "RUN REPLAY", ACCENT, "#a74b3d"), (0.43, "STOP  /  HOLD", "#514f46", "#69665b"), (0.62, "RESET ARM", "#b5ae9e", "#c6bead")):
        button_ax = fig.add_axes([x, 0.158, 0.15, 0.052])
        button = Button(button_ax, label, color=color, hovercolor=hover)
        button.label.set_color("#f5f1e7" if color != "#b5ae9e" else INK)
        button.label.set_fontsize(8)
        buttons.append(button)
    run_button, stop_button, reset_button = buttons
    stop_requested = {"value": False}

    def show_run(_event, selected_run: int) -> None:
        message = load_raw_run_message(RAW_DIR / f"S001R{selected_run:02d}.edf", selected_run, "preview", 1)
        signal = message.signal_volts[:14, :int(4 * message.sampling_frequency_hz)]
        _plot_signal(eeg_ax, signal, message.sampling_frequency_hz, f"Recorded PhysioNet run {selected_run} — first 4 seconds")
        status.set_text(f"Previewing recorded run {selected_run}; the Pi performs real classification when connected.")
        warning.set_text(f"Start the Mac server with --session-id demo-{selected_run} --runs {selected_run}; connect the Pi with the same session ID.")
        fig.canvas.draw_idle()

    fig.text(0.09, 0.112, "RECORDED SIGNAL PREVIEW  ·  Pi performs real inference", ha="left", color=MUTED, fontsize=8)
    for run, x in zip((6, 10, 14), (0.68, 0.77, 0.86)):
        button_ax = fig.add_axes([x, 0.101, 0.065, 0.032])
        button = Button(button_ax, f"RUN {run}", color="#c8c1b2", hovercolor="#d6cfbf")
        button.label.set_color(INK)
        button.label.set_fontsize(7)
        button.on_clicked(lambda event, selected=run: show_run(event, selected))

    def send_viewer(message: dict) -> None:
        try:
            viewer_connection.sendall((json.dumps(message) + "\n").encode("utf-8"))
        except OSError:
            status.set_text("3D simulation disconnected; close and restart the demo.")
            status.set_color(ACCENT)

    def run_sequence(_event) -> None:
        try:
            actions = parse_sequence(sequence_box.text)
        except ValueError as exc:
            status.set_text(str(exc))
            status.set_color(ACCENT)
            return
        run_button.label.set_text("Running…")
        stop_requested["value"] = False
        fig.canvas.draw_idle()
        for index, requested in enumerate(actions):
            if stop_requested["value"]:
                break
            signal = synthesize_epoch(requested, seed=42 + index)
            predicted, _scores = predict_synthetic(model, signal)
            send_viewer({"type": "action", "action": predicted})
            status.set_color("#52675e" if predicted == requested else ACCENT)
            completed = _animate_signal(
                eeg_ax, signal, SFREQ,
                f"Synthetic EEG • step {index + 1}/{len(actions)} • interpreted: {predicted}",
                status, stop_requested,
            )
            if not completed:
                break
        stop_requested["value"] = False
        run_button.label.set_text("Run sequence")
        fig.canvas.draw_idle()

    def stop(_event) -> None:
        stop_requested["value"] = True
        send_viewer({"type": "stop"})
        status.set_text("Stopped in HOLD.")
        status.set_color("#89784f")
        fig.canvas.draw_idle()

    def reset(_event) -> None:
        send_viewer({"type": "reset"})
        fig.canvas.draw_idle()

    run_button.on_clicked(run_sequence)
    stop_button.on_clicked(stop)
    reset_button.on_clicked(reset)
    fig.add_artist(plt.Line2D([0.075, 0.925], [0.075, 0.075], transform=fig.transFigure, color=RULE, lw=0.8))
    fig.text(0.075, 0.045, "ACTIONS  /  APPROACH_A · LOWER · GRASP · LIFT · MOVE_B · RELEASE · RETRACT · HOME · HOLD", ha="left", color=MUTED, fontsize=7)
    fig.text(0.925, 0.045, "REPEAT WITH *N  ·  MAX 24", ha="right", color=MUTED, fontsize=7)
    print("EEG sequence window and 3D MuJoCo viewer are open. Close either window to exit.")
    try:
        plt.show()
    finally:
        try:
            send_viewer({"type": "quit"})
            viewer_connection.close()
            viewer_process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            viewer_process.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
