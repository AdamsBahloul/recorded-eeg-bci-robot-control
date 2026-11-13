#!/usr/bin/env python3
"""Measure preprocessing/inference timing and process resources on this host."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/cache/matplotlib"))
sys.path.insert(0, str(ROOT))

from bci.replay_source import load_raw_run_message
from embedded.inference import PiInference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "data/processed/deployment/csp_lda.joblib")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/deployment/manifest.json")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/MNE-eegbci-data/files/eegmmidb/1.0.0/S001")
    parser.add_argument("--runs", nargs="+", type=int, default=[6, 10, 14])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or any(run not in (6, 10, 14) for run in args.runs):
        parser.error("repeats must be positive and runs must be selected from 6, 10, 14")
    load_start = time.perf_counter()
    inference = PiInference(args.model, args.manifest)
    model_load = time.perf_counter() - load_start
    user_cpu_start = time.process_time()
    wall_start = time.perf_counter()
    runs = []
    for repeat in range(args.repeats):
        for index, run_id in enumerate(args.runs):
            message = load_raw_run_message(
                args.raw_dir / f"S001R{run_id:02d}.edf", run_id,
                session_id=f"benchmark-{repeat + 1}", sequence=index + 1,
            )
            predictions, timing = inference.process(message)
            timing["repeat"] = repeat + 1
            timing["run_id"] = run_id
            timing["prediction_count"] = len(predictions)
            timing["preprocessing_per_epoch_seconds"] = timing["preprocessing_seconds"] / len(predictions)
            timing["feature_extraction_per_epoch_seconds"] = timing["feature_extraction_seconds"] / len(predictions)
            timing["classification_per_epoch_seconds"] = timing["classification_seconds"] / len(predictions)
            runs.append(timing)
    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - user_cpu_start
    rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = int(rss_raw if sys.platform == "darwin" else rss_raw * 1024)
    device_model = "unknown"
    model_path = Path("/proc/device-tree/model")
    if model_path.is_file():
        device_model = model_path.read_bytes().replace(b"\0", b"").decode("utf-8", errors="replace")
    physical_pi = "Raspberry Pi 3 Model B Plus" in device_model
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "measurement_host": platform.node(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "device_model": device_model,
        "hardware_verified": physical_pi,
        "model_version": inference.manifest["model_version"],
        "model_loading_seconds": model_load,
        "wall_inference_loop_seconds": wall_seconds,
        "process_cpu_seconds": cpu_seconds,
        "process_cpu_percent_of_one_core": 100 * cpu_seconds / wall_seconds if wall_seconds else 0,
        "peak_process_rss_bytes": peak_rss_bytes,
        "runs": runs,
        "note": "Times are measured on the current host. They are Pi measurements only when run on the physical Raspberry Pi.",
    }
    output = args.output or ROOT / "data/processed/deployment" / f"benchmark_{platform.node()}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"Run {row['run_id']}" for row in runs]
    index = range(len(runs))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar([i - width for i in index], [row["preprocessing_per_epoch_seconds"] * 1000 for row in runs],
           width, label="Preprocessing")
    ax.bar(list(index), [row["feature_extraction_per_epoch_seconds"] * 1000 for row in runs],
           width, label="CSP transform")
    ax.bar([i + width for i in index], [row["classification_per_epoch_seconds"] * 1000 for row in runs],
           width, label="LDA")
    ax.set_xticks(list(index), labels)
    ax.set_ylabel("Observed time per epoch (ms)")
    ax.set_title(f"NeuroBot inference benchmark — {platform.node()} ({platform.machine()})")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    figure_path = output.with_suffix(".png")
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    report["benchmark_plot"] = figure_path.name
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved benchmark: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
