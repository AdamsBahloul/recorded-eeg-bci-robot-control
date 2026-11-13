#!/usr/bin/env python3
"""Run independent M2 artifact judge and write machine-readable report."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/cache/matplotlib"))
sys.path.insert(0, str(ROOT))

from validation.classifier_judge import judge_m2
from validation.signal_judge import judge_signal_run


def record(report, name, operation) -> None:
    try:
        report["judges"].append(operation())
    except Exception as exc:
        report["judges"].append({"judge": name, "status": "FAIL", "failure": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    report = {
        "project": "NeuroBot",
        "scope": "recorded EEG only; participant 1 runs 6, 10, 14",
        "judges": [],
    }
    record(report, "independent_m2_csp_lda_recomputation", lambda: judge_m2(
            ROOT / "data/processed/m1/S001_epochs.npz",
            ROOT / "data/processed/m1/S001_metadata.json",
            ROOT / "data/processed/m2/out_of_fold_predictions.npz",
            ROOT / "data/processed/m2/metrics.json",
        ))
    raw = ROOT / "data/raw/MNE-eegbci-data/files/eegmmidb/1.0.0/S001"
    for run_id in (6, 10, 14):
        record(report, f"independent_raw_signal_run_{run_id}", lambda run=run_id: judge_signal_run(
            raw / f"S001R{run:02d}.edf", run,
        ))
    report["overall_status"] = "PASS" if report["judges"] and all(item["status"] == "PASS" for item in report["judges"]) else "FAIL"
    output = ROOT / "data/processed/validation/judge_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {output}")
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
