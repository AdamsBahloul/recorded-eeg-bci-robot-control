#!/usr/bin/env python3
"""Download a deliberately small EEGBCI subset through MNE.

Example:
    python scripts/download_eegbci.py --subjects 1 --runs 6 10 14
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=[1],
        help="Participant IDs. Defaults to participant 1 only.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        default=[6, 10, 14],
        help="EEGBCI run IDs. Defaults to imagined left/right fist runs.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Destination directory (default: data/raw).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(s < 1 or s > 109 for s in args.subjects):
        raise SystemExit("EEGBCI participant IDs must be between 1 and 109.")
    if not args.runs or any(r < 1 or r > 14 for r in args.runs):
        raise SystemExit("Provide valid EEGBCI run IDs (1–14).")

    try:
        import mne
    except ImportError as exc:
        raise SystemExit(
            "MNE is not installed. Activate .venv and install requirements-core.txt."
        ) from exc

    destination = args.data_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    print(f"Downloading subjects={args.subjects}, runs={args.runs}")
    print(f"Destination: {destination}")
    print("This downloads only the explicitly selected recordings.")

    downloaded = mne.datasets.eegbci.load_data(
        subjects=args.subjects,
        runs=args.runs,
        path=str(destination),
        update_path=False,
    )
    print("\nDownloaded files:")
    for item in downloaded:
        print(item)


if __name__ == "__main__":
    main()
