# NeuroBot starter kit

NeuroBot is an educational BCI-to-robot-simulation project. The first deliverable is an offline EEG processing and classification pipeline using public motor-imagery EEG data. Later milestones add deterministic replay, a MuJoCo robotic arm, an embedded Raspberry Pi gateway, and a dashboard.

## Start here
1. Open this folder in Codex.
2. Ensure Codex reads `AGENTS.md` first.
3. Paste the start prompt from `CODEX_START_PROMPT.md`.
4. Review and approve the plan before allowing installations or downloads.

## Hardware reality
- MacBook Pro M4, 16 GB: primary development and simulation computer.
- Raspberry Pi: later embedded gateway; model and OS are not yet specified.
- No EEG headset or physical robot hardware: use public recordings and simulation only.
- The initial EEG replay is not live brain acquisition.

## First data source
PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB), accessed with MNE. Begin with participant 1 and runs 6, 10, 14 only. These runs are commonly used for imagined left/right fist tasks; the implementation must inspect and verify annotation labels before assigning classes.

## Setup (after Codex reviews and user approves)
On macOS, use Python 3.11 in a virtual environment. Run:

```bash
./scripts/setup_macos.sh
```

The setup script installs only the core Python stack and development tools. It does not install ROS 2, download the dataset, or install the dashboard stack.

To download the initial small EEG subset after setup:

```bash
source .venv/bin/activate
python scripts/download_eegbci.py --subjects 1 --runs 6 10 14
```

The dataset is downloaded into `data/raw/` and is excluded from Git. Do not use `--all-subjects` unless explicitly needed and disk space has been checked.

## Project phases
See `docs/ROADMAP.md`. Do not skip the baseline/evaluation phase or treat the first model as a reliable BCI.
