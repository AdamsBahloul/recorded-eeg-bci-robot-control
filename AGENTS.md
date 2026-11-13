# NeuroBot — Codex Operating Instructions

## Mission
Build NeuroBot as a reproducible, recorded-data-only motor-imagery system. The MacBook stores and replays the existing PhysioNet EEG signal to a Raspberry Pi 3 B+. The Pi preprocesses EEG, runs CSP+LDA classification and safety logic, and sends validated commands to the MacBook's MuJoCo simulation. There is no live EEG acquisition.

## User hardware and constraints
- Development computer: Apple MacBook Pro M4, 16 GB unified memory, macOS / Apple Silicon.
- Embedded controller: Raspberry Pi 3 Model B+; inspect the installed OS when device access is available.
- No EEG headset, robotic arm, sensors, motors, or other electronics are currently available.
- Dataset and simulation run on the Mac; final signal processing, classification, and command safety run on the Pi 3 B+.
- No physical EEG or robot hardware is used. Do not add live acquisition or actuator control.
- Use only participant 1, runs 6, 10, and 14; do not download other participants or runs.

## Required project principles
1. Build in small, verifiable milestones. Never generate the whole application in one pass.
2. Before editing, inspect the repository and read this file plus relevant `docs/` files.
3. Prefer the smallest implementation that proves the current milestone.
4. Keep modules independently testable; use typed interfaces and clear names.
5. Do not invent experimental results, dataset contents, hardware capabilities, or benchmark numbers.
6. Separate training, offline evaluation, replay inference, simulation, and embedded deployment.
7. The only EEG source is the existing recorded dataset. Label all input and operation `REPLAY`; do not add live acquisition infrastructure.
8. Do not use data from test subjects/sessions during training or preprocessing fit. Prevent subject/session leakage.
9. Record random seeds, dataset version, participant IDs, run IDs, preprocessing parameters, model version, and metrics.
10. Make all data paths configurable. Do not commit datasets, model weights, caches, secrets, or generated logs.
11. Keep the Mac's 16 GB memory and Pi 3 B+ resource limits in mind. Use the existing single-participant dataset and bounded signal messages.
12. Keep BCI processing and safety on the Pi and simulation on the Mac. Do not introduce ROS 2.
13. Do not add Kubernetes, databases, cloud services, agent frameworks, or complex infrastructure without a demonstrated need.
14. Ask before destructive operations, broad downloads, paid services, or changes outside this repository.

## Safety and scientific validity
- This is a non-clinical educational/research prototype, not a medical device.
- No human subject data collection is planned initially.
- Never infer a person's thoughts or intentions beyond the limited labels in the dataset.
- The initial classifier is experimental and may perform poorly on unseen people.
- Robot control must be simulation-only until separately reviewed.
- Use a `HOLD` / no-motion state by default.
- Require confidence threshold, stable predictions across multiple consecutive windows, command timeout, rate limits, joint limits, and an explicit simulation enable flag before movement.
- Low confidence, stale data, invalid commands, communication loss, or model errors must produce `HOLD`.
- Do not implement physical actuator control; Pi commands target only the Mac simulation.
- Never describe confidence as calibrated probability unless calibration is actually implemented and evaluated.

## Architecture and technology choices
- Python 3.11 recommended for the initial macOS environment.
- EEG: MNE-Python, NumPy, SciPy, scikit-learn.
- Baseline classifier: CSP + LDA.
- Model: CSP+LDA; do not add deep learning.
- Dataset: PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB), participant 1, runs 6, 10, and 14 (imagined hands versus feet). The event mapping was verified in M1; preserve the metadata and do not silently remap labels.
- Replay: deterministic transfer of the existing recorded EEG samples/epochs from Mac to Pi. No live EEG, LSL, or acquisition driver.
- Simulation: MuJoCo Python package, native on Apple Silicon.
- Embedded: Raspberry Pi 3 B+ performs signal preprocessing, CSP+LDA inference, confidence/stability validation, and safety state transitions.
- Communication: transfer EEG signal data Mac-to-Pi and validated commands Pi-to-Mac over a documented schema and lightweight transport. Do not couple classification to the transport.

## Required command contract
Commands must be one of: `HOLD`, `HANDS`, `FEET`.
Every command message must include:
- command ID and sequence number
- command type and UTC timestamp
- source epoch ID, predicted class, raw decision score, confidence measure and its type
- model version, safety state, validation status, and schema version

Reject unknown commands, invalid or stale messages, duplicate/out-of-order sequences, and unexpected sources. `HOLD` is always safe and is the fallback.

## Coding standards
- Add tests for data transformations, command validation, safety gates, and interfaces.
- Use `pytest`; use `ruff` for linting/formatting if configured.
- Use docstrings for public functions and comments for non-obvious scientific decisions.
- Avoid giant files and hidden global state.
- Keep configuration in `config/` and paths in environment variables or CLI arguments.
- Log metadata and errors; do not log sensitive personal data.
- Pin dependencies only after a working environment has been tested. Do not invent exact version pins.

## Codex workflow
For each task:
1. State the milestone and acceptance criteria in 3–6 bullets.
2. Inspect relevant files and current git status.
3. Propose a short plan before broad changes.
4. Implement only that milestone.
5. Run focused tests and report exact commands/results.
6. List changed files and any assumptions.
7. Stop and ask before moving to the next milestone.

## First task
The project is past M0/M1. For each new task, inspect the existing implementation and relevant context files, propose a milestone-scoped plan before broad changes, implement only approved scope, run focused tests, and stop at the milestone gate. The final architecture is recorded-data-only: Mac replay sends EEG signal data to the Pi; the Pi preprocesses/classifies and validates; the Mac simulation accepts validated commands from the Pi.
