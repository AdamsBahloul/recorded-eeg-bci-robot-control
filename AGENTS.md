# NeuroBot — Codex Operating Instructions

## Mission
Build NeuroBot: a reproducible, research-oriented motor-imagery EEG-to-robot simulation system. Public EEG recordings are the initial signal source. A replay engine makes offline recordings behave like a stream. A classifier maps EEG windows to conservative commands. MuJoCo simulates a 6-DOF arm. A Raspberry Pi is a later embedded command gateway. A web dashboard is a later visualization layer.

## User hardware and constraints
- Development computer: Apple MacBook Pro M4, 16 GB unified memory, macOS / Apple Silicon.
- Additional computer: Raspberry Pi; exact model and OS are not yet known.
- No EEG headset, robotic arm, sensors, motors, or other electronics are currently available.
- Initial work must run on the Mac without the Raspberry Pi or physical hardware.
- Do not assume a specific Raspberry Pi model. Keep embedded code portable and document requirements before deployment.
- Do not download the full EEG dataset by default. Start with one participant and the minimum required motor-imagery runs.

## Required project principles
1. Build in small, verifiable milestones. Never generate the whole application in one pass.
2. Before editing, inspect the repository and read this file plus relevant `docs/` files.
3. Prefer the smallest implementation that proves the current milestone.
4. Keep modules independently testable; use typed interfaces and clear names.
5. Do not invent experimental results, dataset contents, hardware capabilities, or benchmark numbers.
6. Separate training, offline evaluation, replay inference, simulation, and embedded deployment.
7. Never claim replayed EEG is live EEG acquisition. Label it `REPLAY` in the UI, logs, and reports.
8. Do not use data from test subjects/sessions during training or preprocessing fit. Prevent subject/session leakage.
9. Record random seeds, dataset version, participant IDs, run IDs, preprocessing parameters, model version, and metrics.
10. Make all data paths configurable. Do not commit datasets, model weights, caches, secrets, or generated logs.
11. Keep the Mac's 16 GB memory limit in mind. Use one subject initially, small batches, CPU/MPS only when verified, and avoid loading the whole dataset.
12. Do not install ROS 2 on macOS as the default path. First make the standalone MuJoCo path work. ROS 2 is a later optional integration, preferably on supported Ubuntu 24.04 ARM64.
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
- Do not implement physical actuator control in this project unless the user explicitly requests it and a separate safety review is completed.
- Never describe confidence as calibrated probability unless calibration is actually implemented and evaluated.

## Architecture and technology choices
- Python 3.11 recommended for the initial macOS environment.
- EEG: MNE-Python, NumPy, SciPy, scikit-learn.
- Baseline classifier: CSP + LDA.
- Deep model: EEGNet in PyTorch only after a valid baseline and evaluation pipeline exist.
- Dataset: PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB), accessed through MNE's EEGBCI helper. Initial runs: 6, 10, 14 (imagined left/right fist tasks). Verify event labels from the downloaded recordings rather than assuming them.
- Replay: deterministic offline window replay; later LSL only if a real streaming need exists.
- Simulation: MuJoCo Python package, native on Apple Silicon.
- API/dashboard: defer until core pipeline works; then FastAPI + WebSockets and React + TypeScript.
- Embedded: defer until Pi model/OS are confirmed. Initially use a small Python or C++ command receiver over MQTT only if needed. ROS 2 is optional and later.
- Communication: define a transport-neutral command schema first. Do not couple BCI logic to MQTT/ROS 2.

## Required command contract
Commands must be one of:
`HOLD`, `MOVE_LEFT`, `MOVE_RIGHT`, `MOVE_FORWARD`, `MOVE_BACKWARD`, `GRIP`, `RELEASE`.
Every command message must include:
- `command`: allowed enum value
- `confidence`: numeric 0..1, not necessarily calibrated
- `timestamp`: UTC ISO-8601 string
- `source`: `REPLAY` or `LIVE`
- `model_version`: string
- `sequence`: monotonically increasing integer
- `valid_for_ms`: positive expiry interval

Reject unknown commands, invalid confidence, stale timestamps, expired messages, invalid sequence numbers, and unexpected sources. `HOLD` is always safe and is the fallback.

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
Do not implement the complete project. Read `README.md`, `docs/PROJECT_BRIEF.md`, `docs/ARCHITECTURE.md`, `docs/ROADMAP.md`, `docs/DATA_AND_EVALUATION.md`, and `docs/SAFETY.md`. Then inspect the scaffold. Check macOS/Apple Silicon compatibility of the proposed dependencies. Create a concise setup plan and a milestone-1 plan. Do not install packages, download data, or write application code until the user approves the plan.
