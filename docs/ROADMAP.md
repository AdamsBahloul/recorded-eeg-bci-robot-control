# Roadmap and milestone gates

## M0 — Repository and environment plan
- Inspect Mac/Python compatibility.
- Confirm Python version and disk space.
- Review architecture and dependencies.
Gate: user approves setup plan.

## M1 — Dataset and preprocessing
- Install core dependencies.
- Download participant 1, runs 6/10/14 only.
- Inspect raw metadata and annotations/events.
- Implement loading, filtering, epoching and plotting.
- Add tests for channel selection, event handling, shapes and deterministic preprocessing.
Gate: reproducible epochs with verified labels; no classifier claims yet.

## M2 — Classical baseline
- Implement CSP + LDA.
- Define a defensible split (session/run-aware or subject-wise).
- Avoid fitting scalers/CSP on held-out data.
- Report balanced accuracy, accuracy, macro-F1, confusion matrix and split metadata.
Gate: evaluation script runs end-to-end and documents limitations.

## M3 — EEGNet comparison
- Implement compact EEGNet only after M2.
- Keep training reproducible and resource-bounded.
- Compare against the baseline with the same split.
Gate: model comparison report with honest metrics.

## M4 — Replay inference and command contract
- Implement deterministic chunk/window replay.
- Add command message schema, validation and safety gate.
- Add stale-message timeout, stable-prediction requirement and HOLD fallback.
Gate: invalid, stale, low-confidence and missing predictions always produce HOLD.

## M5 — MuJoCo arm simulation
- Load or create a documented 6-DOF arm model.
- Implement bounded joint commands and smooth trajectories.
- Connect only validated commands.
- Keep simulation enable flag off by default.
Gate: simulation-only movement works and safety tests pass.

## M6 — Raspberry Pi gateway
- First identify Pi model, RAM and OS.
- Decide Raspberry Pi OS vs Ubuntu based on requirements.
- Implement network receiver and command validator.
- Add watchdog, service startup and disconnect fallback.
- Do not add physical actuator drivers.
Gate: Pi can receive and validate commands; disconnected state is HOLD.

## M7 — Dashboard
- FastAPI + WebSockets backend.
- React + TypeScript frontend.
- Signal/replay status, predictions, robot state, latency, logs.
Gate: UI labels REPLAY vs LIVE accurately and exposes safety state.

## M8 — Research packaging
- Reproducible scripts/configs.
- Dataset provenance and licensing notes.
- Model card, limitations, test report and demo video.
- Architecture diagram and technical report.
Gate: another developer can reproduce the demonstration.

## Do not parallelize early
Complete M1 before classifier implementation, M2 before EEGNet, M4 before robot integration, and M5 before Pi integration. This reduces debugging complexity and token/context waste.
