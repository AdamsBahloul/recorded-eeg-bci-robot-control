# Roadmap and milestone gates

The final system is recorded-data-only. No milestone adds live EEG acquisition, additional datasets, participants, runs, dashboard/API, ROS 2, deep learning, or physical actuators.

## M0 — Repository and Mac environment plan (complete)
- Inspect repository and Mac compatibility.
- Gate: user approval.

## M1 — Existing data and preprocessing (complete)
- Participant 1, runs 6/10/14; verify annotations before labels.
- Deterministic 8–30 Hz preprocessing and 1–4 s epochs.
- Gate: 45 verified epochs and reproducible metadata.

## M2 — CSP+LDA baseline (complete)
- CSP(4) + LDA(SVD), stratified run-grouped three-fold CV, seed 42.
- Verify one leakage-safe OOF prediction per epoch and record metrics.
- Gate: verified confusion matrix `[[20, 1], [1, 23]]`; evaluation artifacts kept separate from deployment model.

## M3 — Offline command engine (complete)
- Deterministic replay, decision-margin validation, temporal stability, HOLD/ERROR state machine, schema, logs, plots, tests.
- Active commands disabled by default.
- Gate: safe M3 replay; no robot or transport integration.

## M4 — Pi deployment package and local inference (software complete; hardware verification pending)
- Inspect the actual Raspberry Pi 3 B+ model/OS and available resources.
- Package versioned CSP+LDA deployment model(s) separately from M2 OOF evaluation artifacts.
- Implement Pi signal validation, required preprocessing, CSP+LDA inference, and Pi-side confidence/stability/safety behavior.
- Verify numerical parity against the Mac reference with deterministic fixtures; benchmark startup, memory, CPU, inference latency, and sustained runs on the Pi.
- Gate: the Pi can process EEG signal arrays locally and produce safe commands; measurements are from actual hardware or explicitly marked not run.

## M5 — Mac-to-Pi EEG transfer and Pi-to-Mac commands (software complete; hardware verification pending)
- Define versioned signal, command, status/heartbeat, and acknowledgement schemas.
- Transfer actual recorded EEG samples and required metadata from Mac to Pi; never send precomputed class labels as the inference input.
- Add stale/duplicate/out-of-order/corrupt message handling, reconnect behavior, and HOLD/stop on loss.
- Gate: integration tests prove signal data reaches Pi inference and commands are independently validated on Mac.

## M6 — Mac-hosted robotic-arm simulation (implementation and headless tests complete; graphical viewer check pending)
- Implement a bounded MuJoCo arm, safe HANDS/FEET action mapping, HOLD/ERROR stop, and receiver-side sequence/freshness/safety validation.
- Gate: invalid, stale, duplicate, or disconnected commands do not move the simulation.

## M7 — End-to-end demonstration and test framework (software acceptance checks complete; physical demonstration pending)
- One documented, repeatable offline run: Mac replay -> Pi preprocessing/inference/safety -> Pi command -> Mac validation -> simulation acknowledgement.
- Add integration, system, fault-injection, and endurance tests; independent judge models/oracles; machine-readable validation report; measured performance benchmarks.
- Gate: demonstrate and log Pi identity, EEG epoch/sample receipt, local prediction, safety state, command, acknowledgement, and robot state. Missing hardware tests are explicitly reported as not run.

## M8 — Final packaging and demonstration (documentation prepared; physical demonstration pending)
- Reproduction guide, architecture and data provenance, model/evaluation limitations, test/judge report, benchmark report, and final demonstration recording.
- Gate: another developer can reproduce the complete recorded-data flow with the documented Mac and Pi setup.

## Work discipline

Proceed one milestone at a time, preserve completed M1/M2/M3 behavior and artifacts, and stop at each gate for review. Do not install device-specific packages until the Pi OS is inspected. Never report simulated Pi results as hardware measurements.
