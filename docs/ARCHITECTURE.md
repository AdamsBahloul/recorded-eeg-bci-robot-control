# Architecture

## High-level flow
Public EEG recording
  -> dataset loader
  -> preprocessing / epoching
  -> baseline classifier
  -> offline/replay inference
  -> validated command contract
  -> safety gate
  -> MuJoCo simulation
  -> (later) dashboard

Later embedded path:
BCI host on Mac -> transport adapter -> Raspberry Pi command gateway -> simulation or hardware abstraction.
No physical actuators are in the current scope.

## Runtime boundaries
- `bci/`: dataset handling, preprocessing, feature extraction, training, evaluation, inference.
- `robotics/`: MuJoCo model, simulator adapter, command-to-motion mapping, safety limits.
- `embedded/`: later Raspberry Pi receiver and service management.
- `dashboard/`: later frontend and API.
- `config/`: versioned non-secret configuration.
- `data/`: downloaded source data and derived artifacts; never commit.
- `models/`: trained weights and metadata; never commit large weights by default.
- `reports/`: generated metrics and plots; commit only intentionally selected small reports.
- `tests/`: unit and integration tests.

## Recommended initial execution
Keep the initial pipeline in one Python environment on macOS. Use MuJoCo's native macOS support for simulation. Do not introduce ROS 2 until a standalone BCI-to-MuJoCo vertical slice works.

## ROS 2 decision
ROS 2 is a later integration, not a prerequisite. The preferred path is a supported Ubuntu 24.04 ARM64 environment, subject to the exact Raspberry Pi model and OS. Do not assume ROS 2 Jazzy binaries are natively supported on macOS. Verify official compatibility before installation.

## Communication design
Define and test a transport-independent message model before selecting MQTT or ROS 2. MQTT can be a simple later Mac-to-Pi transport. ROS 2 can be added for robot middleware if justified. Avoid running both until there is a clear reason.

## Command message
Example (illustrative schema only):
```json
{
  "command": "MOVE_LEFT",
  "confidence": 0.82,
  "timestamp": "2026-11-12T12:00:00Z",
  "source": "REPLAY",
  "model_version": "csp-lda-v0",
  "sequence": 42,
  "valid_for_ms": 500
}
```
Confidence is a model score unless probability calibration is explicitly evaluated. `source` must be `REPLAY` for the current data path.

## Mac M4 / 16 GB guidance
- Start with one participant and only needed runs.
- Do not preload all participants or all EDF files.
- Use small batches and close unused applications during training.
- Start with CSP + LDA; add EEGNet only after baseline correctness.
- Prefer native macOS MuJoCo. Avoid a heavy VM unless ROS 2 is needed.
- Keep large datasets and model weights outside Git.
