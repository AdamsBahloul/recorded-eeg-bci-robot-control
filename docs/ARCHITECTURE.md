# Final architecture — recorded-data distributed BCI

## Data path

```text
MacBook: existing PhysioNet recording
  -> recorded EEG signal message (samples + timing/channel metadata)
  -> Raspberry Pi 3 B+: validate signal input
  -> deployment preprocessing -> CSP features -> LDA
  -> confidence + temporal stability + Pi safety state machine
  -> versioned HANDS / FEET / HOLD command
  -> MacBook: independent command validation
  -> MuJoCo robotic-arm simulation
```

All EEG is recorded PhysioNet data from participant 1, runs 6, 10, and 14. The system has no live EEG acquisition path. Mac-to-Pi messages must carry signal values, never just Mac-computed predictions or labels.

## Responsibility boundaries

### MacBook Pro M4

- Stores the existing raw recordings and M1 processed artifacts.
- Replays and transfers recorded EEG signal data with schema, source, run, epoch, sample rate, channel order, dimensions, and timestamp metadata.
- Trains/packages versioned deployment models; retains separate run-held-out M2 evaluation artifacts.
- Hosts the robotic-arm simulation and a receiver that validates Pi commands independently before applying bounded actions.
- Stops or holds the simulation on invalid commands, stale messages, or Pi communication loss.

### Raspberry Pi 3 Model B+

- Receives and validates actual EEG sample arrays from the Mac.
- Performs the documented deployment preprocessing and CSP+LDA inference locally.
- Computes/validates the documented decision margin, stability, timeouts, command rate, sequence, and safety state.
- Emits only versioned HANDS, FEET, HOLD commands. Default state and startup state are HOLD.
- Reports health and communicates failures. The Pi's OS must be inspected before device-specific setup.

## Model boundaries

The M2 evaluation model uses shuffled three-fold StratifiedGroupKFold grouped by run, seed 42, CSP(4)+LDA(SVD). Each epoch is predicted only by a model that excluded its entire recording run. These OOF models and predictions are for evaluation/replay validation, not a model trained on all samples.

The deployment package is distinct, versioned, and contains the fitted preprocessing/classification parameters and metadata. For a replay that must preserve held-out evaluation, select the model that excluded the replayed run. Any model fitted on all permitted data is for demonstration inference only and must not be scored against those training epochs as an unseen test.

The original M1 files and artifacts are immutable inputs to later milestones. Derived packages and logs live in separate M2/M3/deployment output paths and stay out of Git.

## Communication and validation

Define versioned, bounded schemas for EEG input, command output, heartbeat/status, and acknowledgements. Include unique IDs/sequences and timestamps. Reject malformed, duplicate, out-of-order, stale, wrong-source, or wrong-version messages. Keep transport behind interfaces; choose a lightweight transport only after checking the Pi's OS and network environment. Both the Pi and Mac simulation receiver enforce safety independently.

## Runtime and scope exclusions

Python 3.11 is the current development environment. The Mac simulation uses MuJoCo after compatibility is checked. The Pi runtime should use only dependencies supported by its inspected OS and architecture. No ROS 2, dashboard, web API, live EEG API, physical hardware path, or additional dataset is part of the final architecture.
