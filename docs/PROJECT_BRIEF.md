# Project brief

## Objective

NeuroBot is an educational, recorded-data-only BCI prototype. It transfers EEG signal data from the existing PhysioNet participant 1 recordings to a Raspberry Pi 3 Model B+. The Pi preprocesses the signal, runs CSP+LDA, applies confidence, temporal stability, and safety checks, then sends HANDS, FEET, or HOLD commands to a robotic-arm simulation running on the MacBook Pro M4.

There is no live EEG acquisition, EEG headset, physical robot, or actuator control in this project.

## Fixed data scope

- PhysioNet EEG Motor Movement/Imagery Database, participant 1, runs 6, 10, and 14 only.
- The task is imagined hands versus imagined feet.
- The verified M1 artifacts contain 45 epochs: 21 hands and 24 feet, with 64 channels and 481 samples per epoch at 160 Hz.
- Do not acquire additional participants, runs, or datasets.

## System responsibilities

**MacBook Pro M4:** store the existing dataset, prepare and transfer recorded EEG signal input, train/package the CSP+LDA deployment models, host the MuJoCo simulation, and manage the demonstration.

**Raspberry Pi 3 Model B+:** receive actual EEG signal values, perform the deployment preprocessing and CSP+LDA inference, apply confidence/stability/safety logic, and generate validated commands. The Pi must not receive Mac-computed labels as its inference input.

**Communication:** carry documented EEG signal messages Mac-to-Pi and validated command messages Pi-to-Mac. Both sides validate messages. Loss, corruption, stale input, or invalid commands produce a safe HOLD/stop response.

## Evaluation and safety

The M2 evaluation remains leakage-safe, run-held-out cross-validation. Its out-of-fold predictions are evaluation outputs, not a deployment model trained for unseen data. The deployment package must remain distinct and versioned. HANDS/FEET commands stay disabled by default until a decision-margin threshold is explicitly configured and validated. LDA scores are not calibrated probabilities.

The final demonstration must prove that the Pi receives EEG signal data and performs its own preprocessing/inference before its commands affect the Mac simulation. Report the Pi's CPU, memory, startup, inference latency, and sustained-operation results only when measured on the actual device. If hardware is unavailable, mark device tests not run.

## In scope

- Reproducible M1 preprocessing and M2 CSP+LDA evaluation.
- Pi deployment of preprocessing, classification, and safety logic.
- Mac-to-Pi recorded EEG signal transfer and Pi-to-Mac command communication.
- Mac-hosted robotic-arm simulation and validated command receiver.
- Independent judge/reference checks, unit/integration/system/fault-injection tests, and measured benchmarks.
- Reproducible documentation and an end-to-end final demonstration.

## Out of scope

- Live EEG acquisition, headsets, or acquisition infrastructure.
- Additional participants, recordings, or datasets.
- EEGNet/deep learning, ROS 2, MQTT broker infrastructure unless specifically justified for the chosen transport, dashboard/API, cloud services, and physical actuators.
- Clinical use or claims, arbitrary thought decoding, and real-world BCI performance claims.
