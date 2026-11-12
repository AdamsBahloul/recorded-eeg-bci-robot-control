# Project brief

## One-sentence definition
NeuroBot is a reproducible educational brain-computer-interface prototype that classifies motor-imagery patterns in public EEG recordings and converts the resulting predictions into safe commands for a simulated robotic arm.

## Problem and motivation
BCI systems can translate measurable brain-signal patterns into external control signals. Motor imagery (imagining a movement without executing it) is a common research paradigm. NeuroBot demonstrates the software pipeline from EEG data to a robot simulation, while making limitations and uncertainty visible.

## Demonstration target
A user starts a recorded EEG replay. NeuroBot processes windows of EEG, predicts one of a small number of motor-imagery classes, applies a conservative safety gate, and updates a simulated 6-DOF robotic arm. A dashboard may later show signal traces, predictions, confidence, robot state, and latency.

## Initial command mapping
The initial research classifier should distinguish only the classes supported by the selected dataset and evaluation protocol (initially imagined left vs imagined right fist). The command mapping is configurable. For example:
- left imagery -> MOVE_LEFT
- right imagery -> MOVE_RIGHT
- uncertain / rest / invalid -> HOLD

Do not map classes to forward/backward/grip/release unless the dataset and model support those classes and a separate evaluation demonstrates them.

## Scope
### In scope
- Public EEG dataset acquisition and metadata tracking.
- EEG preprocessing and epoching.
- CSP + LDA baseline.
- EEGNet as a later comparison.
- Subject/session-aware evaluation.
- Offline deterministic replay.
- Safe command schema and validation.
- MuJoCo simulation.
- Later optional Raspberry Pi command gateway.
- Later optional dashboard.

### Out of scope initially
- Medical diagnosis, therapy, or clinical claims.
- Reading arbitrary thoughts.
- Real-time acquisition from a human without EEG hardware.
- Physical robot/actuator control.
- Training a foundation model or using cloud AI for EEG inference.
- Complex deployment infrastructure.

## Success criteria
1. A fresh clone can be set up using documented commands.
2. A small, documented EEG subset can be downloaded reproducibly.
3. Data provenance and event labels are inspected and recorded.
4. Preprocessing is deterministic and unit-tested.
5. Baseline evaluation avoids subject/session leakage.
6. Model metrics are reported honestly with limitations.
7. Invalid or uncertain predictions result in HOLD.
8. Simulation can be run without physical hardware.
9. Mac M4 16 GB constraints are respected.
