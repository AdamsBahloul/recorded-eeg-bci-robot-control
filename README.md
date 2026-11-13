# NeuroBot

NeuroBot is an educational, recorded-data BCI prototype. A MacBook replays EEG from the PhysioNet Motor Movement/Imagery dataset to a Raspberry Pi 3 B+. The Pi preprocesses the signal, runs CSP+LDA, applies safety checks, and sends HANDS, FEET, or HOLD commands back to a MuJoCo arm simulation on the Mac.

There is no live EEG acquisition or physical robot control. The dataset is restricted to participant 1, runs 6, 10, and 14 (imagined hands versus feet).

## Choose a starting point

- **Try the interactive EEG and 3D arm demo:** follow [the run guide](docs/GETTING_STARTED.md#1-interactive-synthetic-eeg-demo). This first demo generates artificial signals and uses an illustrative model; it does not classify EEG on the Pi.
- **Run the recorded EEG Pi-to-Mac cycle:** follow [the run guide](docs/GETTING_STARTED.md#4-repeat-a-recorded-eeg-cycle-on-the-pi-and-mac). This sends recorded signal samples to the Pi for CSP+LDA inference.
- **Understand the project and its limits:** read [the architecture](docs/ARCHITECTURE.md), [data and evaluation notes](docs/DATA_AND_EVALUATION.md), and [safety limits](docs/SAFETY.md).

## Repository layout

```text
bci/          EEG preprocessing, classifier, and replay sources
embedded/     Pi signal validation, inference, and controller
mac/          Mac command validation, event logging, and replay server
runtime/      Confidence, stability, state machine, and replay logic
transport/    EEG and command message protocols
simulation/   MuJoCo arm model and simulation controller
demo/         Separate artificial EEG demonstration
config/       Project, Mac, Pi, M2, and M3 settings
scripts/      Setup, preprocessing, evaluation, packaging, and launch tools
tests/        Unit and integration checks
docs/         Architecture, safety, evaluation, and run guides
pi/           Raspberry Pi installation and device tools
data/         Local recordings and generated artifacts; excluded from Git
```

See [docs/FILE_MAP.md](docs/FILE_MAP.md) for the main entry points and module responsibilities.

## Setup and run instructions

The complete Mac and Raspberry Pi instructions, including the synthetic demo, dataset setup, Pi installation, network connection, and infinite replay cycle, are in [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

## Data, safety, and reproducibility

- Only the existing PhysioNet participant 1 recordings, runs 6, 10, and 14, are in scope.
- EEG samples travel from the Mac to the Pi; labels and Mac-side predictions are not inference inputs.
- Cross-validation is run-held-out within one participant. It is not an independent test or evidence of performance on other people.
- `HOLD` is the normal default. The unvalidated demo override is only for the simulated demonstration and is not suitable for physical hardware.
- EEG files, derived datasets, fitted model binaries, caches, and run logs are excluded from Git. Follow the dataset's terms when obtaining the recordings.
- This is a non-clinical research/education prototype, not a medical device.

## Before publishing on GitHub

Review [docs/PUBLISHING.md](docs/PUBLISHING.md) for repository hygiene and the license decision. The project has not been pushed or published by these instructions.
