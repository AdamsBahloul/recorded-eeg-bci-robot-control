# NeuroBot

### Recorded EEG -> Raspberry Pi inference -> MuJoCo arm simulation

NeuroBot is a reproducible educational BCI prototype that turns **recorded** motor-imagery EEG into safe, simulated robot commands. A Mac replays approved PhysioNet EEG to a Raspberry Pi 3 B+, where the signal is preprocessed and classified with CSP+LDA. The Pi then sends validated `HANDS`, `FEET`, or `HOLD` commands back to a MuJoCo arm simulation.

> **This is a research and education prototype, not a medical device.** It does not acquire live EEG and it does not control physical hardware.

## Research paper

Read the [NeuroBot research paper](neurobot.pdf) for the project's motivation, system design, methodology, and experimental scope.

## What makes it useful

- **End-to-end data path:** Mac replay, bounded EEG transport, Pi-side inference, command validation, and simulation.
- **Safety-first runtime:** confidence gates, temporal stability checks, timeouts, rate limits, and `HOLD` as the default fallback.
- **Reproducible scope:** PhysioNet participant 1, runs 6, 10, and 14 only.
- **Two ways to explore:** start with an interactive synthetic demo or run the recorded EEG cycle.
- **Testable architecture:** preprocessing, protocols, runtime state, safety, and simulation are covered by focused tests.

## See it in action

### 1. Start with the interactive demo

The fastest way to explore the project is the synthetic EEG and 3D arm demo:

```bash
./scripts/setup_macos.sh
.venv/bin/python -m pip install -r requirements-simulation.txt
.venv/bin/python scripts/run_synthetic_direction_demo.py
```

This demo uses generated signals and an illustrative classifier. It demonstrates the UI and arm sequence, but it does **not** evaluate the recorded-data CSP+LDA pipeline.

### 2. Run the recorded EEG pipeline

For the full Mac-to-Pi replay workflow, including approved dataset setup, model packaging, device installation, and launch commands, follow the [Getting Started guide](docs/GETTING_STARTED.md).

## Architecture

```text
PhysioNet EEG recordings
          |
          v
Mac replay server ---> bounded EEG protocol ---> Raspberry Pi 3 B+
                                                |
                         preprocessing + CSP/LDA + safety gates
                                                |
                                                v
                         validated command: HANDS / FEET / HOLD
                                                |
                                                v
                                      Mac MuJoCo arm simulation
```

| Area | Responsibility |
| --- | --- |
| `bci/` | EEG replay, preprocessing, and CSP+LDA classification |
| `embedded/` | Pi-side signal validation, inference, and controller |
| `runtime/` | Confidence, stability, state machine, and safety logic |
| `transport/` | Bounded EEG chunks, framing, and message schemas |
| `mac/` | Command validation, logging, and replay server |
| `simulation/` | MuJoCo arm model and simulation controller |
| `demo/` | Separate synthetic signal and arm demonstration |
| `tests/` | Unit and integration coverage |

See the [project file map](docs/FILE_MAP.md) for module responsibilities.

## Scientific and safety boundaries

- Only participant 1, runs 6, 10, and 14 are in scope: imagined hands versus feet.
- EEG samples are sent from the Mac to the Pi; labels and Mac-side predictions are not inference inputs.
- Evaluation is run-held-out within one participant, not evidence of performance on other people.
- `HOLD` is the normal default. The unvalidated demo override is simulation-only.
- Recordings, derived data, fitted models, caches, and logs stay local and are excluded from Git.
- Read the [architecture](docs/ARCHITECTURE.md), [data and evaluation notes](docs/DATA_AND_EVALUATION.md), and [safety limits](docs/SAFETY.md) before extending the system.

## Development

Use Python 3.11 on macOS. Run the focused test suite with:

```bash
.venv/bin/python -m pytest
```

The staged dependency files are documented in the [Getting Started guide](docs/GETTING_STARTED.md). See [docs/PUBLISHING.md](docs/PUBLISHING.md) for repository hygiene and licensing notes.
