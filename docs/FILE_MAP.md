# Project file map

## Start here

- `README.md` — project summary, scope, and links to the guides.
- `docs/GETTING_STARTED.md` — walkthrough from the interactive synthetic demo through the Mac/Pi recorded EEG cycle.
- `docs/PUBLISHING.md` — checks to complete before publishing on GitHub.
- `AGENTS.md` — project constraints and development rules.

For project changes, read context in this order: `AGENTS.md`, `README.md`, then the relevant files in `docs/` (`PROJECT_BRIEF.md`, `ARCHITECTURE.md`, `ROADMAP.md`, `DATA_AND_EVALUATION.md`, `SAFETY.md`, `TECH_DECISIONS.md`, and `M3_RUNTIME.md`). `CODEX_START_PROMPT.md` is the initial Codex task prompt; it is not a user run guide.

## Source packages

- `bci/` — EEG preprocessing, CSP+LDA classification, replay input, and replay controls.
- `embedded/` — Pi-side EEG message validation, preprocessing/inference, and controller.
- `runtime/` — prediction confidence, temporal stability, safety state machine, command generation, and replay logic.
- `transport/` — bounded EEG chunks, message framing, and protocol.
- `mac/` — independent command validation, logging, and replay session handling.
- `simulation/` — MuJoCo arm model and bounded simulation controller.
- `demo/` — artificial EEG signal generator, illustrative classifier, and synthetic arm task.
- `validation/` — independent signal, classifier, and command reference checks.

## Operations and configuration

- `scripts/` — environment checks, data download/preprocessing, M2 evaluation, model packaging, Pi/Mac launchers, benchmarks, and demo launchers.
- `config/` — dataset/project, M2/M3, Mac, and Pi TOML/YAML settings. The unvalidated simulation demo profile is opt-in.
- `pi/` — Pi device inspection, installer, pinned dependencies, and device-specific notes.
- `requirements-m1*.txt`, `requirements-m2*.txt`, `requirements-simulation*.txt` — staged dependency sets and their recorded locks.
- `requirements-core.txt`, `requirements-ml.txt`, `requirements-api.txt` — retained aggregate/deferred dependency lists for reference. They are not used by the documented M1–M3 workflow; install the staged sets above instead.

## Tests, docs, and local data

- `tests/` — unit, integration, protocol, replay, safety, and simulation tests.
- `docs/` — architecture, data/evaluation, roadmap, safety, technical decisions, runtime, and user run guides.
- `data/raw/` — local approved PhysioNet EDF files (ignored by Git).
- `data/processed/` — local preprocessing/evaluation/deployment outputs and run logs (ignored by Git).
- `data/cache/` — local plotting and library caches (ignored by Git).

Do not commit dataset files, derived EEG, fitted models, deployment bundles, caches, or generated logs. For the data flow and module boundaries, see `docs/ARCHITECTURE.md`; for the allowed data and evaluation claims, see `docs/DATA_AND_EVALUATION.md`.
