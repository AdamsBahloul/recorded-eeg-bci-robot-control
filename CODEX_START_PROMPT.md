# NeuroBot project context

Read `AGENTS.md`, `README.md`, and the relevant files in `docs/` before making changes. Inspect the existing implementation and Git status first. Treat the current project context as authoritative; this file is only a short starting pointer.

## Definitive architecture

- Use only recorded PhysioNet EEG from participant 1, runs 6, 10, and 14. No live EEG acquisition or acquisition infrastructure.
- The MacBook Pro M4 stores/replays EEG signal data and hosts the robotic-arm simulation.
- The Raspberry Pi 3 B+ receives actual EEG signal values, performs preprocessing, CSP+LDA inference, confidence/stability validation, and safety checks, then emits HANDS, FEET, or HOLD commands.
- The Mac validates Pi commands independently and controls the simulation. No physical actuators.
- Preserve completed M1/M2/M3 implementations, methodology, and artifacts. Keep M2 OOF evaluation separate from deployment models.
- Keep HOLD by default; do not assume or reuse a decision-margin threshold.
- Work one milestone at a time. Do not add other datasets, participants, live acquisition, EEGNet, ROS 2, dashboard/API, or unrelated infrastructure.

## Workflow

1. Inspect relevant code, tests, configuration, dependency locks, and docs.
2. Identify technical risks and present a concise milestone-scoped plan before broad implementation.
3. Install only dependencies required for the approved milestone. Inspect the actual Pi OS before device-specific setup.
4. Implement incrementally, preserve earlier milestones, and test each module.
5. Report measured results, errors, limitations, and changed files. Stop at the milestone gate; do not proceed without approval.
