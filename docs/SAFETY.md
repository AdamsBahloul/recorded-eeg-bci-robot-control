# Safety, ethics and limitations

## Project status and boundaries

NeuroBot is a non-clinical educational/research prototype. It is not for diagnosis, treatment, rehabilitation, or safety-critical operation. Input is limited to recorded public PhysioNet EEG from participant 1, runs 6, 10, and 14. There is no live EEG acquisition or physical actuator control.

The Pi 3 B+ runs preprocessing, CSP+LDA inference, confidence/stability checks, and a safety state machine. The Mac hosts the robotic simulation and independently validates every Pi command. Simulated replay is not live acquisition or proof of controlling a user's brain.

## Fail-safe behavior

Default and startup state is `HOLD`. Low or invalid confidence, unstable predictions, stale or malformed EEG, duplicate/out-of-order messages, source disconnect, model errors, command timeout, simulation receiver errors, and shutdown must stop movement and return to HOLD. Internal failures enter ERROR and require explicit recovery through HOLD. The Pi and Mac receiver each enforce their own safety checks.

HANDS/FEET transitions stay disabled until an explicit decision-margin threshold is configured and validated. LDA decision scores are not calibrated probabilities. Never select a threshold to improve the existing 45-epoch evaluation result.

The Mac simulation must bound joint positions, velocities, acceleration, and trajectories. Invalid or rejected commands must not move the arm. No network message may directly control a physical actuator.

## Data ethics and claims

Use only the documented public dataset subset and honor its terms. Do not upload EEG to third parties by default. Do not infer arbitrary thoughts or sensitive personal traits. One participant and three runs cannot establish cross-participant generalization or real-world BCI performance. Report replay, hardware, and simulation measurements separately; do not fabricate unavailable Pi hardware tests.
