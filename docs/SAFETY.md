# Safety, ethics and limitations

## Project status
NeuroBot is an educational/research software prototype. It is not a medical device and must not be used for diagnosis, treatment, rehabilitation, or safety-critical operation.

## No physical robot
The current project has no actuators. All movement is in MuJoCo simulation. Do not add physical actuator control or GPIO motor drivers without an explicit new scope and a separate hardware safety review.

## Fail-safe behavior
Default state is `HOLD`.
Return HOLD if:
- confidence/score is below configured threshold;
- prediction is unstable across the configured number of windows;
- timestamp is stale or message expired;
- sequence number is duplicated or out of order;
- command is unknown or malformed;
- source is not permitted;
- model is unavailable or raises an error;
- transport disconnects;
- simulation is disabled;
- joint or workspace limits would be exceeded.

Use rate limits, bounded trajectories and a visible simulation stop control. A confidence score is not automatically a calibrated probability.

## Data ethics
Use public research data under its stated terms. Do not upload EEG data to third-party services by default. Do not infer sensitive personal traits or claim to decode arbitrary thoughts. Document dataset limitations and demographic/generalization limitations.

## Honest demonstrations
- Offline file processing is not real-time acquisition.
- Replaying an EEG file through a stream interface is `REPLAY`, not `LIVE`.
- A classifier trained on public data does not demonstrate control of the developer's own brain.
- Model performance on one participant does not establish generalization.
