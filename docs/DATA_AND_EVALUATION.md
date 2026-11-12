# Data, preprocessing and evaluation protocol

## Initial dataset
Use PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB), via MNE's EEGBCI helper:
https://physionet.org/content/eegmmidb/1.0.0/

Initial download scope:
- Participant: 1 only.
- Runs: 6, 10, 14 only.
- These runs are commonly used for imagined left/right fist tasks. Always inspect the downloaded recording annotations and event descriptions before assigning labels.
- Never download all participants by default.

The helper is `mne.datasets.eegbci.load_data(subjects, runs, path=...)`. Confirm current MNE API before implementation. Keep dataset location configurable.

## Initial preprocessing hypothesis
A reasonable starting experiment for motor imagery:
- EEG channels only; inspect channel names/types first.
- Notch filter at 50 Hz for local mains interference if present and justified by signal inspection.
- Bandpass 8–30 Hz as an initial mu/beta motor-imagery feature band.
- Epoch around task events using a documented time interval; choose and justify it after inspecting annotations and dataset instructions.
- Baseline correction only if appropriate for the chosen epoch and event design.
- Preserve original raw files; write derived data to a separate cache.

Do not silently apply the same pipeline to all datasets. Store preprocessing parameters in config and report them.

## Labels
Do not assume event code mappings. Inspect `raw.annotations`, event IDs and MNE event conversion. Confirm which annotation corresponds to imagined left fist and imagined right fist for the selected runs. Exclude rest or other classes unless explicitly included in the experiment.

## Evaluation
- Do not randomly split overlapping windows from the same recording across train and test.
- For the first single-participant experiment, use run/session-aware validation where feasible and clearly label it as within-subject evaluation.
- For generalization claims, evaluate on held-out participants; do not claim cross-subject generalization from one participant.
- Fit CSP, scalers, feature selection and all learned preprocessing only on training folds.
- Report accuracy, balanced accuracy, macro-F1, confusion matrix, class counts, split strategy, participant IDs, run IDs, and preprocessing settings.
- Include a majority-class baseline.
- Report inference latency separately from data acquisition and replay latency.
- Do not present accuracy as proof of reliable real-world BCI control.

## Reproducibility metadata
Record:
- Dataset and version/URL.
- Participant and run identifiers.
- Download date.
- MNE and dependency versions.
- Channel selection.
- Filters and epoch window.
- Event mapping.
- Random seeds.
- Split strategy.
- Model configuration and checkpoint ID.
- Metrics and confusion matrix.

## Future live EEG
A real headset and its supported acquisition API are required for live human EEG. LSL may be used later if the headset supports it. Until then, label all stream playback as `REPLAY`.
