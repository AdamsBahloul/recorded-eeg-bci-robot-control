# Data, preprocessing and evaluation protocol

## Initial dataset
Use PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB), via MNE's EEGBCI helper:
https://physionet.org/content/eegmmidb/1.0.0/

Initial download scope:
- Participant: 1 only.
- Runs: 6, 10, 14 only.
- These runs are imagined hands versus feet (both fists versus both feet), not left versus right fist. The downloaded EDF annotation descriptions and event counts were inspected before assigning labels. The verified mapping is T1=hands and T2=feet.
- Never download all participants by default.

The helper is `mne.datasets.eegbci.load_data(subjects, runs, path=...)`. Confirm current MNE API before implementation. Keep dataset location configurable.

## Initial preprocessing hypothesis
A reasonable starting experiment for motor imagery:
- EEG channels only; inspect channel names/types first.
- Notch filtering only if justified by signal inspection. The verified M1 preprocessing did not apply a notch filter.
- Bandpass 8–30 Hz as an initial mu/beta motor-imagery feature band.
- Epoch around task events using a documented time interval; choose and justify it after inspecting annotations and dataset instructions.
- Baseline correction only if appropriate for the chosen epoch and event design.
- Preserve original raw files; write derived data to a separate cache.

Do not silently apply the same pipeline to all datasets. Store preprocessing parameters in config and report them.

## Labels
The M1 metadata records the verified annotations, event IDs/counts, and T1=hands/T2=feet mapping for the selected runs. Preserve this mapping. Exclude rest (T0) from the hands-versus-feet task.

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

## Final deployment input

There is no live EEG acquisition in this project. The MacBook stores the existing participant 1, runs 6/10/14 recordings and sends recorded EEG signal values to the Raspberry Pi 3 B+. The Pi must receive the signal array and its sample rate, channel order, dimensions, run/epoch identity, and timestamp; it must not receive a Mac-computed class label as the input to inference.

Preserve a distinction between M1 reference epochs, leakage-safe M2 out-of-fold evaluation, and the versioned deployment model. Preprocessing and CSP+LDA used during deployment must be reproducible and run on the Pi. Keep original EDF files and M1 artifacts unchanged; use separate derived paths for deployment and transferred messages.
