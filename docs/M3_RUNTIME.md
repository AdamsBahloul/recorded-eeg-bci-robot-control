# M3 — BCI command replay and safety layer

M3 is a deterministic, offline replay path over the existing participant 1 M1 epochs. It does not acquire live EEG, control a robot, or provide an API/transport.

## Prediction methodology

`bci/classification.py` restores the M2 baseline: four-component MNE CSP (log average power and mutual-information component order) followed by scikit-learn LDA with the SVD solver. The shuffled three-fold `StratifiedGroupKFold` uses recording run as the group and seed 42. Each fold holds out one of runs 6, 10, and 14. CSP and LDA are fit only on the other two runs. The verified M2 reference has 45 predictions, each assigned once, confusion matrix `[[20, 1], [1, 23]]`, accuracy 43/45, and balanced accuracy `(20/21 + 23/24)/2`.

During M3 replay, each epoch is predicted by its corresponding run-held-out model. Before replay, recomputed predictions and decision scores must match the saved M2 out-of-fold artifact. This is within-participant, run-held-out cross-validation, not an independent test set and not evidence of cross-participant or real-world performance.

Class mapping follows verified M1 annotations: hands = class 0, feet = class 1. The saved LDA `decision_function` is a signed score toward class 1 (feet), not a probability. M3's margin is its absolute value and has no probability calibration.

## Runtime safety behavior

`config/m3.toml` is the runtime configuration. It validates the fixed M2 methodology and the replay/safety parameters. The default `confidence.enabled = false` keeps the controller in HOLD and prevents active HANDS/FEET transitions. An active-command threshold requires both explicitly enabling confidence and setting a finite, positive `confidence.threshold`; any such threshold requires separate review and validation before use. No 0.70 threshold is assumed.

When enabled, a prediction must pass the decision-score margin gate and remain stable for the configured number of consecutive epochs inside the stability window. The controller also enforces maximum prediction age, prediction interval, timeout, cooldown, and active-command rate. Invalid inputs, missing data, replay stop, model failure, and shutdown fall back to HOLD; internal failures enter ERROR and require explicit recovery to HOLD. Commands are transport-neutral JSON records and do not invoke external hardware.

## Artifacts and reproduction

Install the tested M2/M3 stack with `python -m pip install -r requirements-m2.txt`, then run:

```bash
python scripts/evaluate_csp_lda.py
python scripts/run_neurobot_replay.py --deterministic
```

M2 metrics and OOF predictions are written under `data/processed/m2/`. M3 writes `run_summary.json`, `replay_predictions.csv`, structured JSONL events, and plots under `data/processed/m3/`. Paths are configurable in TOML. Generated datasets, predictions, logs, and figures are excluded from Git.

The summary records model settings, replay and safety parameters, dataset/preprocessing references, and dependency versions. `requirements-m2-lock.txt` records the tested environment versions. Do not interpret this offline replay as live BCI control.
