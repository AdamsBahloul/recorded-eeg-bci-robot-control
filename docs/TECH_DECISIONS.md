# Technical decisions and defaults

| Area | Decision | Reason / revisit condition |
|---|---|---|
| Data source | Existing PhysioNet EEGMMIDB, participant 1, runs 6/10/14 only | Fixed project scope; no additional data or live acquisition |
| Language | Python 3.11 | Current Mac environment; verify supported version on Pi OS before deployment |
| EEG preprocessing | Existing documented M1 pipeline, reproduced on Pi for deployment input | Pi must receive signal data and do its own signal processing |
| Evaluation model | MNE CSP(4) + scikit-learn LDA(SVD), 3-fold StratifiedGroupKFold by run, seed 42 | Reproduces the verified M2 evaluation methodology |
| Deployment model | Versioned fitted CSP+LDA package, separate from OOF evaluation artifacts | Enables Pi inference without conflating training and held-out evaluation |
| Pi | Raspberry Pi 3 Model B+ | Final embedded inference and safety controller; inspect OS and benchmark actual device |
| Simulation | MuJoCo on MacBook Pro M4 | Simulation remains on Mac and receives validated Pi commands |
| Communication | Small versioned EEG-signal and command schemas behind transport interfaces | Choose a lightweight transport after checking Pi OS/network; no ROS 2 requirement |
| Safety | HOLD by default; Pi and Mac receiver validate independently | Fail-safe distributed command path |
| Artifacts | Data, deployment packages, logs, and generated plots stay out of Git | Avoid committing datasets and generated/large files |

## Excluded from the final scope

- Live EEG acquisition, headset APIs, and LSL.
- Additional participants or datasets.
- EEGNet/deep learning, ROS 2, dashboard/API, cloud services, and physical actuator control.
- MQTT broker infrastructure unless the selected communication design establishes a concrete need.
