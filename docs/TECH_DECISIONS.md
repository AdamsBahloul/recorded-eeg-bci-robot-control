# Technical decisions and defaults

| Area | Initial decision | Reason / revisit condition |
|---|---|---|
| Language | Python 3.11 | Broad scientific package compatibility |
| EEG | MNE-Python | Established EEG workflows |
| Dataset | PhysioNet EEGMMIDB | Public motor imagery recordings |
| Initial model | CSP + LDA | Interpretable baseline before deep learning |
| Deep model | EEGNet / PyTorch | Add only after baseline is correct |
| Simulation | MuJoCo | Native macOS support and Python API |
| Middleware | None initially | Avoid integration overhead until vertical slice |
| Later transport | MQTT or ROS 2 | Choose after Pi model and use case confirmed |
| Backend | FastAPI | Add when replay/control API is needed |
| Frontend | React + TypeScript | Add after core system works |
| Database | None initially | Files/JSON logs are sufficient for prototype |
| Deployment | Local first | No cloud requirement |

## Do not add without justification
- Docker for the initial Mac pipeline.
- ROS 2 before standalone MuJoCo control works.
- LSL before a genuine stream source is available.
- MQTT before a process/network boundary is needed.
- React before the core command loop exists.
- Cloud APIs or paid services.
- Large language models in the control loop.
