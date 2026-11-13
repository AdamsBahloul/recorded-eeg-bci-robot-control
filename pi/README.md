# Raspberry Pi setup

The Pi runs EEG preprocessing, CSP+LDA inference, and the safety controller. The Mac sends raw recorded signal samples and hosts the MuJoCo simulation. The Pi does not receive the requested sequence labels as inference input.

## Supported setup

The pinned runtime targets a Raspberry Pi 3 B+ with 64-bit Raspberry Pi OS (`aarch64`) and Python 3.11. The installer checks the architecture and installs compatible wheels only; it does not attempt memory-intensive scientific-package builds on the Pi.

## Install

Follow the complete Mac-to-Pi bundle, install, connection, and replay instructions in [the getting started guide](../docs/GETTING_STARTED.md#3-prepare-the-raspberry-pi). The high-level order is:

1. On the Mac, build the deployment model if needed and package the Pi runtime.
2. Copy `data/processed/deployment/neurobot-pi-bundle.tar.gz` to the Pi.
3. On the Pi, extract the bundle into `~/neurobot_starter`.
4. Run `./pi/inspect_device.sh`, then `./pi/install.sh`.
5. Start the Mac simulation server, then run `scripts/run_pi_controller.py` on the Pi with the Mac's LAN IP and the same session ID.

The model and manifest are stored under `models/deployment/` in the Pi bundle. Startup validates the model manifest and dependency versions. The bundle does not contain the EEG dataset or M1 training artifacts.

## Safety and device verification

`config/m3.toml` keeps active commands disabled by default. The `m3_unvalidated_simulation_demo.toml` override enables simulated HANDS/FEET transitions for the documented demo only; its threshold is unvalidated and must never be used to control physical hardware. See [the safety notes](../docs/SAFETY.md).

Local software integration checks do not verify operation on a physical Pi. Record OS, architecture, install result, memory, CPU, latency, network reliability, and endurance only after measuring them on that device.
