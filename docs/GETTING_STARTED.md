# NeuroBot: demo to recorded EEG cycle

This guide starts with the interactive EEG/arm demo and ends with an indefinitely repeating recorded-signal cycle. It assumes the Mac project folder is `neurobot_starter` and the MacBook and Raspberry Pi are on the same network.

## 1. Interactive synthetic EEG demo

On a fresh Mac checkout, create the Python 3.11 environment first. If `.venv` already exists, skip the setup command. Install the simulation dependency if it is not already present:

```bash
cd /Users/adam/Public/Projects/NeuroBot/neurobot_starter
./scripts/setup_macos.sh
.venv/bin/python -m pip install -r requirements-simulation.txt
```

Launch the combined signal-generator and 3D arm windows:

```bash
.venv/bin/python scripts/run_synthetic_direction_demo.py
```

Enter a sequence such as:

```text
APPROACH_A, LOWER, GRASP, LIFT, MOVE_B, LOWER, RELEASE, RETRACT, HOME
```

Choose **Run Replay** to animate the signal chart and arm. **Stop / Hold** stops the sequence safely; **Reset Arm** returns the arm to its starting pose. The run buttons preview four seconds from recorded runs 6, 10, or 14, but do not send those previews to the Pi. Those preview buttons require the recordings from step 2; the artificial sequence demo itself does not.

**What this demo proves:** the interface, artificial signal visualization, and illustrative arm sequence work together. It uses generated EEG and a synthetic classifier. It is not the PhysioNet CSP+LDA pipeline and does not establish EEG decoding performance.

## 2. Prepare the Mac environment and approved recordings

Use Python 3.11. A fresh Mac setup is:

```bash
cd /Users/adam/Public/Projects/NeuroBot/neurobot_starter
./scripts/setup_macos.sh
.venv/bin/python -m pip install -r requirements-m2.txt
.venv/bin/python -m pip install -r requirements-simulation.txt
```

If the approved recordings are not already in `data/raw/`, download only participant 1, runs 6, 10, and 14, then preprocess them:

```bash
.venv/bin/python scripts/download_eegbci.py --subjects 1 --runs 6 10 14
.venv/bin/python scripts/preprocess_eegbci.py
```

The recordings are large local inputs and are ignored by Git. The preprocessing step checks annotations and event counts before applying the verified hands/feet class mapping.

If the deployment model is missing, build it from the processed recordings and create the Pi bundle:

```bash
.venv/bin/python scripts/build_deployment_model.py
.venv/bin/python scripts/package_pi_deployment.py
```

The generated package is `data/processed/deployment/neurobot-pi-bundle.tar.gz`. Rebuild and transfer it after changing the Pi runtime or model. The Pi package contains the inference code and model, not the source dataset.

## 3. Prepare the Raspberry Pi

The tested Pi install targets 64-bit Raspberry Pi OS, ARM64, and Python 3.11. Transfer the bundle from a Mac terminal, replacing the placeholders with the Pi's account and network address:

```bash
scp data/processed/deployment/neurobot-pi-bundle.tar.gz <PI_USER>@<PI_IP>:~/
```

Then on the Pi, extract the bundle, inspect the device, and install its pinned runtime:

```bash
mkdir -p ~/neurobot_starter
tar -xzf ~/neurobot-pi-bundle.tar.gz -C ~
cd ~/neurobot_starter
./pi/inspect_device.sh
./pi/install.sh
```

The archive creates `~/neurobot_starter`. If this is an existing installation, extract the new archive over that folder before running the updated cycle. Full device setup notes are in [pi/README.md](../pi/README.md).

## 4. Repeat a recorded EEG cycle on the Pi and Mac

This cycle sends actual recorded EEG windows from run 6 to the Pi. The Mac uses its local verified annotations to select windows for the requested sequence; it sends only the EEG samples and signal metadata. The Pi independently preprocesses and classifies those samples, then returns its commands. The sequence repeats until stopped.

### Start the Mac simulation server

In the first Mac terminal, run the MuJoCo viewer through `mjpython`:

```bash
cd /Users/adam/Public/Projects/NeuroBot/neurobot_starter
.venv/bin/mjpython scripts/run_mac_simulation_server.py \
  --session-id cycle-001 --source sequence \
  --sequence "HANDS*3 FEET*3 HANDS*3" --sequence-run 6 \
  --config config/mac_simulation_demo.toml \
  --m3-config config/m3_unvalidated_simulation_demo.toml \
  --enable-simulation --repeat-forever --interactive-controls
```

Wait until the terminal says it is listening for the Pi. Keep this terminal and the MuJoCo window open. The Mac IP can be found in macOS **System Settings → Wi-Fi → Details** or with `ipconfig getifaddr en0` when using Wi-Fi.

### Start the Pi controller

In a Pi terminal, use the Mac's LAN IP and the same session ID:

```bash
cd ~/neurobot_starter
.venv-pi/bin/python scripts/run_pi_controller.py \
  --host <MACBOOK_IP> --session-id cycle-001 \
  --runtime-config config/m3_unvalidated_simulation_demo.toml
```

For example, if the Mac's address is `192.168.1.17`, use `--host 192.168.1.17`. The Pi connects to port 8765 by default. The Mac server must be started first.

The requested sequence produces HANDS → FEET → HANDS, then repeats. Returning to HANDS returns to the first **commanded HANDS pose**, not the arm's home pose. Type `stop` in the Mac server terminal or close the MuJoCo window to stop the cycle; the system falls back to HOLD.

### Important safety note

This example explicitly enables `m3_unvalidated_simulation_demo.toml`, whose decision-margin threshold is unvalidated. It is included only for the requested software simulation demonstration. It does not make scores calibrated probabilities, validate the model, or permit physical robot use. The ordinary `config/m3.toml` stays HOLD by default.

The sequence is drawn from the same participant and run family used to train the demonstration deployment model. This is a signal-to-command integration demonstration, not independent model evaluation or evidence of generalization.

## 5. Outputs and troubleshooting

Mac session logs and summaries are written under `data/processed/distributed/`. They are local generated files and are excluded from Git. EEG inputs, models, and processed data also stay out of Git.

- **Pi cannot connect:** confirm both devices share the network, the Mac server is waiting, the Pi command uses the Mac's current LAN IP, and both commands use exactly the same session ID.
- **Pi cannot find the model manifest:** extract the latest deployment bundle into `~/neurobot_starter` and verify `models/deployment/manifest.json` exists there.
- **Arm remains in HOLD:** verify the demo TOML was used on both sides, the Mac command includes `--enable-simulation`, and the latest Pi bundle was transferred and extracted. The normal config intentionally disables active transitions.
- **MuJoCo viewer fails to open on macOS:** launch the server with `.venv/bin/mjpython`, not `.venv/bin/python`.

For device installation and deployment details, see [pi/README.md](../pi/README.md). For system boundaries and safety semantics, see [docs/ARCHITECTURE.md](ARCHITECTURE.md) and [docs/SAFETY.md](SAFETY.md).
