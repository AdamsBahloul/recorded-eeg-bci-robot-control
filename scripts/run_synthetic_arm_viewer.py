#!/usr/bin/env python3
"""Local subprocess bridge: synthetic demo actions to the 3D MuJoCo viewer."""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import mujoco.viewer

from demo.synthetic_arm import PickPlaceTask
from simulation.controller import ArmSimulation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    with socket.create_connection(("127.0.0.1", args.port), timeout=15) as connection:
        connection.settimeout(0.0)
        arm = ArmSimulation(ROOT / "simulation/arm.xml")
        task = PickPlaceTask(arm)
        buffer = bytearray()
        with mujoco.viewer.launch_passive(arm.model, arm.data) as viewer:
            viewer.cam.lookat[:] = [0.72, 0.0, 0.32]
            viewer.cam.distance = 2.25
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -24
            connection.sendall(b'{"type":"ready"}\n')
            running = True
            while running and viewer.is_running():
                try:
                    chunk = connection.recv(4096)
                    if not chunk:
                        break
                    buffer.extend(chunk)
                    while b"\n" in buffer:
                        row, _, remainder = buffer.partition(b"\n")
                        buffer[:] = remainder
                        try:
                            message = json.loads(row)
                            kind = message.get("type")
                            if kind == "action":
                                task.command(message["action"])
                            elif kind == "reset":
                                task.reset()
                                arm.reset()
                            elif kind == "stop":
                                task.stop()
                            elif kind == "quit":
                                running = False
                        except (ValueError, KeyError, json.JSONDecodeError):
                            arm.force_hold("invalid_synthetic_demo_message")
                except BlockingIOError:
                    pass
                task.step()
                viewer.sync()
                time.sleep(arm.model.opt.timestep)
            arm.force_hold("synthetic_demo_viewer_closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
