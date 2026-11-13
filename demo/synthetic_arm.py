"""Bounded joint targets for the synthetic-only arm illustration."""
from __future__ import annotations

import numpy as np

from simulation.controller import HOME, JOINT_LIMITS


ACTION_POSES = {
    "APPROACH_A": [-0.245, 0.889, -1.665, 1.334, 0.0, 0.0],
    "LOWER": [-0.245, 0.211, -0.389, 1.434, 0.0, 0.0],
    "GRASP": [-0.245, 0.211, -0.389, 1.434, 0.0, 0.0],
    "LIFT": [-0.245, 0.889, -1.665, 1.334, 0.0, 0.0],
    "MOVE_B": [0.245, 0.889, -1.665, 1.334, 0.0, 0.0],
    "RELEASE": [0.245, 0.013, 0.233, 1.020, 0.0, 0.0],
    "RETRACT": [0.245, 0.889, -1.665, 1.334, 0.0, 0.0],
    "HOME": [0.0, -0.25, -0.7, -0.35, 0.0, 0.0],
    "HOLD": HOME.tolist(),
}
LOWER_B_POSE = [0.245, 0.211, -0.389, 1.434, 0.0, 0.0]


def set_synthetic_target(arm, action: str) -> None:
    """Set a bounded visual-only target; no transport or M3 safety is involved."""
    if action not in ACTION_POSES:
        raise ValueError(f"unsupported synthetic arm action: {action}")
    target = np.asarray(ACTION_POSES[action], dtype=float)
    if action == "LOWER" and arm.state in ("MOVE_B", "RETRACT"):
        target = np.asarray(LOWER_B_POSE, dtype=float)
    if np.any(target < JOINT_LIMITS[:, 0]) or np.any(target > JOINT_LIMITS[:, 1]):
        raise ValueError(f"synthetic target outside arm joint limits: {action}")
    arm.state = action
    arm.target = target
    left_gripper = arm.mujoco.mj_name2id(arm.model, arm.mujoco.mjtObj.mjOBJ_ACTUATOR, "left_gripper_act")
    right_gripper = arm.mujoco.mj_name2id(arm.model, arm.mujoco.mjtObj.mjOBJ_ACTUATOR, "right_gripper_act")
    if action == "GRASP":
        # Match the 90 mm payload width instead of driving the fingers through it.
        arm.data.ctrl[left_gripper], arm.data.ctrl[right_gripper] = -0.032, 0.032
    elif action in ("APPROACH_A", "LOWER", "LIFT", "MOVE_B", "RELEASE", "RETRACT", "HOME", "HOLD"):
        arm.data.ctrl[left_gripper], arm.data.ctrl[right_gripper] = -0.05, 0.05


class PickPlaceTask:
    """Simulation-only gripper and object latch for a pick-A/drop-B sequence."""

    def __init__(self, arm) -> None:
        self.arm = arm
        mujoco = arm.mujoco
        self.object_body = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.object_joint = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        self.object_qadr = int(arm.model.jnt_qposadr[self.object_joint])
        self.object_dadr = int(arm.model.jnt_dofadr[self.object_joint])
        self.tool_site = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_SITE, "tool_tip")
        self.tool_body = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_BODY, "tool_yaw")
        self.payload_geom = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_GEOM, "payload")
        self.finger_geoms = {
            mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("left_finger_link", "left_finger_tip", "right_finger_link", "right_finger_tip")
        }
        self.payload_collision = (int(arm.model.geom_contype[self.payload_geom]),
                                  int(arm.model.geom_conaffinity[self.payload_geom]))
        self.finger_collision = {}
        for geom_id in self.finger_geoms:
            self.finger_collision[geom_id] = (
                int(arm.model.geom_contype[geom_id]), int(arm.model.geom_conaffinity[geom_id])
            )
            self._set_finger_collision(geom_id, False)
        self.pending_grasp = False
        self.carrying = False
        self.grasp_succeeded = False
        self.grasp_contact_confirmed = False
        self.object_tool_offset = None

    def command(self, action: str) -> None:
        set_synthetic_target(self.arm, action)
        if action == "GRASP":
            self._set_all_finger_collisions(True)
            self.pending_grasp = True
            self.grasp_succeeded = False
            self.grasp_contact_confirmed = False
        elif action == "RELEASE":
            self.carrying = False
            self.pending_grasp = False
            self._set_payload_collision(True)
        elif action in ("APPROACH_A", "LOWER", "LIFT", "MOVE_B", "RETRACT", "HOME", "HOLD"):
            if not self.carrying:
                self._set_all_finger_collisions(False)
        elif action == "HOME":
            self.pending_grasp = False

    def step(self, duration_seconds: float | None = None) -> None:
        self.arm.step(duration_seconds)
        mujoco = self.arm.mujoco
        tool = self.arm.data.site_xpos[self.tool_site].copy()
        obj = self.arm.data.xpos[self.object_body].copy()
        if self.pending_grasp and self._both_fingers_contact_payload():
            self.carrying = True
            self.grasp_succeeded = True
            self.grasp_contact_confirmed = True
            self.pending_grasp = False
            self.object_tool_offset = obj - self.arm.data.site_xpos[self.tool_site]
            self._set_payload_collision(False)
        if self.carrying:
            self.arm.data.qpos[self.object_qadr:self.object_qadr + 3] = (
                self.arm.data.site_xpos[self.tool_site] + self.object_tool_offset
            )
            self.arm.data.qvel[self.object_dadr:self.object_dadr + 6] = 0.0
            mujoco.mj_forward(self.arm.model, self.arm.data)

    def object_position(self) -> np.ndarray:
        return self.arm.data.xpos[self.object_body].copy()

    def reset(self) -> None:
        self._set_payload_collision(True)
        self.pending_grasp = False
        self.carrying = False
        self.grasp_succeeded = False
        self.grasp_contact_confirmed = False
        self.object_tool_offset = None
        self._set_all_finger_collisions(False)

    def stop(self) -> None:
        self.pending_grasp = False
        self.arm.force_hold("synthetic_task_stopped")

    def _set_payload_collision(self, enabled: bool) -> None:
        if enabled:
            self.arm.model.geom_contype[self.payload_geom] = self.payload_collision[0]
            self.arm.model.geom_conaffinity[self.payload_geom] = self.payload_collision[1]
        else:
            # The gripper has physically contacted both sides; suppress contact while latched.
            self.arm.model.geom_contype[self.payload_geom] = 0
            self.arm.model.geom_conaffinity[self.payload_geom] = 0

    def _set_finger_collision(self, geom_id: int, enabled: bool) -> None:
        if enabled:
            self.arm.model.geom_contype[geom_id], self.arm.model.geom_conaffinity[geom_id] = self.finger_collision[geom_id]
        else:
            self.arm.model.geom_contype[geom_id] = 0
            self.arm.model.geom_conaffinity[geom_id] = 0

    def _set_all_finger_collisions(self, enabled: bool) -> None:
        for geom_id in self.finger_geoms:
            self._set_finger_collision(geom_id, enabled)

    def _both_fingers_contact_payload(self) -> bool:
        seen: set[int] = set()
        for contact in self.arm.data.contact[:self.arm.data.ncon]:
            if contact.geom1 == self.payload_geom and contact.geom2 in self.finger_geoms:
                seen.add(contact.geom2)
            elif contact.geom2 == self.payload_geom and contact.geom1 in self.finger_geoms:
                seen.add(contact.geom1)
        left = any(self.arm.mujoco.mj_id2name(self.arm.model, self.arm.mujoco.mjtObj.mjOBJ_GEOM, gid).startswith("left_") for gid in seen)
        right = any(self.arm.mujoco.mj_id2name(self.arm.model, self.arm.mujoco.mjtObj.mjOBJ_GEOM, gid).startswith("right_") for gid in seen)
        return left and right
