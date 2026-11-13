"""Bounded 6-joint arm controller for validated Pi command events."""
from __future__ import annotations

import threading

import numpy as np


JOINT_NAMES = (
    "base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch",
    "wrist_roll_joint", "tool_yaw_joint",
)
JOINT_LIMITS = np.asarray([[-2.8, 2.8], [-1.45, 1.2], [-2.2, 0.25], [-1.7, 1.7], [-2.8, 2.8], [-2.8, 2.8]])
POSES = {
    "HANDS": np.asarray([0.3, -0.15, -0.55, -0.35, 0.25, 0.2]),
    "FEET": np.asarray([-0.3, -0.65, -0.25, 0.15, -0.25, -0.2]),
}
HOME = np.asarray([0.0, -0.25, -0.7, -0.35, 0.0, 0.0])


class ArmSimulation:
    """Headless MuJoCo simulation state and rate-limited command application."""

    def __init__(self, xml_path) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.lock = threading.RLock()
        self.state = "HOLD"
        self.target = HOME.copy()
        self.commanded_position = HOME.copy()
        self.commanded_velocity = np.zeros(6, dtype=float)
        self.maximum_velocity = 0.65
        self.maximum_acceleration = 1.5
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self._joint_ids = np.asarray([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES])
        self._qpos_addresses = self.model.jnt_qposadr[self._joint_ids]
        self._actuator_ids = np.asarray([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in (
            "base_yaw_act", "shoulder_act", "elbow_act", "wrist_pitch_act", "wrist_roll_act", "tool_yaw_act"
        )])

    def apply_command(self, event) -> dict[str, object]:
        """Apply only a well-formed, already Mac-validated command event."""
        event.validate()
        with self.lock:
            if event.new_state in ("HOLD", "ERROR"):
                self.state = event.new_state
                self.target = self.data.qpos[self._qpos_addresses].copy()
                self.commanded_position = self.target.copy()
                self.commanded_velocity[:] = 0
                return {"accepted": True, "state": self.state, "reason": "safe_no_motion_state"}
            if event.command_type not in POSES or event.validation_status != "accepted":
                return {"accepted": False, "reason": "invalid_active_command"}
            if event.new_state not in ("HANDS", "FEET"):
                return {"accepted": False, "reason": "invalid_state"}
            target = POSES[event.command_type].copy()
            if np.any(target < JOINT_LIMITS[:, 0]) or np.any(target > JOINT_LIMITS[:, 1]):
                return {"accepted": False, "reason": "target_outside_joint_limits"}
            self.state = event.new_state
            self.target = target
            return {"accepted": True, "state": self.state, "reason": "bounded_pose_accepted"}

    def reset(self) -> None:
        """Reset arm to the documented safe home pose and HOLD."""
        with self.lock:
            self.mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            self.mujoco.mj_forward(self.model, self.data)
            self.state = "HOLD"
            self.target = HOME.copy()
            self.commanded_position = HOME.copy()
            self.commanded_velocity[:] = 0

    def force_hold(self, reason: str = "external_failure") -> None:
        """Stop target motion after a lost or invalid communication session."""
        del reason  # retained in caller's structured logs
        with self.lock:
            self.state = "HOLD"
            self.target = self.data.qpos[self._qpos_addresses].copy()
            self.commanded_position = self.target.copy()
            self.commanded_velocity[:] = 0

    def step(self, duration_seconds: float | None = None) -> None:
        """Advance the simulation with acceleration/velocity-limited setpoints."""
        with self.lock:
            dt = float(self.model.opt.timestep)
            steps = 1 if duration_seconds is None else max(1, int(round(duration_seconds / dt)))
            for _ in range(steps):
                error = self.target - self.commanded_position
                # Brake using stopping distance so rate-limited joints settle on target
                # instead of overshooting and oscillating around it.
                desired_speed = np.minimum(
                    self.maximum_velocity,
                    np.sqrt(2.0 * self.maximum_acceleration * np.abs(error)),
                )
                desired_velocity = np.sign(error) * desired_speed
                delta_velocity = np.clip(
                    desired_velocity - self.commanded_velocity,
                    -self.maximum_acceleration * dt,
                    self.maximum_acceleration * dt,
                )
                self.commanded_velocity += delta_velocity
                next_position = self.commanded_position + self.commanded_velocity * dt
                crossed = np.sign(self.target - next_position) != np.sign(error)
                self.commanded_position = np.where(crossed, self.target, next_position)
                self.commanded_velocity = np.where(crossed, 0.0, self.commanded_velocity)
                self.commanded_position = np.clip(self.commanded_position, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
                self.data.ctrl[self._actuator_ids] = self.commanded_position
                self.mujoco.mj_step(self.model, self.data)

    def status(self) -> dict[str, object]:
        """Return simulation state, joint positions, and bounded targets."""
        with self.lock:
            return {
                "state": self.state,
                "simulation_time_seconds": float(self.data.time),
                "joint_positions_rad": self.data.qpos[self._qpos_addresses].copy().tolist(),
                "joint_targets_rad": self.target.copy().tolist(),
                "joint_limits_rad": JOINT_LIMITS.copy().tolist(),
                "maximum_target_velocity_rad_s": self.maximum_velocity,
                "maximum_target_acceleration_rad_s2": self.maximum_acceleration,
            }
