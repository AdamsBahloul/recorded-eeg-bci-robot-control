"""Headless MuJoCo safety, limit, reset, and command mapping tests."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from runtime.command import CommandEvent
from simulation.controller import ArmSimulation, HOME, JOINT_LIMITS, POSES


ROOT = Path(__file__).resolve().parents[1]


def event(command: str, previous: str, new: str, sequence: int) -> CommandEvent:
    active = command in ("HANDS", "FEET")
    predicted = 0 if command == "HANDS" else 1 if command == "FEET" else None
    score = 3.0 if predicted == 1 else -3.0 if predicted == 0 else None
    return CommandEvent(
        command_id=f"sim-{sequence}", command_type=command,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        source_epoch_id="run-epoch" if active else None,
        predicted_class=predicted, decision_score=score,
        confidence_measure=abs(score) if score is not None else None,
        confidence_type="decision_score_margin" if active else None,
        previous_state=previous, new_state=new,
        validation_status="accepted", reason="test_validated_command",
        sequence=sequence,
    )


def test_arm_starts_at_safe_home_and_has_six_bounded_joints() -> None:
    arm = ArmSimulation(ROOT / "simulation/arm.xml")
    status = arm.status()
    assert status["state"] == "HOLD"
    np.testing.assert_allclose(status["joint_positions_rad"], HOME, atol=1e-12)
    assert len(status["joint_positions_rad"]) == 6


def test_hands_feet_pose_mapping_and_joint_targets_are_bounded() -> None:
    arm = ArmSimulation(ROOT / "simulation/arm.xml")
    for command in ("HANDS", "HOLD", "FEET"):
        prior = arm.state
        next_state = command
        result = arm.apply_command(event(command, prior, next_state, 1 if command == "HANDS" else 2 if command == "HOLD" else 3))
        assert result["accepted"] is True
        if command in POSES:
            np.testing.assert_allclose(arm.target, POSES[command])
        for _ in range(100):
            arm.step()
        qpos = np.asarray(arm.status()["joint_positions_rad"])
        assert np.all(qpos >= JOINT_LIMITS[:, 0] - 1e-5)
        assert np.all(qpos <= JOINT_LIMITS[:, 1] + 1e-5)


def test_error_and_disconnect_fallback_hold_motion() -> None:
    arm = ArmSimulation(ROOT / "simulation/arm.xml")
    arm.apply_command(event("HANDS", "HOLD", "HANDS", 1))
    arm.step(0.3)
    arm.apply_command(event("HOLD", "HANDS", "ERROR", 2))
    assert arm.status()["state"] == "ERROR"
    np.testing.assert_array_equal(arm.commanded_velocity, np.zeros(6))
    arm.force_hold("communication_loss")
    assert arm.status()["state"] == "HOLD"
    arm.reset()
    assert arm.status()["state"] == "HOLD"
    np.testing.assert_allclose(arm.target, HOME)


def test_rate_limited_joint_motion_settles_without_overshoot() -> None:
    arm = ArmSimulation(ROOT / "simulation/arm.xml")
    target = np.asarray([0.4, 0.15, -0.9, 0.25, 0.2, -0.15])
    arm.target = target.copy()
    arm.step(4.0)
    np.testing.assert_allclose(arm.commanded_position, target, atol=1e-8)
    np.testing.assert_allclose(arm.commanded_velocity, np.zeros(6), atol=1e-8)
