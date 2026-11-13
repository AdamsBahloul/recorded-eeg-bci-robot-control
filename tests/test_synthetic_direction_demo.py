import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")

from demo.synthetic_directions import ACTIONS, make_synthetic_model, predict_synthetic, synthesize_epoch
from demo.synthetic_arm import ACTION_POSES
from scripts.run_synthetic_direction_demo import _animate_signal, parse_sequence, plt


@pytest.fixture(scope="module")
def model():
    return make_synthetic_model(seed=42)


@pytest.mark.parametrize("direction", ACTIONS)
def test_synthetic_demo_generates_and_classifies_requested_action(model, direction):
    signal = synthesize_epoch(direction, seed=123)
    assert signal.shape == (14, 481)
    assert np.isfinite(signal).all()
    prediction, scores = predict_synthetic(model, signal)
    assert prediction == direction
    assert scores.shape == (len(ACTIONS),)


def test_synthetic_demo_rejects_malformed_signal(model):
    with pytest.raises(ValueError, match="expected finite EEG"):
        predict_synthetic(model, np.zeros((64, 481)))


def test_sequence_parser_supports_repetition_and_complex_actions():
    assert parse_sequence("APPROACH_A, LOWER GRASP, LIFT, MOVE_B, RELEASE, HOME*2, HOLD") == [
        "APPROACH_A", "LOWER", "GRASP", "LIFT", "MOVE_B", "RELEASE", "HOME", "HOME", "HOLD"
    ]


def test_every_synthetic_action_has_bounded_arm_pose():
    from simulation.controller import JOINT_LIMITS

    from demo.synthetic_arm import LOWER_B_POSE

    for pose in [*ACTION_POSES.values(), LOWER_B_POSE]:
        array = np.asarray(pose)
        assert np.all(array >= JOINT_LIMITS[:, 0])
        assert np.all(array <= JOINT_LIMITS[:, 1])


def test_pick_and_place_sequence_moves_object_from_a_to_b():
    from demo.synthetic_arm import PickPlaceTask
    from simulation.controller import ArmSimulation

    arm = ArmSimulation("simulation/arm.xml")
    task = PickPlaceTask(arm)
    for action in ("APPROACH_A", "LOWER", "GRASP", "LIFT", "MOVE_B", "LOWER"):
        task.command(action)
        for _ in range(1000):
            task.step()
    assert task.grasp_succeeded
    assert task.grasp_contact_confirmed
    assert task.carrying
    assert arm.model.geom_contype[task.payload_geom] == 0
    task.command("RELEASE")
    for _ in range(1000):
        task.step()
    assert not task.carrying
    assert arm.model.geom_contype[task.payload_geom] == 2
    x, y, _z = task.object_position()
    assert abs(x - 0.72) < 0.12
    assert abs(y - 0.18) < 0.10
    assert task.object_position()[2] >= 0.084


def test_collision_masks_allow_gripper_payload_and_payload_table_only():
    import mujoco
    from simulation.controller import ArmSimulation

    model = ArmSimulation("simulation/arm.xml").model
    geom = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def collides(left, right):
        return bool((model.geom_contype[left] & model.geom_conaffinity[right])
                    or (model.geom_contype[right] & model.geom_conaffinity[left]))

    assert collides(geom("left_finger_link"), geom("payload"))
    assert collides(geom("right_finger_link"), geom("payload"))
    assert collides(geom("payload"), geom("work_table"))
    assert not collides(geom("upper_link"), geom("payload"))
    assert not collides(geom("upper_link"), geom("work_table"))
    assert collides(geom("payload"), geom("floor"))
    assert not collides(geom("left_finger_tip"), geom("floor"))
    assert not collides(geom("right_finger_tip"), geom("floor"))


def test_synthetic_eeg_chart_reveals_samples_over_time(monkeypatch):
    class Status:
        text = ""

        def set_text(self, value):
            self.text = value

    figure, axis = plt.subplots()
    monkeypatch.setattr(plt, "pause", lambda _seconds: None)
    signal = synthesize_epoch("GRASP", seed=9)[:4]
    status = Status()
    assert _animate_signal(axis, signal, 160.0, "test signal", status, {"value": False})
    assert len(axis.lines) == 4
    assert all(len(line.get_xdata()) == signal.shape[1] for line in axis.lines)
    assert "3.00 s" in status.text
    plt.close(figure)
