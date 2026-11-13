"""Deterministic recorded-epoch replay controls and validation."""
from __future__ import annotations

from datetime import datetime, timezone

from runtime.replay import EpochReplay


def test_sequential_replay_ids_order_and_eof(m1_dataset) -> None:
    replay = EpochReplay(m1_dataset, deterministic_mode=True)
    samples = list(replay)

    assert len(samples) == 45
    assert [sample.sample_index for sample in samples] == list(range(45))
    assert samples[0].epoch_id == "S001-R06-E000"
    assert samples[14].epoch_id == "S001-R06-E014"
    assert samples[15].epoch_id == "S001-R10-E000"
    assert samples[30].epoch_id == "S001-R14-E000"
    assert samples[0].run_id == 6 and samples[15].run_id == 10 and samples[30].run_id == 14
    assert replay.complete
    assert replay.step() is None


def test_pause_resume_step_and_reset(m1_dataset) -> None:
    replay = EpochReplay(m1_dataset)
    replay.pause()
    assert replay.next_sample() is None
    first = replay.step()
    assert first is not None and first.sample_index == 0
    assert replay.paused
    replay.resume()
    assert replay.next_sample().sample_index == 1
    replay.reset()
    assert replay.position == 0 and not replay.paused and not replay.complete
    assert replay.next_sample().epoch_id == "S001-R06-E000"


def test_replay_speed_and_delay_are_applied_and_repeatable(m1_dataset) -> None:
    sleeps: list[float] = []
    replay = EpochReplay(m1_dataset, delay_between_epochs_seconds=0.4, speed=2.0,
                         deterministic_mode=False, sleep=sleeps.append,
                         monotonic=lambda: 5.0,
                         utc_now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))
    first_three = []
    iterator = replay.iter_samples(poll_seconds=0.5)
    first_three.append(next(iterator))
    first_three.append(next(iterator))
    first_three.append(next(iterator))

    assert [sample.replay_elapsed_seconds for sample in first_three] == [0.0, 0.2, 0.4]
    assert sleeps == [0.2, 0.2]


def test_invalid_speed_and_delay_are_rejected(m1_dataset) -> None:
    import pytest

    with pytest.raises(ValueError, match="speed"):
        EpochReplay(m1_dataset, speed=0)
    with pytest.raises(ValueError, match="delay"):
        EpochReplay(m1_dataset, delay_between_epochs_seconds=-1)
