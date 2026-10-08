"""Shaping bonuses fire once, on the step the float32 progress label crosses a tier."""

import numpy as np
import pytest

from bigym.loco.config import EnvConfig
from bigym.loco.event_progress import EventProgress


@pytest.fixture
def progress():
    config = EnvConfig(
        event_progress_enabled=True,
        event_reward_shaping_enabled=True,
        event_reward_holding_bonus=0.1,
        event_reward_lift_bonus=0.2,
        event_reward_target_bonus=0.3,
    )
    return EventProgress("move_plate", config)


@pytest.mark.parametrize(
    "previous, current, bonus",
    [
        (0.0, 0.35, 0.1),  # grasp: float32(0.35) is 0.34999999
        (0.0, 0.60, 0.3),
        (0.0, 0.85, 0.6),
        (0.20, 1.0, 0.6),
        (0.35, 0.35, 0.0),  # held on: no second bonus
        (0.35, 0.60, 0.2),
        (0.60, 0.80, 0.0),
    ],
)
def test_each_tier_bonus_fires_once(progress, previous, current, bonus):
    assert progress.shaping(previous, np.float32(current)) == pytest.approx(bonus)
