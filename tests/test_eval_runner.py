"""The reference runner takes success from the env and scores a fall as a failure."""

from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest

from bigym.loco.env import BiGym
from bigym.loco.eval import evaluate, is_success


class Step(NamedTuple):
    done: bool
    reward: float

    def last(self) -> bool:
        return self.done


class RewardedEnv:
    """One-step episodes that earn the full reward, with set success and fall latches."""

    config = None
    official_config = None
    episode_termination = BiGym.episode_termination

    def __init__(self, succeeded: bool, fell: bool):
        self.succeeded = succeeded
        self._episode_succeeded = False
        self._episode_fell = fell
        self._last_truncation = None

    def reset(self, seed: int) -> Step:
        self._episode_succeeded = False
        return Step(False, 0.0)

    def step(self, action: np.ndarray) -> Step:
        self._episode_succeeded = self.succeeded
        return Step(True, 1.0)

    def episode_succeeded(self) -> bool:
        return self._episode_succeeded

    def episode_fell(self) -> bool:
        return self._episode_fell

    def substrate_fingerprint(self) -> dict:
        return {}


def outcome(succeeded: bool, fell: bool) -> SimpleNamespace:
    return SimpleNamespace(
        episode_succeeded=lambda: succeeded, episode_fell=lambda: fell
    )


def test_is_success_needs_the_task_done_and_no_fall():
    assert is_success(outcome(True, False))
    assert not is_success(outcome(True, True))
    assert not is_success(outcome(False, False))


@pytest.mark.parametrize(
    "succeeded, fell, success, termination",
    [
        (True, False, 1, "success"),
        (True, True, 0, "fell"),
        # Reward the task did not hand out (shaping) is not a success.
        (False, False, 0, "terminated"),
    ],
)
def test_runner_takes_success_from_the_env(succeeded, fell, success, termination):
    result = evaluate(
        lambda timestep: np.zeros(1),
        task_name="move_plate",
        method="test",
        episodes=2,
        env=RewardedEnv(succeeded, fell),
        verbose=False,
    )
    assert result["success_rate"] == success
    for record in result["episode_records"]:
        assert record["reward"] == 1.0
        assert record["success"] == success
        assert record["fell"] == int(fell)
        assert record["termination"] == termination
