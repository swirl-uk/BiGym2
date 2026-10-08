"""Every published task's success predicate, checked against a real demonstration.

``tests/fixtures/demo_endpoints.npz`` holds, for each task the dataset
publishes, the reset seed and the first and last simulator state of one
real, successful human demonstration (regenerate it with
``python -m tests.fixtures.make_demo_endpoints``). Each test resets the
protocol env on that seed, which puts the props that are not in qpos (racks,
reach targets) where the demonstration found them, writes the stored state
and asks the task's predicate:

- at the LAST frame it must be True: a predicate that was inverted or
  tightened past what a human demonstration achieves fails here;
- at the FIRST frame it must be False: a predicate that reports success
  without the task being done (``return True``, a tolerance blown up) fails
  here;
- for the plate tasks, the final state with the plate slid off its slot
  must be False: a clause that neither endpoint exercises on its own.

This is the instantaneous predicate (``BiGymEnv._success``); the protocol's
``success_hold_seconds`` window cannot be judged from one stored frame. No
dataset, network or rendering is needed: the env is built with
``camera_keys=()`` and no physics step is taken, so one task costs about one
env construction.
"""

import numpy as np
import pytest

from bigym.loco.tasks import TASK_MAP
from tests.fixtures.make_demo_endpoints import (
    FIXTURE,
    build_env,
    instantaneous_success,
)

with np.load(FIXTURE) as _fixture:
    ENDPOINTS = {key: _fixture[key] for key in _fixture.files}

TASKS = sorted({key.split("__")[0] for key in ENDPOINTS if "__" in key})


def test_the_fixture_covers_the_published_tasks():
    """Twenty published tasks at the pinned revision, all of them canonical."""
    assert len(TASKS) == 20
    assert set(TASKS) <= set(TASK_MAP)


@pytest.mark.parametrize("task_name", TASKS)
def test_a_real_demo_ends_in_success_and_starts_without_it(task_name):
    seed = int(ENDPOINTS[f"{task_name}__seed"])
    episode = int(ENDPOINTS[f"{task_name}__episode_index"])
    env = build_env(task_name)
    try:
        env.reset(seed=seed)
        assert instantaneous_success(
            env,
            ENDPOINTS[f"{task_name}__qpos_last"],
            ENDPOINTS[f"{task_name}__qvel_last"],
        ), f"{task_name}: demo episode {episode} ends in a state judged a failure"

        # Reset again so nothing the first pose left behind (the reach-target
        # highlight colour) leaks into the second judgement.
        env.reset(seed=seed)
        assert not instantaneous_success(
            env,
            ENDPOINTS[f"{task_name}__qpos_first"],
            ENDPOINTS[f"{task_name}__qvel_first"],
        ), f"{task_name}: demo episode {episode} is judged a success at reset"
    finally:
        env.close()


@pytest.mark.parametrize("task_name", ["move_plate", "move_two_plates"])
def test_a_plate_slid_off_its_slot_is_not_a_success(task_name):
    """A plate touching the target rack but 6 cm off every slot is a failure.

    Both demo endpoints are decided by other clauses too (at reset the plate
    is nowhere near the target rack), so they cannot tell whether the
    slot-distance clause works. Sliding the placed plate 6 cm along its slot,
    away from the robot, keeps it upright, released and resting in the rack
    (asserted below) and takes it more than ``_SUCCESSFUL_DIST`` from every
    slot site: only that clause can reject it.
    """
    seed = int(ENDPOINTS[f"{task_name}__seed"])
    qpos = ENDPOINTS[f"{task_name}__qpos_last"].copy()
    qvel = ENDPOINTS[f"{task_name}__qvel_last"]
    env = build_env(task_name)
    try:
        env.reset(seed=seed)
        inner = env.inner_env
        model = inner.model
        plate = inner.plates[0]
        body = model.bind(plate.body).id
        # The plate's free joint: qpos[adr:adr + 3] is its position.
        adr = int(model.jnt_qposadr[model.body_jntadr[body]])
        qpos[adr] += 0.06  # world +x: along the slot, away from the robot

        success = instantaneous_success(env, qpos, qvel)
        assert plate.is_colliding(inner.rack_target), "slid out of the rack"
        assert not success, f"{task_name}: a plate 6 cm off its slot counts"
    finally:
        env.close()
