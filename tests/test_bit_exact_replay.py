"""Save/restore and demonstration replay are bit-exact.

Restoring a state runs ``mj_forward``, so the replay matches the original
run only if every step leaves kinematics, contacts and sensors at the
current state. On ``drawer_top_open`` nothing else refreshes them before the
lower-body controller reads the pelvis IMU.
"""

import os
import warnings

import mujoco
import numpy as np
import pytest

from bigym.loco import make
from bigym.robots.configs.g1 import G1_DEX1_QPOS_CLOSED

TASK = "drawer_top_open"


def make_env(**overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make(TASK, **overrides)


def restore(env, state: dict, lowerbody_state: dict) -> None:
    """Write a saved simulator state and controller state, as demo replay does."""
    inner = env.bigym.inner_env
    model, data = inner.model, inner.data
    for name, value in state.items():
        getattr(data, name)[:] = value
    mujoco.mj_forward(model, data)
    data.qacc_warmstart[:] = state["qacc_warmstart"]
    env.set_lowerbody_state(lowerbody_state)


def test_mid_episode_save_restore_is_bit_exact():
    env = make_env(camera_keys=())
    try:
        outer = env.bigym
        data = outer.inner_env.data
        env.reset(seed=3)
        hold = outer.normalize_action(outer.raw_hold_action())
        rng = np.random.default_rng(0)
        actions = np.clip(
            hold + rng.normal(0.0, 0.02, (30, hold.shape[0])), -1.0, 1.0
        ).astype(np.float32)
        for action in actions[:10]:
            env.step(action)
        state = {
            name: getattr(data, name).copy()
            for name in ("qpos", "qvel", "ctrl", "act", "qacc_warmstart")
        }
        lowerbody_state = {
            key: np.array(value, copy=True)
            for key, value in env.get_lowerbody_state().items()
        }
        first = []
        for action in actions[10:]:
            env.step(action)
            first.append(data.qpos.copy())
        restore(env, state, lowerbody_state)
        for step, action in enumerate(actions[10:]):
            env.step(action)
            np.testing.assert_array_equal(data.qpos, first[step])
    finally:
        env.close()


@pytest.mark.parametrize("reset_between", [False, True])
def test_save_restore_while_the_grippers_move_is_bit_exact(reset_between):
    """The snapshot is taken while the grippers' slew limiter is still closing them.

    The limiter's last target is neither MuJoCo state nor controller state.
    A reset leaves it at the reset pose and the run after the snapshot leaves
    it at closed, so both ways of getting back to the snapshot are covered.
    The simulation time is not restored either way.
    """
    env = make_env(camera_keys=())
    try:
        outer = env.bigym
        data = outer.inner_env.data
        grippers = outer.inner_env.robot.grippers
        env.reset(seed=3)
        hold = outer.normalize_action(outer.raw_hold_action())
        rng = np.random.default_rng(0)
        actions = np.clip(
            hold + rng.normal(0.0, 0.02, (30, hold.shape[0])), -1.0, 1.0
        ).astype(np.float32)
        actions[8:, -len(grippers) :] = 1.0  # close both from step 8 on
        for action in actions[:10]:
            env.step(action)
        state = {
            name: getattr(data, name).copy()
            for name in ("qpos", "qvel", "ctrl", "act", "qacc_warmstart")
        }
        lowerbody_state = {
            key: np.array(value, copy=True)
            for key, value in env.get_lowerbody_state().items()
        }
        for side in grippers:
            target = lowerbody_state[f"env.gripper_target.{side.value.lower()}"]
            assert np.all(target > G1_DEX1_QPOS_CLOSED + 0.004), "not mid-slew"
        first = []
        for action in actions[10:]:
            env.step(action)
            first.append(data.qpos.copy())
        if reset_between:
            env.reset(seed=3)
        restore(env, state, lowerbody_state)
        for step, action in enumerate(actions[10:]):
            env.step(action)
            np.testing.assert_array_equal(data.qpos, first[step])
    finally:
        env.close()


def test_restoring_a_snapshot_from_before_the_first_gripper_command():
    """Without reset warmup the limiter has no target yet at the snapshot.

    Restoring has to bring that back too, or the limiter continues from the
    target the later run left.
    """
    env = make_env(camera_keys=(), controller={"reset_warmup_steps": 0})
    try:
        outer = env.bigym
        data = outer.inner_env.data
        grippers = outer.inner_env.robot.grippers
        env.reset(seed=3)
        assert all(gripper.get_state().size == 0 for gripper in grippers.values())
        state = {
            name: getattr(data, name).copy()
            for name in ("qpos", "qvel", "ctrl", "act", "qacc_warmstart")
        }
        lowerbody_state = {
            key: np.array(value, copy=True)
            for key, value in env.get_lowerbody_state().items()
        }
        actions = np.tile(outer.normalize_action(outer.raw_hold_action()), (10, 1))
        actions = actions.astype(np.float32)
        actions[:, -len(grippers) :] = 1.0  # close both
        first = []
        for action in actions:
            env.step(action)
            first.append(data.qpos.copy())
        restore(env, state, lowerbody_state)
        for step, action in enumerate(actions):
            env.step(action)
            np.testing.assert_array_equal(data.qpos, first[step])
    finally:
        env.close()


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("BIGYM_DATASET_TESTS"),
    reason="set BIGYM_DATASET_TESTS=1 to download real demonstrations",
)
def test_a_demonstration_replays_bit_exactly():
    env = make_env()
    try:
        outer = env.bigym
        data = outer.inner_env.data
        episode = outer.load_demo_episodes(1)[0]
        env.reset(seed=int(np.asarray(episode["seed"]).reshape(-1)[0]))
        state = {
            name: episode[f"init_{name}"]
            for name in ("qpos", "qvel", "ctrl", "act", "qacc_warmstart")
            if f"init_{name}" in episode
        }
        lowerbody_state = {
            key.removeprefix("lb_state."): np.asarray(value)
            for key, value in episode.items()
            if key.startswith("lb_state.")
        }
        restore(env, state, lowerbody_state)
        actions = np.asarray(episode["action"], dtype=np.float32)
        for step in range(1, len(actions)):
            time_step = env.step(actions[step])
            control = outer.get_last_lowerbody_control_info()
            np.testing.assert_array_equal(data.qpos, episode["full_qpos"][step])
            np.testing.assert_array_equal(
                control["lowerbody_action"], episode["lowerbody_action"][step]
            )
            if time_step.last():
                break
        assert outer.episode_succeeded()
    finally:
        env.close()
