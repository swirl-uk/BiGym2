"""G1 drawer ID initialization profile stays bounded and seed-reproducible."""

from __future__ import annotations

import numpy as np
import pytest

from bigym.envs.cupboards import DrawerTopCloseG1, DrawerTopOpenG1
from bigym.loco.config import EnvConfig
from bigym.loco.env import BiGym


class _CabinetStub:
    def __init__(self):
        self.state = None

    def set_state(self, state):
        self.state = np.asarray(state, dtype=np.float64).copy()


def _bare_env(env_cls, profile: str):
    env = object.__new__(env_cls)
    env._initialization_profile = profile
    env.cabinet_drawers = _CabinetStub()
    return env


def _sample(env_cls, seed: int):
    env = _bare_env(env_cls, "g1_id_v1")
    np.random.seed(seed)
    pos, quat = env._sample_reset_robot_pose()
    env._on_reset()
    return pos, quat, env.cabinet_drawers.state


def test_g1_drawer_id_reset_is_seed_reproducible():
    for env_cls in (DrawerTopOpenG1, DrawerTopCloseG1):
        first = _sample(env_cls, 42)
        second = _sample(env_cls, 42)
        for lhs, rhs in zip(first, second, strict=True):
            np.testing.assert_array_equal(lhs, rhs)


def test_g1_drawer_id_state_is_held_during_controller_warmup():
    env = _bare_env(DrawerTopCloseG1, "g1_id_v1")
    np.random.seed(42)
    env._on_reset()
    sampled = env.cabinet_drawers.state.copy()
    env.cabinet_drawers.state[-1] = 1.0  # emulate settle-time drawer motion
    env._on_reset_warmup_step()
    np.testing.assert_array_equal(env.cabinet_drawers.state, sampled)


def test_g1_drawer_id_reset_stays_in_frozen_bounds():
    for env_cls, state_bounds in (
        (DrawerTopOpenG1, (0.0, 0.15)),
        (DrawerTopCloseG1, (0.85, 1.0)),
    ):
        samples = [_sample(env_cls, seed) for seed in range(200)]
        positions = np.stack([sample[0] for sample in samples])
        quaternions = np.stack([sample[1] for sample in samples])
        states = np.stack([sample[2] for sample in samples])

        assert np.all((0.12 <= positions[:, 0]) & (positions[:, 0] <= 0.14))
        assert np.all((-0.05 <= positions[:, 1]) & (positions[:, 1] <= 0.05))
        np.testing.assert_array_equal(positions[:, 2], np.zeros(200))

        # The sampled quaternion is a pure world-z rotation [w, 0, 0, z].
        np.testing.assert_allclose(quaternions[:, 1:3], 0.0, atol=1e-12)
        yaw = 2.0 * np.arctan2(quaternions[:, 3], quaternions[:, 0])
        assert np.all(np.abs(yaw) <= np.deg2rad(5.0) + 1e-12)

        np.testing.assert_array_equal(states[:, :2], np.zeros((200, 2)))
        assert np.all(
            (state_bounds[0] <= states[:, 2]) & (states[:, 2] <= state_bounds[1])
        )

        # Guard against accidentally collapsing any distribution to a constant.
        assert np.ptp(positions[:, 0]) > 0.015
        assert np.ptp(positions[:, 1]) > 0.08
        assert np.ptp(yaw) > np.deg2rad(8.0)
        assert np.ptp(states[:, 2]) > 0.12


def test_upstream_profile_preserves_historical_drawer_resets():
    open_env = _bare_env(DrawerTopOpenG1, "upstream")
    close_env = _bare_env(DrawerTopCloseG1, "upstream")

    np.testing.assert_array_equal(
        open_env._sample_reset_robot_pose()[0], np.array([0.12, 0.0, 0.0])
    )
    open_env._on_reset()
    assert open_env.cabinet_drawers.state is None

    np.testing.assert_array_equal(
        close_env._sample_reset_robot_pose()[0], np.array([0.12, 0.0, 0.0])
    )
    close_env._on_reset()
    np.testing.assert_array_equal(
        close_env.cabinet_drawers.state, np.array([0.0, 0.0, 1.0])
    )


def test_g1_drawer_profile_rejects_unrelated_tasks():
    # Only the rejection and the names it points at are the contract; the
    # task whitelist in the message grows with every seeded task.
    with pytest.raises(ValueError, match="g1_id_v1.*move_plate"):
        BiGym("move_plate", EnvConfig(initialization_profile="g1_id_v1"))
