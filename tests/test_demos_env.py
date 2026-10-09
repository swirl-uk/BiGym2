"""Demo observations through the env: stacking parity and the gym view.

The Hub-backed loader hands the env raw (unstacked) frames; the env must
turn them into exactly what ``reset``/``step`` would have produced. The
network-gated test at the bottom exercises the real dataset.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from bigym.loco import make


@pytest.fixture(scope="module")
def stacked_env():
    """Floating-base G1 env with frame stacking, no lower-body runtime needed."""
    env = make(
        "reach_target_dual",
        controller=None,
        episode_length=50,
        demo_down_sample_rate=25,
        frame_stack=3,
        camera_shape=(16, 16),
    )
    yield env
    env.close()


def test_stacking_parity_with_live_observations(stacked_env):
    bigym = stacked_env.bigym
    raw = []
    bigym.reset(seed=1)
    raw.append(bigym.inner_env.get_observation())
    for _ in range(4):
        bigym.step(np.zeros(bigym.action_space.shape, dtype=np.float32))
        raw.append(bigym.inner_env.get_observation())

    # Reference: the env's own path, from a fresh stack.
    bigym._low_dim_obses.clear()
    for frames in bigym._frames.values():
        frames.clear()
    reference = [bigym._extract_obs(obs) for obs in raw]

    # Demo path: the same frames handed over as one replay-format episode.
    episode = {
        "rgb_obs": np.stack(
            [
                np.stack([obs[f"rgb_{cam}"] for cam in bigym.config.camera_keys])
                for obs in raw
            ]
        ),
        "low_dim_obs": np.stack([bigym._extract_low_dim_from_obs(obs) for obs in raw]),
    }
    rgb, low_dim = bigym.demo_observations(episode)
    assert rgb.shape[1:] == tuple(bigym.rgb_observation_spec().shape)
    assert low_dim.shape[1:] == tuple(bigym.low_dim_observation_spec().shape)
    for t, ref in enumerate(reference):
        np.testing.assert_array_equal(rgb[t], ref["rgb_obs"])
        np.testing.assert_array_equal(low_dim[t], ref["low_dim_obs"])


def test_episode_to_timesteps_layout(stacked_env):
    bigym = stacked_env.bigym
    cams = len(bigym.config.camera_keys)
    length = 6
    episode = {
        "rgb_obs": np.zeros((length, cams, 3, 16, 16), dtype=np.uint8),
        "low_dim_obs": np.zeros(
            (length, bigym.low_dim_raw_observation_spec().shape[0]), np.float32
        ),
        "action": np.zeros((length, bigym.action_space.shape[0]), np.float32),
        "reward": np.array([[0.0]] * (length - 1) + [[1.0]], np.float32),
        "discount": np.array([[1.0]] * (length - 1) + [[0.0]], np.float32),
    }
    timesteps, successful = bigym._episode_to_timesteps(episode)
    assert successful and len(timesteps) == length
    assert timesteps[0].first() and timesteps[-1].last()
    assert all(ts.mid() for ts in timesteps[1:-1])
    assert timesteps[-1].reward == 1.0 and timesteps[-1].discount == 0.0
    assert timesteps[0].demo == 1.0  # no demo column: falls back to the success flag


def test_gym_get_demos_applies_low_dim_normalization(monkeypatch):
    from bigym.loco import env as env_module
    from bigym.loco.gym_adapter import GymnasiumEnv

    env = make(
        "reach_target_dual",
        controller=None,
        episode_length=50,
        demo_down_sample_rate=25,
        camera_shape=(16, 16),
        normalize_low_dim_obs=True,
    )
    bigym = env.bigym
    cams = len(bigym.config.camera_keys)
    dim = bigym.low_dim_raw_observation_spec().shape[0]
    length = 3
    low_dim = np.tile(np.array([[10.0], [20.0], [30.0]], np.float32), (1, dim))
    episode = {
        "rgb_obs": np.zeros((length, cams, 3, 16, 16), dtype=np.uint8),
        "low_dim_obs": low_dim,
        "action": np.zeros((length, bigym.action_space.shape[0]), np.float32),
        "reward": np.array([[0.0], [0.0], [1.0]], np.float32),
        "discount": np.array([[1.0], [1.0], [0.0]], np.float32),
    }
    monkeypatch.setattr(
        env_module.demo_episodes,
        "load_task_episodes",
        lambda *args, **kwargs: ([episode], dict(bigym.action_stats)),
    )
    try:
        state = GymnasiumEnv(env).get_demos(1)[0]["obs"]["state"]
    finally:
        env.close()
    expected = (low_dim - low_dim.mean(0)) / (low_dim.std(0) + 1e-8)
    np.testing.assert_allclose(state, expected, rtol=1e-5)


@pytest.mark.skipif(
    not os.environ.get("BIGYM_DATASET_TESTS"),
    reason="set BIGYM_DATASET_TESTS=1 to download real demonstrations",
)
def test_real_dataset_through_the_paper_api():
    from bigym.loco import make_gym

    env = make_gym("reach_target_single")
    demos = env.get_demos(2)
    assert len(demos) == 2
    demo = demos[0]
    assert demo["obs"]["rgb"].shape[1:] == env.observation_space["rgb"].shape
    assert demo["obs"]["state"].shape[1:] == env.observation_space["state"].shape
    steps = len(demo["action"])
    assert demo["obs"]["rgb"].shape[0] == steps + 1
    assert demo["reward"].shape == demo["terminated"].shape == (steps,)
    assert env.action_space.contains(demo["action"][0])
    assert demo["success"] and demo["reward"][-1] > 0
    obs, _ = env.reset(seed=620000)
    env.step(demo["action"][0])
    env.close()
