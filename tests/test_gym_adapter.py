"""Tests for the gymnasium adapter (bigym.loco.gym_adapter).

The fast tests build a floating-base G1 env (no lower-body backend, so no
policy checkpoints or ONNX runtime are needed) and check that the adapter
is a well-behaved ``gymnasium.Env``. The slow tests build the real
official env (a canonical G1 task, and the paper's snippet), which needs
the GR00T-WBC runtime (vendored in ``bigym.loco.adapters._groot``, always
available).
"""

import inspect
import os

import numpy as np
import pytest

gymnasium = pytest.importorskip(
    "gymnasium",
    reason="gymnasium is not installed in this environment; the adapter "
    "targets the gymnasium API and cannot be exercised without it",
)

from bigym.loco.gym_adapter import (  # noqa: E402
    GymnasiumEnv,
    make_gym,
)


@pytest.fixture(scope="module")
def env():
    """Floating-base G1 reach env behind the gymnasium API."""
    gym_env = make_gym(
        "reach_target_dual",
        controller=None,
        episode_length=200,
        demo_down_sample_rate=25,
    )
    yield gym_env
    gym_env.close()


def test_is_a_gymnasium_env(env):
    assert isinstance(env, gymnasium.Env)
    assert env.unwrapped is env
    assert env.render_mode in env.metadata["render_modes"]
    assert env.metadata["bigym.substrate"]["task"] == "reach_target_dual"


def test_spaces_match_the_observations(env):
    assert isinstance(env.observation_space, gymnasium.spaces.Dict)
    assert set(env.observation_space.spaces) == {"rgb", "state"}

    rgb_space = env.observation_space["rgb"]
    state_space = env.observation_space["state"]
    assert rgb_space.dtype == np.uint8
    assert state_space.dtype == np.float32
    # (num_cameras, 3 * frame_stack, H, W)
    assert len(rgb_space.shape) == 4
    assert rgb_space.shape == tuple(env.rgb_observation_spec().shape)
    assert state_space.shape == tuple(env.low_dim_observation_spec().shape)

    obs, info = env.reset(seed=0)
    assert sorted(obs) == ["rgb", "state"]
    assert obs["rgb"].shape == rgb_space.shape
    assert obs["rgb"].dtype == np.uint8
    assert obs["state"].shape == state_space.shape
    assert obs["state"].dtype == np.float32
    assert obs in env.observation_space
    assert isinstance(info, dict)

    # The action space is the env's real (outer, normalized) one.
    assert isinstance(env.action_space, gymnasium.spaces.Box)
    assert env.action_space.shape == tuple(env.action_spec().shape)
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)
    assert env.action_space.shape[0] == env.wholebody_action_layout()["action_dim"]


def test_reset_info_carries_the_substrate_fingerprint(env):
    _obs, info = env.reset(seed=1)
    assert "substrate" in info
    fingerprint = info["substrate"]
    assert fingerprint is env.metadata["bigym.substrate"]
    assert fingerprint["robot_model"] == "g1_dex1"
    assert fingerprint["lowerbody_backend"] is None  # floating base
    assert (
        fingerprint["substrate_version"]
        == env.substrate_fingerprint()["substrate_version"]
    )
    assert info["success"] is False
    assert info["fell"] is False

    # Reset info reuses the same snapshot object. The direct fingerprint
    # is also stable across episode seeds (see the next test).
    _obs, info2 = env.reset(seed=2)
    assert info2["substrate"] is fingerprint


def test_substrate_fingerprint_is_stable_across_episodes(env):
    """The fingerprint is an env-instance identity, not a per-episode nonce.

    ``compiled_model_sha256`` hashes the compiled MuJoCo model, and task
    randomization rewrites body poses into that model on every reset. So
    ``BiGym`` pins the digest before the first reset and reports the pinned
    value from then on. ``GymnasiumEnv`` additionally snapshots the whole
    fingerprint once at construction because it is expensive; this test
    guards the property that makes that snapshot equal to the live value.
    """
    env.reset(seed=11)
    first = env.substrate_fingerprint()
    env.reset(seed=12)
    second = env.substrate_fingerprint()

    assert first["compiled_model_sha256"] == second["compiled_model_sha256"]
    assert first == second
    assert env.metadata["bigym.substrate"] == first


def test_five_random_steps(env):
    env.reset(seed=2)
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs in env.observation_space
        assert type(reward) is float
        assert type(terminated) is bool
        assert type(truncated) is bool
        assert isinstance(info, dict)
        assert set(info) >= {"success", "fell", "event_progress", "discount"}
        assert type(info["success"]) is bool
        assert type(info["fell"]) is bool
        # Terminal-only keys.
        assert ("termination" in info) == (terminated or truncated)
        if terminated or truncated:
            break


def test_out_of_range_actions_are_clipped_not_asserted(env):
    """A squashed-Gaussian policy emitting 1.0000001 must not crash the env."""
    env.reset(seed=3)
    action = np.full(env.action_space.shape, 1.0 + 1e-6, dtype=np.float32)
    obs, _reward, _terminated, _truncated, _info = env.step(action)
    assert obs in env.observation_space


def test_truncation_at_the_episode_budget(env):
    """episode_length=200 / down_sample 25 -> 8 outer steps, then truncation."""
    env.reset(seed=4)
    budget = 200 // 25
    for step in range(budget):
        _obs, _reward, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        if step < budget - 1:
            assert not (terminated or truncated)
    assert truncated and not terminated
    assert info["discount"] == 1.0
    assert info["termination"] == "timeout"
    assert info["truncation_reason"] == "time_limit"
    assert info["success"] is False


def test_render_returns_the_head_camera_frame(env):
    env.reset(seed=5)
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.dtype == np.uint8
    # HWC, and the camera shape (84x84) rather than the free-camera viewport.
    assert frame.shape == (84, 84, 3)


def test_gymnasium_env_checker(env):
    """Full gymnasium API conformance.

    ``check_env`` exercises seeded/unseeded reset determinism, step
    determinism (observations, reward and the whole info dict), the passive
    reset/step/render checkers and the space limits. If a future gymnasium
    version chokes on the Dict/uint8 observation space, fall back to the
    assertions in the other tests in this file rather than deleting this one.
    """
    from gymnasium.utils.env_checker import check_env

    check_env(env, skip_render_check=False)


def test_a_fallen_episode_the_task_finished_is_not_a_success(env):
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    outer = env.bigym_env.bigym

    env.reset(seed=0)
    _, _, _, _, info = env.step(action)
    assert not info["success"]

    env.reset(seed=0)
    outer._episode_succeeded = True
    _, _, _, _, info = env.step(action)
    assert info["success"] and not info["fell"]

    env.reset(seed=0)
    outer._episode_succeeded = True
    # The floating-base env has no controller to poll, so the latch stays as set.
    outer._episode_fell = True
    _, _, _, _, info = env.step(action)
    assert info["fell"] and not info["success"]
    assert env.bigym_env.episode_termination() == "fell"


def test_shaped_reward_is_not_a_success(monkeypatch):
    """Shaped reward above any size is not the task being done."""
    shaped = make_gym(
        "move_plate",
        controller=None,
        episode_length=50,
        demo_down_sample_rate=25,
        event_progress_enabled=True,
        event_reward_shaping_enabled=True,
        event_reward_progress_scale=1.0,
        event_reward_holding_bonus=0.1,
        event_reward_lift_bonus=0.2,
    )
    try:
        progress = shaped.bigym_env.bigym._event_progress
        # The plate lifted on every step, the task never done.
        monkeypatch.setattr(
            progress, "update", lambda inner, succeeded: np.float32(0.6)
        )
        shaped.reset(seed=0)
        action = np.zeros(shaped.action_space.shape, dtype=np.float32)
        rewards = []
        while True:
            _, reward, terminated, truncated, info = shaped.step(action)
            rewards.append(reward)
            assert not info["success"]
            if terminated or truncated:
                break
        assert min(rewards) == pytest.approx(0.9)
        assert info["termination"] == "timeout"
    finally:
        shaped.close()


@pytest.mark.slow
def test_paper_snippet():
    """The snippet printed in the paper: official env, spaces, 60 demos.

    The demo download needs the network, so ``get_demos(60)`` runs only
    with ``BIGYM_DATASET_TESTS=1``; otherwise its signature is checked.
    """
    from bigym.loco import make_gym as public_make_gym

    env = public_make_gym("move_plate")
    try:
        assert isinstance(env.observation_space, gymnasium.spaces.Dict)
        assert isinstance(env.action_space, gymnasium.spaces.Box)
        assert env.action_space.shape == (20,)  # move_plate: no pitch command
        parameters = inspect.signature(env.get_demos).parameters
        assert list(parameters) == ["num_demos", "only_successful", "decode_images"]
        assert parameters["num_demos"].default == -1
        assert parameters["decode_images"].default is True
        if os.environ.get("BIGYM_DATASET_TESTS"):
            assert len(env.get_demos(60)) == 60
    finally:
        env.close()


@pytest.mark.slow
def test_official_env_for_a_canonical_g1_task():
    """The real benchmark substrate (G1 + groot_wbc_g1) behind the gym API."""
    gym_env = make_gym("reach_target_dual")
    try:
        assert isinstance(gym_env, GymnasiumEnv)
        fingerprint = gym_env.metadata["bigym.substrate"]
        assert fingerprint["task"] == "reach_target_dual"
        assert fingerprint["robot_model"] == "g1_dex1"
        assert fingerprint["lowerbody_backend"] == "groot_wbc_g1"

        obs, info = gym_env.reset(seed=620000)
        assert obs in gym_env.observation_space
        assert info["substrate"] == fingerprint
        assert info["fell"] is False

        for _ in range(3):
            obs, reward, terminated, truncated, info = gym_env.step(
                np.zeros(gym_env.action_space.shape, dtype=np.float32)
            )
            assert obs in gym_env.observation_space
            assert type(reward) is float
            assert type(terminated) is bool
            assert type(truncated) is bool
            if terminated or truncated:
                break
    finally:
        gym_env.close()
