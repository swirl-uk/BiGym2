import copy

import numpy as np
import pytest

from bigym.loco.action_representation import (
    UpperDeltaAccumulator,
    convert_episode_action_representation,
    upper_delta_scale_from_metadata,
)
from bigym.loco.env import BiGym


def _metadata():
    return {
        "action_semantics": {
            "base_action_mode": "lowerbody_cmd",
            "upperbody_ik_conditioning": {
                "max_joint_speed_rad_s": 6.0,
            },
        },
        "control_step_seconds": 0.02,
        "outer_action_floating_dofs": [
            "pelvis_x",
            "pelvis_y",
            "pelvis_z",
            "pelvis_rz",
        ],
        "outer_limb_actuator_names": ["left_joint", "right_joint"],
        "action_stats": {
            "min": [-0.35, -0.25, 0.4, -0.5, -2.0, -1.0, 0.0, 0.0],
            "max": [0.35, 0.25, 1.0, 0.5, 2.0, 1.0, 1.0, 1.0],
        },
    }


def _normalize(raw, metadata):
    low = np.asarray(metadata["action_stats"]["min"], dtype=np.float32)
    high = np.asarray(metadata["action_stats"]["max"], dtype=np.float32)
    return ((raw - low) / (high - low + 1e-8) * 2.0 - 1.0).astype(np.float32)


def test_episode_upper_delta_preserves_base_gripper_and_source():
    metadata = _metadata()
    raw = np.asarray(
        [
            [0.1, -0.1, 0.75, 0.2, 0.20, -0.30, 1.0, 0.0],
            [0.2, -0.2, 0.80, 0.1, 0.26, -0.42, 0.0, 1.0],
            [0.0, 0.00, 0.70, 0.0, 0.20, -0.36, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    action = _normalize(raw, metadata)
    episode = {
        "action": action.copy(),
        "reward": np.zeros((len(action), 1), dtype=np.float32),
    }
    source = copy.deepcopy(episode)

    converted = convert_episode_action_representation(
        episode,
        metadata,
        action_representation="upper_delta",
    )

    np.testing.assert_array_equal(converted["action"][:, :4], action[:, :4])
    np.testing.assert_array_equal(converted["action"][:, 6:], action[:, 6:])
    np.testing.assert_allclose(
        converted["action"][:, 4:6],
        [[0.0, 0.0], [0.5, -1.0], [-0.5, 0.5]],
        atol=2e-6,
    )
    np.testing.assert_array_equal(episode["action"], source["action"])
    assert converted["reward"] is episode["reward"]


def test_upper_delta_accumulator_integrates_and_clips_joint_targets():
    accumulator = UpperDeltaAccumulator(0.12)
    accumulator.reset(np.asarray([0.20, -0.30], dtype=np.float32))
    target = accumulator.apply(
        np.asarray([0.5, -1.0], dtype=np.float32),
        np.asarray([-1.0, -0.4], dtype=np.float32),
        np.asarray([0.25, 1.0], dtype=np.float32),
    )
    np.testing.assert_allclose(target, [0.25, -0.4], atol=1e-7)
    np.testing.assert_array_equal(accumulator.target, target)


def test_loco_env_decodes_base_and_gripper_but_integrates_only_upper():
    metadata = _metadata()
    env = BiGym.__new__(BiGym)
    low, high = (
        np.asarray(metadata["action_stats"][key], dtype=np.float32)
        for key in ("min", "max")
    )
    env.set_action_stats(low, high)
    env.outer_action_floating_dofs = tuple(range(4))  # ty: ignore[invalid-assignment]
    env.outer_limb_actuator_names = ("left_joint", "right_joint")
    env._outer_action_low = low
    env._outer_action_high = high
    env.upper_delta_accumulator = UpperDeltaAccumulator(0.12)
    env.upper_delta_accumulator.reset(np.asarray([0.20, -0.30], dtype=np.float32))
    normalized = np.asarray(
        [0.5, -0.5, 0.0, 0.2, 0.5, -1.0, -1.0, 1.0],
        dtype=np.float32,
    )

    raw = env.denormalize_action(normalized)

    np.testing.assert_allclose(raw[:4], [0.175, -0.125, 0.7, 0.1], atol=1e-6)
    np.testing.assert_allclose(raw[4:6], [0.26, -0.42], atol=1e-6)
    np.testing.assert_allclose(raw[6:], [0.0, 1.0], atol=1e-6)


def test_upper_delta_scale_is_metadata_driven_and_explicitly_overridable():
    metadata = _metadata()
    assert upper_delta_scale_from_metadata(metadata) == pytest.approx(0.12)
    assert upper_delta_scale_from_metadata(metadata, 0.2) == pytest.approx(0.2)


def test_upper_delta_rejects_out_of_envelope_demo_step():
    metadata = _metadata()
    raw = np.asarray(
        [
            [0, 0, 0.7, 0, 0.0, 0.0, 0, 0],
            [0, 0, 0.7, 0, 0.2, 0.0, 0, 0],
        ],
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="exceeds"):
        convert_episode_action_representation(
            {"action": _normalize(raw, metadata)},
            metadata,
            action_representation="upper_delta",
        )
