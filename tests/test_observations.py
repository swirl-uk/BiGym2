"""Test observations."""

import mujoco
import numpy as np
import pytest
from numpy.testing import assert_allclose
from pyquaternion import Quaternion

from bigym.action_modes import JointPositionActionMode
from bigym.envs.reach_target import ReachTarget
from bigym.utils.observation_config import CameraConfig, ObservationConfig


@pytest.mark.parametrize(
    "rgb",
    [True, False],
)
@pytest.mark.parametrize(
    "depth",
    [True, False],
)
@pytest.mark.parametrize(
    "resolution",
    [(4, 4), (8, 8)],
)
class TestObservations:
    """Test observations."""

    def test_visual_observations_exist(self, rgb, depth, resolution):
        """Test visual observations exist."""
        camera_name = "head"
        env = ReachTarget(
            action_mode=JointPositionActionMode(floating_base=True, absolute=True),
            observation_config=ObservationConfig(
                cameras=[
                    CameraConfig(
                        name=camera_name,
                        rgb=rgb,
                        depth=depth,
                        resolution=resolution,
                    )
                ],
            ),
        )
        observation, _ = env.reset()
        assert (f"rgb_{camera_name}" in observation) == rgb
        if rgb:
            assert observation[f"rgb_{camera_name}"].shape == (3, *resolution)
        assert (f"depth_{camera_name}" in observation) == depth
        if depth:
            assert observation[f"depth_{camera_name}"].shape == resolution


def test_no_visual_observations():
    """Test no visual observations."""
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        observation_config=ObservationConfig(),
    )
    observation, _ = env.reset()
    assert not any(key.startswith(("rgb_", "depth_")) for key in observation)


def test_camera_pose_config():
    """Test camera pose is correctly set via observation config."""
    camera_name = "external"
    camera_pos = np.random.uniform(-1, 1, 3).tolist()
    camera_quat = Quaternion.random().elements.tolist()
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig(
                    name=camera_name,
                    pos=camera_pos,
                    quat=camera_quat,
                )
            ],
        ),
    )
    env.reset()
    camera = env._cameras_map[camera_name][1]
    assert_allclose(env.data.bind(camera).xpos, camera_pos)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, env.data.bind(camera).xmat)
    assert np.allclose(quat, camera_quat) or np.allclose(quat, np.negative(camera_quat))
