"""Hand-controller poses in the virtual space."""

import numpy as np
from pyquaternion import Quaternion
from xr import Posef

from bigym.vr.viewer import Side
from bigym.vr.viewer.pyopenxr_to_mujoco_converter import (
    quaternion_from_pyopenxr,
    rotate_quaternion_yaw,
    rotate_vector_yaw,
    vector_from_pyopenxr,
)
from bigym.vr.viewer.xr_context import XRContextObject


def controller_pose(
    context: XRContextObject,
    side: Side,
    offset: Posef,
    yaw_offset: float = 0.0,
) -> tuple[np.ndarray, Quaternion]:
    """Position and orientation of one hand controller's aim pose.

    Args:
        context: XR context to read the input state from.
        side: Which hand.
        offset: Virtual space offset added to the position.
        yaw_offset: Virtual space yaw applied to position and orientation.
    """
    pose = context.input.state[side].pose_aim
    pos = (
        rotate_vector_yaw(vector_from_pyopenxr(pose.position), yaw_offset)
        + offset.position.as_numpy()
    )
    quat = rotate_quaternion_yaw(
        Quaternion(quaternion_from_pyopenxr(pose.orientation)), yaw_offset
    )
    return pos, quat
