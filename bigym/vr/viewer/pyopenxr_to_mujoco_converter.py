"""Converts vectors and quaternions from pyopenxr to mujoco space."""

import numpy as np
from pyquaternion import Quaternion
from xr import Quaternionf, Vector3f
from xr.utils import Matrix4x4f


def matrix_as_flat16(matrix: Matrix4x4f) -> np.ndarray:
    """The matrix as a flat column-major 16-vector.

    The callers index columns via flat offsets ([8:11] = 3rd column).
    """
    return np.asarray(matrix.as_numpy()).ravel(order="F")


def vector_from_pyopenxr(xr_vector: Vector3f | np.ndarray) -> np.ndarray:
    """Convert pyopenxr vector to mujoco space.

    To convert from pyopenxr to mujoco, a 90-degree rotation along the X-axis
    has to be applied, i.e., multiplication by the following offset matrix:

    | 1 0  0 |
    | 0 0 -1 |
    | 0 1  0 |

    mujoco_vector = [xr_vector[0], -xr_vector[2], xr_vector[1]]
    """
    if isinstance(xr_vector, Vector3f):
        xr_vector = xr_vector.as_numpy()
    return np.array([xr_vector[0], -xr_vector[2], xr_vector[1]])


def rotate_vector_yaw(vector: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a MuJoCo-space vector around the vertical axis."""
    vector = np.asarray(vector, dtype=np.float64)
    c, s = np.cos(float(yaw)), np.sin(float(yaw))
    out = vector.copy()
    out[0] = c * vector[0] - s * vector[1]
    out[1] = s * vector[0] + c * vector[1]
    return out


def rotate_quaternion_yaw(quaternion: Quaternion, yaw: float) -> Quaternion:
    """Apply a MuJoCo-world yaw rotation to an orientation."""
    quaternion = Quaternion(quaternion)
    if float(yaw) == 0.0:
        return quaternion
    return Quaternion(axis=[0, 0, 1], radians=float(yaw)) * quaternion


def quaternion_from_pyopenxr(xr_quaternion: Quaternionf) -> np.ndarray:
    """Convert pyopenxr quaternion to mujoco space."""
    quaternion = Quaternion(
        xr_quaternion.w, xr_quaternion.x, xr_quaternion.y, xr_quaternion.z
    )
    quaternion = Quaternion(axis=[1, 0, 0], degrees=90).rotate(quaternion)
    return quaternion.elements


def camera_axes_from_pyopenxr(
    xr_quaternion: Quaternionf,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert pyopenxr quaternion to mujoco forward and up axes."""
    orientation = matrix_as_flat16(Matrix4x4f.create_from_quaternion(xr_quaternion))
    forward = vector_from_pyopenxr(orientation[8:11])
    up = vector_from_pyopenxr(orientation[4:7])
    return forward, up
