"""Camera geometry shared by the sandbox client and the evaluator.

The cameras are pinhole cameras with a vertical field of view ``fovy_deg`` and
a world pose (``pos``, ``rot``) from ``camera_info()``; ``rot`` columns are the
camera x = right, y = up, z = backward axes (MuJoCo convention: the camera
looks along -z).

This module is copied into an agent sandbox as ``harness/geometry.py`` under
the ``tools`` interface, so it must keep working when imported as a top-level
module.
"""

from __future__ import annotations

import numpy as np


def pixel_to_ray(
    cam: dict, u: float, v: float, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """World-frame ray through one pixel of an image from ``cam``.

    Pixel indices are 0-based with (0, 0) at the top-left corner, u along the
    width; a pixel's centre is used, so the image centre is
    ``((width - 1) / 2, (height - 1) / 2)``. Equivalent to ROS
    image_geometry's ``projectPixelTo3dRay`` followed by a transform into the
    world frame.

    Args:
        cam: One entry of ``camera_info()`` (``fovy_deg``, ``pos``, ``rot``).
        u: Pixel column.
        v: Pixel row.
        width: Width the image was rendered at.
        height: Height the image was rendered at.

    Returns:
        ``(origin, direction)``: the camera position (3,) and a unit vector (3,).
    """
    f = (height / 2.0) / np.tan(np.deg2rad(float(cam["fovy_deg"])) / 2.0)
    d_cam = np.array(
        [
            (float(u) - (width - 1) / 2.0) / f,
            -(float(v) - (height - 1) / 2.0) / f,
            -1.0,
        ]
    )
    d = np.asarray(cam["rot"], dtype=np.float64).reshape(3, 3) @ d_cam
    return np.asarray(cam["pos"], dtype=np.float64).copy(), d / np.linalg.norm(d)
