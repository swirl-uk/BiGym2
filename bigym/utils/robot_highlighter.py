"""Highlight actuated joints of the robot."""

import warnings

import mujoco
import numpy as np

from bigym import scene
from bigym.robots.robot import Robot


class RobotHighlighter:
    """Highlight actuated joints of the robot."""

    def __init__(self, robot: Robot):
        """Collect the geoms moved by each limb actuator and each gripper."""
        self._model = robot.simulation.model
        self._actuated_geoms: list[list[mujoco.MjsGeom]] = [
            list(joint.parent.geoms) for joint in robot.limb_joints
        ]
        for gripper in robot.grippers.values():
            self._actuated_geoms.append(scene.descendants(gripper.body, "geoms"))
        self._original_colors = [
            [self._model.bind(geom).rgba.copy() for geom in geoms]
            for geoms in self._actuated_geoms
        ]

    def reset(self):
        """Clean highlight."""
        for geoms, colors in zip(
            self._actuated_geoms, self._original_colors, strict=True
        ):
            for geom, color in zip(geoms, colors, strict=True):
                self._model.bind(geom).rgba = color

    def highlight(self, index: int, tint: np.ndarray | None = None):
        """Highlight joint by index."""
        allowed_range = range(len(self._actuated_geoms))
        if index not in allowed_range:
            warnings.warn(f"Index {index} out of {allowed_range} range.", stacklevel=2)
            return
        if tint is None:
            tint = np.ones(4)
        for geom, color in zip(
            self._actuated_geoms[index], self._original_colors[index], strict=True
        ):
            self._model.bind(geom).rgba = color + tint
