"""Robot Gripper."""

import mujoco
import numpy as np

from bigym import scene
from bigym.const import HandSide
from bigym.robots.config import GripperConfig
from bigym.simulation import Simulation
from bigym.utils.physics_utils import Colliders, has_collided_collections


class Gripper:
    """A gripper body of the robot, with its actuators and pads."""

    _NORMAL_RANGE = (0, 1)
    _ROUND_DECIMALS = 1

    def __init__(
        self,
        side: HandSide,
        wrist_site: mujoco.MjsSite,
        config: GripperConfig,
        simulation: Simulation,
        model: mujoco.MjSpec,
    ):
        """Find the gripper in the robot ``model`` and add its pinch site if needed."""
        self._side = side
        self._wrist_site = wrist_site
        self._config = config
        self.simulation = simulation

        self._body = model.body(config.body)
        if config.pinch_site:
            self._pinch_site = model.site(config.pinch_site)
        else:
            self._pinch_site = self._body.add_site(
                size=[0.01, 0.01, 0.01], rgba=[1, 1, 1, 1], group=5
            )
        self._actuators: list[mujoco.MjsActuator] = []
        self._actuated_joints: list[mujoco.MjsJoint] = []
        for actuator in model.actuators:
            if (actuator.name or actuator.target) in config.actuators:
                self._actuators.append(actuator)
                self._actuated_joints.append(model.joint(actuator.target))
        self._pad_geoms = [
            geom
            for pad in config.pad_bodies
            for geom in scene.descendants(model.body(pad), "geoms")
            if geom.contype == 1 and geom.conaffinity == 1
        ]

    @property
    def body(self) -> mujoco.MjsBody:
        """Get gripper body."""
        return self._body

    @property
    def wrist_site(self) -> mujoco.MjsSite:
        """Get wrist site."""
        return self._wrist_site

    @property
    def pinch_site(self) -> mujoco.MjsSite:
        """The site between the finger pads."""
        return self._pinch_site

    @property
    def pinch_position(self) -> np.ndarray:
        """Get position of the pinch site."""
        return self.simulation.data.bind(self._pinch_site).xpos.copy()

    @property
    def wrist_position(self) -> np.ndarray:
        """Get position of the wrist site."""
        return self.simulation.data.bind(self._wrist_site).xpos.copy()

    @property
    def range(self) -> np.ndarray:
        """Get gripper control range."""
        return self._config.range

    @property
    def actuators(self) -> list[mujoco.MjsActuator]:
        """Get list of gripper actuators."""
        return self._actuators

    @property
    def actuated_joints(self) -> list[mujoco.MjsJoint]:
        """The joints of :attr:`actuators`, in the same order."""
        return self._actuated_joints

    @property
    def pad_geoms(self) -> list[mujoco.MjsGeom]:
        """The collider geoms of the finger pads."""
        return self._pad_geoms

    @property
    def qpos(self) -> float:
        """Get average qpos of actuated joints."""
        positions = [
            np.interp(
                self.simulation.data.bind(joint).qpos.item(),
                self.simulation.model.bind(joint).range,
                self._config.range,
            )
            for joint in self._actuated_joints
        ]
        return np.round(np.average(positions), decimals=self._ROUND_DECIMALS)

    @property
    def qvel(self) -> float:
        """Get current velocity of gripper actuators."""
        velocities = [
            self.simulation.data.bind(joint).qvel.item()
            for joint in self._actuated_joints
        ]
        return np.round(np.average(velocities), decimals=self._ROUND_DECIMALS)

    def is_holding_object(self, other: Colliders) -> bool:
        """Check if gripper is holding object."""
        return has_collided_collections(self.simulation, self._pad_geoms, other)

    def reset(self) -> None:
        """Clear per-episode controller state; called from ``Robot.reset``.

        The base gripper is stateless. Subclasses that filter their targets
        across control steps override this so a reset never inherits the
        previous episode's targets.
        """

    def set_control(self, ctrl: float):
        """Set state of the gripper."""
        ctrl = np.interp(ctrl, self._config.range, self._NORMAL_RANGE)
        if self._config.discrete:
            ctrl = np.round(ctrl)
        ctrl = np.interp(ctrl, self._NORMAL_RANGE, self._config.range)
        for actuator in self._actuators:
            ctrlrange = self.simulation.model.bind(actuator).ctrlrange
            ctrl = np.interp(ctrl, self._config.range, ctrlrange)
            self.simulation.data.bind(actuator).ctrl = ctrl
