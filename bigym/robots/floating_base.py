"""Robot floating base."""

from typing import Optional

import mujoco
import numpy as np
from pyquaternion import Quaternion

from bigym.action_modes import PelvisDof
from bigym.const import TOLERANCE_ANGULAR, TOLERANCE_LINEAR
from bigym.robots.config import FloatingBaseConfig
from bigym.simulation import Simulation
from bigym.utils.physics_utils import is_target_reached

DOFS_ORDER = {PelvisDof.X: 1, PelvisDof.Y: 2, PelvisDof.Z: 3, PelvisDof.RZ: 6}
PASSIVE_DOFS_ORDER = {"pelvis_rx": 4, "pelvis_ry": 5}


class RobotFloatingBase:
    """Floating base of the robot to simplify control.

    Adds the base joints to the pelvis in the robot's spec and their position
    actuators to the scene's spec, where they precede the robot's own
    actuators.
    """

    DELTA_RANGE_POS: tuple[float, float] = (-0.01, 0.01)
    DELTA_RANGE_ROT: tuple[float, float] = (-0.05, 0.05)

    def __init__(
        self,
        config: FloatingBaseConfig,
        pelvis: mujoco.MjsBody,
        floating_dofs: list[PelvisDof],
        simulation: Simulation,
        namespace: str,
    ):
        """Add the base joints to ``pelvis``, a body of the robot attached under ``namespace``."""
        self._config = config
        self._pelvis = pelvis
        self.simulation = simulation
        self._offset_position = np.array(self._config.offset_position)
        self._position_actuators: list[Optional[mujoco.MjsActuator]] = [
            None,
            None,
            None,
        ]
        self._rotation_actuators: list[Optional[mujoco.MjsActuator]] = [
            None,
            None,
            None,
        ]
        self.joints: dict[str, mujoco.MjsJoint] = {}
        self._passive_joints: list[mujoco.MjsJoint] = []

        joint_specs = []
        for floating_dof in floating_dofs:
            if floating_dof not in self._config.dofs:
                raise ValueError(
                    f"Floating DOF {floating_dof} is not supported by this robot."
                )
            joint_specs.append(
                (
                    DOFS_ORDER[floating_dof],
                    floating_dof.value,
                    self._config.dofs[floating_dof],
                    True,
                )
            )
        for name, dof in self._config.passive_dofs.items():
            joint_specs.append((PASSIVE_DOFS_ORDER[name], name, dof, False))

        for _, name, dof, actuated in sorted(joint_specs):
            joint = self._pelvis.add_joint(
                type=dof.joint_type, name=name, axis=dof.axis
            )
            if dof.joint_range:
                joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
                joint.range[:] = dof.joint_range
            else:
                joint.limited = mujoco.mjtLimited.mjLIMITED_FALSE

            if not actuated:
                self._passive_joints.append(joint)
                continue

            actuator = simulation.spec.add_actuator(
                name=namespace + name,
                target=namespace + name,
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
            )
            actuator.set_to_position(kp=dof.stiffness)
            if dof.action_range:
                actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
                actuator.ctrlrange[:] = dof.action_range
            else:
                actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
            self.joints[actuator.name] = joint

            axis_index = int(np.argmax(dof.axis))
            if dof.joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                self._position_actuators[axis_index] = actuator
            else:
                self._rotation_actuators[axis_index] = actuator

        self._accumulated_actions: np.ndarray = np.zeros(len(self.all_actuators))
        self._last_action: np.ndarray = np.zeros(len(self.all_actuators))

    def joint(self, actuator: mujoco.MjsActuator) -> mujoco.MjsJoint:
        """The joint ``actuator`` drives."""
        return self.joints[actuator.name]

    def reset(self, position: np.ndarray, quaternion: np.ndarray):
        """Set position and orientation of the floating base."""
        self._accumulated_actions *= 0
        self._last_action *= 0

        self._set_position(position)
        self._set_quaternion(quaternion)
        for passive_joint in self._passive_joints:
            bound_joint = self.simulation.data.bind(passive_joint)
            bound_joint.qpos = 0.0
            bound_joint.qvel = 0.0
            bound_joint.qacc = 0.0

    def get_action_bounds(self) -> list[tuple[float, float]]:
        """Get action bounds of all actuators."""
        bounds = []
        for actuator in self._position_actuators:
            if actuator:
                bounds.append(self._config.delta_range_position)
        for actuator in self._rotation_actuators:
            if actuator:
                bounds.append(self._config.delta_range_rotation)
        return bounds

    def set_control(self, control: np.ndarray):
        """Set control of all actuators."""
        self._accumulated_actions += self._last_action
        self._last_action = control.copy()
        for actuator, value in zip(self.all_actuators, control, strict=True):
            bound_actuator = self.simulation.data.bind(actuator)
            bound_actuator.ctrl += value
            if actuator.ctrllimited == mujoco.mjtLimited.mjLIMITED_TRUE:
                bound_actuator.ctrl = np.clip(bound_actuator.ctrl, *actuator.ctrlrange)

    @property
    def is_target_reached(self) -> bool:
        """Check if the target state of all actuators is reached."""
        for actuator in self._position_actuators:
            if actuator and not is_target_reached(
                self.simulation, actuator, self.joint(actuator), TOLERANCE_LINEAR
            ):
                return False
        for actuator in self._rotation_actuators:
            if actuator and not is_target_reached(
                self.simulation, actuator, self.joint(actuator), TOLERANCE_ANGULAR
            ):
                return False
        return True

    @property
    def dof_amount(self) -> int:
        """Get number of actuated DOF."""
        return len(self.all_actuators)

    @property
    def qpos(self) -> np.ndarray:
        """Get positions of actuated joints."""
        return np.array(
            [
                self.simulation.data.bind(self.joint(actuator)).qpos.item()
                for actuator in self.all_actuators
            ],
            np.float32,
        )

    @property
    def qvel(self) -> np.ndarray:
        """Get velocities of actuated joints."""
        return np.array(
            [
                self.simulation.data.bind(self.joint(actuator)).qvel.item()
                for actuator in self.all_actuators
            ],
            np.float32,
        )

    @property
    def get_accumulated_actions(self) -> np.ndarray:
        """Get accumulated actions since last reset."""
        return np.array(self._accumulated_actions, np.float32)

    @property
    def all_actuators(self) -> list[mujoco.MjsActuator]:
        """Get all actuators."""
        return [a for a in self._position_actuators if a] + [
            a for a in self._rotation_actuators if a
        ]

    @property
    def position_actuators(self) -> list[Optional[mujoco.MjsActuator]]:
        """Get all position actuators."""
        return self._position_actuators

    @property
    def rotation_actuators(self) -> list[Optional[mujoco.MjsActuator]]:
        """Get all rotation actuators."""
        return self._rotation_actuators

    @property
    def passive_joints(self) -> list[mujoco.MjsJoint]:
        """Get physical floating-base joints which have no action channel."""
        return list(self._passive_joints)

    @property
    def _pelvis_z(self) -> float:
        if self._position_actuators[2]:
            return float(
                self.simulation.data.bind(
                    self.joint(self._position_actuators[2])
                ).qpos.item()
            )
        return float(self.simulation.model.bind(self._pelvis).pos[2])

    def _set_position(self, position: np.ndarray):
        position = position + self._offset_position
        self._set_value(True, position)

    def _set_quaternion(self, quaternion: np.ndarray):
        rotation = np.flip(np.array(Quaternion(quaternion).yaw_pitch_roll))
        self._set_value(False, rotation)

    def _set_value(self, position: bool, values: np.ndarray):
        actuators = self._position_actuators if position else self._rotation_actuators
        assert len(values) == len(actuators)
        for i, value, actuator in zip(
            range(len(values)), values, actuators, strict=True
        ):
            if actuator:
                bound_joint = self.simulation.data.bind(self.joint(actuator))
                bound_joint.qpos = value
                bound_joint.qvel *= 0
                bound_joint.qacc *= 0
                self.simulation.data.bind(actuator).ctrl = value
            elif position:
                pelvis = self.simulation.model.bind(self._pelvis)
                pelvis.pos[i] = value
                pelvis.sameframe = 0
