"""BiGym Robot."""

from abc import ABC, abstractmethod
from typing import Optional

import mujoco
import numpy as np

from bigym import scene
from bigym.action_modes import ActionMode, JointPositionActionMode
from bigym.const import HandSide
from bigym.robots.config import RobotConfig
from bigym.robots.floating_base import RobotFloatingBase
from bigym.robots.gripper import Gripper
from bigym.simulation import Simulation
from bigym.utils.physics_utils import (
    Colliders,
    critical_damping,
    set_body_position,
    set_body_quaternion,
    set_position_servo,
)


class Robot(ABC):
    """A robot model attached to the scene, driven through its action mode.

    The robot edits its own spec (floating base, actuators, gains, grippers)
    before attaching it to the scene and keeps the spec elements it controls.
    """

    def __init__(self, action_mode: ActionMode, simulation: Simulation):
        """Build the robot from its config and attach it to the simulation's spec."""
        self._action_mode = action_mode
        self.simulation = simulation
        model = scene.load(self.config.model)
        self.namespace = scene.namespace(simulation.spec, model)
        self._on_loaded(model)
        self._grippers = self._get_grippers(model)
        self._body = scene.attach(simulation.spec, model)
        self._joints = scene.descendants(self._body, "joints")
        if self._floating_base:
            # Floating base joints follow the parsed ones, as in the published layout.
            floating = self._floating_base.passive_joints
            floating += self._floating_base.joints.values()
            names = {joint.name for joint in floating}
            self._joints = [joint for joint in self._joints if joint.name not in names]
            self._joints += self._floating_base.joints.values()
        if not self._action_mode.floating_base:
            self._body.add_freejoint(name=self._body.name)
        self._action_mode.bind_robot(self, simulation)

    @property
    @abstractmethod
    def config(self) -> RobotConfig:
        """Get robot config."""
        pass

    @property
    def action_mode(self) -> ActionMode:
        """Get action mode."""
        return self._action_mode

    @property
    def body(self) -> mujoco.MjsBody:
        """The frame body the robot is attached under."""
        return self._body

    @property
    def pelvis(self) -> mujoco.MjsBody:
        """Get pelvis."""
        return self._pelvis

    @property
    def limb_actuators(self) -> list[mujoco.MjsActuator]:
        """Get all limb actuators."""
        return self._limb_actuators

    @property
    def limb_joints(self) -> list[mujoco.MjsJoint]:
        """The joints of :attr:`limb_actuators`, in the same order."""
        return self._limb_joints

    @property
    def joints(self) -> list[mujoco.MjsJoint]:
        """The robot joints reported in :attr:`qpos` and :attr:`qvel`."""
        return self._joints

    @property
    def grippers(self) -> dict[HandSide, Gripper]:
        """Get robot grippers."""
        return self._grippers

    @property
    def floating_base(self) -> Optional[RobotFloatingBase]:
        """Get floating base."""
        return self._floating_base

    @property
    def cameras(self) -> list[mujoco.MjsCamera]:
        """The robot cameras, in the order of ``config.cameras``."""
        return self._cameras

    @property
    def qpos(self) -> np.ndarray:
        """Get positions of all joints."""
        model, data = self.simulation.model, self.simulation.data
        ids = [joint.id for joint in self._joints]
        return data.qpos[model.jnt_qposadr[ids]].astype(np.float32)

    @property
    def qpos_grippers(self) -> np.ndarray:
        """Get current state of gripper actuators."""
        return np.array(
            [gripper.qpos for gripper in self.grippers.values()], np.float32
        )

    @property
    def qpos_actuated(self) -> np.ndarray:
        """Get positions of actuated joints."""
        qpos = []
        if self._floating_base:
            qpos.extend(self._floating_base.qpos)
        data = self.simulation.data
        qpos.extend(float(data.bind(joint).qpos.item()) for joint in self._limb_joints)
        qpos.extend(self.qpos_grippers)
        return np.array(qpos, np.float32)

    @property
    def qvel(self) -> np.ndarray:
        """Get velocities of all joints."""
        model, data = self.simulation.model, self.simulation.data
        ids = [joint.id for joint in self._joints]
        return data.qvel[model.jnt_dofadr[ids]].astype(np.float32)

    @property
    def qvel_actuated(self) -> np.ndarray:
        """Get velocities of actuated joints."""
        qvel = []
        if self._floating_base:
            qvel.extend(self._floating_base.qvel)
        data = self.simulation.data
        qvel.extend(float(data.bind(joint).qvel.item()) for joint in self._limb_joints)
        qvel.extend(gripper.qvel for gripper in self._grippers.values())
        return np.array(qvel, np.float32)

    def get_hand_pos(self, side: HandSide) -> np.ndarray:
        """Get position of robot hand site."""
        if side not in self.config.arms.keys():
            return np.zeros(3)
        return self._grippers[side].wrist_position

    def is_gripper_holding_object(self, other: Colliders, side: HandSide) -> bool:
        """Check if the gripper is holding an object."""
        if side not in self.config.arms.keys():
            return False
        return self._grippers[side].is_holding_object(other)

    def reset(self, position: np.ndarray, orientation: np.ndarray):
        """Reset robot."""
        self._set_pose(position, orientation)
        for gripper in self._grippers.values():
            gripper.reset()
        reset_state = (
            self.config.floating_base.reset_state
            if self._action_mode.floating_base
            else self.config.full_body.reset_state
        )
        if reset_state is not None and len(reset_state) == len(self.limb_actuators):
            self._action_mode.reset(reset_state)

    def _set_pose(self, position: np.ndarray, orientation: np.ndarray):
        """Instantly set pose of the robot pelvis."""
        if self._floating_base:
            self._floating_base.reset(position, orientation)
        else:
            model, data = self.simulation.model, self.simulation.data
            set_body_position(
                model, self._pelvis, position + self.config.full_body.offset_position
            )
            set_body_quaternion(model, data, self._pelvis, orientation)

    def get_limb_control_range(
        self, actuator: mujoco.MjsActuator, absolute: bool
    ) -> np.ndarray:
        """Get control range of the limb actuator."""
        if not absolute:
            return np.array(self.config.delta_range)
        return self.simulation.model.bind(actuator).ctrlrange

    def _get_grippers(self, model: mujoco.MjSpec) -> dict[HandSide, Gripper]:
        return {
            side: Gripper(
                side,
                self._wrist_sites[side],
                self.config.gripper,
                self.simulation,
                model,
            )
            for side in self.config.arms
        }

    def _on_loaded(self, model: mujoco.MjSpec):
        self._cameras = [model.camera(name) for name in self.config.cameras]
        self._wrist_sites = {
            side: model.site(arm.site) for side, arm in self.config.arms.items()
        }
        self._pelvis = model.body(self.config.pelvis_body)
        for joint in list(self._pelvis.joints):
            model.delete(joint)
        self._pelvis.pos = np.zeros(3)
        self._pelvis.quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Joints of new position actuators, with the stiffness to damp critically
        stiffened: list[tuple[mujoco.MjsJoint, float]] = []

        self._floating_base: Optional[RobotFloatingBase] = None
        if self._action_mode.floating_base:
            self._floating_base = RobotFloatingBase(
                self.config.floating_base,
                self._pelvis,
                self._action_mode.floating_dofs,
                self.simulation,
                self.namespace,
            )
            stiffened.extend(
                (self._floating_base.joint(actuator), actuator.gainprm[0])
                for actuator in self._floating_base.all_actuators
            )

        limbs: list[tuple[mujoco.MjsActuator, mujoco.MjsJoint]] = []
        for actuator in list(model.actuators):
            name = actuator.name or actuator.target
            if name not in self.config.actuators:
                continue
            joint = model.joint(actuator.target)
            if self._floating_base and not self.config.actuators[name]:
                model.delete(actuator)
                model.delete(joint)
                continue
            if not isinstance(self._action_mode, JointPositionActionMode):
                continue
            if actuator.biastype == mujoco.mjtBias.mjBIAS_NONE:
                # A motor becomes a position actuator, appended after the others.
                model.delete(actuator)
                actuator = model.add_actuator(
                    name=name, target=joint.name, trntype=mujoco.mjtTrn.mjTRN_JOINT
                )
                set_position_servo(actuator, self.config.position_kp)
                actuator.ctrlrange = joint.range
                stiffened.append((joint, self.config.position_kp))
            limbs.append((actuator, joint))

        joint_order = scene.descendants(model.worldbody, "joints")
        limbs.sort(key=lambda limb: joint_order.index(limb[1]))
        self._limb_actuators = [actuator for actuator, _ in limbs]
        self._limb_joints = [joint for _, joint in limbs]

        initial = model.compile()
        for joint, stiffness in stiffened:
            joint.damping[0] = critical_damping(stiffness, joint, initial)
