"""Action modes for the BiGym robots."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Optional

import numpy as np
from gymnasium import spaces

from bigym.const import TOLERANCE_ANGULAR
from bigym.utils.physics_utils import (
    is_target_reached,
)

if TYPE_CHECKING:
    from bigym.robots.floating_base import RobotFloatingBase
    from bigym.robots.robot import Robot
    from bigym.simulation import Simulation


class TargetStateNotReachedWarning(Warning):
    """Warning raised when the target state is not reached within the maximum steps."""

    pass


class PelvisDof(Enum):
    """Set of floating base DOFs."""

    X = "pelvis_x"
    Y = "pelvis_y"
    Z = "pelvis_z"
    RZ = "pelvis_rz"
    # Torso-pitch command channel (pitch-capable lower-body backends): an ABSOLUTE
    # waist-pitch target in rad, + = lean forward. Riding in the action
    # vector keeps recorded demos replayable; tasks opt in, existing demos
    # without this slot are unaffected.
    RY = "pelvis_ry"


DEFAULT_DOFS = [PelvisDof.X, PelvisDof.Y, PelvisDof.RZ]


class ActionMode(ABC):
    """Base action mode class used for controlling the robot."""

    def __init__(
        self,
        floating_base: bool = True,
        floating_dofs: Optional[list[PelvisDof]] = None,
    ):
        """Init.

        :param floating_base: If True, then legs are frozen, and the robot base
            controlled by positional actuators.
            If False, then user has full control of legs (i.e. for whole-body control).
        :param floating_dofs: Set of floating DOFs. By default, it is: [X, Y, RZ].
        """
        self._floating_base = floating_base
        self._floating_dofs = DEFAULT_DOFS if floating_dofs is None else floating_dofs

        # Will be assigned later
        self._simulation: Optional[Simulation] = None
        self._robot: Optional[Robot] = None

    def bind_robot(self, robot: Robot, simulation: Simulation):
        """Bind action mode to robot."""
        self._robot = robot
        self._simulation = simulation

    @property
    def _bound_robot(self) -> Robot:
        assert self._robot is not None, "Action mode is not bound to a robot."
        return self._robot

    @property
    def _bound_simulation(self) -> Simulation:
        assert self._simulation is not None, "Action mode is not bound to a robot."
        return self._simulation

    @property
    def _bound_floating_base(self) -> RobotFloatingBase:
        # A robot builds its floating base iff its action mode is floating.
        floating_base = self._bound_robot.floating_base
        assert floating_base is not None
        return floating_base

    @property
    def floating_base(self) -> bool:
        """Is floating base enabled."""
        return self._floating_base

    @property
    def floating_dofs(self) -> list[PelvisDof]:
        """Set of floating DOFs."""
        return self._floating_dofs

    @abstractmethod
    def action_space(
        self, action_scale: float, seed: Optional[int] = None
    ) -> spaces.Box:
        """The action space for this action mode."""
        pass

    @abstractmethod
    def step(self, action: np.ndarray):
        """Apply the control command and step the physics.

        Note: This function has the responsibility of stepping the simulation.

        :param action: The entire action passed to the action mode.
        """
        pass

    @abstractmethod
    def reset(self, reset_state: np.ndarray):
        """Reset state of the robot accordingly to the action mode.

        :param reset_state: Target reset state of robot actuators.
        """
        pass


class JointPositionActionMode(ActionMode):
    """Control all joints through joint position.

    Allows to control joint positions, supporting both absolute and delta positions.
    For absolute control, set 'absolute' to True. If the floating base is enabled,
    only delta position control is applied to it.

    Notes:
        - `block_until_reached` does not guarantee reaching the target position because
          the target position could be unreachable due to collisions.
        - Joints of the `floating_base` are always controlled in delta position mode.
    """

    # Settle budget for ``block_until_reached``: 200 physics steps at the
    # nominal 2 ms timestep. Robots whose XML compiles to a finer timestep
    # (the G1: 1 ms) get the same 0.4 s of simulated time, not half of it.
    MAX_STEPS = 200
    _NOMINAL_TIMESTEP = 0.002

    def __init__(
        self,
        absolute: bool = False,
        block_until_reached: bool = False,
        floating_base: bool = True,
        floating_dofs: Optional[list[PelvisDof]] = None,
    ):
        """See base.

        :param absolute: Use absolute or delta joint positions.
        :param block_until_reached: Continue stepping until the target
            position is reached or the step threshold is exceeded.
        """
        super().__init__(
            floating_base=floating_base,
            floating_dofs=floating_dofs,
        )
        self.absolute = absolute
        self.block_until_reached = block_until_reached

    def action_space(
        self, action_scale: float, seed: Optional[int] = None
    ) -> spaces.Box:
        """See base."""
        robot = self._bound_robot
        bounds = []
        if self.floating_base:
            action_bounds = self._bound_floating_base.get_action_bounds()
            action_bounds = [np.array(b) * action_scale for b in action_bounds]
            bounds.extend(action_bounds)
        for actuator in robot.limb_actuators:
            action_bounds = np.array(
                robot.get_limb_control_range(actuator, self.absolute)
            )
            action_bounds *= 1 if self.absolute else action_scale
            bounds.append(action_bounds)
        for _, gripper in robot.grippers.items():
            bounds.append(gripper.range)
        bounds = np.array(bounds).copy().astype(np.float32)
        low, high = bounds.T
        return spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
            seed=seed,
        )

    def step(self, action: np.ndarray):
        """See base."""
        robot = self._bound_robot
        simulation = self._bound_simulation
        if self.floating_base:
            floating_base = self._bound_floating_base
            base_action = action[: floating_base.dof_amount]
            action = action[floating_base.dof_amount :]
            floating_base.set_control(base_action)
        for i, actuator in enumerate(robot.limb_actuators):
            actuator = simulation.data.bind(actuator)
            actuator.ctrl = action[i] if self.absolute else actuator.ctrl + action[i]
        gripper_actions = action[-len(robot.grippers) :]
        for side, action in zip(robot.grippers, gripper_actions, strict=True):
            robot.grippers[side].set_control(action)

        if self.block_until_reached:
            self._step_until_reached()
        else:
            simulation.step()

    def reset(self, reset_state: np.ndarray):
        """See base."""
        robot = self._bound_robot
        simulation = self._bound_simulation
        if len(reset_state) != len(robot.limb_actuators):
            raise ValueError(
                f"Mismatch between reset_state length "
                f"({len(reset_state)}) "
                f"and number of actuators ({len(robot.limb_actuators)}). "
                f"Ensure reset_state matches the actuators count in the model."
            )
        for value, actuator, joint in zip(
            reset_state, robot.limb_actuators, robot.limb_joints, strict=True
        ):
            bound_joint = simulation.data.bind(joint)
            bound_joint.qpos = value
            bound_joint.qvel *= 0
            bound_joint.qacc *= 0
            simulation.data.bind(actuator).ctrl = value

    def _step_until_reached(self):
        """Step physics until the target position is reached."""
        simulation = self._bound_simulation
        steps_counter = 0
        timestep = float(simulation.model.opt.timestep)
        max_steps = max(
            1, int(round(self.MAX_STEPS * self._NOMINAL_TIMESTEP / timestep))
        )
        while True:
            simulation.step()
            steps_counter += 1
            if self._is_target_state_reached() or steps_counter >= max_steps:
                if steps_counter >= max_steps:
                    warnings.warn(
                        f"Failed to reach target state in {max_steps} steps!",
                        TargetStateNotReachedWarning,
                        stacklevel=2,
                    )
                break

    def _is_target_state_reached(self):
        if self.floating_base:
            if not self._bound_floating_base.is_target_reached:
                return False
        robot = self._bound_robot
        for actuator, joint in zip(
            robot.limb_actuators, robot.limb_joints, strict=True
        ):
            if not is_target_reached(
                self._bound_simulation, actuator, joint, TOLERANCE_ANGULAR
            ):
                return False
        return True
