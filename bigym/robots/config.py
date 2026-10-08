"""Robot configs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from bigym.action_modes import PelvisDof
from bigym.const import HandSide
from bigym.utils.dof import Dof


@dataclass
class GripperConfig:
    """Configuration for a gripper embedded into the robot model.

    Attributes:
        actuators: A list of gripper's actuators names.
        range: Range of gripper control.
        body: Name of the gripper body in the robot model.
        pad_bodies: A list root pads bodies.
        pinch_site: Site between the pads; a site is added to ``body`` if None.
        discrete: Round control signal to min or max range value.
    """

    actuators: list[str]
    range: np.ndarray
    body: str
    pad_bodies: list[str] = field(default_factory=list)
    pinch_site: Optional[str] = None
    discrete: bool = True


@dataclass
class ArmConfig:
    """Configuration for a robot arm.

    Attributes:
        site: The wrist site of the arm.
        links: A list of body links of the hand.
    """

    site: str
    links: list[str]


@dataclass
class FloatingBaseConfig:
    """Configuration for a floating base."""

    dofs: dict[PelvisDof, Dof]
    delta_range_position: tuple[float, float]
    delta_range_rotation: tuple[float, float]
    offset_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reset_state: Optional[np.ndarray] = None
    # Physical base joints which are not exposed as actions or public
    # floating-base proprioception.  Lower-body policies still observe them
    # through their IMU sensors.
    passive_dofs: dict[str, Dof] = field(default_factory=dict)


@dataclass
class FullBodyConfig:
    """Configuration for full-body mode."""

    offset_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reset_state: Optional[np.ndarray] = None


@dataclass
class RobotConfig:
    """Configuration for a robot.

    Attributes:
        model: The path to the robot XML model.
        delta_range: Action range for delta position action mode.
        position_kp: Stiffness of actuators for absolute position action mode.
        pelvis_body: Name of the pelvis body element.
        full_body: Config for full-body mode.
        floating_base: Configuration for the floating base.
        gripper: Configuration for the robot's gripper.
        arms: Configuration for the robot's hands.
        actuators: Dictionary containing all actuators
            and indicating if it is used in floating action mode.
        cameras: List of available cameras.
    """

    model: Path
    delta_range: tuple[float, float]
    position_kp: float
    pelvis_body: str
    full_body: FullBodyConfig
    floating_base: FloatingBaseConfig
    gripper: GripperConfig
    arms: dict[HandSide, ArmConfig]
    actuators: dict[str, bool]
    cameras: list[str] = field(default_factory=list)
