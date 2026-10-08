"""Degree of freedom description."""

from dataclasses import dataclass
from typing import Optional, Tuple

import mujoco


@dataclass(frozen=True)
class Dof:
    """Degree of freedom description."""

    joint_type: mujoco.mjtJoint
    axis: Tuple[int, int, int]
    joint_range: Optional[Tuple[float, float]] = None
    action_range: Optional[Tuple[float, float]] = None
    stiffness: float = 0
