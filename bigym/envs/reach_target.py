"""Set of reach target tasks."""

from abc import ABC
from dataclasses import dataclass, field

import mujoco
import numpy as np
from gymnasium import spaces

from bigym.bigym_env import BiGymEnv
from bigym.const import HandSide
from bigym.robots.robot import Robot
from bigym.simulation import Simulation
from bigym.utils.physics_utils import set_body_position


@dataclass
class TargetConfig:
    """Target Config."""

    target_hands: list[HandSide]
    reset_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    size: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.05, 0.05]))
    color_default: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 1]))
    color_highlight: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 1]))


class Target:
    """Target sphere."""

    def __init__(self, simulation: Simulation, robot: Robot, config: TargetConfig):
        """Add the target body to the simulation's spec."""
        self.simulation = simulation
        self._robot = robot
        self._config = config
        self.body = simulation.spec.worldbody.add_body()
        self.geom = self.body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=config.size.tolist(),
            rgba=config.color_default.tolist(),
            group=1,
            density=1000,
            mass=0,
            contype=0,
            conaffinity=0,
        )

    def reset_position(self, offset: np.ndarray):
        """Reset position of the target."""
        set_body_position(
            self.simulation.model, self.body, self._config.reset_position + offset
        )

    def get_position(self) -> np.ndarray:
        """World position of the target."""
        return self.simulation.data.bind(self.body).xpos.copy()

    def distance(self, pos: np.ndarray) -> float:
        """Get distance to target."""
        return float(np.linalg.norm(self.get_position() - pos))

    def is_reached(self, tolerance: float) -> bool:
        """Check if target is reached."""
        for side in self._config.target_hands:
            if side not in self._robot.grippers:
                continue
            hand_pos = self._robot.get_hand_pos(side)
            is_reached = self.distance(hand_pos) <= tolerance
            self.set_highlight(is_reached)
            if is_reached:
                return True
        return False

    def set_highlight(self, highlight: bool):
        """Toggle target highlight."""
        self.simulation.model.bind(self.geom).rgba = (
            self._config.color_highlight if highlight else self._config.color_default
        )


class _ReachTargetEnv(BiGymEnv, ABC):
    """Base reach target environment."""

    TARGET_CONFIGS = [
        TargetConfig(
            target_hands=[HandSide.LEFT, HandSide.RIGHT],
            reset_position=np.array([0.5, 0, 1]),
            color_default=np.array([0.3, 0, 0, 1]),
            color_highlight=np.array([1, 0, 0, 1]),
        )
    ]

    # x sampling is tighter than y/z: far+low draws (x>0.55 with z<0.95) are
    # standing-reachable but account for most operator fumbles in VR demos.
    # x in [center-0.05, center+0.05] removes that corner and keeps the reach
    # tasks saturable as pure-manipulation benchmarks.
    POSITION_BOUNDS = np.array([0.05, 0.1, 0.1])
    TOLERANCE = 0.1

    @property
    def reach_tolerance(self) -> float:
        """Success radius in meters (env override wins over the class constant).

        The G1 protocol pins 0.05 (= the target sphere's radius, i.e.
        the pinch-center site must be INSIDE the ball — "ball between the
        finger pads"); the class default 0.1 leaves a 5 cm shell around
        the ball, which reads as success on a fingertip graze.
        """
        override = getattr(self, "_reach_tolerance", None)
        return self.TOLERANCE if override is None else float(override)

    def _initialize_env(self):
        self.targets: list[Target] = []
        for config in self.TARGET_CONFIGS:
            self.targets.append(Target(self.simulation, self.robot, config))

    def _on_reset(self):
        for target in self.targets:
            offset = np.random.uniform(-self.POSITION_BOUNDS, self.POSITION_BOUNDS)
            target.reset_position(offset)
            target.set_highlight(False)

    def _success(self) -> bool:
        for target in self.targets:
            if not target.is_reached(self.reach_tolerance):
                return False
        return True

    def _on_step(self):
        """Highlight spheres even in fast mode."""
        self._success()


class ReachTarget(_ReachTargetEnv):
    """Reach the target with either left or right wrist."""

    def _get_task_privileged_obs_space(self):
        return {
            "target_position": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            )
        }

    def _get_task_privileged_obs(self):
        return {
            "target_position": np.array(
                self.targets[0].get_position(), np.float32
            ).copy()
        }


class ReachTargetSingle(ReachTarget):
    """Reach the target with specific wrist."""

    TARGET_CONFIGS = [
        TargetConfig(
            target_hands=[HandSide.LEFT],
            reset_position=np.array([0.5, 0, 1]),
            color_default=np.array([0.3, 0, 0, 1]),
            color_highlight=np.array([1, 0, 0, 1]),
        )
    ]


class ReachTargetDual(_ReachTargetEnv):
    """Reach 2 targets, one with each arm."""

    TARGET_CONFIGS = [
        TargetConfig(
            target_hands=[HandSide.LEFT],
            reset_position=np.array([0.5, 0.2, 1]),
            color_default=np.array([0.3, 0, 0, 1]),
            color_highlight=np.array([1, 0, 0, 1]),
        ),
        TargetConfig(
            target_hands=[HandSide.RIGHT],
            reset_position=np.array([0.5, -0.2, 1]),
            color_default=np.array([0, 0.3, 0, 1]),
            color_highlight=np.array([0, 1, 0, 1]),
        ),
    ]

    def _get_task_privileged_obs_space(self):
        return {
            "target_position_left": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
            "target_position_right": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
        }

    def _get_task_privileged_obs(self):
        return {
            "target_position_left": np.array(
                self.targets[0].get_position(), np.float32
            ).copy(),
            "target_position_right": np.array(
                self.targets[1].get_position(), np.float32
            ).copy(),
        }


# G1 workspace variants: the upstream tasks sample targets at world
# z = 1.0 +- 0.1 (chest height for a 1.8 m humanoid), at/above the 1.3 m G1's
# shoulder line. These variants lower the sampling box into the G1's
# comfortable manipulation band and pull it slightly closer. The upstream
# tasks remain runnable with the G1 as a deliberate high-reach stress variant.
G1_TARGET_Z = 0.8
G1_TARGET_X = 0.45


class ReachTargetG1(ReachTarget):
    """ReachTarget (either hand) with the target box in the G1 workspace."""

    TARGET_CONFIGS = [
        TargetConfig(
            target_hands=[HandSide.LEFT, HandSide.RIGHT],
            reset_position=np.array([G1_TARGET_X, 0, G1_TARGET_Z]),
            color_default=np.array([0.3, 0, 0, 1]),
            color_highlight=np.array([1, 0, 0, 1]),
        )
    ]


class ReachTargetSingleG1(ReachTargetSingle):
    """ReachTargetSingle (left hand) with the target box in the G1 workspace."""

    TARGET_CONFIGS = [
        TargetConfig(
            target_hands=[HandSide.LEFT],
            reset_position=np.array([G1_TARGET_X, 0, G1_TARGET_Z]),
            color_default=np.array([0.3, 0, 0, 1]),
            color_highlight=np.array([1, 0, 0, 1]),
        )
    ]


class ReachTargetDualG1(ReachTargetDual):
    """ReachTargetDual (one target per hand) in the G1 workspace."""

    TARGET_CONFIGS = [
        TargetConfig(
            target_hands=[HandSide.LEFT],
            reset_position=np.array([G1_TARGET_X, 0.2, G1_TARGET_Z]),
            color_default=np.array([0.3, 0, 0, 1]),
            color_highlight=np.array([1, 0, 0, 1]),
        ),
        TargetConfig(
            target_hands=[HandSide.RIGHT],
            reset_position=np.array([G1_TARGET_X, -0.2, G1_TARGET_Z]),
            color_default=np.array([0, 0.3, 0, 1]),
            color_highlight=np.array([0, 1, 0, 1]),
        ),
    ]
