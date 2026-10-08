"""Manipulation tasks."""

from abc import ABC

import mujoco
import numpy as np
from pyquaternion import Quaternion

from bigym.bigym_env import BiGymEnv
from bigym.const import PRESETS_PATH
from bigym.envs.props.cabintets import BaseCabinet
from bigym.envs.props.cutlery import Spoon
from bigym.envs.props.items import Cube
from bigym.envs.props.kitchenware import Mug
from bigym.utils.env_utils import get_random_points_on_plane
from bigym.utils.physics_utils import set_body_position


class _ManipulationEnv(BiGymEnv, ABC):
    """Base manipulation environment."""

    _PRESET_PATH = PRESETS_PATH / "cabinet.yaml"

    def _initialize_env(self):
        self.cabinet = self._preset.get_props(BaseCabinet)[0]


class FlipCup(_ManipulationEnv):
    """Flip cup upside-up task."""

    _CUP_POS = np.array([0.8, 0, 1])
    _CUP_ROT_X = np.deg2rad(180)
    _CUP_ROT_Z = np.deg2rad(180)
    _CUP_STEP = 0.1
    _CUP_POS_EXTENTS = np.array([0.1, 0.25])
    _CUP_POS_BOUNDS = np.array([0.03, 0.03, 0])
    _CUP_ROT_BOUNDS = np.deg2rad(30)

    _TOLERANCE = np.deg2rad(5)

    def _initialize_env(self):
        super()._initialize_env()
        self.cup = Mug(self.simulation)

    def _success(self) -> bool:
        up = np.array([0, 0, 1])
        cup_up = Quaternion(self.cup.get_quaternion()).rotate(up)
        angle_to_up = np.arccos(np.clip(np.dot(cup_up, up), -1.0, 1.0))
        if angle_to_up > self._TOLERANCE:
            return False
        if not self.cup.is_colliding(self.cabinet.counter):
            return False
        for side in self.robot.grippers:
            if self.robot.is_gripper_holding_object(self.cup, side):
                return False
        return True

    def _on_reset(self):
        spawn_point = get_random_points_on_plane(
            1,
            self._CUP_POS,
            self._CUP_POS_EXTENTS,
            self._CUP_STEP,
            self._CUP_POS_BOUNDS,
        )[0]
        self.cup.set_position(spawn_point, True)
        quat = Quaternion(axis=[1, 0, 0], angle=self._CUP_ROT_X)
        angle = np.random.uniform(-self._CUP_ROT_BOUNDS, self._CUP_ROT_BOUNDS)
        quat *= Quaternion(axis=[0, 0, 1], angle=self._CUP_ROT_Z + angle)
        self.cup.set_quaternion(quat.elements, True)


class FlipCutlery(_ManipulationEnv):
    """Flip cutlery item task."""

    _CUP_POS = np.array([0.8, 0, 0.86])
    _CUP_ROT_Z = np.deg2rad(180)
    _CUP_STEP = 0.1
    _CUP_POS_EXTENTS = np.array([0.1, 0.25])
    _CUP_POS_BOUNDS = np.array([0.03, 0.03, 0])
    _CUP_ROT_BOUNDS = np.deg2rad(180)

    _SPOON_OFFSET = np.array([0, 0, 0.15])
    _SPOON_QUAT = Quaternion(axis=[1, 0, 0], degrees=90)

    _TOLERANCE = np.deg2rad(50)

    def _initialize_env(self):
        super()._initialize_env()
        self.cup = Mug(self.simulation, kinematic=False)
        self.spoon = Spoon(self.simulation)

    def _success(self) -> bool:
        down = np.array([0, 0, -1])
        fwd = np.array([0, 1, 0])
        spoon_fwd = Quaternion(self.spoon.get_quaternion()).rotate(fwd)
        angle_to_up = np.arccos(np.clip(np.dot(spoon_fwd, down), -1.0, 1.0))
        if angle_to_up > self._TOLERANCE:
            return False
        if not self.spoon.is_colliding(self.cup):
            return False
        # Upstream only refused a held cup, so a spoon still pinched inside
        # the mug counted as success; the spoon must be released too.
        for side in self.robot.grippers:
            if self.robot.is_gripper_holding_object(self.spoon, side):
                return False
            if self.robot.is_gripper_holding_object(self.cup, side):
                return False
        return True

    def _on_reset(self):
        spawn_point = get_random_points_on_plane(
            1,
            self._CUP_POS,
            self._CUP_POS_EXTENTS,
            self._CUP_STEP,
            self._CUP_POS_BOUNDS,
        )[0]
        self.cup.set_position(spawn_point)
        angle = np.random.uniform(-self._CUP_ROT_BOUNDS, self._CUP_ROT_BOUNDS)
        quat = Quaternion(axis=[0, 0, 1], angle=self._CUP_ROT_Z + angle)
        self.cup.set_quaternion(quat.elements)

        self.spoon.set_position(spawn_point + self._SPOON_OFFSET, True)
        self.spoon.set_quaternion(self._SPOON_QUAT.elements, True)


class StackBlocks(BiGymEnv):
    """Stack blocks in the correct area of the table."""

    _PRESET_PATH = PRESETS_PATH / "stack_blocks.yaml"

    _NUM_BLOCKS = 3
    _BLOCKS_POS = np.array([0.7, 0, 1])
    _BLOCKS_POS_EXTENTS = np.array([0.2, 0.5])
    _BLOCKS_STEP = 0.15
    _BLOCKS_POS_BOUNDS = np.array([0.05, 0.05, 0])
    _BLOCKS_ROT_BOUNDS = np.deg2rad([0, 0, 180])

    _TARGET_SIZE = np.array([0.05, 0.05, 0.001])
    _TARGET_COLOR = np.array([0.3, 0.8, 0.3, 1.0])
    _TARGET_POS = np.array([1.4, 0, 0.95])
    _TARGET_POS_BOUNDS = np.array([0.05, 0.2, 0.0])

    def _initialize_env(self):
        self.blocks = [Cube(self.simulation) for _ in range(self._NUM_BLOCKS)]
        self.target = self.spec.worldbody.add_body()
        self.target_collider = self.target.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=self._TARGET_SIZE.tolist(),
            rgba=self._TARGET_COLOR.tolist(),
            group=1,
            density=1000,
        )

    def _on_reset(self):
        points = get_random_points_on_plane(
            len(self.blocks),
            self._BLOCKS_POS,
            self._BLOCKS_POS_EXTENTS,
            self._BLOCKS_STEP,
        )
        for block, point in zip(self.blocks, points, strict=True):
            block.set_pose(
                point,
                position_bounds=self._BLOCKS_POS_BOUNDS,
                rotation_bounds=self._BLOCKS_ROT_BOUNDS,
            )
        offset = np.random.uniform(-self._TARGET_POS_BOUNDS, self._TARGET_POS_BOUNDS)
        set_body_position(self.model, self.target, self._TARGET_POS + offset)

    # Cube edge is 0.05 m (props/cube/cube.xml half-size 0.025). A block
    # resting on another sits one edge higher; allow tilt/settling slack.
    _STACK_MIN_RISE = 0.03
    _STACK_MAX_XY_OFFSET = 0.05

    def _success(self) -> bool:
        blocks_sorted = sorted(self.blocks, key=lambda b: b.get_position()[2])
        if not blocks_sorted[0].is_colliding(self.target_collider):
            return False
        if not blocks_sorted[1].is_colliding(blocks_sorted[0]):
            return False
        if not blocks_sorted[2].is_colliding(blocks_sorted[1]):
            return False
        # Pairwise contact alone (the upstream predicate) also passes a
        # horizontal chain of touching cubes; require each block to sit above
        # and over the previous one.
        for lower, upper in zip(blocks_sorted, blocks_sorted[1:], strict=False):
            p_low = np.asarray(lower.get_position())
            p_up = np.asarray(upper.get_position())
            if p_up[2] - p_low[2] < self._STACK_MIN_RISE:
                return False
            if np.linalg.norm(p_up[:2] - p_low[:2]) > self._STACK_MAX_XY_OFFSET:
                return False
        for block in self.blocks:
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(block, side):
                    return False
        return True

    def _fail(self) -> bool:
        if super()._fail():
            return True
        for block in self.blocks:
            if block.is_colliding(self.floor):
                return True
        return False


# G1 workspace variants: the 0.84 m counters sit below the G1 shoulder line
# (1.05 m), so the spawn moves forward to put the objects 0.45-0.60 m from the
# settled pelvis (groot_wbc_g1 settles ~0.07 behind the spawn), and the
# flip-task object band is pulled in from x=0.8 to x=0.72 so its far corner
# stays reachable standing. StackBlocks also sinks both tables to a ~0.68 m
# top and lowers the block/target spawns with them; the cross-table walk
# stays part of the task.


class FlipCupG1(FlipCup):
    """FlipCup with the spawn and cup band in the G1 workspace."""

    # 0.24, not closer: the resting hands sit at z=0.845, exactly counter
    # height, so the fingertips (pelvis +0.32) must clear the 0.6 counter
    # face or the settle pushes the robot off its stance (seen at 0.30).
    RESET_ROBOT_POS = np.array([0.24, 0, 0])
    _CUP_POS = np.array([0.72, 0, 1])


class FlipCutleryG1(FlipCutlery):
    """FlipCutlery with the spawn and cup band in the G1 workspace."""

    RESET_ROBOT_POS = np.array([0.24, 0, 0])  # see FlipCupG1
    _CUP_POS = np.array([0.72, 0, 0.86])


class StackBlocksG1(StackBlocks):
    """StackBlocks with G1-height tables and edge-accessible work zones."""

    _PRESET_PATH = PRESETS_PATH / "stack_blocks_g1.yaml"
    # Groot settles ~0.07 m behind the requested spawn. At x=0.18 the home
    # fingertips retain ~9 cm clearance from the first table's x=0.40 edge.
    RESET_ROBOT_POS = np.array([0.18, 0, 0])
    # One shallow row near the exposed edge. Including +/-5 cm reset jitter,
    # centres stay x=0.525..0.625 and |y|<=0.275 instead of reaching x=0.825.
    _BLOCKS_POS = np.array([0.575, 0, 0.75])
    _BLOCKS_POS_EXTENTS = np.array([0.10, 0.35])
    # Put the goal 10-18 cm inside the far exposed edge of table two and
    # keep jitter small enough that every episode has the same approach.
    _TARGET_POS = np.array([1.56, 0, 0.70])
    _TARGET_POS_BOUNDS = np.array([0.04, 0.08, 0.0])
