"""Load/unload cutlery to/from dishwasher."""

from abc import ABC

import numpy as np
from pyquaternion import Quaternion

from bigym.bigym_env import BiGymEnv
from bigym.const import PRESETS_PATH
from bigym.envs.dishwasher_cups import G1_DISHWASHER_SPAWN
from bigym.envs.props.cabintets import BaseCabinet
from bigym.envs.props.cutlery import Fork, Knife
from bigym.envs.props.dishwasher import Dishwasher
from bigym.envs.props.holders import CutleryTray
from bigym.envs.props.kitchenware import Mug
from bigym.utils.env_utils import get_random_sites
from bigym.utils.physics_utils import zyz_euler_to_quaternion


class _DishwasherCutleryEnv(BiGymEnv, ABC):
    """Base cutlery environment."""

    _PRESET_PATH = PRESETS_PATH / "counter_dishwasher.yaml"
    _CUTLERY = [Knife, Fork]

    RESET_ROBOT_POS = np.array([0, -0.6, 0])

    def _initialize_env(self):
        self.dishwasher = self._preset.get_props(Dishwasher)[0]
        self.cutlery = [item_cls(self.simulation) for item_cls in self._CUTLERY]

    def _fail(self) -> bool:
        if super()._fail():
            return True
        for item in self.cutlery:
            if item.is_colliding(self.floor):
                return True
        return False

    def _on_reset(self):
        self.dishwasher.set_state(door=1, bottom_tray=1, middle_tray=0)


class _DishwasherUnloadCutleryEnv(_DishwasherCutleryEnv):
    """Base unload cutlery from dishwasher task."""

    _SITES_SLICE = -2

    _CUTLERY_OFFSET_POS = np.array([0, 0, 0.1])
    _CUTLERY_BOUNDS_ANGLE = np.deg2rad(30)
    _CUTLERY_SPAWN_ROT = Quaternion(axis=[1, 0, 0], degrees=90)

    def _on_reset(self):
        super()._on_reset()
        # The site poses are read after the trays moved.
        self.simulation.forward()
        sites = self.dishwasher.basket.site_sets[0]
        sites = get_random_sites(sites, len(self.cutlery), segment=self._SITES_SLICE)
        for site, item in zip(sites, self.cutlery, strict=True):
            item_pos = self.data.bind(site).xpos + self._CUTLERY_OFFSET_POS
            angle = np.random.uniform(
                -self._CUTLERY_BOUNDS_ANGLE, self._CUTLERY_BOUNDS_ANGLE
            )
            item_quat = self._CUTLERY_SPAWN_ROT * Quaternion(
                axis=[0, 1, 0], angle=angle
            )
            item.set_quaternion(item_quat.elements, True)
            item.set_position(item_pos, True)


class DishwasherUnloadCutlery(_DishwasherUnloadCutleryEnv):
    """Unload cutlery from dishwasher task."""

    _TRAY_POS = np.array([0.65, -0.6, 0.86])
    _TRAY_BOUNDS = np.array([0.05, 0.05, 0])
    _TRAY_ROT = np.array([0, 0, -np.pi / 2])

    def _initialize_env(self):
        super()._initialize_env()
        self.tray = CutleryTray(self.simulation)

    def _success(self) -> bool:
        for item in self.cutlery:
            if not item.is_colliding(self.tray):
                return False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(item, side):
                    return False
        return True

    def _on_reset(self):
        super()._on_reset()
        offset = np.random.uniform(-self._TRAY_BOUNDS, self._TRAY_BOUNDS)
        self.tray.set_position(self._TRAY_POS + offset)
        self.tray.set_quaternion(zyz_euler_to_quaternion(self._TRAY_ROT))


class DishwasherUnloadCutleryLong(_DishwasherUnloadCutleryEnv):
    """Unload cutlery from dishwasher to drawer task."""

    _PRESET_PATH = PRESETS_PATH / "counter_dishwasher_cutlery_cabinet.yaml"
    _CUTLERY = [Fork]
    _TOLERANCE = 0.1

    def _initialize_env(self):
        super()._initialize_env()
        self.cutlery_cabinet = self._preset.get_props(BaseCabinet)[-1]
        self.tray = self._preset.get_props(CutleryTray)[0]

    def _success(self) -> bool:
        if not np.allclose(self.dishwasher.get_state(), 0, atol=self._TOLERANCE):
            return False
        if not np.allclose(self.cutlery_cabinet.get_state(), 0, atol=self._TOLERANCE):
            return False
        for item in self.cutlery:
            if not item.is_colliding(self.tray):
                return False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(item, side):
                    return False
        return True


class DishwasherLoadCutlery(_DishwasherCutleryEnv):
    """Load cutlery to dishwasher task."""

    _MUG_POS = np.array([0.65, -0.6, 0.86])
    _MUG_BOUNDS = np.array([0.05, 0.05, 0])
    _MUG_BOUNDS_ANGLE = np.deg2rad(90)

    _BASKET_OFFSET_POS = np.array([0, 0, 0.15])
    _CUTLERY_SPAWN_ROT = Quaternion(axis=[1, 0, 0], degrees=90)
    _CUTLERY_OFFSET_ANGLE = np.deg2rad(90)
    _CUTLERY_OFFSET_ANGLE_RANGE = np.deg2rad(5)
    _CUTLERY_SPAWN_OFFSET = 0.02

    def _initialize_env(self):
        super()._initialize_env()
        self.mug = Mug(self.simulation, False)

    def _success(self) -> bool:
        for item in self.cutlery:
            if not item.is_colliding(self.dishwasher.basket.colliders):
                return False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(item, side):
                    return False
        return True

    def _on_reset(self):
        super()._on_reset()
        mug_angle = np.random.uniform(-self._MUG_BOUNDS_ANGLE, self._MUG_BOUNDS_ANGLE)
        self.mug.set_quaternion(zyz_euler_to_quaternion(np.array([0, 0, mug_angle])))
        offset = np.random.uniform(-self._MUG_BOUNDS, self._MUG_BOUNDS)
        mug_pos = self._MUG_POS + offset
        self.mug.set_position(mug_pos, True)
        for i, item in enumerate(self.cutlery):
            item.set_quaternion(self._CUTLERY_SPAWN_ROT.elements, True)
            offset_angle = self._CUTLERY_OFFSET_ANGLE * i
            offset_angle += np.random.uniform(
                -self._CUTLERY_OFFSET_ANGLE_RANGE, self._CUTLERY_OFFSET_ANGLE_RANGE
            )
            item_offset = np.array([np.cos(offset_angle), np.sin(offset_angle), 0])
            item_offset *= self._CUTLERY_SPAWN_OFFSET
            item_pos = mug_pos.copy() + self._BASKET_OFFSET_POS + item_offset
            item.set_position(item_pos, True)


class DishwasherLoadCutleryG1(DishwasherLoadCutlery):
    """DishwasherLoadCutlery with a collision-free G1 start."""

    RESET_ROBOT_POS = G1_DISHWASHER_SPAWN


class DishwasherUnloadCutleryG1(DishwasherUnloadCutlery):
    """DishwasherUnloadCutlery with a collision-free G1 start."""

    RESET_ROBOT_POS = G1_DISHWASHER_SPAWN


class DishwasherUnloadCutleryLongG1(DishwasherUnloadCutleryLong):
    """UnloadCutleryLong under the G1 embodiment.

    Unlike the cups/plates long variants the target is the cutlery tray in a
    base-cabinet drawer (z 0.66), not the wall cabinet, so the upstream
    preset is kept.
    """

    RESET_ROBOT_POS = G1_DISHWASHER_SPAWN
