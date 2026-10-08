"""Load/unload cups to/from dishwasher."""

from abc import ABC

import numpy as np
from pyquaternion import Quaternion

from bigym.bigym_env import BiGymEnv
from bigym.const import PRESETS_PATH
from bigym.envs.props.cabintets import BaseCabinet, WallCabinet
from bigym.envs.props.dishwasher import Dishwasher
from bigym.envs.props.kitchenware import Mug
from bigym.utils.env_utils import get_random_sites


class _DishwasherCupsEnv(BiGymEnv, ABC):
    """Base cups environment."""

    RESET_ROBOT_POS = np.array([0, -0.6, 0])

    _PRESET_PATH = PRESETS_PATH / "counter_dishwasher.yaml"
    _CUPS_COUNT = 2

    def _initialize_env(self):
        self.dishwasher = self._preset.get_props(Dishwasher)[0]
        self.cabinets = self._preset.get_props(BaseCabinet)
        self.cups = [Mug(self.simulation) for _ in range(self._CUPS_COUNT)]

    def _fail(self) -> bool:
        if super()._fail():
            return True
        for cup in self.cups:
            if cup.is_colliding(self.floor):
                return True
        return False

    def _on_reset(self):
        self.dishwasher.set_state(door=1, bottom_tray=0, middle_tray=1)


class DishwasherUnloadCups(_DishwasherCupsEnv):
    """Unload cups from dishwasher task."""

    _SITES_STEP = 3
    _SITES_SLICE = 3

    _CUPS_ROT_X = np.deg2rad(180)
    _CUPS_ROT_Z = np.deg2rad(90)
    _CUPS_ROT_BOUNDS = np.deg2rad(5)
    _CUPS_POS = np.array([0, -0.05, 0.05])
    _CUPS_STEP = np.array([0.115, 0, 0])

    def _success(self) -> bool:
        for cup in self.cups:
            if not any([cup.is_colliding(cabinet) for cabinet in self.cabinets]):
                return False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(cup, side):
                    return False
        return True

    def _on_reset(self):
        super()._on_reset()
        # The site poses are read after the trays moved.
        self.simulation.forward()
        sites = self.dishwasher.tray_middle.site_sets[0]
        sites = get_random_sites(
            sites, len(self.cups), self._SITES_STEP, self._SITES_SLICE
        )
        for site, cup in zip(sites, self.cups, strict=True):
            quat = Quaternion(axis=[1, 0, 0], angle=self._CUPS_ROT_X)
            angle = np.random.uniform(-self._CUPS_ROT_BOUNDS, self._CUPS_ROT_BOUNDS)
            quat *= Quaternion(axis=[0, 0, 1], angle=self._CUPS_ROT_Z + angle)
            cup.set_quaternion(quat.elements, True)
            pos = self.data.bind(site).xpos.copy()
            pos += self._CUPS_POS
            cup.set_position(pos, True)


class DishwasherUnloadCupsLong(DishwasherUnloadCups):
    """Unload cup from dishwasher in wall cabinet task."""

    _PRESET_PATH = PRESETS_PATH / "counter_dishwasher_wall_cabinet.yaml"
    _CUPS_COUNT = 1
    _SITES_SLICE = 2
    _TOLERANCE = 0.1

    def _initialize_env(self):
        super()._initialize_env()
        self.wall_cabinet = self._preset.get_props(WallCabinet)[0]

    def _success(self) -> bool:
        if not np.allclose(self.dishwasher.get_state(), 0, atol=self._TOLERANCE):
            return False
        if not np.allclose(self.wall_cabinet.get_state(), 0, atol=self._TOLERANCE):
            return False
        for cup in self.cups:
            if not cup.is_colliding(self.wall_cabinet.shelf_bottom):
                return False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(cup, side):
                    return False
        return True


class DishwasherLoadCups(_DishwasherCupsEnv):
    """Load cups to dishwasher task."""

    _CUPS_POS = np.array([0.6, -0.6, 1])
    _CUPS_POS_STEP = np.array([0, 0.15, 0])
    _CUPS_POS_BOUNDS = np.array([0.05, 0.02, 0])
    _CUPS_ROT_X = np.deg2rad(180)
    _CUPS_ROT_Z = np.deg2rad(180)
    _CUPS_ROT_BOUNDS = np.deg2rad(30)

    def _success(self) -> bool:
        for cup in self.cups:
            if not cup.is_colliding(self.dishwasher.tray_middle.colliders):
                return False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(cup, side):
                    return False
        return True

    def _on_reset(self):
        super()._on_reset()
        for i, cup in enumerate(self.cups):
            quat = Quaternion(axis=[1, 0, 0], angle=self._CUPS_ROT_X)
            angle = np.random.uniform(-self._CUPS_ROT_BOUNDS, self._CUPS_ROT_BOUNDS)
            quat *= Quaternion(axis=[0, 0, 1], angle=self._CUPS_ROT_Z + angle)
            cup.set_quaternion(quat.elements, True)
            pos = self._CUPS_POS + i * self._CUPS_POS_STEP
            pos += np.random.uniform(-self._CUPS_POS_BOUNDS, self._CUPS_POS_BOUNDS)
            cup.set_position(pos, True)


# G1 spawn for the whole dishwasher family (cups, cutlery, plates): the
# upstream x=0. The drawer-family x=0.12 wedges the settled Dex1 hands up to
# 34 mm into the counter front; at x=0 the objects stay one short step away
# and the counter/dishwasher turn keeps its locomotion.
G1_DISHWASHER_SPAWN = np.array([0.0, -0.6, 0])


class DishwasherLoadCupsG1(DishwasherLoadCups):
    """DishwasherLoadCups with a collision-free G1 start."""

    RESET_ROBOT_POS = G1_DISHWASHER_SPAWN


class DishwasherUnloadCupsG1(DishwasherUnloadCups):
    """DishwasherUnloadCups with a collision-free G1 start."""

    RESET_ROBOT_POS = G1_DISHWASHER_SPAWN


class DishwasherUnloadCupsLongG1(DishwasherUnloadCupsLong):
    """UnloadCupsLong with the lowered G1 wall cabinet (shelf ~1.15 m)."""

    _PRESET_PATH = PRESETS_PATH / "counter_dishwasher_wall_cabinet_g1.yaml"
    RESET_ROBOT_POS = G1_DISHWASHER_SPAWN
