"""Groceries tasks."""

import numpy as np

from bigym.bigym_env import BiGymEnv
from bigym.const import PRESETS_PATH
from bigym.envs.props.cabintets import BaseCabinet, OpenShelf, WallCabinet
from bigym.envs.props.items import Beer, Cereals, Ketchup, Mustard, Soap, Soda, Wine
from bigym.envs.props.prop import Prop
from bigym.utils.env_utils import get_random_points_on_plane


class GroceriesStoreLower(BiGymEnv):
    """Put groceries to lower cabinets tasks."""

    _PRESET_PATH = PRESETS_PATH / "counter_base_2.yaml"

    _PROP_TYPES = [Wine, Soap, Beer, Cereals, Ketchup, Mustard, Soda]

    _PROPS_PER_CATEGORY = 1
    _PROPS_COUNT = 4

    _PROPS_POS = np.array([0.7, -0.3, 0.9])
    _PROPS_STEP = 0.2
    _PROPS_POS_EXTENTS = np.array([0.1, 0.5])
    _PROPS_POS_BOUNDS = np.array([0.1, 0.05, 0])
    _PROPS_ROT_BOUNDS = np.deg2rad([0, 0, 180])

    def _initialize_env(self):
        self.cabinet_1 = self._preset.get_props(BaseCabinet)[0]
        self.cabinet_2 = self._preset.get_props(BaseCabinet)[1]
        self.props: list[Prop] = []
        self.selected_props: list[Prop] | np.ndarray = []
        for prop_type in self._PROP_TYPES:
            self.props.extend(
                [prop_type(self.simulation) for _ in range(self._PROPS_PER_CATEGORY)]
            )

    def _on_reset(self):
        for prop in self.props:
            prop.disable()

        self.selected_props = np.random.choice(
            np.array(self.props), size=self._PROPS_COUNT, replace=False
        )
        points = get_random_points_on_plane(
            len(self.selected_props),
            self._PROPS_POS,
            self._PROPS_POS_EXTENTS,
            self._PROPS_STEP,
        )
        for prop, point in zip(self.selected_props, points, strict=True):
            prop.enable()
            prop.set_pose(
                point,
                position_bounds=self._PROPS_POS_BOUNDS,
                rotation_bounds=self._PROPS_ROT_BOUNDS,
            )

    def _success(self) -> bool:
        for prop in self.selected_props:
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(prop, side):
                    return False
            if not (
                prop.is_colliding(self.cabinet_1.shelf)
                or prop.is_colliding(self.cabinet_1.shelf_bottom)
                or prop.is_colliding(self.cabinet_2.shelf_bottom)
            ):
                return False
        return True


class GroceriesStoreUpper(GroceriesStoreLower):
    """Put groceries to upper cabinets tasks."""

    _PRESET_PATH = PRESETS_PATH / "counter_base_wall_2x2.yaml"

    def _initialize_env(self):
        super()._initialize_env()
        self.cabinet_wall = self._preset.get_props(WallCabinet)[0]
        self.shelf = self._preset.get_props(OpenShelf)[0]

    def _success(self) -> bool:
        for prop in self.selected_props:
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(prop, side):
                    return False
            if not (
                prop.is_colliding(self.cabinet_wall.shelf)
                or prop.is_colliding(self.cabinet_wall.shelf_bottom)
                or prop.is_colliding(self.shelf.shelf)
            ):
                return False
        return True


class GroceriesStoreLowerG1(GroceriesStoreLower):
    """GroceriesStoreLower with the spawn pulled in for the G1 workspace.

    Item band x 0.6-0.8 on the counter and the low cabinet-shelf target
    (0.47 m) are both inside the G1 band; only the spawn moves forward.
    """

    # counter_base_2 cabinets sit at x=1.1 (front face 0.5). From 0.15 on, the
    # settled Dex1 fingers wedge into the counter front after the GR00T reset
    # warmup; 0.14 leaves 46 mm finger-to-counter clearance and the pelvis
    # settles at x~0.07 with the item band 0.6-0.9 m ahead.
    RESET_ROBOT_POS = np.array([0.14, 0, 0])


class GroceriesStoreUpperG1(GroceriesStoreUpper):
    """GroceriesStoreUpper with the wall cabinet and open shelf lowered.

    Both target fixtures drop 0.26 m and come 0.2 m forward (shelf bottom
    1.198 m); the counter item band is unchanged and the spawn matches
    GroceriesStoreLowerG1.

    The items are placed 1 cm above the counter top (0.859 m) instead of
    the stock 4 cm. Together with the 0.26 m (not 0.32 m) drop this keeps
    the 0.300 m wine bottle below the shelf bottom at placement, so it is
    not born inside the shelf and ejected.
    """

    _PRESET_PATH = PRESETS_PATH / "counter_base_wall_2x2_g1.yaml"
    RESET_ROBOT_POS = np.array([0.14, 0, 0])
    _PROPS_POS = np.array([0.7, -0.3, 0.87])
