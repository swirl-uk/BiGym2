"""Dishwasher interaction tasks."""

from abc import ABC

import numpy as np

from bigym.bigym_env import BiGymEnv
from bigym.const import PRESETS_PATH
from bigym.envs.props.dishwasher import Dishwasher


class _DishwasherEnv(BiGymEnv, ABC):
    """Base dishwasher environment."""

    RESET_ROBOT_POS = np.array([0, -0.8, 0])

    _PRESET_PATH = PRESETS_PATH / "dishwasher.yaml"
    _TOLERANCE = 0.05

    def _initialize_env(self):
        self.dishwasher = self._preset.get_props(Dishwasher)[0]


class DishwasherOpen(_DishwasherEnv):
    """Open the dishwasher door and pull out all trays."""

    def _success(self) -> bool:
        return np.allclose(self.dishwasher.get_state(), 1, atol=self._TOLERANCE)

    def _on_reset(self):
        self.dishwasher.set_state(door=0, bottom_tray=0, middle_tray=0)


class DishwasherClose(_DishwasherEnv):
    """Push back all trays and close the door of the dishwasher."""

    def _success(self) -> bool:
        return np.allclose(self.dishwasher.get_state(), 0, atol=self._TOLERANCE)

    def _on_reset(self):
        self.dishwasher.set_state(door=1, bottom_tray=1, middle_tray=1)


class DishwasherCloseTrays(DishwasherClose):
    """Push the dishwasher’s trays back with the door initially open."""

    def _success(self) -> bool:
        return np.allclose(self.dishwasher.get_state()[1:], 0, atol=self._TOLERANCE)


class DishwasherOpenTrays(DishwasherClose):
    """Pull out the dishwasher’s trays with the door initially open."""

    def _success(self) -> bool:
        return np.allclose(self.dishwasher.get_state()[1:], 1, atol=self._TOLERANCE)

    def _on_reset(self):
        self.dishwasher.set_state(door=1, bottom_tray=0, middle_tray=0)


# G1 door/tray tasks use the raised dishwasher preset (0.25 m plinth). The
# lateral y=-0.8 spawn is unchanged to clear the fold-down door; approaching
# the appliance remains part of the task.


class DishwasherOpenG1(DishwasherOpen):
    """DishwasherOpen on the raised G1 dishwasher (plinth 0.25 m, same spawn)."""

    _PRESET_PATH = PRESETS_PATH / "dishwasher_g1_raised.yaml"


class DishwasherCloseG1(DishwasherClose):
    """DishwasherClose on the raised G1 dishwasher (plinth 0.25 m, same spawn)."""

    _PRESET_PATH = PRESETS_PATH / "dishwasher_g1_raised.yaml"


class DishwasherOpenTraysG1(DishwasherOpenTrays):
    """DishwasherOpenTrays on the raised G1 dishwasher (plinth 0.25 m, same spawn)."""

    _PRESET_PATH = PRESETS_PATH / "dishwasher_g1_raised.yaml"


class DishwasherCloseTraysG1(DishwasherCloseTrays):
    """DishwasherCloseTrays on the raised G1 dishwasher (plinth 0.25 m, same spawn)."""

    _PRESET_PATH = PRESETS_PATH / "dishwasher_g1_raised.yaml"
