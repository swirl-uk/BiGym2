"""Utensil holder."""

from abc import ABC
from pathlib import Path

import mujoco

from bigym.const import ASSETS_PATH
from bigym.envs.props.prop import Prop


class _Holder(Prop, ABC):
    """Base Holder Prop."""

    _CACHE_COLLIDERS = True
    _CACHE_SITES = True


class CutleryTray(_Holder):
    """Cutlery tray."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/cutlery_tray/cutlery_tray.xml"


class DishDrainer(_Holder):
    """Dish Drainer."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/dish_drainer/dish_drainer.xml"

    def _post_init(self):
        self.holders_left: list[mujoco.MjsGeom] = self.colliders[0:7]
        self.holders_right: list[mujoco.MjsGeom] = self.colliders[7:14]
