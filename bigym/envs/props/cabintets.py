"""Modular cabinets."""

from abc import ABC
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt

from bigym import scene
from bigym.const import ASSETS_PATH
from bigym.envs.props.prop import CollidableProp
from bigym.utils.physics_utils import get_joint_position, set_joint_position


class ModularCabinet(CollidableProp, ABC):
    """Base modular prop."""

    def _post_init(self):
        self.joints = scene.descendants(self.body, "joints")

    def set_state(self, state: np.ndarray):
        """Set normalized state of joints."""
        for value, joint in zip(state, self.joints, strict=True):
            set_joint_position(self.simulation, joint, value, True)

    def get_state(self) -> npt.NDArray[np.float64]:
        """Get normalized state of joints."""
        return np.array(
            [get_joint_position(self.simulation, joint, True) for joint in self.joints]
        )

    @staticmethod
    def _toggle_body(model: mujoco.MjSpec, name: str, enable: bool):
        if not enable:
            model.delete(model.body(name))

    @staticmethod
    def _last_geom(model: mujoco.MjSpec, body: str) -> mujoco.MjsGeom:
        return scene.descendants(model.body(body), "geoms")[-1]


class BaseCabinet(ModularCabinet):
    """Modular base cabinet."""

    _BIG_DRAWERS = ["drawer_big_1", "drawer_big_2"]
    _SMALL_DRAWERS = [
        "drawer_small_1",
        "drawer_small_2",
        "drawer_small_3",
        "drawer_small_4",
    ]
    _DOOR_LEFT = "door_left"
    _DOOR_RIGHT = "door_right"
    _PANEL = "panel"
    _SHELF = "shelf"
    _SHELF_BOTTOM = "shelf_bottom"
    _HOB = "hob"
    _WALLS = "walls"
    _COUNTER = "counter"

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/kitchen/base_cabinet_600.xml"

    def _parse_kwargs(self, kwargs: dict[str, Any]):
        self._walls_enable = kwargs.get("walls_enable", True)
        big_drawers_enable = kwargs.get("big_drawers_enable", None)
        small_drawers_enable = kwargs.get("small_drawers_enable", None)
        self._hob_enable = kwargs.get("hob_enable", False)
        self._panel_enable = kwargs.get("panel_enable", False)
        self._door_left_enable = kwargs.get("door_left_enable", False)
        self._door_right_enable = kwargs.get("door_right_enable", False)
        self._shelf_enable = kwargs.get("shelf_enable", False)

        if big_drawers_enable is None:
            big_drawers_enable = [False] * len(self._BIG_DRAWERS)
        if small_drawers_enable is None:
            small_drawers_enable = [False] * len(self._SMALL_DRAWERS)
        self._big_drawers_enable: list[bool] = big_drawers_enable
        self._small_drawers_enable: list[bool] = small_drawers_enable

        assert len(self._big_drawers_enable) == len(self._BIG_DRAWERS)
        assert len(self._small_drawers_enable) == len(self._SMALL_DRAWERS)

        self._CACHE_SITES = (
            self._hob_enable
            or any(self._big_drawers_enable)
            or any(self._small_drawers_enable)
        )

    def _on_loaded(self, model: mujoco.MjSpec):
        self.shelf = self._last_geom(model, self._SHELF)
        self.shelf_bottom = self._last_geom(model, self._SHELF_BOTTOM)
        self.counter = self._last_geom(model, self._COUNTER)
        self.hob = self._last_geom(model, self._HOB)

        for name, enable in zip(
            self._BIG_DRAWERS, self._big_drawers_enable, strict=True
        ):
            self._toggle_body(model, name, enable)
        for name, enable in zip(
            self._SMALL_DRAWERS, self._small_drawers_enable, strict=True
        ):
            self._toggle_body(model, name, enable)
        self._toggle_body(model, self._WALLS, self._walls_enable)
        self._toggle_body(model, self._HOB, self._hob_enable)
        self._toggle_body(model, self._PANEL, self._panel_enable)
        self._toggle_body(model, self._DOOR_LEFT, self._door_left_enable)
        self._toggle_body(model, self._DOOR_RIGHT, self._door_right_enable)
        self._toggle_body(model, self._SHELF, self._shelf_enable)


class WallCabinet(ModularCabinet):
    """Modular wall cabinet."""

    _DOORS = ["door_right", "door_left"]
    _GLASS_DOORS = ["door_right_glass", "door_left_glass"]
    _VENT = "vent"
    _SHELF = "shelf"
    _SHELF_BOTTOM = "shelf_bottom"

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/kitchen/wall_cabinet_600.xml"

    def _parse_kwargs(self, kwargs: dict[str, Any]):
        self._doors_enable = kwargs.get("doors_enable", False)
        self._glass_doors_enable = kwargs.get("glass_doors_enable", False)
        self._vent_enable = kwargs.get("vent_enable", False)

    def _on_loaded(self, model: mujoco.MjSpec):
        self.shelf = self._last_geom(model, self._SHELF)
        self.shelf_bottom = self._last_geom(model, self._SHELF_BOTTOM)

        for door_name in self._DOORS:
            self._toggle_body(model, door_name, self._doors_enable)
        for door_name in self._GLASS_DOORS:
            self._toggle_body(model, door_name, self._glass_doors_enable)
        self._toggle_body(model, self._VENT, self._vent_enable)


class OpenShelf(ModularCabinet):
    """Modular open shelf."""

    _SHELF = "shelf"

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/kitchen/open_shelf_600.xml"

    def _on_loaded(self, model: mujoco.MjSpec):
        self.shelf = self._last_geom(model, self._SHELF)
