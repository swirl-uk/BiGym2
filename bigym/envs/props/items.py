"""Different pickable items."""

from pathlib import Path
from typing import Any

import mujoco

from bigym.const import ASSETS_PATH
from bigym.envs.props.prop import KinematicProp


class Box(KinematicProp):
    """Box."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/box/box.xml"


class LightBox(Box):
    """The stock box at 3 kg instead of 5 kg (G1 bimanual-squeeze carry)."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/box/box_light.xml"


class Cube(KinematicProp):
    """Cube."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/cube/cube.xml"


class Sandwich(KinematicProp):
    """Sandwich."""

    _COLLIDER = "collider"
    _COLLIDER_ROUNDED = "collider_rounded"

    _NORMAL_MODEL = "sandwich.xml"
    _TOASTED_MODEL = "sandwich_toasted.xml"

    @property
    def _model_path(self) -> Path:
        model = self._TOASTED_MODEL if self.toasted else self._NORMAL_MODEL
        return ASSETS_PATH / "props/sandwich" / model

    def _parse_kwargs(self, kwargs: dict[str, Any]):
        self.toasted = kwargs.get("toasted", False)
        self.rounded_collider = kwargs.get("rounded_collider", False)

    def _on_loaded(self, model: mujoco.MjSpec):
        collider_to_remove = (
            self._COLLIDER if self.rounded_collider else self._COLLIDER_ROUNDED
        )
        model.delete(model.geom(collider_to_remove))


class Wine(KinematicProp):
    """Wine."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/groceries/wine/wine.xml"


class Detergent(KinematicProp):
    """Detergent."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/groceries/detergent/detergent.xml"


class Soap(KinematicProp):
    """Soap."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/groceries/soap/soap.xml"


class Beer(KinematicProp):
    """Beer."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/groceries/beer.xml"


class Cereals(KinematicProp):
    """Cereals."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/groceries/cereal.xml"


class Crisps(KinematicProp):
    """Crisps."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/groceries/crisps.xml"


class Ketchup(KinematicProp):
    """Ketchup."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/groceries/ketchup.xml"


class Mustard(KinematicProp):
    """Mustard."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/groceries/mustard.xml"


class Soda(KinematicProp):
    """Soda."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/groceries/soda.xml"
