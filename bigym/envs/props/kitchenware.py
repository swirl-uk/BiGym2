"""Kitchenware."""

from pathlib import Path

from bigym.const import ASSETS_PATH
from bigym.envs.props.prop import KinematicProp


class Plate(KinematicProp):
    """Plate."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/plate/plate.xml"


class Mug(KinematicProp):
    """Mug."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/mug/mug.xml"


class Pan(KinematicProp):
    """Pan."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/pan/pan.xml"


class Saucepan(KinematicProp):
    """Saucepan."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/saucepan/saucepan.xml"


class ChoppingBoard(KinematicProp):
    """Chopping Board."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/board/board.xml"
