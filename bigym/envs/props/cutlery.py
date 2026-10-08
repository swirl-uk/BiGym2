"""Cutlery props."""

from pathlib import Path

from bigym.const import ASSETS_PATH
from bigym.envs.props.prop import KinematicProp


class Fork(KinematicProp):
    """Fork."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/cutlery/fork/fork.xml"


class Knife(KinematicProp):
    """Knife."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/cutlery/knife/knife.xml"


class Spoon(KinematicProp):
    """Spoon."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/cutlery/spoon/spoon.xml"


class Spatula(KinematicProp):
    """Spatula."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/spatula/spatula.xml"
