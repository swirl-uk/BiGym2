"""Tables."""

from pathlib import Path

from bigym.const import ASSETS_PATH
from bigym.envs.props.prop import CollidableProp


class Table(CollidableProp):
    """Default Table."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/table/table.xml"


class SmallTable(CollidableProp):
    """Shorter version of the default table."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/table_dishwasher/table_dishwasher.xml"
