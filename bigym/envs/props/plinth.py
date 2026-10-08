"""Static appliance plinth for the G1 workspace."""

from pathlib import Path

from bigym.const import ASSETS_PATH
from bigym.envs.props.prop import CollidableProp


class Plinth(CollidableProp):
    """A fixed 0.62 x 0.58 x 0.25 m block an appliance stands on.

    Used by ``dishwasher_g1_raised.yaml`` to lift the free-standing
    dishwasher so its fold-down door (0.16-0.25 m high when open) lands at
    0.41-0.50 m, inside the G1's reach at a moderate squat with the torso
    pitched.
    """

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/plinth/plinth.xml"
