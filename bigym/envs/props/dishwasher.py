"""Dishwasher."""

from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from bigym.const import ASSETS_PATH
from bigym.envs.props.prop import Prop
from bigym.utils.physics_utils import get_joint_position, set_joint_position


class DishwasherPart:
    """Part of the dishwasher."""

    def __init__(
        self,
        spec: mujoco.MjSpec,
        namespace: str,
        body_name: str,
        joint_name: Optional[str] = None,
        site_sets: Optional[list[tuple[str, int]]] = None,
    ):
        """Find the part's elements in ``spec`` under the dishwasher's ``namespace``."""
        self.body: mujoco.MjsBody = spec.body(namespace + body_name)
        self.joint: Optional[mujoco.MjsJoint] = (
            spec.joint(namespace + joint_name) if joint_name else None
        )
        self.site_sets: list[list[mujoco.MjsSite]] = [
            [spec.site(f"{namespace}{sites_name}_{i + 1}") for i in range(sites_count)]
            for sites_name, sites_count in site_sets or []
        ]
        self.colliders: list[mujoco.MjsGeom] = Prop.get_body_colliders(self.body)


class Dishwasher(Prop):
    """Dishwasher."""

    DOOR_BODY = "door"
    DOOR_JOINT = "door_hinge"

    TRAY_BOTTOM_BODY = "tray_bottom"
    TRAY_BOTTOM_JOINT = "tray_bottom_linear"
    TRAY_BOTTOM_SITES = [
        ("tray_bottom_holder_1", 11),
        ("tray_bottom_holder_2", 11),
    ]

    TRAY_MIDDLE_BODY = "tray_mid"
    TRAY_MIDDLE_JOINT = "tray_mid_linear"
    TRAY_MIDDLE_SITES = [
        ("tray_mid_holder_sites_1", 11),
        ("tray_mid_holder_sites_2", 11),
    ]

    BASKET_BODY = "cuttlery_basket"
    BASKET_SITES = [("cuttlery_basket", 6)]

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/dishwasher/dishwasher.xml"

    def _post_init(self):
        spec, namespace = self.simulation.spec, self.body.name
        self.door = DishwasherPart(spec, namespace, self.DOOR_BODY, self.DOOR_JOINT)
        self.tray_bottom = DishwasherPart(
            spec,
            namespace,
            self.TRAY_BOTTOM_BODY,
            self.TRAY_BOTTOM_JOINT,
            self.TRAY_BOTTOM_SITES,
        )
        self.tray_middle = DishwasherPart(
            spec,
            namespace,
            self.TRAY_MIDDLE_BODY,
            self.TRAY_MIDDLE_JOINT,
            self.TRAY_MIDDLE_SITES,
        )
        self.basket = DishwasherPart(
            spec, namespace, self.BASKET_BODY, site_sets=self.BASKET_SITES
        )

    def set_state(self, door: float, bottom_tray: float, middle_tray: float):
        """Set state of dishwasher joints."""
        assert self.door.joint and self.tray_bottom.joint and self.tray_middle.joint
        set_joint_position(self.simulation, self.door.joint, door, True)
        set_joint_position(self.simulation, self.tray_bottom.joint, bottom_tray, True)
        set_joint_position(self.simulation, self.tray_middle.joint, middle_tray, True)

    def get_state(self) -> np.ndarray:
        """Get state of dishwasher joints."""
        assert self.door.joint and self.tray_bottom.joint and self.tray_middle.joint
        return np.array(
            [
                get_joint_position(self.simulation, self.door.joint, True),
                get_joint_position(self.simulation, self.tray_bottom.joint, True),
                get_joint_position(self.simulation, self.tray_middle.joint, True),
            ]
        )
