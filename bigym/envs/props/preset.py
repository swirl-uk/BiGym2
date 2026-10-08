"""Props Preset."""

from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, Type, TypeVar, cast, overload

import numpy as np
from yaml import safe_load

import bigym.envs.props as props_module
from bigym.envs.props.prop import Prop
from bigym.simulation import Simulation
from bigym.utils.physics_utils import zyz_euler_to_quaternion
from bigym.utils.shared import find_class_in_module

PT = TypeVar("PT", bound=Prop)

Placement = tuple[Prop, Optional[np.ndarray], Optional[np.ndarray]]


class Preset:
    """Props of a scene layout, loaded from a YAML file.

    Each prop is placed in the spec at its configured position and rotation,
    except the last prop loaded: its placement waits in :attr:`last_placement`
    until :meth:`place_last_prop` is called.
    """

    def __init__(self, simulation: Simulation, path: Optional[Path]):
        """Load the props of ``path`` into the simulation's spec."""
        self._props: list[Prop] = []
        self._props_lookup: dict[type[Prop], list[Prop]] = defaultdict(list)
        self.last_placement: Optional[Placement] = None

        if path is None:
            return
        self._load_file(simulation, path)
        for prop in self._props:
            self._props_lookup[type(prop)].append(prop)

    @overload
    def get_props(self, prop_type: None = None) -> list[Prop]: ...

    @overload
    def get_props(self, prop_type: Type[PT]) -> list[PT]: ...

    def get_props(self, prop_type: Optional[Type[PT]] = None) -> list:
        """Get preset props."""
        if prop_type is None:
            return self._props
        else:
            # The lookup is keyed by exact type, so the list holds `prop_type`s.
            return cast(list[PT], self._props_lookup.get(prop_type, []))

    def place_last_prop(self, compiled: bool):
        """Place the last prop loaded, in the spec or, once compiled, in the model."""
        if self.last_placement is None:
            return
        prop, position, quaternion = self.last_placement
        if compiled:
            if position is not None:
                prop.set_position(position)
            if quaternion is not None:
                prop.set_quaternion(quaternion)
        else:
            if position is not None:
                prop.body.pos = position
            if quaternion is not None:
                prop.body.quat = quaternion
        self.last_placement = None

    def _load_file(self, simulation: Simulation, path: Path):
        with open(path) as f:
            config = safe_load(f)
        for include in config.get("include", []):
            self._load_file(simulation, path.parent / include)
        for prop_config in config.get("props", []):
            self._load_prop(prop_config, simulation)

    def _load_prop(self, config, simulation: Simulation, parent: Optional[Prop] = None):
        prop_type = config.pop("type")
        prop_cls = cast(
            Optional[type[Prop]], find_class_in_module(props_module, prop_type)
        )
        if not prop_cls:
            return
        position = self._get_float_array(config.pop("position", None), 3)
        euler = self._get_float_array(config.pop("euler", None), 3, np.deg2rad)
        children = config.pop("children", [])
        if parent is None:
            prop = prop_cls(simulation, **config)
        else:
            parent_site_name = config.pop("parent_site")
            parent_site = next(
                (
                    site
                    for site in parent.sites
                    if site.name == parent.body.name + parent_site_name
                ),
                None,
            )
            if parent_site is None:
                raise ValueError(
                    f"Site with name '{parent_site_name}' not found in {parent}"
                )
            prop = prop_cls(simulation, parent=parent_site, **config)
        self.place_last_prop(compiled=False)
        self._props.append(prop)
        quaternion = None if euler is None else zyz_euler_to_quaternion(euler)
        self.last_placement = (prop, position, quaternion)
        for child_config in children:
            self._load_prop(child_config, simulation, prop)

    @staticmethod
    def _get_float_array(
        value: Optional[list],
        target_length: int,
        post_process: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Optional[np.ndarray]:
        if value is None:
            return None
        if len(value) != target_length:
            raise ValueError(
                f"Incorrect array length: {value}, expected length: {target_length}"
            )
        result = np.array([float(item) for item in value])
        if post_process:
            result = post_process(result)
        return result
