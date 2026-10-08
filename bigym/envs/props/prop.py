"""Abstract prop."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import mujoco
import numpy as np
from pyquaternion import Quaternion

from bigym import scene
from bigym.simulation import Simulation
from bigym.utils.physics_utils import (
    Colliders,
    get_quaternion,
    has_collided_collections,
    set_body_position,
    set_body_quaternion,
)


class Prop(ABC):
    """A model attached to the scene under its own frame body.

    The prop adds itself to the simulation's spec on construction and keeps
    its spec elements; they index the compiled model and data once the scene
    is compiled.
    """

    _KINEMATIC = False
    _CACHE_COLLIDERS = False
    _CACHE_SITES = False

    HIDDEN_POSITION = np.array([0, 0, -100])

    def __init__(
        self,
        simulation: Simulation,
        kinematic: Optional[bool] = None,
        cache_colliders: Optional[bool] = None,
        cache_sites: Optional[bool] = None,
        parent: Optional[mujoco.MjsSite] = None,
        **kwargs,
    ):
        """Load the prop model and attach it to the scene, at ``parent`` if given."""
        self._parse_kwargs(kwargs or {})
        kinematic = kinematic or self._KINEMATIC
        cache_colliders = cache_colliders or self._CACHE_COLLIDERS
        cache_sites = cache_sites or self._CACHE_SITES

        self.simulation = simulation
        model = scene.load(self._model_path)
        self._on_loaded(model)
        self.body: mujoco.MjsBody = scene.attach(simulation.spec, model, parent)

        self.geoms: list[mujoco.MjsGeom] = scene.descendants(self.body, "geoms")
        self.colliders: list[mujoco.MjsGeom] = []
        if cache_colliders:
            self.colliders = self.get_body_colliders(self.body)
        self.sites: list[mujoco.MjsSite] = []
        if cache_sites:
            self.sites = scene.descendants(self.body, "sites")
        self.kinematic = kinematic
        self.freejoint: Optional[mujoco.MjsJoint] = None
        if self.kinematic:
            self.freejoint = self.body.add_freejoint(name=self.body.name)
        self.collision_settings = [
            (geom.contype, geom.conaffinity) for geom in self.geoms
        ]
        self._post_init()

    @property
    @abstractmethod
    def _model_path(self) -> Path:
        raise NotImplementedError

    def _parse_kwargs(self, kwargs: dict[str, Any]):  # noqa: B027 - optional hook
        """Process initialization kwargs."""
        pass

    def _on_loaded(self, model: mujoco.MjSpec):  # noqa: B027 - optional hook
        """Customize the prop model before it is attached."""
        pass

    def _post_init(self):  # noqa: B027 - optional hook
        """Customize prop initialization."""
        pass

    def get_position(self) -> np.ndarray:
        """World position; a kinematic prop reports its free joint position."""
        if self.freejoint:
            return self.simulation.data.bind(self.freejoint).qpos[:3].copy()
        return self.simulation.data.bind(self.body).xpos.copy()

    def get_quaternion(self) -> np.ndarray:
        """World rotation as a quaternion (w, x, y, z)."""
        return get_quaternion(self.simulation.data, self.body)

    def set_position(self, position: np.ndarray, reset_dynamics: bool = True):
        """Move the prop: its free joint if kinematic, else its frame in the model."""
        if self.freejoint:
            joint = self.simulation.data.bind(self.freejoint)
            joint.qpos[:3] = position
            if reset_dynamics:
                joint.qvel *= 0
                joint.qacc *= 0
        else:
            set_body_position(self.simulation.model, self.body, position)

    def set_quaternion(self, quaternion: np.ndarray, reset_dynamics: bool = True):
        """Rotate the prop: its free joint if kinematic, and its frame in the model."""
        if self.freejoint:
            joint = self.simulation.data.bind(self.freejoint)
            joint.qpos[3:] = quaternion
            if reset_dynamics:
                joint.qvel *= 0
                joint.qacc *= 0
        set_body_quaternion(
            self.simulation.model, self.simulation.data, self.body, quaternion
        )

    def get_pose(self) -> np.ndarray:
        """Get pose in the world space."""
        return np.concatenate((self.get_position(), self.get_quaternion()), axis=-1)

    def set_pose(
        self,
        position: np.ndarray | None = None,
        quat: np.ndarray | None = None,
        position_bounds: np.ndarray | None = None,
        rotation_bounds: np.ndarray | None = None,
    ):
        """Set pose in the world space (by default the origin, unrotated)."""
        position = np.zeros(3) if position is None else position
        quat = Quaternion().elements if quat is None else quat
        position_bounds = np.zeros(3) if position_bounds is None else position_bounds
        rotation_bounds = np.zeros(3) if rotation_bounds is None else rotation_bounds
        offset_pos = np.random.uniform(-position_bounds, position_bounds)
        pos = position + offset_pos

        offset_rot = np.random.uniform(-rotation_bounds, rotation_bounds)
        quat = (
            Quaternion(quat)
            * Quaternion(axis=[1, 0, 0], angle=offset_rot[0])
            * Quaternion(axis=[0, 1, 0], angle=offset_rot[1])
            * Quaternion(axis=[0, 0, 1], angle=offset_rot[2])
        )

        self.set_position(pos)
        self.set_quaternion(quat.elements)

    def disable(self):
        """Make the prop intangible and move it out of the way."""
        for geom in self.geoms:
            bound = self.simulation.model.bind(geom)
            bound.contype = 0
            bound.conaffinity = 0
        if self.freejoint:
            self.simulation.model.bind(self.freejoint).damping = 10e6
        self.set_position(self.HIDDEN_POSITION)

    def enable(self):
        """Restore the collisions and free motion of a disabled prop."""
        for geom, (contype, conaffinity) in zip(
            self.geoms, self.collision_settings, strict=True
        ):
            bound = self.simulation.model.bind(geom)
            bound.contype = contype
            bound.conaffinity = conaffinity
        if self.freejoint:
            self.simulation.model.bind(self.freejoint).damping = 0

    def get_velocities(self) -> np.ndarray:
        """Get velocities of the free body."""
        if self.freejoint:
            return np.array(self.simulation.data.bind(self.freejoint).qvel)
        return np.zeros(0)

    def is_static(self, atol_pos: float = 1.0e-3):
        """Check if object's position is static."""
        velocities = self.get_velocities()
        return np.allclose(velocities[:3], 0, atol=atol_pos)

    def is_colliding(self, other: Colliders) -> bool:
        """Whether a collider of this prop touches ``other``."""
        return has_collided_collections(self.simulation, self.colliders, other)

    @staticmethod
    def get_body_colliders(body: mujoco.MjsBody) -> list[mujoco.MjsGeom]:
        """Get all colliders of the body."""
        return [
            geom
            for geom in scene.descendants(body, "geoms")
            if geom.contype == 1 and geom.conaffinity == 1
        ]


class CollidableProp(Prop, ABC):
    """Collidable prop."""

    _CACHE_COLLIDERS = True


class KinematicProp(Prop, ABC):
    """Kinematic collidable prop."""

    _KINEMATIC = True
    _CACHE_COLLIDERS = True
