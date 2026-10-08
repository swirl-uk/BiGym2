"""Cupboard interaction tasks."""

from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
from pyquaternion import Quaternion

from bigym.bigym_env import BiGymEnv
from bigym.const import PRESETS_PATH
from bigym.envs.props.cabintets import BaseCabinet, WallCabinet

TOLERANCE = 0.1


class _CupboardsInteractionEnv(BiGymEnv, ABC):
    """Base cupboards environment."""

    RESET_ROBOT_POS = np.array([-0.2, 0, 0])

    _PRESET_PATH = PRESETS_PATH / "counter_base_wall_3x1.yaml"

    def _initialize_env(self):
        self.cabinet_drawers = self._preset.get_props(BaseCabinet)[0]
        self.cabinet_door_left = self._preset.get_props(BaseCabinet)[1]
        self.cabinet_door_right = self._preset.get_props(BaseCabinet)[2]
        self.cabinet_wall = self._preset.get_props(WallCabinet)[0]
        self.all_cabinets = [
            self.cabinet_drawers,
            self.cabinet_door_left,
            self.cabinet_door_right,
            self.cabinet_wall,
        ]


class DrawerTopOpen(_CupboardsInteractionEnv):
    """Open top drawer of the cupboard task."""

    def _success(self) -> bool:
        return np.allclose(self.cabinet_drawers.get_state()[-1], 1, atol=TOLERANCE)


class DrawerTopClose(_CupboardsInteractionEnv):
    """Close top drawer of the cupboard task."""

    def _success(self) -> bool:
        return np.allclose(self.cabinet_drawers.get_state()[-1], 0, atol=TOLERANCE)

    def _on_reset(self):
        self.cabinet_drawers.set_state(np.array([0, 0, 1]))


class DrawersAllOpen(_CupboardsInteractionEnv):
    """Open all drawers of the cupboard task."""

    def _success(self) -> bool:
        return np.allclose(self.cabinet_drawers.get_state(), 1, atol=TOLERANCE)


class DrawersAllClose(_CupboardsInteractionEnv):
    """Close all drawers of the cupboard task."""

    def _success(self) -> bool:
        return np.allclose(self.cabinet_drawers.get_state(), 0, atol=TOLERANCE)

    def _on_reset(self):
        self.cabinet_drawers.set_state(np.array([1, 1, 1]))


class WallCupboardOpen(_CupboardsInteractionEnv):
    """Open doors of the wall cupboard task."""

    def _success(self) -> bool:
        return np.allclose(self.cabinet_wall.get_state(), 1, atol=TOLERANCE)


class WallCupboardClose(_CupboardsInteractionEnv):
    """Close doors of the wall cupboard task."""

    def _success(self) -> bool:
        return np.allclose(self.cabinet_wall.get_state(), 0, atol=TOLERANCE)

    def _on_reset(self):
        self.cabinet_wall.set_state(np.array([1, 1]))


class CupboardsOpenAll(_CupboardsInteractionEnv):
    """Open all doors/drawers of the kitchen counter task."""

    def _success(self) -> bool:
        for cabinet in self.all_cabinets:
            if not np.allclose(cabinet.get_state(), 1, atol=TOLERANCE):
                return False
        return True


class CupboardsCloseAll(_CupboardsInteractionEnv):
    """Close all doors/drawers of the kitchen counter task."""

    def _success(self) -> bool:
        for cabinet in self.all_cabinets:
            if not np.allclose(cabinet.get_state(), 0, atol=TOLERANCE):
                return False
        return True

    def _on_reset(self):
        for cabinet in self.all_cabinets:
            cabinet.set_state(np.ones_like(cabinet.get_state()))


# G1 workspace spawn: from the upstream spawn the closed top handle is 0.84 m
# from the pelvis, beyond G1 standing reach. groot_wbc_g1 settles the pelvis
# ~0.07 behind the spawn, so x=0.12 puts the closed handle 0.59 m ahead while
# the fully-open drawer front still clears the toes by >10 cm. Scene geometry
# is unchanged.
G1_RESET_ROBOT_POS = np.array([0.12, 0, 0])

if TYPE_CHECKING:
    _G1InitializationMixinBase = _CupboardsInteractionEnv
else:
    _G1InitializationMixinBase = object


class _DrawerTopG1InitializationMixin(_G1InitializationMixinBase):
    """Seeded ID reset distribution for the two canonical G1 drawer tasks."""

    _G1_ID_PROFILE = "g1_id_v1"
    _G1_ID_RESET_X_BOUNDS = (0.12, 0.14)
    _G1_ID_RESET_Y_BOUNDS = (-0.05, 0.05)
    _G1_ID_RESET_YAW_BOUNDS = np.deg2rad((-5.0, 5.0))
    _G1_ID_DRAWER_TOP_STATE_BOUNDS: tuple[float, float]

    def _sample_reset_robot_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if self.initialization_profile != self._G1_ID_PROFILE:
            return super()._sample_reset_robot_pose()
        x = np.random.uniform(*self._G1_ID_RESET_X_BOUNDS)
        y = np.random.uniform(*self._G1_ID_RESET_Y_BOUNDS)
        yaw = np.random.uniform(*self._G1_ID_RESET_YAW_BOUNDS)
        return (
            np.array([x, y, 0.0], dtype=np.float64),
            Quaternion(axis=[0, 0, 1], angle=yaw).elements,
        )

    def _on_reset(self):
        if self.initialization_profile != self._G1_ID_PROFILE:
            return super()._on_reset()
        state = np.zeros(3, dtype=np.float64)
        state[-1] = np.random.uniform(*self._G1_ID_DRAWER_TOP_STATE_BOUNDS)
        self._g1_id_sampled_drawer_state = state.copy()
        self.cabinet_drawers.set_state(state)

    def _on_reset_warmup_step(self):
        super()._on_reset_warmup_step()
        if self.initialization_profile == self._G1_ID_PROFILE:
            # The G1 settle otherwise pushes a partly open top drawer outward
            # (up to ~0.1 normalized). Hold the sampled state through warmup
            # so the first observation realizes the sampled distribution
            # without a final-frame teleport.
            self.cabinet_drawers.set_state(self._g1_id_sampled_drawer_state)


class DrawerTopOpenG1(_DrawerTopG1InitializationMixin, DrawerTopOpen):
    """DrawerTopOpen with the spawn pulled in for the G1 workspace."""

    RESET_ROBOT_POS = G1_RESET_ROBOT_POS
    _G1_ID_DRAWER_TOP_STATE_BOUNDS = (0.0, 0.15)


class DrawerTopCloseG1(_DrawerTopG1InitializationMixin, DrawerTopClose):
    """DrawerTopClose with the spawn pulled in for the G1 workspace."""

    RESET_ROBOT_POS = G1_RESET_ROBOT_POS
    _G1_ID_DRAWER_TOP_STATE_BOUNDS = (0.85, 1.0)


class DrawersAllOpenG1(DrawersAllOpen):
    """DrawersAllOpen with the spawn pulled in for the G1 workspace."""

    RESET_ROBOT_POS = G1_RESET_ROBOT_POS


class DrawersAllCloseG1(DrawersAllClose):
    """DrawersAllClose with the spawn pulled in for the G1 workspace."""

    RESET_ROBOT_POS = G1_RESET_ROBOT_POS


# G1 wall-cabinet variants: the upstream wall cabinet (shelf bottom 1.47 m,
# doors ~1.8 m) is above the G1 fingertip ceiling (~1.55 m standing). The G1
# presets lower it 0.32 m and pull it 0.2 m forward (shelf bottom ~1.15 m,
# keeping >=0.25 m above the counter for objects to pass under; handles
# ~1.2 m). The *_all tasks spawn at x=0.0 (see CupboardsOpenAllG1).


class _WallCupboardG1InitializationMixin(_G1InitializationMixinBase):
    """Seeded ID reset distribution for the two G1 wall-cabinet tasks.

    Same shape as _DrawerTopG1InitializationMixin, applied to the wall
    cabinet: robot x/y/yaw plus the two door hinges. Upstream leaves both
    tasks fully deterministic (open has no _on_reset, close pins [1, 1]),
    so every evaluation episode would replay one trajectory.

    The x band is per task because the two spawns differ by 0.44 m and the
    reach geometry recorded on each subclass is tight at both ends; y, yaw
    and the hinge band follow the drawer profile. The doors are sampled
    independently, so one leaf can start wider than the other.
    """

    _G1_ID_PROFILE = "g1_id_v1"
    _G1_ID_RESET_X_BOUNDS: tuple[float, float]
    _G1_ID_RESET_Y_BOUNDS = (-0.05, 0.05)
    _G1_ID_RESET_YAW_BOUNDS = np.deg2rad((-5.0, 5.0))
    _G1_ID_WALL_STATE_BOUNDS: tuple[float, float]

    def _sample_reset_robot_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if self.initialization_profile != self._G1_ID_PROFILE:
            return super()._sample_reset_robot_pose()
        x = np.random.uniform(*self._G1_ID_RESET_X_BOUNDS)
        y = np.random.uniform(*self._G1_ID_RESET_Y_BOUNDS)
        yaw = np.random.uniform(*self._G1_ID_RESET_YAW_BOUNDS)
        return (
            np.array([x, y, 0.0], dtype=np.float64),
            Quaternion(axis=[0, 0, 1], angle=yaw).elements,
        )

    def _on_reset(self):
        if self.initialization_profile != self._G1_ID_PROFILE:
            return super()._on_reset()
        state = np.random.uniform(*self._G1_ID_WALL_STATE_BOUNDS, size=2)
        self._g1_id_sampled_wall_state = state.copy()
        self.cabinet_wall.set_state(state)

    def _on_reset_warmup_step(self):
        super()._on_reset_warmup_step()
        if self.initialization_profile == self._G1_ID_PROFILE:
            # Same hold as the drawer profile: the G1 settle drags a partly
            # open leaf.
            self.cabinet_wall.set_state(self._g1_id_sampled_wall_state)


class WallCupboardOpenG1(_WallCupboardG1InitializationMixin, WallCupboardOpen):
    """WallCupboardOpen against the lowered G1 wall cabinet."""

    _PRESET_PATH = PRESETS_PATH / "counter_base_wall_3x1_g1.yaml"
    RESET_ROBOT_POS = np.array([0.24, 0, 0])
    _G1_ID_RESET_X_BOUNDS = (0.22, 0.26)
    _G1_ID_WALL_STATE_BOUNDS = (0.0, 0.15)


class WallCupboardCloseG1(_WallCupboardG1InitializationMixin, WallCupboardClose):
    """WallCupboardClose against the lowered G1 wall cabinet.

    Spawns at the upstream -0.2, not WallCupboardOpenG1's 0.24: the reset
    opens both doors toward the robot, and from 0.24 the door edges sit
    0.20 m ahead of the settled pelvis, so the task would close in place.
    From -0.2 they are 0.64 m ahead, one step away, with no settled hand
    contact.
    """

    _PRESET_PATH = PRESETS_PATH / "counter_base_wall_3x1_g1.yaml"
    RESET_ROBOT_POS = np.array([-0.2, 0, 0])
    _G1_ID_RESET_X_BOUNDS = (-0.22, -0.18)
    _G1_ID_WALL_STATE_BOUNDS = (0.85, 1.0)


class CupboardsOpenAllG1(CupboardsOpenAll):
    """CupboardsOpenAll with the lowered G1 wall cabinet."""

    _PRESET_PATH = PRESETS_PATH / "counter_base_wall_3x1_g1.yaml"
    # Not the drawer-family 0.12: the raised arms need elbow room at the
    # ~1.2 m wall-door handles, and the task steps between fixtures anyway.
    RESET_ROBOT_POS = np.array([0.0, 0, 0])


class CupboardsCloseAllG1(CupboardsCloseAll):
    """CupboardsCloseAll with the lowered G1 wall cabinet."""

    _PRESET_PATH = PRESETS_PATH / "counter_base_wall_3x1_g1.yaml"
    RESET_ROBOT_POS = np.array([0.0, 0, 0])  # see CupboardsOpenAllG1
