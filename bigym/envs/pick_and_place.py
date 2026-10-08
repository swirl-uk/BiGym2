"""Pick and place tasks."""

import mujoco
import numpy as np
from pyquaternion import Quaternion

from bigym.bigym_env import BiGymEnv
from bigym.const import PRESETS_PATH
from bigym.envs.props.cabintets import BaseCabinet, WallCabinet
from bigym.envs.props.cutlery import Spatula
from bigym.envs.props.items import Box, LightBox, Sandwich
from bigym.envs.props.kitchenware import ChoppingBoard, Mug, Pan, Saucepan
from bigym.envs.props.prop import Prop
from bigym.utils.env_utils import get_random_points_on_plane
from bigym.utils.physics_utils import get_colliders


class PutCups(BiGymEnv):
    """Put cups in the wall cabinet."""

    _PRESET_PATH = PRESETS_PATH / "counter_base_wall_1x1.yaml"

    _CUPS_COUNT = 2

    _CUPS_POS = np.array([0.8, 0, 1])
    _CUPS_ROT = np.deg2rad(180)
    _CUPS_STEP = 0.15
    _CUPS_POS_EXTENTS = np.array([0.1, 0.25])
    _CUPS_POS_BOUNDS = np.array([0.03, 0.03, 0])
    _CUPS_ROT_BOUNDS = np.deg2rad(30)

    def _initialize_env(self):
        self.cabinet_base = self._preset.get_props(BaseCabinet)[0]
        self.cabinet_wall = self._preset.get_props(WallCabinet)[0]
        self.cups = [Mug(self.simulation) for _ in range(self._CUPS_COUNT)]

    def _success(self) -> bool:
        for cup in self.cups:
            if not cup.is_colliding(self.cabinet_wall.shelf_bottom):
                return False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(cup, side):
                    return False
        return True

    def _on_reset(self):
        points = get_random_points_on_plane(
            len(self.cups),
            self._CUPS_POS,
            self._CUPS_POS_EXTENTS,
            self._CUPS_STEP,
            self._CUPS_POS_BOUNDS,
        )
        for cup, point in zip(self.cups, points, strict=True):
            cup.set_position(point)
            angle = np.random.uniform(-self._CUPS_ROT_BOUNDS, self._CUPS_ROT_BOUNDS)
            cup.set_quaternion(
                Quaternion(axis=[0, 0, 1], angle=self._CUPS_ROT + angle).elements
            )


class TakeCups(PutCups):
    """Take cups from the wall cupboard."""

    _CUPS_POS = np.array([1.05, 0, 1.5])

    def _success(self) -> bool:
        for cup in self.cups:
            if not cup.is_colliding(self.cabinet_base.counter):
                return False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(cup, side):
                    return False
        return True


class StoreBox(BiGymEnv):
    """Put box in the cupboard task."""

    _PRESET_PATH = PRESETS_PATH / "cabinet_door.yaml"

    _BOX_POS = np.array([0.8, 0, 1])
    _BOX_POS_BOUNDS = np.array([0.03, 0.03, 0])
    _BOX_ROT_BOUNDS = np.deg2rad(180)
    _BOX_CLS = Box

    def _initialize_env(self):
        self.cabinet_base = self._preset.get_props(BaseCabinet)[0]
        self.box = self._BOX_CLS(self.simulation, True)

    def _on_reset(self):
        offset = np.random.uniform(-self._BOX_POS_BOUNDS, self._BOX_POS_BOUNDS)
        self.box.set_position(self._BOX_POS + offset, True)
        angle = np.random.uniform(-self._BOX_ROT_BOUNDS, self._BOX_ROT_BOUNDS)
        self.box.set_quaternion(Quaternion(axis=[0, 0, 1], angle=angle).elements)

    def _success(self) -> bool:
        if not self.box.is_colliding(self.cabinet_base.shelf):
            return False
        for side in self.robot.grippers:
            if self.robot.is_gripper_holding_object(self.box, side):
                return False
        return True


class PickBox(StoreBox):
    """Pick up box from and place it on the counter task."""

    _BOX_POS = np.array([0.8, -1, 0.2])
    _BOX_QUAT = Quaternion(axis=[0, 1, 0], degrees=90)
    # Upstream success only asks for contact between the box and the counter,
    # which a box pressed against the counter's side edge satisfies; the box
    # must also rest on the counter top.
    _REQUIRE_BOX_ON_COUNTER = True
    _ON_COUNTER_Z_TOL = 0.03

    def _world_aabb(self, geoms) -> tuple[np.ndarray, np.ndarray]:
        """World AABB of box/mesh geoms (meshes via their bounding sphere)."""
        model, data = self.model, self.data
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for g in geoms:
            gid = g.id
            pos = data.geom_xpos[gid]
            half = model.geom_size[gid]
            if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
                rot = data.geom_xmat[gid].reshape(3, 3)
                corners = np.array(
                    [
                        [sx, sy, sz]
                        for sx in (-half[0], half[0])
                        for sy in (-half[1], half[1])
                        for sz in (-half[2], half[2])
                    ]
                )
                pts = pos + corners @ rot.T
                lo, hi = np.minimum(lo, pts.min(0)), np.maximum(hi, pts.max(0))
            else:
                r = float(half.max())
                lo, hi = np.minimum(lo, pos - r), np.maximum(hi, pos + r)
        return lo, hi

    def _box_on_counter_top(self) -> bool:
        """Box centre inside the counter footprint, bottom face on its top."""
        clo, chi = self._world_aabb(get_colliders(self.cabinet_base.counter))
        blo, _ = self._world_aabb(self.box.colliders)
        centre = np.asarray(self.box.get_position())
        inside = bool(clo[0] <= centre[0] <= chi[0] and clo[1] <= centre[1] <= chi[1])
        return inside and abs(float(blo[2]) - float(chi[2])) <= self._ON_COUNTER_Z_TOL

    def _success(self) -> bool:
        if not self.box.is_colliding(self.cabinet_base.counter):
            return False
        for side in self.robot.grippers:
            if self.robot.is_gripper_holding_object(self.box, side):
                return False
        if self._REQUIRE_BOX_ON_COUNTER and not self._box_on_counter_top():
            return False
        return True

    def _on_reset(self):
        offset = np.random.uniform(-self._BOX_POS_BOUNDS, self._BOX_POS_BOUNDS)
        self.box.set_position(self._BOX_POS + offset, True)
        angle = np.random.uniform(-self._BOX_ROT_BOUNDS, self._BOX_ROT_BOUNDS)
        quat = self._BOX_QUAT
        quat *= Quaternion(axis=[1, 0, 0], angle=angle)
        self.box.set_quaternion(quat.elements)


class SaucepanToHob(BiGymEnv):
    """Take saucepan from cabinet and place it to hob."""

    _PRESET_PATH = PRESETS_PATH / "cabinet_hob.yaml"

    _SAUCEPAN_POS = np.array([0.85, 0.1, 0.5])
    _SAUCEPAN_QUAT = Quaternion(axis=[0, 0, 1], degrees=90)
    _SAUCEPAN_POS_BOUNDS = np.array([0.05, 0.05, 0])
    _SAUCEPAN_ROT_BOUNDS = np.deg2rad([0, 0, 20])

    def _initialize_env(self):
        self.cabinet_base = self._preset.get_props(BaseCabinet)[0]
        self.saucepan = Saucepan(self.simulation)

    def _success(self) -> bool:
        if not self.saucepan.is_colliding(self.cabinet_base.hob):
            return False
        for side in self.robot.grippers:
            if self.robot.is_gripper_holding_object(self.saucepan, side):
                return False
        return True

    def _on_reset(self):
        self.saucepan.set_pose(
            self._SAUCEPAN_POS,
            self._SAUCEPAN_QUAT.elements,
            self._SAUCEPAN_POS_BOUNDS,
            self._SAUCEPAN_ROT_BOUNDS,
        )


class StoreKitchenware(BiGymEnv):
    """Put all kitchenware to cupboard."""

    _PRESET_PATH = PRESETS_PATH / "cabinet_hob.yaml"

    _ITEMS = [Saucepan, Pan]
    _ITEMS_QUAT = Quaternion(axis=[0, 0, 1], degrees=15)
    _ITEMS_POS_BOUNDS = np.array([0.02, 0.02, 0])
    _ITEMS_ROT_BOUNDS = np.deg2rad([0, 0, 30])

    def _initialize_env(self):
        self.cabinet_base = self._preset.get_props(BaseCabinet)[0]
        self.items: list[Prop] = [item(self.simulation) for item in self._ITEMS]

    def _success(self) -> bool:
        for item in self.items:
            if not item.is_colliding(self.cabinet_base.shelf):
                return False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(item, side):
                    return False
        return True

    def _on_reset(self):
        sites = [self.cabinet_base.sites[0], self.cabinet_base.sites[2]]
        np.random.shuffle(sites)  # ty: ignore[invalid-argument-type]
        for item, site in zip(self.items, sites, strict=True):
            item.set_pose(
                self.data.bind(site).xpos,
                self._ITEMS_QUAT.elements,
                self._ITEMS_POS_BOUNDS,
                self._ITEMS_ROT_BOUNDS,
            )


class ToastSandwich(BiGymEnv):
    """Move sandwich on the frying pan."""

    _PRESET_PATH = PRESETS_PATH / "counter_base_2_hob.yaml"

    _PAN_QUAT = Quaternion(axis=[0, 0, 1], degrees=-90)
    _PAN_POS_BOUNDS = np.array([0.02, 0.02, 0])
    _PAN_ROT_BOUNDS = np.deg2rad([0, 0, 30])

    _SPATULA_OFFSET = np.array([-0.02, 0.02, 0.08])
    _SPATULA_QUAT = Quaternion(axis=[0, 0, 1], degrees=90)

    _BOARD_POS = np.array([0.7, -0.6, 0.88])
    _BOARD_ROT_BOUNDS = np.deg2rad([0, 0, 5])

    _TOLERANCE = np.deg2rad(10)
    _TOASTED = False
    _ROUNDED = False

    _SANDWICH_OFFSET = np.array([0, 0, 0.05])
    _SANDWICH_POS_BOUNDS = np.array([0.05, 0.05, 0])
    _SANDWICH_ROT_BOUNDS = np.deg2rad([0, 0, 180])

    @property
    def _sandwich_anchor(self) -> Prop:
        return self.board

    def _initialize_env(self):
        self.cabinet_base = self._preset.get_props(BaseCabinet)[0]
        self.pan = Pan(self.simulation)
        self.spatula = Spatula(self.simulation)
        self.board = ChoppingBoard(self.simulation)
        self.sandwich = Sandwich(
            self.simulation, toasted=self._TOASTED, rounded_collider=self._ROUNDED
        )

    def _on_reset(self):
        site = self.cabinet_base.sites[0]
        self.pan.set_pose(
            self.data.bind(site).xpos,
            self._PAN_QUAT.elements,
            self._PAN_POS_BOUNDS,
            self._PAN_ROT_BOUNDS,
        )
        self.spatula.set_pose(
            self.pan.get_position() + self._SPATULA_OFFSET,
            self._SPATULA_QUAT.elements,
        )
        self.board.set_pose(self._BOARD_POS, rotation_bounds=self._BOARD_ROT_BOUNDS)
        self.sandwich.set_pose(
            self._sandwich_anchor.get_position() + self._SANDWICH_OFFSET,
            position_bounds=self._SANDWICH_POS_BOUNDS,
            rotation_bounds=self._SANDWICH_ROT_BOUNDS,
        )

    def _success(self) -> bool:
        up = np.array([0, 0, 1])
        sandwich_up = Quaternion(self.sandwich.get_quaternion()).rotate(up)
        angle_to_up = np.arccos(np.clip(np.dot(sandwich_up, up), -1.0, 1.0))
        angle_to_down = np.arccos(np.clip(np.dot(sandwich_up, -up), -1.0, 1.0))
        if not (angle_to_up <= self._TOLERANCE or angle_to_down <= self._TOLERANCE):
            return False
        if not self.pan.is_colliding(self.cabinet_base.hob):
            return False
        if not self.sandwich.is_colliding(self.pan):
            return False
        return True

    def _fail(self) -> bool:
        if super()._fail():
            return True
        for side in self.robot.grippers:
            if self.robot.is_gripper_holding_object(self.sandwich, side):
                return True
        return False


class FlipSandwich(ToastSandwich):
    """Flip sandwich using spatula."""

    _SANDWICH_OFFSET = np.array([0, 0, 0.04])
    _SANDWICH_POS_BOUNDS = np.array([0.01, 0.01, 0])
    _SANDWICH_ROT_BOUNDS = np.deg2rad([0, 0, 180])

    _ROUNDED = True

    @property
    def _sandwich_anchor(self) -> Prop:
        return self.pan

    def _success(self) -> bool:
        up = np.array([0, 0, 1])
        sandwich_up = Quaternion(self.sandwich.get_quaternion()).rotate(up)
        angle_to_down = np.arccos(np.clip(np.dot(sandwich_up, -up), -1.0, 1.0))
        if angle_to_down > self._TOLERANCE:
            return False
        if not self.pan.is_colliding(self.cabinet_base.hob):
            return False
        if not self.sandwich.is_colliding(self.pan):
            return False
        return True


class RemoveSandwich(FlipSandwich):
    """Remove sandwich from the frying pan."""

    _TOASTED = True

    def _success(self) -> bool:
        up = np.array([0, 0, 1])
        sandwich_up = Quaternion(self.sandwich.get_quaternion()).rotate(up)
        angle_to_up = np.arccos(np.clip(np.dot(sandwich_up, up), -1.0, 1.0))
        angle_to_down = np.arccos(np.clip(np.dot(sandwich_up, -up), -1.0, 1.0))
        if not (angle_to_up <= self._TOLERANCE or angle_to_down <= self._TOLERANCE):
            return False
        if not self.sandwich.is_colliding(self.board):
            return False
        return True


# G1 counter/hob tasks retain the original work-surface heights and move
# the spawn forward. Multi-station tasks still require walking/stepping.
# The variants below also include geometry changes: PickBox raises pickup
# onto a low side table, PickBox/StoreBox use a 3 kg box (Dex1 has no palm, so
# the box is carried by squeezing it between the open grippers; 5 kg slips
# out while walking), and PutCups/TakeCups lower the wall cabinet. PickBox
# keeps the upstream spawn and requires approaching the box at y=-1.


# The sandwich family spawns the G1 0.25 m to the left of the upstream spawn
# so the left shoulder faces the pan handle (pointing +y along the counter,
# tip at y ~0.45) and the right shoulder faces the spatula; from y = 0 the
# handle sits at the 0.42 m shoulder-to-gripper limit. The pan yaw jitter is
# narrowed from +-30 to +-15 deg, which keeps the handle within reach at every
# seed. x stays at 0.10 because further forward the reset settle shoves the
# robot against the counter; the operator walks the last 0.25 m.
SANDWICH_G1_SPAWN = np.array([0.10, 0.25, 0])
SANDWICH_G1_PAN_ROT_BOUNDS = np.deg2rad([0, 0, 15])


class ToastSandwichG1(ToastSandwich):
    """ToastSandwich with the G1 spawn facing the pan handle and spatula."""

    RESET_ROBOT_POS = SANDWICH_G1_SPAWN
    _PAN_ROT_BOUNDS = SANDWICH_G1_PAN_ROT_BOUNDS


class FlipSandwichG1(FlipSandwich):
    """FlipSandwich with the G1 spawn facing the pan handle and spatula."""

    RESET_ROBOT_POS = SANDWICH_G1_SPAWN
    _PAN_ROT_BOUNDS = SANDWICH_G1_PAN_ROT_BOUNDS


class RemoveSandwichG1(RemoveSandwich):
    """RemoveSandwich with the G1 spawn facing the pan handle and spatula."""

    RESET_ROBOT_POS = SANDWICH_G1_SPAWN
    _PAN_ROT_BOUNDS = SANDWICH_G1_PAN_ROT_BOUNDS


class SaucepanToHobG1(SaucepanToHob):
    """SaucepanToHob with the saucepan moved to the front of the shelf.

    Upstream parks the saucepan at x=0.85, 0.25 m behind the door face,
    which at the G1 squat floor (0.4 m) puts the handle at the very end of
    a straight arm. x=0.77 is as far forward as the pan can go while staying
    fully on the shelf under the +-0.05 jitter.
    """

    RESET_ROBOT_POS = np.array([0.24, 0, 0])  # fingertip clearance, see FlipCupG1
    _SAUCEPAN_POS = np.array([0.77, 0.1, 0.5])


class StoreKitchenwareG1(StoreKitchenware):
    """StoreKitchenware with the spawn pulled in for the G1 workspace."""

    RESET_ROBOT_POS = np.array([0.26, 0, 0])  # fingertip clearance, see FlipCupG1


class PickBoxG1(PickBox):
    """PickBox with the 3 kg box on a low side table."""

    _PRESET_PATH = PRESETS_PATH / "cabinet_door_pick_box_g1.yaml"
    # The upright 0.25 m-tall box rests with its centre 0.125 m above the
    # side table's 0.50 m top, i.e. z = 0.625 m, 0.10 m inside the table's
    # front edge (x = 0.6): reachable with a moderate squat plus torso pitch,
    # and close enough to the edge that the knees clear the table. Spawn it
    # 0.5 mm above the resting height rather than dropping it, which keeps
    # the solver stable. The preset also sinks the cabinet 0.15 m (counter
    # top 0.71 m).
    _BOX_POS = np.array([0.70, -1.0, 0.6255])
    _BOX_CLS = LightBox


class StoreBoxG1(StoreBox):
    """StoreBox with the spawn pulled in and the 3 kg box."""

    # Furthest-forward spawn that settles cleanly: at 0.25 the settled Dex1
    # fingers end up inside the counter front, and at 0.23 a transient touch
    # kicks the pelvis back. 0.21 settles at x~0.14 with 75 mm
    # finger-to-counter clearance and the box 0.67 m ahead.
    RESET_ROBOT_POS = np.array([0.21, 0, 0])
    _BOX_CLS = LightBox


class PutCupsG1(PutCups):
    """PutCups into the lowered G1 wall cabinet (shelf bottom ~1.15 m)."""

    _PRESET_PATH = PRESETS_PATH / "counter_base_wall_1x1_g1.yaml"
    RESET_ROBOT_POS = np.array([0.24, 0, 0])


class TakeCupsG1(TakeCups):
    """TakeCups from the lowered G1 wall cabinet.

    The cup spawn moves with the cabinet (-0.2 m x, -0.32 m z).
    """

    _PRESET_PATH = PRESETS_PATH / "counter_base_wall_1x1_g1.yaml"
    RESET_ROBOT_POS = np.array([0.24, 0, 0])
    _CUPS_POS = np.array([0.85, 0, 1.18])
