"""Regression tests for G1-only task accessibility adaptations."""

from collections.abc import Iterator

import mujoco
import numpy as np
import pytest

from bigym import scene
from bigym.action_modes import JointPositionActionMode
from bigym.envs.dishwasher_cups import (
    DishwasherLoadCupsG1,
    DishwasherUnloadCupsG1,
    DishwasherUnloadCupsLongG1,
)
from bigym.envs.dishwasher_cutlery import (
    DishwasherLoadCutleryG1,
    DishwasherUnloadCutleryG1,
    DishwasherUnloadCutleryLongG1,
)
from bigym.envs.dishwasher_plates import (
    DishwasherLoadPlatesG1,
    DishwasherUnloadPlatesG1,
    DishwasherUnloadPlatesLongG1,
)
from bigym.envs.manipulation import StackBlocksG1
from bigym.envs.pick_and_place import (
    FlipSandwichG1,
    PickBox,
    PickBoxG1,
    RemoveSandwichG1,
    SaucepanToHob,
    SaucepanToHobG1,
    StoreBox,
    StoreBoxG1,
    ToastSandwichG1,
)
from bigym.envs.props.cabintets import BaseCabinet
from bigym.envs.props.items import Box, LightBox
from bigym.envs.props.tables import Table
from bigym.robots.configs.g1 import G1Dex1
from bigym.utils.physics_utils import has_collided_collections


@pytest.fixture(scope="module")
def pick_box_g1() -> Iterator[PickBoxG1]:
    env = PickBoxG1(
        action_mode=JointPositionActionMode(floating_base=True),
        robot_cls=G1Dex1,
    )
    try:
        yield env
    finally:
        env.close()


@pytest.fixture(scope="module")
def stack_blocks_g1() -> Iterator[StackBlocksG1]:
    env = StackBlocksG1(
        action_mode=JointPositionActionMode(floating_base=True),
        robot_cls=G1Dex1,
    )
    try:
        yield env
    finally:
        env.close()


def test_light_box_is_g1_only() -> None:
    """The base task classes keep the stock 5 kg box; the G1 subclasses carry the 3 kg one."""
    assert PickBox._BOX_CLS is Box
    assert StoreBox._BOX_CLS is Box
    assert PickBoxG1._BOX_CLS is LightBox
    assert StoreBoxG1._BOX_CLS is LightBox


def test_pick_box_g1_settles_on_low_side_table(pick_box_g1: PickBoxG1) -> None:
    pick_box_g1.reset(seed=620000)
    pick_box_g1.simulation.step(500)
    pick_box_g1.simulation.forward()

    table = pick_box_g1._preset.get_props(Table)[0]
    box_position = pick_box_g1.box.get_position()
    assert box_position[2] == pytest.approx(0.625, abs=0.003)
    assert abs(box_position[0] - 0.70) < 0.06 and abs(box_position[1] + 1.0) < 0.06
    assert pick_box_g1.box.is_colliding(table)


def test_pick_box_g1_box_is_stock_geometry_at_3kg(pick_box_g1: PickBoxG1) -> None:
    """The G1 box keeps the stock collider and friction; only the mass drops."""
    box_bodies = [
        body
        for body in scene.descendants(pick_box_g1.box.body, "bodies")
        if body.name.split("/")[-1] == "box"
    ]
    assert len(box_bodies) == 1
    box_mass = pick_box_g1.model.bind(box_bodies[0]).mass.item()
    assert box_mass == pytest.approx(3.0)

    colliders = [
        geom for geom in pick_box_g1.box.geoms if geom.name.split("/")[-1] == "collider"
    ]
    assert len(colliders) == 1
    bound = pick_box_g1.model.bind(colliders[0])
    np.testing.assert_allclose(bound.size, [0.125, 0.1, 0.06])
    assert float(bound.friction[0]) == pytest.approx(1.5)
    assert not any("handle" in g.name.split("/")[-1] for g in pick_box_g1.box.geoms)


def test_stack_blocks_g1_reset_stays_near_exposed_edges(
    stack_blocks_g1: StackBlocksG1,
) -> None:
    for seed in range(16):
        stack_blocks_g1.reset(seed=seed)
        block_positions = np.asarray(
            [block.get_position() for block in stack_blocks_g1.blocks]
        )
        target_position = stack_blocks_g1.data.bind(stack_blocks_g1.target).xpos.copy()

        assert np.all((0.525 <= block_positions[:, 0]))
        assert np.all((block_positions[:, 0] <= 0.625))
        assert np.all(np.abs(block_positions[:, 1]) <= 0.275)

        # Table two's exposed far edge is x=1.70: the pad is always 10-18 cm in.
        target_edge_inset = 1.70 - target_position[0]
        assert 0.10 <= target_edge_inset <= 0.18
        assert abs(target_position[1]) <= 0.08

        requested_spawn_xy = StackBlocksG1.RESET_ROBOT_POS[:2]
        distances = np.linalg.norm(
            block_positions[:, :2] - requested_spawn_xy,
            axis=1,
        )
        assert np.all(distances <= 0.525)


G1_DISHWASHER_TASKS = [
    DishwasherLoadCupsG1,
    DishwasherUnloadCupsG1,
    DishwasherUnloadCupsLongG1,
    DishwasherLoadCutleryG1,
    DishwasherUnloadCutleryG1,
    DishwasherUnloadCutleryLongG1,
    DishwasherLoadPlatesG1,
    DishwasherUnloadPlatesG1,
    DishwasherUnloadPlatesLongG1,
]


@pytest.mark.parametrize("env_cls", G1_DISHWASHER_TASKS, ids=lambda c: c.__name__)
def test_dishwasher_g1_hands_start_off_counter(env_cls) -> None:
    """The whole G1 dishwasher family spawns at the upstream x=0.

    See dishwasher_cups.G1_DISHWASHER_SPAWN.
    """
    np.testing.assert_allclose(env_cls.RESET_ROBOT_POS, [0.0, -0.6, 0.0])
    env = env_cls(
        action_mode=JointPositionActionMode(floating_base=True),
        robot_cls=G1Dex1,
    )
    try:
        env.reset(seed=620000)
        env.simulation.step(500)
        env.simulation.forward()
        cabinets = env._preset.get_props(BaseCabinet)
        assert cabinets
        for gripper in env.robot.grippers.values():
            for cabinet in cabinets:
                assert not has_collided_collections(
                    env.simulation,
                    scene.descendants(gripper.body, "geoms"),
                    cabinet.colliders,
                )
    finally:
        env.close()


@pytest.fixture(scope="module")
def remove_sandwich_g1() -> Iterator[RemoveSandwichG1]:
    env = RemoveSandwichG1(
        action_mode=JointPositionActionMode(floating_base=True),
        robot_cls=G1Dex1,
    )
    try:
        yield env
    finally:
        env.close()


def test_sandwich_g1_spawn_faces_pan_handle_and_spatula(
    remove_sandwich_g1: RemoveSandwichG1,
) -> None:
    """The G1 sandwich spawn (0.10, 0.25) keeps handle and spatula in reach.

    The pan handle points along the counter to the robot's left (tip at
    y ~0.43-0.46) and the spatula lies at y ~0.19. Standing 0.15 m off the
    counter's front edge (pelvis x = 0.35, i.e. 0.25 m ahead of the spawn)
    the right shoulder must be within a bent-arm reach (0.40 m) of the
    spatula, and the left shoulder within 0.46 m of the handle geom's centre
    (0.42 m shoulder-to-gripper-base plus the finger pads; the IK sweep
    places the gripper base within 1.5 cm at the -100 deg tail) at every
    seed, which needs the +-15 deg pan yaw jitter; the reset settle must
    not shove the robot or the pan (IK sweep, 12 seeds).
    """
    env = remove_sandwich_g1
    for cls in (ToastSandwichG1, FlipSandwichG1, RemoveSandwichG1):
        assert np.allclose(cls.RESET_ROBOT_POS, [0.10, 0.25, 0]), cls
        assert np.allclose(cls._PAN_ROT_BOUNDS, np.deg2rad([0, 0, 15])), cls

    model = env.model
    data = env.data

    def body_position(suffix: str) -> np.ndarray:
        for body in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            if name.endswith(suffix):
                return np.asarray(data.xpos[body])
        raise AssertionError(suffix)

    def geom_position(suffix: str) -> np.ndarray:
        for geom in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
            if name.endswith(suffix):
                return np.asarray(data.geom_xpos[geom])
        raise AssertionError(suffix)

    step_forward = np.array([0.25, 0.0, 0.0])
    for seed in range(620000, 620012):
        env.reset(seed=seed)
        env.simulation.step(200)
        env.simulation.forward()

        pelvis = body_position("/pelvis")
        assert np.allclose(pelvis[:2], [0.10, 0.25], atol=0.01), (seed, pelvis)
        assert env.pan.is_colliding(env.cabinet_base.hob), seed
        pan = env.pan.get_position()
        assert abs(pan[0] - 0.72) < 0.05 and abs(pan[1] - 0.16) < 0.05, (seed, pan)
        quat = env.pan.get_quaternion()
        yaw = np.degrees(2 * np.arctan2(quat[3], quat[0]))
        assert -106 <= yaw <= -74, (seed, yaw)

        handle = geom_position("collider_handle_1")
        spatula = env.spatula.get_position()
        left_shoulder = body_position("left_shoulder_pitch_link") + step_forward
        right_shoulder = body_position("right_shoulder_pitch_link") + step_forward
        assert np.linalg.norm(handle - left_shoulder) < 0.46, (seed, handle)
        assert np.linalg.norm(spatula - right_shoulder) < 0.40, (seed, spatula)


def test_saucepan_to_hob_g1_pan_sits_at_shelf_front() -> None:
    """The G1 saucepan spawns at the front of the shelf, fully on it.

    Upstream keeps x=0.85; the G1 subclass moves it to 0.77 so the handle
    is within a squatting G1's reach. Under the +-0.05 jitter
    the pan must stay on the shelf (front edge x=0.652) and never touch the
    door.
    """
    np.testing.assert_allclose(SaucepanToHob._SAUCEPAN_POS, [0.85, 0.1, 0.5])
    np.testing.assert_allclose(SaucepanToHobG1._SAUCEPAN_POS, [0.77, 0.1, 0.5])
    env = SaucepanToHobG1(
        action_mode=JointPositionActionMode(floating_base=True),
        robot_cls=G1Dex1,
    )
    try:
        model = env.model
        data = env.data

        def body_name(geom: int) -> str:
            name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom]
            )
            return (name or "").split("/")[-1]

        for seed in range(620000, 620010):
            env.reset(seed=seed)
            env.simulation.step(300)
            env.simulation.forward()
            pan = env.saucepan.get_position()
            assert pan[2] == pytest.approx(0.482, abs=0.005), (seed, pan)
            assert pan[0] - 0.067 >= 0.652, (seed, pan)
            assert env.saucepan.is_colliding(env.cabinet_base.shelf), seed
            partners = set()
            for i in range(data.ncon):
                contact = data.contact[i]
                names = {body_name(contact.geom1), body_name(contact.geom2)}
                if "saucepan" in names:
                    partners |= names - {"saucepan"}
            assert partners <= {"shelf"}, (seed, partners)
    finally:
        env.close()
