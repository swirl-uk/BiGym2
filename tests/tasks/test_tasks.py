"""Test tasks."""

import numpy as np
import pytest

from bigym.action_modes import ActionMode, JointPositionActionMode
from bigym.bigym_env import MAX_DISTANCE_FROM_ORIGIN, BiGymEnv
from bigym.envs.cupboards import (
    CupboardsCloseAll,
    CupboardsOpenAll,
    DrawersAllClose,
    DrawersAllOpen,
    DrawerTopClose,
    DrawerTopOpen,
    WallCupboardClose,
    WallCupboardOpen,
)
from bigym.envs.dishwasher import (
    DishwasherClose,
    DishwasherCloseTrays,
    DishwasherOpen,
    DishwasherOpenTrays,
)
from bigym.envs.dishwasher_cups import (
    DishwasherLoadCups,
    DishwasherUnloadCups,
    DishwasherUnloadCupsLong,
)
from bigym.envs.dishwasher_cutlery import (
    DishwasherLoadCutlery,
    DishwasherUnloadCutlery,
    DishwasherUnloadCutleryLong,
)
from bigym.envs.dishwasher_plates import (
    DishwasherLoadPlates,
    DishwasherUnloadPlates,
    DishwasherUnloadPlatesLong,
)
from bigym.envs.groceries import GroceriesStoreLower, GroceriesStoreUpper
from bigym.envs.manipulation import FlipCup, FlipCutlery, StackBlocks
from bigym.envs.move_plates import MovePlate, MoveTwoPlates
from bigym.envs.pick_and_place import (
    FlipSandwich,
    PickBox,
    PutCups,
    RemoveSandwich,
    SaucepanToHob,
    StoreBox,
    StoreKitchenware,
    TakeCups,
    ToastSandwich,
)
from bigym.envs.reach_target import ReachTarget, ReachTargetDual, ReachTargetSingle
from bigym.utils.physics_utils import set_body_position

# Every floating-base task env.
ENVIRONMENTS: list[type[BiGymEnv]] = [
    ReachTarget,
    ReachTargetSingle,
    ReachTargetDual,
    StackBlocks,
    MovePlate,
    MoveTwoPlates,
    DishwasherOpen,
    DishwasherClose,
    DishwasherOpenTrays,
    DishwasherCloseTrays,
    DishwasherUnloadPlates,
    DishwasherUnloadPlatesLong,
    DishwasherLoadPlates,
    DishwasherUnloadCutlery,
    DishwasherUnloadCutleryLong,
    DishwasherLoadCutlery,
    DishwasherUnloadCups,
    DishwasherUnloadCupsLong,
    DishwasherLoadCups,
    DrawerTopOpen,
    DrawerTopClose,
    DrawersAllOpen,
    DrawersAllClose,
    WallCupboardOpen,
    WallCupboardClose,
    CupboardsOpenAll,
    CupboardsCloseAll,
    TakeCups,
    PutCups,
    FlipCup,
    FlipCutlery,
    PickBox,
    StoreBox,
    SaucepanToHob,
    StoreKitchenware,
    ToastSandwich,
    FlipSandwich,
    RemoveSandwich,
    GroceriesStoreLower,
    GroceriesStoreUpper,
]


@pytest.mark.parametrize("env_class", ENVIRONMENTS)
@pytest.mark.parametrize(
    "action_mode_class",
    [JointPositionActionMode],
)
@pytest.mark.slow
class TestEnvs:
    def test_terminates_when_robot_out_of_bounds(
        self, env_class: type[BiGymEnv], action_mode_class: type[ActionMode]
    ):
        action_mode = action_mode_class(floating_base=True)
        env = env_class(action_mode=action_mode)
        obs, _ = env.reset()
        set_body_position(
            env.model, env.robot.pelvis, np.array([MAX_DISTANCE_FROM_ORIGIN] * 3)
        )
        obs, rew, term, trunc, info = env.step(env.action_space.sample())
        assert term
        env.close()
