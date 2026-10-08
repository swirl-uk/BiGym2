"""The task registry: every benchmark task and its official configuration.

``TASKS`` maps each task name to a :class:`TaskSpec`: the environment class,
the episode budget and the fields where the task's official config departs
from the :class:`~bigym.loco.config.EnvConfig` defaults. ``task_config(name)``
is that official config; ``bigym.loco.make(name)`` builds it. The canonical
names are the G1 tasks; there is deliberately no alias layer.

Tasks from another package: :func:`register_task` adds one to ``TASKS``,
and a name with a colon (``"pkg.module:ATTR"``) names a :class:`TaskSpec`
in an importable module.

Budgets: G1 budgets started from the upstream floating-base table. Tasks
marked ``data_derived`` have since had their budget replaced by
:data:`BUDGET_RULE` applied to the published 60-demo batch;
``budget_provenance(name)`` tells those apart from the remaining upstream
placeholders, which are not evidence of a sufficient controller-in-the-loop
budget.

Exceptions: the six tasks collected before the torso-pitch command existed
keep it off (their demonstrations are 20-dim; every other task is 21-dim).
The three reach tasks pin ``reach_tolerance=0.05``: the pinch centre must be
inside the target sphere, where the class default of 0.1 passes on a graze.
The two top-drawer tasks use the seeded ``g1_id_v1`` reset distribution
(robot x/y/yaw and drawer state). The wall-cupboard tasks reset
deterministically, as upstream: their demonstrations were collected that way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from bigym.bigym_env import BiGymEnv
from bigym.envs.cupboards import (
    CupboardsCloseAllG1,
    CupboardsOpenAllG1,
    DrawersAllCloseG1,
    DrawersAllOpenG1,
    DrawerTopCloseG1,
    DrawerTopOpenG1,
    WallCupboardCloseG1,
    WallCupboardOpenG1,
)
from bigym.envs.dishwasher import (
    DishwasherCloseG1,
    DishwasherCloseTraysG1,
    DishwasherOpenG1,
    DishwasherOpenTraysG1,
)
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
from bigym.envs.groceries import (
    GroceriesStoreLowerG1,
    GroceriesStoreUpperG1,
)
from bigym.envs.manipulation import (
    FlipCupG1,
    FlipCutleryG1,
    StackBlocksG1,
)
from bigym.envs.move_plates import (
    MovePlateG1,
    MoveTwoPlatesG1,
)
from bigym.envs.pick_and_place import (
    FlipSandwichG1,
    PickBoxG1,
    PutCupsG1,
    RemoveSandwichG1,
    SaucepanToHobG1,
    StoreBoxG1,
    StoreKitchenwareG1,
    TakeCupsG1,
    ToastSandwichG1,
)
from bigym.envs.reach_target import (
    ReachTargetDualG1,
    ReachTargetG1,
    ReachTargetSingleG1,
)
from bigym.loco.config import EnvConfig
from bigym.loco.objref import import_object, is_object_ref


@dataclass(frozen=True)
class TaskSpec:
    """One registered task: its class, budget and official-config differences.

    ``env_cls`` is a :class:`~bigym.bigym_env.BiGymEnv` subclass; the env
    builds it with the ``BiGymEnv`` constructor arguments.
    """

    env_cls: type[BiGymEnv]
    # Episode budget in env steps (outer steps = budget / demo_down_sample_rate).
    episode_length: int
    # True when the budget came from the published demos by BUDGET_RULE.
    data_derived: bool = False
    # EnvConfig fields where this task's official config departs from the
    # defaults, in EnvConfig.override form.
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def config(self) -> EnvConfig:
        """The task's official configuration."""
        return EnvConfig(episode_length=self.episode_length).override(self.overrides)


def _task(
    env_cls: type[BiGymEnv],
    episode_length: int,
    *,
    data_derived: bool = False,
    **overrides: Any,
) -> TaskSpec:
    return TaskSpec(env_cls, episode_length, data_derived, overrides)


# The six tasks recorded before the torso-pitch command existed.
NO_PITCH: dict[str, Any] = {"controller": {"pitch_command": False}}
REACH = dict(NO_PITCH, reach_tolerance=0.05)
TOP_DRAWER = dict(NO_PITCH, initialization_profile="g1_id_v1")

TASKS: dict[str, TaskSpec] = {
    "reach_target_multi_modal": _task(ReachTargetG1, 7000, data_derived=True, **REACH),
    "reach_target_single": _task(ReachTargetSingleG1, 9000, data_derived=True, **REACH),
    "reach_target_dual": _task(ReachTargetDualG1, 7000, data_derived=True, **REACH),
    "stack_blocks": _task(StackBlocksG1, 87500, data_derived=True),
    "move_plate": _task(MovePlateG1, 17000, data_derived=True, **NO_PITCH),
    "move_two_plates": _task(MoveTwoPlatesG1, 23000, data_derived=True),
    "flip_cup": _task(FlipCupG1, 18500, data_derived=True),
    "flip_cutlery": _task(FlipCutleryG1, 19500, data_derived=True),
    "dishwasher_open": _task(DishwasherOpenG1, 7500),
    "dishwasher_close": _task(DishwasherCloseG1, 46500, data_derived=True),
    "dishwasher_open_trays": _task(DishwasherOpenTraysG1, 9500),
    "dishwasher_close_trays": _task(DishwasherCloseTraysG1, 8000),
    "dishwasher_load_cups": _task(DishwasherLoadCupsG1, 20000, data_derived=True),
    "dishwasher_unload_cups": _task(DishwasherUnloadCupsG1, 10000),
    "dishwasher_unload_cups_long": _task(DishwasherUnloadCupsLongG1, 18000),
    "dishwasher_load_cutlery": _task(DishwasherLoadCutleryG1, 26500, data_derived=True),
    "dishwasher_unload_cutlery": _task(DishwasherUnloadCutleryG1, 15500),
    "dishwasher_unload_cutlery_long": _task(DishwasherUnloadCutleryLongG1, 18000),
    "dishwasher_load_plates": _task(DishwasherLoadPlatesG1, 31500, data_derived=True),
    "dishwasher_unload_plates": _task(DishwasherUnloadPlatesG1, 20000),
    "dishwasher_unload_plates_long": _task(DishwasherUnloadPlatesLongG1, 26000),
    "drawer_top_open": _task(DrawerTopOpenG1, 13500, data_derived=True, **TOP_DRAWER),
    "drawer_top_close": _task(DrawerTopCloseG1, 8500, data_derived=True, **TOP_DRAWER),
    "drawers_open_all": _task(DrawersAllOpenG1, 12000),
    "drawers_close_all": _task(DrawersAllCloseG1, 5000),
    "wall_cupboard_open": _task(WallCupboardOpenG1, 14500, data_derived=True),
    "wall_cupboard_close": _task(WallCupboardCloseG1, 17000, data_derived=True),
    "cupboards_open_all": _task(CupboardsOpenAllG1, 22500),
    "cupboards_close_all": _task(CupboardsCloseAllG1, 15500),
    "take_cups": _task(TakeCupsG1, 10500),
    "put_cups": _task(PutCupsG1, 30000, data_derived=True),
    "pick_box": _task(PickBoxG1, 44000, data_derived=True),
    "store_box": _task(StoreBoxG1, 15000),
    "saucepan_to_hob": _task(SaucepanToHobG1, 46500, data_derived=True),
    "store_kitchenware": _task(StoreKitchenwareG1, 20000),
    "sandwich_toast": _task(ToastSandwichG1, 16500),
    "sandwich_flip": _task(FlipSandwichG1, 15500),
    "sandwich_remove": _task(RemoveSandwichG1, 31500, data_derived=True),
    "store_groceries_lower": _task(GroceriesStoreLowerG1, 32000),
    "store_groceries_upper": _task(GroceriesStoreUpperG1, 19000),
}

# Class dispatch only, derived from TASKS.
TASK_MAP: dict[str, type[BiGymEnv]] = {
    name: spec.env_cls for name, spec in TASKS.items()
}

# How the data-derived budgets were computed; each published batch's metadata
# records it as ``recommended_episode_length_rule``.
BUDGET_RULE = (
    "2x the longest successful demonstration, rounded up to the next 500 "
    "env steps, minimum 2000"
)
DATA_DERIVED_BUDGET_TASKS: frozenset[str] = frozenset(
    name for name, spec in TASKS.items() if spec.data_derived
)


def register_task(name: str, spec: TaskSpec) -> None:
    """Register ``spec`` under ``name`` so ``make(name)`` builds it.

    A name can be registered once; registering it again raises ValueError.
    """
    if not isinstance(spec, TaskSpec):
        raise TypeError(f"task {name!r} must be a TaskSpec, got {type(spec).__name__}")
    if is_object_ref(name):
        raise ValueError(
            f"task name {name!r} contains ':', which marks a 'pkg.module:ATTR' "
            "reference; register a plain name or pass the reference itself"
        )
    if name in TASKS:
        raise ValueError(f"task {name!r} is already registered")
    TASKS[name] = spec
    TASK_MAP[name] = spec.env_cls


def resolve_task_name(task_name: str) -> str:
    """Validate a task name: registered, or an importable TaskSpec reference."""
    if is_object_ref(task_name):
        task_spec(task_name)
        return task_name
    if task_name not in TASKS:
        raise KeyError(
            f"unknown task {task_name!r}; registered tasks: {', '.join(TASKS)}"
        )
    return task_name


def check_task_initialization_profile(task_name: str, profile: str) -> None:
    """Raise ValueError when ``task_name``'s env class lacks ``profile``."""
    if profile == "g1_id_v1" and task_name not in (
        "drawer_top_open",
        "drawer_top_close",
        "wall_cupboard_open",
        "wall_cupboard_close",
    ):
        raise ValueError(
            "initialization_profile='g1_id_v1' is only defined for "
            "drawer_top_open, drawer_top_close, wall_cupboard_open and "
            f"wall_cupboard_close, got task_name={task_name!r}"
        )


def task_spec(name: str) -> TaskSpec:
    """The :class:`TaskSpec` a task name refers to (KeyError if unknown)."""
    if is_object_ref(name):
        spec = import_object(name)
        if not isinstance(spec, TaskSpec):
            raise TypeError(f"task {name!r} is a {type(spec).__name__}, not a TaskSpec")
        return spec
    return TASKS[resolve_task_name(name)]


def task_config(name: str) -> EnvConfig:
    """The official configuration of a task."""
    return task_spec(name).config()


def budget_provenance(name: str) -> str:
    """``data_derived`` when the budget came from demos, else ``upstream_placeholder``."""
    return "data_derived" if task_spec(name).data_derived else "upstream_placeholder"


def all_task_names() -> tuple[str, ...]:
    """Every registered task name, sorted."""
    return tuple(sorted(TASKS))
