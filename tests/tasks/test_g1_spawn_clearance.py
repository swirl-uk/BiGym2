"""G1 spawns whose settled Dex1 hands must start clear of the counter.

After the GR00T reset settle the fingers of the ``store_groceries_*`` and
``store_box`` G1 spawns sit close to the counter front; the spawns are the
closest contact-free values. The check has to run on the benchmark
substrate: the floating-base env parks the base at the spawn with a
different arm pose (its fingers touch the counter at x=0.14, while the
controller settles the pelvis ~7 cm behind the spawn).
"""

import mujoco
import numpy as np
import pytest

from bigym.envs.groceries import GroceriesStoreLowerG1, GroceriesStoreUpperG1
from bigym.envs.pick_and_place import StoreBoxG1

FOOT_LINKS = ("ankle", "foot")

SPAWNS = [
    ("store_groceries_lower", GroceriesStoreLowerG1, [0.14, 0.0, 0.0]),
    ("store_groceries_upper", GroceriesStoreUpperG1, [0.14, 0.0, 0.0]),
    ("store_box", StoreBoxG1, [0.21, 0.0, 0.0]),
]


def _robot_scene_contacts(env) -> list[tuple[str, str, float]]:
    """(robot body, scene body, penetration m) for non-foot robot contacts."""
    inner = env.inner_env
    model = inner.model
    data = inner.data
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        for i in range(model.nbody)
    ]
    pelvis = next(i for i, n in enumerate(names) if n.endswith("pelvis"))
    root = model.body_rootid[pelvis]
    rows = []
    for i in range(data.ncon):
        contact = data.contact[i]
        body_1 = model.geom_bodyid[contact.geom1]
        body_2 = model.geom_bodyid[contact.geom2]
        robot_1 = model.body_rootid[body_1] == root
        robot_2 = model.body_rootid[body_2] == root
        if robot_1 == robot_2:
            continue
        robot_body, scene_body = (body_1, body_2) if robot_1 else (body_2, body_1)
        link = names[robot_body].rsplit("/", 1)[-1]
        if any(k in link for k in FOOT_LINKS):
            continue
        rows.append((link, names[scene_body], float(-contact.dist)))
    return rows


@pytest.mark.parametrize("task,env_cls,spawn", SPAWNS, ids=[s[0] for s in SPAWNS])
def test_g1_store_spawn_values(task, env_cls, spawn) -> None:
    np.testing.assert_allclose(env_cls.RESET_ROBOT_POS, spawn)


@pytest.mark.parametrize("task,env_cls,spawn", SPAWNS, ids=[s[0] for s in SPAWNS])
def test_g1_store_spawns_settle_clear_of_the_counter(task, env_cls, spawn) -> None:
    from bigym.loco import make

    env = make(task)
    try:
        env.reset(seed=620000)
        contacts = _robot_scene_contacts(env)
        assert not contacts, (task, sorted({(a, b) for a, b, _ in contacts}))
    finally:
        env.close()
