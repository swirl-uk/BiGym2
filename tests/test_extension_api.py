"""Controllers and tasks from outside bigym: registration and "module:ATTR".

The backends and the task live in tests/fixtures/external_extension.py,
written against public names only. The envs are built without cameras, so
nothing renders.
"""

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import bigym.loco
from bigym.loco import adapters, make, register_backend, register_task, tasks
from bigym.loco.objref import import_object, path_name
from bigym.vr.collect.config import (
    COLLECT_MIN_OUTER_STEPS,
    CollectConfig,
    collection_env_config,
    default_out_dir,
)
from tests.fixtures import external_extension

BACKEND = "test_external_wbc"
TASK = "test_external_move_plate"
BACKEND_REF = "tests.fixtures.external_extension:BINDING"
TASK_REF = "tests.fixtures.external_extension:TASK"
HOLD_POSE_REF = "tests.fixtures.external_extension:HOLD_POSE"
FAST = {"camera_keys": (), "episode_length": 2000}


@pytest.fixture
def registered():
    register_backend(BACKEND, external_extension.BINDING)
    register_task(TASK, external_extension.TASK)
    yield
    adapters.BACKENDS.pop(BACKEND, None)
    tasks.TASKS.pop(TASK, None)
    tasks.TASK_MAP.pop(TASK, None)
    import_object.cache_clear()


def _run(env, steps=3):
    try:
        env.reset(seed=620000)
        for _ in range(steps):
            ts = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            assert np.all(np.isfinite(ts.low_dim_obs))
    finally:
        env.close()


def test_the_extension_api_is_public():
    for name in ("BackendBinding", "register_backend", "register_task"):
        assert name in bigym.loco.__all__
    assert bigym.loco.BackendBinding is adapters.BackendBinding
    assert "groot_wbc_g1" in adapters.BACKENDS


def test_the_fixture_uses_public_names_only():
    tree = ast.parse(Path(external_extension.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not any(p.startswith("_") for p in (node.module or "").split("."))
            assert not any(a.name.startswith("_") for a in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(p.startswith("_") for p in alias.name.split("."))


def _check_backend_env(backend):
    env = make(
        "move_plate",
        controller={"backend": backend, "reset_warmup_steps": 5},
        **FAST,
    )
    assert env.config_overrides["controller.backend"] == backend
    assert env.substrate_fingerprint()["lowerbody_backend"] == backend
    _run(env)


def test_registered_backend_drives_the_env(registered):
    _check_backend_env(BACKEND)


def test_backend_named_by_module_attr(registered):
    _check_backend_env(BACKEND_REF)


def test_a_lower_body_base_subclass_drives_the_env():
    env = make(
        "move_plate",
        controller={"backend": HOLD_POSE_REF, "reset_warmup_steps": 5},
        **FAST,
    )
    try:
        controller = env.lowerbody.controller
        assert isinstance(controller, external_extension.HoldPoseController)
        env.reset(seed=620000)
        state = controller.get_state()
        assert set(state) == {"cmd", "height_cmd"}
        for _ in range(3):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        controller.set_state(state)
        assert controller.get_state().keys() == state.keys()
        np.testing.assert_array_equal(
            controller.step(), external_extension.STANDING_POSE
        )
        assert not controller.is_failed()
    finally:
        env.close()


def test_registered_task(registered):
    assert tasks.task_config(TASK).episode_length == 17000
    env = make(TASK, controller=None, **FAST)
    assert env.official_config == external_extension.TASK.config()
    assert env.substrate_fingerprint()["task"] == TASK
    _run(env)


def test_task_named_by_module_attr():
    env = make(TASK_REF, **FAST)
    assert env.action_space.shape == (20,)  # the TaskSpec turns pitch off
    assert env.substrate_fingerprint()["task"] == TASK_REF
    _run(env)


def test_a_name_registers_once(registered):
    with pytest.raises(ValueError, match="already registered"):
        register_backend(BACKEND, external_extension.BINDING)
    with pytest.raises(ValueError, match="already registered"):
        register_backend("groot_wbc_g1", external_extension.BINDING)
    with pytest.raises(ValueError, match="already registered"):
        register_task(TASK, external_extension.TASK)
    with pytest.raises(ValueError, match="already registered"):
        register_task("move_plate", external_extension.TASK)


def test_invalid_registrations_and_references():
    with pytest.raises(ValueError, match="pkg.module:ATTR"):
        register_backend("a:b", external_extension.BINDING)
    not_an_extension: Any = object()
    with pytest.raises(TypeError, match="BackendBinding"):
        register_backend("not_a_binding", not_an_extension)
    with pytest.raises(TypeError, match="TaskSpec"):
        register_task("not_a_spec", not_an_extension)
    with pytest.raises(ImportError, match="no_such_bigym_pkg"):
        make("move_plate", controller={"backend": "no_such_bigym_pkg.wbc:BINDING"})
    with pytest.raises(AttributeError, match="MISSING"):
        make(
            "move_plate",
            controller={"backend": "tests.fixtures.external_extension:MISSING"},
        )
    with pytest.raises(TypeError, match="not a BackendBinding"):
        make("move_plate", controller={"backend": TASK_REF})
    with pytest.raises(TypeError, match="not a TaskSpec"):
        make(BACKEND_REF)
    assert "not_a_binding" not in adapters.BACKENDS
    assert "not_a_spec" not in tasks.TASKS


def test_a_module_attr_task_gets_a_path_safe_name():
    assert path_name(TASK_REF) == "tests.fixtures.external_extension-TASK"
    assert path_name("move_plate") == "move_plate"
    out = default_out_dir(TASK_REF)
    assert out.parent.name == "tests.fixtures.external_extension-TASK"
    collection, training = collection_env_config(CollectConfig(task=TASK_REF))
    assert training == tasks.task_config(TASK_REF)
    assert collection.episode_length == COLLECT_MIN_OUTER_STEPS * 10
