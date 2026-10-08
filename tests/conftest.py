import os

import pytest

if os.environ.get("MUJOCO_GL") == "egl":
    import mujoco

    # Importing pyopenxr switches PyOpenGL to GLX, after which no EGL display
    # can be opened; open it before any test module imports xr.
    mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"), 1, 1).close()


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run slow tests")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
