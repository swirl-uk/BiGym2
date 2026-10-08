"""The evaluator's policy process: the policy runs where the simulator is not.

Fast tests (no MuJoCo): a stand-in environment is driven by policies run both
in-process and through :class:`PolicyProcess`, and the two must agree bit for
bit; a policy that rewrites the task through the garbage collector cheats
in-process and must not through the pipe; failures surface as they would
in-process; the parent refuses anything but plain data from the child; and a
policy that imports the simulator is found by the static scan and refused by
the child at run time.
"""

from __future__ import annotations

import json
import pickle
import textwrap
from pathlib import Path

import numpy as np
import pytest

from bigym.loco.agent.episode import Tools, load_policy, run_episode
from bigym.loco.agent.policy_process import (
    ForbiddenImport,
    PolicyError,
    PolicyProcess,
    decode,
    encode,
    forbidden_imports,
    tool_names,
)


class TaskEnv:
    """A stand-in env whose reward comes from its own ``_success`` test."""

    def __init__(self, steps: int = 6):
        """Set the episode length; the task itself never succeeds."""
        self.info = {"action_dim": 4, "task": "fake", "raw_low": np.zeros(4)}
        self.steps = steps
        self.t = 0
        self.actions: list[np.ndarray] = []
        self.calls: list[tuple] = []

    def _success(self) -> bool:
        return False

    def _obs(self) -> dict:
        return {
            "t": self.t,
            "time_limit": 100,
            "fell": False,
            "low_dim_obs": np.linspace(0.0, 1.0, 5, dtype=np.float32) * self.t,
        }

    def reset(self, seed):
        """Start an episode."""
        self.t = 0
        self.seed = seed
        return self._obs()

    def step(self, raw):
        """Record the action; reward 1 on the step the task reports success."""
        self.actions.append(np.asarray(raw).copy())
        self.t += 1
        reward = 1.0 if self._success() else 0.0
        done = reward > 0 or self.t >= self.steps
        return self._obs(), reward, done, {"termination": None}

    def hold_action(self):
        """The action that holds still."""
        return np.full(4, 0.1, dtype=np.float32)

    def image(self, camera="head", width=84, height=84):
        """A deterministic image; refuses sizes above 8x8 like the real cap."""
        self.calls.append(("image", camera, width, height))
        if width > 8 or height > 8:
            raise ValueError(f"requested {width}x{height}; the cap is 8x8")
        return np.arange(width * height * 3, dtype=np.uint8).reshape(height, width, 3)

    def ik(self, **kw):
        """Echo the target, so the test sees the arguments arrive intact."""
        self.calls.append(
            ("ik", {k: (type(v), np.asarray(v).tobytes()) for k, v in kw.items()})
        )
        return np.asarray(kw["left_pos"], dtype=np.float64) * 2.0


def write(tmp_path: Path, name: str, source: str) -> Path:
    """Write a policy file from indented source and return its path."""
    path = tmp_path / name
    path.write_text(textwrap.dedent(source))
    return path


HONEST = """
    import numpy as np


    class Policy:
        def reset(self, obs, tools):
            self.hold = np.asarray(tools.hold_action(), dtype=np.float32)
            self.task = tools.info["task"]

        def act(self, obs, tools):
            img = tools.image("head", 4, 4)
            arm = tools.ik(left_pos=np.array([0.1, 0.2, 0.3]) * obs["t"])
            raw = self.hold + np.float32(img.mean() / 1000.0)
            raw[:3] += arm.astype(np.float32)
            return list(raw.astype(np.float64) / 3.0)
"""

TAMPER = """
    import gc

    import numpy as np


    class Policy:
        def reset(self, obs, tools):
            self.hold = np.asarray(tools.hold_action(), dtype=np.float32)
            for o in gc.get_objects():
                try:
                    if not isinstance(o, type) and callable(
                        getattr(type(o), "_success", None)
                    ):
                        o._success = lambda *a, **k: True
                except Exception:
                    pass

        def act(self, obs, tools):
            return self.hold
"""


def test_an_isolated_policy_matches_the_in_process_run_bit_for_bit(tmp_path):
    """Actions, records and tool calls are identical to running the policy here."""
    path = write(tmp_path, "policy.py", HONEST)
    here, there = TaskEnv(), TaskEnv()
    rec_here = run_episode(here, load_policy(path), seed=3)
    with PolicyProcess() as proc:
        rec_there = run_episode(there, proc.load(path), seed=3)
    for rec in (rec_here, rec_there):
        rec.pop("wall_s")
    assert rec_there == rec_here
    assert here.calls == there.calls
    assert len(here.actions) == len(there.actions) == 6
    for a, b in zip(here.actions, there.actions, strict=True):
        assert a.dtype == b.dtype == np.float32
        assert a.tobytes() == b.tobytes()


def test_a_policy_cannot_patch_the_task_through_the_garbage_collector(tmp_path):
    """The gc patch scores 1.0 in-process; through the pipe there is no env to find."""
    path = write(tmp_path, "tamper.py", TAMPER)
    assert run_episode(TaskEnv(), load_policy(path), seed=0)["success"] == 1
    env = TaskEnv()
    with PolicyProcess() as proc:
        rec = run_episode(env, proc.load(path), seed=0)
    assert rec["success"] == 0
    assert rec["reward"] == 0.0
    assert rec["length"] == env.steps
    assert env._success() is False


def test_the_child_tools_have_the_same_surface_as_episode_tools():
    """Every public Tools method is forwarded, and nothing else is."""
    expected = {
        "hold_action",
        "ik",
        "render",
        "image",
        "camera_info",
        "pixel_to_ray",
    }
    assert tool_names() == expected
    assert expected <= set(vars(Tools))


def test_a_tool_error_reaches_the_policy_which_may_handle_it(tmp_path):
    """A refused request raises inside the policy, as it does in-process."""
    path = write(
        tmp_path,
        "policy.py",
        """
        import numpy as np


        class Policy:
            def reset(self, obs, tools):
                try:
                    tools.image("head", 84, 84)
                except ValueError as exc:
                    self.err = str(exc)

            def act(self, obs, tools):
                assert "cap is 8x8" in self.err
                return np.zeros(4)
        """,
    )
    with PolicyProcess() as proc:
        rec = run_episode(TaskEnv(steps=2), proc.load(path), seed=0)
    assert rec["length"] == 2


def test_an_unknown_tool_is_refused(tmp_path):
    """The parent runs only the public Tools methods, never other env attributes."""
    path = write(
        tmp_path,
        "policy.py",
        """
        import numpy as np


        class Policy:
            def reset(self, obs, tools):
                tools._env.reset(0)

            def act(self, obs, tools):
                return np.zeros(4)
        """,
    )
    env = TaskEnv()
    with PolicyProcess() as proc:
        with pytest.raises(PolicyError, match="tools has no method 'reset'"):
            run_episode(env, proc.load(path), seed=5)
    assert env.seed == 5  # only the parent's own reset ran


def test_a_policy_that_raises_fails_the_run_and_the_process_survives(tmp_path):
    """The error carries the policy's traceback; the next load still works."""
    bad = write(
        tmp_path,
        "bad.py",
        """
        class Policy:
            def reset(self, obs, tools):
                pass

            def act(self, obs, tools):
                raise KeyError("left_hand_pos")
        """,
    )
    good = write(tmp_path, "good.py", HONEST)
    with PolicyProcess() as proc:
        with pytest.raises(PolicyError, match="KeyError: 'left_hand_pos'") as info:
            run_episode(TaskEnv(), proc.load(bad), seed=0)
        assert "bad.py" in str(info.value)
        assert run_episode(TaskEnv(), proc.load(good), seed=0)["length"] == 6


def test_a_file_without_a_policy_class_is_refused(tmp_path):
    path = write(tmp_path, "nothing.py", "X = 1\n")
    with PolicyProcess() as proc:
        with pytest.raises(PolicyError, match="must define a class Policy"):
            proc.load(path)


def test_a_policy_process_that_dies_fails_the_run(tmp_path):
    path = write(
        tmp_path,
        "policy.py",
        """
        import os


        class Policy:
            def reset(self, obs, tools):
                os._exit(3)
        """,
    )
    with PolicyProcess() as proc:
        with pytest.raises(PolicyError, match="exit status 3"):
            run_episode(TaskEnv(), proc.load(path), seed=0)


def test_the_wire_round_trips_plain_data_exactly():
    value = {
        "f32": np.array([np.nan, 1e-45, -0.0], dtype=np.float32),
        "f64": np.arange(6.0).reshape(2, 3).T,
        "scalar": np.float64(0.1),
        "flag": np.bool_(True),
        "nested": [1, 2.5, None, "x", (np.int64(7), [True])],
        "path": Path("/tmp/x.png"),
    }
    got = decode(encode(value))
    assert got["f32"].dtype == np.float32
    assert got["f32"].tobytes() == value["f32"].tobytes()
    assert np.array_equal(got["f64"], value["f64"])
    assert type(got["scalar"]) is np.float64 and got["scalar"] == 0.1
    assert type(got["flag"]) is np.bool_
    assert got["nested"] == [1, 2.5, None, "x", (7, [True])]
    assert type(got["nested"][4][0]) is np.int64
    assert got["path"] == "/tmp/x.png"


def test_the_wire_refuses_anything_but_plain_data():
    """No pickles, no object arrays, no arbitrary objects: in either direction."""
    with pytest.raises(TypeError):
        encode(object())
    with pytest.raises(TypeError):
        encode(np.array([object()]))
    with pytest.raises(ValueError):
        decode(pickle.dumps({"op": "action"}))
    forged = {
        "__bigym_wire__": "ndarray",
        "dtype": "|O",
        "shape": [1],
        "data": "AAAAAAAAAAA=",
    }
    with pytest.raises(ValueError, match="refused"):
        decode(json.dumps(forged).encode())
    short = dict(forged, dtype="<f4", shape=[3])
    with pytest.raises(ValueError, match="does not match"):
        decode(json.dumps(short).encode())


# --------------------------------------------------------------------------
# Forbidden imports
# --------------------------------------------------------------------------


def scan(tmp_path: Path, source: str) -> list[str]:
    """Scan a directory holding only ``policy.py`` with this source."""
    return forbidden_imports(write(tmp_path, "policy.py", source))


@pytest.mark.parametrize(
    "source, module",
    [
        ("import mujoco", "mujoco"),
        ("import mujoco.viewer as v", "mujoco.viewer"),
        ("import numpy, bigym", "bigym"),
        ("from bigym.loco.agent import EnvTools", "bigym.loco.agent"),
        ("from dm_control import mjcf", "dm_control"),
        ("from mink import Configuration", "mink"),
        ("m = __import__('mujoco')", "mujoco"),
        ("import builtins\nm = builtins.__import__('mink')", "mink"),
        ("import importlib\nm = importlib.import_module('bigym.envs')", "bigym.envs"),
        (
            "from importlib import import_module\nimport_module('dm_control')",
            "dm_control",
        ),
        ("def f():\n    import mujoco\n", "mujoco"),
    ],
)
def test_the_scan_finds_every_spelling_of_a_forbidden_import(tmp_path, source, module):
    hits = scan(tmp_path, source)
    assert len(hits) == 1
    assert hits[0].startswith("policy.py:") and hits[0].endswith(f" imports {module}")


@pytest.mark.parametrize(
    "source",
    [
        "import numpy as bigym_np",
        "import numpy as mujoco",
        "from . import helper",
        "from .bigym import x",
        "import bigymnastics, mujoco_menagerie_notes",
        "import scipy.optimize as mink",
        'S = \'import mujoco\'\nT = """from bigym import x"""',
        "name = 'mujoco'\nm = __import__(name)",
        "import importlib\nimportlib.import_module('.mujoco', 'helpers')",
    ],
)
def test_the_scan_ignores_what_is_not_a_forbidden_import(tmp_path, source):
    assert scan(tmp_path, source) == []


def test_the_scan_covers_every_helper_next_to_the_policy(tmp_path):
    """Any sibling is importable, so each one is scanned; a broken one is skipped."""
    policy = write(tmp_path, "policy.py", "import numpy as np\nimport planner\n")
    write(tmp_path, "planner.py", "import os\n\nimport mujoco\n")
    write(tmp_path, "scratch.py", "def broken(:\n    import bigym\n")
    (tmp_path / "notes.txt").write_text("import bigym\n")
    assert forbidden_imports(policy) == ["planner.py:3 imports mujoco"]


def test_the_policy_process_refuses_a_dynamic_simulator_import(tmp_path):
    """An import the scan cannot see is refused in the child and rejects the policy."""
    path = write(
        tmp_path,
        "policy.py",
        """
        import importlib

        import numpy as np


        class Policy:
            def reset(self, obs, tools):
                pass

            def act(self, obs, tools):
                importlib.import_module("mu" + "joco")
                return np.zeros(4)
        """,
    )
    assert forbidden_imports(path) == []
    env = TaskEnv()
    with PolicyProcess() as proc:
        with pytest.raises(ForbiddenImport, match=r"^policy\.py:12 imports mujoco$"):
            run_episode(env, proc.load(path), seed=0)
    assert env.actions == []
    assert issubclass(ForbiddenImport, PolicyError)


def test_a_caught_simulator_import_still_rejects_the_policy(tmp_path):
    """Catching the ImportError does not hide the attempt."""
    path = write(
        tmp_path,
        "policy.py",
        """
        import numpy as np

        try:
            sim = __import__("".join(["big", "ym"]))
        except ImportError as exc:
            sim = str(exc)


        class Policy:
            def reset(self, obs, tools):
                pass

            def act(self, obs, tools):
                return np.zeros(4)
        """,
    )
    with PolicyProcess() as proc:
        with pytest.raises(ForbiddenImport, match=r"^policy\.py:5 imports bigym$"):
            proc.load(path)


def test_the_child_refuses_only_the_simulator(tmp_path):
    """The finder is in front of every import, yet other modules import as usual."""
    path = write(
        tmp_path,
        "policy.py",
        """
        import sys

        import numpy as np


        class Policy:
            def reset(self, obs, tools):
                import fractions
                import numpy.linalg

                self.ok = fractions.Fraction(1, 2) + float(numpy.linalg.norm([0.0]))

            def act(self, obs, tools):
                import colorsys

                finder = type(sys.meta_path[0]).__name__ == "_ForbiddenFinder"
                loaded = any(m.split(".")[0] in ("bigym", "mujoco") for m in sys.modules)
                return np.array([finder, loaded, self.ok, colorsys.ONE_THIRD])
        """,
    )
    env = TaskEnv(steps=2)
    with PolicyProcess() as proc:
        rec = run_episode(env, proc.load(path), seed=0)
    assert rec["length"] == 2
    assert env.actions[0].tolist() == pytest.approx([1.0, 0.0, 0.5, 1 / 3])
