"""``bigym-view`` on the dataset layout: discovery, sources and the episode store.

A tiny LeRobot v3 task export (state columns only, alignment v2) stands in
for the published dataset or a ``bigym-download --local-dir`` copy. No
environment is built: the store is driven with a stub env.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from bigym.loco.agent.demo_video import TaskDemos
from bigym.vr.viewer import view_demos
from tests.fixtures.synthetic_dataset import write_export
from tests.fixtures.viser_gui import FakeServer

NQ = 4
LENGTHS = (5, 3)


def make_episode(index: int, length: int) -> dict:
    """Collector-convention rows: reward[t] is the transition that produced t."""
    qpos = np.arange(length * NQ, dtype=np.float64).reshape(length, NQ) + 100 * index
    reward = np.zeros((length, 1), np.float32)
    reward[-1] = 1.0
    return {"full_qpos": qpos, "reward": reward, "seed": 620000 + index}


@pytest.fixture
def local_copy(tmp_path):
    """A ``--local-dir`` download of two tasks, with the Hub's own cache dir."""
    root = tmp_path / "bigym-data"
    episodes = [make_episode(i, n) for i, n in enumerate(LENGTHS)]
    for task in ("move_plate", "pick_box"):
        write_export(root / task, episodes, task)
    # huggingface_hub keeps its bookkeeping in <local_dir>/.cache; never a task.
    write_export(root / ".cache" / "huggingface" / "stray", episodes, "stray")
    return root, episodes


def test_discovers_task_folders_and_skips_hidden(local_copy):
    root, _ = local_copy
    found = view_demos.discover_dataset_tasks(root)
    assert [f.name for f in found] == ["move_plate", "pick_box"]
    assert view_demos.discover_dataset_tasks(root / "pick_box") == [
        (root / "pick_box").resolve()
    ]


def test_local_copy_is_a_dataset_source(local_copy):
    root, _ = local_copy
    source = view_demos.find_source(str(root))
    assert isinstance(source, view_demos.DatasetSource)
    assert source.labels == ["move_plate", "pick_box"]
    assert source.tasks == ["move_plate", "pick_box"]
    assert all(d is not None and d.is_dir() for d in source.dirs)
    assert source.where == str(root.resolve())
    assert view_demos.pick_task(source, "pick_box") == 1
    with pytest.raises(SystemExit, match="not in"):
        view_demos.pick_task(source, "stack_blocks")


def test_batches_win_over_dataset_folders(local_copy):
    root, _ = local_copy
    batch = root / "mine" / "20260101_000000"
    batch.mkdir(parents=True)
    (batch / "metadata.json").write_text("{}")
    np.savez(batch / "episode_0.npz", full_qpos=np.zeros((2, NQ)))
    source = view_demos.find_source(str(root))
    assert isinstance(source, view_demos.BatchSource)
    assert source.batches == [batch.resolve()]


def test_empty_or_missing_directory_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No demos found"):
        view_demos.find_source(str(tmp_path))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        view_demos.find_source(str(tmp_path / "nope"))


def test_empty_field_is_the_published_dataset(monkeypatch):
    from bigym.loco.demos import hub

    monkeypatch.setattr(hub, "available_tasks", lambda *a: ("move_plate", "reach"))
    source = view_demos.find_source("  ")
    assert isinstance(source, view_demos.DatasetSource)
    assert source.labels == ["move_plate", "reach"]
    assert source.dirs == [None, None]
    assert source.roots == []


def test_columns_undo_the_transition_shift(local_copy):
    root, episodes = local_copy
    demos = TaskDemos("move_plate", root / "move_plate")
    assert [e.seed for e in demos.episodes] == [620000, 620001]
    for episode, want in zip(demos.episodes, episodes, strict=True):
        got = demos.columns(episode, ("full_qpos", "reward", "lowerbody_command"))
        assert set(got) == {"full_qpos", "reward"}
        np.testing.assert_array_equal(got["full_qpos"], want["full_qpos"])
        np.testing.assert_array_equal(got["reward"], want["reward"])
        np.testing.assert_array_equal(demos.qpos(episode), want["full_qpos"])


class _Env:
    def __init__(self):
        self.seeds = []

    def reset(self, seed=None):
        self.seeds.append(seed)


class _Inner:
    def __init__(self):
        self.hooks = 0

    class _Cache:
        def clean(self):
            pass

    _step_cache = _Cache()

    def _on_step(self):
        self.hooks += 1


def test_dataset_store_loads_episodes_on_their_seed(local_copy):
    root, episodes = local_copy
    env, inner = _Env(), _Inner()
    store = view_demos.DatasetStore(
        env, inner, TaskDemos("pick_box", root / "pick_box")
    )
    assert store.episode_labels() == ["00 · seed 620000 · 5", "01 · seed 620001 · 3"]
    assert [i["seed"] for i in store.episode_infos()] == [620000, 620001]
    assert store.load(1) and not store.load(1)
    assert env.seeds == [620001]
    np.testing.assert_array_equal(store.qpos, episodes[1]["full_qpos"])
    np.testing.assert_array_equal(store.rewards, episodes[1]["reward"].reshape(-1))
    assert store.qvel is None and store.cmd is None
    assert store.files[1].name == "001_episode_1"
    store.after_forward(0)
    assert inner.hooks == 1
    assert not store.load(99) and store.ep == 1  # clamped to the last episode


def test_folder_marks_are_one_level_deep(local_copy, tmp_path):
    root, _ = local_copy
    assert view_demos.demo_marker(root / "move_plate")
    assert view_demos.has_demos(root)  # task folders directly inside
    assert view_demos.has_demos(root / "move_plate")
    assert not view_demos.has_demos(tmp_path)  # two levels above: not marked
    assert not view_demos.has_demos(tmp_path / "nope")
    names, total = view_demos.subfolders(root)
    assert names == ["move_plate", "pick_box"] and total == 2  # .cache hidden
    assert view_demos.subfolders(root, 1) == (["move_plate"], 2)


def test_browse_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a b").mkdir()
    assert view_demos.browse_dir("") == tmp_path.resolve()
    assert view_demos.browse_dir("'a b'") == (tmp_path / "a b").resolve()
    assert view_demos.browse_dir("x y") is None
    assert view_demos.browse_dir("missing") is None
    assert view_demos.browse_dir("'unbalanced") is None


def test_directory_panel_browses_and_loads(local_copy, monkeypatch):
    root, _ = local_copy
    monkeypatch.chdir(root.parent)
    monkeypatch.setattr(view_demos, "_RECENT", [])
    server = FakeServer()
    opened = []
    done = threading.Event()
    view_demos.directory_panel(server, "", lambda src: (opened.append(src), done.set()))
    h = server.gui.handles
    (note,) = server.gui.htmls
    assert h["folders"].options[1:] == ["● bigym-data"]  # the working directory
    h["folders"].fire("● bigym-data")
    assert h["path"].value == str(root.resolve())
    assert h["folders"].options[1:] == ["● move_plate", "● pick_box"]
    assert h["folders"].value == h["folders"].options[0]
    assert "2 subfolders" in note.content
    h["up"].fire()
    assert h["path"].value == str(root.parent.resolve())
    h["path"].fire(str(root / "nope"))
    assert "not a folder" in note.content and h["folders"].options[1:] == []
    h["path"].fire(str(root / "move_plate"))
    assert "this folder holds demos" in note.content
    h["path"].value = str(root)
    h["load"].fire()
    assert done.wait(30)
    assert isinstance(opened[0], view_demos.DatasetSource)
    assert view_demos._RECENT == [str(root)]
    assert f"loaded: {root}" in view_demos.quick_places()
    h["quick"].fire(f"working directory: {root.parent}")
    assert h["path"].value == str(root.parent) and h["quick"].value == "jump to …"
