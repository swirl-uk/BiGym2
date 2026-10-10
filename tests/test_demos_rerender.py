"""bigym-rerender-lerobot on a downloaded LeRobot task folder.

A fake renderer stands in for the simulator, so the test checks what the
copy does: the camera columns, their statistics and the recorded camera
resolution change, and every other column and file stays as it was.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from bigym.loco.demos import rerender
from tests.fixtures.synthetic_dataset import CAMERAS, make_episode, write_export

SIZE = (6, 10)
STATS = ("mean", "count")


class FakeRenderer:
    """Renders every frame of an episode as one flat colour per camera."""

    camera_keys = CAMERAS
    size = SIZE

    def __init__(self):
        self.seeds = []

    def render_episode(self, npz):
        self.seeds.append(int(npz["seed"]))
        qpos = np.asarray(npz["full_qpos"])
        frames = np.empty((len(qpos), len(CAMERAS), 3, *SIZE), np.uint8)
        for t, row in enumerate(qpos):
            for i in range(len(CAMERAS)):
                frames[t, i] = colour(row, i)
        return frames


def colour(qpos_row: np.ndarray, camera: int) -> int:
    return int(abs(qpos_row[0]) * 40 + camera * 50) % 256


def fake_image_stats(frames):
    return {
        "mean": frames.mean(axis=(0, 2, 3)).reshape(3, 1, 1) / 255.0,
        "count": np.array([len(frames)]),
    }


def fake_aggregate(stats):
    counts = np.array([s["count"][0] for s in stats], float)
    means = np.stack([s["mean"] for s in stats])
    return {
        "mean": (means * counts[:, None, None, None]).sum(0) / counts.sum(),
        "count": np.array([int(counts.sum())]),
    }


@pytest.fixture
def task(tmp_path, monkeypatch):
    monkeypatch.setattr(rerender, "image_stats", fake_image_stats)
    monkeypatch.setattr(rerender, "aggregate_image_stats", fake_aggregate)
    rng = np.random.default_rng(0)
    episodes = []
    for index, length in enumerate((5, 3, 7)):
        episode = make_episode(rng, length, action_dim=4, state_dim=5, nq=6)
        episode["seed"] = np.int64(620000 + index)
        episodes.append(episode)
    root = write_export(tmp_path / "src", episodes, episodes_per_file=2)
    add_stats(root, len(episodes))
    metadata = json.loads((root / "metadata.json").read_text())
    metadata["env_config"] = {"camera_shape": [8, 8], "camera_keys": list(CAMERAS)}
    (root / "metadata.json").write_text(json.dumps(metadata))
    return root, episodes


def add_stats(root: Path, count: int) -> None:
    """The image statistics a LeRobot export carries, as placeholders."""
    names = [f"observation.images.{cam}" for cam in CAMERAS]
    path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    for name in names:
        table = table.append_column(
            f"stats/{name}/mean",
            pa.array(
                [[[[0.0]]] * 3] * count, pa.list_(pa.list_(pa.list_(pa.float64())))
            ),
        )
        table = table.append_column(
            f"stats/{name}/count", pa.array([[0]] * count, pa.list_(pa.int64()))
        )
    pq.write_table(table, path)
    stats = {"action": {"mean": [0.0]}}
    stats.update({name: {"mean": [[[0.0]]] * 3, "count": [0]} for name in names})
    (root / "meta" / "stats.json").write_text(json.dumps(stats))


def decode(cell) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(cell["bytes"])))


def test_only_the_cameras_change(task, tmp_path):
    src, episodes = task
    out = tmp_path / "out"
    renderer = FakeRenderer()
    rerender.rerender_task(src, out, renderer)

    assert renderer.seeds == [620000, 620001, 620002]
    files = sorted(p.relative_to(src) for p in src.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file())
    cameras = [f"observation.images.{cam}" for cam in CAMERAS]
    for path in sorted(src.glob("data/chunk-*/*.parquet")):
        before, after = pq.read_table(path), pq.read_table(out / path.relative_to(src))
        assert after.schema.equals(before.schema, check_metadata=True)
        for name in before.column_names:
            if name not in cameras:
                assert after.column(name).equals(before.column(name)), name
        qpos = np.asarray(before.column("full_qpos").to_pylist())
        for i, name in enumerate(cameras):
            for row, cell in zip(qpos, after.column(name).to_pylist(), strict=True):
                image = decode(cell)
                assert image.shape == (*SIZE, 3)
                assert (image == colour(row, i)).all()

    for name in ("meta/alignment.json", "meta/episode_init_states.json"):
        assert (out / name).read_bytes() == (src / name).read_bytes()
    info = json.loads((out / "meta" / "info.json").read_text())
    assert all(info["features"][name]["shape"] == [3, *SIZE] for name in cameras)
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["task"]["camera_shape"] == list(SIZE)
    assert metadata["env_config"]["camera_shape"] == list(SIZE)


def test_image_statistics_are_recomputed(task, tmp_path):
    src, episodes = task
    out = tmp_path / "out"
    rerender.rerender_task(src, out, FakeRenderer())

    head = "observation.images.head"
    table = pq.read_table(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    counts = [c[0] for c in table.column(f"stats/{head}/count").to_pylist()]
    assert counts == [len(e["full_qpos"]) for e in episodes]
    means = [
        np.asarray(m).ravel() for m in table.column(f"stats/{head}/mean").to_pylist()
    ]
    for mean, episode in zip(means, episodes, strict=True):
        expected = np.mean([colour(row, 0) for row in episode["full_qpos"]]) / 255.0
        np.testing.assert_allclose(mean, expected)
    totals = json.loads((out / "meta" / "stats.json").read_text())
    assert totals[head]["count"] == [sum(counts)]
    assert totals["action"] == {"mean": [0.0]}


def test_refuses_a_folder_that_is_not_empty(task, tmp_path):
    src, _ = task
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("")
    with pytest.raises(SystemExit, match="not empty"):
        rerender.rerender_task(src, out, FakeRenderer())


def test_refuses_a_folder_inside_the_task(task):
    src, _ = task
    with pytest.raises(SystemExit, match="inside"):
        rerender.rerender_task(src, src / "224", FakeRenderer())


def test_default_folder_names_the_size():
    assert rerender.default_root("move_plate", (224, 224)) == Path(
        "bigym-g1-native60-224/move_plate"
    )
    assert rerender.default_root("move_plate", (240, 320)) == Path(
        "bigym-g1-native60-240x320/move_plate"
    )


@pytest.mark.parametrize(
    "flags",
    [["--repo-id", "me/x"], ["--max-episodes", "2"], ["--push"], ["--videos"]],
)
def test_npz_only_flags_are_refused_for_a_task_folder(task, monkeypatch, flags):
    src, _ = task
    argv = ["bigym-rerender-lerobot", "--demo-dir", str(src), *flags]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="npz batches only"):
        rerender.main()


@pytest.mark.parametrize("flags", [[], ["--task", "move_plate", "--demo-dir", "x"]])
def test_needs_exactly_one_source(monkeypatch, flags):
    monkeypatch.setattr(sys, "argv", ["bigym-rerender-lerobot", *flags])
    with pytest.raises(SystemExit, match="one of --task or --demo-dir"):
        rerender.main()
