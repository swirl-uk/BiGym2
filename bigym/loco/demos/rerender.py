"""Re-render the cameras of demonstrations at another resolution.

A task of the published LeRobot dataset (``--task``, or a downloaded task
folder as ``--demo-dir``) is copied with only its three camera columns
re-rendered: state, action, every other column and file stay as they are,
and the image statistics and the recorded camera resolution are updated::

    uv sync --extra lerobot
    uv run bigym-rerender-lerobot --task move_plate --size 224 224

writes ``bigym-g1-native60-224/move_plate``. ``--root`` writes elsewhere.

A replay-format npz batch is exported to LeRobot v3 instead, with the
same output as ``bigym.loco.demos.lerobot_export`` in every respect — state, action,
per-step extras, action alignment v2, ``meta/episode_init_states.json``,
``meta/alignment.json``, the root ``metadata.json`` copy — EXCEPT that the
three camera features are re-rendered at an arbitrary resolution instead of
being copied from the recorded 84x84 ``rgb_obs``.  VLA stacks (pi0.5 / openpi)
want 224x224; the recorded demos are 84x84.

How it stays a re-render and not a re-simulation
------------------------------------------------
Every frame of a full-state replay-format demo stores ``data.qpos`` (key
``full_qpos``: robot + props; its width depends on the recorded model).
The frame loop writes that vector and calls ``mj_forward`` without physics
integration or lower-body policy inference, as in ``bigym/vr/viewer/view_demos.py``.
Environment construction/reset still runs the configured controller warmup.
Two pieces of scene state are NOT in qpos:

* reach-target spheres are static bodies moved through ``model.body_pos`` by
  ``_ReachTargetEnv._on_reset`` from the GLOBAL numpy RNG, which
  ``BiGymEnv.reset(seed=...)`` seeds — so each episode's stored ``seed`` is
  replayed before its frames;
* the target highlight colour lives in ``model.geom_rgba`` and is written by
  the task's ``_on_step`` hook, which the real ``BiGymEnv.step`` runs BEFORE
  ``get_observation``; the loop below does the same (with the step cache
  cleaned first, so ``_success()`` is not answered from the previous frame).

Images come from the env's own ``BiGymEnv._get_visual_obs`` with
``camera_shape`` set to the requested size, so shadow map, visual flags,
camera intrinsics and the ``(3, H, W)`` uint8 channel-first layout are exactly
the recording path's — only bigger.  Camera identity/order also comes from the
batch (``metadata["task"]["camera_keys"]``, canonically
``head, right_wrist, left_wrist``); the wrist cameras ride on the robot, so
posing qpos moves them.

Nothing about the LeRobot writing is re-implemented here: ``lerobot_export``
is imported and its ``main()`` is called with the source batch, with the
module's ``np.load`` redirected so that ``rgb_obs`` (and only ``rgb_obs``)
comes back re-rendered.  Fidelity gate: re-render at 84x84 and compare
against the stored pixels.

Usage (needs bigym AND lerobot in one process; the project venv is 3.12 and
already has bigym, so layer lerobot in ephemerally)::

    CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl \
    uv run --no-sync --with "lerobot[dataset]>=0.6" bigym-rerender-lerobot \
        --demo-dir <batch-dir> \
        --repo-id  YOUR_HF_USER/bigym-g1-move-plate-224 \
        --root     <output-dir> \
        --size 224 224 \
        --task-prompt "Move the plate between two draining racks."

``--task-prompt`` is the language instruction written into the LeRobot tasks
table (``meta/tasks.parquet`` in v3; one row, referenced per frame by
``task_index``).  Omit it and the instruction is resolved from
``bigym/loco/demos/task_instructions.yaml`` exactly as ``lerobot_export`` does.

Resolution ceiling: MuJoCo offscreen buffers are capped by the compiled
model's ``<visual><global offwidth= offheight=>``.  224 fits the default
640x480; larger sizes abort with the measured limits.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from bigym.loco import make
from bigym.loco.config import EnvConfig
from bigym.loco.demos import lerobot_export
from bigym.loco.demos.kinematic import pose
from bigym.loco.objref import path_name


class BatchRenderer:
    """Kinematic re-renderer for one replay-format demo batch.

    Builds the batch's own env (task, robot, lower-body backend, substrate) at
    the requested camera resolution and replays each episode's ``full_qpos``.

    One episode is rendered at a time and held whole in RAM (600 frames x 3
    cameras at 224x224 is ~270 MB); the writer consumes it frame by frame.
    """

    def __init__(self, metadata: dict, size: tuple[int, int], *, verbose: bool = True):
        """Build the recorded environment at the requested camera resolution."""
        self._verbose = verbose
        config = EnvConfig.from_metadata(metadata)
        self.camera_keys = tuple(config.camera_keys)
        self.size = (int(size[0]), int(size[1]))
        # The recorded env at the requested camera resolution. The success
        # criterion stays as recorded: reach_tolerance decides when the
        # target sphere lights up, and that highlight is part of the pixels.
        self.env = make(metadata["task"]["task_name"], config, camera_shape=self.size)
        self.inner = self.env.inner_env
        self.model = self.inner.model
        self.data = self.inner.data
        limit = (
            int(self.model.vis.global_.offheight),
            int(self.model.vis.global_.offwidth),
        )
        if self.size[0] > limit[0] or self.size[1] > limit[1]:
            raise SystemExit(
                f"--size {self.size[0]}x{self.size[1]} exceeds this scene's "
                f"MuJoCo offscreen buffer {limit[1]}x{limit[0]} (width x height). "
                "Raise <visual><global offwidth/offheight> in the scene XML "
                "(a bigym asset) before rendering this large."
            )

    def close(self):
        """Close the environment and release its rendering resources."""
        if self.env is not None:
            self.env.close()
            self.env = None

    def render_episode(self, npz) -> np.ndarray:
        """Return ``(T, n_cam, 3, H, W)`` uint8 for one episode's npz."""
        files = set(npz.files)
        if "seed" in files:
            # Replays the reset RNG draws that place targets/props and the
            # task's own reset bookkeeping. Cheap relative to rendering.
            assert self.env is not None
            self.env.reset(seed=int(np.asarray(npz["seed"]).reshape(-1)[0]))
        qpos = np.asarray(npz["full_qpos"], dtype=np.float64)
        qvel = (
            np.asarray(npz["full_qvel"], dtype=np.float64)
            if "full_qvel" in files
            else None
        )
        steps = qpos.shape[0]
        out = np.empty((steps, len(self.camera_keys), 3, *self.size), dtype=np.uint8)
        started = time.time()
        for t in range(steps):
            # Same order as the real step(): pose, then the task's _on_step
            # (reach-target highlight), then the observation.
            pose(self.inner, qpos[t], qvel[t] if qvel is not None else None)
            obs = self.inner._get_visual_obs()
            for i, camera in enumerate(self.camera_keys):
                out[t, i] = obs[f"rgb_{camera}"]
        if self._verbose:
            elapsed = time.time() - started
            print(
                f"  rendered {steps} frames x {len(self.camera_keys)} cams @ "
                f"{self.size[0]}x{self.size[1]} in {elapsed:.1f}s "
                f"({1000 * elapsed / max(1, steps):.1f} ms/frame)",
                flush=True,
            )
        return out


class _RerenderedNpz:
    """``NpzFile`` stand-in that serves a freshly rendered ``rgb_obs``.

    Everything except ``rgb_obs`` passes straight through to the real npz, so
    lerobot_export reads the recorded state/action/extras/init snapshots
    verbatim.
    """

    def __init__(self, npz, path, renderer, cache):
        self._npz = npz
        self._path = str(path)
        self._renderer = renderer
        self._cache = cache

    @property
    def files(self):
        return self._npz.files

    def __contains__(self, key):
        return key in self._npz

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._npz.close()
        return False

    def close(self):
        self._npz.close()

    def __getitem__(self, key):
        if key != "rgb_obs":
            return self._npz[key]
        if self._cache.get("path") != self._path:
            print(f"re-rendering {Path(self._path).name}", flush=True)
            self._cache.clear()  # one episode at a time: 224s are ~270 MB each
            self._cache["rgb"] = self._renderer.render_episode(self._npz)
            self._cache["path"] = self._path
        return self._cache["rgb"]


class _NumpyProxy:
    """numpy module proxy with ``load`` swapped, scoped to one module."""

    def __init__(self, real, loader):
        self._real = real
        self.load = loader

    def __getattr__(self, name):
        return getattr(self._real, name)


class EpisodeArrays:
    """An ``NpzFile`` stand-in over in-memory arrays.

    :meth:`BatchRenderer.render_episode` reads an episode through the
    ``files`` / ``__getitem__`` npz interface; a dataset episode is handed to
    it through this wrapper instead of being written to disk first.
    """

    def __init__(self, arrays: dict):
        """Wrap a mapping of array name to array.

        Args:
            arrays: The arrays to serve.
        """
        self._arrays = dict(arrays)

    @property
    def files(self) -> list:
        """The array names, as ``NpzFile.files`` reports them."""
        return list(self._arrays)

    def __contains__(self, key) -> bool:
        """Whether an array is present."""
        return key in self._arrays

    def __getitem__(self, key):
        """Return one array."""
        return self._arrays[key]


def is_lerobot_task(path: Path) -> bool:
    """Whether ``path`` is one task of a LeRobot export rather than an npz batch."""
    return (Path(path) / "meta" / "info.json").is_file()


def _column(table, name: str) -> np.ndarray:
    """One list column of equal-length rows as a ``(rows, width)`` array."""
    column = table.column(name).combine_chunks()
    return column.flatten().to_numpy(zero_copy_only=False).reshape(len(column), -1)


def _png_column(frames: np.ndarray, column_type):
    """``(T, 3, H, W)`` frames as an image column of ``column_type``."""
    import io

    import pyarrow as pa
    from PIL import Image

    images = []
    for chw in frames:
        buffer = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(chw.transpose(1, 2, 0))).save(
            buffer, format="PNG"
        )
        images.append({"bytes": buffer.getvalue(), "path": None})
    return pa.array(images, type=column_type)


def image_stats(frames: np.ndarray) -> dict[str, np.ndarray]:
    """LeRobot's per-episode statistics of one camera, ``(T, 3, H, W)`` uint8."""
    from lerobot.datasets import compute_stats  # ty: ignore[unresolved-import]

    picked = [
        compute_stats.auto_downsample_height_width(frames[i])
        for i in compute_stats.sample_indices(len(frames))
    ]
    stats = compute_stats.get_feature_stats(
        np.stack(picked),
        axis=(0, 2, 3),
        keepdims=True,
        quantile_list=compute_stats.DEFAULT_QUANTILES,
    )
    return {
        key: value if key == "count" else np.squeeze(value / 255.0, axis=0)
        for key, value in stats.items()
    }


def aggregate_image_stats(stats: list[dict]) -> dict[str, np.ndarray]:
    """LeRobot's dataset statistics of one camera from its per-episode ones."""
    from lerobot.datasets import compute_stats  # ty: ignore[unresolved-import]

    return compute_stats.aggregate_feature_stats(stats)


def _set_camera_shape(metadata: dict, size: tuple[int, int]) -> dict:
    """``metadata`` with its recorded camera resolution set to ``size``."""
    for block in ("task", "env_config"):
        if "camera_shape" in metadata.get(block, {}):
            metadata[block]["camera_shape"] = list(size)
    return metadata


def rerender_task(src: Path, out: Path, renderer) -> None:
    """Write a copy of one LeRobot task folder with its cameras re-rendered.

    Only the camera columns, their image statistics and the recorded camera
    resolution change; every other column and file is copied as it is.

    Args:
        src: The task folder, as ``bigym-download`` or ``env.get_demos()``
            fetched it.
        out: The folder to write; it must not exist or be empty.
        renderer: Renders one episode, ``(T, n_cam, 3, H, W)``, from its
            ``full_qpos``, ``full_qvel`` and ``seed`` (a :class:`BatchRenderer`).
    """
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq

    src, out = Path(src), Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; pass another --root")
    if out.resolve().is_relative_to(src.resolve()):
        raise SystemExit(f"{out} is inside {src}; pass a --root outside it")
    size = (int(renderer.size[0]), int(renderer.size[1]))
    info = json.loads((src / "meta" / "info.json").read_text())
    cameras = [f"observation.images.{name}" for name in renderer.camera_keys]
    missing = [name for name in cameras if name not in info["features"]]
    if missing:
        raise SystemExit(f"{src} has no {', '.join(missing)} column")
    states = json.loads((src / "meta" / "episode_init_states.json").read_text())
    seeds = {
        int(key): int(np.asarray(entry["arrays"]["seed"]["data"]).ravel()[0])
        for key, entry in states["episodes"].items()
    }

    stats: dict[int, dict[str, dict]] = {}
    for path in sorted(src.glob("data/chunk-*/file-*.parquet")):
        table = pq.read_table(path)
        target = out / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        owners = table.column("episode_index").to_numpy()
        _, starts = np.unique(owners, return_index=True)
        bounds = [*sorted(starts.tolist()), len(owners)]
        with pq.ParquetWriter(target, table.schema) as writer:
            for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
                rows = table.slice(start, stop - start)
                episode = int(owners[start])
                arrays = {
                    "full_qpos": _column(rows, "full_qpos"),
                    "seed": seeds[episode],
                }
                if "full_qvel" in rows.column_names:
                    arrays["full_qvel"] = _column(rows, "full_qvel")
                print(
                    f"re-rendering episode {episode} ({rows.num_rows} frames)",
                    flush=True,
                )
                frames = renderer.render_episode(EpisodeArrays(arrays))
                stats[episode] = {}
                for i, name in enumerate(cameras):
                    index = rows.schema.get_field_index(name)
                    rows = rows.set_column(
                        index,
                        rows.schema.field(index),
                        _png_column(frames[:, i], rows.schema.field(index).type),
                    )
                    stats[episode][name] = image_stats(frames[:, i])
                writer.write_table(rows)

    for path in sorted(src.glob("meta/episodes/chunk-*/file-*.parquet")):
        table = pq.read_table(path)
        episodes = table.column("episode_index").to_pylist()
        for name in cameras:
            for key in stats[episodes[0]][name]:
                column = f"stats/{name}/{key}"
                index = table.schema.get_field_index(column)
                values = [np.asarray(stats[e][name][key]).tolist() for e in episodes]
                table = table.set_column(
                    index,
                    table.schema.field(index),
                    pa.array(values, type=table.schema.field(index).type),
                )
        target = out / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target)

    for name in cameras:
        info["features"][name]["shape"] = [3, *size]
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    totals = json.loads((src / "meta" / "stats.json").read_text())
    for name in cameras:
        merged = aggregate_image_stats([stats[e][name] for e in sorted(stats)])
        totals[name] = {
            key: np.asarray(value).tolist() for key, value in merged.items()
        }
    (out / "meta" / "stats.json").write_text(json.dumps(totals, indent=4))
    for name in ("metadata.json", "meta/source_metadata.json"):
        if (src / name).is_file():
            metadata = json.loads((src / name).read_text())
            blob = json.dumps(
                _set_camera_shape(metadata, size), indent=1, ensure_ascii=False
            )
            (out / name).write_text(blob)
    written = {
        "meta/info.json",
        "meta/stats.json",
        "metadata.json",
        "meta/source_metadata.json",
    }
    for path in sorted(src.rglob("*")):
        relative = path.relative_to(src).as_posix()
        if (
            path.is_file()
            and relative not in written
            and not relative.startswith(("data/", "meta/episodes/", "."))
        ):
            (out / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, out / relative)
    print(f"wrote {out} ({len(stats)} episodes at {size[0]}x{size[1]})")


def default_root(task: str, size: tuple[int, int]) -> Path:
    """``<dataset>-<size>/<task>``: 224 for a square 224x224, else HxW."""
    from bigym.loco.demos import hub

    label = str(size[0]) if size[0] == size[1] else f"{size[0]}x{size[1]}"
    return Path(f"{hub.dataset_repo().rsplit('/', 1)[-1]}-{label}") / path_name(task)


@dataclass
class RerenderConfig:
    """bigym-rerender-lerobot settings."""

    task: str | None = None
    """Task of the published dataset to re-render, fetched like
    ``bigym-download --task``."""
    demo_dir: str | None = None
    """A downloaded LeRobot task folder, or a replay-format npz batch."""
    root: str | None = None
    """Output directory; default ``<dataset>-<size>/<task>``. Required for an
    npz batch."""
    repo_id: str | None = None
    """LeRobot repo id of an npz batch's export, e.g.
    YOUR_HF_USER/bigym-g1-move-plate-224."""
    size: tuple[int, int] = (224, 224)
    """Re-render resolution H W (224 224 is what openpi/pi0.5 expects)."""
    task_prompt: str | None = None
    """npz batches: language instruction written to the LeRobot tasks table;
    default resolves the demo's task_name in
    bigym/loco/demos/task_instructions.yaml."""
    max_episodes: int = -1
    """npz batches: export at most this many episodes; -1 exports all."""
    fps: int | None = None
    """npz batches: passed to bigym-export-lerobot (only for batches whose
    metadata lacks control_step_seconds)."""
    videos: bool = False
    """npz batches: passed to bigym-export-lerobot, mp4 instead of lossless
    PNG (visualization only; the training loader rejects video-mode datasets)."""
    push: bool = False
    """npz batches: push the dataset to the Hub (created private)."""


def main() -> None:
    """Re-render and export demos using command-line options."""
    # Offscreen only. Set before the env stack loads: mujoco picks its
    # GL backend when it is first imported. Unset or empty both mean "pick".
    if not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = "egl"
    args = tyro.cli(RerenderConfig, description=__doc__)
    if (args.task is None) == (args.demo_dir is None):
        raise SystemExit("pass one of --task or --demo-dir")
    if args.task is not None:
        from bigym.loco.demos import hub

        src = hub.task_dir(args.task)
    else:
        src = Path(str(args.demo_dir))

    metadata = json.loads((src / "metadata.json").read_text())
    task = metadata["task"]["task_name"]
    recorded = tuple(metadata["task"].get("camera_shape", (84, 84)))
    print(
        f"{src} | task={task} robot={metadata['task'].get('robot_model')} "
        f"cameras={tuple(metadata['task'].get('camera_keys', ()))} "
        f"recorded {recorded[0]}x{recorded[1]} -> re-render "
        f"{args.size[0]}x{args.size[1]}"
    )

    if is_lerobot_task(src):
        npz_only = {
            "--repo-id": args.repo_id is not None,
            "--task-prompt": args.task_prompt is not None,
            "--max-episodes": args.max_episodes != -1,
            "--fps": args.fps is not None,
            "--videos": args.videos,
            "--push": args.push,
        }
        passed = [flag for flag, given in npz_only.items() if given]
        if passed:
            raise SystemExit(
                f"{', '.join(passed)} apply to npz batches only; a LeRobot task "
                "folder is copied with only its cameras re-rendered"
            )
        out = Path(args.root) if args.root else default_root(task, tuple(args.size))
        renderer = BatchRenderer(metadata, tuple(args.size))
        try:
            rerender_task(src, out, renderer)
        finally:
            renderer.close()
        return

    if args.root is None or args.repo_id is None:
        raise SystemExit("an npz batch needs --root and --repo-id")
    export = lerobot_export
    renderer = BatchRenderer(metadata, tuple(args.size))
    cache: dict = {}
    real_load = np.load

    def rerendering_load(file, *rest, **kwargs):
        return _RerenderedNpz(real_load(file, *rest, **kwargs), file, renderer, cache)

    export.np = _NumpyProxy(np, rerendering_load)  # ty: ignore[invalid-assignment]

    argv = [
        "bigym-export-lerobot",
        "--demo-dir",
        str(src),
        "--repo-id",
        args.repo_id,
        "--root",
        args.root,
        "--max-episodes",
        str(args.max_episodes),
    ]
    if args.task_prompt is not None:
        argv += ["--task-text", args.task_prompt]
    if args.fps is not None:
        argv += ["--fps", str(args.fps)]
    if args.videos:
        argv.append("--videos")

    saved_argv = sys.argv
    sys.argv = argv
    try:
        export.main()
    finally:
        sys.argv = saved_argv
        export.np = np  # ty: ignore[invalid-assignment]
        renderer.close()
    size = (int(args.size[0]), int(args.size[1]))
    for name in ("metadata.json", "meta/source_metadata.json"):
        path = Path(args.root) / name
        if path.is_file():
            metadata = _set_camera_shape(json.loads(path.read_text()), size)
            path.write_text(json.dumps(metadata, indent=1, ensure_ascii=False))
    if args.push:
        from lerobot.datasets.lerobot_dataset import (  # ty: ignore[unresolved-import]
            LeRobotDataset,
        )

        LeRobotDataset(args.repo_id, root=args.root).push_to_hub(private=True)
        print(f"pushed to hub: {args.repo_id} (private)")


if __name__ == "__main__":
    main()
