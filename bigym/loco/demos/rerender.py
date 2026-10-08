"""Export a replay-format npz demo batch to LeRobot v3 with re-rendered cameras.

Same output as ``bigym.loco.demos.lerobot_export`` in every respect — state, action,
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


@dataclass
class RerenderConfig:
    """bigym-rerender-lerobot settings."""

    demo_dir: str
    """Replay-format npz batch to re-render."""
    repo_id: str
    """LeRobot repo id, e.g. YOUR_HF_USER/bigym-g1-move-plate-224."""
    root: str
    """Output directory of the dataset."""
    size: tuple[int, int] = (224, 224)
    """Re-render resolution H W (224 224 is what openpi/pi0.5 expects)."""
    task_prompt: str | None = None
    """Language instruction written to the LeRobot tasks table; default
    resolves the demo's task_name in bigym/loco/demos/task_instructions.yaml."""
    max_episodes: int = -1
    """Export at most this many episodes; -1 exports all."""
    fps: int | None = None
    """Passed to bigym-export-lerobot (only for batches whose metadata lacks
    control_step_seconds)."""
    videos: bool = False
    """Passed to bigym-export-lerobot: mp4 instead of lossless PNG
    (visualization only; the training loader rejects video-mode datasets)."""
    push: bool = False
    """Push the dataset to the Hub (created private)."""


def main() -> None:
    """Re-render and export demos using command-line options."""
    # Offscreen only. Set before the env stack loads: mujoco picks its
    # GL backend when it is first imported. Unset or empty both mean "pick".
    if not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = "egl"
    args = tyro.cli(RerenderConfig, description=__doc__)

    src = Path(args.demo_dir)
    metadata = json.loads((src / "metadata.json").read_text())
    recorded = tuple(metadata["task"].get("camera_shape", (84, 84)))
    print(
        f"batch {src} | task={metadata['task']['task_name']} "
        f"robot={metadata['task'].get('robot_model')} "
        f"cameras={tuple(metadata['task'].get('camera_keys', ()))} "
        f"recorded {recorded[0]}x{recorded[1]} -> re-render "
        f"{args.size[0]}x{args.size[1]}"
    )

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
    if args.push:
        argv.append("--push")

    saved_argv = sys.argv
    sys.argv = argv
    try:
        export.main()
    finally:
        sys.argv = saved_argv
        export.np = np  # ty: ignore[invalid-assignment]
        renderer.close()


if __name__ == "__main__":
    main()
