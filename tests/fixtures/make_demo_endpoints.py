"""Regenerate tests/fixtures/demo_endpoints.npz from the demonstration dataset.

Usage (needs Hub access to the dataset, see :mod:`bigym.loco.demos.hub`)::

    python -m tests.fixtures.make_demo_endpoints     # every published task
    python -m tests.fixtures.make_demo_endpoints --task move_plate --task pick_box

For every published task this stores the first and the last frame of one
real, successful demonstration: the episode's reset seed and the full
simulator state (``full_qpos`` / ``full_qvel``) of both frames. That is all
``tests/test_task_success_from_demos.py`` needs to check each task's success
predicate against a state a human actually reached, with no dataset and no
network at test time.

Only three columns of one episode are read, straight out of the remote
parquet file (``HfFileSystem`` fetches the needed byte ranges), so no camera
PNG is downloaded; regenerating all twenty tasks takes about three minutes,
mostly Hub round trips.

Each task's episodes are tried in dataset order and the first one whose
endpoints the CURRENT code judges correctly (predicate False on the first
frame, True on the last) is kept; a skipped episode is reported. Regenerate
only on purpose -- when the dataset is re-exported or the substrate changes
and ``SUBSTRATE_VERSION`` is bumped -- and review the diff of the report, not
just the file: the fixture pins today's predicates, so regenerating it on a
broken tree would bless the breakage.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

FIXTURE = Path(__file__).resolve().parent / "demo_endpoints.npz"

# Each episode is tried in dataset order; give up on a task after this many.
MAX_EPISODES_TRIED = 5


def build_env(task: str):
    """The official env of ``task`` with no cameras and no reset warmup.

    The warmup only settles the robot under its controller, and every pose
    the fixture holds overwrites the robot's whole state, so skipping it
    changes nothing a success predicate reads. The props that are NOT in
    qpos (rack and target poses written into the model by ``_on_reset``)
    come from the episode's seed, which :func:`instantaneous_success`'s
    caller resets on.
    """
    from bigym.loco import make

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make(task, camera_keys=(), controller={"reset_warmup_steps": 0})


def instantaneous_success(env, qpos: np.ndarray, qvel: np.ndarray) -> bool:
    """Pose ``env`` at a stored simulator state and evaluate the task predicate.

    This is the INSTANTANEOUS predicate (``BiGymEnv._success``), not the
    held success the protocol reports after ``success_hold_seconds``: a
    single stored frame cannot satisfy a hold window.

    Args:
        env: A :func:`build_env` env, already reset on the episode's seed.
        qpos: The stored ``full_qpos`` of one frame.
        qvel: The stored ``full_qvel`` of the same frame.

    Returns:
        Whether the task's success predicate holds at that state.
    """
    import mujoco

    inner = env.inner_env
    model, data = inner.model, inner.data
    if qpos.shape != (model.nq,) or qvel.shape != (model.nv,):
        raise ValueError(
            f"stored state is qpos{qpos.shape} qvel{qvel.shape} but the model "
            f"has nq={model.nq} nv={model.nv}; the fixture predates a model "
            "change -- regenerate it with "
            "python -m tests.fixtures.make_demo_endpoints"
        )
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    # Predicates are memoised per step; drop the previous frame's answer.
    inner._step_cache.clean()
    return bool(inner._success())


def _read_episode(fs, root: str, task: str, episode: dict[str, Any]):
    """Return ``(full_qpos, full_qvel)`` of one episode, ``(T, nq)``/``(T, nv)``."""
    import pyarrow.parquet as pq

    index = int(episode["episode_index"])
    path = (
        f"{root}/{task}/data/chunk-{int(episode['data/chunk_index']):03d}"
        f"/file-{int(episode['data/file_index']):03d}.parquet"
    )
    table = pq.read_table(
        path,
        columns=["episode_index", "full_qpos", "full_qvel"],
        filters=[("episode_index", "==", index)],
        filesystem=fs,
    )
    if table.num_rows != int(episode["length"]):
        raise RuntimeError(
            f"{task} episode {index}: read {table.num_rows} rows, the episode "
            f"index says {int(episode['length'])}"
        )
    qpos = np.stack([np.asarray(v, np.float64) for v in table.column("full_qpos")])
    qvel = np.stack([np.asarray(v, np.float64) for v in table.column("full_qvel")])
    return qpos, qvel


def endpoints_for_task(fs, root: str, task: str) -> dict[str, Any]:
    """Pick one demonstration of ``task`` and return its two endpoint states."""
    import pyarrow.parquet as pq

    init_states = json.loads(
        fs.read_text(f"{root}/{task}/meta/episode_init_states.json")
    )
    episodes = pq.read_table(
        f"{root}/{task}/meta/episodes/chunk-000/file-000.parquet",
        columns=[
            "episode_index",
            "length",
            "data/chunk_index",
            "data/file_index",
        ],
        filesystem=fs,
    ).to_pylist()
    episodes.sort(key=lambda e: int(e["episode_index"]))
    env = build_env(task)
    try:
        for episode in episodes[:MAX_EPISODES_TRIED]:
            index = int(episode["episode_index"])
            record = init_states["episodes"][str(index)]
            seed = int(np.asarray(record["arrays"]["seed"]["data"]).ravel()[0])
            qpos, qvel = _read_episode(fs, root, task, episode)
            env.reset(seed=seed)
            first = instantaneous_success(env, qpos[0], qvel[0])
            env.reset(seed=seed)
            last = instantaneous_success(env, qpos[-1], qvel[-1])
            if last and not first:
                return {
                    "seed": seed,
                    "episode_index": index,
                    "source_file": str(record.get("source_file", "")),
                    "length": int(qpos.shape[0]),
                    "qpos_first": qpos[0],
                    "qvel_first": qvel[0],
                    "qpos_last": qpos[-1],
                    "qvel_last": qvel[-1],
                }
            print(
                f"[skip] {task} episode {index}: predicate {first} on the first "
                f"frame, {last} on the last (want False, True)",
                flush=True,
            )
    finally:
        env.close()
    raise RuntimeError(
        f"{task}: none of the first {MAX_EPISODES_TRIED} demonstrations starts "
        "unsuccessful and ends successful under the current predicate"
    )


def main(argv: list[str] | None = None) -> int:
    """Write the fixture for the requested (default: every published) task."""
    from huggingface_hub import HfApi, HfFileSystem

    from bigym.loco.demos import hub

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="task to (re)generate (repeatable); default: every published task, "
        "and the fixture is rewritten from scratch",
    )
    parser.add_argument("--out", type=Path, default=FIXTURE)
    args = parser.parse_args(argv)

    repo = hub.dataset_repo()
    # Pin one dataset commit for the whole run and record it in the fixture.
    revision = hub.dataset_revision() or HfApi().dataset_info(repo).sha
    assert revision is not None
    root = f"datasets/{repo}@{revision}"
    fs = HfFileSystem()
    tasks = list(args.task) or list(hub.available_tasks(repo, revision))

    arrays: dict[str, np.ndarray] = {}
    if args.task and args.out.is_file():
        with np.load(args.out) as old:
            arrays = {key: old[key] for key in old.files}
    for task in tasks:
        found = endpoints_for_task(fs, root, task)
        for key, value in found.items():
            arrays[f"{task}__{key}"] = np.asarray(value)
        print(
            f"[ok] {task}: episode {found['episode_index']} "
            f"({found['source_file']}, {found['length']} frames, seed "
            f"{found['seed']}, nq={found['qpos_last'].shape[0]})",
            flush=True,
        )
    arrays["dataset_repo"] = np.asarray(repo)
    arrays["dataset_revision"] = np.asarray(revision)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)  # ty: ignore[invalid-argument-type]
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e3:.1f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
