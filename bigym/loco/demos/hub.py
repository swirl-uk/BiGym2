"""Hugging Face Hub access to the BiGym 2.0 demonstration dataset.

The public demonstrations live in one Hugging Face *dataset* repository with
one top-level folder per task, each a LeRobot v3 lossless export::

    <repo>/
      move_plate/            data/chunk-000/file-*.parquet + meta/ + metadata.json
      reach_target_single/   ...

Two ways to get them onto a machine:

- **Lazy, per task**: :func:`task_dir` fetches one task's folder the first
  time it is needed. ``env.get_demos()`` calls it, so running a task pulls
  exactly that task's demonstrations (a few hundred MB to a few GB).
- **Ahead of time**: ``bigym-download --all`` (or :func:`download_all`)
  mirrors the whole dataset; ``bigym-download --task move_plate
  pick_box`` fetches a subset. ``--local-dir PATH`` writes the files into a
  plain folder instead of the cache, which ``bigym-view --demo-dir PATH``
  opens.

Both go through ``huggingface_hub.snapshot_download`` and share its cache
(``$HF_HOME`` / ``$HF_HUB_CACHE``; default ``~/.cache/huggingface``), so a
pre-download and a later lazy load never fetch a file twice, and the usual
Hub knobs apply (``HF_TOKEN`` for private/gated repos, ``HF_HUB_OFFLINE=1``
to refuse network access and use the cache only).

The repository defaults to :data:`DEFAULT_DATASET_REPO`; override it with the
``BIGYM_DATASET_REPO`` environment variable (and optionally pin a revision
with ``BIGYM_DATASET_REVISION``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Iterable

import tyro
from huggingface_hub import HfApi, snapshot_download

from bigym.loco.objref import path_name

DEFAULT_DATASET_REPO = "SWIRL-Lab/bigym-g1-native60"
REPO_ENV_VAR = "BIGYM_DATASET_REPO"
REVISION_ENV_VAR = "BIGYM_DATASET_REVISION"

TASK_MARKER = "meta/info.json"
AGENT_DEMO_FOLDER = "agent_demos"


class DemosUnavailableError(RuntimeError):
    """The dataset repository has no demonstrations for the requested task.

    For a benchmark task this means its demonstrations have not been
    published yet: the dataset is released task by task, and the message
    lists what is published and what is still pending.
    """


def dataset_url(repo: str | None = None) -> str:
    """The dataset's page on the Hub."""
    return f"https://huggingface.co/datasets/{dataset_repo(repo)}"


def dataset_repo(repo: str | None = None) -> str:
    """Return the dataset repo id: argument, else env override, else default."""
    return str(repo or os.environ.get(REPO_ENV_VAR) or DEFAULT_DATASET_REPO)


def dataset_revision(revision: str | None = None) -> str | None:
    """Return the pinned dataset revision, if any (argument, else env)."""
    value = revision or os.environ.get(REVISION_ENV_VAR)
    return str(value) if value else None


def available_tasks(
    repo: str | None = None, revision: str | None = None
) -> tuple[str, ...]:
    """List the tasks that have a demonstration folder in the dataset."""
    files = HfApi().list_repo_files(
        dataset_repo(repo), repo_type="dataset", revision=dataset_revision(revision)
    )
    tasks = {
        f.split("/", 1)[0]
        for f in files
        if f.count("/") >= 2 and f.split("/", 1)[1] == TASK_MARKER
    }
    return tuple(sorted(tasks))


def pending_tasks(
    repo: str | None = None, revision: str | None = None
) -> tuple[str, ...]:
    """Benchmark tasks whose demonstrations are not in the dataset yet."""
    from bigym.loco.tasks import TASK_MAP

    published = set(available_tasks(repo, revision))
    return tuple(sorted(name for name in TASK_MAP if name not in published))


def _snapshot(
    repo: str | None,
    revision: str | None,
    allow_patterns: Iterable[str] | None,
    local_dir: Path | None = None,
) -> Path:
    return Path(
        snapshot_download(
            dataset_repo(repo),
            repo_type="dataset",
            revision=dataset_revision(revision),
            allow_patterns=list(allow_patterns) if allow_patterns else None,
            local_dir=str(local_dir) if local_dir is not None else None,
        )
    )


def task_dir(
    task: str,
    repo: str | None = None,
    revision: str | None = None,
    local_dir: Path | None = None,
) -> Path:
    """Return the local folder of one task's demonstrations, downloading it if needed.

    Only that task's files are fetched (``<task>/**``; a
    ``"pkg.module:ATTR"`` task's folder is ``pkg.module-ATTR``, see
    :func:`bigym.loco.objref.path_name`), into the Hub cache or, with
    ``local_dir``, into ``local_dir/<task>``. Raises
    :class:`DemosUnavailableError` when the dataset has no folder for the
    task, naming the tasks it does have.
    """
    return task_files(task, None, repo, revision, local_dir)


def task_files(
    task: str,
    patterns: Iterable[str] | None,
    repo: str | None = None,
    revision: str | None = None,
    local_dir: Path | None = None,
) -> Path:
    """Fetch some files of a task's folder and return the folder.

    ``patterns`` are glob patterns relative to the task's folder, such as
    ``meta/**``; None fetches the whole folder. The task's metadata
    (``metadata.json`` and ``meta/``) always comes along. Raises
    :class:`DemosUnavailableError` like :func:`task_dir`.
    """
    task = str(task)
    folder_name = path_name(task)
    wanted = ["**"] if patterns is None else ["metadata.json", "meta/**", *patterns]
    root = _snapshot(
        repo, revision, [f"{folder_name}/{pattern}" for pattern in wanted], local_dir
    )
    folder = root / folder_name
    if (folder / TASK_MARKER).is_file():
        return folder
    raise DemosUnavailableError(_unavailable_message(task, repo, revision))


def agent_demo_dir(
    task: str, repo: str | None = None, revision: str | None = None
) -> Path | None:
    """Return the published agent demonstration videos of a task, or None.

    They sit in ``agent_demos/<task>/`` of the dataset: the files
    ``bigym-agent`` puts in a sandbox under its default demonstration
    settings, as the benchmark's sessions were given them.
    """
    folder = AGENT_DEMO_FOLDER + "/" + path_name(str(task))
    found = _snapshot(repo, revision, [f"{folder}/**"]) / folder
    return found if any(found.glob("*.mp4")) else None


def snapshot_revision(folder: Path) -> str | None:
    """The commit a folder of the Hub cache was downloaded at, or None.

    Pass it as ``revision`` to fetch more files of the same dataset version.
    """
    snapshot = Path(folder).parent
    return snapshot.name if snapshot.parent.name == "snapshots" else None


def _unavailable_message(task: str, repo: str | None, revision: str | None) -> str:
    from bigym.loco.tasks import TASK_MAP

    published = available_tasks(repo, revision)
    pending = pending_tasks(repo, revision)
    total = len(TASK_MAP)
    if task not in TASK_MAP:
        return (
            f"{task!r} is not a BiGym 2.0 benchmark task, so dataset "
            f"{dataset_repo(repo)!r} has no demonstrations for it "
            f"(see bigym.loco.tasks.TASK_MAP)"
        )
    lines = [
        f"Demonstrations for task {task!r} have not been published yet.",
        f"Dataset {dataset_repo(repo)!r} currently covers {len(published)}/{total} "
        "benchmark tasks; the remaining ones are released in later dataset "
        f"updates. Watch {dataset_url(repo)} for the next batch, or point "
        f"${REPO_ENV_VAR} at your own export of this task.",
    ]
    if published:
        lines.append(f"Published ({len(published)}): {', '.join(published)}")
    if pending:
        lines.append(f"Pending ({len(pending)}): {', '.join(pending)}")
    return "\n".join(lines)


def download_all(
    repo: str | None = None,
    revision: str | None = None,
    local_dir: Path | None = None,
) -> Path:
    """Mirror the whole dataset (Hub cache, or ``local_dir``); return its root."""
    return _snapshot(repo, revision, None, local_dir)


@dataclass
class DownloadConfig:
    """Download BiGym 2.0 demonstrations from the Hugging Face Hub.

    They land in the Hub cache, so env.get_demos() never waits on the network.
    """

    task: list[str] = field(default_factory=list)
    """Tasks to fetch; default: nothing unless --all."""
    all: bool = False
    """Fetch every task."""
    agent: bool = False
    """Fetch only what bigym-agent needs: each task's metadata and its
    demonstration videos, not the demonstrations."""
    list_tasks: Annotated[bool, tyro.conf.arg(name="list")] = False
    """List the tasks the dataset provides."""
    repo: str | None = None
    """Dataset repo id (default: $BIGYM_DATASET_REPO or
    SWIRL-Lab/bigym-g1-native60)."""
    revision: str | None = None
    """Dataset revision to pin."""
    local_dir: Path | None = None
    """Download into this folder (one subfolder per task) instead of the Hub
    cache; bigym-view --demo-dir PATH opens it."""


def main(argv: list[str] | None = None) -> int:
    """``bigym-download``: fetch demonstrations ahead of time."""
    args = tyro.cli(DownloadConfig, args=argv, prog="bigym-download")

    repo = dataset_repo(args.repo)
    if args.list_tasks or not (args.all or args.task):
        from bigym.loco.tasks import TASK_MAP

        published = available_tasks(repo, args.revision)
        pending = pending_tasks(repo, args.revision)
        print(f"{repo}: {len(published)}/{len(TASK_MAP)} benchmark tasks published")
        for name in published:
            print(f"  {name}")
        if pending:
            print(f"pending ({len(pending)}, released in later dataset updates):")
            for name in pending:
                print(f"  {name}")
        if not (args.all or args.task):
            print("Pass --all or --task NAME to download.")
        return 0
    if args.agent:
        if args.local_dir is not None:
            raise SystemExit(
                "--agent fills the Hub cache bigym-agent reads; drop --local-dir"
            )
        for name in args.task or available_tasks(repo, args.revision):
            task_files(name, [], repo, args.revision)
            videos = agent_demo_dir(name, repo, args.revision)
            print(f"{name}: {videos or 'metadata only, no published videos'}")
        return 0
    if args.all:
        root = download_all(repo, args.revision, args.local_dir)
        print(f"downloaded every task of {repo} to {root}")
        return 0
    for name in args.task:
        folder = task_dir(name, repo, args.revision, args.local_dir)
        if args.local_dir is None:
            agent_demo_dir(name, repo, args.revision)  # what bigym-agent reads
        print(f"{name}: {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
