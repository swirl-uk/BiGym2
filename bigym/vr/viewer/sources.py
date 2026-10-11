"""What ``bigym-view`` can open: demo batches, agent runs and dataset tasks."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from bigym.loco.demos import hub
from bigym.loco.demos.dataset import load_task_metadata
from bigym.vr.viewer.agent.runs import (
    batch_version,
    discover_cells,
    eval_success,
    is_agent_cell,
    sessions_of,
    tasks_of,
)
from bigym.vr.viewer.episodes import batch_success, episode_files, is_episode_file

# Training-run internals that also hold .npz files — never demo batches, and
# walking them costs minutes (replay buffers store one file per episode).
DISCOVER_PRUNE = {
    # Half-written development episodes live here until they are linked
    # into their batch (bigym.loco.agent.server writes dev/.staging/).
    ".staging",
    "buffer",
    "demo_buffer",
    "expert_demo_buffer",
    "eval_video",
    "eval_snapshots",
    "tb",
    "wandb",
    ".hydra",
    ".git",
}


def discover_batches(root: Path) -> list[Path]:
    """Find demo batch dirs (metadata.json + at least one .npz) under root."""
    root = root.resolve()
    if (root / "metadata.json").exists() and episode_files(root):
        return [root]
    batches = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in DISCOVER_PRUNE)
        if "metadata.json" in filenames and any(map(is_episode_file, filenames)):
            batches.append(Path(dirpath))
            dirnames[:] = []  # a batch has no nested batches
    return sorted(batches)


def _is_dataset_task(path: Path) -> bool:
    """True for a LeRobot task export: ``meta/info.json`` plus data parquet."""
    return (path / "meta" / "info.json").is_file() and any(
        path.glob("data/chunk-*/file-*.parquet")
    )


def discover_dataset_tasks(root: Path) -> list[Path]:
    """Find LeRobot task folders under root (a local copy of the dataset)."""
    root = root.resolve()
    if _is_dataset_task(root):
        return [root]
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in DISCOVER_PRUNE and not d.startswith(".")
        )
        if _is_dataset_task(Path(dirpath)):
            found.append(Path(dirpath))
            dirnames[:] = []  # a task folder holds no other task folder
    return sorted(found)


@dataclass
class DatasetSource:
    """Dataset tasks to offer in the Task dropdown.

    Attributes:
        labels: What the dropdown shows, one per task.
        tasks: The task names the replay environments are built for.
        dirs: Each task's local folder, or None to fetch it from the Hub.
        roots: The directories the tasks were found under; empty for the
            published dataset.
    """

    labels: list[str]
    tasks: list[str]
    dirs: list[Path | None]
    roots: list[Path] = field(default_factory=list)

    @property
    def where(self) -> str:
        """Where the tasks come from, for the log and the UI."""
        if not self.roots:
            return f"{hub.dataset_repo()} (Hugging Face Hub cache)"
        return shlex.join(str(r) for r in self.roots)


@dataclass
class BatchSource:
    """Collector batches and agent-run cells found under some directories."""

    roots: list[Path]
    batches: list[Path]
    labels: list[str]
    cells: list
    label_root: Path

    @property
    def where(self) -> str:
        """The directories, as the "Demo directory" field shows them."""
        return shlex.join(str(r) for r in self.roots)


def published_source() -> DatasetSource:
    """The published dataset's tasks (``hub.available_tasks``)."""
    tasks = list(hub.available_tasks())
    if not tasks:
        raise FileNotFoundError(f"dataset {hub.dataset_repo()} lists no tasks")
    return DatasetSource(labels=tasks, tasks=tasks, dirs=[None] * len(tasks))


def local_source(roots: list[Path]) -> DatasetSource | BatchSource:
    """What a list of directories holds: batches/agent runs, else dataset tasks.

    Raises:
        FileNotFoundError: A path does not exist, or nothing viewable is
            under them.
    """
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"{root} does not exist")
    roots = [root.resolve() for root in roots]
    batches = discover_all(roots)
    if batches:
        label_root = _label_root(roots)
        return BatchSource(
            roots=roots,
            batches=batches,
            labels=batch_labels(label_root, batches),
            cells=discover_cells(roots),
            label_root=label_root,
        )
    folders: list[Path] = []
    for root in roots:
        folders += [f for f in discover_dataset_tasks(root) if f not in folders]
    if not folders:
        raise FileNotFoundError(
            "No demos found under "
            + ", ".join(str(r) for r in roots)
            + ": looked for demo batches (metadata.json + *.npz), agent runs "
            "and dataset task folders (meta/info.json + data/*.parquet)"
        )
    label_root = _label_root(roots)
    labels, tasks = [], []
    for folder in folders:
        try:
            rel = folder.relative_to(label_root)
        except ValueError:
            rel = folder
        labels.append(str(rel) if str(rel) != "." else folder.name)
        tasks.append(str(load_task_metadata(folder)["task"]["task_name"]))
    return DatasetSource(labels=labels, tasks=tasks, dirs=list(folders), roots=roots)


def find_source(text: str) -> DatasetSource | BatchSource:
    """Resolve the "Demo directory" field: empty is the published dataset.

    Several directories are separated by spaces (quote one that has a space
    in its name), as on the command line.
    """
    parts = shlex.split(text.strip())
    if not parts:
        return published_source()
    return local_source([Path(part).expanduser() for part in parts])


# The folder browser of the "Demo directory" panel. It lists the viewer
# machine's own filesystem (often a remote server), so the marks stay cheap:
# marker files in the folder itself and one level down, never a full walk.
BROWSE_LIMIT = 200


PEEK_LIMIT = 200


MARK = "● "


RECENT: list[str] = []  # directories loaded this session, newest first


def demo_marker(path: Path) -> bool:
    """True when path is a dataset task folder, a demo batch or an agent cell."""
    return (
        (path / "meta" / "info.json").is_file()
        or (path / "metadata.json").is_file()
        or is_agent_cell(path)
    )


def subfolders(path: Path, limit: int | None = None) -> tuple[list[str], int]:
    """Visible subfolder names of path, sorted: the first ``limit`` and the total.

    Hidden folders and training-run internals are left out.

    Raises:
        OSError: path cannot be listed (missing, not a folder, no permission).
    """
    names = []
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.name.startswith(".") or entry.name in DISCOVER_PRUNE:
                continue
            try:
                if entry.is_dir():
                    names.append(entry.name)
            except OSError:
                continue
    names.sort()
    return (names[:limit] if limit is not None else names), len(names)


def has_demos(path: Path) -> bool:
    """True when path, or a folder directly inside it, holds something to view."""
    try:
        if demo_marker(path):
            return True
        with os.scandir(path) as entries:
            for count, entry in enumerate(entries):
                if count >= PEEK_LIMIT:
                    break
                if entry.name.startswith(".") or entry.name in DISCOVER_PRUNE:
                    continue
                if entry.is_dir() and demo_marker(Path(entry.path)):
                    return True
    except OSError:
        return False
    return False


def browse_dir(text: str) -> Path | None:
    """The folder the browser lists for the field's text.

    Empty lists the working directory; several paths or a path that is not a
    folder list nothing.
    """
    try:
        parts = shlex.split(text.strip())
    except ValueError:
        return None
    if not parts:
        return Path.cwd()
    if len(parts) > 1:
        return None
    path = Path(parts[0]).expanduser().resolve()
    return path if path.is_dir() else None


def quick_places() -> dict[str, str]:
    """Starting points for the browser: label -> what goes in the field."""
    places = {"published dataset (Hugging Face cache)": ""}
    cwd = Path.cwd()
    if (cwd / "bigym_demos").is_dir():
        places["./bigym_demos"] = shlex.quote(str(cwd / "bigym_demos"))
    places[f"working directory: {cwd}"] = shlex.quote(str(cwd))
    places[f"home: {Path.home()}"] = shlex.quote(str(Path.home()))
    for text in RECENT:
        places.setdefault(f"loaded: {text}", text)
    return places


def newest_batch(batches: list[Path]) -> int | None:
    """Index of the batch with the most recently written episode file.

    Used by --follow's "auto-jump to newest": a session that is running
    keeps adding episodes to the batch it is writing, so the freshest
    ``*.npz`` mtime is what "newest" means here.
    """
    best_index, best_time = None, -1.0
    for i, batch in enumerate(batches):
        newest = -1.0
        for path in episode_files(batch):
            try:
                newest = max(newest, path.stat().st_mtime)
            except OSError:
                continue
        if newest > best_time:
            best_index, best_time = i, newest
    return best_index


def _batch_shown_name(rel: Path, batch: Path, kind: str, version: int | None) -> str:
    """The path a run-layout batch is labelled with (``batch/`` dropped)."""
    leaf = batch.parent.name
    if kind == "dev" and version is not None:
        # dev/v006_<stamp> is one directory per version and start time; the
        # label names the version.
        leaf = f"v{version:03d}"
    grand = rel.parent.parent
    if str(grand) in (".", ""):  # --demo-dir pointed at (or just above) it
        return f"{kind}/{leaf}"
    return str(grand / leaf)


def batch_labels(root: Path, batches: list[Path]) -> list[str]:
    """Dropdown/terminal labels for discovered batches.

    Ordinary batches read ``<path relative to root> · <n> eps``. A batch of
    the agent run layout drops its ``batch/`` leaf (``eval/v007``,
    ``replays/v007``, ``replays/policy_ab12cd34``, ``dev/v006``); an
    evaluation appends the version's hidden-seed success rate when
    ``eval/vNNN/summary.json`` is there, and a development batch the mean of
    its episodes' stored ``success``.
    """
    labels = []
    for b in batches:
        try:
            rel = b.relative_to(root.resolve())
        except ValueError:
            rel = b
        count = len(episode_files(b))
        info = batch_version(b)
        if info is None:
            labels.append(f"{rel} · {count} eps")
            continue
        cell, kind, version = info
        label = f"{_batch_shown_name(rel, b, kind, version)} · {count} eps"
        rate = None
        if kind == "eval" and version is not None:
            rate = eval_success(cell, version)
        elif kind == "dev":
            rate = batch_success(b)
        if rate is not None:
            label += f" · success {rate:.2f}"
        labels.append(label)
    return labels


def rescan_into(root: Path, batches: list[Path], labels: list[str]) -> bool:
    """Re-discover the batches under root, updating both lists in place.

    The viewer hands the same two lists to the frontend, which indexes into
    them, so a rescan must mutate them rather than replace them.

    Args:
        root: The directory batches were discovered under.
        batches: The live list of batch directories.
        labels: The live list of their labels.

    Returns:
        True when the set of batches changed (a new one appeared, or one
        that was still being written now has episodes).
    """
    fresh = discover_batches(root)
    changed = fresh != batches
    batches[:] = fresh
    # Labels carry episode counts and success rates, which grow while a
    # batch is being written, so they are rebuilt on every scan.
    labels[:] = batch_labels(root, fresh)
    return changed


def pick_batch(labels: list[str], selector: str | None) -> int:
    """Resolve --batch (an index or a label substring) to a batch index."""
    if selector is None:
        return 0
    text = str(selector).strip()
    if text.isdigit() and int(text) < len(labels):
        return int(text)
    matches = [i for i, label in enumerate(labels) if text in label]
    if not matches:
        raise SystemExit(
            f"--batch {selector!r} matches no batch; found: " + ", ".join(labels)
        )
    if len(matches) > 1:
        raise SystemExit(
            f"--batch {selector!r} is ambiguous: "
            + ", ".join(labels[i] for i in matches)
        )
    return matches[0]


def _label_root(roots: list[Path]) -> Path:
    """The directory batch labels are shown relative to."""
    if len(roots) == 1:
        return roots[0]
    return Path(os.path.commonpath([str(r) for r in roots]))


def discover_all(roots: list[Path]) -> list[Path]:
    """Every batch under every --demo-dir path, deduplicated and sorted."""
    found: list[Path] = []
    for root in roots:
        for batch in discover_batches(root):
            if batch not in found:
                found.append(batch)
    return sorted(found)


def rescan_all(
    roots: list[Path], label_root: Path, batches: list[Path], labels: list[str]
) -> bool:
    """Re-discover every root into the viewer's live lists; True when changed."""
    fresh = discover_all(roots)
    changed = fresh != batches
    batches[:] = fresh
    labels[:] = batch_labels(label_root, fresh)
    return changed


def print_discovery(roots: list[Path], cells: list) -> None:
    """The first lines of the startup log: what --demo-dir turned out to be."""
    if not cells:
        return
    sessions = sessions_of(cells)
    tasks = tasks_of(cells)
    if len(cells) == 1:
        print(f"[viewer] agent cell {cells[0].path}")
        return
    print(
        f"[viewer] agent runs under {_label_root(roots)}: {len(sessions)} "
        f"sessions ({', '.join(sessions)}) x {len(tasks)} tasks, "
        f"{len(cells)} cells"
    )


def pick_task(source: DatasetSource, selector: str | None) -> int:
    """Resolve --task (a task name or label) to an index into the source."""
    if selector is None:
        return 0
    for names in (source.tasks, source.labels):
        if selector in names:
            return names.index(selector)
    raise SystemExit(
        f"--task {selector!r} is not in {source.where}; it has: "
        + ", ".join(source.labels)
    )
