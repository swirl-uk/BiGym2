#!/usr/bin/env python3
"""Demo viewer: the published demonstrations, demo batches and agent-run cells.

Kinematic playback: each frame poses the scene from the demo's stored
``full_qpos`` (no physics, no lower-body policy stepping). Two frontends:

--viewer viser (default)
    Browser 3D scene via mjviser (infinite grid ground,
    no skybox, IBL lighting), remote-friendly (ssh -L <port>). The "Demo
    directory" field at the top opens another directory (or, left empty, the
    published dataset) without restarting; it browses the folders of the
    machine running the viewer, ● marking those with demos. The "Figure
    capture" panel grabs the current view at an explicit pixel size (viser
    renders in the browser, so the window caps it) with the target spheres
    and velocity arrows hidden, and writes it to --figure-dir. Target spheres
    are drawn by the viewer itself (the in-model geoms are hidden from
    mjviser, whose meshes are baked once at build time and would go stale
    when an episode reset moves the target), with a glow when reached. The
    per-frame stored reward is shown in the Playback panel. "Camera panels"
    shows MuJoCo's own pixels for the current frame: the robot's onboard
    cameras through the environment's observation path, plus one
    third-person view.

--viewer mujoco
    Native MuJoCo window (needs a local display). Keyboard: SPACE play/pause,
    LEFT/RIGHT step frame, UP/DOWN switch episode, ``[`` / ``]`` switch batch
    or task (the scene is rebuilt the way the viser dropdown rebuilds it).
    Task visuals (e.g. the reach-target highlight) work natively because the
    renderer reads live geom colors. ``--batch`` / ``--task`` and
    ``--episode`` pick what opens first; with several batches and no
    ``--batch`` the list is printed once and the first one opens. The native
    frontend has no compare mode, but ``--demo-dir <root>`` still opens the
    first cell's evaluation in it.

What opens is decided by ``--demo-dir`` (or the "Demo directory" field):

nothing: the published dataset
    The demonstrations on the Hugging Face Hub (``bigym.loco.demos.hub``:
    ``BIGYM_DATASET_REPO`` / ``BIGYM_DATASET_REVISION`` apply), read from the
    Hub cache. The viser frontend offers the dataset's tasks in a "Task"
    dropdown and fetches a task's folder the first time it is picked;
    ``--task`` picks the one that opens first.

plain demo directories
    A collector batch, an exported dataset or any folder above them. Batches
    are discovered recursively and the viser frontend offers them in one
    "Demo batch" dropdown.

a local copy of the dataset
    A LeRobot task folder (``meta/info.json`` + ``data/*.parquet``) or a
    folder of them, such as ``bigym-download --local-dir PATH``. Opens like
    the published dataset, with the Task dropdown listing the folders found.
    A directory that holds collector batches opens as batches.

agent runs
    A cell (one task x one session of the coding-agent benchmark, i.e. a
    directory with ``policies/index.json``), a session root of cells, or a
    directory of such roots. ``--demo-dir`` takes several paths, so
    ``--demo-dir runs/s1 runs/s2 runs/s3`` is three sessions. The sidebar
    then shows Task and Session dropdowns, a two-level **Rollouts** picker
    (Evaluation / Development / Replays, then the item), the session card
    from ``run.json``, the Policy versions / Policy code / Transcript panels
    and a **Compare** panel that puts up to four policies of the same task
    side by side on one seed. ``bigym/vr/viewer/agent_cells.py`` holds all of it.

Dataset episodes are replayed on the task's official environment
(``make(task, camera_keys=())``), reset on the episode's seed from
``meta/episode_init_states.json``, with the task's ``_on_step`` hook run per
frame (:mod:`bigym.loco.demos.kinematic`).

``--follow`` rescans the cell every ``--follow-seconds`` so a session being
watched live grows new rollouts, versions and summary figures in place; the
scan runs on the main loop and never switches the rollout on screen unless
"auto-jump to newest" is ticked.

Threading note: env.reset() renders camera observations, and the EGL context
belongs to the main thread — episode loads, camera renders and subprocess
launches therefore ONLY happen on the main loop; UI/keyboard callbacks just
set a request flag (downloads and directory scans run on a worker thread
and then do the same).

Usage:
    # the published demonstrations, in the browser
    bigym-view
    bigym-view --task move_plate --episode 3

    # point at any folder; demo batches inside are discovered automatically
    MUJOCO_GL=egl bigym-view --demo-dir <demo-root> --port 8080

    # or a specific batch, in a native window
    bigym-view --demo-dir <demo-root>/<task>/<timestamp> --viewer mujoco

    # one agent cell, or every session under a directory of run roots
    MUJOCO_GL=egl bigym-view --demo-dir <runs>/<session>/<task>
    MUJOCO_GL=egl bigym-view --demo-dir <runs>

    # compare three sessions' policies on one hidden seed, from the shell
    MUJOCO_GL=egl bigym-view \
        --demo-dir <runs>/s1/move_plate <runs>/s2/move_plate \
        --compare "s1:v008,s2:v007,s3:last" --seed 620003

    # headless smoke: serve for 20 s, then exit
    MUJOCO_GL=egl bigym-view --demo-dir <cell> --exit-after-seconds 20
"""

from __future__ import annotations

import html
import json
import os
import shlex
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import imageio.v2 as imageio
import mujoco
import numpy as np
import trimesh
import tyro
import viser
import viser.transforms as vtf

from bigym.cli import usage_error
from bigym.envs.reach_target import Target
from bigym.loco import EnvConfig, make
from bigym.loco.agent.demo_video import TaskDemos
from bigym.loco.demos import hub
from bigym.loco.demos.dataset import load_task_metadata
from bigym.loco.demos.kinematic import replay_env, run_task_hook
from bigym.loco.objref import path_name
from bigym.vr.viewer import agent_cells, video
from bigym.vr.viewer.mjviser_compat import (
    hide_reach_targets,
    open_viser_scene,
    tint_robot_geoms,
)

END_HOLD_SECONDS = 1.5  # freeze on the last frame before looping
# Nominal collection rate. Playback prefers metadata["control_step_seconds"],
# falling back to model.opt.timestep x substeps when a batch does not stamp it.
DEMO_HZ = 50.0

SKY_OPTIONS = ("grey", "black", "transparent")

# --follow rescans the cell this often by default (seconds).
FOLLOW_SECONDS = 5.0
SPARKLINE_SIZE = (252, 26)  # width, height in px of the reward strip
# Cameras a task XML may provide for a third-person view, best first; without
# one the panel falls back to a free camera orbiting the robot.
THIRD_PERSON_CAMERAS = ("rec_third", "third_person", "external")
THIRD_PERSON_SIZE = (256, 256)
CAMERA_PANEL_HZ = 10.0  # image widgets are re-encoded per update; 10 Hz is plenty
# Lighting panel state, kept across batch switches (see LightingPanel). Black is
# the default sky: it is what slides, videos and the compare scene want, and
# a capture keeps it.
_LIGHTING_PREFS: dict = {
    "sky": "black",
    "environment": 0.7,
    "default_lights": True,
    "shadows": True,
    "key_light": 0.0,
}

RECORD_SECONDS = 10.0


@dataclass
class ViewConfig:
    """bigym-view settings."""

    demo_dir: list[Path] | None = None
    """One or more paths: a demo batch dir, any ancestor folder (batches are
    discovered recursively), a local copy of the dataset (LeRobot task
    folders), an agent cell, a session root of cells, or a directory holding
    such roots (default: the published demonstrations from the Hugging Face
    Hub)."""
    viewer: Literal["mujoco", "viser"] = "viser"
    """viser: browser UI, works over ssh -L; mujoco: native window on a local
    display."""
    port: int = 8080
    """viser only: HTTP port."""
    robot_tint: Literal["none", "dark"] = "none"
    """viser only: darken the robot's geoms so it separates from white
    fixtures (display only; the model and observations are untouched)."""
    figure_dir: str = "figure_captures"
    """viser only: where the Figure capture button writes its PNGs."""
    batch: str | None = None
    """Which discovered batch to open first: an index into the printed list,
    or a substring of its label (default: the first)."""
    task: str | None = None
    """Which dataset task to open first (published or a local copy; default:
    the first)."""
    episode: int = 0
    """Episode index to open first."""
    follow: bool = False
    """viser only: rescan the cell every --follow-seconds so a live session's
    new batches, policy versions and run.json show up."""
    follow_seconds: float = FOLLOW_SECONDS
    """How often --follow rescans."""
    compare: str | None = None
    """viser only: start in compare mode, e.g. --compare "s1:v008,s2:v007,s3:last"
    — a comma-separated list of up to 6 <session>:<version> slots (version:
    vNNN, NNN, last or submission; session matched by name or a unique
    substring). The task is the one the viewer would open, so point
    --demo-dir at it."""
    seed: int = agent_cells.EVAL_SEED_LO
    """The seed compare mode poses every slot on."""
    compare_gap: float = agent_cells.COMPARE_GAP
    """Metres between neighbouring compare slots along +y."""
    layout: Literal["auto", "row", "2 columns", "3 columns"] = "auto"
    """How compare mode places its slots: a column steps along +y and a row
    steps back along -x (auto: one row up to 3 slots, two columns from four)."""
    exit_after_seconds: float = 0.0
    """viser only: serve for this many seconds and exit (0 = until Ctrl-C).
    Headless smoke tests use it to check a cell renders."""
    record: str | None = None
    """Compare mode only: once the scene is built, record it from the overview
    camera to this H.264 mp4 and exit. A headless Chrome renders the frames
    unless --browser says otherwise."""
    record_seconds: float = RECORD_SECONDS
    """Length of the recording; the episodes play at their recorded speed and
    wrap around."""
    record_size: str = video.RECORD_SIZE
    """Frame size of the recording, WxH."""
    record_fps: float = 0.0
    """Frame rate written into the file (0: the episodes' own rate, so one
    video frame per episode step)."""
    browser: str = "headless"
    """Who renders a recording: headless starts a headless Chrome/Chromium
    when nobody is connected after 3s, none waits for you to open the URL,
    anything else is a browser executable."""
    record_then_stay: bool = False
    """Keep serving the scene after the recording instead of exiting."""
    record_view: Literal["overview", "oblique"] = "overview"
    """Where the recording camera stands: overview (the live compare's camera,
    behind the robots) or oblique (in front and to one side, lower and
    closer: hands and tables, not backs)."""
    record_zoom: float = 1.0
    """Narrow the recording lens by this factor (2 = twice as close)."""
    record_speed: float = 1.0
    """Playback speed of the recording (2 = twice real time)."""
    record_label_scale: float = agent_cells.RECORD_LABEL_SCALE
    """How much bigger the billboards are drawn in a recording than live."""

    def __post_init__(self) -> None:
        """Reject values the viewer cannot act on.

        Raises:
            ValueError: A flag is out of range or needs another one.
        """
        if self.record and not self.compare:
            raise ValueError("--record needs --compare")
        if self.episode < 0:
            raise ValueError(f"--episode must be >= 0, got {self.episode}")
        if self.follow_seconds <= 0.0:
            raise ValueError(
                f"--follow-seconds must be > 0, got {self.follow_seconds:g}"
            )
        for flag, value in (
            ("--exit-after-seconds", self.exit_after_seconds),
            ("--record-fps", self.record_fps),
        ):
            if value < 0.0:
                raise ValueError(f"{flag} must be >= 0, got {value:g}")
        for flag, value in (
            ("--record-seconds", self.record_seconds),
            ("--record-zoom", self.record_zoom),
            ("--record-speed", self.record_speed),
            ("--record-label-scale", self.record_label_scale),
        ):
            if value <= 0.0:
                raise ValueError(f"{flag} must be > 0, got {value:g}")


def _parse_args() -> ViewConfig:
    try:
        return tyro.cli(ViewConfig, prog="bigym-view", description=__doc__)
    except ValueError as exc:
        usage_error(str(exc), prog="bigym-view")


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
    if (root / "metadata.json").exists() and agent_cells.episode_files(root):
        return [root]
    batches = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in DISCOVER_PRUNE)
        if "metadata.json" in filenames and any(
            map(agent_cells.is_episode_file, filenames)
        ):
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


def _published_source() -> DatasetSource:
    """The published dataset's tasks (``hub.available_tasks``)."""
    tasks = list(hub.available_tasks())
    if not tasks:
        raise FileNotFoundError(f"dataset {hub.dataset_repo()} lists no tasks")
    return DatasetSource(labels=tasks, tasks=tasks, dirs=[None] * len(tasks))


def _local_source(roots: list[Path]) -> DatasetSource | BatchSource:
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
            cells=agent_cells.discover_cells(roots),
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
        return _published_source()
    return _local_source([Path(part).expanduser() for part in parts])


# The folder browser of the "Demo directory" panel. It lists the viewer
# machine's own filesystem (often a remote server), so the marks stay cheap:
# marker files in the folder itself and one level down, never a full walk.
BROWSE_LIMIT = 200
PEEK_LIMIT = 200
MARK = "● "
_RECENT: list[str] = []  # directories loaded this session, newest first


def demo_marker(path: Path) -> bool:
    """True when path is a dataset task folder, a demo batch or an agent cell."""
    return (
        (path / "meta" / "info.json").is_file()
        or (path / "metadata.json").is_file()
        or agent_cells.is_agent_cell(path)
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
    for text in _RECENT:
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
        for path in agent_cells.episode_files(batch):
            try:
                newest = max(newest, path.stat().st_mtime)
            except OSError:
                continue
        if newest > best_time:
            best_index, best_time = i, newest
    return best_index


def episode_labels(infos: list[dict]) -> list[str]:
    """Dropdown labels such as ``620003 ✓ 1599`` / ``620002 ✗ 1700 timeout``."""
    labels = []
    for i, info in enumerate(infos):
        seed = info["seed"]
        head = str(seed) if seed is not None else info["name"]
        mark = "·"
        if info["success"] is not None:
            mark = "✓" if info["success"] > 0 else "✗"
        parts = [head, mark]
        if info["length"] is not None:
            parts.append(str(info["length"]))
        end = str(info["termination"] or "")
        if end and end != "success":
            parts.append(end)
        label = " ".join(parts)
        # viser dropdowns key on the option string, so equal labels (an old
        # batch without seeds) must still differ.
        labels.append(f"{label}  [{i}]" if label in labels else label)
    return labels


def episode_list_html(infos: list[dict], current: int) -> str:
    """A scrollable list of the batch's episodes, the open one highlighted."""
    if not infos:
        return '<div style="font-size:11px;opacity:.6;">no episodes</div>'
    rows = []
    for i, info in enumerate(infos):
        ok = info["success"] is not None and info["success"] > 0
        colour = "#16a34a" if ok else "#dc2626"
        if info["success"] is None:
            colour = "inherit"
        mark = "✓" if ok else ("✗" if info["success"] is not None else "·")
        seed = info["seed"]
        head = str(seed) if seed is not None else info["name"]
        end = str(info["termination"] or "")
        tail = f"{info['length'] if info['length'] is not None else '—'}"
        if end and end != "success":
            tail += f" · {end}"
        background = "rgba(128,128,128,.18)" if i == current else "transparent"
        rows.append(
            f'<div style="display:flex;justify-content:space-between;gap:6px;'
            f"padding:2px 5px;border-radius:4px;background:{background};"
            'font-family:ui-monospace,monospace;font-size:10px;">'
            f'<span style="color:{colour};">{mark} {html.escape(head)}</span>'
            f'<span style="opacity:.6;">{html.escape(tail)}</span></div>'
        )
    return (
        '<div style="max-height:160px;overflow-y:auto;color:inherit;">'
        + "".join(rows)
        + "</div>"
    )


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
        count = len(agent_cells.episode_files(b))
        info = agent_cells.batch_version(b)
        if info is None:
            labels.append(f"{rel} · {count} eps")
            continue
        cell, kind, version = info
        label = f"{_batch_shown_name(rel, b, kind, version)} · {count} eps"
        rate = None
        if kind == "eval" and version is not None:
            rate = agent_cells.eval_success(cell, version)
        elif kind == "dev":
            rate = agent_cells.batch_success(b)
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


def load_metadata(demo_dir: Path) -> dict:
    """The batch's ``metadata.json``, at its root or one folder down."""
    for path in sorted(demo_dir.glob("metadata.json")) + sorted(
        demo_dir.glob("*/metadata.json")
    ):
        with path.open() as f:
            return json.load(f)
    raise FileNotFoundError(f"No metadata.json under {demo_dir}")


def build_env(metadata: dict):
    """A ``make()`` env built the way the batch's metadata says it was recorded."""
    return make(metadata["task"]["task_name"], EnvConfig.from_metadata(metadata))


class Store(Protocol):
    """The episodes a frontend plays: a demo batch or one dataset task.

    load() must run on the main thread: env.reset renders camera obs on the
    main thread's EGL context, and the main loop is the only mutator of
    (model, data).
    """

    files: list[Path]
    """One path per episode; the frontends show ``.name`` and use ``.stem``."""
    ep: int
    """The loaded episode, -1 before the first load."""
    info: dict
    """Per-episode scalars (success, fell, length, termination) of the loaded
    episode, as far as the source stores them."""
    qvel: np.ndarray | None
    """``(T, nv)`` full qvel of the loaded episode, when stored."""
    cmd: np.ndarray | None
    """``(T, 3)`` lower-body command of the loaded episode, when stored."""

    @property
    def qpos(self) -> np.ndarray:
        """``(T, nq)`` full qpos of the loaded episode."""
        ...

    @property
    def rewards(self) -> np.ndarray:
        """Per-frame reward of the loaded episode."""
        ...

    def load(self, idx: int) -> bool:
        """Load episode ``idx`` and reseed the env; False if already loaded."""
        ...

    def episode_labels(self) -> list[str]:
        """One episode-dropdown label per file."""
        ...

    def episode_infos(self) -> list[dict]:
        """Per-episode seed, outcome and length, for the episode list."""
        ...

    def after_forward(self, frame: int) -> None:
        """Run once a frame is posed and forwarded."""
        ...


class DemoStore:
    """Episode loading + reset-with-seed for stored-qpos demo batches."""

    def __init__(self, env, files: list[Path]):
        """Bind the store to an env and its list of episode .npz files."""
        self.env = env
        self.files = files
        self.ep = -1
        self._qpos: np.ndarray | None = None
        self._rewards: np.ndarray | None = None
        self.qvel: np.ndarray | None = None
        self.cmd: np.ndarray | None = None
        self.info: dict = {}

    @property
    def qpos(self) -> np.ndarray:
        """``(T, nq)`` full qpos of the loaded episode."""
        assert self._qpos is not None, "no episode loaded"
        return self._qpos

    @property
    def rewards(self) -> np.ndarray:
        """Per-frame reward of the loaded episode."""
        assert self._rewards is not None, "no episode loaded"
        return self._rewards

    def load(self, idx: int) -> bool:
        """Load episode ``idx`` and reseed the env; False if already loaded."""
        idx = int(np.clip(idx, 0, len(self.files) - 1))
        if idx == self.ep:
            return False
        ep = np.load(self.files[idx])
        if "seed" in ep.files:
            self.env.reset(seed=int(np.asarray(ep["seed"]).reshape(-1)[0]))
        self._qpos = np.asarray(ep["full_qpos"], dtype=np.float64)
        self._rewards = np.asarray(ep["reward"]).reshape(-1)
        # The velocity arrows need both; agent rollouts store no command.
        self.qvel = np.asarray(ep["full_qvel"]) if "full_qvel" in ep.files else None
        self.cmd = (
            np.asarray(ep["lowerbody_command"])
            if "lowerbody_command" in ep.files
            else None
        )
        self.info = {
            key: (
                str(np.asarray(ep[key]).reshape(-1)[0])
                if key == "termination"
                else float(np.asarray(ep[key]).reshape(-1)[0])
            )
            for key in ("success", "fell", "length", "termination")
            if key in ep.files
        }
        self.ep = idx
        return True

    def episode_labels(self) -> list[str]:
        """Seed + outcome when the batch stores them, else the file name.

        The shared timestamp prefix eats the narrow dropdown width, so a
        batch without seeds shows the distinctive tail of each name, which
        already carries ``<episode>_<steps>``.
        """
        infos = agent_cells.episode_infos(self.files[0].parent, self.files)
        if any(info["seed"] is not None for info in infos):
            return episode_labels(infos)
        names = [f.name for f in self.files]
        prefix = len(os.path.commonprefix(names)) if len(names) > 1 else 0
        return [name[prefix:] for name in names]

    def episode_infos(self) -> list[dict]:
        """Per-episode seed, outcome and length read from the batch."""
        return agent_cells.episode_infos(self.files[0].parent.resolve(), self.files)

    def after_forward(self, frame: int) -> None:
        """Nothing: a batch episode is fully posed by its stored qpos."""


class DatasetStore:
    """A :class:`Store` over one task of a dataset.

    The episodes come from a LeRobot task export (the published dataset or
    a local copy) through :class:`bigym.loco.agent.demo_video.TaskDemos`,
    which reads only the state columns, never a camera PNG. Each load resets
    the env on the episode's seed; ``after_forward`` runs the task's
    ``_on_step`` hook once a frame is posed.
    """

    COLUMNS = ("full_qpos", "full_qvel", "reward", "lowerbody_command")

    def __init__(self, env, inner, demos):
        """Bind the store to a replay env and a task's :class:`TaskDemos`."""
        self.env = env
        self.inner = inner
        self.demos = demos
        self.episodes = list(demos.episodes)
        # Stand-in paths: the frontends show .name and use .stem for
        # capture file names; nothing is read from them.
        self.files = [
            demos.dir / f"{e.index:03d}_{Path(e.source_file).stem}"
            for e in self.episodes
        ]
        self.labels = [
            f"{e.index:02d} · seed {e.seed} · {e.length}" for e in self.episodes
        ]
        self.infos = [
            {
                "name": path.name,
                "seed": e.seed,
                "success": None,
                "length": e.length,
                "termination": "",
                "fell": None,
            }
            for path, e in zip(self.files, self.episodes, strict=True)
        ]
        self.ep = -1
        self._qpos: np.ndarray | None = None
        self._rewards: np.ndarray | None = None
        self.qvel: np.ndarray | None = None
        self.cmd: np.ndarray | None = None
        self.info: dict = {}

    @property
    def qpos(self) -> np.ndarray:
        """``(T, nq)`` full qpos of the loaded episode."""
        assert self._qpos is not None, "no episode loaded"
        return self._qpos

    @property
    def rewards(self) -> np.ndarray:
        """Per-frame reward of the loaded episode."""
        assert self._rewards is not None, "no episode loaded"
        return self._rewards

    def load(self, idx: int) -> bool:
        """Load episode ``idx`` and reseed the env; False if already loaded."""
        idx = int(np.clip(idx, 0, len(self.episodes) - 1))
        if idx == self.ep:
            return False
        episode = self.episodes[idx]
        arrays = self.demos.columns(episode, self.COLUMNS)
        self.env.reset(seed=episode.seed)
        self._qpos = np.asarray(arrays["full_qpos"], dtype=np.float64)
        steps = self._qpos.shape[0]
        self._rewards = np.asarray(
            arrays.get("reward", np.zeros(steps)), dtype=float
        ).reshape(-1)
        self.qvel = arrays.get("full_qvel")
        self.cmd = arrays.get("lowerbody_command")
        self.info = {"length": float(steps)}
        self.ep = idx
        return True

    def episode_labels(self) -> list[str]:
        """``<index> · seed <seed> · <length>`` per episode."""
        return self.labels

    def episode_infos(self) -> list[dict]:
        """Seed and length per episode; the dataset stores no outcome."""
        return self.infos

    def after_forward(self, frame: int) -> None:
        """Run the task's step hook on the posed frame (target highlight)."""
        run_task_hook(self.inner)


class Playhead:
    """Frame advance with an end-of-episode hold."""

    def __init__(self):
        """Start at frame 0 with no end-of-episode hold pending."""
        self.frame = 0
        self.hold_until = 0.0
        self.last_advance = time.time()

    def reset(self):
        """Rewind to frame 0 and clear the end-of-episode hold."""
        self.frame = 0
        self.hold_until = 0.0

    def tick(self, n: int, playing: bool, fps: float) -> bool:
        """Advance one frame if due; returns True when the frame changed."""
        now = time.time()
        if not playing or now < self.hold_until or now - self.last_advance < 1.0 / fps:
            return False
        nxt = self.frame + 1
        if nxt >= n:
            if self.hold_until <= 0.0 or now < self.hold_until:
                self.hold_until = now + END_HOLD_SECONDS
                nxt = n - 1
            else:
                nxt = 0
                self.hold_until = 0.0
        changed = nxt != self.frame
        self.frame = nxt
        self.last_advance = now
        return changed


class CameraPanels:
    """MuJoCo-rendered image widgets for the viser sidebar.

    The onboard cameras come from the environment's own observation path
    (``BiGymEnv._get_visual_obs``), at the batch's recorded resolution, so
    the panels show the pixels a policy would have seen. One extra view is
    rendered from a third-person camera (a named one from the task XML if it
    has it, else a free camera orbiting the robot).

    Every render runs on the main loop: MuJoCo's GL context belongs to it.
    """

    def __init__(
        self,
        server,
        inner,
        model,
        data,
        camera_keys,
        camera_shape=(84, 84),
    ):
        """Build one image widget per onboard camera plus the third view."""
        self.inner = inner
        self.model = model
        self.data = data
        self.camera_keys = tuple(camera_keys)
        self.shape = (int(camera_shape[0]), int(camera_shape[1]))
        self.images: dict[str, Any] = {}
        self.renderer = None
        self.camera: int | mujoco.MjvCamera | None = None
        self.third_label = ""
        self.error = ""
        with server.gui.add_folder("Camera panels"):
            self.enabled = server.gui.add_checkbox("show", initial_value=True)
            blank = np.zeros((self.shape[0], self.shape[1], 3), dtype=np.uint8)
            for key in self.camera_keys:
                self.images[key] = server.gui.add_image(blank, label=key, format="jpeg")
            self._add_third_person(server)
            self.message = server.gui.add_html("")

    def _add_third_person(self, server) -> None:
        """Add the third-person widget, if an offscreen buffer allows one."""
        height = min(THIRD_PERSON_SIZE[0], int(self.model.vis.global_.offheight))
        width = min(THIRD_PERSON_SIZE[1], int(self.model.vis.global_.offwidth))
        name = ""
        cam_id = -1
        for candidate in THIRD_PERSON_CAMERAS:
            found = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, candidate)
            if found >= 0:
                name, cam_id = candidate, int(found)
                break
        self._third_size = (height, width)
        try:
            self.renderer = mujoco.Renderer(self.model, height, width)
        except Exception as exc:
            self.error = f"third-person renderer unavailable: {exc}"
            return
        if cam_id >= 0:
            self.camera = cam_id
            self.third_label = f"third ({name})"
        else:
            free = mujoco.MjvCamera()
            free.distance = 3.0
            free.elevation = -15.0
            free.azimuth = 160.0
            self.camera = free
            self.third_label = "third (free)"
        self.images[self.third_label] = server.gui.add_image(
            np.zeros((height, width, 3), dtype=np.uint8),
            label=self.third_label,
            format="jpeg",
        )

    def describe(self) -> str:
        """A one-line summary for the startup log."""
        onboard = ", ".join(self.camera_keys) or "none"
        return (
            f"onboard {onboard} @ {self.shape[1]}x{self.shape[0]}"
            f" + {self.third_label or 'no third-person view'}"
        )

    def update(self, lookat=None) -> None:
        """Re-render every panel for the scene currently in ``data``.

        Args:
            lookat: world point the free third-person camera orbits; ignored
                when the scene provides a named third-person camera.
        """
        if not bool(self.enabled.value):
            for handle in self.images.values():
                handle.visible = False
            return
        for handle in self.images.values():
            handle.visible = True
        # The real step() runs the task's _on_step() (which paints scene
        # state such as the reach-target highlight) before the observation
        # is rendered. Same order as the re-renderer.
        run_task_hook(self.inner)
        try:
            obs = self.inner._get_visual_obs()
        except Exception as exc:
            self.message.content = f"<i>camera render failed: {exc}</i>"
            return
        for key in self.camera_keys:
            image = obs.get(f"rgb_{key}")
            if image is not None:
                self.images[key].image = np.ascontiguousarray(
                    np.moveaxis(np.asarray(image), 0, -1)
                )
        if self.renderer is not None and self.third_label:
            camera = self.camera
            if lookat is not None and not isinstance(camera, int):
                assert camera is not None
                camera.lookat[:] = np.asarray(lookat, dtype=float)
            try:
                self.renderer.update_scene(self.data, camera=camera)
                self.images[self.third_label].image = self.renderer.render()
            except Exception as exc:
                self.message.content = f"<i>third-person render failed: {exc}</i>"
                return
        self.message.content = f"<i>{html.escape(self.error)}</i>" if self.error else ""

    def rebind(self, inner, model, data) -> None:
        """Point the existing widgets at another environment.

        Compare mode builds one environment per slot and lets a dropdown
        choose whose cameras the panels show. The widgets are unchanged
        (every slot is the same task, so the camera names and resolution
        match); only the offscreen renderer has to be rebuilt, because a
        ``mujoco.Renderer`` is bound to one model.

        Args:
            inner: The slot's native BiGym env.
            model: Its ``mujoco.MjModel``.
            data: Its ``mujoco.MjData``.
        """
        self.inner, self.model, self.data = inner, model, data
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if not self.third_label:
            return
        if not isinstance(self.camera, int):
            pass  # the free camera carries no model-bound state
        else:
            name = self.third_label[len("third (") : -1]
            found = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            self.camera = int(found) if found >= 0 else 0
        try:
            self.renderer = mujoco.Renderer(model, *self._third_size)
        except Exception as exc:
            self.error = f"third-person renderer unavailable: {exc}"

    def close(self) -> None:
        """Release the third-person renderer."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def sparkline_svg(
    rewards,
    frame: int,
    size: tuple[int, int] = SPARKLINE_SIZE,
    flags: str = "",
) -> str:
    """An inline-SVG strip of the episode's reward with a playhead marker.

    Args:
        rewards: The per-frame reward values.
        frame: The frame the playhead sits on.
        size: ``(width, height)`` of the strip in pixels.
        flags: Extra text (termination, fell) shown under the strip.

    Returns:
        The strip as HTML.
    """
    values = np.asarray(rewards, dtype=float).reshape(-1)
    width, height = int(size[0]), int(size[1])
    if values.size == 0:
        return '<div style="font-size:10px;opacity:.5;">no reward recorded</div>'
    lo, hi = float(values.min()), float(values.max())
    span = hi - lo
    n = values.size
    pad = 2.0

    def x_at(i: int) -> float:
        return pad + (width - 2 * pad) * (i / max(1, n - 1))

    def y_at(v: float) -> float:
        norm = 0.5 if span <= 0 else (v - lo) / span
        return pad + (height - 2 * pad) * (1.0 - norm)

    points = " ".join(f"{x_at(i):.1f},{y_at(v):.1f}" for i, v in enumerate(values))
    at = int(np.clip(frame, 0, n - 1))
    marker_x = x_at(at)
    here = float(values[at])
    svg = (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'style="display:block;">'
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="4" '
        'fill="rgba(128,128,128,.12)"/>'
        f'<polyline points="{points}" fill="none" stroke="#60a5fa" '
        'stroke-width="1.2"/>'
        f'<line x1="{marker_x:.1f}" y1="0" x2="{marker_x:.1f}" y2="{height}" '
        'stroke="#f97316" stroke-width="1"/>'
        f'<circle cx="{marker_x:.1f}" cy="{y_at(here):.1f}" r="2" fill="#f97316"/>'
        "</svg>"
    )
    caption = f"reward {here:.2f} (min {lo:.2f}, max {hi:.2f})"
    if flags:
        caption += f" · {flags}"
    return (
        '<div style="color:inherit;">'
        f'{svg}<div style="font-size:9px;opacity:.5;margin-top:2px;">'
        f"{html.escape(caption)}</div></div>"
    )


def _note_html(text: str, error: bool = False) -> str:
    """A small status line for a sidebar panel."""
    colour = "color:#dc2626;" if error else "opacity:.6;"
    return (
        f'<div style="font-size:10px;{colour}white-space:pre-wrap;">'
        f"{html.escape(text)}</div>"
    )


def directory_panel(server, current: str, request) -> None:
    """The "Demo directory" panel: browse and open another directory live.

    The browser lists folders on the machine running the viewer: a "quick"
    dropdown of starting points, the path (editable), "up", and a dropdown
    of the subfolders, ``●`` marking those that hold demos (dataset task
    folders, demo batches, agent cells, or folders directly above them).
    Load scans the path on a worker thread (the same discovery as
    ``--demo-dir``; empty is the published dataset) and hands what it found
    to ``request``; a directory with nothing to show leaves the scene alone
    and says so in the panel.

    Args:
        server: The viser server.
        current: What the path starts with.
        request: Called with the :class:`DatasetSource` or
            :class:`BatchSource` to open.
    """
    jump, open_sub = "jump to …", "open a subfolder …"
    hint = "empty: the published dataset · several: separate with spaces"
    places = quick_places()
    listed: dict[str, Path] = {}
    shown: dict[str, str | None] = {"text": None}
    with server.gui.add_folder("Demo directory"):
        quick = server.gui.add_dropdown("quick", options=[jump, *places])
        path = server.gui.add_text("path", initial_value=current)
        up = server.gui.add_button("up")
        folders = server.gui.add_dropdown("folders", options=[open_sub])
        load = server.gui.add_button("load")
        note = server.gui.add_html("")

    def refresh() -> None:
        text = str(path.value)
        if text == shown["text"]:
            return
        shown["text"] = text
        listed.clear()
        base = browse_dir(text)
        error = False
        if base is None:
            try:
                several = len(shlex.split(text)) > 1
            except ValueError:
                several = False
            status = "several folders: load opens them together"
            if not several:
                status, error = f"{text.strip()} is not a folder", True
        else:
            try:
                names, total = subfolders(base, BROWSE_LIMIT)
            except OSError as exc:
                names, total = [], 0
                status, error = f"cannot list {base}: {exc.strerror or exc}", True
            else:
                for name in names:
                    child = base / name
                    listed[(MARK if has_demos(child) else "") + name] = child
                status = f"{base}: {total} subfolder" + ("" if total == 1 else "s")
                if total > len(names):
                    status += f" (first {len(names)} listed)"
                if demo_marker(base):
                    status += " · this folder holds demos"
                status += " · ● holds demos"
        folders.options = [open_sub, *listed]
        folders.value = open_sub
        note.content = _note_html(f"{status}\n{hint}", error)

    def go(target: Path | str) -> None:
        path.value = target if isinstance(target, str) else shlex.quote(str(target))
        refresh()

    def on_quick(_) -> None:
        if quick.value in places:
            target = places[quick.value]
            quick.value = jump
            go(target)

    def on_up(_) -> None:
        go((browse_dir(str(path.value)) or Path.cwd()).parent)

    def on_folder(_) -> None:
        child = listed.get(str(folders.value))
        if child is not None:
            go(child)

    def scan(text: str) -> None:
        try:
            source = find_source(text)
        except Exception as exc:
            note.content = _note_html(str(exc).splitlines()[0], error=True)
            return
        if isinstance(source, DatasetSource) and source.dirs[0] is None:
            note.content = _note_html(f"fetching {source.labels[0]} …")
            try:
                hub.task_dir(source.tasks[0])
            except Exception as exc:
                note.content = _note_html(str(exc).splitlines()[0], error=True)
                return
        if text.strip():
            if text.strip() in _RECENT:
                _RECENT.remove(text.strip())
            _RECENT.insert(0, text.strip())
        note.content = _note_html(f"opening {source.where} …")
        request(source)

    def on_load(_) -> None:
        text = str(path.value)
        shown["text"] = None  # the next edit or navigation lists again
        note.content = _note_html(
            f"looking for demos in {text} …" if text.strip() else "listing tasks …"
        )
        threading.Thread(target=scan, args=(text,), daemon=True).start()

    quick.on_update(on_quick)
    path.on_update(lambda _: refresh())
    up.on_click(on_up)
    folders.on_update(on_folder)
    load.on_click(on_load)
    refresh()


def _task_picker(source: DatasetSource, idx: int):
    """Build the "Task" dropdown for a dataset (a :class:`Picker`'s ``build``).

    Picking a task the Hub cache does not hold yet downloads its folder on a
    worker thread, with a status line, while the open task keeps playing;
    the switch is requested once the files are there.

    Args:
        source: The dataset's tasks.
        idx: The task on screen.

    Returns:
        ``build(server, request_switch)``.
    """

    def build(server, request_switch) -> None:
        with server.gui.add_folder("Task"):
            dropdown = server.gui.add_dropdown(
                "task", options=list(source.labels), initial_value=source.labels[idx]
            )
            note = server.gui.add_html(_note_html(source.where))
        wanted = {"index": idx}

        def fetch(target: int) -> None:
            if source.dirs[target] is None:
                try:
                    hub.task_dir(source.tasks[target])
                except Exception as exc:
                    note.content = _note_html(str(exc).splitlines()[0], error=True)
                    return
            if wanted["index"] == target:  # not overtaken by a later pick
                note.content = _note_html(f"building {source.labels[target]} …")
                request_switch(target)

        def on_pick(_) -> None:
            target = source.labels.index(str(dropdown.value))
            wanted["index"] = target
            if target == idx:
                note.content = _note_html(source.where)
                return
            label = source.labels[target]
            note.content = _note_html(
                f"fetching {label} …"
                if source.dirs[target] is None
                else f"building {label} …"
            )
            threading.Thread(target=fetch, args=(target,), daemon=True).start()

        dropdown.on_update(on_pick)

    return build


def run_mujoco(
    store: Store,
    inner,
    model,
    data,
    realtime_fps: float = DEMO_HZ,
    batch_labels: list[str] | None = None,
    batch_idx: int = 0,
    item: str = "batch",
) -> int | None:
    """Play the store's episodes in a native MuJoCo window until it closes.

    Returns the index of another batch (or dataset task, ``item``) to open
    (``[`` / ``]``), or None when the window was closed.
    """
    import mujoco.viewer

    ctl: dict = {"playing": True, "dirty": True, "want": None, "batch": None}
    head = Playhead()
    labels = list(batch_labels or [])

    def on_key(keycode: int) -> None:
        # Runs on the viewer thread: only request; loads happen on the main loop.
        if keycode == 32:  # SPACE
            ctl["playing"] = not ctl["playing"]
        elif keycode == 262:  # RIGHT
            head.frame = min(head.frame + 1, store.qpos.shape[0] - 1)
            ctl["dirty"] = True
        elif keycode == 263:  # LEFT
            head.frame = max(head.frame - 1, 0)
            ctl["dirty"] = True
        elif keycode == 265:  # UP
            ctl["want"] = store.ep + 1
        elif keycode == 264:  # DOWN
            ctl["want"] = store.ep - 1
        elif keycode in (91, 93) and len(labels) > 1:  # [ and ]
            step = 1 if keycode == 93 else -1
            ctl["batch"] = (batch_idx + step) % len(labels)

    keys = "[viewer] SPACE play/pause · LEFT/RIGHT frame · UP/DOWN episode"
    print(keys + (f" · [ / ] {item}" if len(labels) > 1 else ""))
    if labels:
        print(f"[viewer] {item} [{batch_idx}] {labels[batch_idx]}")
    print(f"[viewer] episode {store.ep}: {store.files[store.ep].name}")
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running():
            nxt, ctl["batch"] = ctl["batch"], None
            if nxt is not None and nxt != batch_idx:
                print(f"[viewer] switching to {item} [{nxt}] {labels[nxt]}")
                return nxt
            want, ctl["want"] = ctl["want"], None
            if want is not None and store.load(want):
                print(f"[viewer] episode {store.ep}: {store.files[store.ep].name}")
                head.reset()
                ctl["dirty"] = True
            if head.tick(store.qpos.shape[0], ctl["playing"], realtime_fps):
                ctl["dirty"] = True
            if ctl["dirty"]:
                data.qpos[: store.qpos.shape[1]] = store.qpos[head.frame]
                mujoco.mj_forward(model, data)
                # kinematic playback skips task logic; run the success
                # check so task visuals (target highlight) update live
                inner._success()
                store.after_forward(head.frame)
                viewer.sync()
                ctl["dirty"] = False
            time.sleep(0.004)
    return None


@dataclass
class Picker:
    """What the browser viewer can switch to; the caller rebuilds the scene.

    Attributes:
        labels: One label per batch (or dataset task). A rescan updates the
            list in place, so an index the viewer returns stays valid.
        index: The one on screen.
        item: What a label names, ``batch`` or ``task``, for the log.
        paths: The batch directories behind the labels, updated in place with
            them; None for dataset tasks.
        rescan: Re-discovers ``paths`` and ``labels`` (a finished job wrote a
            new batch); None when nothing can change.
        build: Builds a custom picker instead of the flat "Demo batch"
            dropdown, as ``build(server, request_switch)``; request_switch
            takes the index to open.
    """

    labels: list[str]
    index: int = 0
    item: str = "batch"
    paths: list[Path] | None = None
    rescan: Callable[[], object] | None = None
    build: Callable[[Any, Callable[[int], None]], None] | None = None


def rotation_z_to(direction: np.ndarray) -> vtf.SO3:
    """The rotation taking +z onto the unit vector ``direction``."""
    c = float(direction[2])
    if c > 0.9999:
        return vtf.SO3.identity()
    if c < -0.9999:
        return vtf.SO3.from_x_radians(np.pi)
    axis = np.cross([0.0, 0.0, 1.0], direction)
    axis /= np.linalg.norm(axis)
    return vtf.SO3.exp(axis * np.arccos(c))


def rgb255(colour) -> tuple[int, ...]:
    """A MuJoCo rgba (0-1 floats) as viser's 0-255 integers."""
    values = np.asarray(colour, dtype=float).reshape(-1)[:3]
    return tuple(int(round(255 * v)) for v in values)


class VelocityArrows:
    """Commanded and measured base velocity, drawn above the pelvis.

    Four arrows from the anchor A = pelvis + 0.575 m up: commanded and
    measured planar velocity (horizontal, 0.5 m per m/s) and commanded and
    measured yaw rate (vertical, 0.5 m per rad/s). Each is a shaft (80 % of
    the length, radius w) under a cone head (20 %, base radius 2w).
    """

    WIDTH = 0.015
    # commanded linear, measured linear, commanded yaw, measured yaw
    COLOURS = np.array(
        [(51, 51, 153), (0, 153, 255), (51, 153, 51), (0, 255, 102)], dtype=np.uint8
    )

    def __init__(self, server, model) -> None:
        """Find the floating-base joints and add the two batched meshes."""
        self.base_adr: dict[str, tuple[int, int]] = {}
        for j in range(model.njnt):
            name = model.jnt(j).name
            for suffix in ("pelvis_x", "pelvis_y", "pelvis_z", "pelvis_rz"):
                if name.endswith(suffix):
                    self.base_adr[suffix] = (
                        int(model.jnt_qposadr[j]),
                        int(model.jnt_dofadr[j]),
                    )
        self.available = {"pelvis_x", "pelvis_y"} <= self.base_adr.keys()
        shaft = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=16)
        shaft.apply_translation([0.0, 0.0, 0.5])  # spans z in [0, 1]
        head = trimesh.creation.cone(radius=2.0, height=1.0, sections=16)
        count = len(self.COLOURS)
        self.identity = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (count, 1))
        self.shafts = server.scene.add_batched_meshes_simple(
            "/velocity/shafts",
            shaft.vertices,
            shaft.faces,
            batched_wxyzs=self.identity,
            batched_positions=np.zeros((count, 3)),
            batched_scales=np.ones((count, 3)),
            batched_colors=self.COLOURS,
        )
        self.heads = server.scene.add_batched_meshes_simple(
            "/velocity/heads",
            head.vertices,
            head.faces,
            batched_wxyzs=self.identity,
            batched_positions=np.zeros((count, 3)),
            batched_scales=np.ones((count, 3)),
            batched_colors=self.COLOURS,
        )

    def pelvis_lookat(self, qpos: np.ndarray) -> tuple[float, float, float] | None:
        """A point above the pelvis for the free camera, or None without one."""
        if not self.available:
            return None
        adr = self.base_adr
        return (qpos[adr["pelvis_x"][0]], qpos[adr["pelvis_y"][0]], 0.9)

    def update(self, store: Store, frame: int, visible: bool) -> None:
        """Pose the arrows for ``frame``; hidden when asked or nothing is stored."""
        shown = (
            visible
            and self.available
            and store.cmd is not None
            and store.qvel is not None
        )
        self.shafts.visible = shown
        self.heads.visible = shown
        if not shown:
            return
        assert store.cmd is not None and store.qvel is not None
        adr = self.base_adr
        q = store.qpos[frame]
        pelvis_z = q[adr["pelvis_z"][0]] if "pelvis_z" in adr else 0.8
        yaw = q[adr["pelvis_rz"][0]] if "pelvis_rz" in adr else 0.0
        anchor = np.array(
            [q[adr["pelvis_x"][0]], q[adr["pelvis_y"][0]], pelvis_z + 0.575]
        )
        cmd = store.cmd[min(frame, len(store.cmd) - 1)]
        qv = store.qvel[min(frame, len(store.qvel) - 1)]
        c, s = np.cos(yaw), np.sin(yaw)
        yaw_rate = qv[adr["pelvis_rz"][1]] if "pelvis_rz" in adr else 0.0
        vectors = [
            0.5 * np.array([c * cmd[0] - s * cmd[1], s * cmd[0] + c * cmd[1], 0.0]),
            0.5 * np.array([qv[adr["pelvis_x"][1]], qv[adr["pelvis_y"][1]], 0.0]),
            0.5 * np.array([0.0, 0.0, cmd[2]]),
            0.5 * np.array([0.0, 0.0, yaw_rate]),
        ]
        count = len(self.COLOURS)
        positions = np.zeros((count, 3))
        wxyz = self.identity.copy()
        shaft_scales = np.full((count, 3), 1e-4)
        head_positions = np.tile(anchor, (count, 1))
        head_scales = np.full((count, 3), 1e-4)
        width = self.WIDTH
        for k, vector in enumerate(vectors):
            length = float(np.linalg.norm(vector))
            positions[k] = anchor
            if length < 0.01:  # collapse near-zero arrows instead of jittering
                continue
            direction = vector / length
            wxyz[k] = rotation_z_to(direction).wxyz
            shaft_scales[k] = (width, width, 0.8 * length)
            head_positions[k] = anchor + direction * 0.8 * length
            head_scales[k] = (width, width, 0.2 * length)
        self.shafts.batched_positions = positions
        self.shafts.batched_wxyzs = wxyz
        self.shafts.batched_scales = shaft_scales
        self.heads.batched_positions = head_positions
        self.heads.batched_wxyzs = wxyz
        self.heads.batched_scales = head_scales


class TargetSpheres:
    """Reach-target balls drawn by the viewer, with a glow when reached.

    They stand in for the model's own target geoms, which are hidden from
    mjviser: its meshes are baked once at scene build, so a reset moving a
    target would leave a stale ball behind.
    """

    def __init__(self, server, targets: list[Target]) -> None:
        """Add a ball and a hidden glow per target."""
        self.targets = targets
        self.balls, self.glows = [], []
        for i, target in enumerate(targets):
            radius = float(np.asarray(target._config.size).reshape(-1)[0])
            self.balls.append(
                server.scene.add_icosphere(
                    f"/targets/{i}/ball",
                    radius=radius,
                    color=rgb255(target._config.color_default),
                )
            )
            self.glows.append(
                server.scene.add_icosphere(
                    f"/targets/{i}/glow",
                    radius=radius * 1.35,
                    color=rgb255(target._config.color_highlight),
                    opacity=0.5,
                    visible=False,
                )
            )

    def measure(self, inner) -> list[tuple[bool, np.ndarray]]:
        """Each target's reached flag and position in the posed frame."""
        reached = [target.is_reached(inner.reach_tolerance) for target in self.targets]
        positions = [target.get_position() for target in self.targets]
        return list(zip(reached, positions, strict=True))

    def show(self, measured: list[tuple[bool, np.ndarray]]) -> None:
        """Move the balls and light up the reached ones."""
        for ball, glow, (hit, position) in zip(
            self.balls, self.glows, measured, strict=True
        ):
            ball.position = tuple(position)
            glow.position = tuple(position)
            glow.visible = bool(hit)


class EpisodePanel:
    """The "Episode" folder: info card, progress, slider, list and picker."""

    def __init__(self, server, labels: list[str], count: int) -> None:
        """Add the folder's widgets."""
        with server.gui.add_folder("Episode"):
            self.card = server.gui.add_html("")
            self.progress = server.gui.add_progress_bar(0.0, color=(96, 165, 250))
            self.slider = server.gui.add_slider(
                "index", min=0, max=count - 1, step=1, initial_value=0
            )
            # The list is read-only (viser HTML cannot call back); the dropdown
            # next to it is what picks an episode, with the same labels.
            self.list = server.gui.add_html("")
            self.dropdown = server.gui.add_dropdown(
                "episode", options=labels, initial_value=labels[0]
            )
            self.prev = server.gui.add_button("prev")
            self.next = server.gui.add_button("next")


def episode_card_html(store: Store, frame: int, fps: float) -> str:
    """The episode card: file name, length and the reward at ``frame``.

    Theme-neutral, no outer frame (the folder already draws one): it
    inherits the sidebar colours, mutes text by opacity and uses translucent
    surfaces. Episodes count from 0 like the file names, and the total is a
    count so "0 of 50" cannot be misread as 49 episodes.
    """
    n = store.qpos.shape[0]
    reward = float(store.rewards[min(frame, len(store.rewards) - 1)])
    reward_style = "color:#16a34a;" if reward >= 1.0 else "opacity:.55;"
    name = store.files[store.ep].name
    return (
        '<div style="padding:1px 2px;color:inherit;">'
        f'<div title="{name}" style="font-family:ui-monospace,'
        "monospace;font-size:11px;"
        "opacity:.75;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
        f'margin-bottom:4px;">{name}</div>'
        '<div style="font-size:10px;opacity:.5;margin-bottom:8px;">'
        f"{len(store.files)} episodes · this one: {n} steps "
        f"({n / fps:.1f}s)</div>"
        '<div style="display:flex;justify-content:space-between;align-items:center;'
        'background:rgba(128,128,128,.12);border-radius:6px;padding:6px 9px;">'
        f'<span style="font-size:10px;opacity:.6;">REWARD @ FRAME {frame}</span>'
        '<span style="font-family:ui-monospace,monospace;'
        "font-size:16px;font-weight:750;"
        f'{reward_style}">{reward:.0f}</span></div>'
        '<div style="font-size:9px;opacity:.45;margin-top:6px;">'
        f"fps {fps:.0f} = realtime · frames span "
        f"{1000.0 / fps:.0f} ms sim each</div></div>"
    )


def episode_flags(info: dict) -> str:
    """Outcome, termination and fell flags of an episode, for the reward strip."""
    parts = []
    if info.get("success") is not None:
        parts.append("success" if float(info["success"]) > 0 else "fail")
    end = str(info.get("termination") or "")
    if end:
        parts.append(end)
    if info.get("fell"):
        parts.append("fell")
    return " · ".join(parts)


class PlaybackPanel:
    """The "Playback" folder: frame slider, reward strip, play and fps."""

    def __init__(self, server, realtime_fps: float) -> None:
        """Add the folder's widgets, playing at the recorded rate."""
        with server.gui.add_folder("Playback"):
            self.frame = server.gui.add_slider(
                "frame", min=0, max=1, step=1, initial_value=0
            )
            self.reward_strip = server.gui.add_html("")
            self.playing = server.gui.add_checkbox("play", initial_value=True)
            self.fps = server.gui.add_slider(
                "fps",
                min=1,
                max=max(100, int(round(realtime_fps))),
                step=1,
                initial_value=int(round(realtime_fps)),
            )


class LightingPanel:
    """The "Lighting" folder: sky, environment map, default and key lights.

    mjviser carries geometry but not MuJoCo's lights, so viser lights the
    scene with a flat environment map: every face gets the same intensity
    and the model -- 298 of 339 geoms are plain grey rgba, 0.5 or 0.7 --
    reads as one white blob. The fix is to take intensity away, not to add
    sources. Sliders, because the right value is a judgement made looking at
    the render.

    Sky: viser draws no sky, so the page colour shows through and captures
    carry alpha there. "grey" shows the environment map itself, blurred and
    dimmed, as a neutral backdrop that both the white fixtures and the
    darkened robot read against; "black" is the same map at zero intensity,
    for slides and video; "transparent" keeps the cut-out for compositing.
    All of it is baked into captures.

    The panel is rebuilt with the scene on every batch switch, so each
    control is seeded from ``prefs`` (process-wide) and writes back on
    change: pick a sky once, keep it for the next batches.
    """

    def __init__(self, server, prefs: dict) -> None:
        """Add the key light and the folder, and apply the preferences."""
        self.server = server
        self.prefs = prefs
        # Aimed like the MJCF main_light (pos 0 0 3, dir 0 0 -1), off by
        # default: turn it up only if the flatter render needs shape.
        self.key = server.scene.add_light_directional(
            "/lighting/key", intensity=0.0, cast_shadow=True, visible=False
        )
        with server.gui.add_folder("Lighting"):
            self.sky = server.gui.add_dropdown(
                "sky", options=SKY_OPTIONS, initial_value=prefs["sky"]
            )
            self.environment = server.gui.add_slider(
                "environment",
                min=0.0,
                max=1.5,
                step=0.05,
                initial_value=prefs["environment"],
            )
            self.default_lights = server.gui.add_checkbox(
                "default lights", initial_value=prefs["default_lights"]
            )
            self.shadows = server.gui.add_checkbox(
                "shadows", initial_value=prefs["shadows"]
            )
            self.key_light = server.gui.add_slider(
                "key light",
                min=0.0,
                max=3.0,
                step=0.05,
                initial_value=prefs["key_light"],
            )
            self.message = server.gui.add_html("")
        for control in (
            self.sky,
            self.environment,
            self.default_lights,
            self.shadows,
            self.key_light,
        ):
            control.on_update(self.apply)
        # An untouched control never fires on_update, so a fresh scene needs
        # one explicit pass for the preferences to hold.
        self.apply()

    def apply(self, _=None) -> None:
        """Push the controls to the scene and remember them."""
        try:
            sky = str(self.sky.value)
            # Every sky is the environment cube map shown as background;
            # black is that map at zero intensity. Not set_background_image:
            # its quad is rescaled per frame from the camera's film size, and
            # a capture whose aspect differs from the browser canvas can go
            # out before the quad catches up, leaving transparent strips at
            # the sides.
            self.server.scene.configure_environment_map(
                environment_intensity=float(self.environment.value),
                background=sky != "transparent",
                background_blurriness=1.0,
                background_intensity=0.5 if sky == "grey" else 0.0,
            )
            self.server.scene.configure_default_lights(
                enabled=bool(self.default_lights.value),
                cast_shadow=bool(self.shadows.value),
            )
            self.key.intensity = float(self.key_light.value)
            self.key.visible = float(self.key_light.value) > 0.0
        except Exception as exc:
            self.message.content = f"<i>{exc}</i>"
            return
        self.message.content = ""
        self.prefs.update(
            sky=sky,
            environment=float(self.environment.value),
            default_lights=bool(self.default_lights.value),
            shadows=bool(self.shadows.value),
            key_light=float(self.key_light.value),
        )


class FigureCapture:
    """The "Figure capture" folder: the current view as a PNG at print size.

    Frame a pose by dragging, then grab it. viser renders in the browser, so
    the capture is capped by the window: open it wide. Only the velocity
    arrows (a debug overlay) are hidden for the shot; the target balls and
    their glow stay, as MuJoCo shows the reached state too.
    """

    def __init__(self, server, overlays, figure_dir: str, task_name: str) -> None:
        """Add the folder; ``overlays`` are the scene handles hidden in a shot."""
        self.server = server
        self.overlays = overlays
        self.figure_dir = figure_dir
        self.task_name = task_name
        with server.gui.add_folder("Figure capture"):
            self.width = server.gui.add_number(
                "width", initial_value=1000, min=64, step=8
            )
            self.height = server.gui.add_number(
                "height", initial_value=1000, min=64, step=8
            )
            self.button = server.gui.add_button("capture png")
            self.message = server.gui.add_html("")

    def capture(self, stem: str, frame: int) -> None:
        """Write the first client's view to ``<figure_dir>/<task>_<stem>_f<frame>.png``."""
        clients = list(self.server.get_clients().values())
        if not clients:
            self.message.content = "<i>no browser connected</i>"
            return
        was = [handle.visible for handle in self.overlays]
        try:
            for handle in self.overlays:
                handle.visible = False
            image = clients[0].get_render(
                height=int(self.height.value),
                width=int(self.width.value),
                transport_format="png",
            )
        except Exception as exc:
            self.message.content = f"<i>capture failed: {exc}</i>"
            return
        finally:
            for handle, visible in zip(self.overlays, was, strict=True):
                handle.visible = visible
        out = Path(self.figure_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Batch stems are collection timestamps, so a directory of captures
        # from several tasks is unreadable without the task in the name.
        prefix = f"{path_name(self.task_name)}_" if self.task_name else ""
        path = out / f"{prefix}{stem}_f{frame:05d}.png"
        imageio.imwrite(path, image)
        print(f"[viewer] captured {path}", flush=True)
        self.message.content = f"saved <code>{path.name}</code>"


class BatchDropdown:
    """The "Demo batch" folder: a flat picker, plus the --follow controls."""

    def __init__(self, server, picker: Picker, request, follow: bool) -> None:
        """Add the dropdown; ``request`` is called with the picked index."""
        self.picker = picker
        self.auto_jump = None
        self.follow_note = None
        with server.gui.add_folder("Demo batch"):
            self.dropdown = server.gui.add_dropdown(
                "batch",
                options=list(picker.labels),
                initial_value=picker.labels[picker.index],
            )
            self.dropdown.on_update(
                lambda _: request(picker.labels.index(self.dropdown.value))
            )
            if follow:
                self.auto_jump = server.gui.add_checkbox(
                    "auto-jump to newest", initial_value=False
                )
                self.follow_note = server.gui.add_html("")

    def point_at(self, open_batch: Path) -> None:
        """Re-list the batches and select the one on screen."""
        picker = self.picker
        here = picker.index
        if picker.paths is not None:
            resolved = [b.resolve() for b in picker.paths]
            if open_batch in resolved:
                here = resolved.index(open_batch)
        self.dropdown.options = list(picker.labels)
        self.dropdown.value = picker.labels[min(here, len(picker.labels) - 1)]


class ViserViewer:
    """The browser viewer for one batch or dataset task.

    The scene is built in ``__init__`` and served by :meth:`run`. GUI
    callbacks run on viser's thread pool and only record a request; loads,
    renders and switches happen on the main loop (see the module doc).

    With ``cell`` (the agent cell the batch belongs to) the flat batch
    dropdown gives way to the agent panels of
    :class:`~bigym.vr.viewer.agent_cells.CellPanels`, whose Task and Session
    dropdowns offer ``cells``. With --follow (batches only) the main loop
    rescans every --follow-seconds: new batches, policy versions, a grown
    run.json or transcript, new episodes in the open batch. The batch on
    screen only changes when "auto-jump to newest" is ticked.
    """

    def __init__(
        self,
        store: Store,
        inner,
        picker: Picker,
        args: ViewConfig,
        directory: str,
        *,
        realtime_fps: float = DEMO_HZ,
        task_name: str = "",
        camera_keys: tuple[str, ...] = (),
        camera_shape: tuple[int, int] = (84, 84),
        cell: Path | None = None,
        cells: list | None = None,
    ) -> None:
        """Open the server and build the scene and every panel.

        Args:
            store: The episodes to play, one already loaded.
            inner: The native env the store poses.
            picker: What can be switched to.
            args: The command line.
            directory: What the "Demo directory" field starts with (empty
                for the published dataset).
            realtime_fps: The rate the episodes were recorded at.
            task_name: Prefixes capture file names.
            camera_keys: The onboard cameras the camera panels show.
            camera_shape: Their recorded ``(height, width)``.
            cell: The agent cell the batch belongs to, if any.
            cells: Every cell found, for the Task and Session dropdowns.
        """
        self.store = store
        self.inner = inner
        self.model, self.data = inner.model, inner.data
        self.picker = picker
        self.port = args.port
        self.realtime_fps = realtime_fps
        self.exit_after = args.exit_after_seconds
        self.follow_seconds = (
            args.follow_seconds if args.follow and picker.paths is not None else 0.0
        )
        self.open_batch = store.files[0].parent.resolve()
        self.head = Playhead()
        self.labels = list(store.episode_labels())
        self.dirty = True
        self.wanted_episode: int | None = None
        self.wanted_batch: int | None = None
        self.wanted_compare: agent_cells.CompareRequest | None = None
        self.wanted_source: DatasetSource | BatchSource | None = None
        self.card_at = 0.0
        self.card_reward: float | None = None
        self.cameras_at = 0.0
        self.cameras_logged = False
        self.follow_at = time.time()
        self.follow_scans = 0

        targets = hide_reach_targets(inner)
        self.server, self.scene = open_viser_scene(
            self.model, inner, port=self.port, label="demo-viewer"
        )
        self.arrows = VelocityArrows(self.server, self.model)
        self.targets = TargetSpheres(self.server, targets)
        directory_panel(self.server, directory, self._request_source)
        self.cell_panels = self._add_cell_panels(args, cell, cells or [])
        self.batch_dropdown = self._add_batch_picker(cell)
        self.episode = EpisodePanel(self.server, self.labels, len(store.files))
        self.playback = PlaybackPanel(self.server, realtime_fps)
        with self.server.gui.add_folder("View"):
            self.show_arrows = self.server.gui.add_checkbox(
                "velocity arrows", initial_value=True
            )
            self.show_arrows.on_update(self._mark_dirty)
        # MuJoCo's own pixels next to the viser scene: the onboard cameras the
        # env renders for observations, plus a third-person view.
        self.cameras = CameraPanels(
            self.server, inner, self.model, self.data, camera_keys, camera_shape
        )
        self.cameras.enabled.on_update(self._mark_dirty)
        print(f"[viewer] camera panels: {self.cameras.describe()}", flush=True)
        LightingPanel(self.server, _LIGHTING_PREFS)
        self.figure = FigureCapture(
            self.server,
            (self.arrows.shafts, self.arrows.heads),
            args.figure_dir,
            task_name,
        )
        self.figure.button.on_click(self._capture)
        self._wire_episode_controls()
        self.server.on_client_connect(self._place_camera)

    def _add_cell_panels(
        self, args: ViewConfig, cell: Path | None, cells: list
    ) -> agent_cells.CellPanels | None:
        """The agent panels (Session, Rollouts, Policy, Transcript, Compare)."""
        if cell is None:
            return None
        infos = agent_cells.episode_infos(self.open_batch, self.store.files)
        seeds = [i["seed"] for i in infos if i["seed"]]
        return agent_cells.CellPanels(
            self.server,
            cell=cell,
            cells=list(cells),
            batches=self.picker.paths if self.picker.paths is not None else [],
            open_batch=self.open_batch,
            request_switch=self._request_batch,
            request_compare=self._request_compare,
            follow=self.follow_seconds > 0.0,
            seed_hint=seeds[0] if seeds else agent_cells.EVAL_SEED_LO,
            compare_gap=args.compare_gap,
            compare_layout=args.layout,
        )

    def _add_batch_picker(self, cell: Path | None) -> BatchDropdown | None:
        """The caller's own picker, or the flat dropdown for several batches."""
        if self.picker.build is not None:
            self.picker.build(self.server, self._request_batch)
            return None
        if cell is None and len(self.picker.labels) > 1:
            return BatchDropdown(
                self.server,
                self.picker,
                self._request_batch,
                follow=self.follow_seconds > 0.0,
            )
        return None

    def _wire_episode_controls(self) -> None:
        """Episode and frame controls: requests only, served by the main loop."""
        episode, playback = self.episode, self.playback
        episode.slider.on_update(
            lambda _: self._request_episode(int(episode.slider.value))
        )
        episode.dropdown.on_update(
            lambda _: self._request_episode(self.labels.index(episode.dropdown.value))
        )
        episode.prev.on_click(lambda _: self._request_episode(self.store.ep - 1))
        episode.next.on_click(lambda _: self._request_episode(self.store.ep + 1))
        playback.frame.on_update(self._seek)

    def _mark_dirty(self, _=None) -> None:
        self.dirty = True

    def _request_episode(self, index: int) -> None:
        self.wanted_episode = index

    def _request_batch(self, index: int) -> None:
        self.wanted_batch = index

    def _request_compare(self, request: agent_cells.CompareRequest) -> None:
        self.wanted_compare = request

    def _request_source(self, source: DatasetSource | BatchSource) -> None:
        self.wanted_source = source

    def _seek(self, _=None) -> None:
        self.head.frame = int(self.playback.frame.value)
        self.dirty = True

    def _capture(self, _=None) -> None:
        self.figure.capture(self.store.files[self.store.ep].stem, self.head.frame)

    def _place_camera(self, client: viser.ClientHandle) -> None:
        client.camera.position = (-1.8, 1.2, 1.4)
        client.camera.look_at = (0.2, 0.0, 0.7)

    def _sync_ui(self) -> None:
        """Point every episode control at the loaded episode and frame."""
        store, head = self.store, self.head
        n = store.qpos.shape[0]
        self.playback.frame.max = max(1, n - 1)
        self.playback.frame.value = min(head.frame, n - 1)
        self.episode.slider.value = store.ep
        self.episode.dropdown.value = self.labels[store.ep]
        self.episode.list.content = episode_list_html(store.episode_infos(), store.ep)
        self.episode.card.content = episode_card_html(
            store, head.frame, self.realtime_fps
        )
        self.playback.reward_strip.content = sparkline_svg(
            store.rewards, head.frame, flags=episode_flags(store.info)
        )
        self.card_at = time.time()

    def _refresh_batches(self) -> None:
        """Re-discover batches and re-point the picker at the open one."""
        if self.picker.rescan is None:
            return
        if self.cell_panels is not None:
            self.picker.rescan()
            self.cell_panels.refresh_rollouts()
            return
        if self.batch_dropdown is None:
            return
        self.picker.rescan()
        self.batch_dropdown.point_at(self.open_batch)

    def _is_other_batch(self, index: int) -> bool:
        """True when the picked batch is not the one on screen."""
        paths = self.picker.paths
        if paths is not None and 0 <= index < len(paths):
            return paths[index].resolve() != self.open_batch
        return index != self.picker.index

    def _refresh_open_batch(self) -> bool:
        """Pick up episodes added to the batch on screen; True when it grew."""
        store, files = self.store, self.store.files
        fresh = agent_cells.episode_files(self.open_batch)
        if [f.name for f in fresh] == [f.name for f in files]:
            return False
        here = files[store.ep].name if 0 <= store.ep < len(files) else ""
        files[:] = fresh
        # A new episode can sort before the open one.
        self.labels[:] = store.episode_labels()
        store.ep = next((i for i, f in enumerate(files) if f.name == here), 0)
        self.episode.slider.max = max(0, len(files) - 1)
        self.episode.dropdown.options = list(self.labels)
        self._sync_ui()
        print(
            f"[viewer] follow: the open batch now has {len(files)} episodes "
            f"(last {self.labels[-1]!r})",
            flush=True,
        )
        return True

    def _follow_tick(self, now: float) -> None:
        """Rescan the cell (main loop, throttled): batches, versions, run.json."""
        if self.follow_seconds <= 0.0 or now - self.follow_at < self.follow_seconds:
            return
        self.follow_at = now
        self.follow_scans += 1
        labels = self.picker.labels
        before = list(labels)
        self._refresh_batches()
        grew = self._refresh_open_batch()
        if self.cell_panels is not None:
            self.cell_panels.refresh_versions()
        if labels and list(labels) != before:
            fresh = [label for label in labels if label not in before]
            print(
                f"[viewer] follow: {len(labels)} batches"
                + (f" · new/changed: {', '.join(fresh)}" if fresh else ""),
                flush=True,
            )
        note = f"scan {self.follow_scans} · {len(labels)} rollouts"
        if grew:
            note += f" · this one: {len(self.store.files)} episodes"
        note += f" · {time.strftime('%H:%M:%S')}"
        if self.cell_panels is not None:
            self.cell_panels.note_follow(note)
            jump = self.cell_panels.auto_jump
        elif self.batch_dropdown is not None:
            if self.batch_dropdown.follow_note is not None:
                self.batch_dropdown.follow_note.content = (
                    f'<div style="font-size:10px;opacity:.55;">'
                    f"{html.escape(note)}</div>"
                )
            jump = self.batch_dropdown.auto_jump
        else:
            jump = None
        paths = self.picker.paths
        if jump is not None and bool(jump.value) and paths:
            newest = newest_batch(paths)
            if newest is not None and self._is_other_batch(newest):
                self.wanted_batch = newest

    def _exit_request(self, started: float) -> tuple[str, Any] | None:
        """What to say and hand back when the viewer should close, else None."""
        index = self.wanted_batch
        if index is not None and self._is_other_batch(index):
            return (
                f"[viewer] switching to {self.picker.item} "
                f"{self.picker.labels[index]} — "
                "rebuilding scene, the browser tab reconnects by itself",
                index,
            )
        self.wanted_batch = None
        if self.wanted_source is not None:
            return (
                f"[viewer] opening {self.wanted_source.where} — "
                "rebuilding scene, the browser tab reconnects by itself",
                self.wanted_source,
            )
        if self.wanted_compare is not None:
            request, self.wanted_compare = self.wanted_compare, None
            return (
                "[viewer] entering compare mode — rebuilding the scene, "
                "the browser tab reconnects by itself",
                request,
            )
        if self.exit_after > 0.0 and time.time() - started > self.exit_after:
            return (
                f"[viewer] --exit-after-seconds {self.exit_after:g} elapsed, closing",
                None,
            )
        return None

    def _show_frame(self) -> None:
        """Pose the scene at the playhead and refresh what depends on it."""
        store, head = self.store, self.head
        self.data.qpos[: store.qpos.shape[1]] = store.qpos[head.frame]
        mujoco.mj_forward(self.model, self.data)
        store.after_forward(head.frame)
        measured = self.targets.measure(self.inner)
        self.scene.update_from_mjdata(self.data)
        self.arrows.update(store, head.frame, self.show_arrows.value)
        n = store.qpos.shape[0]
        self.episode.progress.value = 100.0 * head.frame / max(1, n - 1)
        # The styled card is throttled: per-frame HTML churn makes the
        # sidebar jitter (~5-10 Hz is plenty). A reward change or the final
        # frame always refreshes, so the success frame never shows a stale 0
        # during the end hold.
        now = time.time()
        reward = float(store.rewards[min(head.frame, len(store.rewards) - 1)])
        if (
            now - self.card_at > 0.15
            or head.frame >= n - 1
            or reward != self.card_reward
        ):
            self.episode.card.content = episode_card_html(
                store, head.frame, self.realtime_fps
            )
            self.playback.reward_strip.content = sparkline_svg(
                store.rewards, head.frame, flags=episode_flags(store.info)
            )
            self.card_at = now
            self.card_reward = reward
        self.targets.show(measured)
        # MuJoCo renders cost real time, so the panels run at their own rate
        # (never faster than the playback rate) on the main loop.
        rate = min(CAMERA_PANEL_HZ, max(self.playback.fps.value, 1))
        if now - self.cameras_at > 1.0 / rate:
            self.cameras.update(self.arrows.pelvis_lookat(store.qpos[head.frame]))
            self.cameras_at = now
            if not self.cameras_logged:
                print(f"[viewer] camera panels rendering ({self.cameras.describe()})")
                self.cameras_logged = True

    def run(
        self,
    ) -> int | agent_cells.CompareRequest | DatasetSource | BatchSource | None:
        """Serve until the viewer is asked for another scene.

        Returns:
            The index of another batch (or task) to open, the
            :class:`~bigym.vr.viewer.agent_cells.CompareRequest` the Compare
            button made, the :class:`DatasetSource` / :class:`BatchSource`
            the "Demo directory" field loaded, or None once
            --exit-after-seconds elapsed. Ctrl-C is the other way out.
        """
        self._sync_ui()
        print(
            f"[viewer] episode list: {len(self.store.files)} entries, "
            f"first {self.labels[0]!r}",
            flush=True,
        )
        if self.cell_panels is not None:
            for line in self.cell_panels.describe():
                print(line, flush=True)
        if self.follow_seconds > 0.0:
            print(
                f"[viewer] --follow: rescanning every {self.follow_seconds:g}s",
                flush=True,
            )
        port = self.port
        print(f"[viewer] open http://localhost:{port} (ssh -L {port}:localhost:{port})")
        started = time.time()
        store, head = self.store, self.head
        while True:
            leaving = self._exit_request(started)
            if leaving is not None:
                message, result = leaving
                print(message)
                self.cameras.close()
                self.server.stop()
                return result
            if self.cell_panels is not None:
                self.cell_panels.tick(time.time(), self._refresh_batches)
            if self.follow_seconds > 0.0:
                self._follow_tick(time.time())
            wanted, self.wanted_episode = self.wanted_episode, None
            if wanted is not None and store.load(wanted):
                head.reset()
                self._sync_ui()
                self.dirty = True
            playback = self.playback
            if head.tick(
                store.qpos.shape[0], playback.playing.value, playback.fps.value
            ):
                playback.frame.value = head.frame
                self.dirty = True
            if self.dirty:
                self._show_frame()
                self.dirty = False
            time.sleep(0.004)


def _open_batch(
    args: ViewConfig,
    demo_dir: Path,
    batch_labels: list[str],
    batch_idx: int,
    directory: str,
    cell: Path | None = None,
    batch_paths: list[Path] | None = None,
    rescan=None,
    cells: list | None = None,
):
    """Build env + viewer for one batch.

    Returns another batch index to switch to (the Rollouts dropdowns, the
    "Demo batch" one or the native frontend's ``[`` / ``]``), a
    :class:`~vr.viewer.agent_cells.CompareRequest` when the Compare button
    was pressed, a source when the "Demo directory" field loaded another
    directory, or None when the viewer exits.
    """
    files = agent_cells.episode_files(demo_dir)
    if not files:
        raise FileNotFoundError(f"No .npz demos in {demo_dir}")
    metadata = load_metadata(demo_dir)
    print(f"[viewer] {len(files)} episodes | task={metadata['task']['task_name']}")

    env = build_env(metadata)
    env.reset(seed=0)
    inner = env.inner_env
    model = inner.model
    data = inner.data

    _tint(args, model)

    # Prefer the collection-time value stamped in metadata: the env's current
    # step duration can differ from what a demo was recorded under.
    meta_step_seconds = metadata.get("control_step_seconds")
    if meta_step_seconds:
        realtime_fps = 1.0 / float(meta_step_seconds)
    else:
        realtime_fps = 1.0 / float(inner.control_step_seconds)
    if abs(realtime_fps - DEMO_HZ) > 1e-6:
        print(
            f"[viewer] recorded frames span {1000.0 / realtime_fps:.0f} ms of "
            f"sim time each -> realtime playback is {realtime_fps:.0f} fps, "
            f"not the nominal {DEMO_HZ:.0f}"
        )

    store = DemoStore(env, files)
    store.load(args.episode)

    task = metadata["task"]
    try:
        if args.viewer == "viser":
            return ViserViewer(
                store,
                inner,
                Picker(batch_labels, batch_idx, paths=batch_paths, rescan=rescan),
                args,
                directory,
                realtime_fps=realtime_fps,
                task_name=task["task_name"],
                camera_keys=tuple(task.get("camera_keys") or ("head",)),
                camera_shape=tuple(task.get("camera_shape") or (84, 84)),
                cell=cell,
                cells=cells,
            ).run()
        try:
            return run_mujoco(
                store,
                inner,
                model,
                data,
                realtime_fps,
                batch_labels=batch_labels,
                batch_idx=batch_idx,
            )
        except Exception as exc:
            raise SystemExit(
                f"native viewer failed ({exc}); over SSH use --viewer viser"
            ) from exc
    finally:
        env.close()


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


def _rescan_all(
    roots: list[Path], label_root: Path, batches: list[Path], labels: list[str]
) -> bool:
    """Re-discover every root into the viewer's live lists; True when changed."""
    fresh = discover_all(roots)
    changed = fresh != batches
    batches[:] = fresh
    labels[:] = batch_labels(label_root, fresh)
    return changed


def _print_discovery(roots: list[Path], cells: list) -> None:
    """The first lines of the startup log: what --demo-dir turned out to be."""
    if not cells:
        return
    sessions = agent_cells.sessions_of(cells)
    tasks = agent_cells.tasks_of(cells)
    if len(cells) == 1:
        print(f"[viewer] agent cell {cells[0].path}")
        return
    print(
        f"[viewer] agent runs under {_label_root(roots)}: {len(sessions)} "
        f"sessions ({', '.join(sessions)}) x {len(tasks)} tasks, "
        f"{len(cells)} cells"
    )


def _opening_index(
    args: ViewConfig, cells: list, batches: list[Path], labels: list[str]
) -> int:
    """Which batch opens first: --batch, else the default rollout of a cell."""
    if args.batch is not None:
        return pick_batch(labels, args.batch)
    if cells:
        rollout, _ = agent_cells.default_rollout(
            agent_cells.rollouts_of(cells[0].path, batches)
        )
        if rollout is not None and rollout.batch in batches:
            return batches.index(rollout.batch)
    return 0


def _run_compare(args: ViewConfig, request, back_to: int) -> int:
    """Produce any missing rollout, then play the slots side by side.

    Args:
        args: The parsed command line.
        request: The resolved compare request.
        back_to: The batch index to return to afterwards.

    Returns:
        ``back_to`` — leaving compare mode returns to the single-rollout
        scene that asked for it, whether the scene was built or the user
        left while the missing rollouts were still running.
    """
    if request.error:
        print(f"[viewer] compare: {request.error}", flush=True)
        return back_to
    if not request.slots:
        print("[viewer] compare: no slots to compare", flush=True)
        return back_to
    print(
        f"[viewer] compare: layout {request.layout} — "
        f"{agent_cells.grid_note(len(request.slots), request.gap, request.layout)}",
        flush=True,
    )
    for i, slot in enumerate(request.slots):
        place = agent_cells.slot_grid(len(request.slots), request.layout)[i]
        print(
            f"[viewer] compare slot {i}: {slot.session} {slot.version_name} "
            f"seed {request.seed} at row {place[0]} column {place[1]} "
            f"<- {slot.source or 'not rolled out yet'}",
            flush=True,
        )
    if request.missing:
        # The rollouts are produced by run_compare itself, inside the viser
        # server it opens first: one progress bar per job, a working Leave
        # button, and the scene built when the last episode lands.
        print(
            f"[viewer] compare: {len(request.missing)} slot(s) have no episode "
            f"on seed {request.seed}; running bigym-agent replay --no-video",
            flush=True,
        )
    else:
        print(
            f"[viewer] compare: every slot already stores seed {request.seed} "
            "(evaluation batches hold every hidden seed) — no replay needed",
            flush=True,
        )
    agent_cells.run_compare(
        request,
        port=args.port,
        build_env=build_env,
        load_metadata=load_metadata,
        camera_panels=CameraPanels,
        lighting_prefs=_LIGHTING_PREFS,
        exit_after=args.exit_after_seconds,
        record=record_request(args),
    )
    return back_to


def record_request(args: ViewConfig) -> dict | None:
    """What ``--record`` asks :func:`agent_cells.run_compare` to write."""
    if not args.record:
        return None
    return {
        "path": Path(args.record),
        "seconds": args.record_seconds,
        "size": video.parse_record_size(args.record_size),
        "fps": args.record_fps,
        "browser": args.browser,
        "stay": args.record_then_stay,
        "view": args.record_view,
        "zoom": args.record_zoom,
        "speed": args.record_speed,
        "label_scale": args.record_label_scale,
    }


def _cli_compare(args: ViewConfig, cells: list, batches: list[Path], idx: int):
    """Build the compare request ``--compare`` asked for, or None."""
    if not args.compare:
        return None
    if args.viewer != "viser":
        raise SystemExit("--compare needs --viewer viser")
    if not cells:
        raise SystemExit("--compare needs agent cells under --demo-dir")
    here = agent_cells.cell_of(cells, batches[idx]) or cells[0]
    try:
        specs = agent_cells.parse_compare(args.compare)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    request = agent_cells.resolve_compare(cells, here.task, specs, args.seed)
    request.gap = args.compare_gap
    request.layout = args.layout
    request.back_to = idx
    return request


def _tint(args: ViewConfig, model) -> None:
    """Display-only robot tint, applied before mjviser converts the meshes."""
    if args.robot_tint == "dark" and args.viewer == "viser":
        n = tint_robot_geoms(model)
        print(f"[viewer] tinted {n} robot geoms (display only)")


def _open_dataset_task(args: ViewConfig, source: DatasetSource, idx: int):
    """Build the replay env + viewer for one task of a dataset.

    Returns another task index (the Task dropdown or ``[`` / ``]``), a
    source when the "Demo directory" field loaded another directory, or None
    when the viewer exits.
    """
    task, folder, label = source.tasks[idx], source.dirs[idx], source.labels[idx]
    if folder is None:
        print(f"[viewer] {task}: fetching from {hub.dataset_repo()}", flush=True)
    demos = TaskDemos(task, folder)
    episodes = demos.episodes
    print(f"[viewer] {label}: {len(episodes)} episodes | task={task} | {demos.dir}")
    step = demos.metadata.get("control_step_seconds")
    realtime_fps = 1.0 / float(step) if step else DEMO_HZ

    env = replay_env(task)
    inner = env.inner_env
    model = inner.model
    data = inner.data
    _tint(args, model)
    store = DatasetStore(env, inner, demos)
    store.load(args.episode)
    try:
        if args.viewer == "viser":
            picker = Picker(
                source.labels, idx, item="task", build=_task_picker(source, idx)
            )
            directory = shlex.join(str(r) for r in source.roots)
            return ViserViewer(
                store,
                inner,
                picker,
                args,
                directory,
                realtime_fps=realtime_fps,
                task_name=task,
            ).run()
        try:
            return run_mujoco(
                store,
                inner,
                model,
                data,
                realtime_fps,
                batch_labels=source.labels,
                batch_idx=idx,
                item="task",
            )
        except Exception as exc:
            raise SystemExit(
                f"native viewer failed ({exc}); over SSH use --viewer viser"
            ) from exc
    finally:
        env.close()


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


def _run_dataset(args: ViewConfig, source: DatasetSource):
    """Play a dataset's tasks until the viewer exits or opens another source."""
    if args.batch is not None or args.compare:
        raise SystemExit(
            "--batch and --compare apply to demo batches and agent runs; "
            f"{source.where} is a dataset (use --task)"
        )
    idx = pick_task(source, args.task)
    print(
        f"[viewer] {len(source.labels)} tasks in {source.where}; opening "
        f"{source.labels[idx]}"
    )
    while True:
        result = _open_dataset_task(args, source, idx)
        if not isinstance(result, int):
            return result
        idx = result


def _run_batches(args: ViewConfig, source: BatchSource):
    """Open the demo batches, or the agent rollouts, of a batch source.

    Returns the source the "Demo directory" field loaded, or None when the
    viewer exits.
    """
    if args.task is not None:
        raise SystemExit(
            f"--task picks a dataset task; {source.where} holds demo batches "
            "(use --batch)"
        )
    roots, label_root = source.roots, source.label_root
    batches, labels, cells = source.batches, source.labels, source.cells
    _print_discovery(roots, cells)

    def rescan() -> list[str]:
        """Re-discover batches in place (a finished job wrote a new one)."""
        if _rescan_all(roots, label_root, batches, labels):
            print(f"[viewer] {len(batches)} batches under {label_root}")
        return labels

    idx = _opening_index(args, cells, batches, labels)
    here = agent_cells.cell_of(cells, batches[idx])
    if here is not None and args.viewer != "viser":
        # The viser frontend prints this from the panels it builds; the
        # native one has no panels, so say it here.
        rollouts = agent_cells.rollouts_of(here.path, batches)
        opened, reason = agent_cells.default_rollout(rollouts)
        for line in agent_cells.describe_opening(here, rollouts, opened, reason):
            print(line, flush=True)
    if not cells and len(batches) > 1 and args.batch is None:
        # No picker: list the batches once, open the first, and let the
        # dropdown ("[" / "]" natively) move between them.
        print(f"[viewer] {len(batches)} demo batches under {label_root}:")
        for i, label in enumerate(labels):
            print(f"  [{i:2d}] {label}")
        print(
            f"[viewer] opening [{idx}] {labels[idx]} "
            "(--batch <index-or-substring> picks another)"
        )
    compare = _cli_compare(args, cells, batches, idx)

    while True:
        if compare is not None:
            request, compare = compare, None
            idx = _run_compare(args, request, request.back_to)
            if args.exit_after_seconds > 0.0:
                return None  # headless smoke: one compare pass and out
            if args.record and not args.record_then_stay:
                return None  # the file is written; that is what was asked for
            continue
        here = agent_cells.cell_of(cells, batches[idx])
        result = _open_batch(
            args,
            batches[idx],
            labels,
            idx,
            source.where,
            cell=here.path if here is not None else None,
            batch_paths=batches,
            rescan=rescan,
            cells=cells,
        )
        if result is None or isinstance(result, (DatasetSource, BatchSource)):
            return result
        if isinstance(result, agent_cells.CompareRequest):
            result.back_to = idx
            compare = result
            continue
        idx = result


def main() -> None:
    """Open the published demonstrations, or what --demo-dir points at."""
    args = _parse_args()

    if args.demo_dir:
        source = _local_source([Path(raw).expanduser() for raw in args.demo_dir])
    else:
        try:
            source = _published_source()
        except Exception as exc:  # offline: --task still works
            if args.task is None:
                raise SystemExit(
                    f"cannot list the published tasks ({exc}); pass --task NAME "
                    "to open a cached task, or --demo-dir PATH"
                ) from exc
            source = DatasetSource(labels=[args.task], tasks=[args.task], dirs=[None])
    while source is not None:
        if isinstance(source, DatasetSource):
            source = _run_dataset(args, source)
        else:
            source = _run_batches(args, source)
        # What to open first applies to the command line's source only.
        args.batch = args.task = args.compare = None
        args.episode = 0


if __name__ == "__main__":
    main()
