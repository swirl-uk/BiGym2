#!/usr/bin/env python3
"""The agent-run half of ``bigym-view``: cells, rollouts, panels, compare.

A "cell" is one task x one session of the coding-agent benchmark
(``bigym.loco.agent``). Its directory holds ``policies/index.json`` (every
policy version the agent wrote), ``run.json``, ``transcript.jsonl`` and three
kinds of rollout, each of whose ``batch/`` subdirectory is an ordinary
replay-format batch:

``eval/vNNN/``
    The hidden-seed evaluation of one version (100 episodes, seeds
    620000-620099).
``dev/vNNN_<stamp>/``
    The episodes the agent itself ran while it worked. They are grouped by
    the ``policy.py`` version that was current at the time, which is NOT the
    same as "episodes of that version": agents often develop in other files
    while ``policy.py`` still holds the template, so the labels say
    "while policy.py was v001".
``replays/vNNN/`` or ``replays/policy_<sha8>/``
    On-demand rollouts written by ``bigym-agent replay``.

Sessions and roots: a *root* is a directory of cells (``<root>/<task>/``),
one root per session, and a directory of roots is a set of sessions. The
viewer accepts any of the three (``--demo-dir`` takes several paths) and
turns them into a Task x Session choice over one flat list of rollouts.

``bigym/vr/viewer/view_demos.py`` keeps the generic viewer (batch discovery, env
build, playback, camera panels, figure capture) and calls into this module
for everything that knows the run layout. The dependency runs one way: this
module never imports the viewer.

Threading: every function here is called from the viewer's main loop. GUI
callbacks only set request flags (env builds, renders and subprocess
launches belong to the main thread, which owns the EGL context).
"""

from __future__ import annotations

import contextlib
import difflib
import hashlib
import html
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import cast

import mjviser.scene as mjscene
import mujoco
import numpy as np
import viser
from mjviser import ViserMujocoScene
from PIL import Image, ImageDraw, ImageFont
from pygments import highlight
from pygments.formatters.html import HtmlFormatter
from pygments.lexers.python import PythonLexer

from bigym.loco.agent.snapshots import version_dir
from bigym.loco.agent.transcript import TOKEN_FIELDS, price_of
from bigym.loco.eval.protocol import EVAL_SEED_BASE
from bigym.vr.viewer import video
from bigym.vr.viewer.mjviser_compat import (
    hide_reach_targets,
    patch_mjviser,
    patch_mjviser_track_props,
)

# ---------------------------------------------------------------------------
# Run layout (the viewer only ever reads these)
# ---------------------------------------------------------------------------

POLICY_INDEX = Path("policies") / "index.json"
AGENT_BATCH_KINDS = ("eval", "replays", "dev")
# A batch of a policy file that is not one of the recorded versions lives in
# ``<kind>/policy_<first 8 of the sha256>/batch`` (bigym-agent replay --policy).
POLICY_BATCH_PREFIX = "policy_"
RUN_JSON = "run.json"
TRANSCRIPT_JSONL = "transcript.jsonl"
# Command blocks in the transcript panel and the policy source are quoted up
# to this many lines (the panel is a sidebar, not a pager).
TRANSCRIPT_COMMAND_LINES = 20
# The first five hidden evaluation seeds are the conventional "show me what it
# does" set for the Replay button.
DEFAULT_REPLAY_SEEDS = f"{EVAL_SEED_BASE}-{EVAL_SEED_BASE + 4}"
EVAL_SEED_LO = EVAL_SEED_BASE
AGENT_CLI_MODULE = "bigym.loco.agent"
# Subprocess output goes to a file under the cell so the main loop never
# blocks reading a pipe (and so a failure can be quoted back into the panel).
JOB_LOG_DIR = "viewer_jobs"
# A running replay or evaluation writes this file in its output directory
# (bigym.loco.agent.batch.RunProgress); it is what the progress bars read.
PROGRESS_NAME = ".progress.json"
# What ``bigym-agent evaluate`` runs when nothing says otherwise, so a bar
# has a denominator before the first episode lands in episodes.csv.
DEFAULT_EVAL_EPISODES = 100

# The Rollouts picker's first level: what the user calls each kind, in the
# order the dropdown offers them.
ROLLOUT_KINDS = (("Evaluation", "eval"), ("Development", "dev"), ("Replays", "replays"))
KIND_TITLES = {kind: title for title, kind in ROLLOUT_KINDS}

# Compare mode: up to six environments in a grid. A column steps along +y
# and a row steps along -x, so several rows stand behind one another rather
# than side by side for ever. The panel starts with one slot per session
# that holds the task (its submission) and "Add policy" fills the rest.
COMPARE_SLOTS = 6
COMPARE_GAP = 2.5
# What the layout dropdown offers, and what each one means in columns.
LAYOUT_AUTO = "auto"
LAYOUT_ROW = "row"
COMPARE_LAYOUTS = (LAYOUT_AUTO, LAYOUT_ROW, "2 columns", "3 columns")
# ``auto`` keeps a single row while it stays readable and folds into two
# columns from the fourth slot on.
AUTO_ROW_MAX = 3
# One tint per slot so the robots read apart at a glance. A tint is a
# per-channel multiplier on the robot's geom colours and is DISPLAY ONLY: it
# is written to the model only while mjviser bakes that slot's meshes, and
# the original ``geom_rgba`` is put back before anything renders from the
# model again -- the camera panels go through MuJoCo's renderer, which reads
# ``geom_rgba`` live, and they must show the policy's own view.
SLOT_TINTS = (
    (0.45, 0.68, 1.00),
    (1.00, 0.62, 0.32),
    (0.48, 0.88, 0.52),
    (0.90, 0.56, 0.95),
    (1.00, 0.85, 0.35),
    (0.40, 0.85, 0.92),
)
# The same six colours as CSS, for the legend's swatches and the stripe down
# the side of each 3D billboard.
SLOT_COLOURS = (
    "#73aeff",
    "#ff9e52",
    "#7ae085",
    "#e68ef2",
    "#ffd95a",
    "#66d9eb",
)
# How far above the pelvis a slot's billboard hangs, in metres, and how
# wide the plate is drawn.
BILLBOARD_Z = 1.15
BILLBOARD_WIDTH = 0.9
# The billboard is a flat image, so it has one face: it looks back along -x,
# which is the side both the single-rollout viewer and the compare camera
# watch the robots from.
PLATE_FACE = (-1.0, 0.0, 0.0)
PLATE_UP = (0.0, 0.0, 1.0)
# The plate is drawn on a canvas of this fixed size so the image node's
# geometry never has to change when the text does.
PLATE_SIZE = (720, 176)
PLATE_FONT_SIZE = 36
# A long label is drawn smaller rather than clipped, down to this size.
PLATE_FONT_MIN = 14
# DejaVu carries the glyphs the labels use (· ✓ ✗); PIL's own default face
# is the fallback when no DejaVu is installed.
PLATE_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSans.ttf",
)
SLOT_EMPTY = "(empty)"
VERSION_LAST = "last"
FOCUS_FREE = "(free view)"
# What a label says instead of a dash: a dash reads as "broken", and these
# two absences have names.
NO_TRAIN = "no dev episodes"
NO_EVAL = "not evaluated"
# A cell keeps the policy the agent started from here; a v001 that still
# matches it byte for byte is the template, not a first attempt.
TEMPLATE_POLICY = "initial_policy.py"
TEMPLATE_MARK = "(template)"
# The builder's untouched policy is recorded as a version (v000 in new cells,
# v001 in older ones) so diffs and the void verdict have a baseline; a viewer
# hides it unless asked, because it is not something the agent wrote.
SHOW_TEMPLATE = False
# Session names in one benchmark run share a prefix (``astra_high_s2``);
# labels drop it when every slot has it, but only at one of these.
NAME_SEPARATORS = "_-./"


# ---------------------------------------------------------------------------
# Small readers
# ---------------------------------------------------------------------------


def read_json(path: Path) -> dict | None:
    """Parse a JSON file; None when it is missing, unreadable or malformed."""
    try:
        with Path(path).open() as f:
            loaded = json.load(f)
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def as_float(value) -> float | None:
    """Coerce a JSON value to float; None when it is missing or not one."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value) -> int | None:
    """Coerce a JSON value to int; None when it is missing or not one."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def iso_seconds(text: str) -> float | None:
    """Unix seconds of an ISO-8601 timestamp, None when it cannot be read."""
    try:
        return datetime.fromisoformat(str(text)).timestamp()
    except (TypeError, ValueError):
        return None


def file_digest(path: Path) -> str:
    """The sha256 of a file, or "" when it cannot be read.

    ``bigym-agent replay --policy <file>`` names its output directory after
    the first eight characters of this digest, so the panel can point at the
    batch the job is about to write.
    """
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def percent(rate) -> str:
    """A success rate as ``41%``; "" when there is none to show."""
    value = as_float(rate)
    return "" if value is None else f"{100.0 * value:.0f}%"


def compact(value) -> str:
    """Render a token count as ``105k`` / ``0.6M``; "—" when unknown."""
    number = as_float(value)
    if number is None:
        return "—"
    if abs(number) >= 5e5:  # 593,640 reads as 0.6M; 105,312 stays 105k
        return f"{number / 1e6:.1f}M"
    if abs(number) >= 1e3:
        return f"{number / 1e3:.0f}k"
    return f"{number:.0f}"


def duration(seconds) -> str:
    """Render a duration as ``1h 37m`` / ``12m 03s``; "—" when unknown."""
    value = as_float(seconds)
    if value is None:
        return "—"
    value = int(round(value))
    hours, rest = divmod(value, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {secs:02d}s"


# ---------------------------------------------------------------------------
# Progress of a running rollout (the ``.progress.json`` its process writes)
# ---------------------------------------------------------------------------


def read_progress(out_dir: Path) -> dict | None:
    """Read the ``.progress.json`` of a running replay or evaluation.

    Args:
        out_dir: The job's output directory (``replays/vNNN``, ``eval/vNNN``).

    Returns:
        The state dict written by ``bigym.loco.agent.batch.RunProgress``
        (``seed``, ``step``, ``max_steps``, ``episode``, ``episodes``,
        ``done``, ``pid``), or None when the job has not written one yet —
        which is also what a job that has just started looks like.
    """
    return read_json(Path(out_dir) / PROGRESS_NAME)


def progress_fraction(state: dict | None) -> float:
    """How far a run has got, 0 to 1; 0 when nothing is known yet.

    One episode is ``step / max_steps``. A run of several episodes counts
    the finished ones too, so the bar crosses the whole run once rather
    than snapping back to the left at every episode. An evaluation reports
    no steps at all (its episodes run in worker processes), and then the
    fraction is simply the finished episodes over the requested ones.

    Args:
        state: A dict from :func:`read_progress`, or None.

    Returns:
        A fraction in ``[0, 1]``.
    """
    if not state:
        return 0.0
    step = as_int(state.get("step")) or 0
    total = as_int(state.get("max_steps")) or 0
    episodes = as_int(state.get("episodes")) or 0
    finished = as_int(state.get("episode")) or 0
    inside = min(1.0, step / total) if total > 0 else 0.0
    if episodes > 1:
        return max(0.0, min(1.0, (finished + inside) / episodes))
    if total > 0:
        return inside
    return max(0.0, min(1.0, finished / episodes)) if episodes > 0 else 0.0


def progress_label(head: str, state: dict | None) -> str:
    """Name one running job and say where it is.

    ``astra_high_s2 · v002 · seed 500 · 812/3523`` — ``head`` is whatever
    names the job (a compare slot, a version), and what follows is the step
    inside the episode, plus ``k/n episodes`` when the run is more than one.
    A job that has not written its progress file yet reads ``starting…``.

    Args:
        head: The job's name.
        state: A dict from :func:`read_progress`, or None.

    Returns:
        The label line.
    """
    if not state:
        return f"{head} · starting…"
    step = as_int(state.get("step")) or 0
    total = as_int(state.get("max_steps")) or 0
    episodes = as_int(state.get("episodes")) or 0
    finished = as_int(state.get("episode")) or 0
    parts = [head]
    if total > 0:
        parts.append(f"{step}/{total}")
    if episodes > 1 or total <= 0:
        parts.append(f"{finished}/{episodes} episodes")
    return " · ".join(parts)


def progress_note_html(label: str) -> str:
    """The text line that sits above one progress bar."""
    return (
        '<div style="font-size:10px;opacity:.7;font-family:ui-monospace,'
        f'monospace;">{html.escape(label)}</div>'
    )


# ---------------------------------------------------------------------------
# Episode files inside a replay-format batch
# ---------------------------------------------------------------------------


def is_episode_file(name: str) -> bool:
    """True for a finished episode file.

    Writers produce an episode under a temporary name and rename (or
    hard-link) it into place, so a name that is hidden or carries a ``.tmp``
    / ``.part`` marker is a file still being written, not an episode.
    """
    if not name.endswith(".npz") or name.startswith("."):
        return False
    return ".tmp" not in name and ".part" not in name


def episode_files(batch: Path) -> list[Path]:
    """The batch's episode files, in the order the viewer shows them."""
    return sorted(
        path for path in Path(batch).glob("*.npz") if is_episode_file(path.name)
    )


def episode_info(path: Path) -> dict:
    """Read one episode's scalars (seed, success, length, termination)."""
    info: dict = {
        "name": path.name,
        "seed": None,
        "success": None,
        "length": None,
        "termination": "",
        "fell": None,
    }
    try:
        with np.load(path, allow_pickle=False) as ep:
            keys = set(ep.files)
            if "seed" in keys:
                info["seed"] = int(np.asarray(ep["seed"]).reshape(-1)[0])
            if "success" in keys:
                info["success"] = float(np.asarray(ep["success"]).reshape(-1)[0])
            if "fell" in keys:
                info["fell"] = float(np.asarray(ep["fell"]).reshape(-1)[0])
            if "length" in keys:
                info["length"] = int(np.asarray(ep["length"]).reshape(-1)[0])
            elif "reward" in keys:
                info["length"] = int(np.asarray(ep["reward"]).reshape(-1).shape[0])
            if "termination" in keys:
                info["termination"] = str(np.asarray(ep["termination"]).reshape(-1)[0])
    except (OSError, ValueError, KeyError, EOFError):
        pass
    return info


# Per-episode scalars are read once per batch and re-read only when the set
# of .npz files changes (a running evaluation adds one file per episode).
_EPISODE_CACHE: dict[str, tuple[tuple[str, ...], list[dict]]] = {}


def episode_infos(batch: Path, files: list[Path] | None = None) -> list[dict]:
    """Scalars of every episode in a batch, cached until its files change."""
    batch = Path(batch)
    paths = list(files) if files is not None else episode_files(batch)
    signature = tuple(p.name for p in paths)
    cached = _EPISODE_CACHE.get(str(batch))
    if cached is not None and cached[0] == signature:
        return cached[1]
    infos = [episode_info(p) for p in paths]
    _EPISODE_CACHE[str(batch)] = (signature, infos)
    return infos


def batch_success(batch: Path) -> float | None:
    """Mean of the stored per-episode ``success`` values, when they exist."""
    values = [i["success"] for i in episode_infos(batch) if i["success"] is not None]
    return float(np.mean(values)) if values else None


# ---------------------------------------------------------------------------
# Cells, versions, sessions
# ---------------------------------------------------------------------------


def is_agent_cell(root: Path) -> bool:
    """True when root is an agent cell (it has policies/index.json)."""
    return (Path(root) / POLICY_INDEX).exists()


def is_policy_dirname(name: str) -> bool:
    """True for a ``policy_<8 hex>`` directory (``replay --policy <file>``)."""
    if not name.startswith(POLICY_BATCH_PREFIX):
        return False
    digest = name[len(POLICY_BATCH_PREFIX) :]
    return len(digest) == 8 and all(c in "0123456789abcdef" for c in digest)


def version_of_dirname(name: str) -> int | None:
    """Version number of ``vNNN`` / ``vNNN_<stamp>``, None when it is neither.

    Development batches are one directory per version *and* start time
    (``dev/v006_20260921_010203``), so the digits stop at the first
    underscore.
    """
    if not name.startswith("v"):
        return None
    digits = name[1:].split("_", 1)[0]
    return int(digits) if digits.isdigit() else None


def batch_version(batch: Path) -> tuple[Path, str, int | None] | None:
    """Split an agent-run-layout batch into (cell dir, kind, version).

    Args:
        batch: a discovered batch directory.

    Returns:
        ``(cell, "eval" | "replays" | "dev", version)`` for
        ``<cell>/<kind>/vNNN/batch`` (``dev`` also accepts
        ``vNNN_<stamp>``), ``(cell, kind, None)`` for a
        ``policy_<sha8>/batch`` written by ``replay --policy``, or None for
        any other batch (an exported dataset dir, a collector batch, ...).
    """
    batch = Path(batch)
    if batch.name != "batch":
        return None
    version_dir_ = batch.parent
    kind_dir = version_dir_.parent
    name = version_dir_.name
    if kind_dir.name not in AGENT_BATCH_KINDS:
        return None
    if is_policy_dirname(name):
        return kind_dir.parent, kind_dir.name, None
    version = version_of_dirname(name)
    if version is None:
        return None
    return kind_dir.parent, kind_dir.name, version


def eval_success(cell: Path, version: int) -> float | None:
    """Hidden-seed success rate from eval/vNNN/summary.json, when written."""
    summary = read_json(version_dir(cell, version, "eval") / "summary.json")
    if not summary or summary.get("success_rate") is None:
        return None
    try:
        return float(summary["success_rate"])
    except (TypeError, ValueError):
        return None


def entry_tokens(entry: dict) -> dict[str, int] | None:
    """The ``{input, cached, output, reasoning}`` block of an index entry."""
    tokens = entry.get("tokens")
    if not isinstance(tokens, dict):
        return None
    out = {}
    for field_name in TOKEN_FIELDS:
        out[field_name] = as_int(tokens.get(field_name)) or 0
    return out


def policy_rows(cell: Path) -> list[dict]:
    """The cell's policy versions, oldest first, with their known scores.

    Reads ``policies/index.json`` (version, time, trigger, train success as
    the agent's own runs measured it, and the provenance the snapshot
    watcher records: ``ts``, ``command_index``, ``budget_used``, ``tokens``,
    ``elapsed_s``) and joins each version with its hidden-seed
    ``eval/vNNN/summary.json`` where one exists. Provenance fields written
    by newer sessions only are None for an older cell.
    """
    index = read_json(Path(cell) / POLICY_INDEX) or {}
    rows: list[dict] = []
    for entry in index.get("versions") or []:
        if not isinstance(entry, dict):
            continue
        try:
            version = int(entry["version"])
        except (KeyError, TypeError, ValueError):
            continue
        when = str(entry.get("time") or "")
        rows.append(
            {
                "version": version,
                "time": when,
                "trigger": str(entry.get("trigger") or ""),
                "train_success": entry.get("train_success"),
                "train_episodes": entry.get("train_episodes"),
                "eval_success": eval_success(cell, version),
                # Provenance (absent in cells written before it existed).
                "ts": as_float(entry.get("ts")) or iso_seconds(when),
                "command_index": as_int(entry.get("command_index")),
                "budget_used": as_int(entry.get("budget_used")),
                "tokens": entry_tokens(entry),
                "elapsed_s": as_float(entry.get("elapsed_s")),
                "dir": str(entry.get("dir") or f"v{version:03d}"),
                "template": bool(entry.get("template")),
            }
        )
    rows = sorted(rows, key=lambda row: row["version"])
    for row in rows:
        if not row["template"] and row["version"] <= 1:
            # Older cells carry no flag: recognise the template by content.
            row["template"] = is_template_version(cell, row)
    return rows


def visible_rows(rows: list[dict]) -> list[dict]:
    """The versions a viewer lists: all of them, or the non-template ones."""
    if SHOW_TEMPLATE:
        return list(rows)
    return [row for row in rows if not row.get("template")]


def template_versions(cell: Path) -> set[int]:
    """The version numbers of the cell's template snapshots."""
    return {row["version"] for row in policy_rows(cell) if row.get("template")}


def submission_version(cell: Path, rows: list[dict] | None = None) -> int | None:
    """The version the session submitted: the last entry of the index."""
    rows = policy_rows(cell) if rows is None else rows
    return rows[-1]["version"] if rows else None


def read_run(cell: Path) -> dict:
    """The cell's ``run.json``; an empty dict when there is none yet."""
    return read_json(Path(cell) / RUN_JSON) or {}


@dataclass(frozen=True)
class Cell:
    """One task x one session: where it lives and what it is called."""

    path: Path
    task: str
    session: str

    @property
    def title(self) -> str:
        """``<session> / <task>``, how the startup log names the cell."""
        return f"{self.session} / {self.task}"


def _cells_under(root: Path) -> list[Path]:
    """The cell directories that are direct children of ``root``."""
    try:
        children = sorted(p for p in Path(root).iterdir() if p.is_dir())
    except OSError:
        return []
    return [p for p in children if is_agent_cell(p)]


def discover_cells(roots: list[Path] | tuple[Path, ...]) -> list[Cell]:
    """Find every agent cell under the given ``--demo-dir`` paths.

    Three shapes are accepted, tried in this order for each path:

    1. the path is a cell itself (``policies/index.json``) — its session is
       the name of its parent directory;
    2. the path is a session root (``<root>/<task>/`` cells) — its session
       is the name of the path;
    3. the path holds session roots (``<parent>/<root>/<task>/``) — one
       session per root.

    Args:
        roots: The resolved ``--demo-dir`` paths.

    Returns:
        The cells, sorted by session then task, with duplicates (the same
        directory reached through two paths) removed. Sessions whose
        directory names collide are disambiguated with their parent's name.
    """
    found: list[tuple[Path, str, Path]] = []  # (cell path, task, session dir)
    seen: set[Path] = set()
    for raw in roots:
        root = Path(raw).expanduser().resolve()
        candidates: list[tuple[Path, Path]] = []
        if is_agent_cell(root):
            candidates = [(root, root.parent)]
        else:
            direct = _cells_under(root)
            if direct:
                candidates = [(cell, root) for cell in direct]
            else:
                try:
                    children = sorted(p for p in root.iterdir() if p.is_dir())
                except OSError:
                    children = []
                for child in children:
                    candidates += [(cell, child) for cell in _cells_under(child)]
        for cell_path, session_dir in candidates:
            if cell_path in seen:
                continue
            seen.add(cell_path)
            found.append((cell_path, cell_path.name, session_dir))
    # Two roots can share a basename ("runs/a/nightly" and "runs/b/nightly");
    # only then is the parent name worth showing.
    by_name: dict[str, set[Path]] = {}
    for _, _, session_dir in found:
        by_name.setdefault(session_dir.name, set()).add(session_dir)
    cells = []
    for cell_path, task, session_dir in found:
        name = session_dir.name
        if len(by_name[name]) > 1:
            name = f"{session_dir.parent.name}/{name}"
        cells.append(Cell(path=cell_path, task=task, session=name))
    return sorted(cells, key=lambda c: (c.session, c.task))


def cell_of(cells: list[Cell], batch: Path) -> Cell | None:
    """The cell a discovered batch belongs to, None when it is a plain one."""
    info = batch_version(batch)
    if info is None:
        return None
    wanted = Path(info[0]).resolve()
    for cell in cells:
        if cell.path == wanted:
            return cell
    return None


def tasks_of(cells: list[Cell]) -> list[str]:
    """Every task name the discovered cells cover, sorted."""
    return sorted({cell.task for cell in cells})


def sessions_of(cells: list[Cell], task: str = "") -> list[str]:
    """Sessions that hold ``task`` (every session when ``task`` is empty)."""
    return sorted({c.session for c in cells if not task or c.task == task})


def find_cell(cells: list[Cell], task: str, session: str) -> Cell | None:
    """The cell for one task x session pair, None when there is none."""
    for cell in cells:
        if cell.task == task and cell.session == session:
            return cell
    return None


# ---------------------------------------------------------------------------
# Rollouts (the two-level picker)
# ---------------------------------------------------------------------------


@dataclass
class Rollout:
    """One rollout batch of a cell, as the Rollouts panel names it."""

    batch: Path
    cell: Path
    kind: str
    version: int | None
    policy_dir: str
    submission: bool

    @property
    def title(self) -> str:
        """The kind as the first dropdown spells it."""
        return KIND_TITLES.get(self.kind, self.kind)


def rollout_item_label(rollout: Rollout) -> str:
    """The second dropdown's text for one rollout.

    Evaluations and replays are named by their version (``v007 · 100
    episodes · 2%``); the submission is marked as such. Development batches
    are grouped by the ``policy.py`` version that was current while they
    ran, which is not the same as "episodes of that version" — agents often
    develop in other files — so they read ``while policy.py was v001 · 66
    episodes · 3%``.
    """
    count = len(episode_files(rollout.batch))
    if rollout.version is None:
        head = rollout.policy_dir or rollout.batch.parent.name
    elif rollout.kind == "dev":
        head = f"while policy.py was v{rollout.version:03d}"
    else:
        head = f"v{rollout.version:03d}"
    parts = [head, f"{count} episodes"]
    rate = None
    if rollout.kind == "eval" and rollout.version is not None:
        rate = eval_success(rollout.cell, rollout.version)
    if rate is None:
        rate = batch_success(rollout.batch)
    text = percent(rate)
    if text:
        parts.append(text)
    if rollout.submission:
        parts.append("submission")
    return " · ".join(parts)


def rollouts_of(cell: Path, batches: list[Path]) -> list[Rollout]:
    """The rollouts of one cell, taken from the viewer's batch list.

    Args:
        cell: The cell directory.
        batches: Every batch the viewer discovered (all cells, all kinds).

    Returns:
        The batches that belong to this cell, evaluations first, then
        development, then replays, each group newest version first.
    """
    wanted = Path(cell).resolve()
    submission = submission_version(wanted)
    hidden = set() if SHOW_TEMPLATE else template_versions(wanted)
    found: list[Rollout] = []
    for batch in batches:
        info = batch_version(batch)
        if info is None or Path(info[0]).resolve() != wanted:
            continue
        owner, kind, version = info
        if kind == "dev" and version in hidden:
            continue
        found.append(
            Rollout(
                batch=Path(batch),
                cell=Path(owner).resolve(),
                kind=kind,
                version=version,
                policy_dir=Path(batch).parent.name,
                submission=(
                    kind == "eval" and version is not None and version == submission
                ),
            )
        )
    order = {kind: i for i, (_, kind) in enumerate(ROLLOUT_KINDS)}
    return sorted(
        found,
        key=lambda r: (
            order.get(r.kind, 9),
            -(r.version if r.version is not None else -1),
            r.batch.parent.name,
        ),
    )


def default_rollout(rollouts: list[Rollout]) -> tuple[Rollout | None, str]:
    """Which rollout a cell opens on, and the sentence that says why.

    The evaluation of the submitted policy is what a reader wants first:
    it is the number the benchmark reports. Failing that, any evaluation,
    then any rollout at all.
    """
    if not rollouts:
        return None, "this cell has no rollouts yet"
    for rollout in rollouts:
        if rollout.submission:
            return rollout, "the evaluation of the submission"
    evaluations = [r for r in rollouts if r.kind == "eval"]
    if evaluations:
        return evaluations[0], "the newest evaluation (no submission scored yet)"
    return rollouts[0], f"no evaluation yet, so the newest {rollouts[0].title.lower()}"


def describe_opening(
    cell: Cell, rollouts: list[Rollout], opened, reason: str
) -> list[str]:
    """The startup log for an agent cell: what it is and what opened.

    Args:
        cell: The cell being opened.
        rollouts: Its rollouts.
        opened: The rollout on screen, or None.
        reason: Why that one (from :func:`default_rollout`).

    Returns:
        The lines to print, in order.
    """
    run = read_run(cell.path)
    lines = [f"[viewer] task {cell.task} · session {cell.session} · cell {cell.path}"]
    if run:
        verdict = (run.get("verdict") or {}).get("state")
        lines.append(
            f"[viewer] session: {run.get('harness')} {run.get('model')} "
            f"effort={run.get('effort')} interface={run.get('interface')} "
            f"verdict={verdict}"
        )
    lines.append(
        f"[viewer] rollouts: {rollout_counts(rollouts)} · "
        f"{len(visible_rows(policy_rows(cell.path)))} policy versions"
    )
    if opened is not None:
        lines.append(
            f"[viewer] opening {opened.title} {rollout_item_label(opened)} — {reason}"
        )
    return lines


def rollout_counts(rollouts: list[Rollout]) -> str:
    """``evaluation 1 · development 6 · replays 0`` for the startup log."""
    parts = []
    for title, kind in ROLLOUT_KINDS:
        parts.append(f"{title.lower()} {sum(1 for r in rollouts if r.kind == kind)}")
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# run.json, policy table, code and transcript panels
# ---------------------------------------------------------------------------


def token_cost(tokens: dict | None, price: dict | None) -> float | None:
    """USD for a token block at the given per-million prices."""
    if not tokens or not price:
        return None
    return (
        tokens.get("input", 0) * price.get("input", 0.0)
        + tokens.get("cached", 0) * price.get("cached", 0.0)
        + tokens.get("output", 0) * price.get("output", 0.0)
    ) / 1e6


def summary_html(run: dict) -> str:
    """The session summary card: what ``run.json`` says about this session.

    Shows the harness/model/effort/interface, the budget the environment
    server metered, the verdict, the wall clock, the token block and the
    cost (the run's own figure when it recorded one, else the price table).
    """
    if not run:
        return '<div style="font-size:11px;opacity:.6;">no run.json yet</div>'
    budget = cast(
        dict, run.get("budget") if isinstance(run.get("budget"), dict) else {}
    )
    usage = cast(dict, run.get("usage") if isinstance(run.get("usage"), dict) else {})
    verdict = cast(
        dict, run.get("verdict") if isinstance(run.get("verdict"), dict) else {}
    )
    cost = as_float(run.get("cost_usd"))
    if cost is None:
        cost = token_cost(usage, price_of(run.get("model")))
    used, cap = as_int(budget.get("used")), as_int(budget.get("cap"))
    state = str(verdict.get("state") or ("running" if not run.get("end") else "—"))
    reason = str(verdict.get("reason") or "")
    pairs = [
        ("harness", str(run.get("harness") or "—")),
        ("model", str(run.get("model") or "—")),
        ("effort", str(run.get("effort") or "—")),
        ("interface", str(run.get("interface") or "—")),
        (
            "budget",
            "—" if used is None else f"{used:,} / {cap:,}" if cap else f"{used:,}",
        ),
        ("verdict", state + (f" ({reason})" if reason else "")),
        ("wall clock", duration(run.get("wall_clock_s"))),
        (
            "tokens",
            "in {} · cached {} · out {} · reasoning {}".format(
                compact(usage.get("input")),
                compact(usage.get("cached")),
                compact(usage.get("output")),
                compact(usage.get("reasoning")),
            ),
        ),
        ("cost", "—" if cost is None else f"${cost:,.2f}"),
    ]
    body = "".join(
        '<tr><td style="opacity:.55;padding-right:8px;white-space:nowrap;">'
        f"{html.escape(key)}</td>"
        f'<td style="font-family:ui-monospace,monospace;">{html.escape(value)}</td>'
        "</tr>"
        for key, value in pairs
    )
    task = str(run.get("task") or "")
    head = (
        f'<div style="font-size:11px;font-weight:600;margin-bottom:4px;">'
        f"{html.escape(task)}</div>"
        if task
        else ""
    )
    return (
        '<div style="font-size:10px;color:inherit;">'
        f'{head}<table style="width:100%;border-collapse:collapse;">{body}</table>'
        "</div>"
    )


def table_columns(rows: list[dict]) -> list[str]:
    """Columns the policy table shows: provenance only when it was recorded."""
    columns = ["ver", "time", "trigger", "train", "eval"]
    if any(row.get("command_index") is not None for row in rows):
        columns.append("cmd")
    if any(row.get("budget_used") is not None for row in rows):
        columns.append("budget")
    if any(row.get("tokens") for row in rows):
        columns += ["tokens", "cost"]
    return columns


def policy_table_html(rows: list[dict], price: dict | None = None) -> str:
    """The "Policy versions" table.

    Always: version, time, trigger, the agent's own training score and the
    hidden-seed evaluation. The provenance columns (command index, budget
    used, ``input+output`` tokens, cost) are added only when at least one
    version carries them, so an older cell's table looks exactly as before.

    ``train`` and ``eval`` say what is absent rather than dashing it out
    (``no dev episodes``, ``not evaluated``): those two blanks have a
    meaning, where a missing token count is simply unknown and stays "—".

    Args:
        rows: Rows from :func:`policy_rows`.
        price: USD per million tokens for the session's model, or None (the
            cost column then reads "—").

    Returns:
        The table as HTML.
    """
    if not rows:
        return (
            '<div style="font-size:11px;opacity:.6;">no policies/index.json '
            "entries</div>"
        )

    def two_decimals(value, absent: str = "—") -> str:
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return absent

    columns = table_columns(rows)
    show_cmd = "cmd" in columns
    show_budget = "budget" in columns
    show_tokens = "tokens" in columns
    head = '<tr style="opacity:.55;text-align:left;">' + "".join(
        f"<th>{c}</th>" for c in columns
    )
    head += "</tr>"
    body = []
    for row in rows:
        when = row["time"][:19].replace("T", " ") if row["time"] else "—"
        cells = [
            f'<td style="font-family:ui-monospace,monospace;">v{row["version"]:03d}</td>',
            f"<td>{html.escape(when)}</td>",
            f"<td>{html.escape(row['trigger'] or '—')}</td>",
            f'<td style="white-space:nowrap;">'
            f"{two_decimals(row['train_success'], NO_TRAIN)}</td>",
            f'<td style="white-space:nowrap;">'
            f"{two_decimals(row['eval_success'], NO_EVAL)}</td>",
        ]
        if show_cmd:
            index = row.get("command_index")
            cells.append(f"<td>{'—' if index is None else index}</td>")
        if show_budget:
            cells.append(f"<td>{compact(row.get('budget_used'))}</td>")
        if show_tokens:
            tokens = row.get("tokens") or None
            if tokens:
                pair = f"{compact(tokens.get('input'))}/{compact(tokens.get('output'))}"
            else:
                pair = "—"
            cost = token_cost(tokens, price)
            cells.append(f'<td style="white-space:nowrap;">{pair}</td>')
            cells.append(f"<td>{'—' if cost is None else f'${cost:,.2f}'}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div style="font-size:10px;color:inherit;overflow-x:auto;">'
        '<table style="width:100%;border-collapse:collapse;">'
        f"{head}{''.join(body)}</table></div>"
    )


def policy_file(cell: Path, row: dict) -> Path:
    """Where a version's ``policy.py`` lives (``policies/vNNN/policy.py``)."""
    name = str(row.get("dir") or f"v{int(row['version']):03d}")
    return Path(cell) / "policies" / name / "policy.py"


def highlight_python(source: str) -> str:
    """Syntax-highlight Python as standalone HTML."""
    formatter = HtmlFormatter(nowrap=True)
    marked = highlight(source, PythonLexer(), formatter)
    style = formatter.get_style_defs(".bigym-code")
    return (
        f"<style>{style}</style>"
        '<pre class="bigym-code" style="margin:0;font-family:ui-monospace,'
        f'monospace;font-size:10px;white-space:pre;">{marked}</pre>'
    )


def code_html(path: Path, max_height: int = 420) -> str:
    """The policy source of one version, scrollable and highlighted."""
    path = Path(path)
    try:
        source = path.read_text(errors="replace")
    except OSError as exc:
        return f'<div style="font-size:11px;opacity:.6;">{html.escape(str(exc))}</div>'
    lines = len(source.splitlines())
    return (
        '<div style="font-size:10px;color:inherit;">'
        f'<div style="opacity:.5;margin-bottom:4px;">{lines} lines</div>'
        f'<div style="max-height:{max_height}px;overflow:auto;'
        'background:rgba(128,128,128,.10);border-radius:6px;padding:6px 8px;">'
        f"{highlight_python(source)}</div></div>"
    )


def vscode_link_html(path: Path) -> str:
    """A ``vscode://file/<abs path>`` link for the selected policy file."""
    absolute = str(Path(path).resolve())
    href = "vscode://file/" + absolute.lstrip("/")
    return (
        '<div style="font-size:10px;color:inherit;">'
        f'<a href="{html.escape(href, quote=True)}" style="color:#60a5fa;">'
        "open in VS Code</a>"
        '<span style="opacity:.5;"> — works when the browser runs on the '
        "machine that holds the files</span></div>"
    )


def diff_html(
    left: str, right: str, left_label: str, right_label: str, max_height: int = 360
) -> str:
    """A unified diff of two policy sources, added/removed lines coloured."""
    lines = list(
        difflib.unified_diff(
            left.splitlines(),
            right.splitlines(),
            fromfile=left_label,
            tofile=right_label,
            lineterm="",
        )
    )
    if not lines:
        return (
            '<div style="font-size:11px;opacity:.6;">identical to '
            f"{html.escape(right_label)}</div>"
        )
    rendered = []
    for line in lines:
        if line.startswith("+++") or line.startswith("---"):
            style = "opacity:.55;"
        elif line.startswith("@@"):
            style = "color:#a855f7;"
        elif line.startswith("+"):
            style = "color:#16a34a;background:rgba(22,163,74,.10);"
        elif line.startswith("-"):
            style = "color:#dc2626;background:rgba(220,38,38,.10);"
        else:
            style = "opacity:.75;"
        rendered.append(f'<div style="{style}">{html.escape(line) or "&nbsp;"}</div>')
    added = sum(
        1 for line in lines if line.startswith("+") and not line.startswith("+++")
    )
    removed = sum(
        1 for line in lines if line.startswith("-") and not line.startswith("---")
    )
    return (
        '<div style="font-size:10px;color:inherit;">'
        f'<div style="opacity:.55;margin-bottom:4px;">+{added} / -{removed} '
        f"({html.escape(left_label)} → {html.escape(right_label)})</div>"
        f'<div style="max-height:{max_height}px;overflow:auto;'
        "font-family:ui-monospace,monospace;white-space:pre;"
        'background:rgba(128,128,128,.10);border-radius:6px;padding:6px 8px;">'
        + "".join(rendered)
        + "</div></div>"
    )


def read_transcript(cell: Path) -> list[dict]:
    """Parse ``<cell>/transcript.jsonl``; an empty list when there is none."""
    path = Path(cell) / TRANSCRIPT_JSONL
    records: list[dict] = []
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue  # a line still being written
                if isinstance(record, dict):
                    records.append(record)
    except OSError:
        return []
    return records


def version_bounds(rows: list[dict], index: int) -> tuple[float | None, float | None]:
    """Timestamps bracketing what the agent did to produce the next version.

    Args:
        rows: Rows from :func:`policy_rows`, oldest first.
        index: Position of the selected version in ``rows``.

    Returns:
        ``(start, end)`` unix seconds; ``end`` is None for the last version
        (the slice runs to the end of the transcript) and ``start`` is None
        when the selected version carries no timestamp at all.
    """
    if not rows or not 0 <= index < len(rows):
        return None, None
    start = rows[index].get("ts")
    end = rows[index + 1].get("ts") if index + 1 < len(rows) else None
    return start, end


def transcript_slice(
    records: list[dict], start: float | None, end: float | None
) -> list[dict]:
    """The records between two version timestamps.

    Records without a ``t`` are kept when no record has one at all (a
    transcript written before timestamps were recorded cannot be sliced, and
    showing everything beats showing nothing).
    """
    timed = [r for r in records if as_float(r.get("t")) is not None]
    if not timed:
        return list(records)
    out = []
    for record in timed:
        when = as_float(record.get("t"))
        assert when is not None
        if start is not None and when < start:
            continue
        if end is not None and when >= end:
            continue
        out.append(record)
    return out


def slice_by_commands(
    records: list[dict], start: int | None, end: int | None
) -> list[dict]:
    """Fallback slice: records around the Nth command of the stream.

    Used when the transcript carries no timestamps but the policy index
    recorded how many commands the harness had run when each version was
    snapshotted.
    """
    if start is None:
        return list(records)
    out: list[dict] = []
    seen = 0
    for record in records:
        if seen >= start and (end is None or seen < end):
            out.append(record)
        if str(record.get("kind")) == "command":
            seen += 1
    return out


def version_records(records: list[dict], rows: list[dict], index: int) -> list[dict]:
    """What the agent did between the selected version and the next one."""
    start, end = version_bounds(rows, index)
    if any(as_float(r.get("t")) is not None for r in records):
        return transcript_slice(records, start, end)
    starts = rows[index].get("command_index") if 0 <= index < len(rows) else None
    ends = rows[index + 1].get("command_index") if index + 1 < len(rows) else None
    return slice_by_commands(records, starts, ends)


def transcript_title(rows: list[dict], index: int) -> str:
    """The panel title naming the two versions a slice sits between."""
    if not rows or not 0 <= index < len(rows):
        return "no policy versions"
    here = f"v{rows[index]['version']:03d}"
    if index + 1 < len(rows):
        nxt = f"v{rows[index + 1]['version']:03d}"
        return f"after {here}, up to {nxt} — what produced {nxt}"
    return f"after {here} — the tail of the session (no later version yet)"


def clip_lines(text: str, limit: int = TRANSCRIPT_COMMAND_LINES) -> str:
    """Clip a block to ``limit`` lines, noting how many were dropped."""
    lines = str(text).splitlines()
    if len(lines) <= limit:
        return "\n".join(lines)
    return "\n".join(lines[:limit]) + f"\n… {len(lines) - limit} more lines"


def transcript_html(records: list[dict], max_height: int = 420) -> str:
    """Render transcript records: messages as prose, commands as blocks."""
    if not records:
        return '<div style="font-size:11px;opacity:.6;">nothing in this slice</div>'
    blocks = []
    for record in records:
        kind = str(record.get("kind") or "")
        role = str(record.get("role") or "")
        if kind == "command":
            code = clip_lines(record.get("command") or "")
            exit_code = record.get("exit_code")
            tail = "" if exit_code in (None, 0) else f" (exit {exit_code})"
            blocks.append(
                f'<div style="opacity:.5;margin-top:6px;">$ command{tail}</div>'
                '<pre style="margin:2px 0;padding:5px 7px;border-radius:5px;'
                "background:rgba(128,128,128,.14);font-family:ui-monospace,monospace;"
                'font-size:10px;white-space:pre-wrap;">'
                f"{html.escape(code)}</pre>"
            )
        elif kind == "output":
            blocks.append(
                '<pre style="margin:2px 0;padding:5px 7px;border-radius:5px;'
                "background:rgba(128,128,128,.07);font-family:ui-monospace,monospace;"
                'font-size:10px;white-space:pre-wrap;opacity:.7;">'
                f"{html.escape(clip_lines(record.get('text') or ''))}</pre>"
            )
        elif kind in ("message", "reasoning"):
            label = "reasoning" if kind == "reasoning" else role or "agent"
            blocks.append(
                f'<div style="opacity:.5;margin-top:6px;">{html.escape(label)}</div>'
                f'<div style="font-size:11px;white-space:pre-wrap;">'
                f"{html.escape(clip_lines(record.get('text') or '', 40))}</div>"
            )
        else:
            text = record.get("text") or record.get("command") or ""
            blocks.append(
                f'<div style="opacity:.45;margin-top:4px;">{html.escape(kind)}: '
                f"{html.escape(clip_lines(str(text), 4))}</div>"
            )
    return (
        f'<div style="max-height:{max_height}px;overflow:auto;color:inherit;'
        'font-size:10px;">' + "".join(blocks) + "</div>"
    )


# ---------------------------------------------------------------------------
# bigym-agent subprocesses
# ---------------------------------------------------------------------------


class AgentJob:
    """One ``bigym-agent`` subcommand run for the viser panels.

    The subprocess writes its output to a log file under the cell, so the
    viewer's main loop never blocks on a pipe; progress is read instead from
    the files the subcommand produces (replay npz count, evaluation CSV
    rows). Start/poll are main-loop operations: GUI callbacks only request.

    The same class serves the Policy versions panel (one cell, the cell on
    screen) and compare mode, which fills a missing seed in any cell of any
    session — the cell is a constructor argument, so nothing about the job
    is tied to the rollout being viewed.
    """

    def __init__(
        self,
        kind: str,
        cell: Path,
        version: int,
        seeds: str = "",
        policy: str | Path = "",
        extra: tuple[str, ...] = (),
    ):
        """Describe a run of one policy version, or of a loose policy file.

        Args:
            kind: ``replay`` or ``evaluate``.
            cell: The cell directory. Any cell: compare mode passes the cell
                of the slot that is missing an episode.
            version: The policy version, ignored when ``policy`` is given.
            seeds: Seed specification for ``replay``.
            policy: A policy file to replay instead of a recorded version;
                its batch lands in ``<cell>/replays/policy_<sha8>``.
            extra: Further command-line arguments (compare mode passes
                ``--no-video``: it needs the batch, not the mp4).
        """
        self.kind = str(kind)
        self.cell = Path(cell)
        self.version = int(version)
        self.seeds = str(seeds or "")
        self.policy = str(policy or "")
        self.extra = tuple(extra)
        self.digest = file_digest(Path(self.policy)) if self.policy else ""
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"policy_{self.digest[:8]}" if self.policy else f"v{self.version:03d}"
        self.log_path = self.cell / JOB_LOG_DIR / f"{self.kind}_{name}_{stamp}.log"
        self.proc: subprocess.Popen | None = None
        self.error: str | None = None

    @property
    def label(self) -> str:
        """What this job is called in the panel and the log lines."""
        return f"policy_{self.digest[:8]}" if self.policy else f"v{self.version:03d}"

    @property
    def command(self) -> list[str]:
        """The argv this job runs (the ``bigym-agent`` module entry point)."""
        cmd = [sys.executable, "-m", AGENT_CLI_MODULE, self.kind, str(self.cell)]
        if self.policy:
            cmd += ["--policy", self.policy]
        else:
            cmd += ["--version", str(self.version)]
        if self.kind == "replay" and self.seeds:
            cmd += ["--seeds", self.seeds]
        return cmd + list(self.extra)

    @property
    def out_dir(self) -> Path:
        """Where the subcommand writes this job's outputs."""
        kind_dir = "replays" if self.kind == "replay" else "eval"
        if self.policy:
            # Matches bigym-agent replay --policy: the file's content hash
            # names the directory, so replaying the same file twice reuses it.
            return self.cell / kind_dir / f"policy_{self.digest[:8]}"
        return version_dir(self.cell, self.version, kind_dir)

    def start(self) -> None:
        """Launch the subprocess; a launch failure lands in ``error``."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            log = self.log_path.open("w")
        except OSError as exc:
            self.error = f"cannot write {self.log_path}: {exc}"
            return
        try:
            self.proc = subprocess.Popen(
                self.command, stdout=log, stderr=subprocess.STDOUT
            )
        except OSError as exc:
            log.close()
            self.error = f"cannot run {' '.join(self.command)}: {exc}"

    @property
    def episodes(self) -> int:
        """How many episodes an evaluation was asked for (``--episodes``)."""
        argv = self.command
        for i, arg in enumerate(argv):
            if arg == "--episodes" and i + 1 < len(argv):
                return as_int(argv[i + 1]) or DEFAULT_EVAL_EPISODES
            if arg.startswith("--episodes="):
                return as_int(arg.split("=", 1)[1]) or DEFAULT_EVAL_EPISODES
        return DEFAULT_EVAL_EPISODES

    def csv_rows(self) -> int:
        """Episodes already scored, counted in ``episodes.csv``."""
        episodes_csv = self.out_dir / "episodes.csv"
        if not episodes_csv.exists():
            return 0
        try:
            with episodes_csv.open() as f:
                return max(0, sum(1 for _ in f) - 1)
        except OSError:
            return 0

    def progress(self) -> str:
        """One line of progress, polled from the output directory."""
        if self.kind == "replay":
            written = len(episode_files(self.out_dir / "batch"))
            return f"replay {self.label}: {written} episodes written"
        return f"evaluate {self.label}: {self.csv_rows()} episodes scored"

    def progress_state(self) -> tuple[float, str, bool]:
        """``(fraction, label, animated)`` for this job's progress bar.

        Prefers the ``.progress.json`` the subprocess writes (step level for
        a replay, episode level for an evaluation) and falls back to the
        files it has finished: an evaluation's ``episodes.csv`` rows over
        ``--episodes``, a replay's episode count. Before either exists the
        bar is animated and says ``starting…``, because a rollout can take a
        minute to build its environment and a bar stuck at zero with no
        motion reads as a hang.

        Returns:
            The fraction in ``[0, 1]``, the label line, and whether the bar
            should animate (nothing is known yet).
        """
        state = read_progress(self.out_dir)
        head = f"{self.kind} {self.label}"
        if state:
            return progress_fraction(state), progress_label(head, state), False
        if self.kind == "evaluate":
            rows, total = self.csv_rows(), max(1, self.episodes)
            if rows:
                return (
                    min(1.0, rows / total),
                    f"{head} · {rows}/{total} episodes",
                    False,
                )
        return 0.0, f"{head} · starting…", True

    def poll(self) -> bool:
        """True once the job is over; a non-zero exit sets ``error``."""
        if self.error is not None or self.proc is None:
            return True
        code = self.proc.poll()
        if code is None:
            return False
        if code != 0:
            self.error = self.diagnose(code)
        return True

    def stop(self) -> None:
        """Kill the subprocess: nobody is waiting for its output any more.

        Pressing Leave while compare mode is still producing its rollouts
        must not leave a replay running for another twenty minutes, writing
        into a cell the viewer has stopped looking at.
        """
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        with contextlib.suppress(OSError):
            proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                proc.wait(timeout=5.0)

    def diagnose(self, code: int) -> str:
        """Turn a non-zero exit into a message the panel can show."""
        tail = ""
        try:
            tail = self.log_path.read_text(errors="replace")[-4000:]
        except OSError:
            pass
        # Only the dispatcher's own refusals mean "no such subcommand"; a
        # missing module inside the agent's policy is a job failure and is
        # reported with its own last line below.
        unknown = any(
            marker in tail
            for marker in (
                f"No module named '{AGENT_CLI_MODULE}'",
                "No module named 'bigym'",
                "cannot be directly executed",
                "Unrecognized options",
                "invalid choice",
                "unknown command",
                "unknown subcommand",
                "unrecognized arguments",
            )
        )
        if unknown:
            return (
                f"this install has no working 'bigym-agent {self.kind}' "
                f"subcommand ({AGENT_CLI_MODULE} refused it); install "
                "bigym[agent] and try again"
            )
        lines = [line for line in tail.splitlines() if line.strip()]
        detail = lines[-1] if lines else f"see {self.log_path}"
        return f"{self.kind} failed (exit {code}): {detail}"


# ---------------------------------------------------------------------------
# Compare: N policies of the same task, side by side on one seed
# ---------------------------------------------------------------------------


def session_prefix(sessions) -> str:
    """The name part every session shares, cut at a separator.

    A benchmark run names its sessions ``astra_high_s1``, ``astra_high_s2``,
    ``astra_high_s3``; in a compare of those three, ``astra_high_`` is on
    every label and tells the reader nothing. The raw common prefix would be
    ``astra_high_s`` and would leave ``1``/``2``/``3``, so it is cut back to
    the last separator — the result is ``s1``, ``s2``, ``s3``.

    Args:
        sessions: The session names being compared.

    Returns:
        The prefix to drop, or "" when there is nothing worth dropping (one
        session, no shared start, or a prefix that would empty a name).
    """
    names = [str(name) for name in sessions]
    if len(set(names)) < 2:
        return ""
    shared = names[0]
    for name in names[1:]:
        while not name.startswith(shared):
            shared = shared[:-1]
            if not shared:
                return ""
    cut = max(shared.rfind(sep) for sep in NAME_SEPARATORS)
    prefix = shared[: cut + 1] if cut >= 0 else ""
    if not prefix or any(name == prefix for name in names):
        return ""
    return prefix


def template_policy_path(cell: Path) -> Path:
    """Where a cell keeps the policy its agent started from."""
    return Path(cell) / TEMPLATE_POLICY


def is_template_version(cell: Path, row: dict) -> bool:
    """True when this version's ``policy.py`` is still the untouched template.

    Every session's v001 is a snapshot taken before the agent wrote
    anything, so a compare of three v001s is a compare of one policy with
    itself; saying so on the label costs nothing and explains three
    identical robots.

    Args:
        cell: The cell directory.
        row: One row of :func:`policy_rows`.

    Returns:
        True when ``policies/vNNN/policy.py`` matches ``initial_policy.py``
        byte for byte.
    """
    template = template_policy_path(cell)
    try:
        return template.read_bytes() == policy_file(cell, row).read_bytes()
    except OSError:
        return False


def layout_columns(layout: str, count: int) -> int:
    """How many columns a layout puts ``count`` slots in.

    Args:
        layout: One of :data:`COMPARE_LAYOUTS` (anything else reads as
            ``auto``).
        count: How many slots there are.

    Returns:
        The number of columns, at least one.
    """
    count = max(1, int(count))
    name = str(layout or LAYOUT_AUTO).strip().lower()
    if name == LAYOUT_ROW:
        return count
    digits = name.split(" ", 1)[0]
    if digits.isdigit() and int(digits) > 0:
        return min(count, int(digits))
    return count if count <= AUTO_ROW_MAX else 2


def slot_grid(count: int, layout: str = LAYOUT_AUTO) -> list[tuple[int, int]]:
    """The ``(row, column)`` of every slot, filling each row left to right."""
    columns = layout_columns(layout, count)
    return [(i // columns, i % columns) for i in range(max(0, int(count)))]


def slot_offset(
    row: int, column: int, gap: float, gap_x: float | None = None
) -> tuple[float, float, float]:
    """Where the scene of the slot at ``(row, column)`` is moved to.

    A column steps along +y, so one row reads left to right exactly as it
    always did; a row steps along -x, so the second row stands behind the
    first rather than continuing the line off the side of the screen.

    Args:
        row: Its row index.
        column: Its column index.
        gap: Metres between neighbouring columns.
        gap_x: Metres between rows, or None for ``gap``.

    Returns:
        The ``(x, y, z)`` offset of that slot's frames.
    """
    depth = float(gap if gap_x is None else gap_x)
    # ``+ 0.0`` so row 0 reads as 0.0 rather than -0.0 in the logs.
    return (-depth * int(row) + 0.0, float(gap) * int(column), 0.0)


def grid_offsets(
    count: int, gap: float, layout: str = LAYOUT_AUTO
) -> list[tuple[float, float, float]]:
    """One offset per slot, in slot order."""
    return [slot_offset(r, c, gap) for r, c in slot_grid(count, layout)]


def grid_shape(count: int, layout: str = LAYOUT_AUTO) -> tuple[int, int]:
    """``(rows, columns)`` of the grid a layout puts ``count`` slots in."""
    cells = slot_grid(count, layout)
    if not cells:
        return (0, 0)
    return (cells[-1][0] + 1, layout_columns(layout, count))


def grid_note(
    count: int, gap: float, layout: str = LAYOUT_AUTO, noun: str = "slot"
) -> str:
    """``4 slots as 2 columns x 2 rows, gap 2.5 m`` for the log and header."""
    rows, columns = grid_shape(count, layout)
    return (
        f"{count} {noun}{'' if count == 1 else 's'} as {columns} column"
        f"{'' if columns == 1 else 's'} x {rows} row{'' if rows == 1 else 's'}"
        f", gap {gap:g} m"
    )


def compare_camera_pose(
    offsets, height: float = 1.6, distance: float = 2.8
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Where a client's camera starts so the whole grid is in view.

    The camera stands on the -x side, which is the side the single-rollout
    viewer watches a robot from, far enough out to hold the width of the
    grid (+y) and — once there is more than one row — high enough to look
    over the heads of the nearer row at the one behind it. Row 0 is the
    far row: rows recede along -x, towards the camera's own side, so the
    grid is read from the front of the last row backwards.

    Args:
        offsets: Every slot's offset, from :func:`grid_offsets`.
        height: Eye height above the floor for a single row.
        distance: How far out a single slot is watched from.

    Returns:
        ``(position, look_at)`` in world metres.
    """
    places = [tuple(float(v) for v in o) for o in offsets] or [(0.0, 0.0, 0.0)]
    xs = [o[0] for o in places]
    ys = [o[1] for o in places]
    span = max(ys) - min(ys)
    depth = max(xs) - min(xs)
    mid_y = 0.5 * (min(ys) + max(ys))
    # 0.8 m of eye height per metre of depth clears a 1.7 m robot standing
    # in the near row on the line of sight to the far one.
    return (
        (min(xs) - distance - 0.7 * span, mid_y, height + 0.8 * depth),
        (max(xs) - 0.5 * depth, mid_y, 0.8),
    )


def focus_pose(
    offset, pelvis, distance: float = 2.0, height: float = 1.35
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Where a camera stands to watch one slot on its own.

    In front of that robot (the -x side, the face every other view uses),
    looking at its pelvis a little above the hips.

    Args:
        offset: The slot's grid offset.
        pelvis: Its pelvis position inside its own model.
        distance: Metres in front of the pelvis.
        height: Eye height above the floor.

    Returns:
        ``(position, look_at)`` in world metres.
    """
    anchor = (
        float(pelvis[0]) + float(offset[0]),
        float(pelvis[1]) + float(offset[1]),
        float(pelvis[2]),
    )
    return (
        (anchor[0] - float(distance), anchor[1], float(height)),
        (anchor[0], anchor[1], anchor[2] + 0.15),
    )


def fly_clients_to(server, position, look_at) -> int:
    """Point every connected client's camera at one pose (main loop only).

    Args:
        server: The viser server.
        position: Where the camera goes.
        look_at: What it looks at.

    Returns:
        How many clients were moved.
    """
    moved = 0
    try:
        clients = list(server.get_clients().values())
    except Exception:  # a disconnect mid-iteration is not an error
        return 0
    for client in clients:
        with contextlib.suppress(Exception):
            client.camera.position = tuple(float(v) for v in position)
            client.camera.look_at = tuple(float(v) for v in look_at)
            moved += 1
    return moved


@dataclass
class CompareSlot:
    """One environment of the compare scene."""

    session: str
    task: str
    cell: Path
    version: int | None = None
    policy: str = ""
    episode: Path | None = None
    source: str = ""
    train_success: float | None = None
    eval_success: float | None = None
    submission: bool = False
    template: bool = False
    prefix: str = ""

    @property
    def name(self) -> str:
        """The session as the labels spell it, with the shared prefix gone."""
        text = str(self.session)
        if self.prefix and text.startswith(self.prefix) and text != self.prefix:
            return text[len(self.prefix) :]
        return text

    @property
    def version_name(self) -> str:
        """``v007`` for a recorded version, ``policy_<sha8>`` for a file."""
        if self.policy:
            return f"policy_{file_digest(Path(self.policy))[:8]}"
        return "—" if self.version is None else f"v{self.version:03d}"

    @property
    def label(self) -> str:
        """The floating label above this slot's pelvis.

        ``s2 · v007 · submission · train 40% · eval 2%`` — who ran, which
        policy, and the two scores the panels already show, so the scene
        alone answers "which of these is the good one". The session drops
        the prefix every slot shares, a v001 that is still the untouched
        template says so, and a score that was never measured is named
        (``no dev episodes``, ``not evaluated``) rather than dashed out.
        """
        version = self.version_name
        if self.template:
            version = f"{version} {TEMPLATE_MARK}"
        parts = [self.name, version]
        if self.submission:
            parts.append("submission")
        train = percent(self.train_success)
        parts.append(f"train {train}" if train else NO_TRAIN)
        rate = percent(self.eval_success)
        parts.append(f"eval {rate}" if rate else NO_EVAL)
        return " · ".join(parts)


@dataclass
class CompareRequest:
    """What the Compare button (or ``--compare``) asked for."""

    slots: list[CompareSlot]
    seed: int
    gap: float = COMPARE_GAP
    layout: str = LAYOUT_AUTO
    prefix: str = ""
    back_to: int = 0
    error: str = ""
    task: str = ""
    missing: list[CompareSlot] = field(default_factory=list)

    def offsets(self) -> list[tuple[float, float, float]]:
        """Where each slot's scene is placed, in slot order."""
        return grid_offsets(len(self.slots), self.gap, self.layout)


def seed_episode(
    cell: Path, version: int | None, seed: int, policy: str = ""
) -> Path | None:
    """The stored episode of one version on one seed, when there is one.

    Evaluation batches hold every hidden seed (620000-620099), so comparing
    two submissions on a hidden seed needs no new rollout at all; replays
    and development batches are searched after them.

    Args:
        cell: The cell directory.
        version: The policy version, or None for ``policy``.
        seed: The seed to find.
        policy: A loose policy file, whose batches live under
            ``replays/policy_<sha8>/``.

    Returns:
        The ``.npz`` path, or None when nothing has rolled this seed out.
    """
    cell = Path(cell)
    candidates: list[Path] = []
    if policy:
        digest = file_digest(Path(policy))[:8]
        candidates.append(cell / "replays" / f"policy_{digest}" / "batch")
    elif version is not None:
        candidates.append(version_dir(cell, version, "eval") / "batch")
        candidates.append(version_dir(cell, version, "replays") / "batch")
        candidates += sorted(cell.glob(f"dev/v{version:03d}_*/batch"))
    for batch in candidates:
        for path in episode_files(batch):
            if episode_info(path)["seed"] == int(seed):
                return path.resolve()
    return None


def episode_source(cell: Path, episode: Path) -> str:
    """``eval/v007`` — where an episode file came from, for the log."""
    try:
        rel = Path(episode).resolve().relative_to(Path(cell).resolve())
    except ValueError:
        return str(episode)
    return str(rel.parent.parent)


def fill_slot(slot: CompareSlot, seed: int) -> CompareSlot:
    """Attach the stored episode, the version's scores and its provenance."""
    rows = policy_rows(slot.cell)
    if slot.version is not None:
        for row in rows:
            if row["version"] == slot.version:
                slot.train_success = as_float(row.get("train_success"))
                slot.eval_success = as_float(row.get("eval_success"))
                slot.template = is_template_version(slot.cell, row)
        slot.submission = slot.version == submission_version(slot.cell, rows)
    slot.episode = seed_episode(slot.cell, slot.version, seed, slot.policy)
    slot.source = episode_source(slot.cell, slot.episode) if slot.episode else ""
    return slot


def resolve_version(cell: Path, text: str) -> int | None:
    """Turn ``v008`` / ``8`` / ``last`` / ``submission`` into a version number."""
    rows = policy_rows(cell)
    name = str(text or "").strip().lower()
    if not name or name in (VERSION_LAST, "submission"):
        return rows[-1]["version"] if rows else None
    digits = name[1:] if name.startswith("v") else name
    if digits.isdigit():
        return int(digits)
    return None


def parse_compare(spec: str) -> list[tuple[str, str]]:
    """Parse ``--compare "s1:v008,s2:v007,s3:last"`` into (session, version).

    Args:
        spec: A comma-separated list of ``<session>:<version>`` slots; the
            version may be ``vNNN``, ``NNN``, ``last`` or ``submission``.

    Returns:
        One ``(session, version)`` pair per slot, in the order given.

    Raises:
        ValueError: when a slot is not ``session:version``.
    """
    slots = []
    for chunk in str(spec).split(","):
        text = chunk.strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError(
                f"--compare slot {text!r} is not '<session>:<version>' "
                "(versions: vNNN, NNN, last, submission)"
            )
        session, version = text.split(":", 1)
        slots.append((session.strip(), version.strip()))
    if not slots:
        raise ValueError("--compare needs at least one '<session>:<version>' slot")
    return slots[:COMPARE_SLOTS]


def match_session(sessions: list[str], text: str) -> str | None:
    """Resolve a session name the user typed: exact, then unique substring."""
    name = str(text).strip()
    if name in sessions:
        return name
    matches = [s for s in sessions if name and name in s]
    return matches[0] if len(matches) == 1 else None


def resolve_compare(
    cells: list[Cell], task: str, specs: list[tuple[str, str]], seed: int
) -> CompareRequest:
    """Turn ``--compare`` / the panel's choices into a filled request.

    Args:
        cells: Every discovered cell.
        task: The task to compare on (compare is one task at a time).
        specs: ``(session, version)`` pairs.
        seed: The seed every slot is posed on.

    Returns:
        The request; ``error`` is set (and ``slots`` may be short) when a
        session or version could not be resolved, and ``missing`` lists the
        slots whose episode has to be produced by ``bigym-agent replay``.
    """
    sessions = sessions_of(cells, task)
    slots: list[CompareSlot] = []
    problems: list[str] = []
    for session_text, version_text in specs:
        session = match_session(sessions, session_text)
        if session is None:
            problems.append(
                f"no session matches {session_text!r} for task {task} "
                f"(have: {', '.join(sessions) or 'none'})"
            )
            continue
        cell = find_cell(cells, task, session)
        if cell is None:
            problems.append(f"session {session} has no cell for task {task}")
            continue
        version = resolve_version(cell.path, version_text)
        if version is None:
            problems.append(f"{session}: no version matches {version_text!r}")
            continue
        slots.append(
            fill_slot(
                CompareSlot(
                    session=session, task=task, cell=cell.path, version=version
                ),
                seed,
            )
        )
    prefix = session_prefix([slot.session for slot in slots])
    for slot in slots:
        slot.prefix = prefix
    return CompareRequest(
        slots=slots,
        seed=int(seed),
        task=task,
        prefix=prefix,
        error="; ".join(problems),
        missing=[s for s in slots if s.episode is None],
    )


def missing_job(slot: CompareSlot, seed: int) -> AgentJob:
    """The ``bigym-agent replay --no-video`` job one empty slot needs."""
    if slot.policy:
        return AgentJob(
            "replay",
            slot.cell,
            0,
            str(seed),
            policy=slot.policy,
            extra=("--no-video",),
        )
    return AgentJob(
        "replay",
        slot.cell,
        int(slot.version or 0),
        str(seed),
        extra=("--no-video",),
    )


class MissingWork:
    """The rollouts a compare request is still waiting for, as a state machine.

    Every slot with no stored episode on the compare seed becomes a
    ``bigym-agent replay <cell> --version N --seeds <seed> --no-video``
    subprocess. They all start at once (each is its own process with its own
    environment, and physics is CPU-bound), and two slots naming the same
    policy share one job, so no episode file is written twice.

    Waiting is the caller's business, one :meth:`poll` per main-loop tick,
    so the GUI keeps its progress bars and Leave button while jobs run.
    :meth:`bars` is what the Compare folder draws.
    """

    def __init__(self, request: CompareRequest, timeout: float = 3600.0):
        """Plan the work for one request (nothing starts yet).

        Args:
            request: The request whose ``missing`` slots need an episode.
            timeout: Give up on a job that takes longer than this.
        """
        self.request = request
        self.timeout = float(timeout)
        self.jobs: list[AgentJob] = []
        self.heads: list[str] = []
        self.pending: set[int] = set()
        self.started = 0.0
        self.error = ""
        self.done = False

    def start(self) -> list[AgentJob]:
        """Launch one job per distinct missing policy (main loop only).

        Returns:
            The jobs started, in slot order; empty when every slot already
            has its episode, in which case the work is done at once.
        """
        seen: set[tuple] = set()
        for slot in list(self.request.missing):
            key = (str(slot.cell), slot.policy or int(slot.version or 0))
            if key in seen:
                continue
            seen.add(key)
            job = missing_job(slot, self.request.seed)
            self.jobs.append(job)
            self.heads.append(
                f"{slot.name} · {slot.version_name} · seed {self.request.seed}"
            )
            print(f"[viewer] {' '.join(job.command)} > {job.log_path}", flush=True)
            job.start()
        self.pending = set(range(len(self.jobs)))
        self.started = time.time()
        if self.jobs:
            print(
                f"[viewer] compare: {len(self.jobs)} replay job(s) running in parallel",
                flush=True,
            )
        else:
            self.resolve()
        return list(self.jobs)

    def poll(self) -> bool:
        """Poll every running job; True once there is nothing left to wait for.

        A job that fails ends the work: the others are killed and
        :attr:`error` says what went wrong.
        """
        if self.done:
            return True
        for index in sorted(self.pending):
            job = self.jobs[index]
            if not job.poll():
                continue
            self.pending.discard(index)
            if job.error:
                self.error = job.error
                self.stop()
                self.done = True
                return True
        if self.pending:
            if time.time() - self.started > self.timeout:
                names = ", ".join(self.jobs[i].label for i in sorted(self.pending))
                self.error = f"replay timed out for: {names}"
                self.stop()
                self.done = True
            return self.done
        self.resolve()
        return True

    def resolve(self) -> None:
        """Re-read every slot now the jobs are over.

        The scene is built only once every file exists, so this re-runs the
        whole resolution (not just the slots that had jobs) and sets
        :attr:`error` when something is still not there.
        """
        for slot in self.request.slots:
            fill_slot(slot, self.request.seed)
        self.request.missing = [s for s in self.request.slots if s.episode is None]
        if self.request.missing:
            names = ", ".join(
                f"{s.name} {s.version_name}" for s in self.request.missing
            )
            self.error = f"still no episode on seed {self.request.seed} for: {names}"
        self.done = True

    def stop(self) -> None:
        """Kill every job still running (the Leave button, or a failure)."""
        for index in sorted(self.pending):
            self.jobs[index].stop()
        self.pending.clear()

    def bars(self) -> list[tuple[float, str, bool]]:
        """One ``(fraction, label, animated)`` per job, in the order started."""
        out: list[tuple[float, str, bool]] = []
        for index, job in enumerate(self.jobs):
            if index not in self.pending:
                out.append((1.0, f"{self.heads[index]} · done", False))
                continue
            state = read_progress(job.out_dir)
            out.append(
                (
                    progress_fraction(state),
                    progress_label(self.heads[index], state),
                    state is None,
                )
            )
        return out


def compare_status_html(request: CompareRequest) -> str:
    """What the Compare panel says about the slots it is about to build."""
    if request.error:
        return f'<div style="font-size:10px;color:#dc2626;">{html.escape(request.error)}</div>'
    if not request.slots:
        return '<div style="font-size:11px;opacity:.6;">pick at least one slot</div>'
    rows = []
    for i, slot in enumerate(request.slots):
        where = slot.source or "needs a replay"
        colour = "opacity:.6;" if slot.episode else "color:#f97316;"
        rows.append(
            f'<div style="font-size:10px;font-family:ui-monospace,monospace;">'
            f"slot {i} {html.escape(slot.name)} {html.escape(slot.version_name)} "
            f'<span style="{colour}">{html.escape(where)}</span></div>'
        )
    return '<div style="color:inherit;">' + "".join(rows) + "</div>"


# ---------------------------------------------------------------------------
# Compare scene construction
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def prefixed_body_names(prefix: str):
    """Make mjviser name this scene's nodes ``/bodies/<prefix>/<body>/...``.

    mjviser hard-codes the viser node paths it builds (``/bodies/<body>`` and
    ``/fixed_bodies/<body>``), so N scenes on one server would collide. Every
    path goes through ``scene.get_body_name``, so prefixing that with a path
    segment gives each slot its own subtree — which is also how the slot is
    moved: a frame at ``/bodies/<prefix>`` offsets everything below it.

    Args:
        prefix: The path segment for this slot.

    Yields:
        None, with the patch in place for the duration of the block.
    """
    original = mjscene.get_body_name

    def named(mj_model, body_id) -> str:
        return f"{prefix}/{original(mj_model, body_id)}"

    mjscene.get_body_name = named  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        mjscene.get_body_name = original


@contextlib.contextmanager
def slot_scene_build(server, prefix: str, *, keep_grid: bool, fixed_frame):
    """Everything one compare slot needs mjviser to do differently.

    Three patches, live only while ``ViserMujocoScene`` is constructing:

    * node names get the slot's prefix (see :func:`prefixed_body_names`);
    * the ``/fixed_bodies`` root frame is created once by the caller and
      handed to every slot — mjviser adds it per scene, and adding a viser
      node twice at the same path removes the first one, taking the
      previous slot's static geometry with it;
    * the ground grid is captured so only the first slot keeps one (every
      slot draws its own infinite plane at z = 0, and coplanar copies
      z-fight).

    Args:
        server: The viser server.
        prefix: This slot's path segment.
        keep_grid: True for the slot whose ground plane stays visible.
        fixed_frame: The shared ``/fixed_bodies`` frame handle.

    Yields:
        None, with the patches in place for the duration of the block.
    """
    api = type(server.scene)
    original_grid = api.add_grid
    original_frame = api.add_frame
    created = []

    def add_grid(self, name, *args, **kwargs):
        handle = original_grid(self, name, *args, **kwargs)
        created.append(handle)
        return handle

    def add_frame(self, name, *args, **kwargs):
        if str(name) == "/fixed_bodies":
            return fixed_frame
        return original_frame(self, name, *args, **kwargs)

    api.add_grid = add_grid
    api.add_frame = add_frame
    try:
        with prefixed_body_names(prefix):
            yield
    finally:
        api.add_grid = original_grid
        api.add_frame = original_frame
        if not keep_grid:
            for handle in created:
                handle.visible = False


def tint_slot_robot(model, colour: tuple[float, float, float]) -> int:
    """Tint one slot's robot so the copies read apart (display only).

    Multiplies the robot's geom rgb by ``colour`` in this process's copy of
    the model, exactly as the viewer's ``--robot-tint`` does: nothing is
    written to disk and no observation changes (compare mode replays stored
    ``full_qpos``, it does not step a policy).

    Args:
        model: The slot's ``mujoco.MjModel``.
        colour: Per-channel multiplier.

    Returns:
        The number of geoms tinted.
    """
    roots = set()
    for body in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        if name.endswith("/pelvis"):
            roots.add(int(model.body_rootid[body]))
    if not roots:
        return 0
    mask = np.isin(model.body_rootid[model.geom_bodyid], list(roots))
    model.geom_rgba[mask, :3] = np.clip(
        model.geom_rgba[mask, :3] * np.asarray(colour, dtype=float), 0.0, 1.0
    )
    return int(mask.sum())


def apply_lighting(server, prefs: dict) -> None:
    """Apply the viewer's lighting preferences to a freshly built scene."""
    sky = str(prefs.get("sky", "black"))
    try:
        server.scene.configure_environment_map(
            environment_intensity=float(prefs.get("environment", 0.7)),
            background=sky != "transparent",
            background_blurriness=1.0,
            background_intensity=0.5 if sky == "grey" else 0.0,
        )
        server.scene.configure_default_lights(
            enabled=bool(prefs.get("default_lights", True)),
            cast_shadow=bool(prefs.get("shadows", True)),
        )
    except Exception as exc:  # lighting is cosmetic
        print(f"[viewer] lighting unavailable: {exc}", flush=True)


def pelvis_body_id(model) -> int:
    """The id of the robot's pelvis body, or -1 when the model has none.

    Args:
        model: A ``mujoco.MjModel``.

    Returns:
        The body id whose name ends in ``/pelvis``, else -1.
    """
    for body in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        if name.endswith("/pelvis"):
            return int(body)
    return -1


def pelvis_position(data, body: int) -> tuple[float, float, float]:
    """The pelvis's world position after ``mj_forward``.

    Reading ``data.xpos`` rather than decoding qpos means the anchor is
    right whatever the robot's root joint looks like, and it is already the
    position mjviser drew the body at.

    Args:
        data: A ``mujoco.MjData`` that has been forwarded.
        body: The pelvis body id, or -1.

    Returns:
        ``(x, y, z)``; ``(0, 0, 0.8)`` when there is no pelvis body.
    """
    if body < 0:
        return (0.0, 0.0, 0.8)
    x, y, z = (float(v) for v in data.xpos[body])
    return (x, y, z)


def billboard_anchor(
    pelvis, offset, z_offset: float = BILLBOARD_Z
) -> tuple[float, float, float]:
    """Where one slot's billboard hangs in the compare scene.

    Every slot's environment is modelled at the origin and moved into place
    by a frame that carries its grid offset, so a pelvis position read out
    of that slot's ``MjData`` has to be offset the same way before it can be
    used as a world anchor for a label, which hangs off the scene root.

    Args:
        pelvis: The pelvis world position inside the slot's own model.
        offset: The slot's grid offset (see :func:`slot_offset`).
        z_offset: How far above the pelvis the billboard hangs.

    Returns:
        The pelvis plus the slot's offset plus ``z_offset`` in z.
    """
    return (
        float(pelvis[0]) + float(offset[0]),
        float(pelvis[1]) + float(offset[1]),
        float(pelvis[2]) + float(offset[2]) + float(z_offset),
    )


def slot_colour(index: int) -> str:
    """The CSS colour of slot ``index`` (the palette repeats)."""
    return SLOT_COLOURS[int(index) % len(SLOT_COLOURS)]


def slot_outcome(info: dict) -> str:
    """``✓`` / ``✗ · timeout · fell`` for the label under a finished slot."""
    parts = []
    success = info.get("success")
    if success is not None:
        parts.append("✓" if float(success) > 0 else "✗")
    end = str(info.get("termination") or "")
    if end and end != "success":
        parts.append(end)
    if info.get("fell"):
        parts.append("fell")
    return " · ".join(parts)


def css_rgb(colour: str) -> tuple[int, int, int]:
    """``#73aeff`` as ``(115, 174, 255)``; grey when it is not a hex colour."""
    text = str(colour).strip().lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    try:
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    except (ValueError, IndexError):
        return (160, 160, 160)


def quaternion_from_axes(x, y, z) -> tuple[float, float, float, float]:
    """The ``(w, x, y, z)`` of the rotation whose axes are the three given.

    Args:
        x: Where the local +x axis points, in world coordinates.
        y: Where the local +y axis points.
        z: Where the local +z axis points.

    Returns:
        The quaternion, with a non-negative ``w`` so the same rotation
        always reads the same way.
    """
    m = np.column_stack([np.asarray(v, dtype=float).reshape(3) for v in (x, y, z)])
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = [
            0.25 * scale,
            (m[2, 1] - m[1, 2]) / scale,
            (m[0, 2] - m[2, 0]) / scale,
            (m[1, 0] - m[0, 1]) / scale,
        ]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        scale = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        quat = [
            (m[2, 1] - m[1, 2]) / scale,
            0.25 * scale,
            (m[0, 1] + m[1, 0]) / scale,
            (m[0, 2] + m[2, 0]) / scale,
        ]
    elif m[1, 1] > m[2, 2]:
        scale = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        quat = [
            (m[0, 2] - m[2, 0]) / scale,
            (m[0, 1] + m[1, 0]) / scale,
            0.25 * scale,
            (m[1, 2] + m[2, 1]) / scale,
        ]
    else:
        scale = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        quat = [
            (m[1, 0] - m[0, 1]) / scale,
            (m[0, 2] + m[2, 0]) / scale,
            (m[1, 2] + m[2, 1]) / scale,
            0.25 * scale,
        ]
    if quat[0] < 0.0:
        quat = [-v for v in quat]
    return cast(tuple[float, float, float, float], tuple(float(v) for v in quat))


def plate_wxyz(face=PLATE_FACE, up=PLATE_UP) -> tuple[float, float, float, float]:
    """The orientation that shows an image plane's picture along ``face``.

    viser's image node is a plane that the client pre-rotates by half a turn
    about its x axis: the picture is on the side its local -z points to and
    the top of the picture is along its local -y (viewing the plane down its
    local +z shows the picture mirrored; keeping local +y up shows it upside
    down). So the local -z is sent along ``face``, from the plate towards
    whoever reads it, and the local -y is sent along ``up``. Looking along -x
    with +z up is what makes the billboards readable from the side the
    compare camera starts on.

    Args:
        face: The world direction the picture looks along (plate to reader).
        up: Which way is up on the picture.

    Returns:
        The ``(w, x, y, z)`` for ``scene.add_image``.
    """
    normal = np.asarray(face, dtype=float)
    normal = normal / np.linalg.norm(normal)
    vertical = np.asarray(up, dtype=float)
    vertical = vertical - normal * float(vertical @ normal)
    vertical = vertical / np.linalg.norm(vertical)
    local_z = -normal
    local_y = -vertical
    return quaternion_from_axes(np.cross(local_y, local_z), local_y, local_z)


def plate_font(size: int):
    """A face for the billboard text, DejaVu when the system has one."""
    for path in PLATE_FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:  # Pillow >= 10.1 can scale its own bundled face
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def plate_status(status: str) -> str:
    """The part of a slot's status that is worth drawing on its plate.

    The step counter changes every frame and the plate is an image, which
    has to be re-drawn and re-encoded to say anything new; the outcome
    changes once. The counter stays in the sidebar legend, which is text
    and costs nothing to rewrite.

    Args:
        status: A line from :func:`slot_status_text`.

    Returns:
        The same line with the ``step a/b`` part removed.
    """
    parts = [p for p in str(status).split(" · ") if not p.startswith("step ")]
    return " · ".join(parts)


def plate_image(
    label: str,
    status: str,
    colour: str,
    *,
    size: tuple[int, int] = PLATE_SIZE,
    font_size: int = PLATE_FONT_SIZE,
):
    """Draw one billboard: white text on an opaque dark plate.

    The plate is laid out on a ``size`` canvas (the text is drawn smaller
    when it would not fit that width) and the image returned is cropped to
    the plate itself, so the only transparent pixels are the rounded
    corners. That matters in the scene: a transparent pixel still writes
    depth, so a wide clear margin around the plate blanks out whatever is
    drawn after it and behind it — the semi-transparent floor grid, when
    the plate is farther from the camera than the grid's origin — into a
    black box. :func:`plate_extent` sizes the image node to match.

    Args:
        label: The slot title (session, version, scores).
        status: The line under it, or "" for a single-line plate.
        colour: The slot's CSS colour, drawn as a stripe down the side.
        size: ``(width, height)`` of the layout canvas in pixels.
        font_size: Text size in pixels.

    Returns:
        An ``(h, w, 4)`` uint8 RGBA array no larger than ``size``; a 2x2
        transparent one when there is nothing to say.
    """
    width, height = int(size[0]), int(size[1])
    lines = [line for line in (str(label), str(status)) if line]
    if not lines:
        return np.zeros((2, 2, 4), dtype=np.uint8)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    pad_x, pad_y, stripe = 14, 10, 7
    room = width - 2 * pad_x - stripe - 8
    font_px = int(font_size)
    font = plate_font(font_px)
    boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    text_w = max(box[2] - box[0] for box in boxes)
    # A label as long as "s2 · v001 (template) · no dev episodes · not
    # evaluated" is drawn smaller rather than run off the side of the plate.
    for _ in range(3):
        if text_w <= room or font_px <= PLATE_FONT_MIN:
            break
        font_px = max(PLATE_FONT_MIN, int(font_px * room / text_w))
        font = plate_font(font_px)
        boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
        text_w = max(box[2] - box[0] for box in boxes)
    spacing = int(round(0.35 * font_px))
    line_h = max(box[3] - box[1] for box in boxes)
    plate_w = min(width, text_w + 2 * pad_x + stripe + 8)
    plate_h = min(height, len(lines) * line_h + (len(lines) - 1) * spacing + 2 * pad_y)
    left = (width - plate_w) // 2
    top = (height - plate_h) // 2
    draw.rounded_rectangle(
        (left, top, left + plate_w - 1, top + plate_h - 1),
        radius=10,
        fill=(15, 15, 18, 255),
    )
    draw.rectangle(
        (left + 4, top + 4, left + 4 + stripe, top + plate_h - 5),
        fill=css_rgb(colour) + (255,),
    )
    y = top + pad_y
    for index, (line, box) in enumerate(zip(lines, boxes, strict=True)):
        draw.text(
            (left + stripe + pad_x, y - box[1]),
            line,
            font=font,
            fill=(255, 255, 255, 255) if index == 0 else (214, 214, 222, 255),
        )
        y += line_h + spacing
    return np.ascontiguousarray(
        np.asarray(image)[top : top + plate_h, left : left + plate_w]
    )


def plate_extent(image, width: float) -> tuple[float, float]:
    """The metres a plate image should span in the scene.

    ``width`` is what a plate as wide as the whole :data:`PLATE_SIZE`
    canvas would span; a narrower plate (the image is cropped to its text)
    spans proportionally less, so the text is the same size on every plate.

    Args:
        image: The ``(h, w, 4)`` array from :func:`plate_image`.
        width: The full-canvas width in metres.

    Returns:
        ``(width, height)`` in metres.
    """
    shown = float(width) * float(image.shape[1]) / float(PLATE_SIZE[0])
    return shown, shown * float(image.shape[0]) / float(image.shape[1])


def viewer_camera_position(server) -> tuple[float, float, float] | None:
    """The camera position of the most recently connected client, or None."""
    try:
        clients = list(server.get_clients().values())
    except Exception:  # a closed server
        return None
    if not clients:
        return None
    try:
        return cast(
            tuple[float, float, float],
            tuple(float(v) for v in clients[-1].camera.position),
        )
    except Exception:  # camera not reported yet
        return None


def facing_wxyz(plate, camera) -> tuple[float, float, float, float]:
    """Orientation that turns a plate at ``plate`` towards ``camera``.

    The picture looks along the line from the plate to the camera and its
    text keeps world +z up; when the camera is straight above or below, +x
    is used as the up hint instead so the orientation stays defined.

    Args:
        plate: The plate's world position.
        camera: The camera's world position.

    Returns:
        The ``(w, x, y, z)`` quaternion for the image node.
    """
    face = np.asarray(camera, dtype=float) - np.asarray(plate, dtype=float)
    norm = float(np.linalg.norm(face))
    if norm < 1e-6:
        return plate_wxyz()
    face = face / norm
    up = (
        PLATE_UP
        if abs(float(face @ np.asarray(PLATE_UP, dtype=float))) < 0.99
        else (1.0, 0.0, 0.0)
    )
    return plate_wxyz(face=tuple(face), up=up)


def add_plate_image(server, name: str, image, width: float, *, position, visible):
    """Add one billboard plane, without shadows when viser allows it."""
    width, height = plate_extent(image, width)
    common = {
        "format": "png",
        "wxyz": plate_wxyz(),
        "position": tuple(float(v) for v in position),
        "visible": bool(visible),
    }
    return server.scene.add_image(
        name,
        image,
        width,
        height,
        cast_shadow=False,
        receive_shadow=False,
        **common,
    )


class SlotBillboard:
    """The plate of text that hangs above one compare slot's pelvis.

    It is a PIL-drawn RGBA image, not an HTML node. viser 1.1 has no
    ``scene.add_html``; its only HTML-in-the-scene node is
    ``scene.add_3d_gui_container``, whose contents the client wraps in a
    Mantine ``Paper`` styled ``background-color: var(--mantine-color-body)``
    — a white card behind any label drawn in it. The container
    takes ``name``, ``wxyz``, ``position`` and ``visible`` and nothing else,
    and ``gui.configure_theme`` carries no custom CSS, so there is no
    supported way to switch that card off.

    An image is a plane, so the plate is turned towards the viewer by hand:
    every time it is moved (once per playback frame) it reads the camera
    position of the connected client and orients its face along the line
    from the plate to that camera, with world +z kept up, so the text is
    read head-on from wherever the user looks. With no client connected it
    faces -x, the side the compare camera starts on. The sidebar legend is
    the copy that carries the live step counter — an image has to be
    re-encoded to change, so the plate is re-drawn only when its text
    actually differs (see :func:`plate_status`).
    """

    def __init__(
        self,
        server,
        name: str,
        *,
        position,
        visible: bool = True,
        width: float = BILLBOARD_WIDTH,
    ):
        """Create the plate, empty, at ``position``.

        Args:
            server: The viser server.
            name: The scene node path.
            position: World anchor (see :func:`billboard_anchor`).
            visible: False while the compare scene is still being built.
            width: How wide the plate is drawn, in metres.
        """
        self.kind = "image"
        self.width = float(width)
        self.drawn = ""
        self.server = server
        self._faced: tuple | None = None
        self.node = add_plate_image(
            server,
            name,
            plate_image("", "", SLOT_COLOURS[0]),
            self.width,
            position=position,
            visible=visible,
        )

    def write(self, label: str, status: str, colour: str) -> None:
        """Set what the plate says, re-drawing it only when that changes."""
        short = plate_status(status)
        key = f"{label}\n{short}\n{colour}"
        if key == self.drawn:
            return
        self.drawn = key
        image = plate_image(label, short, colour)
        self.node.image = image
        shown_w, shown_h = plate_extent(image, self.width)
        try:
            self.node.render_width = shown_w
            self.node.render_height = shown_h
        except AttributeError:  # a fake node in tests
            pass

    def move(self, position) -> None:
        """Re-anchor the plate (the pelvis walked) and turn it to the viewer."""
        self.node.position = tuple(float(v) for v in position)
        self.face()

    def face(self) -> bool:
        """Turn the plate towards the connected client's camera.

        Uses the most recently connected client (one viewer is the common
        case). The orientation is recomputed only when the camera or the
        plate has moved by more than a centimetre since the last time.

        Returns:
            True when the orientation was updated.
        """
        camera = viewer_camera_position(self.server)
        if camera is None:
            return False
        key = tuple(round(v, 2) for v in (*camera, *self.position))
        if key == self._faced:
            return False
        self._faced = key
        self.node.wxyz = facing_wxyz(self.position, camera)
        return True

    @property
    def position(self) -> tuple:
        """Where the plate currently hangs."""
        return tuple(float(v) for v in self.node.position)

    def reveal(self) -> None:
        """Show the plate once its slot has been posed."""
        self.node.visible = True


def legend_row_html(colour: str, title: str, status: str) -> str:
    """One legend row: a colour swatch, the slot title, its live status."""
    swatch = (
        "display:inline-block;width:10px;height:10px;border-radius:2px;"
        f"margin:3px 6px 0 0;vertical-align:top;background:{colour};"
    )
    mono = "font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
    return (
        '<div style="margin:4px 0;">'
        f'<span style="{swatch}"></span>'
        '<span style="display:inline-block;vertical-align:top;">'
        f'<span style="font-size:11px;{mono}">{html.escape(title)}</span><br/>'
        f'<span style="font-size:10px;opacity:.7;{mono}">{html.escape(status)}</span>'
        "</span></div>"
    )


def legend_html(slots, statuses, building: str = "") -> str:
    """The Compare folder's legend, one row per slot.

    The 3D billboards can end up behind a robot, off screen, or edge-on to
    the camera; the legend is in the sidebar and says which colour is which
    policy and where its playhead is no matter what the 3D view does.

    Args:
        slots: The request's :class:`CompareSlot` list.
        statuses: One live status string per slot (short lists are padded).
        building: A line to show above the rows while the scene is still
            being built, e.g. ``building 2/3 …``.

    Returns:
        An HTML fragment for a single ``gui.add_html`` handle.
    """
    rows = []
    if building:
        rows.append(
            f'<div style="font-size:10px;opacity:.75;">{html.escape(building)}</div>'
        )
    for i, slot in enumerate(slots):
        status = statuses[i] if i < len(statuses) else ""
        rows.append(legend_row_html(slot_colour(i), slot.label, status))
    return '<div style="color:inherit;">' + "".join(rows) + "</div>"


def compare_header_html(request: CompareRequest) -> str:
    """``task move_plate · seed 620003 · 3 slots as 3 columns x 1 row``.

    The shared session prefix is named here because the labels drop it: the
    legend says ``s2`` and this line says which ``s2`` that is.
    """
    prefix = f" · sessions {html.escape(request.prefix)}*" if request.prefix else ""
    return (
        '<div style="font-size:10px;opacity:.6;">'
        f"task {html.escape(request.task)} · seed {request.seed} · "
        f"{html.escape(grid_note(len(request.slots), request.gap, request.layout))}"
        f"{prefix}</div>"
    )


def progress_bar_html(fraction: float, label: str, animated: bool = False) -> str:
    """One progress bar drawn in HTML: a label line and a filled track.

    viser's own progress bar and folder widgets take a ``visible`` flag at
    creation but do not reliably show or hide afterwards, so the bars are
    plain HTML written into a text node, which always updates.
    """
    pct = max(0.0, min(100.0, 100.0 * float(fraction)))
    fill = (
        "background:repeating-linear-gradient(45deg,#60a5fa 0 8px,#93c5fd 8px 16px);"
        if animated and pct <= 0.0
        else "background:#3b82f6;"
    )
    width = 100.0 if animated and pct <= 0.0 else pct
    return (
        '<div style="margin:4px 0 6px 0;">'
        f'<div style="font-size:10px;font-family:ui-monospace,monospace;">{html.escape(label)}</div>'
        '<div style="height:7px;border-radius:4px;background:rgba(127,127,127,.25);overflow:hidden;">'
        f'<div style="height:100%;width:{width:.1f}%;{fill}"></div></div></div>'
    )


class JobProgressBars:
    """The running jobs' progress, drawn as HTML into one text node.

    One line and one bar per job; the node is emptied when the work is over.
    """

    def __init__(self, node):
        """Bind the bars to the ``gui.add_html`` handle they are drawn into."""
        self.node = node
        self.rendered = ""

    def sync(self, rows: list[tuple[float, str, bool]], note: str = "") -> None:
        """Draw ``[(fraction, label, animated)]`` under an optional note line."""
        parts = []
        if note:
            parts.append(
                f'<div style="font-size:10px;opacity:.6;">{html.escape(note)}</div>'
            )
        parts += [progress_bar_html(f, label, animated) for f, label, animated in rows]
        markup = "".join(parts)
        if markup != self.rendered:
            self.rendered = markup
            self.node.content = markup

    def clear(self) -> None:
        """Remove every bar: the jobs are over and the scene is coming up."""
        if self.rendered:
            self.rendered = ""
            self.node.content = ""


class CompareLegend:
    """The compare scene's "Compare" folder: header, legend, Leave button.

    The legend is written from the main loop at the playback rate, and is
    also what says ``building 2/3 …`` while the slots are being built with
    their meshes hidden, so the sidebar is never blank and never lies about
    what is on screen. Before any of that, while the missing rollouts are
    still being produced, the same folder holds one progress bar per job
    (:class:`JobProgressBars`) and its Leave button already works.
    """

    def __init__(self, server, request: CompareRequest):
        """Build the folder for ``request``'s slots.

        Args:
            server: The viser server.
            request: The resolved request being played.
        """
        self.request = request
        self.statuses = ["" for _ in request.slots]
        self.building = ""
        self.rendered = ""
        self.left = False
        self.focus_asked: int | None = None
        self.folder = server.gui.add_folder("Compare")
        with self.folder:
            self.header = server.gui.add_html(compare_header_html(request))
            self.sources = server.gui.add_html(compare_status_html(request))
            self.jobs_note = server.gui.add_html("")
            self.body = server.gui.add_html("")
            self.focus_dd = server.gui.add_dropdown(
                "focus", options=self.focus_options(), initial_value=FOCUS_FREE
            )
            # The camera panels render four MuJoCo views every tick for one
            # slot out of six; in compare mode the scene is the point, so
            # they start off and the reader asks for them.
            self.cameras = server.gui.add_checkbox("camera panels", initial_value=False)
            self.leave = server.gui.add_button("Leave compare")
        self.jobs = JobProgressBars(self.jobs_note)
        self.leave.on_click(self.note_left)
        self.focus_dd.on_update(self.note_focus)
        self.refresh()

    def focus_options(self) -> list[str]:
        """What the focus dropdown offers: the free view, then every slot."""
        return [FOCUS_FREE] + [
            f"slot {i} · {slot.name} {slot.version_name}"
            for i, slot in enumerate(self.request.slots)
        ]

    def note_focus(self, _=None) -> None:
        """GUI callback: remember which slot to fly to (the loop does it)."""
        options = self.focus_options()
        value = str(self.focus_dd.value)
        index = options.index(value) - 1 if value in options else -1
        self.focus_asked = index if index >= 0 else None

    def take_focus(self) -> int | None:
        """The slot the focus dropdown asked for, once (main loop only)."""
        index, self.focus_asked = self.focus_asked, None
        return index

    def note_left(self, _=None) -> None:
        """GUI callback: the Leave button was pressed (the loop reads it)."""
        self.left = True

    def show_jobs(self, rows: list[tuple[float, str, bool]], note: str = "") -> None:
        """Draw the per-job progress bars, with an optional line above them.

        Args:
            rows: ``[(fraction, label, animated)]``, one per running job.
            note: A sentence above the bars ("2 slots need a rollout …").
        """
        self.jobs.sync(rows, note)

    def note_error(self, text: str) -> None:
        """Show a failed job in red, where the bars were."""
        self.jobs.clear()
        self.jobs_note.content = (
            f'<div style="font-size:10px;color:#dc2626;">{html.escape(text)}</div>'
        )
        self.jobs.rendered = self.jobs_note.content

    def jobs_done(self) -> None:
        """Take the bars down: every rollout is there and the scene follows."""
        self.jobs.clear()
        self.jobs_note.content = ""
        self.sources.content = compare_status_html(self.request)

    def refresh(self) -> None:
        """Re-render the legend rows, if anything they say has changed."""
        markup = legend_html(self.request.slots, self.statuses, self.building)
        if markup != self.rendered:
            self.rendered = markup
            self.body.content = markup

    def describe(self) -> list[str]:
        """The legend as plain text, one line per slot, for the log."""
        return [
            f"[viewer] compare legend {i}: {slot_colour(i)} {slot.label} · "
            f"{self.statuses[i] if i < len(self.statuses) else ''}"
            for i, slot in enumerate(self.request.slots)
        ]

    def set_status(self, index: int, text: str) -> None:
        """Record one slot's live status (call :meth:`refresh` after)."""
        if 0 <= index < len(self.statuses):
            self.statuses[index] = text

    def note_building(self, done: int, total: int) -> None:
        """Say which slot is being built while the scene is still hidden."""
        self.building = f"building {done}/{total} …"
        self.refresh()

    def note_ready(self) -> None:
        """Clear the building line: every slot is posed and revealed."""
        self.building = ""
        self.refresh()


def compare_note_html(count: int, maximum: int = COMPARE_SLOTS) -> str:
    """The Compare folder's status line, above the slots.

    Says what is already there and how to get more of it -- the panel opens
    with one slot per session and nothing else told the user that, or that
    a fourth policy is one button away.

    Args:
        count: How many slots the panel currently holds.
        maximum: The most it will hold.

    Returns:
        ``3 policies: one submission per session · Add adds another (up to 6)``
        as an HTML fragment.
    """
    word = "policy" if count == 1 else "policies"
    return (
        f'<div style="font-size:10px;opacity:.55;">{count} {word}: one '
        f"submission per session · Add adds another (up to {maximum})</div>"
    )


class ComparePanel:
    """The "Compare" folder: one slot per policy, one seed, one button.

    A slot is ``session x version`` of the task on screen (the task is
    fixed: comparing two different tasks side by side would compare two
    different scenes). The panel opens with one slot per session that holds
    the task, each on that session's submission -- which is what the status
    line says, because three sessions silently becoming three slots reads
    like a limit rather than a default. *Add policy* appends another slot
    (up to :data:`COMPARE_SLOTS`) and every slot carries its own *Remove*.

    *layout* says how the slots are placed: ``auto`` keeps one row up to
    three of them and folds into two columns from the fourth, and ``row`` /
    ``2 columns`` / ``3 columns`` say it outright.

    Clicking Compare hands a :class:`CompareRequest` to the viewer's main
    loop, which rebuilds the scene with one environment per slot; nothing
    is loaded here.
    """

    def __init__(
        self,
        server,
        *,
        cells,
        task,
        session,
        seed,
        gap,
        request_compare,
        layout: str = LAYOUT_AUTO,
    ):
        """Build the folder and seed its slots from the sessions available.

        Args:
            server: The viser server.
            cells: Every discovered cell.
            task: The task on screen (slots are all of this task).
            session: The session on screen (the first slot starts here).
            seed: The seed the field starts on.
            gap: Metres between neighbouring slots along +y.
            request_compare: Called from the main loop with the request.
            layout: Which grid the layout dropdown starts on.
        """
        self.server = server
        self.cells = list(cells)
        self.task = task
        self.request_compare = request_compare
        self.flags: set[str] = set()
        self.sessions = sessions_of(self.cells, task)
        self.rows: list[dict] = []
        with server.gui.add_folder("Compare", expand_by_default=False):
            self.note = server.gui.add_html(compare_note_html(0))
            self.slots_folder = server.gui.add_folder("policies")
            self.add_button = server.gui.add_button("Add policy")
            self.seed_text = server.gui.add_text("seed", initial_value=str(int(seed)))
            self.gap_number = server.gui.add_number(
                "gap (m)", initial_value=float(gap), min=0.5, step=0.1
            )
            self.layout_dd = server.gui.add_dropdown(
                "layout",
                options=list(COMPARE_LAYOUTS),
                initial_value=(layout if layout in COMPARE_LAYOUTS else LAYOUT_AUTO),
            )
            self.button = server.gui.add_button("Compare")
            self.status = server.gui.add_html("")
        self.add_button.on_click(lambda _: self.flags.add("add"))
        self.button.on_click(lambda _: self.flags.add("compare"))
        self.reseat(session)

    # -- slots -------------------------------------------------------------

    def session_options(self) -> list[str]:
        """What a slot's session dropdown offers for this task."""
        return [SLOT_EMPTY] + self.sessions

    def version_options(self, session: str) -> list[str]:
        """``last`` plus every recorded version of that session's cell."""
        if session == SLOT_EMPTY:
            return [VERSION_LAST]
        cell = find_cell(self.cells, self.task, session)
        if cell is None:
            return [VERSION_LAST]
        rows = visible_rows(policy_rows(cell.path))
        return [VERSION_LAST] + [f"v{row['version']:03d}" for row in rows]

    def add_row(self, session: str, version: str = VERSION_LAST) -> dict | None:
        """Append one slot to the ``policies`` folder.

        Args:
            session: The session it starts on, or :data:`SLOT_EMPTY`.
            version: The version it starts on.

        Returns:
            The new row, or None when the panel is already full.
        """
        if len(self.rows) >= COMPARE_SLOTS:
            return None
        index = len(self.rows)
        gui = self.server.gui
        with self.slots_folder:
            folder = gui.add_folder(f"slot {index}", expand_by_default=index < 2)
            with folder:
                session_dd = gui.add_dropdown(
                    "session", options=self.session_options(), initial_value=session
                )
                version_dd = gui.add_dropdown(
                    "version",
                    options=self.version_options(session),
                    initial_value=version,
                )
                policy_text = gui.add_text("policy file (optional)", initial_value="")
                remove = gui.add_button("Remove")
        row = {
            "folder": folder,
            "session": session_dd,
            "version": version_dd,
            "policy": policy_text,
            "remove": remove,
            "drop": False,
        }

        def on_remove(_, row=row) -> None:
            row["drop"] = True
            self.flags.add("slots")

        session_dd.on_update(lambda _: self.flags.add("slots"))
        remove.on_click(on_remove)
        self.rows.append(row)
        self.renumber()
        return row

    def drop_row(self, row: dict) -> None:
        """Remove one slot's widgets from the sidebar (main loop only)."""
        for key in ("remove", "policy", "version", "session", "folder"):
            handle = row.get(key)
            with contextlib.suppress(Exception):
                handle.remove()  # ty: ignore[unresolved-attribute]
        if row in self.rows:
            self.rows.remove(row)

    def renumber(self) -> None:
        """Re-label the slot folders after an add or a remove."""
        for i, row in enumerate(self.rows):
            with contextlib.suppress(Exception):
                row["folder"].label = f"slot {i}"
        self.note.content = compare_note_html(len(self.rows))

    def reseat(self, session: str) -> None:
        """Rebuild the default slots: one per session, the current one first.

        Args:
            session: The session on screen, which takes the first slot.
        """
        for row in list(self.rows):
            self.drop_row(row)
        starters = [s for s in self.sessions if s == session]
        starters += [s for s in self.sessions if s != session]
        for name in starters[:COMPARE_SLOTS]:
            self.add_row(name)
        if not self.rows:
            self.add_row(SLOT_EMPTY)
        self.renumber()

    def retask(self, cells, task: str, session: str) -> None:
        """Re-point the panel at another task (the Task dropdown moved)."""
        self.cells = list(cells)
        self.task = task
        self.sessions = sessions_of(self.cells, task)
        self.reseat(session)

    def refresh_versions(self) -> None:
        """Re-fill each slot's version dropdown for its chosen session."""
        for row in self.rows:
            options = self.version_options(str(row["session"].value))
            keep = str(row["version"].value)
            row["version"].options = options
            row["version"].value = keep if keep in options else VERSION_LAST

    def active_rows(self) -> list[dict]:
        """The slots that name a session (an empty slot is not compared)."""
        return [r for r in self.rows if str(r["session"].value) != SLOT_EMPTY]

    def specs(self) -> list[tuple[str, str]]:
        """The (session, version) pairs the slots currently name."""
        return [
            (str(r["session"].value), str(r["version"].value))
            for r in self.active_rows()
        ]

    # -- requests ----------------------------------------------------------

    def build_request(self) -> CompareRequest:
        """Resolve the slots into a request (main loop: it reads files)."""
        try:
            seed = int(str(self.seed_text.value).strip())
        except ValueError:
            request = CompareRequest(slots=[], seed=0, task=self.task)
            request.error = f"seed {self.seed_text.value!r} is not a number"
            return request
        rows = self.active_rows()
        request = resolve_compare(self.cells, self.task, self.specs(), seed)
        request.gap = float(self.gap_number.value)
        request.layout = str(self.layout_dd.value)
        if len(request.slots) == len(rows):
            for i, (slot, row) in enumerate(zip(request.slots, rows, strict=True)):
                path = str(row["policy"].value).strip()
                if path and Path(path).is_file():
                    slot.policy = path
                    slot.version = None
                    fill_slot(slot, seed)
                elif path:
                    request.error = f"slot {i}: no such policy file {path}"
        request.missing = [s for s in request.slots if s.episode is None]
        return request

    def tick(self) -> None:
        """Handle the flags GUI callbacks set (main loop only)."""
        flags, self.flags = set(self.flags), set()
        if "add" in flags:
            first = self.sessions[0] if self.sessions else SLOT_EMPTY
            if self.add_row(first) is None:
                self.status.content = (
                    '<div style="font-size:10px;opacity:.6;">'
                    f"{COMPARE_SLOTS} policies is the most the scene holds</div>"
                )
        if "slots" in flags:
            for row in [r for r in self.rows if r["drop"]]:
                self.drop_row(row)
            self.renumber()
            self.refresh_versions()
        if "compare" in flags:
            request = self.build_request()
            self.status.content = compare_status_html(request)
            if request.error or not request.slots:
                return
            self.request_compare(request)


class CellPanels:
    """The agent half of the viser sidebar: Session, Rollouts, policies.

    Builds, in sidebar order, ``Session`` (Task and Session dropdowns plus
    the ``run.json`` card), ``Rollouts`` (kind then item), ``Policy
    versions``, ``Policy code``, ``Transcript`` and ``Compare``. Every GUI
    callback only records a flag; :meth:`tick` does the file reads, the
    subprocess launches and the dropdown rewrites on the viewer's main loop.
    """

    def __init__(
        self,
        server,
        *,
        cell: Path,
        cells: list[Cell],
        batches: list[Path],
        open_batch: Path,
        request_switch,
        request_compare,
        follow: bool = False,
        seed_hint: int = EVAL_SEED_LO,
        compare_gap: float = COMPARE_GAP,
        compare_layout: str = LAYOUT_AUTO,
    ):
        """Create every cell panel for the batch currently on screen.

        Args:
            server: The viser server.
            cell: The cell the open batch belongs to.
            cells: Every cell discovered under ``--demo-dir``.
            batches: The viewer's live list of batch directories.
            open_batch: The batch on screen.
            request_switch: ``callable(index into batches)`` — called from
                GUI callbacks, so it must only set a flag.
            request_compare: ``callable(CompareRequest)``, same contract.
            follow: True when ``--follow`` is on (adds the scan line and the
                auto-jump checkbox). Rescanning itself belongs to the
                viewer, which hands the result to :meth:`refresh_rollouts`.
            seed_hint: Seed the Compare panel starts on.
            compare_gap: Metres between compare slots.
            compare_layout: Which grid the Compare panel starts on.
        """
        self.server = server
        self.cell = Path(cell)
        self.cells = list(cells)
        self.batches = batches
        self.open_batch = Path(open_batch).resolve()
        self.request_switch = request_switch
        self.flags: set[str] = set()
        self.here = cell_of(self.cells, self.open_batch) or Cell(
            path=self.cell, task=self.cell.name, session=self.cell.parent.name
        )
        self.rollouts = rollouts_of(self.cell, self.batches)
        self.opened, self.reason = self._opened_rollout()
        self.run = read_run(self.cell)
        self.rows = visible_rows(policy_rows(self.cell))
        self.price = price_of(self.run.get("model"))
        self.transcript: list[dict] = []
        self.transcript_key = None
        self.warned: set[str] = set()
        self.last: dict = {}
        self.job: AgentJob | None = None
        self.job_request: str | None = None
        self.polled = 0.0

        tasks = tasks_of(self.cells) or [self.here.task]
        sessions = sessions_of(self.cells, self.here.task) or [self.here.session]
        with server.gui.add_folder("Session"):
            self.task_dd = server.gui.add_dropdown(
                "task", options=tasks, initial_value=self.here.task
            )
            self.session_dd = server.gui.add_dropdown(
                "session", options=sessions, initial_value=self.here.session
            )
            self.summary = server.gui.add_html(summary_html(self.run))
        with server.gui.add_folder("Rollouts"):
            self.kind_dd = server.gui.add_dropdown(
                "kind", options=self._kinds(), initial_value=self._opened_kind()
            )
            self.item_dd = server.gui.add_dropdown(
                "rollout",
                options=self._items(self._opened_kind()),
                initial_value=self._opened_item(),
            )
            self.rollout_note = server.gui.add_html(self._note_html())
            # --follow only: the same "auto-jump to newest" the plain demo
            # dropdown offers, so a session being watched live can pull the
            # newest batch onto the screen by itself.
            self.auto_jump = (
                server.gui.add_checkbox("auto-jump to newest", initial_value=False)
                if follow
                else None
            )
            self.follow_note = server.gui.add_html("") if follow else None
        options = [f"v{row['version']:03d}" for row in self.rows] or ["(none)"]
        with server.gui.add_folder("Policy versions"):
            self.policy_table = server.gui.add_html(
                policy_table_html(self.rows, self.price)
            )
            self.version_dd = server.gui.add_dropdown(
                "version", options=options, initial_value=options[-1]
            )
            self.template_cb = server.gui.add_checkbox(
                "show template", initial_value=SHOW_TEMPLATE
            )
            self.seeds_text = server.gui.add_text(
                "seeds", initial_value=DEFAULT_REPLAY_SEEDS
            )
            self.replay_btn = server.gui.add_button("Replay")
            self.eval_btn = server.gui.add_button("Evaluate (100 hidden seeds)")
            self.policy_path_text = server.gui.add_text("policy path", initial_value="")
            self.run_policy_btn = server.gui.add_button("Replay this file")
            self.job_msg = server.gui.add_html("")
            # A rollout of 100 hidden seeds takes minutes; the bar says how
            # far in it is (from the job's .progress.json, or the files it
            # has finished) instead of a line that never changes.
            self.job_bar = server.gui.add_html("")
        with server.gui.add_folder("Policy code", expand_by_default=False):
            self.code_path_text = server.gui.add_text(
                "path (select + copy; viser has no clipboard API)",
                initial_value="",
            )
            self.code_link = server.gui.add_html("")
            self.code_html = server.gui.add_html("")
            self.diff_dd = server.gui.add_dropdown(
                "diff against", options=["(none)"] + options, initial_value="(none)"
            )
            self.diff_html = server.gui.add_html("")
        with server.gui.add_folder("Transcript", expand_by_default=False):
            self.tr_title = server.gui.add_html("")
            self.tr_prev = server.gui.add_button("prev version")
            self.tr_next = server.gui.add_button("next version")
            self.tr_body = server.gui.add_html("")
        self.compare = ComparePanel(
            server,
            cells=self.cells,
            task=self.here.task,
            session=self.here.session,
            seed=seed_hint,
            gap=compare_gap,
            layout=compare_layout,
            request_compare=request_compare,
        )
        if not self.rows:
            self.replay_btn.disabled = True
            self.eval_btn.disabled = True
        # Callbacks only request; every read happens in tick() on the main loop.
        self.task_dd.on_update(lambda _: self.flags.add("cell"))
        self.session_dd.on_update(lambda _: self.flags.add("cell"))
        self.kind_dd.on_update(lambda _: self.flags.add("kind"))
        self.item_dd.on_update(lambda _: self.flags.add("item"))
        self.version_dd.on_update(lambda _: self.flags.add("panels"))
        self.template_cb.on_update(lambda _: self.flags.add("template"))
        self.diff_dd.on_update(lambda _: self.flags.add("panels"))
        self.tr_prev.on_click(lambda _: self.flags.update({"panels", "prev"}))
        self.tr_next.on_click(lambda _: self.flags.update({"panels", "next"}))
        self.replay_btn.on_click(lambda _: self._ask("replay"))
        self.eval_btn.on_click(lambda _: self._ask("evaluate"))
        self.run_policy_btn.on_click(lambda _: self._ask("replay-file"))
        self.refresh_panels()

    # -- rollout tree ----------------------------------------------------

    def _opened_rollout(self) -> tuple[Rollout | None, str]:
        """The rollout on screen, or the default when it is not a cell one."""
        fallback, reason = default_rollout(self.rollouts)
        for rollout in self.rollouts:
            if rollout.batch.resolve() == self.open_batch:
                if fallback is not None and fallback.batch == rollout.batch:
                    return rollout, reason
                return rollout, "chosen with --batch"
        return fallback, reason

    def _kinds(self) -> list[str]:
        """Kind names that this cell actually has rollouts for."""
        present = {r.kind for r in self.rollouts}
        names = [title for title, kind in ROLLOUT_KINDS if kind in present]
        return names or ["(none)"]

    def _opened_kind(self) -> str:
        """The kind dropdown's value for the rollout on screen."""
        return self.opened.title if self.opened is not None else "(none)"

    def _items(self, title: str) -> list[str]:
        """Item labels of one kind, in the order the dropdown offers them."""
        labels = [rollout_item_label(r) for r in self.rollouts if r.title == title]
        return labels or ["(none)"]

    def _opened_item(self) -> str:
        """The item dropdown's value for the rollout on screen."""
        return rollout_item_label(self.opened) if self.opened is not None else "(none)"

    def _note_html(self) -> str:
        """The line under the Rollouts dropdowns saying what opened and why."""
        if self.opened is None:
            return '<div style="font-size:10px;opacity:.6;">no rollouts yet</div>'
        return (
            '<div style="font-size:10px;opacity:.55;">showing '
            f"{html.escape(self.opened.title.lower())} "
            f"{html.escape(rollout_item_label(self.opened))} — "
            f"{html.escape(self.reason)}</div>"
        )

    def _rollout_at(self, title: str, item: str) -> Rollout | None:
        """The rollout a (kind, item) pair names."""
        for rollout in self.rollouts:
            if rollout.title == title and rollout_item_label(rollout) == item:
                return rollout
        return None

    def batch_index(self, batch: Path) -> int | None:
        """Where a batch sits in the viewer's flat list, None when it left."""
        wanted = Path(batch).resolve()
        for i, candidate in enumerate(self.batches):
            if Path(candidate).resolve() == wanted:
                return i
        return None

    def describe(self) -> list[str]:
        """The startup log lines for the cell that just opened."""
        lines = describe_opening(self.here, self.rollouts, self.opened, self.reason)
        lines.append(
            f"[viewer] policy table columns: {', '.join(table_columns(self.rows))}"
        )
        lines.append(
            f"[viewer] code panel: {self.last.get('path')} "
            f"({self.last.get('lines')} lines) · transcript slice "
            f"{self.last.get('records')} records ({self.last.get('title')})"
        )
        return lines

    # -- panel refreshes (main loop) --------------------------------------

    def version_index(self) -> int:
        """Position of the selected version in the rows, -1 when none."""
        name = str(self.version_dd.value)
        for i, row in enumerate(self.rows):
            if f"v{row['version']:03d}" == name:
                return i
        return len(self.rows) - 1

    def transcript_records(self) -> list[dict]:
        """The cell's transcript, re-read only when the file changed."""
        path = self.cell / TRANSCRIPT_JSONL
        try:
            stat = path.stat()
            key = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            key = None
        if key != self.transcript_key:
            self.transcript = read_transcript(self.cell) if key else []
            self.transcript_key = key
        return self.transcript

    def refresh_panels(self, step: int = 0) -> None:
        """Rebuild the code, diff and transcript panels (main loop only)."""
        index = self.version_index()
        if step and self.rows:
            index = int(np.clip(index + step, 0, len(self.rows) - 1))
            self.version_dd.value = f"v{self.rows[index]['version']:03d}"
        if not self.rows or index < 0:
            self.code_html.content = (
                '<div style="font-size:11px;opacity:.6;">no versions</div>'
            )
            self.tr_title.content = transcript_title(self.rows, -1)
            self.tr_body.content = ""
            return
        here = f"v{self.rows[index]['version']:03d}"
        path = policy_file(self.cell, self.rows[index])
        try:
            source = path.read_text(errors="replace")
        except OSError as exc:
            source = ""
            # A version whose directory is gone would otherwise repeat this
            # on every rescan; say it once per path.
            if str(path) not in self.warned:
                self.warned.add(str(path))
                print(f"[viewer] cannot read {path}: {exc}", flush=True)
        self.code_path_text.value = str(path)
        self.code_link.content = vscode_link_html(path)
        self.code_html.content = code_html(path)
        other = str(self.diff_dd.value)
        match = [
            r for r in self.rows if f"v{r['version']:03d}" == other and other != here
        ]
        if match:
            base = policy_file(self.cell, match[0])
            try:
                self.diff_html.content = diff_html(
                    base.read_text(errors="replace"), source, other, here
                )
            except OSError as exc:
                self.diff_html.content = f'<div style="font-size:11px;">{exc}</div>'
        else:
            self.diff_html.content = (
                '<div style="font-size:11px;opacity:.6;">pick another version to '
                "diff against</div>"
            )
        records = version_records(self.transcript_records(), self.rows, index)
        self.last = {
            "path": str(path),
            "lines": len(source.splitlines()),
            "records": len(records),
            "title": transcript_title(self.rows, index),
        }
        self.tr_title.content = (
            f'<div style="font-size:10px;opacity:.6;">'
            f"{html.escape(transcript_title(self.rows, index))} · "
            f"{len(records)} records</div>"
        )
        self.tr_body.content = transcript_html(records)

    def refresh_versions(self) -> None:
        """Re-read run.json and policies/index.json into the panels."""
        run = read_run(self.cell)
        if run != self.run:
            self.run = run
            self.price = price_of(run.get("model"))
            self.summary.content = summary_html(run)
        rows = visible_rows(policy_rows(self.cell))
        known = [row["version"] for row in self.rows]
        self.rows = rows
        self.policy_table.content = policy_table_html(rows, self.price)
        versions = [row["version"] for row in rows]
        if versions != known:
            options = [f"v{v:03d}" for v in versions] or ["(none)"]
            selected = str(self.version_dd.value)
            self.version_dd.options = options
            self.version_dd.value = selected if selected in options else options[-1]
            other = str(self.diff_dd.value)
            self.diff_dd.options = ["(none)"] + options
            self.diff_dd.value = other if other in self.diff_dd.options else "(none)"
            self.replay_btn.disabled = self.eval_btn.disabled = not rows
            print(f"[viewer] {len(rows)} policy versions", flush=True)
            self.compare.refresh_versions()
        self.flags.add("panels")

    def refresh_rollouts(self) -> None:
        """Re-read this cell's rollouts and re-point the two dropdowns."""
        self.rollouts = rollouts_of(self.cell, self.batches)
        self.opened, _ = self._opened_rollout()
        kinds = self._kinds()
        self.kind_dd.options = kinds
        if str(self.kind_dd.value) not in kinds:
            self.kind_dd.value = kinds[0]
        items = self._items(str(self.kind_dd.value))
        keep = str(self.item_dd.value)
        self.item_dd.options = items
        self.item_dd.value = keep if keep in items else items[0]
        self.rollout_note.content = self._note_html()
        self.flags.discard("kind")
        self.flags.discard("item")

    def note_follow(self, text: str) -> None:
        """Write the ``--follow`` scan line under the Rollouts dropdowns."""
        if self.follow_note is not None:
            self.follow_note.content = (
                f'<div style="font-size:10px;opacity:.55;">{html.escape(text)}</div>'
            )

    # -- jobs -------------------------------------------------------------

    def _ask(self, kind: str) -> None:
        """GUI callback: request a job (the main loop starts it)."""
        self.job_request = kind

    def start_job(self, kind: str) -> None:
        """Launch one bigym-agent subcommand (main loop only)."""
        policy = ""
        if kind == "replay-file":
            kind = "replay"
            policy = str(self.policy_path_text.value).strip()
            if not policy or not Path(policy).is_file():
                self.job_msg.content = "<b>no such policy file</b>"
                return
            job = AgentJob(
                kind, self.cell, 0, str(self.seeds_text.value), policy=policy
            )
        else:
            name = str(self.version_dd.value)
            if not name.startswith("v") or not name[1:].isdigit():
                self.job_msg.content = "<b>no policy version selected</b>"
                return
            job = AgentJob(kind, self.cell, int(name[1:]), str(self.seeds_text.value))
        job.start()
        self.job = job
        self.polled = 0.0
        self.replay_btn.disabled = True
        self.eval_btn.disabled = True
        self.run_policy_btn.disabled = True
        self.show_progress(0.0, True)
        print(f"[viewer] {' '.join(job.command)} > {job.log_path}", flush=True)
        self.job_msg.content = (
            f"<i>running {html.escape(kind)} {html.escape(job.label)} … "
            f"log: {html.escape(job.log_path.name)}</i>"
        )

    def show_progress(self, fraction: float, animated: bool, label: str = "") -> None:
        """Draw the job bar at ``fraction`` (0-1); striped while unknown."""
        self.job_bar.content = progress_bar_html(fraction, label, animated)

    def hide_progress(self) -> None:
        """Take the job bar down: nothing is running any more."""
        self.job_bar.content = ""

    def poll_job(self, now: float, refresh_batches=None) -> None:
        """Update the panel from the running job; rescan when it ends."""
        job = self.job
        if job is None or now - self.polled < 0.5:
            return
        self.polled = now
        if not job.poll():
            fraction, label, animated = job.progress_state()
            self.show_progress(fraction, animated, label)
            self.job_msg.content = progress_note_html(label)
            return
        self.job = None
        self.hide_progress()
        self.replay_btn.disabled = False
        self.eval_btn.disabled = False
        self.run_policy_btn.disabled = False
        if job.error:
            print(f"[viewer] {job.error}", flush=True)
            self.job_msg.content = f"<b>{html.escape(job.error)}</b>"
            return
        if refresh_batches is not None:
            refresh_batches()
        # An evaluation adds the version's hidden-seed score to the table.
        self.rows = visible_rows(policy_rows(self.cell))
        self.policy_table.content = policy_table_html(self.rows, self.price)
        self.refresh_rollouts()
        self.job_msg.content = f"done — {html.escape(job.progress())}"
        print(f"[viewer] {job.progress()}", flush=True)

    # -- main-loop entry point --------------------------------------------

    def tick(self, now: float, refresh_batches=None) -> None:
        """Do everything the GUI asked for since the last tick."""
        flags, self.flags = set(self.flags), set()
        if "template" in flags:
            global SHOW_TEMPLATE
            SHOW_TEMPLATE = bool(self.template_cb.value)
            self.rows = []  # so refresh_versions sees the list change
            self.refresh_versions()
            self.refresh_rollouts()
        if "cell" in flags:
            self._switch_cell()
        elif "kind" in flags:
            items = self._items(str(self.kind_dd.value))
            self.item_dd.options = items
            self.item_dd.value = items[0]
            self._switch_rollout()
        elif "item" in flags:
            self._switch_rollout()
        if "panels" in flags:
            step = (1 if "next" in flags else 0) - (1 if "prev" in flags else 0)
            self.refresh_panels(step)
        request, self.job_request = self.job_request, None
        if request is not None and self.job is None:
            self.start_job(request)
        self.poll_job(now, refresh_batches)
        self.compare.tick()

    def _switch_cell(self) -> None:
        """Task / Session moved: open that cell's default rollout."""
        task = str(self.task_dd.value)
        sessions = sessions_of(self.cells, task)
        if str(self.session_dd.value) not in sessions:
            self.session_dd.options = sessions or ["(none)"]
            if sessions:
                self.session_dd.value = sessions[0]
        else:
            self.session_dd.options = sessions
        # Keep Compare on the task the dropdowns now name, whether or not
        # the switch below finds a rollout to open.
        self.compare.retask(self.cells, task, str(self.session_dd.value))
        target = find_cell(self.cells, task, str(self.session_dd.value))
        if target is None:
            self.rollout_note.content = (
                '<div style="font-size:10px;color:#dc2626;">no cell for '
                f"{html.escape(task)} in {html.escape(str(self.session_dd.value))}"
                "</div>"
            )
            return
        rollouts = rollouts_of(target.path, self.batches)
        rollout, reason = default_rollout(rollouts)
        if rollout is None:
            self.rollout_note.content = (
                '<div style="font-size:10px;color:#dc2626;">'
                f"{html.escape(target.title)} has no rollouts yet</div>"
            )
            return
        index = self.batch_index(rollout.batch)
        if index is not None:
            print(
                f"[viewer] switching to {target.title}: {rollout.title} "
                f"{rollout_item_label(rollout)} — {reason}",
                flush=True,
            )
            self.request_switch(index)

    def _switch_rollout(self) -> None:
        """Kind / rollout moved: open that batch."""
        rollout = self._rollout_at(str(self.kind_dd.value), str(self.item_dd.value))
        if rollout is None:
            return
        index = self.batch_index(rollout.batch)
        if index is not None:
            self.request_switch(index)


def pose_slot(built: dict, frame: int, mujoco) -> int:
    """Pose one slot at ``frame`` of its replay and push it to the browser.

    The environment was reset on the compare seed, which puts the props
    where that seed wants them but leaves the robot in the reset pose --
    which is not frame 0 of the replay. Every slot is posed from the stored
    ``full_qpos`` before it is ever shown, so the reset pose never reaches
    the screen.

    Args:
        built: One entry of the scene's ``slots`` list.
        frame: The playhead, clamped to this slot's episode.
        mujoco: The ``mujoco`` module.

    Returns:
        The frame actually used.
    """
    qpos = built["qpos"]
    here = max(0, min(int(frame), int(qpos.shape[0]) - 1))
    built["data"].qpos[: qpos.shape[1]] = qpos[here]
    mujoco.mj_forward(built["model"], built["data"])
    built["scene"].update_from_mjdata(built["data"])
    return here


def slot_status_text(built: dict, here: int) -> str:
    """``step 812/1701``, plus the outcome once the playhead is at the end."""
    total = int(built["qpos"].shape[0])
    text = f"step {here + 1}/{total}"
    if here >= total - 1:
        outcome = slot_outcome(built["info"])
        if outcome:
            text = f"{text} · {outcome}"
    return text


def place_billboard(built: dict) -> tuple[float, float, float]:
    """Move one slot's billboard onto its pelvis and return the anchor."""
    anchor = billboard_anchor(
        pelvis_position(built["data"], built["pelvis"]), built["offset"]
    )
    built["billboard"].move(anchor)
    return anchor


def compare_server(port: int):
    """Open the viser server compare mode draws into.

    The port is retried: a just-stopped single-rollout server needs a moment
    to release the socket.

    Args:
        port: The viser port.

    Returns:
        The running server.
    """
    for attempt in range(20):
        try:
            return viser.ViserServer(port=port, label="demo-viewer", verbose=False)
        except OSError:
            if attempt == 19:
                raise
            time.sleep(0.25)
    raise OSError(f"cannot open a viser server on port {port}")


def wait_for_missing(
    request: CompareRequest,
    legend: CompareLegend,
    *,
    timeout: float = 3600.0,
    deadline: float = 0.0,
    tick: float = 0.25,
    log_every: float = 5.0,
) -> bool:
    """Produce the rollouts the request is missing, with the GUI alive.

    Drives :class:`MissingWork` from this loop instead of blocking on it:
    every tick polls the jobs, reads each one's ``.progress.json`` and
    redraws its bar, so the sidebar shows where each rollout has got to and
    the Leave button (which the legend already owns) still works.

    Args:
        request: The request being produced.
        legend: The Compare folder to draw into.
        timeout: Give up on a job that takes longer than this.
        deadline: Wall-clock seconds for the whole wait (0 = no limit);
            ``--exit-after-seconds`` passes its budget here so a headless
            run cannot hang on a rollout that never ends.
        tick: Seconds between polls.
        log_every: Print every bar's label this often, for the log.

    Returns:
        True when every slot has an episode and the scene can be built;
        False when the user left, a job failed or the deadline passed.
    """
    work = MissingWork(request, timeout=timeout)
    work.start()
    note = (
        f"{len(work.jobs)} rollout(s) to produce on seed {request.seed}"
        if work.jobs
        else ""
    )
    started = time.time()
    said = 0.0
    while True:
        if legend.left:
            print("[viewer] compare: Leave pressed, stopping the replays", flush=True)
            work.stop()
            return False
        finished = work.poll()
        rows = work.bars()
        legend.show_jobs(rows, note)
        now = time.time()
        if rows and (now - said >= log_every or finished):
            said = now
            for fraction, label, _ in rows:
                print(
                    f"[viewer] compare: {label} ({100.0 * fraction:.0f}%)", flush=True
                )
        if finished:
            break
        if deadline > 0.0 and now - started > deadline:
            work.stop()
            legend.note_error(f"gave up after {deadline:g}s producing the rollouts")
            print(
                f"[viewer] compare: {deadline:g}s elapsed while producing", flush=True
            )
            return False
        time.sleep(tick)
    if work.error:
        legend.note_error(work.error)
        print(f"[viewer] compare: {work.error}", flush=True)
        return False
    legend.jobs_done()
    return True


def build_compare_scene(
    request: CompareRequest,
    *,
    port: int,
    build_env,
    load_metadata,
    lighting_prefs: dict,
    server=None,
    legend: CompareLegend | None = None,
) -> dict:
    """Build one viser server holding one environment per compare slot.

    Each slot gets its own ``MjModel``/``MjData`` from the ordinary
    ``build_env`` path, reset on the compare seed so the props sit where
    that seed puts them, a distinct mjviser node prefix and a frame that
    carries its place in the grid the request's ``layout`` asks for: one
    column is ``gap`` metres along +y, one row is ``gap`` metres along -x.

    Nothing is shown while that happens. Every slot's two root frames are
    created **before** any of its geometry (so nothing is ever drawn at the
    origin first) and start hidden; when the last slot is built, all of them
    are posed to frame 0 of their replay and only then revealed together.
    The legend says ``building 2/3 …`` in the meantime.

    Args:
        request: The resolved request (every slot must have an episode).
        port: The viser port.
        build_env: ``view_demos.build_env``.
        load_metadata: ``view_demos.load_metadata``.
        lighting_prefs: The viewer's lighting preferences.
        server: A server already showing the Compare folder (the one the
            progress bars ran in), or None to open one on ``port``.
        legend: Its legend, or None to build one.

    Returns:
        A dict with ``server``, ``legend``, ``slots`` (one dict per slot:
        env, inner, model, data, scene, qpos, rewards, info, billboard,
        frames, geom_rgba, cell), ``offsets`` and ``fps``.
    """
    patch_mjviser()
    if server is None:
        server = compare_server(port)
    if legend is None:
        legend = CompareLegend(server, request)
    # One /fixed_bodies root for every slot: see slot_scene_build.
    fixed_frame = server.scene.add_frame("/fixed_bodies", show_axes=False)
    built: list[dict] = []
    total = len(request.slots)
    places = slot_grid(total, request.layout)
    offsets = grid_offsets(total, request.gap, request.layout)
    print(
        f"[viewer] compare layout {request.layout}: "
        f"{grid_note(total, request.gap, request.layout)}",
        flush=True,
    )
    fps = 50.0
    camera_keys: tuple[str, ...] = ("head",)
    camera_shape: tuple[int, int] = (84, 84)
    for i, slot in enumerate(request.slots):
        legend.note_building(i + 1, total)
        legend.set_status(i, "building …")
        legend.refresh()
        assert slot.episode is not None
        batch = Path(slot.episode).parent
        metadata = load_metadata(batch)
        step_seconds = metadata.get("control_step_seconds")
        if step_seconds:
            fps = 1.0 / float(step_seconds)
        task_meta = metadata.get("task") or {}
        camera_keys = tuple(task_meta.get("camera_keys") or ("head",))
        camera_shape = tuple(task_meta.get("camera_shape") or (84, 84))
        env = build_env(metadata)
        env.reset(seed=request.seed)
        inner = env.inner_env
        model, data = inner.model, inner.data
        # Compare mode leaves the targets out rather than drawing N of them.
        hide_reach_targets(inner)
        prefix = f"slot{i}"
        offset = offsets[i]
        # The offset frames exist before a single geom goes under them, so
        # no slot is ever drawn at the origin and then moved.
        body_frame = server.scene.add_frame(
            f"/bodies/{prefix}", show_axes=False, position=offset, visible=False
        )
        slot_fixed = server.scene.add_frame(
            f"/fixed_bodies/{prefix}", show_axes=False, position=offset, visible=False
        )
        patch_mjviser_track_props(model, inner)
        # The tint goes in only for the bake, and comes straight back out:
        # MuJoCo's renderer (the camera panels) reads geom_rgba live, and it
        # must show the policy's own view, not a blue-tinted robot.
        pristine = np.array(model.geom_rgba, copy=True)
        tinted = tint_slot_robot(model, SLOT_TINTS[i % len(SLOT_TINTS)])
        with slot_scene_build(
            server, prefix, keep_grid=i == 0, fixed_frame=fixed_frame
        ):
            scene = ViserMujocoScene(server=server, mj_model=model, num_envs=1)
        model.geom_rgba[:] = pristine
        restored = bool(np.array_equal(np.asarray(model.geom_rgba), pristine))
        scene.camera_tracking_enabled = False
        with np.load(slot.episode) as episode:
            qpos = np.asarray(episode["full_qpos"], dtype=np.float64)
            rewards = np.asarray(episode["reward"]).reshape(-1)
        info = episode_info(Path(slot.episode))
        billboard = SlotBillboard(
            server,
            f"/compare/{prefix}/billboard",
            position=(offset[0], offset[1], offset[2] + BILLBOARD_Z),
            visible=False,
        )
        billboard.write(slot.label, "step 0", slot_colour(i))
        built.append(
            {
                "slot": slot,
                "env": env,
                "inner": inner,
                "model": model,
                "data": data,
                "scene": scene,
                "qpos": qpos,
                "rewards": rewards,
                "info": info,
                "pelvis": pelvis_body_id(model),
                "billboard": billboard,
                "frames": (body_frame, slot_fixed),
                "geom_rgba": pristine,
                "offset": offset,
                "place": places[i],
                "prefix": prefix,
            }
        )
        print(
            f"[viewer] compare slot {i}: {prefix} at row {places[i][0]} "
            f"column {places[i][1]} (x {offset[0]:+.2f}, y {offset[1]:+.2f}), "
            f"built hidden, {qpos.shape[0]} frames from {slot.source}/"
            f"{Path(slot.episode).name}, {tinted} geoms tinted",
            flush=True,
        )
        print(
            f"[viewer] compare slot {i}: geom_rgba restored after the mesh "
            f"bake — model untouched: {restored}",
            flush=True,
        )
        print(f"[viewer] compare slot {i} label: {slot.label}", flush=True)
    apply_lighting(server, lighting_prefs)
    # Every slot is posed to frame 0 of its own replay before anything is
    # shown: the reset pose is not the first frame, and a robot standing in
    # the dishwasher for one update is exactly what that looks like.
    for i, entry in enumerate(built):
        pose_slot(entry, 0, mujoco)
        anchor = place_billboard(entry)
        entry["billboard"].write(
            entry["slot"].label, slot_status_text(entry, 0), slot_colour(i)
        )
        legend.set_status(i, slot_status_text(entry, 0))
        pelvis = pelvis_position(entry["data"], entry["pelvis"])
        print(
            f"[viewer] compare slot {i}: posed to frame 0 of the replay "
            f"(not the reset pose), pelvis "
            f"({pelvis[0]:.2f}, {pelvis[1]:.2f}, {pelvis[2]:.2f})",
            flush=True,
        )
        print(
            f"[viewer] compare slot {i}: billboard ({entry['billboard'].kind}) at "
            f"({anchor[0]:.2f}, {anchor[1]:.2f}, {anchor[2]:.2f}) = pelvis "
            f"+ offset ({entry['offset'][0]:.2f}, {entry['offset'][1]:.2f}) "
            f"+ {BILLBOARD_Z:.2f} m, facing "
            f"({PLATE_FACE[0]:g}, {PLATE_FACE[1]:g}, {PLATE_FACE[2]:g})",
            flush=True,
        )
    for entry in built:
        for frame in entry["frames"]:
            frame.visible = True
        entry["billboard"].reveal()
    legend.note_ready()
    print(f"[viewer] compare: {len(built)} scenes revealed together", flush=True)
    for line in legend.describe():
        print(line, flush=True)
    return {
        "server": server,
        "legend": legend,
        "slots": built,
        "offsets": offsets,
        "fps": fps,
        "mujoco": mujoco,
        "camera_keys": camera_keys,
        "camera_shape": camera_shape,
    }


RECORD_LOG_EVERY = 50
COMPARE_PAD = 1.3
"""Metres a compare slot's content (a table, a robot, the plates) reaches
from its origin, for the recording lens."""
COMPARE_BOX_CENTRE = (0.4, 0.0, 0.9)
"""Where the middle of one slot's content is, relative to its origin."""
COMPARE_BOX_HEIGHT = 1.0
"""Half height of that content."""
COMPARE_PAD_TIGHT = 0.9
"""The pad for the oblique view, which is about the robots and their tables
rather than the whole grid."""
SETTLE_TIMEOUT = 30.0
RECORD_VIEWS = ("overview", "oblique")
RECORD_LABEL_SCALE = 1.5
"""How much bigger the billboards are drawn for a recording than live: a
video is watched from further away than a browser tab."""
OBLIQUE_AZIMUTH = 35.0
"""Degrees the oblique camera stands off the robots' front (+x) towards +y."""
OBLIQUE_ELEVATION = 24.0
"""Degrees above the horizon the oblique camera looks down from."""
RECORD_FOV_MIN = 0.18
RECORD_FOV_MAX = 1.40


def compare_box(
    offsets, pad: float = COMPARE_PAD
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """The box a compare grid's content lives in: ``(centre, half extents)``.

    Args:
        offsets: Every slot's offset.
        pad: Metres a slot's content reaches from its origin.

    Returns:
        ``((cx, cy, cz), (hx, hy, hz))`` in world metres.
    """
    places = [tuple(float(v) for v in o) for o in offsets] or [(0.0, 0.0, 0.0)]
    xs = [o[0] for o in places]
    ys = [o[1] for o in places]
    centre = (
        0.5 * (min(xs) + max(xs)) + COMPARE_BOX_CENTRE[0],
        0.5 * (min(ys) + max(ys)) + COMPARE_BOX_CENTRE[1],
        COMPARE_BOX_CENTRE[2],
    )
    half = (
        0.5 * (max(xs) - min(xs)) + float(pad),
        0.5 * (max(ys) - min(ys)) + float(pad),
        COMPARE_BOX_HEIGHT,
    )
    return centre, half


def compare_record_pose(
    offsets, view: str = "overview", *, home=None
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Where a recording camera stands for ``view``.

    ``overview`` is the live compare's starting camera: behind the robots,
    high, every scene in view. ``oblique`` stands in front of them and to
    one side, lower and closer — the three-quarter view that shows the
    hands and what is on the tables rather than four backs.

    Args:
        offsets: Every slot's offset.
        view: One of :data:`RECORD_VIEWS`.
        home: The live overview pose, when the caller already has it.

    Returns:
        ``(position, look_at)`` in world metres.
    """
    if view not in RECORD_VIEWS:
        raise ValueError(f"record view must be one of {RECORD_VIEWS}, not {view!r}")
    if view == "overview":
        return home if home is not None else compare_camera_pose(offsets)
    centre, half = compare_box(offsets, COMPARE_PAD_TIGHT)
    azimuth = np.radians(OBLIQUE_AZIMUTH)
    elevation = np.radians(OBLIQUE_ELEVATION)
    direction = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    distance = 2.0 * max(half[0], half[1]) + 2.0
    position = np.asarray(centre, dtype=float) + distance * direction
    return cast(
        tuple[tuple[float, float, float], tuple[float, float, float]],
        (tuple(float(v) for v in position), tuple(float(v) for v in centre)),
    )


def box_fov(
    position, look_at, box, aspect: float = 16 / 9, margin: float = 1.05
) -> float:
    """The vertical field of view that fits ``box`` in the frame.

    The box's eight corners are projected into the camera's frame and the
    lens opened just enough, vertically and horizontally (through the
    aspect ratio), to hold the widest of them, so the framing is exact for
    any camera position rather than for one side of the grid.

    Args:
        position: Where the camera is.
        look_at: What it looks at.
        box: ``(centre, half extents)`` in world metres.
        aspect: Frame width over height.
        margin: Room to leave around the box (1.05 = 5%).

    Returns:
        The vertical field of view in radians, clamped.
    """
    eye = np.asarray(position, dtype=float)
    forward = np.asarray(look_at, dtype=float) - eye
    norm = float(np.linalg.norm(forward))
    if norm < 1e-6:
        return RECORD_FOV_MAX
    forward = forward / norm
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ world_up)) > 0.99:
        world_up = np.array([1.0, 0.0, 0.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    centre, half = box
    centre = np.asarray(centre, dtype=float)
    half = np.asarray(half, dtype=float)
    vertical = 0.0
    horizontal = 0.0
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corner = centre + half * np.array([sx, sy, sz]) - eye
                depth = max(0.2, float(corner @ forward))
                vertical = max(vertical, abs(float(corner @ up)) / depth)
                horizontal = max(horizontal, abs(float(corner @ right)) / depth)
    fov_v = 2.0 * np.arctan(margin * vertical)
    fov_h = 2.0 * np.arctan(margin * horizontal)
    from_width = 2.0 * np.arctan(np.tan(0.5 * fov_h) / max(0.1, float(aspect)))
    return float(np.clip(max(fov_v, from_width), RECORD_FOV_MIN, RECORD_FOV_MAX))


def settle_render(
    client, size: tuple[int, int], *, timeout: float = SETTLE_TIMEOUT
) -> int:
    """Render until two pictures in a row are identical; the page has loaded.

    A browser that has just connected is still downloading meshes, and a
    frame asked for too soon shows robots with no tables. The scene is
    static until the playhead moves, so once two consecutive renders match
    everything that is going to appear has appeared.

    Args:
        client: Anything with ``get_render(height, width)``.
        size: ``(width, height)`` to render at.
        timeout: Give up (and record anyway) after this many seconds.

    Returns:
        How many renders it took.
    """
    width, height = int(size[0]), int(size[1])
    previous = None
    renders = 0
    started = time.time()
    while True:
        frame = np.asarray(client.get_render(height, width))
        renders += 1
        if previous is not None and previous.shape == frame.shape:
            if np.array_equal(previous, frame):
                return renders
        previous = frame
        if time.time() - started > timeout:
            print(
                f"[viewer] compare: the page kept changing for {timeout:g}s; "
                "recording anyway",
                flush=True,
            )
            return renders


def compare_record_playheads(
    seconds: float, scene_fps: float, file_fps: float, length: int, speed: float = 1.0
) -> list[int]:
    """The playhead positions a compare recording steps through.

    One video frame per file frame; the playhead moves ``scene_fps /
    file_fps * speed`` episode steps between frames, so the file plays the
    episodes at ``speed`` times their recorded speed whatever rate it is
    written at, and wraps around when the episodes end.

    Args:
        seconds: Length of the video.
        scene_fps: The rate the episodes were recorded at.
        file_fps: The rate the file is written at.
        length: Frames in the longest slot.
        speed: Playback speed (2 = twice real time).

    Returns:
        The playheads, one per video frame.
    """
    count = max(1, int(round(float(seconds) * float(file_fps))))
    step = float(scene_fps) / max(1e-6, float(file_fps)) * max(1e-6, float(speed))
    span = max(1, int(length))
    return [int(round(i * step)) % span for i in range(count)]


def record_compare(
    show,
    client,
    make_sink,
    *,
    playheads,
    size: tuple[int, int],
    log_every: int = RECORD_LOG_EVERY,
) -> tuple[int, tuple[int, int]]:
    """Pose the scene at each playhead, render it in the browser, write it.

    No wall clock: the scene is posed, only then is the client asked for a
    picture, so the file is smooth at its frame rate however slowly the
    software rasteriser draws. The first render decides the real size (a
    human's browser may not return what was asked for) and ``make_sink``
    is called with that.

    Args:
        show: Called with a playhead to pose every slot at it.
        client: Anything with ``get_render(height, width)``.
        make_sink: Called once with ``(width, height)``; returns an object
            with ``write(frame)`` and ``close()``.
        playheads: The playheads to record, in order.
        size: The requested ``(width, height)``.
        log_every: Print progress every this many frames (0 = never).

    Returns:
        ``(frames written, the size they were written at)``.

    Raises:
        RuntimeError: The browser changed the frame size mid-recording.
    """
    width, height = int(size[0]), int(size[1])
    playheads = list(playheads)
    sink = None
    actual = (width, height)
    written = 0
    started = time.time()
    try:
        for index, playhead in enumerate(playheads):
            show(int(playhead))
            frame = np.asarray(client.get_render(height, width))
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[..., :3]
            frame = np.ascontiguousarray(frame.astype(np.uint8, copy=False))
            got = (int(frame.shape[1]), int(frame.shape[0]))
            if sink is None:
                actual = got
                if got != (width, height):
                    print(
                        f"[viewer] the browser returned {got[0]}x{got[1]}, not "
                        f"the requested {width}x{height}; recording at that",
                        flush=True,
                    )
                sink = make_sink(actual)
            elif got != actual:
                raise RuntimeError(
                    f"frame {index} came back {got[0]}x{got[1]} after "
                    f"{actual[0]}x{actual[1]}: do not resize the browser "
                    "window while recording"
                )
            sink.write(frame)
            written += 1
            if log_every and written % int(log_every) == 0:
                rate = written / max(1e-6, time.time() - started)
                print(
                    f"[viewer] recorded {written}/{len(playheads)} frames "
                    f"({rate:.1f} fps captured)",
                    flush=True,
                )
    finally:
        if sink is not None:
            sink.close()
    return written, actual


def record_compare_session(
    server,
    port: int,
    show,
    home,
    *,
    scene_fps: float,
    length: int,
    path,
    offsets=(),
    seconds: float = 10.0,
    size=(1920, 1080),
    fps: float = 0.0,
    browser: str = "headless",
    stay: bool = False,
    view: str = "overview",
    zoom: float = 1.0,
    speed: float = 1.0,
    start_timeout: float = 60.0,
) -> int:
    """Write a video of a built compare scene through a browser.

    viser has no server-side renderer: a headless Chrome is started on the
    page (or a browser somebody already has open is used), parked at the
    overview camera, and asked for one picture per video frame while the
    playhead is stepped by hand (:mod:`bigym.vr.viewer.video`).

    Args:
        server: The viser server the scene is in.
        port: Its port.
        show: Called with a playhead to pose every slot at it.
        home: The overview ``(position, look_at)`` the camera records from.
        scene_fps: The rate the episodes were recorded at.
        length: Frames in the longest slot.
        path: Destination mp4.
        offsets: Every slot's offset; the lens is narrowed to what they
            subtend from ``home`` (viser's default 80 degrees leaves four
            scenes in a small band in the middle of a 16:9 frame).
        seconds: Length of the video.
        size: ``(width, height)`` of the frames.
        fps: Frame rate of the file (0 = ``scene_fps``, real time).
        browser: ``headless``, ``none`` (wait for a person's browser) or a
            browser executable.
        stay: Unused here; :func:`run_compare` reads it.
        view: ``overview`` or ``oblique`` (:func:`compare_record_pose`).
        zoom: Narrows the lens (2 = twice as close-looking).
        speed: Playback speed (2 = twice real time).
        start_timeout: Seconds to wait for a browser to connect.

    Returns:
        The number of frames written (0 when no browser turned up).
    """
    del stay
    file_fps = float(fps) if float(fps) > 0.0 else float(scene_fps)
    width, height = int(size[0]), int(size[1])
    playheads = compare_record_playheads(seconds, scene_fps, file_fps, length, speed)
    video.hide_gui_chrome(server)
    binary = video.browser_binary(browser)
    url = f"http://localhost:{port}"
    process = None
    if binary:
        client = video.wait_for_client(server, timeout=video.BROWSER_GRACE_SECONDS)
        if client is None:
            process = video.launch_browser(binary, url, video.browser_window(size))
            client = (
                video.wait_for_client(server, timeout=start_timeout)
                if process is not None
                else None
            )
    else:
        client = video.wait_for_client(server, timeout=start_timeout)
    if client is None:
        print("[viewer] compare: no browser connected, nothing recorded", flush=True)
        video.stop_browser(process)
        return 0
    pose = compare_record_pose(offsets, view, home=home)
    client.camera.position = pose[0]
    client.camera.look_at = pose[1]
    if len(offsets):
        pad = COMPARE_PAD_TIGHT if view == "oblique" else COMPARE_PAD
        fov = box_fov(
            pose[0], pose[1], compare_box(offsets, pad), width / max(1, height)
        )
        fov = float(
            np.clip(fov / max(0.1, float(zoom)), RECORD_FOV_MIN, RECORD_FOV_MAX)
        )
        client.camera.fov = fov
        print(
            f"[viewer] compare: recording {view} view from ({pose[0][0]:.2f}, "
            f"{pose[0][1]:.2f}, {pose[0][2]:.2f}) looking at ({pose[1][0]:.2f}, "
            f"{pose[1][1]:.2f}, {pose[1][2]:.2f}), lens {np.degrees(fov):.0f} deg, "
            f"speed x{float(speed):g}",
            flush=True,
        )
    show(int(playheads[0]))
    renders = settle_render(client, (width, height))
    print(
        f"[viewer] compare: page settled after {renders} render(s); recording",
        flush=True,
    )
    print(
        f"[viewer] compare: recording {len(playheads)} frames at {width}x{height} "
        f"to {path} ({file_fps:g} fps, {len(playheads) / file_fps:.1f}s of video)",
        flush=True,
    )
    started = time.time()
    try:
        written, actual = record_compare(
            show,
            client,
            lambda real: video.VideoSink(path, real, file_fps),
            playheads=playheads,
            size=(width, height),
        )
    finally:
        video.stop_browser(process)
    elapsed = time.time() - started
    print(
        f"[viewer] compare: wrote {written} frames at {actual[0]}x{actual[1]} "
        f"to {path} in {elapsed:.1f}s ({written / max(1e-6, elapsed):.1f} fps "
        "captured)",
        flush=True,
    )
    return written


def run_compare(
    request: CompareRequest,
    *,
    port: int,
    build_env,
    load_metadata,
    camera_panels,
    lighting_prefs: dict,
    exit_after: float = 0.0,
    timeout: float = 3600.0,
    record: dict | None = None,
) -> None:
    """Play N policies of one task side by side until the user leaves.

    The server comes up first, with the Compare folder in it, so the slots
    that still need a rollout are produced with a progress bar each and a
    working Leave button (:func:`wait_for_missing`); the scene is built into
    that same server once every slot has its episode.

    The scene is laid out as the request's ``layout`` asks (columns along
    +y, rows receding along -x), the camera starts where the whole grid is
    in view, the Compare folder's *focus* dropdown flies every client to one
    slot, and the camera panels start hidden behind its *camera panels*
    checkbox.

    Args:
        request: A resolved request; slots with no stored episode on the
            seed are rolled out first.
        port: The viser port.
        build_env: ``view_demos.build_env``.
        load_metadata: ``view_demos.load_metadata``.
        camera_panels: ``view_demos.CameraPanels``.
        lighting_prefs: The viewer's lighting preferences.
        exit_after: Serve this many seconds, then return (0 = until the
            Leave button or Ctrl-C). It bounds the whole compare, the
            rollouts included, so a headless run always terminates.
        timeout: Give up on a rollout that takes longer than this.
        record: None, or what :func:`record_compare_session` needs to write
            a video of the scene once it is built (``path``, ``seconds``,
            ``size``, ``fps``, ``browser``, ``view``, ``zoom``, ``speed``,
            ``stay``), plus ``label_scale`` for the billboards; the server
            closes after the file unless ``stay`` is true.
    """
    server = compare_server(port)
    legend = CompareLegend(server, request)
    print(f"[viewer] open http://localhost:{port} (ssh -L {port}:localhost:{port})")
    started = time.time()
    if not wait_for_missing(request, legend, timeout=timeout, deadline=exit_after):
        server.stop()
        return
    scene = build_compare_scene(
        request,
        port=port,
        build_env=build_env,
        load_metadata=load_metadata,
        lighting_prefs=lighting_prefs,
        server=server,
        legend=legend,
    )
    server, slots, fps = scene["server"], scene["slots"], scene["fps"]
    legend = scene["legend"]
    mujoco = scene["mujoco"]
    offsets = scene["offsets"]
    length = max(int(s["qpos"].shape[0]) for s in slots)
    state = {"frame": 0, "dirty": True, "leave": False, "camera": 0}
    names = [
        f"slot {i} · {s['slot'].name} {s['slot'].version_name}"
        for i, s in enumerate(slots)
    ]
    home = compare_camera_pose(offsets)

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "frame", min=0, max=max(1, length - 1), step=1, initial_value=0
        )
        playing = server.gui.add_checkbox("play", initial_value=True)
        fps_slider = server.gui.add_slider(
            "fps",
            min=1,
            max=max(100, int(round(fps))),
            step=1,
            initial_value=int(round(fps)),
        )
    camera_dd = server.gui.add_dropdown(
        "camera slot", options=names, initial_value=names[0]
    )
    panels = camera_panels(
        server,
        slots[0]["inner"],
        slots[0]["model"],
        slots[0]["data"],
        scene["camera_keys"],
        scene["camera_shape"],
    )
    # Compare mode is about the scene, so the panels start hidden; the
    # Compare folder's checkbox is what turns them on, and the camera slot
    # dropdown still says whose view they then show.
    panels.enabled.value = False
    legend.leave.on_click(lambda _: state.update(leave=True))
    frame_slider.on_update(
        lambda _: state.update(frame=int(frame_slider.value), dirty=True)
    )
    camera_dd.on_update(lambda _: state.update(camera=names.index(camera_dd.value)))

    @server.on_client_connect
    def _(client) -> None:
        client.camera.position = home[0]
        client.camera.look_at = home[1]

    # Clients that were already watching the progress bars do not fire
    # on_client_connect again, so they are moved by hand.
    fly_clients_to(server, *home)
    print(
        f"[viewer] compare: task {request.task} · seed {request.seed} · "
        f"{len(slots)} scenes built · "
        f"{grid_note(len(slots), request.gap, request.layout)}",
        flush=True,
    )
    print(
        f"[viewer] compare camera: at ({home[0][0]:.2f}, {home[0][1]:.2f}, "
        f"{home[0][2]:.2f}) looking at ({home[1][0]:.2f}, {home[1][1]:.2f}, "
        f"{home[1][2]:.2f}) — every scene in view",
        flush=True,
    )
    head_time = time.time()
    shown_camera = 0
    camera_at = 0.0
    # Only a change of the Compare folder's checkbox writes to the panels'
    # own "show", so the two controls do not fight each other.
    cameras_on = False

    def show(frame: int) -> None:
        """Pose every slot at ``frame`` and say so on its plate and the legend."""
        for i, entry in enumerate(slots):
            here = pose_slot(entry, frame, mujoco)
            text = slot_status_text(entry, here)
            # The billboard rides above the pelvis, so it stays with the
            # robot as it walks instead of marking the origin.
            place_billboard(entry)
            entry["billboard"].write(entry["slot"].label, text, slot_colour(i))
            legend.set_status(i, text)
        legend.refresh()

    try:
        if record:
            scale = float(record.get("label_scale", RECORD_LABEL_SCALE) or 1.0)
            for entry in slots:
                entry["billboard"].width *= scale
                entry["billboard"].drawn = ""  # redrawn at the new size by show()
            written = record_compare_session(
                server,
                port,
                show,
                home,
                scene_fps=fps,
                length=length,
                offsets=offsets,
                **{k: v for k, v in record.items() if k != "label_scale"},
            )
            if not record.get("stay"):
                return
            state.update(frame=0, dirty=True)
            print(
                f"[viewer] compare: {written} frames recorded; still serving",
                flush=True,
            )
        while True:
            if state["leave"]:
                print("[viewer] leaving compare mode", flush=True)
                break
            now = time.time()
            if exit_after > 0.0 and now - started > exit_after:
                print(f"[viewer] --exit-after-seconds {exit_after:g} elapsed, closing")
                break
            rate = max(1.0, float(fps_slider.value))
            if bool(playing.value) and now - head_time >= 1.0 / rate:
                head_time = now
                state["frame"] = (state["frame"] + 1) % length
                frame_slider.value = state["frame"]
                state["dirty"] = True
            focus = legend.take_focus()
            if focus is not None and 0 <= focus < len(slots):
                entry = slots[focus]
                pose = focus_pose(
                    entry["offset"],
                    pelvis_position(entry["data"], entry["pelvis"]),
                )
                moved = fly_clients_to(server, *pose)
                print(
                    f"[viewer] compare: flying {moved} client(s) to "
                    f"{names[focus]} at ({pose[0][0]:.2f}, {pose[0][1]:.2f}, "
                    f"{pose[0][2]:.2f})",
                    flush=True,
                )
            if state["camera"] != shown_camera:
                shown_camera = int(state["camera"])
                panels.rebind(
                    slots[shown_camera]["inner"],
                    slots[shown_camera]["model"],
                    slots[shown_camera]["data"],
                )
                print(
                    f"[viewer] camera panels follow {names[shown_camera]}", flush=True
                )
            if state["dirty"]:
                show(int(state["frame"]))
                state["dirty"] = False
            if bool(legend.cameras.value) != cameras_on:
                cameras_on = bool(legend.cameras.value)
                panels.enabled.value = cameras_on
                print(
                    f"[viewer] compare: camera panels {'on' if cameras_on else 'off'}",
                    flush=True,
                )
            if now - camera_at > 0.2:
                camera_at = now
                panels.update()
            time.sleep(0.004)
    finally:
        panels.close()
        server.stop()
        for entry in slots:
            entry["env"].close()
