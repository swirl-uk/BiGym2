"""What a coding-agent run directory holds: cells, policy versions, rollouts.

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
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from bigym.loco.agent.snapshots import submission_entry, version_dir
from bigym.loco.agent.transcript import TOKEN_FIELDS
from bigym.loco.eval.protocol import EVAL_SEED_BASE
from bigym.vr.viewer.episodes import batch_success, episode_files

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


EVAL_SEED_LO = EVAL_SEED_BASE


# The Rollouts picker's first level: what the user calls each kind, in the
# order the dropdown offers them.
ROLLOUT_KINDS = (("Evaluation", "eval"), ("Development", "dev"), ("Replays", "replays"))


KIND_TITLES = {kind: title for title, kind in ROLLOUT_KINDS}


VERSION_LAST = "last"


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
SHOW_TEMPLATE: bool = False


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
    """The version the session submitted (see :func:`submission_entry`)."""
    rows = policy_rows(cell) if rows is None else rows
    entry = submission_entry(rows)
    return entry["version"] if entry else None


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


def policy_file(cell: Path, row: dict) -> Path:
    """Where a version's ``policy.py`` lives (``policies/vNNN/policy.py``)."""
    name = str(row.get("dir") or f"v{int(row['version']):03d}")
    return Path(cell) / "policies" / name / "policy.py"


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

    A session's first snapshot (v000, or v001 in older cells) is taken before
    the agent wrote anything, so a compare of three templates is a compare of
    one policy with itself; saying so on the label costs nothing and explains
    three identical robots.

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
