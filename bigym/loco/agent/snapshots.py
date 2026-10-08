"""Policy version snapshots: every distinct ``policy.py`` an agent writes.

The environment server runs a :class:`PolicyWatcher` in a daemon thread for
the whole session. It polls the sandbox and records each new content hash of
``policy.py`` (plus the helper modules sitting next to it at that moment) as
``<cell>/policies/vNNN/``, described in ``<cell>/policies/index.json``:

.. code-block:: json

    {"task": "move_plate",
     "versions": [{"version": 1, "dir": "v001", "sha256": "...",
                   "time": "2026-01-02T03:04:05+00:00", "ts": 1767322985.4,
                   "trigger": "write", "helpers": ["add_vision.py"],
                   "train_run": "sandbox/runs/20260102_030405.json",
                   "train_success": 0.4, "train_episodes": 10,
                   "train_seeds": [3, 7, 11],
                   "command_index": 42, "budget_used": 19854,
                   "tokens": {"input": 400, "cached": 600,
                              "output": 50, "reasoning": 30},
                   "elapsed_s": 1618.5}]}

``trigger`` is ``write`` when only the file changed, ``run`` once the agent
executed that exact content through ``run_episodes.py`` (the runner leaves
``runs/<stamp>.json`` and ``runs/<stamp>_policy.py`` behind, which the
watcher matches by content hash), and ``submission`` for the content at the
end of the session. A content hash appears exactly once: a later trigger
updates the entry it already has, so the index is the agent's edit history
with the score its own runs measured.

The training outcome of a version does not depend on the agent using the
runner: every episode the server ran is in ``<cell>/ledger.jsonl``, and an
``episode_end`` event belongs to the version whose snapshot time is the
latest one at or before the event. ``train_episodes`` / ``train_success`` /
``train_seeds`` are that attribution; ``train_run`` still links the runner
records when the agent used them.

Each version also carries what the session had spent when it was written:
``command_index`` (commands the harness had run), ``tokens`` (cumulative
``input`` / ``cached`` / ``output`` / ``reasoning``, read from the raw
harness stream through :mod:`bigym.loco.agent.transcript`), ``budget_used``
(environment steps) and ``elapsed_s`` (seconds since the server started).
Cost is not stored: it is priced at display time.

:func:`rebuild_index` recomputes all of those from the ledger and the raw
stream without touching the version directories, which upgrades a cell
recorded before these fields existed.

``index.json`` is rewritten atomically because the demo viewer reads it live.
"""

from __future__ import annotations

import ast
import bisect
import datetime
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator

from . import transcript
from .cli import EnvToolsConfig, PoliciesConfig, parse_command

INDEX_NAME = "index.json"
POLICY_NAME = "policy.py"
TEMPLATE_NAME = "initial_policy.py"  # the builder's untouched policy, kept in the cell
POLICIES_DIR = "policies"
SANDBOX_CONFIG = "sandbox_config.json"
LEDGER_NAME = "ledger.jsonl"
BUDGET_NAME = "budget.json"
# Development rollouts (server.py writes <cell>/dev/vNNN_<stamp>/batch/).
DEV_DIR = "dev"
# Sibling modules the agent did not write: the runner entry point the sandbox
# ships. Everything under harness/ is out of reach anyway (only the files
# directly beside policy.py are snapshotted).
NOT_HELPERS = frozenset({POLICY_NAME, "run_episodes.py"})
HELPER_SKIP_DIRS = ("harness",)


def sha256_text(data: bytes) -> str:
    """Return the sha256 hex digest of a byte string.

    Args:
        data: The bytes to hash.

    Returns:
        The hex digest.
    """
    return hashlib.sha256(data).hexdigest()


def now_stamp() -> str:
    """Return the current local time as an ISO-8601 string with its offset."""
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def version_name(version: int) -> str:
    """Return the directory name of a version (``v007``).

    Args:
        version: Version number.

    Returns:
        The zero-padded directory name.
    """
    return f"v{int(version):03d}"


def read_index(policies_dir: Path) -> dict:
    """Read ``policies/index.json``, or an empty index when there is none.

    Args:
        policies_dir: The ``policies`` directory of a cell.

    Returns:
        The parsed index (``{"task": ..., "versions": [...]}``).
    """
    path = Path(policies_dir) / INDEX_NAME
    try:
        loaded = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"task": None, "versions": []}
    if not isinstance(loaded, dict):
        return {"task": None, "versions": []}
    loaded.setdefault("versions", [])
    return loaded


def write_index(policies_dir: Path, index: dict) -> Path:
    """Write ``policies/index.json`` atomically.

    The viewer polls this file while a session runs, so it is written to a
    temporary file in the same directory and renamed over the old one: a
    reader sees either the previous index or the new one, never a half file.

    Args:
        policies_dir: The ``policies`` directory of a cell.
        index: The index to write.

    Returns:
        The path written.
    """
    policies_dir = Path(policies_dir)
    policies_dir.mkdir(parents=True, exist_ok=True)
    path = policies_dir / INDEX_NAME
    tmp = policies_dir / f"{INDEX_NAME}.tmp"
    tmp.write_text(json.dumps(index, indent=1, default=str))
    os.replace(tmp, path)
    return path


def version_dir(cell: Path, version: int, kind: str = POLICIES_DIR) -> Path:
    """Return ``<cell>/<kind>/vNNN``, tolerating unpadded directory names.

    Args:
        cell: The cell directory.
        version: Version number.
        kind: ``policies``, ``eval`` or ``replays``.

    Returns:
        The existing directory of that version, or the padded name.
    """
    parent = Path(cell) / kind
    for candidate in sorted(parent.glob("v*")):
        name = candidate.name[1:]
        if candidate.is_dir() and name.isdigit() and int(name) == int(version):
            return candidate
    return parent / version_name(version)


def load_version(cell: Path, version: int) -> Path:
    """Return the ``policy.py`` of one recorded version.

    Helper modules are stored next to it, so
    :func:`bigym.loco.agent.episode.load_policy` on the returned path
    resolves the imports the agent wrote.

    Args:
        cell: The cell directory.
        version: Version number.

    Returns:
        The path of that version's ``policy.py``.

    Raises:
        FileNotFoundError: No such version in this cell.
    """
    cell = Path(cell).resolve()
    path = version_dir(cell, version) / POLICY_NAME
    if not path.exists():
        known = [
            entry.get("version")
            for entry in read_index(cell / POLICIES_DIR)["versions"]
        ]
        raise FileNotFoundError(
            f"no policy version {version} in {cell} (expected {path}); "
            f"recorded versions: {known or 'none'}"
        )
    return path


def resolve_policy(cell: Path, version: int | None, policy: Path | None, kind: str):
    """Resolve ``--version`` / ``--policy`` into a policy file and output dir.

    Args:
        cell: The cell directory.
        version: Version number, or None.
        policy: An explicit policy file, or None.
        kind: ``replays`` or ``eval``.

    Returns:
        ``(policy_path, out_dir, version)``.

    Raises:
        ValueError: Neither or both of version and policy were given.
        FileNotFoundError: The policy file does not exist.
    """
    if (version is None) == (policy is None):
        raise ValueError("give exactly one of --version N and --policy PATH")
    if version is not None:
        policy_path = load_version(cell, int(version))
        return policy_path, cell / kind / version_name(int(version)), int(version)
    assert policy is not None
    policy_path = Path(policy).resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"no such policy file: {policy_path}")
    digest = sha256_text(policy_path.read_bytes())
    return policy_path, cell / kind / f"policy_{digest[:8]}", None


def read_sandbox_config(cell: Path) -> dict:
    """Read ``<cell>/sandbox_config.json``, or an empty dict when absent.

    The sandbox builder writes the session's resolved configuration there:
    the task, the seeds, the demonstrations and the environment settings
    (:func:`read_env_tools`).

    Args:
        cell: The cell directory.

    Returns:
        The parsed configuration, or ``{}``.
    """
    try:
        loaded = json.loads((Path(cell) / SANDBOX_CONFIG).read_text())
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def read_env_tools(cell: Path) -> EnvToolsConfig:
    """Return the environment settings a cell's session ran against.

    Args:
        cell: The cell directory.

    Returns:
        The ``env_tools`` block of ``sandbox_config.json``, whole. Cells built
        before that block existed recorded only the observation settings at
        the top level, and were always scored with the default env tools.
    """
    config = read_sandbox_config(cell)
    recorded = config.get("env_tools")
    if recorded is None:
        names = ("interface", "tier", "image_cap", "pitch", "ik", "hand_pos")
        recorded = {name: config[name] for name in names if name in config}
    return EnvToolsConfig(**recorded)


def cell_task(cell: Path) -> str | None:
    """Return the task a cell was run on.

    Args:
        cell: The cell directory.

    Returns:
        The task name from ``sandbox_config.json`` or ``policies/index.json``,
        or None when neither names one.
    """
    cell = Path(cell)
    task = read_sandbox_config(cell).get("task")
    if not task:
        task = read_index(cell / POLICIES_DIR).get("task")
    return str(task) if task else None


# ------------------------------------------------------------------ time


def stamp_of(when: float) -> str:
    """Return a unix time as a local ISO-8601 string with its offset.

    Args:
        when: Unix time.

    Returns:
        The stamp, at one-second resolution.
    """
    return (
        datetime.datetime.fromtimestamp(float(when))
        .astimezone()
        .isoformat(timespec="seconds")
    )


def compact_stamp(when: float | str | None) -> str:
    """Return a time as ``YYYYMMDD_HHMMSS``, the directory-name form.

    Args:
        when: Unix time, an ISO-8601 stamp, or None for now.

    Returns:
        The compact local stamp.
    """
    moment = parse_time(when)
    if moment is None:
        moment = time.time()
    return datetime.datetime.fromtimestamp(moment).strftime("%Y%m%d_%H%M%S")


def parse_time(stamp: float | str | None) -> float | None:
    """Parse a version time into a unix time.

    Args:
        stamp: A unix time, an ISO-8601 stamp, or None.

    Returns:
        The unix time, or None when it cannot be read.
    """
    if isinstance(stamp, bool) or stamp is None:
        return None
    if isinstance(stamp, (int, float)):
        return float(stamp)
    try:
        return datetime.datetime.fromisoformat(str(stamp)).timestamp()
    except ValueError:
        return None


def entry_time(entry: dict) -> float | None:
    """Return the unix time a version was recorded at.

    ``ts`` is written next to the human-readable ``time``; an index written
    before ``ts`` existed falls back to parsing ``time``.

    Args:
        entry: An index entry.

    Returns:
        The unix time, or None when the entry has neither field.
    """
    when = parse_time(entry.get("ts"))
    return when if when is not None else parse_time(entry.get("time"))


# ------------------------------------------------------------- the ledger


def read_ledger(ledger_dir: Path) -> list[dict]:
    """Read every event of a cell's ``ledger.jsonl``.

    Args:
        ledger_dir: The directory the server writes its ledger into (the
            cell).

    Returns:
        The events, in the order they were appended; ``[]`` when there is no
        ledger.
    """
    path = Path(ledger_dir) / LEDGER_NAME
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    return list(parse_ledger_lines(text))


def parse_ledger_lines(text: str) -> Iterator[dict]:
    """Yield the JSON objects of ledger text, skipping unreadable lines.

    Args:
        text: Ledger text.

    Yields:
        One event per readable line.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            yield event


def ledger_start(events: Iterable[dict]) -> float | None:
    """Return the time the server started, as the ledger records it.

    Args:
        events: Ledger events.

    Returns:
        The ``server_start`` timestamp, the first timestamp of any event, or
        None for an empty ledger.
    """
    first = None
    for event in events:
        when = parse_time(event.get("ts"))
        if when is None:
            continue
        if event.get("event") == "server_start":
            return when
        if first is None:
            first = when
    return first


def ledger_budget(events: Iterable[dict], when: float | None) -> int:
    """Return the interaction steps spent by a point in time.

    Args:
        events: Ledger events.
        when: Unix time, or None for the whole ledger.

    Returns:
        The largest step count any event up to that time reports, or 0.
    """
    used = 0
    for event in events:
        moment = parse_time(event.get("ts"))
        if when is not None and (moment is None or moment > float(when)):
            continue
        value = event.get("budget_used")
        if value is None:
            value = (event.get("budget") or {}).get("used")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            used = max(used, int(value))
    return used


def attribute_training(versions: list[dict], events: Iterable[dict]) -> bool:
    """Attribute the ledger's episodes to the version that was current.

    An ``episode_end`` event belongs to the version whose snapshot time is
    the latest one at or before the event: that is the ``policy.py`` the
    agent had on disk while the episode ran, whether or not it used the
    runner. Episodes that ran before the first snapshot belong to no
    version and are dropped.

    Args:
        versions: The index entries, which are updated in place.
        events: Ledger events.

    Returns:
        True when any entry changed.
    """
    ends = [e for e in events if e.get("event") == "episode_end"]
    ordered = [(entry_time(v), v) for v in versions]
    ordered = sorted((t, i) for i, (t, _) in enumerate(ordered) if t is not None)
    if not ends or not ordered:
        return False
    times = [t for t, _ in ordered]
    buckets: dict[int, list[dict]] = {}
    for event in ends:
        when = parse_time(event.get("ts"))
        if when is None:
            continue
        place = bisect.bisect_right(times, when) - 1
        if place >= 0:
            buckets.setdefault(place, []).append(event)
    changed = False
    for place, (_, index) in enumerate(ordered):
        got = buckets.get(place, [])
        seeds = sorted(
            {int(e["seed"]) for e in got if isinstance(e.get("seed"), (int, float))}
        )
        fields = {
            "train_episodes": len(got),
            "train_success": (
                sum(float(e.get("success", 0) or 0) for e in got) / len(got)
                if got
                else None
            ),
            "train_seeds": seeds,
        }
        entry = versions[index]
        if any(entry.get(key) != value for key, value in fields.items()):
            entry.update(fields)
            changed = True
    return changed


# ------------------------------------------------------- imported helpers


def imported_modules(source: str | bytes) -> set[str]:
    """Return the top-level module names a Python source imports.

    Args:
        source: Python source text.

    Returns:
        The imported top-level names; an empty set when the source does not
        parse (a half-written file, or Python this interpreter cannot read).
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
            else:  # from . import helper
                for alias in node.names:
                    names.add(alias.name.split(".")[0])
    return names


def scan_helpers(sources: dict[str, bytes], entry: str = POLICY_NAME) -> list[str]:
    """Return the sibling modules a policy really imports.

    Agents leave dozens of scratch scripts next to ``policy.py``; all of
    them are copied into the version directory (evaluation must be able to
    import whatever the policy needs), but only the modules reachable from
    ``policy.py``'s imports describe the policy. The scan is static and
    transitive: a module the policy imports is followed into its own
    imports.

    Args:
        sources: File name to source bytes, for every sibling module.
        entry: The module the scan starts at.

    Returns:
        The imported sibling file names, sorted.
    """
    if entry not in sources:
        return []
    found: set[str] = set()
    queue = [entry]
    while queue:
        for module in imported_modules(sources[queue.pop()]):
            name = f"{module}.py"
            if name == entry or name in found or name not in sources:
                continue
            found.add(name)
            queue.append(name)
    return sorted(found)


def read_sources(directory: Path) -> dict[str, bytes]:
    """Read every module of a directory as ``{file name: bytes}``.

    Args:
        directory: A version directory (or a sandbox).

    Returns:
        The sources, skipping files that cannot be read.
    """
    out: dict[str, bytes] = {}
    for path in sorted(Path(directory).glob("*.py")):
        try:
            out[path.name] = path.read_bytes()
        except OSError:
            continue
    return out


# ----------------------------------------------------------- provenance


def zero_tokens() -> dict[str, int]:
    """Return a token counter with every field at zero."""
    return {field: 0 for field in transcript.TOKEN_FIELDS}


class StreamProvenance:
    """The harness stream's counters, queryable at a point in time.

    The raw stream under ``<cell>/raw/`` is read through the transcript
    adapters, so every harness the benchmark supports is covered by the same
    code. A record the adapter gives a time is placed at that time; a record
    without one (the Codex event stream carries no times at all) is placed at
    the file's modification time, the only thing known about it.
    """

    def __init__(self, ledger_dir: Path):
        """Read a cell's raw stream, if it has one.

        Args:
            ledger_dir: The cell the server writes into.
        """
        self.harness: str | None = None
        self.times: list[float] = []
        self.marks: list[tuple[int, tuple[int, ...]]] = []
        self._load(Path(ledger_dir))

    def _load(self, ledger_dir: Path) -> None:
        """Read the stream into cumulative marks, one per record."""
        found = transcript.find_raw(ledger_dir)
        if found is None:
            return
        self.harness, path = found
        try:
            fallback = path.stat().st_mtime
            records = list(transcript.ADAPTERS[self.harness](path))
        except (OSError, ValueError, KeyError):
            return
        commands = 0
        tokens = zero_tokens()
        last = None
        for record in records:
            when = record.get("t")
            when = float(when) if isinstance(when, (int, float)) else fallback
            when = when if last is None else max(when, last)
            last = when
            kind = record.get("kind")
            if kind == "command":
                commands += 1
            elif kind == "usage":
                for field, value in (record.get("tokens") or {}).items():
                    if field in tokens:
                        tokens[field] += int(value or 0)
            self.times.append(when)
            self.marks.append(
                (commands, tuple(tokens[f] for f in transcript.TOKEN_FIELDS))
            )

    def at(self, when: float | None = None) -> dict:
        """Return the counters as of a point in time.

        Args:
            when: Unix time, or None for the whole stream so far.

        Returns:
            ``{"command_index": int, "tokens": {...}}``.
        """
        mark: tuple[int, tuple[int, ...]] | None = None
        if self.marks:
            if when is None:
                mark = self.marks[-1]
            else:
                place = bisect.bisect_right(self.times, float(when)) - 1
                mark = self.marks[place] if place >= 0 else None
        if mark is None:
            return {"command_index": 0, "tokens": zero_tokens()}
        commands, tokens = mark
        return {
            "command_index": int(commands),
            "tokens": dict(
                zip(transcript.TOKEN_FIELDS, (int(t) for t in tokens), strict=True)
            ),
        }


class PolicyWatcher:
    """Record every distinct ``policy.py`` of one session under ``policies/``.

    The watcher is a poller, not a file-system notifier: an agent edits the
    file with whatever tool it likes (some rewrite it through a temporary
    file, which no single inotify event describes), and one second of
    resolution is enough to reconstruct the edit history.
    """

    def __init__(
        self,
        sandbox_dir: Path,
        policies_dir: Path,
        poll_seconds: float = 1.0,
        task: str | None = None,
        ledger_dir: Path | None = None,
        started: float | None = None,
        on_version: Callable[[dict], None] | None = None,
    ):
        """Bind the watcher to one sandbox and its output directory.

        Args:
            sandbox_dir: The agent's sandbox (holds ``policy.py`` and ``runs/``).
            policies_dir: Where versions are written (``<cell>/policies``).
            poll_seconds: Seconds between polls.
            task: Task name recorded in the index.
            ledger_dir: The cell holding ``ledger.jsonl``, ``budget.json`` and
                ``raw/`` (default: the parent of ``policies_dir``).
            started: Unix time the server started, for ``elapsed_s``.
            on_version: Called with the index entry whose content is the one
                on disk, whenever that changes. The server uses it to tell
                its workers which version their episodes belong to.
        """
        self.sandbox = Path(sandbox_dir)
        self.policies = Path(policies_dir)
        self.ledger_dir = Path(ledger_dir) if ledger_dir else self.policies.parent
        self.poll_seconds = float(poll_seconds)
        self.started = float(started) if started is not None else time.time()
        self.on_version = on_version
        self.index = read_index(self.policies)
        if task:
            self.index["task"] = str(task)
        self.seen_runs: set[str] = set()
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.thread: threading.Thread | None = None
        self.current: dict | None = None
        self.episodes: list[dict] = []
        self.ledger_offset = 0
        self.announced: int | None = None

    # ------------------------------------------------------------------ files
    @property
    def policy_path(self) -> Path:
        """The sandbox's ``policy.py``."""
        return self.sandbox / POLICY_NAME

    def helper_files(self) -> list[Path]:
        """Return the helper modules sitting next to ``policy.py`` right now.

        Only files directly beside ``policy.py`` count: the harness the
        sandbox ships lives in its own directory and is not the agent's code.
        """
        out = []
        for path in sorted(self.sandbox.glob("*.py")):
            if path.name in NOT_HELPERS or not path.is_file():
                continue
            if path.parent.name in HELPER_SKIP_DIRS:
                continue
            out.append(path)
        return out

    def versions(self) -> list[dict]:
        """Return the recorded versions, oldest first."""
        return list(self.index["versions"])

    def entry_for(self, sha: str) -> dict | None:
        """Return the index entry with this content hash, if any.

        Args:
            sha: A policy content hash.

        Returns:
            The entry, or None.
        """
        for entry in self.index["versions"]:
            if entry.get("sha256") == sha:
                return entry
        return None

    # ---------------------------------------------------------------- writing
    def add_version(self, content: bytes, trigger: str) -> dict:
        """Write a new version directory and append it to the index.

        Args:
            content: The exact bytes of that version's ``policy.py``.
            trigger: ``write``, ``run`` or ``submission``.

        Returns:
            The new index entry.
        """
        versions = self.index["versions"]
        # The first snapshot is taken before the agent has written anything;
        # when it is still the builder's template it is numbered v000 and
        # marked, so the agent's own first version is v001 and viewers can
        # leave the template out.
        is_template = not versions and content == self.template_bytes()
        number = (
            0
            if is_template
            else (max((int(v.get("version", 0)) for v in versions), default=0) + 1)
        )
        directory = self.policies / version_name(number)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / POLICY_NAME).write_bytes(content)
        # Every sibling module is copied (the policy must still import at
        # evaluation time); only the ones it imports are reported.
        sources = {POLICY_NAME: content}
        for helper in self.helper_files():
            try:
                shutil.copy2(helper, directory / helper.name)
                sources[helper.name] = (directory / helper.name).read_bytes()
            except OSError:
                continue
        now = time.time()
        entry = {
            "version": number,
            "dir": directory.name,
            "sha256": sha256_text(content),
            "time": stamp_of(now),
            "ts": now,
            "trigger": str(trigger),
            "template": bool(is_template),
            "helpers": scan_helpers(sources),
            "train_run": None,
            "train_success": None,
            "train_episodes": None,
            "train_seeds": None,
        }
        entry.update(self.provenance(now))
        self.index["versions"].append(entry)
        return entry

    def template_bytes(self) -> bytes | None:
        """The builder's template (``<cell>/initial_policy.py``), or None."""
        try:
            return (self.ledger_dir / TEMPLATE_NAME).read_bytes()
        except OSError:
            return None

    def provenance(self, now: float) -> dict:
        """Return what the session had spent at this moment.

        Args:
            now: Unix time of the snapshot.

        Returns:
            ``command_index``, ``tokens``, ``budget_used`` and ``elapsed_s``.
            The counters are read live: the stream and ``budget.json`` hold
            exactly what had happened when the version was written.
        """
        fields = StreamProvenance(self.ledger_dir).at()
        fields["budget_used"] = self.budget_used()
        fields["elapsed_s"] = round(float(now) - self.started, 3)
        return fields

    def budget_used(self) -> int:
        """Return the interaction steps spent so far, from ``budget.json``."""
        try:
            state = json.loads((self.ledger_dir / BUDGET_NAME).read_text())
        except (OSError, ValueError):
            return 0
        value = state.get("used") if isinstance(state, dict) else None
        return int(value) if isinstance(value, (int, float)) else 0

    def run_relpath(self, path: Path) -> str:
        """Path of a runner records file as the index reports it.

        Args:
            path: Absolute path of ``runs/<stamp>.json``.

        Returns:
            The path relative to the cell, or an absolute path when the
            policies directory sits outside the cell.
        """
        cell = self.policies.parent.resolve()
        try:
            return str(path.resolve().relative_to(cell))
        except ValueError:
            return str(path.resolve())

    # ---------------------------------------------------------------- polling
    def scan_runs(self) -> int:
        """Attach the agent's own run results to the versions they scored.

        Each ``runs/<stamp>.json`` names the ``policy.py`` snapshot the
        runner took, so the run is matched to a version by content hash. A
        hash no version has (the agent edited and ran between two polls)
        becomes a version of its own.

        Returns:
            The number of run files newly accounted for.
        """
        runs_dir = self.sandbox / "runs"
        if not runs_dir.is_dir():
            return 0
        found = 0
        for records_path in sorted(runs_dir.glob("*.json")):
            if records_path.name in self.seen_runs:
                continue
            try:
                records = json.loads(records_path.read_text())
            except (OSError, ValueError):
                continue  # still being written; the next poll picks it up
            if not isinstance(records, dict):
                self.seen_runs.add(records_path.name)
                continue
            snapshot = runs_dir / str(records.get("policy_snapshot") or "")
            if not snapshot.is_file():
                self.seen_runs.add(records_path.name)
                continue
            content = snapshot.read_bytes()
            entry = self.entry_for(sha256_text(content))
            if entry is None:
                entry = self.add_version(content, "run")
            elif entry["trigger"] == "write":
                entry["trigger"] = "run"
            episodes = [
                e for e in (records.get("episodes") or []) if isinstance(e, dict)
            ]
            entry["train_run"] = self.run_relpath(records_path)
            entry["train_episodes"] = len(episodes)
            entry["train_success"] = (
                float(sum(float(e.get("success", 0)) for e in episodes) / len(episodes))
                if episodes
                else None
            )
            self.seen_runs.add(records_path.name)
            found += 1
        return found

    def scan_policy(self) -> dict | None:
        """Record ``policy.py`` when its content hash is new.

        Returns:
            The new index entry, or None when nothing changed.
        """
        try:
            content = self.policy_path.read_bytes()
        except OSError:
            return None
        if not content:
            # An empty read is an editor mid-write far more often than a real
            # empty policy; the next poll sees the finished file.
            return None
        entry = self.entry_for(sha256_text(content))
        if entry is not None:
            self.current = entry
            return None
        entry = self.add_version(content, "write")
        self.current = entry
        return entry

    def poll(self) -> bool:
        """Run one scan of the sandbox and rewrite the index when it changed.

        Returns:
            True when the index changed.
        """
        with self.lock:
            changed = bool(self.scan_runs())
            changed = bool(self.scan_policy()) or changed
            self.read_new_ledger()
            changed = self.attribute() or changed
            if changed:
                write_index(self.policies, self.index)
            self.announce()
            return changed

    def read_new_ledger(self) -> int:
        """Read the ledger lines appended since the last poll.

        Only whole lines are consumed, so a line the server is in the middle
        of appending is read on the next poll instead of being dropped.

        Returns:
            The number of new episode results.
        """
        path = self.ledger_dir / LEDGER_NAME
        try:
            with open(path, "rb") as handle:
                handle.seek(self.ledger_offset)
                raw = handle.read()
        except OSError:
            return 0
        cut = raw.rfind(b"\n")
        if cut < 0:
            return 0
        self.ledger_offset += cut + 1
        text = raw[: cut + 1].decode(errors="replace")
        found = [
            event
            for event in parse_ledger_lines(text)
            if event.get("event") == "episode_end"
        ]
        self.episodes.extend(found)
        return len(found)

    def attribute(self) -> bool:
        """Attribute the ledger's episodes to the versions that ran them.

        Returns:
            True when the index changed.
        """
        return attribute_training(self.index["versions"], self.episodes)

    def announce(self) -> None:
        """Tell the server which version the sandbox currently holds."""
        entry, callback = self.current, self.on_version
        if entry is None or callback is None:
            return
        version = int(entry.get("version") or 0)
        if version == self.announced:
            return
        self.announced = version
        try:
            callback(entry)
        except Exception as exc:  # the session never depends on this
            print(f"[policies] version callback failed: {exc}", flush=True)

    def finalize(self) -> dict | None:
        """Record the session's final ``policy.py`` as the submission.

        Returns:
            The submission entry, or None when the sandbox has no policy.
        """
        self.stop()
        with self.lock:
            self.scan_runs()
            try:
                content = self.policy_path.read_bytes()
            except OSError:
                content = b""
            entry = None
            if content:
                entry = self.entry_for(sha256_text(content))
                if entry is None:
                    entry = self.add_version(content, "submission")
                else:
                    entry["trigger"] = "submission"
                self.current = entry
            self.read_new_ledger()
            self.attribute()
            write_index(self.policies, self.index)
            return entry

    # ----------------------------------------------------------------- thread
    def start(self) -> None:
        """Poll once, then keep polling in a daemon thread."""
        if self.thread is not None:
            return
        self.poll()
        self.thread = threading.Thread(
            target=self.run_forever, name="policy-watcher", daemon=True
        )
        self.thread.start()

    def run_forever(self) -> None:
        """Poll until :meth:`stop` is called (the thread body)."""
        while not self.stopping.wait(self.poll_seconds):
            try:
                self.poll()
            except Exception as exc:  # a snapshot must never kill the session
                print(
                    f"[policies] poll failed: {type(exc).__name__}: {exc}", flush=True
                )

    def stop(self) -> None:
        """Ask the polling thread to finish and wait briefly for it."""
        self.stopping.set()
        thread, self.thread = self.thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(2.0, self.poll_seconds * 2))


def rebuild_index(cell: Path, policies_dir: Path | None = None) -> dict:
    """Recompute an index's derived fields from the session's own records.

    The version directories are read, never written: content, hashes,
    triggers, times and the ``train_run`` links are exactly what the session
    recorded. Everything that is derived from other files is rebuilt --
    ``helpers`` from the stored sources, the training outcome from
    ``ledger.jsonl``, the provenance from ``budget.json`` and the raw
    harness stream -- which is what gives a cell recorded before those
    fields existed the same index as a cell recorded today.

    Provenance is best effort for a harness whose stream carries no
    timestamps: its records can only be placed at the file's modification
    time, so versions written before that get zero commands and zero tokens.
    ``budget_used`` and ``elapsed_s`` come from the ledger and are exact.

    Args:
        cell: The cell directory.
        policies_dir: The ``policies`` directory (default ``<cell>/policies``).

    Returns:
        The rebuilt index, already written.
    """
    cell = Path(cell).resolve()
    policies = Path(policies_dir) if policies_dir else cell / POLICIES_DIR
    index = read_index(policies)
    events = read_ledger(cell)
    stream = StreamProvenance(cell)
    started = ledger_start(events)
    versions = [v for v in index.get("versions") or [] if isinstance(v, dict)]
    index["versions"] = versions
    for entry in versions:
        when = entry_time(entry)
        if when is not None:
            entry["ts"] = float(when)
        directory = policies / str(
            entry.get("dir") or version_name(entry.get("version") or 0)
        )
        entry["helpers"] = scan_helpers(read_sources(directory))
        try:
            template = (Path(cell) / TEMPLATE_NAME).read_bytes()
            entry["template"] = (directory / POLICY_NAME).read_bytes() == template
        except OSError:
            entry.setdefault("template", False)
        entry.setdefault("train_run", None)
        # A cell with no ledger keeps whatever the runner measured.
        entry.setdefault("train_seeds", None)
        fields = stream.at(when)
        fields["budget_used"] = ledger_budget(events, when)
        fields["elapsed_s"] = (
            round(max(when - started, 0.0), 3)
            if (when is not None and started)
            else None
        )
        entry.update(fields)
    attribute_training(versions, events)
    write_index(policies, index)
    return index


def _eval_summary(cell: Path, version: int) -> dict:
    """Return ``eval/vNNN/summary.json`` of a version, or an empty dict."""
    try:
        loaded = json.loads(
            (version_dir(cell, version, "eval") / "summary.json").read_text()
        )
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _score(rate, episodes) -> str:
    """Render a success rate and its episode count for the table."""
    if rate is None:
        return "-"
    try:
        text = f"{float(rate):.2f}"
    except (TypeError, ValueError):
        return "-"
    return f"{text} ({int(episodes)} eps)" if episodes else text


def policy_table(cell: Path) -> str:
    """Render the cell's policy versions as a text table.

    Args:
        cell: The cell directory.

    Returns:
        The table, one line per version.
    """
    cell = Path(cell).resolve()
    index = read_index(cell / POLICIES_DIR)
    lines = [f"{'version':<8} {'time':<26} {'trigger':<11} {'train':<16} eval"]
    for entry in index["versions"]:
        version = entry.get("version")
        summary = _eval_summary(cell, version) if version is not None else {}
        lines.append(
            f"{version_name(version) if version is not None else '?':<8} "
            f"{str(entry.get('time') or '-'):<26} "
            f"{('template' if entry.get('template') else str(entry.get('trigger') or '-')):<11} "
            f"{_score(entry.get('train_success'), entry.get('train_episodes')):<16} "
            f"{_score(summary.get('success_rate'), summary.get('episodes'))}"
        )
    if len(lines) == 1:
        lines.append("(no versions recorded)")
    return "\n".join(lines)


def main(argv: list[str] | PoliciesConfig | None = None):
    """Print a cell's policy versions with their train and eval scores.

    Args:
        argv: The parsed settings, or command line arguments (None for
            ``sys.argv[1:]``).

    Returns:
        The process exit status.
    """
    args = parse_command(PoliciesConfig, argv)
    cell = args.cell.resolve()
    index_path = cell / POLICIES_DIR / INDEX_NAME
    if not index_path.exists():
        print(f"no policy index: {index_path}", file=sys.stderr)
        return 2
    if args.rebuild:
        rebuilt = rebuild_index(cell)
        print(f"rebuilt {index_path} ({len(rebuilt['versions'])} versions)")
    print(f"{cell}  task={cell_task(cell) or '?'}")
    print(policy_table(cell))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
