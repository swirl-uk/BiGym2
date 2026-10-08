"""The run registry: ``bigym-agent status``, ``kill`` and ``gc``.

Sessions are registered in ``<root>/registry.sqlite`` before they start. The
registry refuses a second live run of the same cell and a duplicate container
name, and remembers which GPU, image and effort a cell ran with, so
``bigym-agent status`` / ``kill`` / ``gc`` can find a session that is still
running. Without it, concurrent launches collide on container names and on the
GPU, and a crashed launcher leaves a container nobody owns.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import time
from pathlib import Path

from . import containers
from .cli import GcConfig, KillConfig, StatusConfig, parse_command


def open_registry(root: Path) -> sqlite3.Connection:
    """Open (creating it if need be) the run registry of a root.

    Args:
        root: The runs root; the database is ``<root>/registry.sqlite``.

    Returns:
        The open connection.
    """
    root.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(root / "registry.sqlite", timeout=30)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY, root TEXT NOT NULL, task TEXT NOT NULL,
        started REAL NOT NULL, pid INTEGER, pgid INTEGER, container TEXT UNIQUE,
        egl INTEGER, gpu INTEGER, effort TEXT, image TEXT, harness TEXT, model TEXT,
        cmd TEXT, ended REAL, note TEXT)"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS runs_root_task ON runs(root, task)")
    return conn


def alive(pid: int | None) -> bool:
    """Say whether a process id is still a running (non-zombie) process.

    Args:
        pid: The process id, or None.

    Returns:
        True when the process exists and has not been reaped.
    """
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return "Z" not in stat.split(")")[-1].split()[0]
    except (OSError, IndexError):
        return True


def claim(conn: sqlite3.Connection, cell: Path, task: str, info: dict) -> int | None:
    """Register a session, refusing one that would collide with a live run.

    Args:
        conn: The registry connection.
        cell: The cell directory (its parent is the root).
        task: The task name.
        info: Row fields (``container``, ``egl``, ``gpu``, ``effort``,
            ``image``, ``harness``, ``model``, ``cmd``).

    Returns:
        The new row id, or None when the claim was refused (the reason is
        printed).
    """
    root = str(cell.parent)
    live = containers.running_containers()
    for rid, pid, name in conn.execute(
        "SELECT id, pid, container FROM runs WHERE root=? AND task=? AND ended IS NULL",
        (root, task),
    ).fetchall():
        if alive(pid) or (name and name in live):
            print(
                f"refused: run {rid} already owns {cell.parent.name}/{task} "
                f"(pid {pid}, container {name})"
            )
            return None
        conn.execute(
            "UPDATE runs SET ended=?, note=COALESCE(note,'')||' [stale, closed by run]'"
            " WHERE id=?",
            (time.time(), rid),
        )
    if info.get("container") and info["container"] in live:
        print(f"refused: container {info['container']} exists")
        return None
    cursor = conn.execute(
        "INSERT INTO runs(root, task, started, pid, pgid, container, egl, gpu, effort,"
        " image, harness, model, cmd) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            root,
            task,
            time.time(),
            os.getpid(),
            os.getpgid(0),
            info.get("container"),
            info.get("egl"),
            info.get("gpu"),
            info.get("effort"),
            info.get("image"),
            info.get("harness"),
            info.get("model"),
            info.get("cmd"),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def close_row(conn: sqlite3.Connection, row_id: int | None, note: str = "") -> None:
    """Mark a registry row finished.

    Args:
        conn: The registry connection.
        row_id: The row to close; None does nothing.
        note: A note to append.
    """
    if row_id is None:
        return
    conn.execute(
        "UPDATE runs SET ended=?, note=COALESCE(note,'')||? WHERE id=?",
        (time.time(), f" {note}" if note else "", row_id),
    )
    conn.commit()


def read_budget(cell: Path) -> dict | None:
    """Read the server's ``budget.json``, when it wrote one."""
    path = cell / "budget.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def cell_state(root: Path, task: str) -> str:
    """One line on a cell's budget, policy versions and evaluation."""
    cell = root / task
    parts = []
    budget = read_budget(cell)
    if budget:
        parts.append(f"budget {budget.get('used', 0):,}/{budget.get('cap', 0):,}")
    index = cell / "policies" / "index.json"
    if index.exists():
        try:
            parts.append(f"v{len(json.loads(index.read_text()).get('versions') or [])}")
        except json.JSONDecodeError:
            pass
    run = cell / "run.json"
    if run.exists():
        try:
            verdict = (json.loads(run.read_text()).get("verdict") or {}).get("state")
            if verdict:
                parts.append(verdict)
        except json.JSONDecodeError:
            pass
    summaries = sorted((cell / "eval").glob("*/summary.json"))
    if summaries:
        try:
            data = json.loads(summaries[-1].read_text())
            parts.append(f"SCORE {round(100 * float(data['success_rate']))}")
            if data.get("rejected"):
                parts.append("REJECTED (imports the simulator)")
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    else:
        rows = sorted((cell / "eval").glob("*/episodes.csv"))
        if rows:
            done = max(0, sum(1 for _ in open(rows[-1])) - 1)
            parts.append(f"eval {done}")
    if run.exists():
        try:
            status = (json.loads(run.read_text()).get("evaluation") or {}).get("status")
        except json.JSONDecodeError:
            status = None
        if status == "failed":
            parts.append("EVAL FAILED (unscored)")
    return ", ".join(parts) or "-"


def main_status(argv: list[str] | StatusConfig | None = None) -> int:
    """List the registry's runs with their budget and evaluation progress.

    Args:
        argv: The parsed settings, or command line arguments (None for
            ``sys.argv[1:]``).

    Returns:
        The process exit status.
    """
    args = parse_command(StatusConfig, argv)
    root = args.root.expanduser().resolve()
    conn = open_registry(root)
    live = containers.running_containers()
    shown = 0
    for (
        rid,
        run_root,
        task,
        started,
        pid,
        name,
        egl,
        gpu,
        effort,
        ended,
    ) in conn.execute(
        "SELECT id, root, task, started, pid, container, egl, gpu, effort, ended"
        " FROM runs ORDER BY id"
    ):
        running = alive(pid) or bool(name and name in live)
        if not args.all and (ended or not running):
            continue
        age = (time.time() - started) / 60
        print(
            f"{rid:4d} {Path(run_root).name}/{task:26s} {str(effort):6s} "
            f"egl{egl}/gpu{gpu} {'LIVE ' if running else 'dead '} {age:6.0f} min  "
            + cell_state(Path(run_root), task)
        )
        shown += 1
    orphans = sorted(
        name
        for name in live
        if name.startswith(containers.CONTAINER_PREFIX)
        and not conn.execute("SELECT 1 FROM runs WHERE container=?", (name,)).fetchone()
    )
    if orphans:
        print("containers not in the registry:", ", ".join(orphans))
    if not shown:
        print("no runs" + ("" if args.all else " running; --all shows finished ones"))
    conn.close()
    return 0


# How a cell's server and evaluators appear in ``ps``: started by the
# launcher (``-m bigym.loco.agent``) or by the console script.
CELL_PROCESSES = tuple(
    f"{prefix} {command}"
    for prefix in ("bigym.loco.agent", "bigym-agent")
    for command in ("serve", "evaluate")
)


def kill_row(conn: sqlite3.Connection, row) -> None:
    """Stop one registered run: its process, its container, its server."""
    rid, run_root, task, pid, pgid, name = row
    if pid and alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    # Kill the group only when it is not our own: a run started in the
    # foreground shares the caller's process group.
    if pgid and pgid != os.getpgid(0):
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    if name:
        subprocess.run(
            containers.docker_argv(["docker", "rm", "-f", name]), capture_output=True
        )
    # The cell's server and evaluators, whatever their process group.
    cell = f"{run_root}/{task}"
    listing = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True)
    for line in listing.stdout.splitlines():
        if cell in line and any(marker in line for marker in CELL_PROCESSES):
            try:
                os.kill(int(line.split()[0]), signal.SIGTERM)
            except (ProcessLookupError, ValueError, PermissionError):
                pass
    close_row(conn, rid, "[killed]")
    print(f"killed run {rid} {Path(run_root).name}/{task}")


def main_kill(argv: list[str] | KillConfig | None = None) -> int:
    """Stop runs by registry id or by task name.

    Args:
        argv: The parsed settings, or command line arguments (None for
            ``sys.argv[1:]``).

    Returns:
        The process exit status.
    """
    args = parse_command(KillConfig, argv)
    root = str(args.root.expanduser().resolve())
    conn = open_registry(Path(root))
    columns = "SELECT id, root, task, pid, pgid, container FROM runs"
    for target in args.targets:
        if target.isdigit():
            rows = conn.execute(f"{columns} WHERE id=?", (int(target),)).fetchall()
        else:
            rows = conn.execute(
                f"{columns} WHERE root=? AND task=? AND ended IS NULL", (root, target)
            ).fetchall()
        if not rows:
            print(f"no live run matches {target!r}")
        for row in rows:
            kill_row(conn, row)
    conn.close()
    return 0


def main_gc(argv: list[str] | GcConfig | None = None) -> int:
    """Close dead registry rows and list (or remove) orphan containers.

    Args:
        argv: The parsed settings, or command line arguments (None for
            ``sys.argv[1:]``).

    Returns:
        The process exit status.
    """
    args = parse_command(GcConfig, argv)
    conn = open_registry(args.root.expanduser().resolve())
    live = containers.running_containers()
    closed = 0
    for rid, pid, name in conn.execute(
        "SELECT id, pid, container FROM runs WHERE ended IS NULL"
    ).fetchall():
        if not alive(pid) and not (name and name in live):
            close_row(conn, rid, "[gc]")
            closed += 1
    orphans = [
        name
        for name in live
        if name.startswith(containers.CONTAINER_PREFIX)
        and not conn.execute(
            "SELECT 1 FROM runs WHERE container=? AND ended IS NULL", (name,)
        ).fetchone()
    ]
    print(
        f"closed {closed} dead registry rows; {len(orphans)} containers not owned by a "
        "live run" + (": " + ", ".join(orphans) if orphans else "")
    )
    if orphans and args.remove_orphans:
        subprocess.run(
            containers.docker_argv(["docker", "rm", "-f", *orphans]),
            capture_output=True,
        )
        print("removed them")
    conn.close()
    return 0
