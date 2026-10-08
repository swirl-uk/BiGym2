"""``bigym-agent audit``: check what a session's tool calls touched.

The benchmark's claim is that the agent saw only its sandbox: the task brief,
the harness, its own files and the environment socket. This command tests that
claim against the record, by reading every command, tool call and file edit out
of a cell's transcript (:mod:`bigym.loco.agent.transcript`, so all three
harnesses are covered) and flagging the ones that name something outside the
sandbox: a path elsewhere on the host, the cell's own bookkeeping (the ledger,
the policy snapshots, another cell), or a command that fetches code or reaches
the network.

The test is relative to the cell: the sandbox path (and, in container mode, the
path it is mounted at) is rewritten to ``<SB>`` first, so what remains is by
construction outside it. Flagged calls are printed, not interpreted -- a
``find /etc`` that returned nothing is as visible here as a download would be --
and the exit status is 1 when anything was flagged, so a batch of sessions can
be audited in CI.

    bigym-agent audit <root> [<root> ...] [--client-root /work]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from bigym.loco.agent import transcript as _transcript
from bigym.loco.agent.cli import AuditConfig, parse_command

# Paths and commands that a sandboxed session has no business naming. The
# patterns describe host locations and egress tools in general; nothing here
# names a particular machine, user or run directory.
OUTSIDE = re.compile(
    # A path that is not in the sandbox ...
    r"(/home/[^/\s\"']+/|/root/|/etc/|/proc/|/sys/|/var/|\.\./|~/|\$HOME"
    r"|site-packages|\.venv/"
    # ... or a command that fetches code or reaches the network. The guards
    # keep an assignment or a longer word (`nc=...`, `my_git_notes`) out of it.
    r"|(?<![\w.-])(pip3?|curl|wget|git|ssh|scp|rsync|nc|ncat|socat|apt|apt-get"
    r"|npm|npx|docker|nvidia-smi|sudo)(?=\s|$))"
)

# What a call looks like in the report.
CALL_KINDS = ("command", "tool", "edit")


def call_texts(records: list[dict]) -> list[str]:
    """Extract the text of every tool call from transcript records.

    Args:
        records: Transcript records of one session.

    Returns:
        One string per call: the shell command, the tool call, or the path of
        an edit.
    """
    calls = []
    for record in records:
        kind = record.get("kind")
        if kind not in CALL_KINDS:
            continue
        text = record.get("command") or ""
        if kind == "edit":
            text = f"edit {record.get('path', '')}"
        elif kind == "tool":
            # A file tool carries its path: a read outside the sandbox counts.
            text = f"{record.get('text', '')} {record.get('path', '')}".strip()
        calls.append(str(text))
    return calls


def is_cell(path: Path) -> bool:
    """Say whether a directory looks like a run cell.

    Args:
        path: The directory to test.

    Returns:
        True when it holds a run record, a raw event stream or a transcript.
    """
    return any(
        (path / name).exists()
        for name in ("run.json", "raw", "transcript.jsonl", "sandbox")
    )


def find_cells(root: Path) -> list[Path]:
    """Find the run cells under a path.

    Args:
        root: A cell, a runs root (``<root>/<task>/``) or a directory of roots.

    Returns:
        The cells found, sorted.
    """
    root = Path(root)
    if is_cell(root):
        return [root]
    cells = [p for p in sorted(root.glob("*")) if p.is_dir() and is_cell(p)]
    if cells:
        return cells
    return [p for p in sorted(root.glob("*/*")) if p.is_dir() and is_cell(p)]


def _records(cell: Path) -> list[dict]:
    """Read a cell's transcript, rebuilding it from the raw stream if need be."""
    path = cell / "transcript.jsonl"
    if path.exists():
        records = []
        for line in path.read_text(errors="replace").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records
    return _transcript.read_records(cell)[1]


def audit_cell(cell: Path, client_root: str | None = "/work") -> tuple[int, list[str]]:
    """Audit one cell's tool calls.

    Args:
        cell: The run cell.
        client_root: The path the sandbox is mounted at inside the agent
            container, rewritten to ``<SB>`` like the sandbox itself.

    Returns:
        ``(number of calls, flagged call texts)``.
    """
    sandbox = str((cell / "sandbox").resolve())
    calls = call_texts(_records(cell))
    flagged = []
    for call in calls:
        text = call.replace(sandbox + "/", "<SB>/").replace(sandbox, "<SB>")
        if client_root:
            root = client_root.rstrip("/")
            text = text.replace(root + "/", "<SB>/")
            text = re.sub(rf"(?<![\w/]){re.escape(root)}(?![\w/])", "<SB>", text)
        # Anything still naming the cell is outside the sandbox: the ledger,
        # the policy snapshots, the evaluation, a neighbouring cell.
        if str(cell.resolve()) in text or OUTSIDE.search(text):
            flagged.append(text)
    return len(calls), flagged


def main(argv: list[str] | AuditConfig | None = None) -> int:
    """Audit every session under the given roots.

    Args:
        argv: The parsed settings, or command line arguments (None for
            ``sys.argv[1:]``).

    Returns:
        1 when any call was flagged, 0 otherwise, 2 when no session was found.
    """
    args = parse_command(AuditConfig, argv)
    cells = []
    for root in args.roots:
        cells += find_cells(root.expanduser().resolve())
    if not cells:
        print("no sessions found under the given roots", file=sys.stderr)
        return 2
    total, flagged_total = 0, 0
    for cell in cells:
        calls, flagged = audit_cell(cell, args.client_root)
        total += calls
        flagged_total += len(flagged)
        label = f"{cell.parent.name}/{cell.name}"
        print(f"{label[-60:]:60s} calls={calls:4d} flagged={len(flagged)}")
        for call in flagged[: args.show]:
            print("      ", call[:220].replace("\n", " | "))
    print(f"total calls {total}, flagged {flagged_total}")
    return 1 if flagged_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
