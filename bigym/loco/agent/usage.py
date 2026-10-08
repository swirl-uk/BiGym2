"""``bigym-agent usage``: token usage and API-equivalent cost per session.

Reads every cell under the given roots and totals the usage records of its
transcript (:mod:`bigym.loco.agent.transcript`), which each harness reports in
its own way: the Codex CLI's cumulative ``turn.completed`` counter and Claude
Code's ``result`` event (which carries the cost itself).

Cost is "API-equivalent": a session run through a subscription is priced as if
it had been billed per token, from :data:`bigym.loco.agent.transcript.MODEL_PRICES`
unless the harness reported a figure of its own. Only the development phase
costs anything -- the hidden-seed evaluation runs ``policy.py`` on the host with
no model in the loop -- and the report checks that rather than asserting it, by
looking for a harness event stream under any cell's ``eval/``.

    bigym-agent usage <root> [<root> ...] [--csv usage.csv]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from bigym.loco.agent import transcript as _transcript
from bigym.loco.agent.audit import find_cells
from bigym.loco.agent.cli import UsageConfig, parse_command

# The report's columns, in order. ``harness`` is what run.json recorded,
# ``stream`` the harness whose raw event stream the transcript came from.
COLUMNS = (
    "root",
    "task",
    "harness",
    "model",
    "effort",
    "interface",
    "stream",
    "verdict",
    "wall_clock_s",
    "input_tokens",
    "cached_tokens",
    "output_tokens",
    "reasoning_tokens",
    "api_equiv_usd",
    "cost_source",
    "pricing_date",
)


def session_row(cell: Path) -> dict | None:
    """Total one cell's usage.

    Args:
        cell: The run cell.

    Returns:
        A row with the columns of :data:`COLUMNS`, or None when the cell holds
        no harness transcript.
    """
    run: dict = {}
    run_json = cell / "run.json"
    if run_json.exists():
        try:
            run = json.loads(run_json.read_text())
        except json.JSONDecodeError:
            run = {}
    harness, records = _transcript.read_records(cell)
    if harness is None:
        return None
    summary = _transcript.usage_summary(records, run.get("model"))
    tokens = summary["tokens"]
    verdict = (run.get("verdict") or {}).get("state", "")
    return {
        "root": cell.parent.name,
        "task": cell.name,
        "harness": run.get("harness", harness),
        "model": run.get("model", ""),
        "effort": run.get("effort", ""),
        "interface": run.get("interface", ""),
        "stream": harness,
        "verdict": verdict,
        "wall_clock_s": run.get("wall_clock_s", ""),
        "input_tokens": tokens["input"],
        "cached_tokens": tokens["cached"],
        "output_tokens": tokens["output"],
        "reasoning_tokens": tokens["reasoning"],
        "api_equiv_usd": summary["usd"],
        "cost_source": summary["cost_source"],
        "pricing_date": summary["pricing_date"] or "",
    }


def eval_transcripts(cells: list[Path]) -> list[str]:
    """Find harness event streams under evaluation directories.

    The hidden-seed evaluation must cost nothing: it runs the submitted policy
    with no agent and no model in the loop. A harness stream under ``eval/``
    would mean otherwise.

    Args:
        cells: The cells to check.

    Returns:
        One line per suspicious file (empty when the claim holds).
    """
    bad = []
    for cell in cells:
        for pattern in ("eval/**/*codex*", "eval/**/*claude_stream*"):
            for stray in sorted(cell.glob(pattern)):
                bad.append(f"{cell.parent.name}/{cell.name}: {stray}")
    return bad


def _summarise(rows: list[dict]) -> str:
    """One line on a set of sessions: count, cost, tokens, cache hit rate."""
    priced = sorted(r["api_equiv_usd"] for r in rows if r["api_equiv_usd"] is not None)
    total_in = sum(r["input_tokens"] + r["cached_tokens"] for r in rows)
    cached = sum(r["cached_tokens"] for r in rows)
    text = f"{len(rows)} sessions"
    if priced:
        median = (
            priced[len(priced) // 2]
            if len(priced) % 2
            else (priced[len(priced) // 2 - 1] + priced[len(priced) // 2]) / 2
        )
        text += (
            f", {sum(priced):.2f} USD total, {median:.2f} USD median "
            f"(min {priced[0]:.2f}, max {priced[-1]:.2f})"
        )
    text += f"; {total_in / 1e6:.1f} M input tokens"
    if total_in:
        text += f", {100 * cached / total_in:.1f}% served from cache"
    return text


def main(argv: list[str] | UsageConfig | None = None) -> int:
    """Print (and optionally write) the per-session usage table.

    Args:
        argv: The parsed settings, or command line arguments (None for
            ``sys.argv[1:]``).

    Returns:
        The process exit status.
    """
    args = parse_command(UsageConfig, argv)
    cells: list[Path] = []
    for root in args.roots:
        cells += find_cells(root.expanduser().resolve())
    rows = []
    for cell in cells:
        row = session_row(cell)
        if row is not None:
            rows.append(row)
    if not rows:
        print("no sessions with a harness transcript found", file=sys.stderr)
        return 1
    for row in rows:
        usd = row["api_equiv_usd"]
        print(
            f"{row['root']}/{row['task']:26s} {str(row['harness']):7s} "
            f"{str(row['model'])[:22]:22s} cache {row['cached_tokens']:12,d} "
            f"input {row['input_tokens']:10,d} output {row['output_tokens']:9,d} "
            + (f"{usd:8.2f} USD ({row['cost_source']})" if usd is not None else "")
        )
    print(_summarise(rows))
    stray = eval_transcripts(cells)
    print(
        "hidden-seed evaluation model calls: "
        + ("none (no harness stream under any eval/)" if not stray else str(stray))
    )
    print(
        f"prices: {_transcript.MODEL_PRICES} USD per million tokens, list prices on "
        f"{_transcript.PRICING_DATE}; edit MODEL_PRICES to reprice"
    )
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
