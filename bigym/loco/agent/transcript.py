"""One transcript format for every agent harness, plus its markdown rendering.

Each harness writes its own event stream into ``<cell>/raw/``:

- Codex CLI: ``codex_events.jsonl`` (``codex exec --json``),
- Claude Code: ``claude_stream.jsonl`` (``--output-format stream-json``),
- ``--harness custom``: ``custom_stdout.txt``, whatever the command printed.

The adapters below turn every one of them into one stream of records, so a reader (or
an auditor, see :mod:`bigym.loco.agent.audit`) never has to know which harness
produced a session. ``bigym-agent report <cell>`` regenerates
``<cell>/transcript.jsonl`` and ``<cell>/transcript.md`` from the raw stream.

Record schema, one JSON object per line of ``transcript.jsonl``; fields that do
not apply are omitted::

    {"t": <unix float>,                       # when the harness timestamps events
     "kind": "message" | "reasoning" | "command" | "output" | "edit" | "tool"
             | "usage" | "meta",
     "role": "agent" | "harness" | "user",
     "text": "...",                           # message / reasoning / output text
     "command": "...",                        # kind == "command"
     "exit_code": <int|null>,                 # command result, when reported
     "path": "...",                           # kind == "edit" / file tool
     "tokens": {"input":, "cached":, "output":, "reasoning":},   # kind == "usage"
     "usd": <float>}                          # kind == "usage", harness-reported

Usage records are additive: summing the ``tokens`` of every usage record of a
session gives the session total. Codex reports a cumulative counter per turn,
so the adapter emits the increment; Claude Code repeats the same usage block on
every content block of a message, so the adapter takes the authoritative
``result`` event (and falls back to one record per distinct message id when a
session was cut short before that event).
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

from .cli import ReportConfig, parse_command

# USD per million tokens, per model: uncached input / cache read / output.
# These are list prices on the date below, quoted so a session can be priced
# without a network call; they are NOT read from any vendor API and go stale.
# Edit them (or add your own model) to price a run at today's rates.
PRICING_DATE = "2026-10-08"
MODEL_PRICES: dict[str, dict[str, float]] = {
    # Reasoning models of the OpenAI Codex CLI row.
    "gpt-6-astra": {"input": 10.0, "cached": 1.0, "output": 50.0},
    # Standard tier, short-context column (developers.openai.com/api/docs/pricing);
    # cache writes and the long-context tier are not applied. gpt-5.6-sol's figure
    # is OpenAI's promotional price, announced through at least 2026-11-21 (list
    # price before 2026-08-21: $5 input / $30 output); reprice after it ends.
    # Runs on a ChatGPT plan draw on its included limits first, so
    # these are list-price equivalents, not what the subscription was charged.
    "gpt-6-sol": {"input": 2.0, "cached": 0.2, "output": 10.0},
    "gpt-6.1-sol": {"input": 2.0, "cached": 0.1, "output": 10.0},
    "gpt-5.6-sol": {"input": 4.0, "cached": 0.4, "output": 20.0},
}

# The harnesses whose event streams the adapters below understand.
HARNESSES = ("codex", "claude", "custom")

TOKEN_FIELDS = ("input", "cached", "output", "reasoning")

# Output blocks are quoted in transcript.md up to this many lines / characters.
MD_OUTPUT_LINES = 40
MD_OUTPUT_CHARS = 4000

# What ``bigym-agent run --harness custom`` captures the command's stdout into.
CUSTOM_STDOUT = "custom_stdout.txt"


def price_of(model: str | None) -> dict[str, float] | None:
    """Look up the price table entry for a model name.

    Args:
        model: The model name a run recorded, or None.

    Returns:
        The USD-per-million-tokens entry, or None when the model is unknown.
    """
    if not model:
        return None
    if model in MODEL_PRICES:
        return MODEL_PRICES[model]
    for key, price in MODEL_PRICES.items():
        if model.startswith(key) or key in model:
            return price
    return None


def _rec(kind: str, role: str, **fields: Any) -> dict:
    """Build a record, dropping the fields that do not apply."""
    out: dict[str, Any] = {"kind": kind, "role": role}
    for key, value in fields.items():
        if value is not None and value != "":
            out[key] = value
    return out


def _tokens(
    input_: int = 0, cached: int = 0, output: int = 0, reasoning: int = 0
) -> dict[str, int]:
    """Build a token dictionary with every field present."""
    return {
        "input": int(input_),
        "cached": int(cached),
        "output": int(output),
        "reasoning": int(reasoning),
    }


def iso_to_unix(stamp: str | None) -> float | None:
    """Parse an ISO-8601 stamp (``Z`` suffix included) into a unix time."""
    if not stamp:
        return None
    try:
        return _dt.datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _json_lines(path: Path) -> Iterator[dict]:
    """Yield the JSON objects of a JSON-lines file, skipping broken lines."""
    with open(path, errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _text_of(content: Any) -> str:
    """Flatten a content field (string, block, or list of blocks) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        return _text_of(content.get("content"))
    if isinstance(content, list):
        return "\n".join(part for part in (_text_of(b) for b in content) if part)
    return str(content)


# --------------------------------------------------------------- Codex CLI


def from_codex_events(path: str | Path) -> Iterator[dict]:
    """Read a Codex CLI ``--json`` event stream into transcript records.

    Args:
        path: Path to ``codex_events.jsonl``.

    Yields:
        Transcript records in event order.
    """
    seen = _tokens()
    for event in _json_lines(Path(path)):
        kind = event.get("type")
        if kind == "thread.started":
            yield _rec("meta", "harness", text=f"codex thread {event.get('thread_id')}")
        elif kind == "turn.completed":
            usage = event.get("usage") or {}
            cached = int(usage.get("cached_input_tokens", 0) or 0)
            total = _tokens(
                max(int(usage.get("input_tokens", 0) or 0) - cached, 0),
                cached,
                int(usage.get("output_tokens", 0) or 0),
                int(usage.get("reasoning_output_tokens", 0) or 0),
            )
            # The counter is cumulative over the thread: report the increment.
            delta = {k: max(total[k] - seen[k], 0) for k in TOKEN_FIELDS}
            seen = total
            yield _rec("usage", "harness", tokens=delta)
        elif kind == "error":
            yield _rec("meta", "harness", text=str(event.get("message", "error")))
        elif kind == "item.completed":
            yield from _codex_item(event.get("item") or {})


def _codex_item(item: dict) -> Iterator[dict]:
    """Turn one completed Codex item into records."""
    kind = item.get("type")
    if kind == "agent_message":
        yield _rec("message", "agent", text=str(item.get("text", "")))
    elif kind == "reasoning":
        yield _rec("reasoning", "agent", text=str(item.get("text", "")))
    elif kind == "command_execution":
        exit_code = item.get("exit_code")
        yield _rec(
            "command",
            "agent",
            command=str(item.get("command", "")),
            exit_code=exit_code,
        )
        output = str(item.get("aggregated_output", "") or "")
        if output:
            yield _rec("output", "harness", text=output, exit_code=exit_code)
    elif kind == "file_change":
        for change in item.get("changes") or []:
            yield _rec(
                "edit",
                "agent",
                path=str(change.get("path", "")),
                text=str(change.get("kind", "")),
            )
    elif kind == "error":
        yield _rec("meta", "harness", text=str(item.get("message", "error")))
    elif kind:
        yield _rec("tool", "agent", text=str(kind))


# -------------------------------------------------------------- Claude Code

# Tools whose call is a shell command, and tools that write a file.
CLAUDE_SHELL_TOOLS = {"Bash", "BashOutput"}
CLAUDE_EDIT_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}


def from_claude_stream(path: str | Path) -> Iterator[dict]:
    """Read a Claude Code ``stream-json`` transcript into transcript records.

    Args:
        path: Path to ``claude_stream.jsonl``.

    Yields:
        Transcript records in event order.
    """
    per_message: dict[str, dict[str, int]] = {}
    final_usage = False
    for event in _json_lines(Path(path)):
        kind = event.get("type")
        when = iso_to_unix(event.get("timestamp"))
        if kind == "system":
            yield from _claude_system(event, when)
        elif kind == "assistant":
            message = event.get("message") or {}
            for block in message.get("content") or []:
                if isinstance(block, dict):
                    yield from _claude_block(block, when)
            usage = message.get("usage")
            if isinstance(usage, dict) and message.get("id"):
                per_message[str(message["id"])] = _claude_tokens(usage)
        elif kind == "user":
            message = event.get("message") or {}
            for block in message.get("content") or []:
                yield from _claude_user_block(block, when)
        elif kind == "result":
            final_usage = True
            yield _rec(
                "meta",
                "harness",
                t=when,
                text="result {}: {} turns, {:.0f} s".format(
                    event.get("subtype", "?"),
                    event.get("num_turns", "?"),
                    float(event.get("duration_ms", 0) or 0) / 1000.0,
                ),
            )
            yield _rec(
                "usage",
                "harness",
                t=when,
                tokens=_claude_tokens(event.get("usage") or {}),
                usd=event.get("total_cost_usd"),
            )
    if not final_usage and per_message:
        # Session cut short before the result event: sum the per-message blocks.
        total = _tokens()
        for tokens in per_message.values():
            for field in TOKEN_FIELDS:
                total[field] += tokens[field]
        yield _rec("usage", "harness", tokens=total)


def _claude_tokens(usage: dict) -> dict[str, int]:
    """Map a Claude usage block onto the transcript's token fields."""
    details = usage.get("output_tokens_details") or {}
    return _tokens(
        int(usage.get("input_tokens", 0) or 0)
        + int(usage.get("cache_creation_input_tokens", 0) or 0),
        int(usage.get("cache_read_input_tokens", 0) or 0),
        int(usage.get("output_tokens", 0) or 0),
        int(details.get("thinking_tokens", 0) or 0),
    )


def _claude_system(event: dict, when: float | None) -> Iterator[dict]:
    """Turn a Claude ``system`` event into a meta record, when it carries news."""
    subtype = event.get("subtype")
    if subtype == "init":
        yield _rec(
            "meta",
            "harness",
            t=when,
            text=(
                f"claude session {event.get('session_id')} "
                f"model {event.get('model')} cwd {event.get('cwd')}"
            ),
        )
    elif subtype == "permission_denied":
        yield _rec(
            "meta",
            "harness",
            t=when,
            text=(
                f"permission denied: {event.get('tool_name')}: "
                f"{event.get('message', '')}"
            ),
        )


def _claude_block(block: dict, when: float | None) -> Iterator[dict]:
    """Turn one assistant content block into records."""
    kind = block.get("type")
    if kind == "thinking":
        text = str(block.get("thinking", "") or "")
        if text:
            yield _rec("reasoning", "agent", t=when, text=text)
    elif kind == "text":
        yield _rec("message", "agent", t=when, text=str(block.get("text", "")))
    elif kind == "tool_use":
        name = str(block.get("name", ""))
        args = block.get("input")
        if not isinstance(args, dict):
            args = {}
        if name in CLAUDE_SHELL_TOOLS and args.get("command"):
            yield _rec(
                "command",
                "agent",
                t=when,
                command=str(args["command"]),
                text=str(args.get("description", "") or ""),
            )
        elif name in CLAUDE_EDIT_TOOLS:
            yield _rec(
                "edit", "agent", t=when, path=str(args.get("file_path", "")), text=name
            )
        else:
            yield _rec(
                "tool",
                "agent",
                t=when,
                text=f"{name} {json.dumps(args, sort_keys=True)[:400]}".strip(),
                path=str(args.get("file_path", "") or "") or None,
            )


def _claude_user_block(block: Any, when: float | None) -> Iterator[dict]:
    """Turn one user content block (tool result or text) into records."""
    if not isinstance(block, dict):
        return
    if block.get("type") == "tool_result":
        yield _rec(
            "output",
            "harness",
            t=when,
            text=_text_of(block.get("content")),
            exit_code=1 if block.get("is_error") else None,
        )
    elif block.get("type") == "text":
        yield _rec("message", "user", t=when, text=str(block.get("text", "")))


# ------------------------------------------------------------ any command


def from_custom(path: str | Path) -> Iterator[dict]:
    """Read a custom agent's captured stdout into transcript records.

    ``--harness custom`` runs an arbitrary program, which reports nothing about
    itself: there are no messages, tool calls or token counts to recover, only
    what it printed. The whole stream becomes one ``output`` record, so
    ``transcript.md`` shows the session and ``usage_summary`` totals to zero.

    Args:
        path: ``<cell>/raw/custom_stdout.txt``, or the cell directory holding
            it.

    Yields:
        One ``output`` record, or nothing when the command printed nothing.
    """
    path = Path(path)
    if path.is_dir():
        found = path / "raw" / CUSTOM_STDOUT
        path = found if found.exists() else path / CUSTOM_STDOUT
    if not path.exists():
        return
    text = path.read_text(errors="replace")
    if not text.strip():
        return
    yield _rec("output", "harness", t=path.stat().st_mtime, text=text)


ADAPTERS = {
    "codex": from_codex_events,
    "claude": from_claude_stream,
    "custom": from_custom,
}


# ------------------------------------------------------------------- a cell


def find_raw(cell: str | Path) -> tuple[str, Path] | None:
    """Find a cell's raw harness event stream.

    Args:
        cell: The run cell (``<root>/<task>/``).

    Returns:
        ``(harness, path)`` for the stream that exists, or None.
    """
    cell = Path(cell)
    for base in (cell / "raw", cell):
        for harness, name in (
            ("codex", "codex_events.jsonl"),
            ("claude", "claude_stream.jsonl"),
            ("custom", CUSTOM_STDOUT),
        ):
            if (base / name).exists():
                return harness, base / name
    return None


def read_records(cell: str | Path) -> tuple[str | None, list[dict]]:
    """Read a cell's raw stream into transcript records.

    Args:
        cell: The run cell.

    Returns:
        ``(harness, records)``; ``(None, [])`` when the cell has no raw stream.
    """
    found = find_raw(cell)
    if found is None:
        return None, []
    harness, path = found
    return harness, list(ADAPTERS[harness](path))


def usage_summary(records: Iterable[dict], model: str | None = None) -> dict:
    """Total a session's token usage and price it.

    Args:
        records: Transcript records.
        model: The model the session ran, for the price table.

    Returns:
        ``{"tokens": {...}, "usd": float|None, "cost_source": str,
        "pricing_date": str}``. ``cost_source`` is ``"harness"`` when the
        harness reported the cost itself, ``"price_table"`` when it was
        computed from :data:`MODEL_PRICES`, and ``"unknown"`` when neither
        applies.
    """
    total = _tokens()
    reported = 0.0
    any_reported = False
    for record in records:
        if record.get("kind") != "usage":
            continue
        tokens = record.get("tokens") or {}
        for field in TOKEN_FIELDS:
            total[field] += int(tokens.get(field, 0) or 0)
        if isinstance(record.get("usd"), (int, float)):
            reported += float(record["usd"])
            any_reported = True
    if any_reported:
        return {
            "tokens": total,
            "usd": round(reported, 4),
            "cost_source": "harness",
            "pricing_date": None,
        }
    price = price_of(model)
    if price is None:
        return {
            "tokens": total,
            "usd": None,
            "cost_source": "unknown",
            "pricing_date": None,
        }
    usd = (
        total["input"] * price["input"]
        + total["cached"] * price["cached"]
        + total["output"] * price["output"]
    ) / 1e6
    return {
        "tokens": total,
        "usd": round(usd, 4),
        "cost_source": "price_table",
        "pricing_date": PRICING_DATE,
    }


def _clip(text: str) -> str:
    """Clip an output block to the markdown quoting limits."""
    lines = text.splitlines()
    clipped = lines[:MD_OUTPUT_LINES]
    body = "\n".join(clipped)[:MD_OUTPUT_CHARS]
    dropped = len(lines) - len(clipped)
    if dropped > 0:
        body += f"\n... [{dropped} more lines]"
    elif len(body) < len("\n".join(clipped)):
        body += "\n... [truncated]"
    return body


def render_markdown(records: list[dict], title: str) -> str:
    """Render transcript records as readable markdown.

    Messages and reasoning become prose, commands become fenced blocks with
    their (truncated) output, edits and tool calls become one-liners.

    Args:
        records: Transcript records.
        title: Heading for the document (usually the cell's name).

    Returns:
        The markdown document.
    """
    counts: dict[str, int] = {}
    for record in records:
        counts[record["kind"]] = counts.get(record["kind"], 0) + 1
    head = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    out = [f"# {title}", "", f"{len(records)} records ({head})", ""]
    index = 0
    while index < len(records):
        record = records[index]
        kind = record["kind"]
        text = str(record.get("text", ""))
        if kind == "message":
            who = {"agent": "Agent", "user": "User", "harness": "Harness"}[
                record["role"]
            ]
            out += [f"**{who}.** {text.strip()}", ""]
        elif kind == "reasoning":
            out += [f"> _thinking_ {text.strip()}", ""]
        elif kind == "command":
            out += ["```console", f"$ {record.get('command', '')}"]
            following = records[index + 1] if index + 1 < len(records) else None
            if following is not None and following["kind"] == "output":
                out += [_clip(str(following.get("text", "")))]
                index += 1
            out += ["```"]
            code = record.get("exit_code")
            if code not in (None, 0):
                out += [f"exit {code}", ""]
            else:
                out += [""]
        elif kind == "output":
            out += ["```console", _clip(text), "```", ""]
        elif kind == "edit":
            out += [f"- edit `{record.get('path', '')}` ({text or 'write'})", ""]
        elif kind == "tool":
            out += [f"- tool `{text}`", ""]
        elif kind == "usage":
            tokens = record.get("tokens") or {}
            out += [
                "- usage: "
                + ", ".join(f"{k} {tokens.get(k, 0):,}" for k in TOKEN_FIELDS),
                "",
            ]
        else:
            out += [f"_{text}_", ""]
        index += 1
    return "\n".join(out).rstrip() + "\n"


def write_transcript(cell: str | Path, model: str | None = None) -> dict:
    """Write ``transcript.jsonl`` and ``transcript.md`` for a run cell.

    Args:
        cell: The run cell (``<root>/<task>/``).
        model: The model the session ran, for pricing the totals.

    Returns:
        ``{"harness", "records", "tokens", "usd", "cost_source", "jsonl",
        "md"}``; ``harness`` is None and ``records`` 0 when the cell holds no
        raw event stream (nothing is written then).
    """
    cell = Path(cell)
    harness, records = read_records(cell)
    if harness is None:
        return {"harness": None, "records": 0, "tokens": _tokens(), "usd": None}
    cell.mkdir(parents=True, exist_ok=True)
    jsonl = cell / "transcript.jsonl"
    with open(jsonl, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    markdown = cell / "transcript.md"
    markdown.write_text(render_markdown(records, f"{cell.name} ({harness})"))
    summary = usage_summary(records, model)
    return {
        "harness": harness,
        "records": len(records),
        "tokens": summary["tokens"],
        "usd": summary["usd"],
        "cost_source": summary["cost_source"],
        "jsonl": str(jsonl),
        "md": str(markdown),
    }


def cell_model(cell: str | Path) -> str | None:
    """Read the model a cell's ``run.json`` recorded, if it has one.

    Args:
        cell: The run cell.

    Returns:
        The model name, or None.
    """
    path = Path(cell) / "run.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("model")
    except (json.JSONDecodeError, OSError):
        return None


def main(argv: list[str] | ReportConfig | None = None) -> int:
    """Regenerate ``transcript.jsonl`` and ``transcript.md`` of a run cell.

    Args:
        argv: The parsed settings, or command line arguments (None for
            ``sys.argv[1:]``).

    Returns:
        The process exit status.
    """
    args = parse_command(ReportConfig, argv)
    cell = args.cell.expanduser().resolve()
    summary = write_transcript(cell, args.model or cell_model(cell))
    if summary["harness"] is None:
        print(f"no harness event stream under {cell}/raw", file=sys.stderr)
        return 1
    tokens = summary["tokens"]
    usd = summary["usd"]
    print(
        f"{cell.name}: {summary['harness']}, {summary['records']} records, "
        + ", ".join(f"{k} {tokens[k]:,}" for k in TOKEN_FIELDS)
        + (f", {usd:.2f} USD ({summary['cost_source']})" if usd is not None else "")
    )
    print(summary["jsonl"])
    print(summary["md"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
