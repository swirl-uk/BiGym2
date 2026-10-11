"""The HTML of a cell's panels: run card, policy table, code, diff, transcript."""

from __future__ import annotations

import difflib
import html
from pathlib import Path
from typing import cast

from pygments import highlight
from pygments.formatters.html import HtmlFormatter
from pygments.lexers.python import PythonLexer

from bigym.loco.agent.transcript import price_of
from bigym.vr.viewer.agent.runs import NO_EVAL, NO_TRAIN, as_float, as_int

# Command blocks in the transcript panel and the policy source are quoted up
# to this many lines (the panel is a sidebar, not a pager).
TRANSCRIPT_COMMAND_LINES = 20


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


def progress_note_html(label: str) -> str:
    """The text line that sits above one progress bar."""
    return (
        '<div style="font-size:10px;opacity:.7;font-family:ui-monospace,'
        f'monospace;">{html.escape(label)}</div>'
    )


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
    version carries them.

    ``train`` and ``eval`` say what is absent rather than dashing it out
    (``no dev episodes``, ``not evaluated``): those two blanks have a
    meaning, where a missing token count is unknown and stays "—".

    Args:
        rows: Rows from :func:`~bigym.vr.viewer.agent.runs.policy_rows`.
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
