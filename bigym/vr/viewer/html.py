"""The HTML of the episode panels: reward sparkline, episode card and list."""

from __future__ import annotations

import html

import numpy as np

from bigym.vr.viewer.stores import Store

SPARKLINE_SIZE = (252, 26)  # width, height in px of the reward strip


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


def note_html(text: str, error: bool = False) -> str:
    """A small status line for a sidebar panel."""
    colour = "color:#dc2626;" if error else "opacity:.6;"
    return (
        f'<div style="font-size:10px;{colour}white-space:pre-wrap;">'
        f"{html.escape(text)}</div>"
    )


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
