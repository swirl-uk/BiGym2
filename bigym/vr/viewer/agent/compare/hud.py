"""The controls of a compare view: its panel, the legend and the progress bars."""

from __future__ import annotations

import contextlib
import html
from pathlib import Path

from bigym.vr.viewer.agent.compare.model import (
    SLOT_EMPTY,
    CompareRequest,
    fill_slot,
    resolve_compare,
)
from bigym.vr.viewer.agent.runs import (
    VERSION_LAST,
    find_cell,
    policy_rows,
    sessions_of,
    visible_rows,
)
from bigym.vr.viewer.layout import (
    COMPARE_LAYOUTS,
    COMPARE_SLOTS,
    FOCUS_FREE,
    LAYOUT_AUTO,
    grid_note,
    slot_colour,
)


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
