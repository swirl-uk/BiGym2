"""The agent half of the viser sidebar."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np

from bigym.loco.agent.transcript import price_of
from bigym.loco.eval.protocol import EVAL_SEED_BASE
from bigym.vr.viewer.agent import runs
from bigym.vr.viewer.agent.compare.hud import ComparePanel, progress_bar_html
from bigym.vr.viewer.agent.html import (
    code_html,
    diff_html,
    policy_table_html,
    progress_note_html,
    summary_html,
    table_columns,
    transcript_html,
    transcript_title,
    vscode_link_html,
)
from bigym.vr.viewer.agent.jobs import AgentJob
from bigym.vr.viewer.agent.runs import (
    EVAL_SEED_LO,
    ROLLOUT_KINDS,
    TRANSCRIPT_JSONL,
    Cell,
    Rollout,
    cell_of,
    default_rollout,
    describe_opening,
    find_cell,
    policy_file,
    policy_rows,
    read_run,
    read_transcript,
    rollout_item_label,
    rollouts_of,
    sessions_of,
    tasks_of,
    version_records,
    visible_rows,
)
from bigym.vr.viewer.layout import COMPARE_GAP, LAYOUT_AUTO

# The first five hidden evaluation seeds are the conventional "show me what it
# does" set for the Replay button.
DEFAULT_REPLAY_SEEDS = f"{EVAL_SEED_BASE}-{EVAL_SEED_BASE + 4}"


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
                "show template", initial_value=runs.SHOW_TEMPLATE
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
            runs.SHOW_TEMPLATE = bool(self.template_cb.value)
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
