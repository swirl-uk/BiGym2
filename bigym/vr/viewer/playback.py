"""The two frontends of ``bigym-view``: the browser viewer and the MuJoCo window.

Threading: env.reset() renders camera observations, and the EGL context
belongs to the main thread, so episode loads, camera renders and subprocess
launches only happen on the main loop; UI and keyboard callbacks just set a
request flag (downloads and directory scans run on a worker thread and then
do the same).
"""

from __future__ import annotations

import html
import time
from pathlib import Path
from typing import Any

import mujoco
import viser

from bigym.vr.viewer.agent.compare.model import CompareRequest
from bigym.vr.viewer.agent.panels import CellPanels
from bigym.vr.viewer.agent.runs import EVAL_SEED_LO
from bigym.vr.viewer.episodes import episode_files, episode_infos
from bigym.vr.viewer.html import (
    episode_card_html,
    episode_flags,
    episode_list_html,
    sparkline_svg,
)
from bigym.vr.viewer.mjviser_compat import hide_reach_targets, open_viser_scene
from bigym.vr.viewer.overlays import TargetSpheres, VelocityArrows
from bigym.vr.viewer.sources import (
    BatchSource,
    DatasetSource,
    newest_batch,
)
from bigym.vr.viewer.stores import Store
from bigym.vr.viewer.view_config import ViewConfig
from bigym.vr.viewer.widgets import (
    CAMERA_PANEL_HZ,
    LIGHTING_PREFS,
    BatchDropdown,
    CameraPanels,
    EpisodePanel,
    FigureCapture,
    LightingPanel,
    Picker,
    PlaybackPanel,
    directory_panel,
)

END_HOLD_SECONDS = 1.5  # freeze on the last frame before looping


# Nominal collection rate. Playback prefers metadata["control_step_seconds"],
# falling back to model.opt.timestep x substeps when a batch does not stamp it.
DEMO_HZ = 50.0


class Playhead:
    """Frame advance with an end-of-episode hold."""

    def __init__(self):
        """Start at frame 0 with no end-of-episode hold pending."""
        self.frame = 0
        self.hold_until = 0.0
        self.last_advance = time.time()

    def reset(self):
        """Rewind to frame 0 and clear the end-of-episode hold."""
        self.frame = 0
        self.hold_until = 0.0

    def tick(self, n: int, playing: bool, fps: float) -> bool:
        """Advance one frame if due; returns True when the frame changed."""
        now = time.time()
        if not playing or now < self.hold_until or now - self.last_advance < 1.0 / fps:
            return False
        nxt = self.frame + 1
        if nxt >= n:
            if self.hold_until <= 0.0 or now < self.hold_until:
                self.hold_until = now + END_HOLD_SECONDS
                nxt = n - 1
            else:
                nxt = 0
                self.hold_until = 0.0
        changed = nxt != self.frame
        self.frame = nxt
        self.last_advance = now
        return changed


def run_mujoco(
    store: Store,
    inner,
    model,
    data,
    realtime_fps: float = DEMO_HZ,
    batch_labels: list[str] | None = None,
    batch_idx: int = 0,
    item: str = "batch",
) -> int | None:
    """Play the store's episodes in a native MuJoCo window until it closes.

    Returns the index of another batch (or dataset task, ``item``) to open
    (``[`` / ``]``), or None when the window was closed.
    """
    import mujoco.viewer

    ctl: dict = {"playing": True, "dirty": True, "want": None, "batch": None}
    head = Playhead()
    labels = list(batch_labels or [])

    def on_key(keycode: int) -> None:
        # Runs on the viewer thread: only request; loads happen on the main loop.
        if keycode == 32:  # SPACE
            ctl["playing"] = not ctl["playing"]
        elif keycode == 262:  # RIGHT
            head.frame = min(head.frame + 1, store.qpos.shape[0] - 1)
            ctl["dirty"] = True
        elif keycode == 263:  # LEFT
            head.frame = max(head.frame - 1, 0)
            ctl["dirty"] = True
        elif keycode == 265:  # UP
            ctl["want"] = store.ep + 1
        elif keycode == 264:  # DOWN
            ctl["want"] = store.ep - 1
        elif keycode in (91, 93) and len(labels) > 1:  # [ and ]
            step = 1 if keycode == 93 else -1
            ctl["batch"] = (batch_idx + step) % len(labels)

    keys = "[viewer] SPACE play/pause · LEFT/RIGHT frame · UP/DOWN episode"
    print(keys + (f" · [ / ] {item}" if len(labels) > 1 else ""))
    if labels:
        print(f"[viewer] {item} [{batch_idx}] {labels[batch_idx]}")
    print(f"[viewer] episode {store.ep}: {store.files[store.ep].name}")
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running():
            nxt, ctl["batch"] = ctl["batch"], None
            if nxt is not None and nxt != batch_idx:
                print(f"[viewer] switching to {item} [{nxt}] {labels[nxt]}")
                return nxt
            want, ctl["want"] = ctl["want"], None
            if want is not None and store.load(want):
                print(f"[viewer] episode {store.ep}: {store.files[store.ep].name}")
                head.reset()
                ctl["dirty"] = True
            if head.tick(store.qpos.shape[0], ctl["playing"], realtime_fps):
                ctl["dirty"] = True
            if ctl["dirty"]:
                data.qpos[: store.qpos.shape[1]] = store.qpos[head.frame]
                mujoco.mj_forward(model, data)
                # kinematic playback skips task logic; run the success
                # check so task visuals (target highlight) update live
                inner._success()
                store.after_forward(head.frame)
                viewer.sync()
                ctl["dirty"] = False
            time.sleep(0.004)
    return None


class ViserViewer:
    """The browser viewer for one batch or dataset task.

    The scene is built in ``__init__`` and served by :meth:`run`. GUI
    callbacks run on viser's thread pool and only record a request; loads,
    renders and switches happen on the main loop (see the module doc).

    With ``cell`` (the agent cell the batch belongs to) the flat batch
    dropdown gives way to the agent panels of
    :class:`~bigym.vr.viewer.agent.panels.CellPanels`, whose Task and Session
    dropdowns offer ``cells``. With --follow (batches only) the main loop
    rescans every --follow-seconds: new batches, policy versions, a grown
    run.json or transcript, new episodes in the open batch. The batch on
    screen only changes when "auto-jump to newest" is ticked.
    """

    def __init__(
        self,
        store: Store,
        inner,
        picker: Picker,
        args: ViewConfig,
        directory: str,
        *,
        realtime_fps: float = DEMO_HZ,
        task_name: str = "",
        camera_keys: tuple[str, ...] = (),
        camera_shape: tuple[int, int] = (84, 84),
        cell: Path | None = None,
        cells: list | None = None,
    ) -> None:
        """Open the server and build the scene and every panel.

        Args:
            store: The episodes to play, one already loaded.
            inner: The native env the store poses.
            picker: What can be switched to.
            args: The command line.
            directory: What the "Demo directory" field starts with (empty
                for the published dataset).
            realtime_fps: The rate the episodes were recorded at.
            task_name: Prefixes capture file names.
            camera_keys: The onboard cameras the camera panels show.
            camera_shape: Their recorded ``(height, width)``.
            cell: The agent cell the batch belongs to, if any.
            cells: Every cell found, for the Task and Session dropdowns.
        """
        self.store = store
        self.inner = inner
        self.model, self.data = inner.model, inner.data
        self.picker = picker
        self.port = args.port
        self.realtime_fps = realtime_fps
        self.exit_after = args.exit_after_seconds
        self.follow_seconds = (
            args.follow_seconds if args.follow and picker.paths is not None else 0.0
        )
        self.open_batch = store.files[0].parent.resolve()
        self.head = Playhead()
        self.labels = list(store.episode_labels())
        self.dirty = True
        self.wanted_episode: int | None = None
        self.wanted_batch: int | None = None
        self.wanted_compare: CompareRequest | None = None
        self.wanted_source: DatasetSource | BatchSource | None = None
        self.card_at = 0.0
        self.card_reward: float | None = None
        self.cameras_at = 0.0
        self.cameras_logged = False
        self.follow_at = time.time()
        self.follow_scans = 0

        targets = hide_reach_targets(inner)
        self.server, self.scene = open_viser_scene(
            self.model, inner, port=self.port, label="demo-viewer"
        )
        self.arrows = VelocityArrows(self.server, self.model)
        self.targets = TargetSpheres(self.server, targets)
        directory_panel(self.server, directory, self._request_source)
        self.cell_panels = self._add_cell_panels(args, cell, cells or [])
        self.batch_dropdown = self._add_batch_picker(cell)
        self.episode = EpisodePanel(self.server, self.labels, len(store.files))
        self.playback = PlaybackPanel(self.server, realtime_fps)
        with self.server.gui.add_folder("View"):
            self.show_arrows = self.server.gui.add_checkbox(
                "velocity arrows", initial_value=True
            )
            self.show_arrows.on_update(self._mark_dirty)
        # MuJoCo's own pixels next to the viser scene: the onboard cameras the
        # env renders for observations, plus a third-person view.
        self.cameras = CameraPanels(
            self.server, inner, self.model, self.data, camera_keys, camera_shape
        )
        self.cameras.enabled.on_update(self._mark_dirty)
        print(f"[viewer] camera panels: {self.cameras.describe()}", flush=True)
        LightingPanel(self.server, LIGHTING_PREFS)
        self.figure = FigureCapture(
            self.server,
            (self.arrows.shafts, self.arrows.heads),
            args.figure_dir,
            task_name,
        )
        self.figure.button.on_click(self._capture)
        self._wire_episode_controls()
        self.server.on_client_connect(self._place_camera)

    def _add_cell_panels(
        self, args: ViewConfig, cell: Path | None, cells: list
    ) -> CellPanels | None:
        """The agent panels (Session, Rollouts, Policy, Transcript, Compare)."""
        if cell is None:
            return None
        infos = episode_infos(self.open_batch, self.store.files)
        seeds = [i["seed"] for i in infos if i["seed"]]
        return CellPanels(
            self.server,
            cell=cell,
            cells=list(cells),
            batches=self.picker.paths if self.picker.paths is not None else [],
            open_batch=self.open_batch,
            request_switch=self._request_batch,
            request_compare=self._request_compare,
            follow=self.follow_seconds > 0.0,
            seed_hint=seeds[0] if seeds else EVAL_SEED_LO,
            compare_gap=args.compare_gap,
            compare_layout=args.layout,
        )

    def _add_batch_picker(self, cell: Path | None) -> BatchDropdown | None:
        """The caller's own picker, or the flat dropdown for several batches."""
        if self.picker.build is not None:
            self.picker.build(self.server, self._request_batch)
            return None
        if cell is None and len(self.picker.labels) > 1:
            return BatchDropdown(
                self.server,
                self.picker,
                self._request_batch,
                follow=self.follow_seconds > 0.0,
            )
        return None

    def _wire_episode_controls(self) -> None:
        """Episode and frame controls: requests only, served by the main loop."""
        episode, playback = self.episode, self.playback
        episode.slider.on_update(
            lambda _: self._request_episode(int(episode.slider.value))
        )
        episode.dropdown.on_update(
            lambda _: self._request_episode(self.labels.index(episode.dropdown.value))
        )
        episode.prev.on_click(lambda _: self._request_episode(self.store.ep - 1))
        episode.next.on_click(lambda _: self._request_episode(self.store.ep + 1))
        playback.frame.on_update(self._seek)

    def _mark_dirty(self, _=None) -> None:
        self.dirty = True

    def _request_episode(self, index: int) -> None:
        self.wanted_episode = index

    def _request_batch(self, index: int) -> None:
        self.wanted_batch = index

    def _request_compare(self, request: CompareRequest) -> None:
        self.wanted_compare = request

    def _request_source(self, source: DatasetSource | BatchSource) -> None:
        self.wanted_source = source

    def _seek(self, _=None) -> None:
        self.head.frame = int(self.playback.frame.value)
        self.dirty = True

    def _capture(self, _=None) -> None:
        self.figure.capture(self.store.files[self.store.ep].stem, self.head.frame)

    def _place_camera(self, client: viser.ClientHandle) -> None:
        client.camera.position = (-1.8, 1.2, 1.4)
        client.camera.look_at = (0.2, 0.0, 0.7)

    def _sync_ui(self) -> None:
        """Point every episode control at the loaded episode and frame."""
        store, head = self.store, self.head
        n = store.qpos.shape[0]
        self.playback.frame.max = max(1, n - 1)
        self.playback.frame.value = min(head.frame, n - 1)
        self.episode.slider.value = store.ep
        self.episode.dropdown.value = self.labels[store.ep]
        self.episode.list.content = episode_list_html(store.episode_infos(), store.ep)
        self.episode.card.content = episode_card_html(
            store, head.frame, self.realtime_fps
        )
        self.playback.reward_strip.content = sparkline_svg(
            store.rewards, head.frame, flags=episode_flags(store.info)
        )
        self.card_at = time.time()

    def _refresh_batches(self) -> None:
        """Re-discover batches and re-point the picker at the open one."""
        if self.picker.rescan is None:
            return
        if self.cell_panels is not None:
            self.picker.rescan()
            self.cell_panels.refresh_rollouts()
            return
        if self.batch_dropdown is None:
            return
        self.picker.rescan()
        self.batch_dropdown.point_at(self.open_batch)

    def _is_other_batch(self, index: int) -> bool:
        """True when the picked batch is not the one on screen."""
        paths = self.picker.paths
        if paths is not None and 0 <= index < len(paths):
            return paths[index].resolve() != self.open_batch
        return index != self.picker.index

    def _refresh_open_batch(self) -> bool:
        """Pick up episodes added to the batch on screen; True when it grew."""
        store, files = self.store, self.store.files
        fresh = episode_files(self.open_batch)
        if [f.name for f in fresh] == [f.name for f in files]:
            return False
        here = files[store.ep].name if 0 <= store.ep < len(files) else ""
        files[:] = fresh
        # A new episode can sort before the open one.
        self.labels[:] = store.episode_labels()
        store.ep = next((i for i, f in enumerate(files) if f.name == here), 0)
        self.episode.slider.max = max(0, len(files) - 1)
        self.episode.dropdown.options = list(self.labels)
        self._sync_ui()
        print(
            f"[viewer] follow: the open batch now has {len(files)} episodes "
            f"(last {self.labels[-1]!r})",
            flush=True,
        )
        return True

    def _follow_tick(self, now: float) -> None:
        """Rescan the cell (main loop, throttled): batches, versions, run.json."""
        if self.follow_seconds <= 0.0 or now - self.follow_at < self.follow_seconds:
            return
        self.follow_at = now
        self.follow_scans += 1
        labels = self.picker.labels
        before = list(labels)
        self._refresh_batches()
        grew = self._refresh_open_batch()
        if self.cell_panels is not None:
            self.cell_panels.refresh_versions()
        if labels and list(labels) != before:
            fresh = [label for label in labels if label not in before]
            print(
                f"[viewer] follow: {len(labels)} batches"
                + (f" · new/changed: {', '.join(fresh)}" if fresh else ""),
                flush=True,
            )
        note = f"scan {self.follow_scans} · {len(labels)} rollouts"
        if grew:
            note += f" · this one: {len(self.store.files)} episodes"
        note += f" · {time.strftime('%H:%M:%S')}"
        if self.cell_panels is not None:
            self.cell_panels.note_follow(note)
            jump = self.cell_panels.auto_jump
        elif self.batch_dropdown is not None:
            if self.batch_dropdown.follow_note is not None:
                self.batch_dropdown.follow_note.content = (
                    f'<div style="font-size:10px;opacity:.55;">'
                    f"{html.escape(note)}</div>"
                )
            jump = self.batch_dropdown.auto_jump
        else:
            jump = None
        paths = self.picker.paths
        if jump is not None and bool(jump.value) and paths:
            newest = newest_batch(paths)
            if newest is not None and self._is_other_batch(newest):
                self.wanted_batch = newest

    def _exit_request(self, started: float) -> tuple[str, Any] | None:
        """What to say and hand back when the viewer should close, else None."""
        index = self.wanted_batch
        if index is not None and self._is_other_batch(index):
            return (
                f"[viewer] switching to {self.picker.item} "
                f"{self.picker.labels[index]} — "
                "rebuilding scene, the browser tab reconnects by itself",
                index,
            )
        self.wanted_batch = None
        if self.wanted_source is not None:
            return (
                f"[viewer] opening {self.wanted_source.where} — "
                "rebuilding scene, the browser tab reconnects by itself",
                self.wanted_source,
            )
        if self.wanted_compare is not None:
            request, self.wanted_compare = self.wanted_compare, None
            return (
                "[viewer] entering compare mode — rebuilding the scene, "
                "the browser tab reconnects by itself",
                request,
            )
        if self.exit_after > 0.0 and time.time() - started > self.exit_after:
            return (
                f"[viewer] --exit-after-seconds {self.exit_after:g} elapsed, closing",
                None,
            )
        return None

    def _show_frame(self) -> None:
        """Pose the scene at the playhead and refresh what depends on it."""
        store, head = self.store, self.head
        self.data.qpos[: store.qpos.shape[1]] = store.qpos[head.frame]
        mujoco.mj_forward(self.model, self.data)
        store.after_forward(head.frame)
        measured = self.targets.measure(self.inner)
        self.scene.update_from_mjdata(self.data)
        self.arrows.update(store, head.frame, self.show_arrows.value)
        n = store.qpos.shape[0]
        self.episode.progress.value = 100.0 * head.frame / max(1, n - 1)
        # The styled card is throttled: per-frame HTML churn makes the
        # sidebar jitter (~5-10 Hz is plenty). A reward change or the final
        # frame always refreshes, so the success frame never shows a stale 0
        # during the end hold.
        now = time.time()
        reward = float(store.rewards[min(head.frame, len(store.rewards) - 1)])
        if (
            now - self.card_at > 0.15
            or head.frame >= n - 1
            or reward != self.card_reward
        ):
            self.episode.card.content = episode_card_html(
                store, head.frame, self.realtime_fps
            )
            self.playback.reward_strip.content = sparkline_svg(
                store.rewards, head.frame, flags=episode_flags(store.info)
            )
            self.card_at = now
            self.card_reward = reward
        self.targets.show(measured)
        # MuJoCo renders cost real time, so the panels run at their own rate
        # (never faster than the playback rate) on the main loop.
        rate = min(CAMERA_PANEL_HZ, max(self.playback.fps.value, 1))
        if now - self.cameras_at > 1.0 / rate:
            self.cameras.update(self.arrows.pelvis_lookat(store.qpos[head.frame]))
            self.cameras_at = now
            if not self.cameras_logged:
                print(f"[viewer] camera panels rendering ({self.cameras.describe()})")
                self.cameras_logged = True

    def run(
        self,
    ) -> int | CompareRequest | DatasetSource | BatchSource | None:
        """Serve until the viewer is asked for another scene.

        Returns:
            The index of another batch (or task) to open, the
            :class:`~bigym.vr.viewer.agent.compare.model.CompareRequest` the Compare
            button made, the :class:`DatasetSource` / :class:`BatchSource`
            the "Demo directory" field loaded, or None once
            --exit-after-seconds elapsed. Ctrl-C is the other way out.
        """
        self._sync_ui()
        print(
            f"[viewer] episode list: {len(self.store.files)} entries, "
            f"first {self.labels[0]!r}",
            flush=True,
        )
        if self.cell_panels is not None:
            for line in self.cell_panels.describe():
                print(line, flush=True)
        if self.follow_seconds > 0.0:
            print(
                f"[viewer] --follow: rescanning every {self.follow_seconds:g}s",
                flush=True,
            )
        port = self.port
        print(f"[viewer] open http://localhost:{port} (ssh -L {port}:localhost:{port})")
        started = time.time()
        store, head = self.store, self.head
        while True:
            leaving = self._exit_request(started)
            if leaving is not None:
                message, result = leaving
                print(message)
                self.cameras.close()
                self.server.stop()
                return result
            if self.cell_panels is not None:
                self.cell_panels.tick(time.time(), self._refresh_batches)
            if self.follow_seconds > 0.0:
                self._follow_tick(time.time())
            wanted, self.wanted_episode = self.wanted_episode, None
            if wanted is not None and store.load(wanted):
                head.reset()
                self._sync_ui()
                self.dirty = True
            playback = self.playback
            if head.tick(
                store.qpos.shape[0], playback.playing.value, playback.fps.value
            ):
                playback.frame.value = head.frame
                self.dirty = True
            if self.dirty:
                self._show_frame()
                self.dirty = False
            time.sleep(0.004)
