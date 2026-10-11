"""The sidebar controls of the browser viewer."""

from __future__ import annotations

import html
import shlex
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np

from bigym.loco.demos import hub
from bigym.loco.demos.kinematic import run_task_hook
from bigym.loco.objref import path_name
from bigym.vr.viewer import sources
from bigym.vr.viewer.html import note_html
from bigym.vr.viewer.sources import (
    BROWSE_LIMIT,
    MARK,
    DatasetSource,
    browse_dir,
    demo_marker,
    find_source,
    has_demos,
    quick_places,
    subfolders,
)

SKY_OPTIONS = ("grey", "black", "transparent")


# Cameras a task XML may provide for a third-person view, best first; without
# one the panel falls back to a free camera orbiting the robot.
THIRD_PERSON_CAMERAS = ("rec_third", "third_person", "external")


THIRD_PERSON_SIZE = (256, 256)


CAMERA_PANEL_HZ = 10.0  # image widgets are re-encoded per update; 10 Hz is plenty


# Lighting panel state, kept across batch switches (see LightingPanel). Black is
# the default sky: it is what slides, videos and the compare scene want, and
# a capture keeps it.
LIGHTING_PREFS: dict = {
    "sky": "black",
    "environment": 0.7,
    "default_lights": True,
    "shadows": True,
    "key_light": 0.0,
}


class CameraPanels:
    """MuJoCo-rendered image widgets for the viser sidebar.

    The onboard cameras come from the environment's own observation path
    (``BiGymEnv._get_visual_obs``), at the batch's recorded resolution, so
    the panels show the pixels a policy would have seen. One extra view is
    rendered from a third-person camera (a named one from the task XML if it
    has it, else a free camera orbiting the robot).

    Every render runs on the main loop: MuJoCo's GL context belongs to it.
    """

    def __init__(
        self,
        server,
        inner,
        model,
        data,
        camera_keys,
        camera_shape=(84, 84),
    ):
        """Build one image widget per onboard camera plus the third view."""
        self.inner = inner
        self.model = model
        self.data = data
        self.camera_keys = tuple(camera_keys)
        self.shape = (int(camera_shape[0]), int(camera_shape[1]))
        self.images: dict[str, Any] = {}
        self.renderer = None
        self.camera: int | mujoco.MjvCamera | None = None
        self.third_label = ""
        self.error = ""
        with server.gui.add_folder("Camera panels"):
            self.enabled = server.gui.add_checkbox("show", initial_value=True)
            blank = np.zeros((self.shape[0], self.shape[1], 3), dtype=np.uint8)
            for key in self.camera_keys:
                self.images[key] = server.gui.add_image(blank, label=key, format="jpeg")
            self._add_third_person(server)
            self.message = server.gui.add_html("")

    def _add_third_person(self, server) -> None:
        """Add the third-person widget, if an offscreen buffer allows one."""
        height = min(THIRD_PERSON_SIZE[0], int(self.model.vis.global_.offheight))
        width = min(THIRD_PERSON_SIZE[1], int(self.model.vis.global_.offwidth))
        name = ""
        cam_id = -1
        for candidate in THIRD_PERSON_CAMERAS:
            found = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, candidate)
            if found >= 0:
                name, cam_id = candidate, int(found)
                break
        self._third_size = (height, width)
        try:
            self.renderer = mujoco.Renderer(self.model, height, width)
        except Exception as exc:
            self.error = f"third-person renderer unavailable: {exc}"
            return
        if cam_id >= 0:
            self.camera = cam_id
            self.third_label = f"third ({name})"
        else:
            free = mujoco.MjvCamera()
            free.distance = 3.0
            free.elevation = -15.0
            free.azimuth = 160.0
            self.camera = free
            self.third_label = "third (free)"
        self.images[self.third_label] = server.gui.add_image(
            np.zeros((height, width, 3), dtype=np.uint8),
            label=self.third_label,
            format="jpeg",
        )

    def describe(self) -> str:
        """A one-line summary for the startup log."""
        onboard = ", ".join(self.camera_keys) or "none"
        return (
            f"onboard {onboard} @ {self.shape[1]}x{self.shape[0]}"
            f" + {self.third_label or 'no third-person view'}"
        )

    def update(self, lookat=None) -> None:
        """Re-render every panel for the scene currently in ``data``.

        Args:
            lookat: world point the free third-person camera orbits; ignored
                when the scene provides a named third-person camera.
        """
        if not bool(self.enabled.value):
            for handle in self.images.values():
                handle.visible = False
            return
        for handle in self.images.values():
            handle.visible = True
        # The real step() runs the task's _on_step() (which paints scene
        # state such as the reach-target highlight) before the observation
        # is rendered. Same order as the re-renderer.
        run_task_hook(self.inner)
        try:
            obs = self.inner._get_visual_obs()
        except Exception as exc:
            self.message.content = f"<i>camera render failed: {exc}</i>"
            return
        for key in self.camera_keys:
            image = obs.get(f"rgb_{key}")
            if image is not None:
                self.images[key].image = np.ascontiguousarray(
                    np.moveaxis(np.asarray(image), 0, -1)
                )
        if self.renderer is not None and self.third_label:
            camera = self.camera
            if lookat is not None and not isinstance(camera, int):
                assert camera is not None
                camera.lookat[:] = np.asarray(lookat, dtype=float)
            try:
                self.renderer.update_scene(self.data, camera=camera)
                self.images[self.third_label].image = self.renderer.render()
            except Exception as exc:
                self.message.content = f"<i>third-person render failed: {exc}</i>"
                return
        self.message.content = f"<i>{html.escape(self.error)}</i>" if self.error else ""

    def rebind(self, inner, model, data) -> None:
        """Point the existing widgets at another environment.

        Compare mode builds one environment per slot and lets a dropdown
        choose whose cameras the panels show. The widgets are unchanged
        (every slot is the same task, so the camera names and resolution
        match); only the offscreen renderer has to be rebuilt, because a
        ``mujoco.Renderer`` is bound to one model.

        Args:
            inner: The slot's native BiGym env.
            model: Its ``mujoco.MjModel``.
            data: Its ``mujoco.MjData``.
        """
        self.inner, self.model, self.data = inner, model, data
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if not self.third_label:
            return
        if not isinstance(self.camera, int):
            pass  # the free camera carries no model-bound state
        else:
            name = self.third_label[len("third (") : -1]
            found = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            self.camera = int(found) if found >= 0 else 0
        try:
            self.renderer = mujoco.Renderer(model, *self._third_size)
        except Exception as exc:
            self.error = f"third-person renderer unavailable: {exc}"

    def close(self) -> None:
        """Release the third-person renderer."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def directory_panel(server, current: str, request) -> None:
    """The "Demo directory" panel: browse and open another directory live.

    The browser lists folders on the machine running the viewer: a "quick"
    dropdown of starting points, the path (editable), "up", and a dropdown
    of the subfolders, ``●`` marking those that hold demos (dataset task
    folders, demo batches, agent cells, or folders directly above them).
    Load scans the path on a worker thread (the same discovery as
    ``--demo-dir``; empty is the published dataset) and hands what it found
    to ``request``; a directory with nothing to show leaves the scene alone
    and says so in the panel.

    Args:
        server: The viser server.
        current: What the path starts with.
        request: Called with the :class:`DatasetSource` or
            :class:`BatchSource` to open.
    """
    jump, open_sub = "jump to …", "open a subfolder …"
    hint = "empty: the published dataset · several: separate with spaces"
    places = quick_places()
    listed: dict[str, Path] = {}
    shown: dict[str, str | None] = {"text": None}
    with server.gui.add_folder("Demo directory"):
        quick = server.gui.add_dropdown("quick", options=[jump, *places])
        path = server.gui.add_text("path", initial_value=current)
        up = server.gui.add_button("up")
        folders = server.gui.add_dropdown("folders", options=[open_sub])
        load = server.gui.add_button("load")
        note = server.gui.add_html("")

    def refresh() -> None:
        text = str(path.value)
        if text == shown["text"]:
            return
        shown["text"] = text
        listed.clear()
        base = browse_dir(text)
        error = False
        if base is None:
            try:
                several = len(shlex.split(text)) > 1
            except ValueError:
                several = False
            status = "several folders: load opens them together"
            if not several:
                status, error = f"{text.strip()} is not a folder", True
        else:
            try:
                names, total = subfolders(base, BROWSE_LIMIT)
            except OSError as exc:
                names, total = [], 0
                status, error = f"cannot list {base}: {exc.strerror or exc}", True
            else:
                for name in names:
                    child = base / name
                    listed[(MARK if has_demos(child) else "") + name] = child
                status = f"{base}: {total} subfolder" + ("" if total == 1 else "s")
                if total > len(names):
                    status += f" (first {len(names)} listed)"
                if demo_marker(base):
                    status += " · this folder holds demos"
                status += " · ● holds demos"
        folders.options = [open_sub, *listed]
        folders.value = open_sub
        note.content = note_html(f"{status}\n{hint}", error)

    def go(target: Path | str) -> None:
        path.value = target if isinstance(target, str) else shlex.quote(str(target))
        refresh()

    def on_quick(_) -> None:
        if quick.value in places:
            target = places[quick.value]
            quick.value = jump
            go(target)

    def on_up(_) -> None:
        go((browse_dir(str(path.value)) or Path.cwd()).parent)

    def on_folder(_) -> None:
        child = listed.get(str(folders.value))
        if child is not None:
            go(child)

    def scan(text: str) -> None:
        try:
            source = find_source(text)
        except Exception as exc:
            note.content = note_html(str(exc).splitlines()[0], error=True)
            return
        if isinstance(source, DatasetSource) and source.dirs[0] is None:
            note.content = note_html(f"fetching {source.labels[0]} …")
            try:
                hub.task_files(source.tasks[0], [])
            except Exception as exc:
                note.content = note_html(str(exc).splitlines()[0], error=True)
                return
        if text.strip():
            if text.strip() in sources.RECENT:
                sources.RECENT.remove(text.strip())
            sources.RECENT.insert(0, text.strip())
        note.content = note_html(f"opening {source.where} …")
        request(source)

    def on_load(_) -> None:
        text = str(path.value)
        shown["text"] = None  # the next edit or navigation lists again
        note.content = note_html(
            f"looking for demos in {text} …" if text.strip() else "listing tasks …"
        )
        threading.Thread(target=scan, args=(text,), daemon=True).start()

    quick.on_update(on_quick)
    path.on_update(lambda _: refresh())
    up.on_click(on_up)
    folders.on_update(on_folder)
    load.on_click(on_load)
    refresh()


def task_picker(source: DatasetSource, idx: int):
    """Build the "Task" dropdown for a dataset (a :class:`Picker`'s ``build``).

    Picking a task the Hub cache does not hold yet fetches its metadata on a
    worker thread, with a status line, while the open task keeps playing;
    the switch is requested once it is there. Episodes are downloaded as
    they are opened, as for the first task.

    Args:
        source: The dataset's tasks.
        idx: The task on screen.

    Returns:
        ``build(server, request_switch)``.
    """

    def build(server, request_switch) -> None:
        with server.gui.add_folder("Task"):
            dropdown = server.gui.add_dropdown(
                "task", options=list(source.labels), initial_value=source.labels[idx]
            )
            note = server.gui.add_html(note_html(source.where))
        wanted = {"index": idx}

        def fetch(target: int) -> None:
            if source.dirs[target] is None:
                try:
                    hub.task_files(source.tasks[target], [])
                except Exception as exc:
                    note.content = note_html(str(exc).splitlines()[0], error=True)
                    return
            if wanted["index"] == target:  # not overtaken by a later pick
                note.content = note_html(f"building {source.labels[target]} …")
                request_switch(target)

        def on_pick(_) -> None:
            target = source.labels.index(str(dropdown.value))
            wanted["index"] = target
            if target == idx:
                note.content = note_html(source.where)
                return
            label = source.labels[target]
            note.content = note_html(
                f"fetching {label} …"
                if source.dirs[target] is None
                else f"building {label} …"
            )
            threading.Thread(target=fetch, args=(target,), daemon=True).start()

        dropdown.on_update(on_pick)

    return build


@dataclass
class Picker:
    """What the browser viewer can switch to; the caller rebuilds the scene.

    Attributes:
        labels: One label per batch (or dataset task). A rescan updates the
            list in place, so an index the viewer returns stays valid.
        index: The one on screen.
        item: What a label names, ``batch`` or ``task``, for the log.
        paths: The batch directories behind the labels, updated in place with
            them; None for dataset tasks.
        rescan: Re-discovers ``paths`` and ``labels`` (a finished job wrote a
            new batch); None when nothing can change.
        build: Builds a custom picker instead of the flat "Demo batch"
            dropdown, as ``build(server, request_switch)``; request_switch
            takes the index to open.
    """

    labels: list[str]
    index: int = 0
    item: str = "batch"
    paths: list[Path] | None = None
    rescan: Callable[[], object] | None = None
    build: Callable[[Any, Callable[[int], None]], None] | None = None


class EpisodePanel:
    """The "Episode" folder: info card, progress, slider, list and picker."""

    def __init__(self, server, labels: list[str], count: int) -> None:
        """Add the folder's widgets."""
        with server.gui.add_folder("Episode"):
            self.card = server.gui.add_html("")
            self.progress = server.gui.add_progress_bar(0.0, color=(96, 165, 250))
            self.slider = server.gui.add_slider(
                "index", min=0, max=count - 1, step=1, initial_value=0
            )
            # The list is read-only (viser HTML cannot call back); the dropdown
            # next to it is what picks an episode, with the same labels.
            self.list = server.gui.add_html("")
            self.dropdown = server.gui.add_dropdown(
                "episode", options=labels, initial_value=labels[0]
            )
            self.prev = server.gui.add_button("prev")
            self.next = server.gui.add_button("next")


class PlaybackPanel:
    """The "Playback" folder: frame slider, reward strip, play and fps."""

    def __init__(self, server, realtime_fps: float) -> None:
        """Add the folder's widgets, playing at the recorded rate."""
        with server.gui.add_folder("Playback"):
            self.frame = server.gui.add_slider(
                "frame", min=0, max=1, step=1, initial_value=0
            )
            self.reward_strip = server.gui.add_html("")
            self.playing = server.gui.add_checkbox("play", initial_value=True)
            self.fps = server.gui.add_slider(
                "fps",
                min=1,
                max=max(100, int(round(realtime_fps))),
                step=1,
                initial_value=int(round(realtime_fps)),
            )


class LightingPanel:
    """The "Lighting" folder: sky, environment map, default and key lights.

    mjviser carries geometry but not MuJoCo's lights, so viser lights the
    scene with a flat environment map: every face gets the same intensity
    and the model -- 298 of 339 geoms are plain grey rgba, 0.5 or 0.7 --
    reads as one white blob. The fix is to take intensity away, not to add
    sources. Sliders, because the right value is a judgement made looking at
    the render.

    Sky: viser draws no sky, so the page colour shows through and captures
    carry alpha there. "grey" shows the environment map itself, blurred and
    dimmed, as a neutral backdrop that both the white fixtures and the
    darkened robot read against; "black" is the same map at zero intensity,
    for slides and video; "transparent" keeps the cut-out for compositing.
    All of it is baked into captures.

    The panel is rebuilt with the scene on every batch switch, so each
    control is seeded from ``prefs`` (process-wide) and writes back on
    change: pick a sky once, keep it for the next batches.
    """

    def __init__(self, server, prefs: dict) -> None:
        """Add the key light and the folder, and apply the preferences."""
        self.server = server
        self.prefs = prefs
        # Aimed like the MJCF main_light (pos 0 0 3, dir 0 0 -1), off by
        # default: turn it up only if the flatter render needs shape.
        self.key = server.scene.add_light_directional(
            "/lighting/key", intensity=0.0, cast_shadow=True, visible=False
        )
        with server.gui.add_folder("Lighting"):
            self.sky = server.gui.add_dropdown(
                "sky", options=SKY_OPTIONS, initial_value=prefs["sky"]
            )
            self.environment = server.gui.add_slider(
                "environment",
                min=0.0,
                max=1.5,
                step=0.05,
                initial_value=prefs["environment"],
            )
            self.default_lights = server.gui.add_checkbox(
                "default lights", initial_value=prefs["default_lights"]
            )
            self.shadows = server.gui.add_checkbox(
                "shadows", initial_value=prefs["shadows"]
            )
            self.key_light = server.gui.add_slider(
                "key light",
                min=0.0,
                max=3.0,
                step=0.05,
                initial_value=prefs["key_light"],
            )
            self.message = server.gui.add_html("")
        for control in (
            self.sky,
            self.environment,
            self.default_lights,
            self.shadows,
            self.key_light,
        ):
            control.on_update(self.apply)
        # An untouched control never fires on_update, so a fresh scene needs
        # one explicit pass for the preferences to hold.
        self.apply()

    def apply(self, _=None) -> None:
        """Push the controls to the scene and remember them."""
        try:
            sky = str(self.sky.value)
            # Every sky is the environment cube map shown as background;
            # black is that map at zero intensity. Not set_background_image:
            # its quad is rescaled per frame from the camera's film size, and
            # a capture whose aspect differs from the browser canvas can go
            # out before the quad catches up, leaving transparent strips at
            # the sides.
            self.server.scene.configure_environment_map(
                environment_intensity=float(self.environment.value),
                background=sky != "transparent",
                background_blurriness=1.0,
                background_intensity=0.5 if sky == "grey" else 0.0,
            )
            self.server.scene.configure_default_lights(
                enabled=bool(self.default_lights.value),
                cast_shadow=bool(self.shadows.value),
            )
            self.key.intensity = float(self.key_light.value)
            self.key.visible = float(self.key_light.value) > 0.0
        except Exception as exc:
            self.message.content = f"<i>{exc}</i>"
            return
        self.message.content = ""
        self.prefs.update(
            sky=sky,
            environment=float(self.environment.value),
            default_lights=bool(self.default_lights.value),
            shadows=bool(self.shadows.value),
            key_light=float(self.key_light.value),
        )


class FigureCapture:
    """The "Figure capture" folder: the current view as a PNG at print size.

    Frame a pose by dragging, then grab it. viser renders in the browser, so
    the capture is capped by the window: open it wide. Only the velocity
    arrows (a debug overlay) are hidden for the shot; the target balls and
    their glow stay, as MuJoCo shows the reached state too.
    """

    def __init__(self, server, overlays, figure_dir: str, task_name: str) -> None:
        """Add the folder; ``overlays`` are the scene handles hidden in a shot."""
        self.server = server
        self.overlays = overlays
        self.figure_dir = figure_dir
        self.task_name = task_name
        with server.gui.add_folder("Figure capture"):
            self.width = server.gui.add_number(
                "width", initial_value=1000, min=64, step=8
            )
            self.height = server.gui.add_number(
                "height", initial_value=1000, min=64, step=8
            )
            self.button = server.gui.add_button("capture png")
            self.message = server.gui.add_html("")

    def capture(self, stem: str, frame: int) -> None:
        """Write the first client's view to ``<figure_dir>/<task>_<stem>_f<frame>.png``."""
        clients = list(self.server.get_clients().values())
        if not clients:
            self.message.content = "<i>no browser connected</i>"
            return
        was = [handle.visible for handle in self.overlays]
        try:
            for handle in self.overlays:
                handle.visible = False
            image = clients[0].get_render(
                height=int(self.height.value),
                width=int(self.width.value),
                transport_format="png",
            )
        except Exception as exc:
            self.message.content = f"<i>capture failed: {exc}</i>"
            return
        finally:
            for handle, visible in zip(self.overlays, was, strict=True):
                handle.visible = visible
        out = Path(self.figure_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Batch stems are collection timestamps, so a directory of captures
        # from several tasks is unreadable without the task in the name.
        prefix = f"{path_name(self.task_name)}_" if self.task_name else ""
        path = out / f"{prefix}{stem}_f{frame:05d}.png"
        imageio.imwrite(path, image)
        print(f"[viewer] captured {path}", flush=True)
        self.message.content = f"saved <code>{path.name}</code>"


class BatchDropdown:
    """The "Demo batch" folder: a flat picker, plus the --follow controls."""

    def __init__(self, server, picker: Picker, request, follow: bool) -> None:
        """Add the dropdown; ``request`` is called with the picked index."""
        self.picker = picker
        self.auto_jump = None
        self.follow_note = None
        with server.gui.add_folder("Demo batch"):
            self.dropdown = server.gui.add_dropdown(
                "batch",
                options=list(picker.labels),
                initial_value=picker.labels[picker.index],
            )
            self.dropdown.on_update(
                lambda _: request(picker.labels.index(self.dropdown.value))
            )
            if follow:
                self.auto_jump = server.gui.add_checkbox(
                    "auto-jump to newest", initial_value=False
                )
                self.follow_note = server.gui.add_html("")

    def point_at(self, open_batch: Path) -> None:
        """Re-list the batches and select the one on screen."""
        picker = self.picker
        here = picker.index
        if picker.paths is not None:
            resolved = [b.resolve() for b in picker.paths]
            if open_batch in resolved:
                here = resolved.index(open_batch)
        self.dropdown.options = list(picker.labels)
        self.dropdown.value = picker.labels[min(here, len(picker.labels) - 1)]
