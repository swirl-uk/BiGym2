"""Running a compare view: fetch the missing rollouts, build the scene, play it."""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np
from mjviser import ViserMujocoScene

from bigym.vr.viewer.agent.compare.hud import CompareLegend
from bigym.vr.viewer.agent.compare.model import CompareRequest, CompareSlot, fill_slot
from bigym.vr.viewer.agent.compare.record import (
    RECORD_LABEL_SCALE,
    record_compare_session,
)
from bigym.vr.viewer.agent.jobs import (
    AgentJob,
    progress_fraction,
    progress_label,
    read_progress,
)
from bigym.vr.viewer.episodes import episode_info
from bigym.vr.viewer.layout import (
    SLOT_TINTS,
    compare_camera_pose,
    focus_pose,
    grid_note,
    grid_offsets,
    slot_colour,
    slot_grid,
)
from bigym.vr.viewer.mjviser_compat import (
    hide_reach_targets,
    patch_mjviser,
    patch_mjviser_track_props,
)
from bigym.vr.viewer.scene import (
    BILLBOARD_Z,
    PLATE_FACE,
    SlotBillboard,
    apply_lighting,
    compare_server,
    fly_clients_to,
    pelvis_body_id,
    pelvis_position,
    place_billboard,
    pose_slot,
    slot_scene_build,
    slot_status_text,
    tint_slot_robot,
)


def missing_job(slot: CompareSlot, seed: int) -> AgentJob:
    """The ``bigym-agent replay --no-video`` job one empty slot needs."""
    if slot.policy:
        return AgentJob(
            "replay",
            slot.cell,
            0,
            str(seed),
            policy=slot.policy,
            extra=("--no-video",),
        )
    return AgentJob(
        "replay",
        slot.cell,
        int(slot.version or 0),
        str(seed),
        extra=("--no-video",),
    )


class MissingWork:
    """The rollouts a compare request is still waiting for, as a state machine.

    Every slot with no stored episode on the compare seed becomes a
    ``bigym-agent replay <cell> --version N --seeds <seed> --no-video``
    subprocess. They all start at once (each is its own process with its own
    environment, and physics is CPU-bound), and two slots naming the same
    policy share one job, so no episode file is written twice.

    Waiting is the caller's business, one :meth:`poll` per main-loop tick,
    so the GUI keeps its progress bars and Leave button while jobs run.
    :meth:`bars` is what the Compare folder draws.
    """

    def __init__(self, request: CompareRequest, timeout: float = 3600.0):
        """Plan the work for one request (nothing starts yet).

        Args:
            request: The request whose ``missing`` slots need an episode.
            timeout: Give up on a job that takes longer than this.
        """
        self.request = request
        self.timeout = float(timeout)
        self.jobs: list[AgentJob] = []
        self.heads: list[str] = []
        self.pending: set[int] = set()
        self.started = 0.0
        self.error = ""
        self.done = False

    def start(self) -> list[AgentJob]:
        """Launch one job per distinct missing policy (main loop only).

        Returns:
            The jobs started, in slot order; empty when every slot already
            has its episode, in which case the work is done at once.
        """
        seen: set[tuple] = set()
        for slot in list(self.request.missing):
            key = (str(slot.cell), slot.policy or int(slot.version or 0))
            if key in seen:
                continue
            seen.add(key)
            job = missing_job(slot, self.request.seed)
            self.jobs.append(job)
            self.heads.append(
                f"{slot.name} · {slot.version_name} · seed {self.request.seed}"
            )
            print(f"[viewer] {' '.join(job.command)} > {job.log_path}", flush=True)
            job.start()
        self.pending = set(range(len(self.jobs)))
        self.started = time.time()
        if self.jobs:
            print(
                f"[viewer] compare: {len(self.jobs)} replay job(s) running in parallel",
                flush=True,
            )
        else:
            self.resolve()
        return list(self.jobs)

    def poll(self) -> bool:
        """Poll every running job; True once there is nothing left to wait for.

        A job that fails ends the work: the others are killed and
        :attr:`error` says what went wrong.
        """
        if self.done:
            return True
        for index in sorted(self.pending):
            job = self.jobs[index]
            if not job.poll():
                continue
            self.pending.discard(index)
            if job.error:
                self.error = job.error
                self.stop()
                self.done = True
                return True
        if self.pending:
            if time.time() - self.started > self.timeout:
                names = ", ".join(self.jobs[i].label for i in sorted(self.pending))
                self.error = f"replay timed out for: {names}"
                self.stop()
                self.done = True
            return self.done
        self.resolve()
        return True

    def resolve(self) -> None:
        """Re-read every slot now the jobs are over.

        The scene is built only once every file exists, so this re-runs the
        whole resolution (not just the slots that had jobs) and sets
        :attr:`error` when something is still not there.
        """
        for slot in self.request.slots:
            fill_slot(slot, self.request.seed)
        self.request.missing = [s for s in self.request.slots if s.episode is None]
        if self.request.missing:
            names = ", ".join(
                f"{s.name} {s.version_name}" for s in self.request.missing
            )
            self.error = f"still no episode on seed {self.request.seed} for: {names}"
        self.done = True

    def stop(self) -> None:
        """Kill every job still running (the Leave button, or a failure)."""
        for index in sorted(self.pending):
            self.jobs[index].stop()
        self.pending.clear()

    def bars(self) -> list[tuple[float, str, bool]]:
        """One ``(fraction, label, animated)`` per job, in the order started."""
        out: list[tuple[float, str, bool]] = []
        for index, job in enumerate(self.jobs):
            if index not in self.pending:
                out.append((1.0, f"{self.heads[index]} · done", False))
                continue
            state = read_progress(job.out_dir)
            out.append(
                (
                    progress_fraction(state),
                    progress_label(self.heads[index], state),
                    state is None,
                )
            )
        return out


def wait_for_missing(
    request: CompareRequest,
    legend: CompareLegend,
    *,
    timeout: float = 3600.0,
    deadline: float = 0.0,
    tick: float = 0.25,
    log_every: float = 5.0,
) -> bool:
    """Produce the rollouts the request is missing, with the GUI alive.

    Drives :class:`MissingWork` from this loop instead of blocking on it:
    every tick polls the jobs, reads each one's ``.progress.json`` and
    redraws its bar, so the sidebar shows where each rollout has got to and
    the Leave button (which the legend already owns) still works.

    Args:
        request: The request being produced.
        legend: The Compare folder to draw into.
        timeout: Give up on a job that takes longer than this.
        deadline: Wall-clock seconds for the whole wait (0 = no limit);
            ``--exit-after-seconds`` passes its budget here so a headless
            run cannot hang on a rollout that never ends.
        tick: Seconds between polls.
        log_every: Print every bar's label this often, for the log.

    Returns:
        True when every slot has an episode and the scene can be built;
        False when the user left, a job failed or the deadline passed.
    """
    work = MissingWork(request, timeout=timeout)
    work.start()
    note = (
        f"{len(work.jobs)} rollout(s) to produce on seed {request.seed}"
        if work.jobs
        else ""
    )
    started = time.time()
    said = 0.0
    while True:
        if legend.left:
            print("[viewer] compare: Leave pressed, stopping the replays", flush=True)
            work.stop()
            return False
        finished = work.poll()
        rows = work.bars()
        legend.show_jobs(rows, note)
        now = time.time()
        if rows and (now - said >= log_every or finished):
            said = now
            for fraction, label, _ in rows:
                print(
                    f"[viewer] compare: {label} ({100.0 * fraction:.0f}%)", flush=True
                )
        if finished:
            break
        if deadline > 0.0 and now - started > deadline:
            work.stop()
            legend.note_error(f"gave up after {deadline:g}s producing the rollouts")
            print(
                f"[viewer] compare: {deadline:g}s elapsed while producing", flush=True
            )
            return False
        time.sleep(tick)
    if work.error:
        legend.note_error(work.error)
        print(f"[viewer] compare: {work.error}", flush=True)
        return False
    legend.jobs_done()
    return True


def build_compare_scene(
    request: CompareRequest,
    *,
    port: int,
    build_env,
    load_metadata,
    lighting_prefs: dict,
    server=None,
    legend: CompareLegend | None = None,
) -> dict:
    """Build one viser server holding one environment per compare slot.

    Each slot gets its own ``MjModel``/``MjData`` from the ordinary
    ``build_env`` path, reset on the compare seed so the props sit where
    that seed puts them, a distinct mjviser node prefix and a frame that
    carries its place in the grid the request's ``layout`` asks for: one
    column is ``gap`` metres along +y, one row is ``gap`` metres along -x.

    Nothing is shown while that happens. Every slot's two root frames are
    created **before** any of its geometry (so nothing is ever drawn at the
    origin first) and start hidden; when the last slot is built, all of them
    are posed to frame 0 of their replay and only then revealed together.
    The legend says ``building 2/3 …`` in the meantime.

    Args:
        request: The resolved request (every slot must have an episode).
        port: The viser port.
        build_env: :func:`bigym.vr.viewer.stores.build_env`.
        load_metadata: :func:`bigym.vr.viewer.stores.load_metadata`.
        lighting_prefs: The viewer's lighting preferences.
        server: A server already showing the Compare folder (the one the
            progress bars ran in), or None to open one on ``port``.
        legend: Its legend, or None to build one.

    Returns:
        A dict with ``server``, ``legend``, ``slots`` (one dict per slot:
        env, inner, model, data, scene, qpos, rewards, info, billboard,
        frames, geom_rgba, cell), ``offsets`` and ``fps``.
    """
    patch_mjviser()
    if server is None:
        server = compare_server(port)
    if legend is None:
        legend = CompareLegend(server, request)
    # One /fixed_bodies root for every slot: see slot_scene_build.
    fixed_frame = server.scene.add_frame("/fixed_bodies", show_axes=False)
    built: list[dict] = []
    total = len(request.slots)
    places = slot_grid(total, request.layout)
    offsets = grid_offsets(total, request.gap, request.layout)
    print(
        f"[viewer] compare layout {request.layout}: "
        f"{grid_note(total, request.gap, request.layout)}",
        flush=True,
    )
    fps = 50.0
    camera_keys: tuple[str, ...] = ("head",)
    camera_shape: tuple[int, int] = (84, 84)
    for i, slot in enumerate(request.slots):
        legend.note_building(i + 1, total)
        legend.set_status(i, "building …")
        legend.refresh()
        assert slot.episode is not None
        batch = Path(slot.episode).parent
        metadata = load_metadata(batch)
        step_seconds = metadata.get("control_step_seconds")
        if step_seconds:
            fps = 1.0 / float(step_seconds)
        task_meta = metadata.get("task") or {}
        camera_keys = tuple(task_meta.get("camera_keys") or ("head",))
        camera_shape = tuple(task_meta.get("camera_shape") or (84, 84))
        env = build_env(metadata)
        env.reset(seed=request.seed)
        inner = env.inner_env
        model, data = inner.model, inner.data
        # Compare mode leaves the targets out rather than drawing N of them.
        hide_reach_targets(inner)
        prefix = f"slot{i}"
        offset = offsets[i]
        # The offset frames exist before a single geom goes under them, so
        # no slot is ever drawn at the origin and then moved.
        body_frame = server.scene.add_frame(
            f"/bodies/{prefix}", show_axes=False, position=offset, visible=False
        )
        slot_fixed = server.scene.add_frame(
            f"/fixed_bodies/{prefix}", show_axes=False, position=offset, visible=False
        )
        patch_mjviser_track_props(model, inner)
        # The tint goes in only for the bake, and comes straight back out:
        # MuJoCo's renderer (the camera panels) reads geom_rgba live, and it
        # must show the policy's own view, not a blue-tinted robot.
        pristine = np.array(model.geom_rgba, copy=True)
        tinted = tint_slot_robot(model, SLOT_TINTS[i % len(SLOT_TINTS)])
        with slot_scene_build(
            server, prefix, keep_grid=i == 0, fixed_frame=fixed_frame
        ):
            scene = ViserMujocoScene(server=server, mj_model=model, num_envs=1)
        model.geom_rgba[:] = pristine
        restored = bool(np.array_equal(np.asarray(model.geom_rgba), pristine))
        scene.camera_tracking_enabled = False
        with np.load(slot.episode) as episode:
            qpos = np.asarray(episode["full_qpos"], dtype=np.float64)
            rewards = np.asarray(episode["reward"]).reshape(-1)
        info = episode_info(Path(slot.episode))
        billboard = SlotBillboard(
            server,
            f"/compare/{prefix}/billboard",
            position=(offset[0], offset[1], offset[2] + BILLBOARD_Z),
            visible=False,
        )
        billboard.write(slot.label, "step 0", slot_colour(i))
        built.append(
            {
                "slot": slot,
                "env": env,
                "inner": inner,
                "model": model,
                "data": data,
                "scene": scene,
                "qpos": qpos,
                "rewards": rewards,
                "info": info,
                "pelvis": pelvis_body_id(model),
                "billboard": billboard,
                "frames": (body_frame, slot_fixed),
                "geom_rgba": pristine,
                "offset": offset,
                "place": places[i],
                "prefix": prefix,
            }
        )
        print(
            f"[viewer] compare slot {i}: {prefix} at row {places[i][0]} "
            f"column {places[i][1]} (x {offset[0]:+.2f}, y {offset[1]:+.2f}), "
            f"built hidden, {qpos.shape[0]} frames from {slot.source}/"
            f"{Path(slot.episode).name}, {tinted} geoms tinted",
            flush=True,
        )
        print(
            f"[viewer] compare slot {i}: geom_rgba restored after the mesh "
            f"bake — model untouched: {restored}",
            flush=True,
        )
        print(f"[viewer] compare slot {i} label: {slot.label}", flush=True)
    apply_lighting(server, lighting_prefs)
    # Every slot is posed to frame 0 of its own replay before anything is
    # shown: the reset pose is not the first frame, and a robot standing in
    # the dishwasher for one update is exactly what that looks like.
    for i, entry in enumerate(built):
        pose_slot(entry, 0, mujoco)
        anchor = place_billboard(entry)
        entry["billboard"].write(
            entry["slot"].label, slot_status_text(entry, 0), slot_colour(i)
        )
        legend.set_status(i, slot_status_text(entry, 0))
        pelvis = pelvis_position(entry["data"], entry["pelvis"])
        print(
            f"[viewer] compare slot {i}: posed to frame 0 of the replay "
            f"(not the reset pose), pelvis "
            f"({pelvis[0]:.2f}, {pelvis[1]:.2f}, {pelvis[2]:.2f})",
            flush=True,
        )
        print(
            f"[viewer] compare slot {i}: billboard ({entry['billboard'].kind}) at "
            f"({anchor[0]:.2f}, {anchor[1]:.2f}, {anchor[2]:.2f}) = pelvis "
            f"+ offset ({entry['offset'][0]:.2f}, {entry['offset'][1]:.2f}) "
            f"+ {BILLBOARD_Z:.2f} m, facing "
            f"({PLATE_FACE[0]:g}, {PLATE_FACE[1]:g}, {PLATE_FACE[2]:g})",
            flush=True,
        )
    for entry in built:
        for frame in entry["frames"]:
            frame.visible = True
        entry["billboard"].reveal()
    legend.note_ready()
    print(f"[viewer] compare: {len(built)} scenes revealed together", flush=True)
    for line in legend.describe():
        print(line, flush=True)
    return {
        "server": server,
        "legend": legend,
        "slots": built,
        "offsets": offsets,
        "fps": fps,
        "mujoco": mujoco,
        "camera_keys": camera_keys,
        "camera_shape": camera_shape,
    }


def run_compare(
    request: CompareRequest,
    *,
    port: int,
    build_env,
    load_metadata,
    camera_panels,
    lighting_prefs: dict,
    exit_after: float = 0.0,
    timeout: float = 3600.0,
    record: dict | None = None,
) -> None:
    """Play N policies of one task side by side until the user leaves.

    The server comes up first, with the Compare folder in it, so the slots
    that still need a rollout are produced with a progress bar each and a
    working Leave button (:func:`wait_for_missing`); the scene is built into
    that same server once every slot has its episode.

    The scene is laid out as the request's ``layout`` asks (columns along
    +y, rows receding along -x), the camera starts where the whole grid is
    in view, the Compare folder's *focus* dropdown flies every client to one
    slot, and the camera panels start hidden behind its *camera panels*
    checkbox.

    Args:
        request: A resolved request; slots with no stored episode on the
            seed are rolled out first.
        port: The viser port.
        build_env: :func:`bigym.vr.viewer.stores.build_env`.
        load_metadata: :func:`bigym.vr.viewer.stores.load_metadata`.
        camera_panels: :class:`bigym.vr.viewer.widgets.CameraPanels`.
        lighting_prefs: The viewer's lighting preferences.
        exit_after: Serve this many seconds, then return (0 = until the
            Leave button or Ctrl-C). It bounds the whole compare, the
            rollouts included, so a headless run always terminates.
        timeout: Give up on a rollout that takes longer than this.
        record: None, or what :func:`record_compare_session` needs to write
            a video of the scene once it is built (``path``, ``seconds``,
            ``size``, ``fps``, ``browser``, ``view``, ``zoom``, ``speed``,
            ``stay``), plus ``label_scale`` for the billboards; the server
            closes after the file unless ``stay`` is true.
    """
    server = compare_server(port)
    legend = CompareLegend(server, request)
    print(f"[viewer] open http://localhost:{port} (ssh -L {port}:localhost:{port})")
    started = time.time()
    if not wait_for_missing(request, legend, timeout=timeout, deadline=exit_after):
        server.stop()
        return
    scene = build_compare_scene(
        request,
        port=port,
        build_env=build_env,
        load_metadata=load_metadata,
        lighting_prefs=lighting_prefs,
        server=server,
        legend=legend,
    )
    server, slots, fps = scene["server"], scene["slots"], scene["fps"]
    legend = scene["legend"]
    mujoco = scene["mujoco"]
    offsets = scene["offsets"]
    length = max(int(s["qpos"].shape[0]) for s in slots)
    state = {"frame": 0, "dirty": True, "leave": False, "camera": 0}
    names = [
        f"slot {i} · {s['slot'].name} {s['slot'].version_name}"
        for i, s in enumerate(slots)
    ]
    home = compare_camera_pose(offsets)

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "frame", min=0, max=max(1, length - 1), step=1, initial_value=0
        )
        playing = server.gui.add_checkbox("play", initial_value=True)
        fps_slider = server.gui.add_slider(
            "fps",
            min=1,
            max=max(100, int(round(fps))),
            step=1,
            initial_value=int(round(fps)),
        )
    camera_dd = server.gui.add_dropdown(
        "camera slot", options=names, initial_value=names[0]
    )
    panels = camera_panels(
        server,
        slots[0]["inner"],
        slots[0]["model"],
        slots[0]["data"],
        scene["camera_keys"],
        scene["camera_shape"],
    )
    # Compare mode is about the scene, so the panels start hidden; the
    # Compare folder's checkbox is what turns them on, and the camera slot
    # dropdown still says whose view they then show.
    panels.enabled.value = False
    legend.leave.on_click(lambda _: state.update(leave=True))
    frame_slider.on_update(
        lambda _: state.update(frame=int(frame_slider.value), dirty=True)
    )
    camera_dd.on_update(lambda _: state.update(camera=names.index(camera_dd.value)))

    @server.on_client_connect
    def _(client) -> None:
        client.camera.position = home[0]
        client.camera.look_at = home[1]

    # Clients that were already watching the progress bars do not fire
    # on_client_connect again, so they are moved by hand.
    fly_clients_to(server, *home)
    print(
        f"[viewer] compare: task {request.task} · seed {request.seed} · "
        f"{len(slots)} scenes built · "
        f"{grid_note(len(slots), request.gap, request.layout)}",
        flush=True,
    )
    print(
        f"[viewer] compare camera: at ({home[0][0]:.2f}, {home[0][1]:.2f}, "
        f"{home[0][2]:.2f}) looking at ({home[1][0]:.2f}, {home[1][1]:.2f}, "
        f"{home[1][2]:.2f}) — every scene in view",
        flush=True,
    )
    head_time = time.time()
    shown_camera = 0
    camera_at = 0.0
    # Only a change of the Compare folder's checkbox writes to the panels'
    # own "show", so the two controls do not fight each other.
    cameras_on = False

    def show(frame: int) -> None:
        """Pose every slot at ``frame`` and say so on its plate and the legend."""
        for i, entry in enumerate(slots):
            here = pose_slot(entry, frame, mujoco)
            text = slot_status_text(entry, here)
            # The billboard rides above the pelvis, so it stays with the
            # robot as it walks instead of marking the origin.
            place_billboard(entry)
            entry["billboard"].write(entry["slot"].label, text, slot_colour(i))
            legend.set_status(i, text)
        legend.refresh()

    try:
        if record:
            scale = float(record.get("label_scale", RECORD_LABEL_SCALE) or 1.0)
            for entry in slots:
                entry["billboard"].width *= scale
                entry["billboard"].drawn = ""  # redrawn at the new size by show()
            written = record_compare_session(
                server,
                port,
                show,
                home,
                scene_fps=fps,
                length=length,
                offsets=offsets,
                **{k: v for k, v in record.items() if k != "label_scale"},
            )
            if not record.get("stay"):
                return
            state.update(frame=0, dirty=True)
            print(
                f"[viewer] compare: {written} frames recorded; still serving",
                flush=True,
            )
        while True:
            if state["leave"]:
                print("[viewer] leaving compare mode", flush=True)
                break
            now = time.time()
            if exit_after > 0.0 and now - started > exit_after:
                print(f"[viewer] --exit-after-seconds {exit_after:g} elapsed, closing")
                break
            rate = max(1.0, float(fps_slider.value))
            if bool(playing.value) and now - head_time >= 1.0 / rate:
                head_time = now
                state["frame"] = (state["frame"] + 1) % length
                frame_slider.value = state["frame"]
                state["dirty"] = True
            focus = legend.take_focus()
            if focus is not None and 0 <= focus < len(slots):
                entry = slots[focus]
                pose = focus_pose(
                    entry["offset"],
                    pelvis_position(entry["data"], entry["pelvis"]),
                )
                moved = fly_clients_to(server, *pose)
                print(
                    f"[viewer] compare: flying {moved} client(s) to "
                    f"{names[focus]} at ({pose[0][0]:.2f}, {pose[0][1]:.2f}, "
                    f"{pose[0][2]:.2f})",
                    flush=True,
                )
            if state["camera"] != shown_camera:
                shown_camera = int(state["camera"])
                panels.rebind(
                    slots[shown_camera]["inner"],
                    slots[shown_camera]["model"],
                    slots[shown_camera]["data"],
                )
                print(
                    f"[viewer] camera panels follow {names[shown_camera]}", flush=True
                )
            if state["dirty"]:
                show(int(state["frame"]))
                state["dirty"] = False
            if bool(legend.cameras.value) != cameras_on:
                cameras_on = bool(legend.cameras.value)
                panels.enabled.value = cameras_on
                print(
                    f"[viewer] compare: camera panels {'on' if cameras_on else 'off'}",
                    flush=True,
                )
            if now - camera_at > 0.2:
                camera_at = now
                panels.update()
            time.sleep(0.004)
    finally:
        panels.close()
        server.stop()
        for entry in slots:
            entry["env"].close()
