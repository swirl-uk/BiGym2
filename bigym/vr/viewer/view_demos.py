#!/usr/bin/env python3
"""Demo viewer: the published demonstrations, demo batches and agent-run cells.

Kinematic playback: each frame poses the scene from the demo's stored
``full_qpos`` (no physics, no lower-body policy stepping). Two frontends:

--viewer viser (default)
    Browser 3D scene via mjviser (infinite grid ground,
    no skybox, IBL lighting), remote-friendly (ssh -L <port>). The "Demo
    directory" field at the top opens another directory (or, left empty, the
    published dataset) without restarting; it browses the folders of the
    machine running the viewer, ● marking those with demos. The "Figure
    capture" panel grabs the current view at an explicit pixel size (viser
    renders in the browser, so the window caps it) with the target spheres
    and velocity arrows hidden, and writes it to --figure-dir. Target spheres
    are drawn by the viewer itself (the in-model geoms are hidden from
    mjviser, whose meshes are baked once at build time and would go stale
    when an episode reset moves the target), with a glow when reached. The
    per-frame stored reward is shown in the Playback panel. "Camera panels"
    shows MuJoCo's own pixels for the current frame: the robot's onboard
    cameras through the environment's observation path, plus one
    third-person view.

--viewer mujoco
    Native MuJoCo window (needs a local display). Keyboard: SPACE play/pause,
    LEFT/RIGHT step frame, UP/DOWN switch episode, ``[`` / ``]`` switch batch
    or task (the scene is rebuilt the way the viser dropdown rebuilds it).
    Task visuals (e.g. the reach-target highlight) work natively because the
    renderer reads live geom colors. ``--batch`` / ``--task`` and
    ``--episode`` pick what opens first; with several batches and no
    ``--batch`` the list is printed once and the first one opens. The native
    frontend has no compare mode, but ``--demo-dir <root>`` still opens the
    first cell's evaluation in it.

What opens is decided by ``--demo-dir`` (or the "Demo directory" field):

nothing: the published dataset
    The demonstrations on the Hugging Face Hub (``bigym.loco.demos.hub``:
    ``BIGYM_DATASET_REPO`` / ``BIGYM_DATASET_REVISION`` apply), read from the
    Hub cache. The viser frontend offers the dataset's tasks in a "Task"
    dropdown and fetches a task's folder the first time it is picked;
    ``--task`` picks the one that opens first.

plain demo directories
    A collector batch, an exported dataset or any folder above them. Batches
    are discovered recursively and the viser frontend offers them in one
    "Demo batch" dropdown.

a local copy of the dataset
    A LeRobot task folder (``meta/info.json`` + ``data/*.parquet``) or a
    folder of them, such as ``bigym-download --local-dir PATH``. Opens like
    the published dataset, with the Task dropdown listing the folders found.
    A directory that holds collector batches opens as batches.

agent runs
    A cell (one task x one session of the coding-agent benchmark, i.e. a
    directory with ``policies/index.json``), a session root of cells, or a
    directory of such roots. ``--demo-dir`` takes several paths, so
    ``--demo-dir runs/s1 runs/s2 runs/s3`` is three sessions. The sidebar
    then shows Task and Session dropdowns, a two-level **Rollouts** picker
    (Evaluation / Development / Replays, then the item), the session card
    from ``run.json``, the Policy versions / Policy code / Transcript panels
    and a **Compare** panel that puts up to six policies of the same task
    side by side on one seed.

Dataset episodes are replayed on the task's official environment
(``make(task, camera_keys=())``), reset on the episode's seed from
``meta/episode_init_states.json``, with the task's ``_on_step`` hook run per
frame (:mod:`bigym.loco.demos.kinematic`).

``--follow`` rescans the cell every ``--follow-seconds`` so a session being
watched live grows new rollouts, versions and summary figures in place; the
scan runs on the main loop and never switches the rollout on screen unless
"auto-jump to newest" is ticked.

Threading note: env.reset() renders camera observations, and the EGL context
belongs to the main thread — episode loads, camera renders and subprocess
launches therefore ONLY happen on the main loop; UI/keyboard callbacks just
set a request flag (downloads and directory scans run on a worker thread
and then do the same).

Usage:
    # the published demonstrations, in the browser
    bigym-view
    bigym-view --task move_plate --episode 3

    # point at any folder; demo batches inside are discovered automatically
    MUJOCO_GL=egl bigym-view --demo-dir <demo-root> --port 8080

    # or a specific batch, in a native window
    bigym-view --demo-dir <demo-root>/<task>/<timestamp> --viewer mujoco

    # one agent cell, or every session under a directory of run roots
    MUJOCO_GL=egl bigym-view --demo-dir <runs>/<session>/<task>
    MUJOCO_GL=egl bigym-view --demo-dir <runs>

    # compare three sessions' policies on one hidden seed, from the shell
    MUJOCO_GL=egl bigym-view \
        --demo-dir <runs>/s1/move_plate <runs>/s2/move_plate \
        --compare "s1:v008,s2:v007,s3:last" --seed 620003

    # headless smoke: serve for 20 s, then exit
    MUJOCO_GL=egl bigym-view --demo-dir <cell> --exit-after-seconds 20
"""

from __future__ import annotations

import shlex
from pathlib import Path

import tyro

from bigym.cli import usage_error
from bigym.loco.agent.demo_video import TaskDemos
from bigym.loco.demos import hub
from bigym.loco.demos.kinematic import replay_env
from bigym.vr.viewer import video
from bigym.vr.viewer.agent.compare.model import (
    CompareRequest,
    parse_compare,
    resolve_compare,
)
from bigym.vr.viewer.agent.compare.run import run_compare
from bigym.vr.viewer.agent.runs import (
    cell_of,
    default_rollout,
    describe_opening,
    rollouts_of,
)
from bigym.vr.viewer.episodes import episode_files
from bigym.vr.viewer.layout import grid_note, slot_grid
from bigym.vr.viewer.mjviser_compat import tint_robot_geoms
from bigym.vr.viewer.playback import DEMO_HZ, ViserViewer, run_mujoco
from bigym.vr.viewer.sources import (
    BatchSource,
    DatasetSource,
    local_source,
    pick_batch,
    pick_task,
    print_discovery,
    published_source,
    rescan_all,
)
from bigym.vr.viewer.stores import DatasetStore, DemoStore, build_env, load_metadata
from bigym.vr.viewer.view_config import ViewConfig
from bigym.vr.viewer.widgets import LIGHTING_PREFS, CameraPanels, Picker, task_picker


def _parse_args() -> ViewConfig:
    try:
        return tyro.cli(ViewConfig, prog="bigym-view", description=__doc__)
    except ValueError as exc:
        usage_error(str(exc), prog="bigym-view")


def _open_batch(
    args: ViewConfig,
    demo_dir: Path,
    batch_labels: list[str],
    batch_idx: int,
    directory: str,
    cell: Path | None = None,
    batch_paths: list[Path] | None = None,
    rescan=None,
    cells: list | None = None,
):
    """Build env + viewer for one batch.

    Returns another batch index to switch to (the Rollouts dropdowns, the
    "Demo batch" one or the native frontend's ``[`` / ``]``), a
    :class:`~bigym.vr.viewer.agent.compare.model.CompareRequest` when the Compare button
    was pressed, a source when the "Demo directory" field loaded another
    directory, or None when the viewer exits.
    """
    files = episode_files(demo_dir)
    if not files:
        raise FileNotFoundError(f"No .npz demos in {demo_dir}")
    metadata = load_metadata(demo_dir)
    print(f"[viewer] {len(files)} episodes | task={metadata['task']['task_name']}")

    env = build_env(metadata)
    env.reset(seed=0)
    inner = env.inner_env
    model = inner.model
    data = inner.data

    _tint(args, model)

    # Prefer the collection-time value stamped in metadata: the env's current
    # step duration can differ from what a demo was recorded under.
    meta_step_seconds = metadata.get("control_step_seconds")
    if meta_step_seconds:
        realtime_fps = 1.0 / float(meta_step_seconds)
    else:
        realtime_fps = 1.0 / float(inner.control_step_seconds)
    if abs(realtime_fps - DEMO_HZ) > 1e-6:
        print(
            f"[viewer] recorded frames span {1000.0 / realtime_fps:.0f} ms of "
            f"sim time each -> realtime playback is {realtime_fps:.0f} fps, "
            f"not the nominal {DEMO_HZ:.0f}"
        )

    store = DemoStore(env, files)
    store.load(args.episode)

    task = metadata["task"]
    try:
        if args.viewer == "viser":
            return ViserViewer(
                store,
                inner,
                Picker(batch_labels, batch_idx, paths=batch_paths, rescan=rescan),
                args,
                directory,
                realtime_fps=realtime_fps,
                task_name=task["task_name"],
                camera_keys=tuple(task.get("camera_keys") or ("head",)),
                camera_shape=tuple(task.get("camera_shape") or (84, 84)),
                cell=cell,
                cells=cells,
            ).run()
        try:
            return run_mujoco(
                store,
                inner,
                model,
                data,
                realtime_fps,
                batch_labels=batch_labels,
                batch_idx=batch_idx,
            )
        except Exception as exc:
            raise SystemExit(
                f"native viewer failed ({exc}); over SSH use --viewer viser"
            ) from exc
    finally:
        env.close()


def _opening_index(
    args: ViewConfig, cells: list, batches: list[Path], labels: list[str]
) -> int:
    """Which batch opens first: --batch, else the default rollout of a cell."""
    if args.batch is not None:
        return pick_batch(labels, args.batch)
    if cells:
        rollout, _ = default_rollout(rollouts_of(cells[0].path, batches))
        if rollout is not None and rollout.batch in batches:
            return batches.index(rollout.batch)
    return 0


def _run_compare(args: ViewConfig, request, back_to: int) -> int:
    """Produce any missing rollout, then play the slots side by side.

    Args:
        args: The parsed command line.
        request: The resolved compare request.
        back_to: The batch index to return to afterwards.

    Returns:
        ``back_to`` — leaving compare mode returns to the single-rollout
        scene that asked for it, whether the scene was built or the user
        left while the missing rollouts were still running.
    """
    if request.error:
        print(f"[viewer] compare: {request.error}", flush=True)
        return back_to
    if not request.slots:
        print("[viewer] compare: no slots to compare", flush=True)
        return back_to
    print(
        f"[viewer] compare: layout {request.layout} — "
        f"{grid_note(len(request.slots), request.gap, request.layout)}",
        flush=True,
    )
    for i, slot in enumerate(request.slots):
        place = slot_grid(len(request.slots), request.layout)[i]
        print(
            f"[viewer] compare slot {i}: {slot.session} {slot.version_name} "
            f"seed {request.seed} at row {place[0]} column {place[1]} "
            f"<- {slot.source or 'not rolled out yet'}",
            flush=True,
        )
    if request.missing:
        # The rollouts are produced by run_compare itself, inside the viser
        # server it opens first: one progress bar per job, a working Leave
        # button, and the scene built when the last episode lands.
        print(
            f"[viewer] compare: {len(request.missing)} slot(s) have no episode "
            f"on seed {request.seed}; running bigym-agent replay --no-video",
            flush=True,
        )
    else:
        print(
            f"[viewer] compare: every slot already stores seed {request.seed} "
            "(evaluation batches hold every hidden seed) — no replay needed",
            flush=True,
        )
    run_compare(
        request,
        port=args.port,
        build_env=build_env,
        load_metadata=load_metadata,
        camera_panels=CameraPanels,
        lighting_prefs=LIGHTING_PREFS,
        exit_after=args.exit_after_seconds,
        record=record_request(args),
    )
    return back_to


def record_request(args: ViewConfig) -> dict | None:
    """What ``--record`` asks :func:`run_compare` to write."""
    if not args.record:
        return None
    return {
        "path": Path(args.record),
        "seconds": args.record_seconds,
        "size": video.parse_record_size(args.record_size),
        "fps": args.record_fps,
        "browser": args.browser,
        "stay": args.record_then_stay,
        "view": args.record_view,
        "zoom": args.record_zoom,
        "speed": args.record_speed,
        "label_scale": args.record_label_scale,
    }


def _cli_compare(args: ViewConfig, cells: list, batches: list[Path], idx: int):
    """Build the compare request ``--compare`` asked for, or None."""
    if not args.compare:
        return None
    if args.viewer != "viser":
        raise SystemExit("--compare needs --viewer viser")
    if not cells:
        raise SystemExit("--compare needs agent cells under --demo-dir")
    here = cell_of(cells, batches[idx]) or cells[0]
    try:
        specs = parse_compare(args.compare)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    request = resolve_compare(cells, here.task, specs, args.seed)
    request.gap = args.compare_gap
    request.layout = args.layout
    request.back_to = idx
    return request


def _tint(args: ViewConfig, model) -> None:
    """Display-only robot tint, applied before mjviser converts the meshes."""
    if args.robot_tint == "dark" and args.viewer == "viser":
        n = tint_robot_geoms(model)
        print(f"[viewer] tinted {n} robot geoms (display only)")


def _open_dataset_task(args: ViewConfig, source: DatasetSource, idx: int):
    """Build the replay env + viewer for one task of a dataset.

    Returns another task index (the Task dropdown or ``[`` / ``]``), a
    source when the "Demo directory" field loaded another directory, or None
    when the viewer exits.
    """
    task, folder, label = source.tasks[idx], source.dirs[idx], source.labels[idx]
    if folder is None:
        print(f"[viewer] {task}: fetching from {hub.dataset_repo()}", flush=True)
    demos = TaskDemos(task, folder)
    episodes = demos.episodes
    print(f"[viewer] {label}: {len(episodes)} episodes | task={task} | {demos.dir}")
    step = demos.metadata.get("control_step_seconds")
    realtime_fps = 1.0 / float(step) if step else DEMO_HZ

    env = replay_env(task)
    inner = env.inner_env
    model = inner.model
    data = inner.data
    _tint(args, model)
    store = DatasetStore(env, inner, demos)
    store.load(args.episode)
    try:
        if args.viewer == "viser":
            picker = Picker(
                source.labels, idx, item="task", build=task_picker(source, idx)
            )
            directory = shlex.join(str(r) for r in source.roots)
            return ViserViewer(
                store,
                inner,
                picker,
                args,
                directory,
                realtime_fps=realtime_fps,
                task_name=task,
            ).run()
        try:
            return run_mujoco(
                store,
                inner,
                model,
                data,
                realtime_fps,
                batch_labels=source.labels,
                batch_idx=idx,
                item="task",
            )
        except Exception as exc:
            raise SystemExit(
                f"native viewer failed ({exc}); over SSH use --viewer viser"
            ) from exc
    finally:
        env.close()


def _run_dataset(args: ViewConfig, source: DatasetSource):
    """Play a dataset's tasks until the viewer exits or opens another source."""
    if args.batch is not None or args.compare:
        raise SystemExit(
            "--batch and --compare apply to demo batches and agent runs; "
            f"{source.where} is a dataset (use --task)"
        )
    idx = pick_task(source, args.task)
    print(
        f"[viewer] {len(source.labels)} tasks in {source.where}; opening "
        f"{source.labels[idx]}"
    )
    while True:
        result = _open_dataset_task(args, source, idx)
        if not isinstance(result, int):
            return result
        idx = result


def _run_batches(args: ViewConfig, source: BatchSource):
    """Open the demo batches, or the agent rollouts, of a batch source.

    Returns the source the "Demo directory" field loaded, or None when the
    viewer exits.
    """
    if args.task is not None:
        raise SystemExit(
            f"--task picks a dataset task; {source.where} holds demo batches "
            "(use --batch)"
        )
    roots, label_root = source.roots, source.label_root
    batches, labels, cells = source.batches, source.labels, source.cells
    print_discovery(roots, cells)

    def rescan() -> list[str]:
        """Re-discover batches in place (a finished job wrote a new one)."""
        if rescan_all(roots, label_root, batches, labels):
            print(f"[viewer] {len(batches)} batches under {label_root}")
        return labels

    idx = _opening_index(args, cells, batches, labels)
    here = cell_of(cells, batches[idx])
    if here is not None and args.viewer != "viser":
        # The viser frontend prints this from the panels it builds; the
        # native one has no panels, so say it here.
        rollouts = rollouts_of(here.path, batches)
        opened, reason = default_rollout(rollouts)
        for line in describe_opening(here, rollouts, opened, reason):
            print(line, flush=True)
    if not cells and len(batches) > 1 and args.batch is None:
        # No picker: list the batches once, open the first, and let the
        # dropdown ("[" / "]" natively) move between them.
        print(f"[viewer] {len(batches)} demo batches under {label_root}:")
        for i, label in enumerate(labels):
            print(f"  [{i:2d}] {label}")
        print(
            f"[viewer] opening [{idx}] {labels[idx]} "
            "(--batch <index-or-substring> picks another)"
        )
    compare = _cli_compare(args, cells, batches, idx)

    while True:
        if compare is not None:
            request, compare = compare, None
            idx = _run_compare(args, request, request.back_to)
            if args.exit_after_seconds > 0.0:
                return None  # headless smoke: one compare pass and out
            if args.record and not args.record_then_stay:
                return None  # the file is written; that is what was asked for
            continue
        here = cell_of(cells, batches[idx])
        result = _open_batch(
            args,
            batches[idx],
            labels,
            idx,
            source.where,
            cell=here.path if here is not None else None,
            batch_paths=batches,
            rescan=rescan,
            cells=cells,
        )
        if result is None or isinstance(result, (DatasetSource, BatchSource)):
            return result
        if isinstance(result, CompareRequest):
            result.back_to = idx
            compare = result
            continue
        idx = result


def main() -> None:
    """Open the published demonstrations, or what --demo-dir points at."""
    args = _parse_args()

    if args.demo_dir:
        source = local_source([Path(raw).expanduser() for raw in args.demo_dir])
    else:
        try:
            source = published_source()
        except Exception as exc:  # offline: --task still works
            if args.task is None:
                raise SystemExit(
                    f"cannot list the published tasks ({exc}); pass --task NAME "
                    "to open a cached task, or --demo-dir PATH"
                ) from exc
            source = DatasetSource(labels=[args.task], tasks=[args.task], dirs=[None])
    while source is not None:
        if isinstance(source, DatasetSource):
            source = _run_dataset(args, source)
        else:
            source = _run_batches(args, source)
        # What to open first applies to the command line's source only.
        args.batch = args.task = args.compare = None
        args.episode = 0


if __name__ == "__main__":
    main()
