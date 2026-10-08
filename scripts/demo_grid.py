#!/usr/bin/env python3
"""One human demonstration of every task, side by side.

One viser server, one environment per published task, every scene playing a
real teleoperated demonstration of that task at the same time and looping
for ever. It exists to be looked at: a screenshot of the whole benchmark in
one frame, or a screen recording for a talk or a project page.

Where the data comes from: the public dataset (``bigym.loco.demos.hub``),
read the way :class:`bigym.loco.agent.demo_video.TaskDemos` reads it — the
``full_qpos`` column of one episode and its seed out of
``meta/episode_init_states.json``, with no camera PNG ever decoded. Each
scene is an official ``make(task)`` env with **no cameras at all**
(``camera_keys=()``), reset on that episode's seed so the props and the
reach targets sit where the demonstration found them, and then posed frame
by frame from ``full_qpos`` with ``mj_forward`` — the same kinematic replay
as ``bigym-view`` and the re-renderer (:mod:`bigym.loco.demos.kinematic`),
task ``_on_step`` hook included, so live scene state (the reach-target
highlight) is right.

Twenty scenes take ~90 s to build, ~7.5 GB of RSS and run at 17-21 grid
updates a second; the cost is almost entirely viser's per-scene transform
pushes, not ``mj_forward``. The playhead therefore follows the *wall
clock* rather than the update loop: demo time runs at ``--fps`` (50 = real
time) and frames that could not be drawn are skipped, so the
demonstrations always play at their own speed however big the grid is.

The scene machinery is compare mode's (``bigym/vr/viewer/agent_cells.py``): one
mjviser scene per task under its own node-name prefix, moved into the grid
by a frame at ``/bodies/<prefix>`` (a column steps +y, a row steps -x), all
of them built hidden and revealed together once every scene holds frame 0
of its demo. Nothing in this module writes to ``agent_cells``; it imports.

For a video the live rate is not good enough, so ``--record`` captures the
same scene offline: the playhead is stepped one frame at a time with no
wall clock, and each frame is rendered by a browser (viser has no
server-side renderer — ``ClientHandle.get_render`` asks a client's canvas)
and piped into ffmpeg. The browser is a headless Chrome the tool starts
itself unless one is already connected. The file is smooth at
``--record-fps`` however long each frame took, which on a software
rasteriser is about 10 s for a 1920x1080 frame of twenty scenes.

Usage (an internal tool for the all-tasks video, run from a checkout)::

    G="python scripts/demo_grid.py"
    $G                                     # every published task, 5 columns
    $G --tasks move_plate,pick_box --columns 2
    $G --camera 3 --tint                   # fly to one scene, tinted robots
    MUJOCO_GL=egl $G --exit-after-seconds 60   # headless smoke
    MUJOCO_GL=egl $G --record grid.mp4 --record-frames 250
    MUJOCO_GL=egl $G --screenshot grid.png --frame 400
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol, cast

import mujoco
import numpy as np
import tyro
from mjviser import ViserMujocoScene

from bigym.loco.demos.kinematic import replay_env, run_task_hook
from bigym.vr.viewer import agent_cells, video
from bigym.vr.viewer.mjviser_compat import (
    hide_reach_targets,
    patch_mjviser,
    patch_mjviser_track_props,
)

# The grid a 20-task dataset wants: 5 columns x 4 rows reads as a poster and
# fits a 16:9 frame without a telephoto camera.
DEFAULT_COLUMNS = 5
# Metres between neighbouring columns and rows. The widest task fixture
# measured across the 20 published tasks is 2.73 m (pick_box, along y;
# drawer/wall-cupboard kitchens are 2.64 m and stack_blocks is 2.31 m along
# x), and a demonstrator walks up to a metre out of that box, so 4.0 m
# leaves better than a metre of floor between neighbours in every task pair
# while keeping the whole 5 x 4 grid inside one camera frame.
DEFAULT_GAP = 4.0
# 50 Hz is the control rate the demonstrations were recorded at, so demo
# time at 50 fps is real time (the dataset metadata's
# ``control_step_seconds`` is read per task and reported at startup).
DEFAULT_FPS = 50.0
DEFAULT_PORT = 8080
DEFAULT_EPISODE = "median"
# ``all``: one playhead for the whole grid, each scene holding its last
# frame until the longest demo ends and everything restarts together.
# ``each``: every scene wraps on its own length.
LOOP_ALL = "all"
LOOP_EACH = "each"
LOOP_MODES = (LOOP_ALL, LOOP_EACH)
SKY_OPTIONS = ("black", "grey", "transparent")
CAMERA_OVERVIEW = "overview"
FOCUS_OVERVIEW = "(overview)"
# How long the "Playback" folder waits between measured-rate log lines.
RATE_LOG_SECONDS = 5.0
RECORD_LOG_EVERY = 50
# How long a capture waits for the page to finish loading its meshes,
# plates and environment map, and what counts as "the picture stopped
# changing" (mean absolute channel difference between two renders, 0-255).
PAGE_SETTLE_SECONDS = 120.0
PAGE_SETTLE_TOLERANCE = 0.4
# Where one scene's content sits relative to its offset, and how much of
# the floor around it belongs to it: the published tasks reach ~2.3 m along
# +x and ~2 m along -y, and the demonstrator walks inside that.
SCENE_CENTRE = (0.9, -0.3, 1.0)
SCENE_PAD = 1.8
SCENE_HEIGHT = 1.4
# Metres of floor left between two scenes' measured footprints. The grid is
# packed on what each scene really occupies, so this is the only spacing
# knob that matters; --gap overrides it with one uniform pitch.
DEFAULT_CLEARANCE = 0.6
# A geom bigger than this is scenery, not a scene: skies, ground planes and
# the odd room-sized visual box would swallow every footprint.
FOOTPRINT_MAX_GEOM = 5.0
# What the overview lens may open to, in radians (viser's own default is
# about 80 degrees, which is far too wide for a grid this size).
FOV_MIN = 0.18
FOV_MAX = 1.40
# How much of the frame the overview shot gives the grid, and how far it
# looks down: a shallower pitch hides the back rows behind the front ones.
OVERVIEW_FILL = 0.80
OVERVIEW_TILT = 0.52
# The flight: how long it runs, how high the camera walks, how far ahead
# it looks down an aisle, and the two lenses it uses.
DEFAULT_DURATION = 30.0
EYE_HEIGHT = 1.4
AISLE_LEAD = 3.0
CLOSE_FOV = 0.90
AISLE_FOV = 1.05
PATH_FLY = "fly"
PATH_NONE = "none"
PATH_PRESETS = (PATH_NONE, PATH_FLY)
# --path-preview: small and slow, just to see the motion.
PREVIEW_SIZE = "480x270"
PREVIEW_FPS = 5.0
# --export-states / --states: the few MB of per-frame state a machine needs
# to draw the grid without the dataset.
STATES_FORMAT = "bigym-demo-grid-states-v1"
# The label plate hangs higher and wider than a compare slot's: it is read
# from the overview camera, twenty metres out, not from two.
PLATE_WIDTH = 1.5
PLATE_Z = 1.35
# Twenty categorical colours: the plate stripe always, and ``--tint`` turns
# the same colour into a per-scene multiplier on the robot's geoms.
GRID_COLOURS = (
    "#73aeff",
    "#ff9e52",
    "#7ae085",
    "#e68ef2",
    "#ffd95a",
    "#66d9eb",
    "#ff8f8f",
    "#9fe06a",
    "#b9a0ff",
    "#ffc17a",
    "#5ec8a8",
    "#f27ab0",
    "#8fd0ff",
    "#d7e06a",
    "#c08fff",
    "#6fe0c8",
    "#ffab5e",
    "#88b4ff",
    "#e0a06a",
    "#a8e0ff",
)


# ---------------------------------------------------------------------------
# Naming, layout and scheduling (no MuJoCo, no viser: all of it is testable)
# ---------------------------------------------------------------------------


def label_text(task: str) -> str:
    """The task name as a label reads it: ``move_plate`` -> ``Move plate``.

    Args:
        task: The task name.

    Returns:
        The name with underscores turned into spaces and the first letter
        capitalised; the rest is left alone so ``reach_target_multi_modal``
        keeps its words.
    """
    words = str(task).replace("_", " ").strip()
    return words[:1].upper() + words[1:] if words else ""


def scene_colour(index: int) -> str:
    """The CSS colour of scene ``index`` (the palette repeats)."""
    return GRID_COLOURS[int(index) % len(GRID_COLOURS)]


def scene_tint(index: int) -> tuple[float, float, float]:
    """The per-channel multiplier ``--tint`` gives scene ``index``.

    The palette colour, scaled so its strongest channel is 1.0: a tint that
    only ever takes light away turns every robot grey.

    Args:
        index: The scene's position in the grid.

    Returns:
        An ``(r, g, b)`` multiplier for :func:`agent_cells.tint_slot_robot`.
    """
    rgb = np.asarray(agent_cells.css_rgb(scene_colour(index)), dtype=float)
    peak = float(rgb.max()) or 1.0
    return cast(tuple[float, float, float], tuple(float(v) for v in rgb / peak))


@dataclass(frozen=True)
class Footprint:
    """How much floor one scene takes, in its own coordinates.

    The box of everything the scene draws — fixtures, props and the robot
    where it stands at frame 0 — measured after ``reset`` so a kitchen that
    reaches 2.3 m along +x and 2 m along -y is described as it really is
    rather than by its origin.

    Attributes:
        lo: ``(x, y)`` of the near-left corner.
        hi: ``(x, y)`` of the far-right corner.
    """

    lo: tuple[float, float]
    hi: tuple[float, float]

    @property
    def depth(self) -> float:
        """How far the scene reaches along x."""
        return max(0.0, float(self.hi[0]) - float(self.lo[0]))

    @property
    def width(self) -> float:
        """How far the scene reaches along y."""
        return max(0.0, float(self.hi[1]) - float(self.lo[1]))

    @property
    def centre(self) -> tuple[float, float]:
        """The middle of the box, which is what a cell is centred on."""
        return (
            0.5 * (float(self.lo[0]) + float(self.hi[0])),
            0.5 * (float(self.lo[1]) + float(self.hi[1])),
        )


@dataclass(frozen=True)
class Layout:
    """Where every scene stands, and what the grid ends up measuring.

    Attributes:
        offsets: One ``(x, y, z)`` per scene, in scene order.
        places: One ``(row, column)`` per scene.
        footprints: Each scene's own box (see :class:`Footprint`).
        rows: ``(x_lo, x_hi)`` of each row's occupied depth, in world
            metres, near row last (rows recede along -x).
        columns: ``(y_lo, y_hi)`` of each column's occupied width.
        clearance: Metres of floor left between neighbouring footprints.
    """

    offsets: tuple[tuple[float, float, float], ...]
    places: tuple[tuple[int, int], ...]
    footprints: tuple[Footprint, ...]
    rows: tuple[tuple[float, float], ...]
    columns: tuple[tuple[float, float], ...]
    clearance: float

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """``((x_lo, y_lo), (x_hi, y_hi))`` of the whole grid."""
        if not self.rows or not self.columns:
            return ((0.0, 0.0), (0.0, 0.0))
        return (
            (self.rows[-1][0], self.columns[0][0]),
            (self.rows[0][1], self.columns[-1][1]),
        )

    @property
    def centre(self) -> tuple[float, float, float]:
        """The middle of the grid, a metre above the floor."""
        (x_lo, y_lo), (x_hi, y_hi) = self.bounds
        return (0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi), SCENE_HEIGHT * 0.5)

    @property
    def extent(self) -> tuple[float, float]:
        """``(depth, width)`` of the grid in metres."""
        (x_lo, y_lo), (x_hi, y_hi) = self.bounds
        return (x_hi - x_lo, y_hi - y_lo)

    def half(self) -> tuple[float, float, float]:
        """Half extents of the grid's box, height included."""
        depth, width = self.extent
        return (0.5 * depth, 0.5 * width, SCENE_HEIGHT)

    def aisle(self, row: int) -> float:
        """The x of the aisle between ``row`` and the row behind it.

        Midway between the two rows' real footprint edges, so the camera
        flies down the gap rather than through a worktop.

        Args:
            row: The nearer row of the pair (0 is the far row).

        Returns:
            The x of the aisle's centre line; the front of row 0 when
            there is no row behind it.
        """
        if not self.rows:
            return 0.0
        first = self.rows[max(0, min(int(row), len(self.rows) - 1))]
        if int(row) + 1 >= len(self.rows):
            return first[0] - max(0.5, 0.5 * self.clearance)
        behind = self.rows[int(row) + 1]
        return 0.5 * (first[0] + behind[1])

    def describe(self) -> list[str]:
        """The startup lines: pitch per row and column, and the extent."""
        depth, width = self.extent
        (x_lo, y_lo), (x_hi, y_hi) = self.bounds
        lines = [
            f"[grid] layout: {len(self.offsets)} scenes, "
            f"{len(self.rows)} rows x {len(self.columns)} columns, "
            f"clearance {self.clearance:g} m",
            f"[grid] extent: {depth:.1f} m deep x {width:.1f} m wide "
            f"(x {x_lo:.1f}..{x_hi:.1f}, y {y_lo:.1f}..{y_hi:.1f})",
        ]
        widths = ", ".join(f"{hi - lo:.1f}" for lo, hi in self.columns)
        depths = ", ".join(f"{hi - lo:.1f}" for lo, hi in self.rows)
        lines.append(f"[grid] column widths (m): {widths}")
        lines.append(f"[grid] row depths (m): {depths}")
        if len(self.columns) > 1:
            centres = [0.5 * (lo + hi) for lo, hi in self.columns]
            steps = ", ".join(
                f"{centres[i + 1] - centres[i]:.1f}" for i in range(len(centres) - 1)
            )
            lines.append(f"[grid] column pitch (m): {steps}")
        if len(self.rows) > 1:
            centres = [0.5 * (lo + hi) for lo, hi in self.rows]
            steps = ", ".join(
                f"{centres[i] - centres[i + 1]:.1f}" for i in range(len(centres) - 1)
            )
            lines.append(f"[grid] row pitch (m): {steps}")
        return lines


def default_footprint() -> Footprint:
    """The box a scene is assumed to fill when it cannot be measured."""
    return Footprint(
        lo=(SCENE_CENTRE[0] - SCENE_PAD, SCENE_CENTRE[1] - SCENE_PAD),
        hi=(SCENE_CENTRE[0] + SCENE_PAD, SCENE_CENTRE[1] + SCENE_PAD),
    )


def pack_layout(
    footprints, columns: int, clearance: float = DEFAULT_CLEARANCE
) -> Layout:
    """Place scenes so their footprints clear each other by ``clearance``.

    A column is as wide as its widest scene and a row as deep as its
    deepest, and each scene is centred in its cell on its footprint's
    middle rather than on its origin — a kitchen that grows along +x and
    -y then lines up with a bare reach-target scene instead of sitting a
    metre off.

    Args:
        footprints: One :class:`Footprint` per scene, in scene order.
        columns: Scenes per row.
        clearance: Metres of floor between neighbouring footprints.

    Returns:
        The :class:`Layout`.
    """
    boxes = list(footprints)
    places = agent_cells.slot_grid(len(boxes), columns_layout(columns))
    if not boxes:
        return Layout((), (), (), (), (), float(clearance))
    gap = max(0.0, float(clearance))
    width = max(1, max(c for _, c in places) + 1)
    rows = max(1, max(r for r, _ in places) + 1)
    # A column is as wide as its widest scene, a row as deep as its deepest.
    column_width = [0.0] * width
    row_depth = [0.0] * rows
    for box, (row, column) in zip(boxes, places, strict=True):
        column_width[column] = max(column_width[column], box.width)
        row_depth[row] = max(row_depth[row], box.depth)
    # Columns step along +y from zero; rows step back along -x.
    column_centre, cursor = [], 0.0
    for index, span in enumerate(column_width):
        cursor = (
            span / 2.0
            if index == 0
            else cursor + column_width[index - 1] / 2.0 + gap + span / 2.0
        )
        column_centre.append(cursor)
    row_centre, cursor = [], 0.0
    for index, span in enumerate(row_depth):
        cursor = (
            -span / 2.0
            if index == 0
            else cursor - row_depth[index - 1] / 2.0 - gap - span / 2.0
        )
        row_centre.append(cursor)
    offsets = []
    for box, (row, column) in zip(boxes, places, strict=True):
        middle = box.centre
        offsets.append(
            (
                float(row_centre[row] - middle[0]),
                float(column_centre[column] - middle[1]),
                0.0,
            )
        )
    return Layout(
        offsets=tuple(offsets),
        places=tuple(places),
        footprints=tuple(boxes),
        rows=tuple(
            (row_centre[r] - row_depth[r] / 2.0, row_centre[r] + row_depth[r] / 2.0)
            for r in range(rows)
        ),
        columns=tuple(
            (
                column_centre[c] - column_width[c] / 2.0,
                column_centre[c] + column_width[c] / 2.0,
            )
            for c in range(width)
        ),
        clearance=gap,
    )


def uniform_layout(footprints, columns: int, gap: float) -> Layout:
    """The old placement: every scene at its origin, ``gap`` metres apart.

    ``--gap`` asks for this — one pitch for every row and column, scenes
    positioned by their origin — for a grid that has to match an older
    capture.

    Args:
        footprints: One :class:`Footprint` per scene (used for the extent
            the camera frames, not for the placement).
        columns: Scenes per row.
        gap: Metres between neighbouring origins.

    Returns:
        The :class:`Layout`.
    """
    boxes = list(footprints)
    places = agent_cells.slot_grid(len(boxes), columns_layout(columns))
    if not boxes:
        return Layout((), (), (), (), (), float(gap))
    offsets = [agent_cells.slot_offset(r, c, float(gap)) for r, c in places]
    width = max(1, max(c for _, c in places) + 1)
    rows = max(1, max(r for r, _ in places) + 1)
    row_span = [[float("inf"), float("-inf")] for _ in range(rows)]
    column_span = [[float("inf"), float("-inf")] for _ in range(width)]
    for box, offset, (row, column) in zip(boxes, offsets, places, strict=True):
        row_span[row][0] = min(row_span[row][0], box.lo[0] + offset[0])
        row_span[row][1] = max(row_span[row][1], box.hi[0] + offset[0])
        column_span[column][0] = min(column_span[column][0], box.lo[1] + offset[1])
        column_span[column][1] = max(column_span[column][1], box.hi[1] + offset[1])
    return Layout(
        offsets=tuple(offsets),
        places=tuple(places),
        footprints=tuple(boxes),
        rows=tuple((lo, hi) for lo, hi in row_span),
        columns=tuple((lo, hi) for lo, hi in column_span),
        clearance=float(gap),
    )


def columns_layout(columns: int) -> str:
    """Compare mode's layout name for ``columns`` scenes per row."""
    return f"{max(1, int(columns))} columns"


def loop_frame(playhead: int, length: int, longest: int, mode: str = LOOP_ALL) -> int:
    """Which frame of one demo a playhead position shows.

    Args:
        playhead: The global tick count (it only ever grows).
        length: Frames in this scene's demonstration.
        longest: Frames in the longest demonstration of the grid.
        mode: ``all`` — one playhead for the grid: a scene that has run out
            holds its last frame until the longest demo ends and every
            scene restarts together. ``each`` — every scene wraps on its
            own length, so the grid drifts apart and never resynchronises.

    Returns:
        A frame index inside ``[0, length)``.
    """
    length = max(1, int(length))
    longest = max(1, int(longest))
    if str(mode) == LOOP_EACH:
        return int(playhead) % length
    return min(int(playhead) % longest, length - 1)


def loop_length(lengths, mode: str = LOOP_ALL) -> int:
    """How many ticks one pass over the grid takes.

    Args:
        lengths: Frames in each scene's demonstration.
        mode: ``all`` or ``each``.

    Returns:
        The longest demo for ``all``; for ``each`` the same number, which is
        only the slider's range — each scene wraps on its own length.
    """
    values = [max(1, int(v)) for v in lengths]
    return max(values) if values else 1


def task_names(spec: str, available: Callable[[], tuple[str, ...]]) -> list[str]:
    """The tasks to show: ``--tasks`` if given, else every published task.

    Args:
        spec: The ``--tasks`` value (a comma-separated list, or empty).
        available: Called only when ``spec`` is empty — normally
            :func:`bigym.loco.demos.hub.available_tasks`, which asks the
            Hub which tasks the dataset publishes.

    Returns:
        The task names, in the order they were asked for, duplicates
        dropped.

    Raises:
        SystemExit: ``spec`` holds nothing but separators, or the dataset
            publishes no tasks at all.
    """
    if str(spec or "").strip():
        wanted = [name.strip() for name in str(spec).split(",") if name.strip()]
        if not wanted:
            raise SystemExit(f"--tasks {spec!r} names no task")
    else:
        wanted = list(available())
        if not wanted:
            raise SystemExit("the dataset publishes no tasks yet")
    seen: list[str] = []
    for name in wanted:
        if name not in seen:
            seen.append(name)
    return seen


def camera_choice(text: str, count: int) -> int:
    """Parse ``--camera``: -1 for the overview, else a scene index.

    Args:
        text: The ``--camera`` value.
        count: How many scenes there are.

    Returns:
        ``-1`` for ``overview``, else the scene index.

    Raises:
        SystemExit: The value is neither ``overview`` nor an index of a
            scene that exists.
    """
    name = str(text or CAMERA_OVERVIEW).strip().lower()
    if name == CAMERA_OVERVIEW:
        return -1
    try:
        index = int(name)
    except ValueError:
        raise SystemExit(
            f"--camera {text!r} is neither {CAMERA_OVERVIEW!r} nor a scene index"
        ) from None
    if not 0 <= index < max(1, int(count)):
        raise SystemExit(f"--camera {index} is outside 0..{max(0, int(count) - 1)}")
    return index


# ---------------------------------------------------------------------------
# The demonstrations (dataset reads only: no image is ever decoded)
# ---------------------------------------------------------------------------


class GridDemo:
    """One task's chosen demonstration: the states, the seed, its name.

    Attributes:
        task: The task name.
        qpos: ``(T, nq)`` of stored simulator states.
        seed: The seed the demonstration was collected on.
        index: Its episode index in the dataset export.
        source_file: The recording it came from.
        count: How many demonstrations the task has (0 when unknown).
        control_seconds: Seconds per recorded frame, or None.
    """

    def __init__(
        self,
        task: str,
        qpos,
        seed: int,
        index: int = -1,
        source_file: str = "",
        count: int = 0,
        control_seconds: float | None = None,
    ):
        """Hold one demonstration's states and where they came from."""
        self.task = str(task)
        self.qpos = np.asarray(qpos, dtype=np.float64)
        self.seed = int(seed)
        self.index = int(index)
        self.source_file = str(source_file)
        self.count = int(count)
        self.control_seconds = control_seconds

    @classmethod
    def from_hub(cls, task: str, which: str = DEFAULT_EPISODE) -> "GridDemo":
        """Read one task's demonstration from the dataset export.

        Args:
            task: The task name (its folder is fetched from the Hub on
                first use).
            which: ``median`` for the demonstration of median length, or an
                episode index.

        Returns:
            The demonstration.

        Raises:
            SystemExit: The task has no such episode.
        """
        from bigym.loco.agent.demo_video import TaskDemos, pick_episodes

        demos = TaskDemos(str(task))
        try:
            episode = pick_episodes(demos.episodes, str(which), 1)[0]
        except ValueError as exc:
            raise SystemExit(f"{task}: {exc}") from None
        step = demos.metadata.get("control_step_seconds")
        return cls(
            task=task,
            qpos=demos.qpos(episode),
            seed=int(episode.seed),
            index=int(episode.index),
            source_file=str(episode.source_file),
            count=len(demos.episodes),
            control_seconds=float(step) if step else None,
        )

    @property
    def length(self) -> int:
        """How many frames the demonstration holds."""
        return int(self.qpos.shape[0])

    @property
    def label(self) -> str:
        """What the plate above this scene's robot says."""
        return label_text(self.task)

    def describe(self, which: str = DEFAULT_EPISODE) -> str:
        """The startup line for this task: episode, seed and length."""
        if self.count:
            chosen = (
                f"median of {self.count}"
                if str(which) == DEFAULT_EPISODE
                else "requested"
            )
            chosen = f" ({chosen})"
        else:
            chosen = ""
        return (
            f"[grid] {self.task}: episode {self.index}{chosen} · "
            f"seed {self.seed} · {self.length} frames · {self.source_file}"
        )


def read_demos(tasks, which: str = DEFAULT_EPISODE) -> list[GridDemo]:
    """Read one demonstration per task from the Hub, printing a line each."""
    demos = []
    for task in tasks:
        demo = GridDemo.from_hub(task, which)
        print(demo.describe(which), flush=True)
        demos.append(demo)
    return demos


def export_states(path, demos, which: str = DEFAULT_EPISODE) -> Path:
    """Write the chosen demonstrations to one compressed npz.

    What a machine needs to draw the grid is the per-frame ``full_qpos``
    and the seed of each task's chosen demonstration — a few MB — not the
    28 GB dataset, which is why this exists: export once where the data
    lives, render anywhere.

    The states are stored as float32. Posing is a kinematic write followed
    by ``mj_forward``, and a float32 qpos is identical to the double one to
    seven digits, which is far below anything a picture shows.

    Args:
        path: Destination ``.npz``.
        demos: The demonstrations to store.
        which: The ``--episode`` rule they were chosen by, for the header.

    Returns:
        The written path.
    """
    from bigym.loco.demos import hub

    chosen = list(demos)
    header = {
        "format": STATES_FORMAT,
        "dataset_repo": hub.dataset_repo(),
        "dataset_revision": hub.dataset_revision(),
        "episode_rule": str(which),
        "qpos_dtype": "float32",
        "tasks": [
            {
                "task": demo.task,
                "seed": int(demo.seed),
                "episode": int(demo.index),
                "source_file": demo.source_file,
                "frames": int(demo.length),
                "control_step_seconds": demo.control_seconds,
            }
            for demo in chosen
        ],
    }
    arrays = {
        f"qpos_{index}": np.asarray(demo.qpos, dtype=np.float32)
        for index, demo in enumerate(chosen)
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        header=json.dumps(header),
        **arrays,  # ty: ignore[invalid-argument-type]
    )
    size = target.stat().st_size
    print(
        f"[grid] wrote {len(chosen)} tasks to {target} "
        f"({size / 1e6:.1f} MB, float32 full_qpos)",
        flush=True,
    )
    return target


def load_states(path, tasks=None) -> list[GridDemo]:
    """Read demonstrations back from an ``--export-states`` npz.

    Args:
        path: The ``.npz`` written by :func:`export_states`.
        tasks: Only these task names, in this order, or None for every
            task in the file in its own order.

    Returns:
        The demonstrations.

    Raises:
        SystemExit: The file is missing, malformed, or lacks a task.
    """
    target = Path(path)
    try:
        with np.load(target, allow_pickle=False) as stored:
            header = json.loads(str(stored["header"]))
            rows = list(header.get("tasks") or [])
            demos = []
            for index, row in enumerate(rows):
                demos.append(
                    GridDemo(
                        task=str(row["task"]),
                        qpos=stored[f"qpos_{index}"],
                        seed=int(row["seed"]),
                        index=int(row.get("episode", -1)),
                        source_file=str(row.get("source_file", "")),
                        count=0,
                        control_seconds=row.get("control_step_seconds"),
                    )
                )
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"--states {target}: {exc}") from None
    if not demos:
        raise SystemExit(f"--states {target}: holds no tasks")
    print(
        f"[grid] {len(demos)} tasks from {target} "
        f"(exported from {header.get('dataset_repo')}, "
        f"episode rule {header.get('episode_rule')})",
        flush=True,
    )
    if tasks:
        by_name = {demo.task: demo for demo in demos}
        missing = [name for name in tasks if name not in by_name]
        if missing:
            raise SystemExit(
                f"--states {target}: no states for {', '.join(missing)} "
                f"(it holds {', '.join(by_name)})"
            )
        demos = [by_name[name] for name in tasks]
    return demos


# ---------------------------------------------------------------------------
# One scene per task
# ---------------------------------------------------------------------------


def measure_footprint(model, data, mujoco) -> Footprint:
    """Measure the floor a posed scene occupies, in its own coordinates.

    Every visible geom's world box (its position plus its bounding radius,
    which is the conservative one MuJoCo already keeps) goes in, except
    ground planes and anything bigger than :data:`FOOTPRINT_MAX_GEOM` —
    those are scenery, and one of them would swallow the whole grid.

    Args:
        model: The scene's ``mujoco.MjModel``.
        data: Its ``MjData``, already posed and forwarded.
        mujoco: The ``mujoco`` module.

    Returns:
        The :class:`Footprint`; the default box when nothing measurable is
        visible.
    """
    kinds = np.asarray(model.geom_type)
    alpha = np.asarray(model.geom_rgba)[:, 3]
    radius = np.asarray(model.geom_rbound, dtype=float)
    keep = (
        (kinds != int(mujoco.mjtGeom.mjGEOM_PLANE))
        & (alpha > 0.0)
        & (radius > 0.0)
        & (radius < FOOTPRINT_MAX_GEOM)
    )
    if not bool(keep.any()):
        return default_footprint()
    centres = np.asarray(data.geom_xpos, dtype=float)[keep]
    reach = radius[keep][:, None]
    low = (centres[:, :2] - reach).min(axis=0)
    high = (centres[:, :2] + reach).max(axis=0)
    return Footprint(
        lo=(float(low[0]), float(low[1])), hi=(float(high[0]), float(high[1]))
    )


def rgb255(colour) -> tuple[int, int, int]:
    """A MuJoCo rgba (0-1 floats) as viser's 0-255 integers."""
    values = np.asarray(colour, dtype=float).reshape(-1)[:3]
    return cast(tuple[int, int, int], tuple(int(round(255 * float(v))) for v in values))


def add_target_spheres(server, prefix: str, targets, offset) -> list[tuple]:
    """Draw one ball (and a highlight glow) per reach target of a scene.

    Args:
        server: The viser server.
        prefix: The scene's node-path segment.
        targets: The task's targets (empty for every non-reach task).
        offset: The scene's grid offset — the balls hang off the scene
            root, not off the scene's own frame, so they carry it
            themselves.

    Returns:
        One ``(target, ball, glow)`` triple per target.
    """
    drawn = []
    for i, target in enumerate(targets):
        radius = float(np.asarray(target._config.size).reshape(-1)[0])
        ball = server.scene.add_icosphere(
            f"/targets/{prefix}/{i}/ball",
            radius=radius,
            color=rgb255(target._config.color_default),
            position=tuple(float(v) for v in offset),
            visible=False,
        )
        glow = server.scene.add_icosphere(
            f"/targets/{prefix}/{i}/glow",
            radius=radius * 1.35,
            color=rgb255(target._config.color_highlight),
            opacity=0.5,
            position=tuple(float(v) for v in offset),
            visible=False,
        )
        drawn.append((target, ball, glow))
    return drawn


def update_targets(entry: dict) -> None:
    """Move a scene's target balls and light the reached one up."""
    offset = entry["offset"]
    for target, ball, glow in entry["targets"]:
        position = np.asarray(target.get_position(), dtype=float)
        place = tuple(float(position[i] + offset[i]) for i in range(3))
        ball.position = place
        glow.position = place
        glow.visible = bool(target.is_reached(entry["inner"].reach_tolerance))


def pose_scene(entry: dict, frame: int, mujoco) -> int:
    """Pose one scene at ``frame`` of its demonstration and push it out.

    Args:
        entry: One scene of the grid.
        frame: The frame to show, clamped to this demo.
        mujoco: The ``mujoco`` module.

    Returns:
        The frame actually used.
    """
    here = agent_cells.pose_slot(entry, frame, mujoco)
    run_task_hook(entry["inner"])
    if entry["targets"]:
        update_targets(entry)
    return here


def place_label(entry: dict, camera=None) -> tuple[float, float, float] | None:
    """Move a scene's plate onto its pelvis; None when labels are off.

    Args:
        entry: One scene of the grid.
        camera: Where the viewer's camera is, so the plate can be turned
            to face it, or None to leave it on compare mode's fixed -x.

    Returns:
        The plate's new anchor, or None when labels are off.
    """
    billboard = entry.get("billboard")
    if billboard is None:
        return None
    anchor = agent_cells.billboard_anchor(
        agent_cells.pelvis_position(entry["data"], entry["pelvis"]),
        entry["offset"],
        PLATE_Z,
    )
    billboard.node.position = tuple(float(v) for v in anchor)
    if camera is not None:
        billboard.node.wxyz = agent_cells.facing_wxyz(anchor, camera)
    return anchor


def build_grid(
    demos: list[GridDemo],
    *,
    columns: int = DEFAULT_COLUMNS,
    clearance: float = DEFAULT_CLEARANCE,
    gap: float = 0.0,
    port: int = DEFAULT_PORT,
    labels: bool = True,
    tint: bool = False,
    sky: str = "black",
    aspect: float = 16 / 9,
    server=None,
) -> dict:
    """Build one viser server holding one posed scene per demonstration.

    Every scene is built the way compare mode builds a slot: its own
    ``MjModel``/``MjData`` from ``make(task)``, reset on its
    demonstration's seed, a distinct mjviser node prefix and a frame that
    carries its place in the grid. Nothing is visible while that happens —
    the frames are created before any geometry and start hidden — and only
    when the last scene is built is every one of them posed to frame 0 of
    its demo and revealed together.

    The placement is measured, not assumed: every environment is built and
    reset first, its footprint taken from the geoms it actually draws, and
    only then are the scenes packed so neighbouring footprints clear each
    other by ``clearance``. A kitchen that reaches 2.3 m one way and 2 m
    the other therefore sits as close to its neighbour as it can.

    Args:
        demos: The demonstrations, in grid order.
        columns: Scenes per row.
        clearance: Metres of floor between neighbouring footprints.
        gap: Non-zero asks for a uniform pitch instead: every scene
            at its origin, ``gap`` metres apart.
        port: The viser port (ignored when ``server`` is given).
        labels: Draw a name plate above each robot's pelvis.
        tint: Give each robot its palette colour (display only: the tint is
            put into the model for the mesh bake and taken straight out).
        sky: ``black``, ``grey`` or ``transparent``.
        server: An open viser server, or None to open one on ``port``.

    Returns:
        A dict with ``server``, ``scenes`` (one entry per demo), ``offsets``
        and ``mujoco``.
    """
    patch_mjviser()
    if server is None:
        server = agent_cells.compare_server(port)
    # One /fixed_bodies root for every scene (see agent_cells.slot_scene_build).
    fixed_frame = server.scene.add_frame("/fixed_bodies", show_axes=False)
    total = len(demos)
    # First pass: build every environment, pose it at frame 0 and measure
    # the floor it takes. Nothing is drawn yet -- the layout is not known
    # until the last footprint is in.
    raw: list[dict] = []
    for i, demo in enumerate(demos):
        print(
            f"[grid] building {i + 1}/{total} {demo.task} "
            f"(seed {demo.seed}, {demo.length} frames)",
            flush=True,
        )
        env = replay_env(demo.task)
        env.reset(seed=demo.seed)
        inner = env.inner_env
        model, data = inner.model, inner.data
        stored = int(demo.qpos.shape[1])
        data.qpos[:stored] = demo.qpos[0]
        mujoco.mj_forward(model, data)
        footprint = measure_footprint(model, data, mujoco)
        print(
            f"[grid] {demo.task}: footprint {footprint.depth:.2f} x "
            f"{footprint.width:.2f} m, centre "
            f"({footprint.centre[0]:+.2f}, {footprint.centre[1]:+.2f})",
            flush=True,
        )
        raw.append(
            {"demo": demo, "env": env, "inner": inner, "model": model, "data": data,
             "footprint": footprint}
        )  # fmt: skip
    footprints = [entry["footprint"] for entry in raw]
    layout = (
        uniform_layout(footprints, columns, gap)
        if gap and gap > 0.0
        else pack_layout(footprints, columns, clearance)
    )
    offsets = list(layout.offsets)
    places = list(layout.places)
    for line in layout.describe():
        print(line, flush=True)
    scenes: list[dict] = []
    for i, source in enumerate(raw):
        demo = source["demo"]
        env, inner = source["env"], source["inner"]
        model, data = source["model"], source["data"]
        targets = hide_reach_targets(inner)
        prefix = f"task{i}"
        offset = offsets[i]
        # The offset frames exist before a single geom goes under them, so
        # no scene is ever drawn at the origin and then moved.
        body_frame = server.scene.add_frame(
            f"/bodies/{prefix}", show_axes=False, position=offset, visible=False
        )
        scene_fixed = server.scene.add_frame(
            f"/fixed_bodies/{prefix}", show_axes=False, position=offset, visible=False
        )
        patch_mjviser_track_props(model, inner)
        pristine = np.array(model.geom_rgba, copy=True)
        tinted = agent_cells.tint_slot_robot(model, scene_tint(i)) if tint else 0
        with agent_cells.slot_scene_build(
            server, prefix, keep_grid=i == 0, fixed_frame=fixed_frame
        ):
            scene = ViserMujocoScene(server=server, mj_model=model, num_envs=1)
        model.geom_rgba[:] = pristine
        scene.camera_tracking_enabled = False
        billboard = None
        if labels:
            billboard = agent_cells.SlotBillboard(
                server,
                f"/grid/{prefix}/label",
                position=(offset[0], offset[1], offset[2] + PLATE_Z),
                visible=False,
                width=PLATE_WIDTH,
            )
            # The plate is drawn once here; nothing it says changes during
            # playback.
            billboard.write(demo.label, "", scene_colour(i))
        entry = {
            "demo": demo,
            "task": demo.task,
            "label": demo.label,
            "env": env,
            "inner": inner,
            "model": model,
            "data": data,
            "scene": scene,
            "qpos": demo.qpos,
            "info": {},
            "pelvis": agent_cells.pelvis_body_id(model),
            "billboard": billboard,
            "frames": (body_frame, scene_fixed),
            "geom_rgba": pristine,
            "offset": offset,
            "place": places[i],
            "prefix": prefix,
            "targets": add_target_spheres(server, prefix, targets, offset),
        }
        scenes.append(entry)
        stored = int(demo.qpos.shape[1])
        if stored != int(model.nq):
            print(
                f"[grid] {demo.task}: demo stores {stored} of the model's "
                f"{int(model.nq)} qpos values; the rest keep their reset value",
                flush=True,
            )
        print(
            f"[grid] scene {i}: {prefix} = {demo.task} at row {places[i][0]} "
            f"column {places[i][1]} (x {offset[0]:+.2f}, y {offset[1]:+.2f}), "
            f"built hidden, {tinted} geoms tinted, {len(targets)} reach target(s)",
            flush=True,
        )
    agent_cells.apply_lighting(
        server,
        {"sky": sky, "environment": 0.7, "default_lights": True, "shadows": True},
    )
    for i, entry in enumerate(scenes):
        pose_scene(entry, 0, mujoco)
        anchor = place_label(entry)
        for _target, ball, _glow in entry["targets"]:
            ball.visible = True
        where = ""
        if anchor is not None:
            where = f", label at ({anchor[0]:.2f}, {anchor[1]:.2f}, {anchor[2]:.2f})"
        print(
            f"[grid] scene {i}: {entry['task']} posed to frame 0 of the demo "
            f"(not the reset pose){where}",
            flush=True,
        )
    for entry in scenes:
        for frame in entry["frames"]:
            frame.visible = True
        if entry["billboard"] is not None:
            entry["billboard"].reveal()
    print(f"[grid] {len(scenes)} scenes revealed together", flush=True)
    return {
        "server": server,
        "scenes": scenes,
        "offsets": offsets,
        "layout": layout,
        "aspect": float(aspect),
        "mujoco": mujoco,
    }


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------


def scene_pose(built: dict, index: int):
    """``(position, look_at)`` for the camera watching one scene alone."""
    entry = built["scenes"][index]
    return agent_cells.focus_pose(
        entry["offset"], agent_cells.pelvis_position(entry["data"], entry["pelvis"])
    )


def overview_pose(
    layout: Layout,
    aspect: float = 16 / 9,
    fill: float = OVERVIEW_FILL,
    tilt: float = OVERVIEW_TILT,
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Where the camera stands to show the whole grid, and through what lens.

    The grid's own box is what is framed: the camera looks at its centre
    from the -x side, tilted down by ``tilt``, and the lens is chosen so
    the box covers ``fill`` of the frame. Aiming at the box's centre rather
    than at a point on the floor is what stops the grid sitting in a band
    across the middle of the picture with an empty sky above it.

    Args:
        layout: The packed grid.
        aspect: Frame width over height.
        fill: How much of the frame the grid should cover (0.8 = 80%).
        tilt: How far the camera looks down, in radians.

    Returns:
        ``(position, look_at, vertical fov)``.
    """
    centre = layout.centre
    half = layout.half()
    look_at = (centre[0], centre[1], centre[2])
    # Far enough out that the near row is not distorted, near enough that
    # the lens stays normal: one grid depth plus its width, whichever leads.
    distance = max(6.0, 1.15 * max(2.0 * half[0], half[1] / max(0.2, float(aspect))))
    position = (
        centre[0] - distance * math.cos(tilt),
        centre[1],
        centre[2] + distance * math.sin(tilt),
    )
    # What the box subtends from there, measured against its near face.
    near = max(1.0, distance - half[0] * math.cos(tilt))
    upright = half[0] * math.sin(tilt) + half[2] * math.cos(tilt)
    vertical = 2.0 * math.atan(upright / near)
    horizontal = 2.0 * math.atan(half[1] / near)
    from_width = 2.0 * math.atan(math.tan(0.5 * horizontal) / max(0.1, float(aspect)))
    needed = max(vertical, from_width)
    fov = float(np.clip(needed / max(0.2, float(fill)), FOV_MIN, FOV_MAX))
    return (position, look_at, fov)


def camera_pose(built: dict, camera: int):
    """``(position, look_at)``: the whole grid (-1) or one scene."""
    if camera < 0:
        position, look_at, _ = overview_pose(
            built["layout"], built.get("aspect", 16 / 9)
        )
        return (position, look_at)
    return scene_pose(built, camera)


def grid_box(offsets) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """The box the whole grid lives in: ``(centre, half extents)``.

    A scene's own content reaches about 2.3 m along +x, 2 m along -y and
    1.3 m along +y of its origin (the widest published task fixture is
    2.73 m across), and a demonstrator walks inside that, so each offset is
    padded by :data:`SCENE_PAD` before the grid's extent is taken.

    Args:
        offsets: Every scene's offset.

    Returns:
        ``((cx, cy, cz), (hx, hy, hz))`` in world metres.
    """
    places = [tuple(float(v) for v in o) for o in offsets] or [(0.0, 0.0, 0.0)]
    xs = [o[0] for o in places]
    ys = [o[1] for o in places]
    centre = (
        0.5 * (min(xs) + max(xs)) + SCENE_CENTRE[0],
        0.5 * (min(ys) + max(ys)) + SCENE_CENTRE[1],
        SCENE_CENTRE[2],
    )
    half = (
        0.5 * (max(xs) - min(xs)) + SCENE_PAD,
        0.5 * (max(ys) - min(ys)) + SCENE_PAD,
        SCENE_HEIGHT,
    )
    return centre, half


def grid_fov(
    offsets,
    position,
    look_at,
    aspect: float = 16 / 9,
    margin: float = 1.08,
    box=None,
) -> float:
    """The vertical field of view that fills the frame with the grid.

    viser's camera opens about 80 degrees vertically, which is right for
    standing next to one robot and wrong for a grid twenty metres wide: the
    scenes end up in a small band in the middle of the picture with black
    sky above and empty floor below (measured: 37% of the frame width).
    Rather than dragging the camera closer — which would foreshorten the
    near row — the lens is narrowed to what the grid actually subtends from
    where :func:`agent_cells.compare_camera_pose` stands.

    Args:
        offsets: Every scene's offset.
        position: Where the camera is.
        look_at: What it looks at.
        aspect: Frame width over height.
        margin: How much room to leave around the grid (1.08 = 8%).
        box: ``(centre, half extents)`` to frame instead of
            :func:`grid_box` of the offsets — compare mode's slots are a
            table and a robot, not a kitchen.

    Returns:
        The vertical field of view in radians, clamped to something a
        camera can sensibly have.
    """
    centre, half = box if box is not None else grid_box(offsets)
    eye = np.asarray(position, dtype=float)
    middle = np.asarray(centre, dtype=float)
    distance = float(np.linalg.norm(middle - eye))
    if distance < 1e-6:
        return FOV_MAX
    # How much of the grid's depth and height stands across the line of
    # sight: the steeper the camera looks down, the more of the depth it is.
    direction = (middle - eye) / distance
    tilt = float(np.arcsin(np.clip(-direction[2], -1.0, 1.0)))
    upright = half[0] * np.sin(tilt) + half[2] * np.cos(tilt)
    wide = half[1]
    # The near row is half the grid's depth closer than its centre, so it
    # subtends more: frame against that distance, or the nearest scenes are
    # cut off at the edges (measured on a 4 x 5 capture).
    near = max(1.0, distance - half[0] * np.cos(tilt))
    vertical = 2.0 * np.arctan(margin * upright / near)
    horizontal = 2.0 * np.arctan(margin * wide / near)
    # A wide grid is framed by the horizontal field, which the aspect ratio
    # turns back into the vertical one the camera is set with.
    from_width = 2.0 * np.arctan(np.tan(0.5 * horizontal) / max(0.1, float(aspect)))
    return float(np.clip(max(vertical, from_width), FOV_MIN, FOV_MAX))


def pose_grid(built: dict, playhead: int, loop: str = LOOP_ALL) -> None:
    """Pose every scene at one playhead position and push it to the browser.

    One ``server.atomic()`` around the lot so the twenty scenes reach the
    client as one update rather than twenty, which matters for a recording:
    ``get_render`` must never catch the grid half-posed.

    Args:
        built: What :func:`build_grid` returned.
        playhead: The global playhead (see :func:`loop_frame`).
        loop: ``all`` or ``each``.
    """
    scenes = built["scenes"]
    longest = loop_length([entry["qpos"].shape[0] for entry in scenes], loop)
    # One camera lookup for the whole grid rather than one per plate.
    camera = built.get("camera")
    if camera is None:
        camera = agent_cells.viewer_camera_position(built["server"])
    with built["server"].atomic():
        for entry in scenes:
            here = loop_frame(playhead, entry["qpos"].shape[0], longest, loop)
            pose_scene(entry, here, built["mujoco"])
            place_label(entry, camera)


def close_grid(built: dict) -> None:
    """Stop the server and close every environment of the grid."""
    with contextlib.suppress(Exception):
        built["server"].stop()
    for entry in built["scenes"]:
        with contextlib.suppress(Exception):
            entry["env"].close()


def run_grid(
    built: dict,
    *,
    fps: float = DEFAULT_FPS,
    loop: str = LOOP_ALL,
    camera: int = -1,
    hide_gui: bool = True,
    exit_after: float = 0.0,
    keys=None,
) -> None:
    """Play every scene until the user leaves (or ``exit_after`` elapses).

    Args:
        built: What :func:`build_grid` returned.
        fps: How fast demo time runs; 50 is real time. It is not a render
            rate: the loop renders as fast as it can and skips the frames
            it could not show, so the demonstrations keep their own speed
            on a grid too big to push at ``fps``. The measured render rate
            is logged every few seconds.
        loop: ``all`` (one playhead, everything restarts together) or
            ``each`` (every scene loops on its own length).
        camera: The scene every client's camera starts on, or -1 for the
            overview of the whole grid.
        hide_gui: Collapse the sidebar and its folder.
        exit_after: Serve this many seconds, then return (0 = until
            Ctrl-C).
        keys: A camera path to fly on the wall clock, looping, instead of
            a fixed camera; None leaves the camera where it is put.
    """
    server, scenes = built["server"], built["scenes"]
    # A recording pins the plates to its own camera; live playback turns
    # them towards whoever is watching instead (--record-then-stay).
    built.pop("camera", None)
    lengths = [entry["qpos"].shape[0] for entry in scenes]
    longest = loop_length(lengths, loop)
    names = [entry["label"] for entry in scenes]
    state: dict = {"playhead": 0, "dirty": True, "focus": None}

    if hide_gui:
        video.hide_gui_chrome(server)
    with server.gui.add_folder("Playback", expand_by_default=not hide_gui):
        playing = server.gui.add_checkbox("play", initial_value=True)
        frame_slider = server.gui.add_slider(
            "frame", min=0, max=max(1, longest - 1), step=1, initial_value=0
        )
        fps_slider = server.gui.add_slider(
            "fps", min=1, max=100, step=1, initial_value=int(round(fps))
        )
        focus_dd = server.gui.add_dropdown(
            "focus",
            options=[FOCUS_OVERVIEW, *names],
            initial_value=FOCUS_OVERVIEW if camera < 0 else names[camera],
        )

    def on_frame(_=None) -> None:
        state.update(playhead=int(frame_slider.value), dirty=True)

    def on_focus(_=None) -> None:
        value = str(focus_dd.value)
        state["focus"] = -1 if value == FOCUS_OVERVIEW else names.index(value)

    frame_slider.on_update(on_frame)
    focus_dd.on_update(on_focus)

    start_pose = camera_pose(built, camera)

    def aim(client, pose, overview: bool) -> None:
        """Put one client's camera on a pose, with the lens the shot wants."""
        client.camera.position = tuple(float(v) for v in pose[0])
        client.camera.look_at = tuple(float(v) for v in pose[1])
        if not overview:
            return
        aspect = 16 / 9
        with contextlib.suppress(Exception):
            aspect = float(client.camera.aspect)
        with contextlib.suppress(Exception):
            client.camera.fov = grid_fov(built["offsets"], pose[0], pose[1], aspect)

    def aim_all(pose, overview: bool) -> int:
        """Move every connected client (main loop only)."""
        moved = 0
        try:
            clients = list(server.get_clients().values())
        except Exception:  # a disconnect mid-iteration is fine
            return 0
        for client in clients:
            with contextlib.suppress(Exception):
                aim(client, pose, overview)
                moved += 1
        return moved

    @server.on_client_connect
    def _(client) -> None:
        aim(client, start_pose, camera < 0)

    aim_all(start_pose, camera < 0)
    flight = path_duration(keys) if keys else 0.0
    if flight > 0.0:
        assert keys is not None
        print(
            f"[grid] flying the camera over {flight:g}s, looping "
            f"({len(list(keys))} keyframes, peak {path_speed(keys):.1f} m/s)",
            flush=True,
        )
    print(
        f"[grid] playing {len(scenes)} scenes at {fps:g} fps, loop={loop}, "
        f"longest demo {longest} frames",
        flush=True,
    )
    print(
        f"[grid] camera: at ({start_pose[0][0]:.2f}, {start_pose[0][1]:.2f}, "
        f"{start_pose[0][2]:.2f}) looking at ({start_pose[1][0]:.2f}, "
        f"{start_pose[1][1]:.2f}, {start_pose[1][2]:.2f})",
        flush=True,
    )

    started = time.time()
    head_time = started
    rate_at = started
    updates = 0
    frames = 0
    measured = 0.0
    try:
        while True:
            now = time.time()
            if exit_after > 0.0 and now - started > exit_after:
                print(
                    f"[grid] --exit-after-seconds {exit_after:g} elapsed, closing",
                    flush=True,
                )
                break
            rate = max(1.0, float(fps_slider.value))
            if not bool(playing.value):
                head_time = now  # a pause must not turn into a jump
            else:
                # The playhead follows the wall clock, not the tick: twenty
                # scenes cost more than 20 ms to push to the browser (see
                # the module docstring), and a demonstration that plays at
                # a third of its speed is worth less than one that plays at
                # its own speed and drops frames. ``steps`` is how many
                # recorded frames have gone by since the last update.
                steps = int((now - head_time) * rate)
                if steps:
                    head_time += steps / rate
                    frames += steps
                    state["playhead"] = (int(state["playhead"]) + steps) % longest
                    frame_slider.value = int(state["playhead"])
                    state["dirty"] = True
            if flight > 0.0:
                # The flight runs on the wall clock and loops; whoever is
                # watching is carried along with it.
                position, look_at, fov = sample_path(keys, (now - started) % flight)
                try:
                    clients = list(server.get_clients().values())
                except Exception:  # a disconnect is not an error
                    clients = []
                for client in clients:
                    aim_client(client, position, look_at, fov)
                built["camera"] = tuple(float(v) for v in position)
            focus, state["focus"] = state["focus"], None
            if focus is not None:
                pose = camera_pose(built, focus)
                moved = aim_all(pose, focus < 0)
                where = "the overview" if focus < 0 else names[focus]
                print(f"[grid] flying {moved} client(s) to {where}", flush=True)
            if state["dirty"]:
                pose_grid(built, int(state["playhead"]), loop)
                state["dirty"] = False
                updates += 1
            if now - rate_at >= RATE_LOG_SECONDS:
                window = now - rate_at
                measured = updates / window
                print(
                    f"[grid] playback {measured:.1f} fps measured "
                    f"({frames / window:.1f} frames of demo time per second, "
                    f"asked for {rate:g})",
                    flush=True,
                )
                updates, frames, rate_at = 0, 0, now
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("[grid] interrupted", flush=True)
    finally:
        window = max(1e-6, time.time() - rate_at)
        if updates:
            measured = updates / window
        print(
            f"[grid] playback {measured:.1f} fps measured over the last pass "
            f"({len(scenes)} scenes)"
        )
        close_grid(built)


# ---------------------------------------------------------------------------
# Camera paths (a flight through the grid, for the promotional clip)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Keyframe:
    """One pose the camera passes through, at one moment of the flight.

    Attributes:
        t: Seconds from the start of the path.
        position: Where the camera is.
        look_at: What it points at.
        fov: Vertical field of view in radians.
        ease: Smooth the approach to this keyframe (used at the start, the
            turn and the end, so the motion starts and stops softly).
    """

    t: float
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    fov: float
    ease: bool = False


def smoothstep(value: float) -> float:
    """The 3t² - 2t³ ease, clamped to ``[0, 1]``."""
    clamped = min(1.0, max(0.0, float(value)))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def catmull_rom(before, start, end, after, s: float, times=None):
    """One Catmull-Rom segment from ``start`` to ``end``.

    Written as a Hermite segment whose end tangents are the neighbours'
    finite difference. With ``times`` those differences are taken over the
    real time between the keyframes, which is what a path with an
    eleven-second dolly next to a one-second corner needs: the uniform
    form gives the long segment a tangent sized for the short one and bows
    the camera a foot sideways out of the aisle (measured: 0.65 m).

    Args:
        before: The point before the segment (``start`` at the ends).
        start: Where the segment begins.
        end: Where it ends.
        after: The point after it (``end`` at the ends).
        s: Position along the segment, 0 to 1.
        times: ``(t0, t1, t2, t3)`` of the four points, or None for the
            uniform spacing the plain Catmull-Rom assumes.

    Returns:
        The interpolated point, as a numpy array.
    """
    p0 = np.asarray(before, dtype=float)
    p1 = np.asarray(start, dtype=float)
    p2 = np.asarray(end, dtype=float)
    p3 = np.asarray(after, dtype=float)
    u = float(s)
    t0, t1, t2, t3 = (
        (0.0, 1.0, 2.0, 3.0) if times is None else [float(v) for v in times]
    )
    span = max(1e-6, t2 - t1)
    # Tangents scaled to this segment's own duration.
    m1 = (p2 - p0) / max(1e-6, t2 - t0) * span
    m2 = (p3 - p1) / max(1e-6, t3 - t1) * span
    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2
    return h00 * p1 + h10 * m1 + h01 * p2 + h11 * m2


def path_duration(keys) -> float:
    """How long a path runs, in seconds."""
    frames = list(keys)
    return float(frames[-1].t) if frames else 0.0


def sample_path(keys, t: float):
    """The camera pose at time ``t`` of a path.

    Position and look-at follow a Catmull-Rom spline through the
    keyframes, so the motion has no corners; the field of view follows the
    same curve. A segment whose end keyframe asks for it is eased in and
    out with :func:`smoothstep`, which is what keeps the start, the turn
    and the arrival on the overview from snapping.

    Args:
        keys: The keyframes, in time order.
        t: Seconds from the start (clamped to the path).

    Returns:
        ``(position, look_at, fov)``.
    """
    frames = list(keys)
    if not frames:
        raise ValueError("a camera path needs at least one keyframe")
    if len(frames) == 1:
        one = frames[0]
        return (tuple(one.position), tuple(one.look_at), float(one.fov))
    when = min(max(float(t), frames[0].t), frames[-1].t)
    index = 0
    for i in range(len(frames) - 1):
        if when <= frames[i + 1].t:
            index = i
            break
        index = i
    start, end = frames[index], frames[index + 1]
    span = max(1e-6, float(end.t) - float(start.t))
    s = (when - float(start.t)) / span
    if start.ease or end.ease:
        s = smoothstep(s)
    before = frames[index - 1] if index > 0 else start
    after = frames[index + 2] if index + 2 < len(frames) else end
    clock = (before.t, start.t, end.t, after.t)
    if before is start:
        clock = (start.t - span, start.t, end.t, clock[3])
    if after is end:
        clock = (clock[0], start.t, end.t, end.t + span)
    position = catmull_rom(
        before.position, start.position, end.position, after.position, s, clock
    )
    look_at = catmull_rom(
        before.look_at, start.look_at, end.look_at, after.look_at, s, clock
    )
    fov = float(
        catmull_rom([before.fov], [start.fov], [end.fov], [after.fov], s, clock)[0]
    )
    return (
        tuple(float(v) for v in position),
        tuple(float(v) for v in look_at),
        float(np.clip(fov, FOV_MIN, FOV_MAX)),
    )


def fly_keyframes(
    layout: Layout,
    duration: float = DEFAULT_DURATION,
    aspect: float = 16 / 9,
    anchor=None,
) -> tuple[Keyframe, ...]:
    """The preset flight: one scene, two aisles, then up to the overview.

    Built from the packed layout, so it works for any task count, column
    count and clearance: the dolly runs down the real aisle between the
    first two rows (:meth:`Layout.aisle`, midway between their footprint
    edges), turns at the far end and comes back down the aisle between the
    last two, then rises to the overview and holds it.

    Args:
        layout: The packed grid.
        duration: Seconds for the whole path.
        aspect: Frame width over height (the overview's lens needs it).
        anchor: The world position of the first scene's pelvis, or None to
            use the middle of its footprint.

    Returns:
        The keyframes, in time order.
    """
    total = max(1.0, float(duration))
    rows = max(1, len(layout.rows))
    near_aisle = layout.aisle(0)
    back_aisle = layout.aisle(max(0, rows - 2))
    first_y = (
        0.5 * (layout.columns[0][0] + layout.columns[0][1]) if layout.columns else 0.0
    )
    last_y = (
        0.5 * (layout.columns[-1][0] + layout.columns[-1][1]) if layout.columns else 0.0
    )
    out_y = (layout.columns[-1][1] + 2.0) if layout.columns else 2.0
    row0_x = 0.5 * (layout.rows[0][0] + layout.rows[0][1])
    back_x = 0.5 * (layout.rows[-1][0] + layout.rows[-1][1])
    if anchor is None:
        first = layout.footprints[0] if layout.footprints else default_footprint()
        offset = layout.offsets[0] if layout.offsets else (0.0, 0.0, 0.0)
        middle = first.centre
        anchor = (middle[0] + offset[0], middle[1] + offset[1], EYE_HEIGHT)
    anchor = tuple(float(v) for v in anchor)
    eye = EYE_HEIGHT
    overview = overview_pose(layout, aspect)
    # 0-3s: standing in front of the first scene, drifting sideways.
    keys = [
        Keyframe(
            t=0.0,
            position=(near_aisle, anchor[1] - 0.8, eye),
            look_at=(anchor[0], anchor[1], anchor[2] + 0.3),
            fov=CLOSE_FOV,
            ease=True,
        ),
        Keyframe(
            t=0.10 * total,
            position=(near_aisle, first_y, eye),
            look_at=(anchor[0], anchor[1] + 0.4, anchor[2] + 0.3),
            fov=CLOSE_FOV,
        ),
        # 3-14s: down the aisle between the first two rows, looking ahead
        # and a little towards row 0 as its scenes go by.
        Keyframe(
            t=0.467 * total,
            position=(near_aisle, last_y, eye),
            look_at=(row0_x, last_y + AISLE_LEAD, eye - 0.2),
            fov=AISLE_FOV,
        ),
        # 14-22s: out past the last column, across to the other aisle and
        # back down it. Each aisle keeps a keyframe just outside the grid
        # as well as one at its end, so a dolly segment's neighbours are
        # both on its own aisle: the spline then has no sideways tangent
        # and cannot bow the camera into a worktop.
        Keyframe(
            t=0.500 * total,
            position=(near_aisle, out_y, eye),
            look_at=(row0_x, out_y + 1.0, eye - 0.2),
            fov=AISLE_FOV,
            ease=True,
        ),
        Keyframe(
            t=0.533 * total,
            position=(0.5 * (near_aisle + back_aisle), out_y + 1.2, eye + 0.2),
            look_at=(back_x, last_y, eye - 0.2),
            fov=AISLE_FOV,
        ),
        Keyframe(
            t=0.567 * total,
            position=(back_aisle, out_y, eye),
            look_at=(back_x, last_y - 1.0, eye - 0.2),
            fov=AISLE_FOV,
            ease=True,
        ),
        Keyframe(
            t=0.600 * total,
            position=(back_aisle, last_y, eye),
            look_at=(back_x, last_y - AISLE_LEAD, eye - 0.2),
            fov=AISLE_FOV,
        ),
        Keyframe(
            t=0.733 * total,
            position=(back_aisle, first_y, eye),
            look_at=(back_x, first_y - AISLE_LEAD, eye - 0.2),
            fov=AISLE_FOV,
        ),
        Keyframe(
            t=0.767 * total,
            position=(
                back_aisle,
                layout.columns[0][0] - 1.6 if layout.columns else -1.6,
                eye,
            ),
            look_at=(back_x, first_y - AISLE_LEAD, eye - 0.2),
            fov=AISLE_FOV,
            ease=True,
        ),
        # 22-28s: rise and pull back onto the overview, and hold it.
        Keyframe(
            t=0.933 * total,
            position=overview[0],
            look_at=overview[1],
            fov=overview[2],
            ease=True,
        ),
        Keyframe(
            t=total,
            position=overview[0],
            look_at=overview[1],
            fov=overview[2],
            ease=True,
        ),
    ]
    return tuple(keys)


def keyframes_to_json(keys) -> list[dict]:
    """The keyframes as the JSON ``--path-dump`` writes (fov in degrees)."""
    return [
        {
            "t": round(float(key.t), 4),
            "position": [round(float(v), 4) for v in key.position],
            "look_at": [round(float(v), 4) for v in key.look_at],
            "fov": round(float(np.degrees(key.fov)), 3),
            "ease": bool(key.ease),
        }
        for key in keys
    ]


def keyframes_from_json(rows) -> tuple[Keyframe, ...]:
    """Read keyframes back from ``--path-keyframes`` JSON.

    Args:
        rows: A list of ``{"t", "position", "look_at", "fov"}`` objects;
            ``fov`` is in degrees and ``ease`` is optional.

    Returns:
        The keyframes, sorted by time.

    Raises:
        SystemExit: The file is not a list of such objects.
    """
    keys = []
    if not isinstance(rows, list) or not rows:
        raise SystemExit("--path-keyframes: expected a non-empty list of keyframes")
    for row in rows:
        try:
            position = tuple(float(v) for v in row["position"])
            look_at = tuple(float(v) for v in row["look_at"])
            when = float(row["t"])
            fov = float(np.radians(float(row.get("fov", np.degrees(AISLE_FOV)))))
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            raise SystemExit(
                f"--path-keyframes: bad keyframe {row!r} ({exc})"
            ) from None
        if len(position) != 3 or len(look_at) != 3:
            raise SystemExit(f"--path-keyframes: {row!r} needs 3 numbers per point")
        keys.append(
            Keyframe(
                t=when,
                position=position,
                look_at=look_at,
                fov=float(np.clip(fov, FOV_MIN, FOV_MAX)),
                ease=bool(row.get("ease", False)),
            )
        )
    return tuple(sorted(keys, key=lambda key: key.t))


def load_keyframes(path) -> tuple[Keyframe, ...]:
    """Read a path from a JSON file written by ``--path-dump`` or by hand."""
    try:
        rows = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"--path-keyframes {path}: {exc}") from None
    return keyframes_from_json(rows)


def dump_keyframes(path, keys) -> Path:
    """Write a path to JSON so it can be edited and fed back in."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(keyframes_to_json(keys), indent=2) + "\n")
    print(f"[grid] wrote {len(list(keys))} keyframes to {target}", flush=True)
    return target


def path_speed(keys, samples: int = 400) -> float:
    """The fastest the camera moves along a path, in metres per second."""
    frames = list(keys)
    total = path_duration(frames)
    if total <= 0.0 or len(frames) < 2:
        return 0.0
    step = total / max(2, int(samples))
    fastest, previous = 0.0, np.asarray(sample_path(frames, 0.0)[0])
    for index in range(1, int(samples) + 1):
        here = np.asarray(sample_path(frames, index * step)[0])
        fastest = max(fastest, float(np.linalg.norm(here - previous)) / step)
        previous = here
    return fastest


# ---------------------------------------------------------------------------
# Offline recording (the browser renders; ffmpeg writes)
# ---------------------------------------------------------------------------


def record_schedule(lengths, loop: str = LOOP_ALL, frames: int = 0) -> list[int]:
    """The playhead positions a recording steps through, in order.

    A recording is deterministic: it walks the playhead one frame at a time
    with no wall clock anywhere, so every frame of every demonstration is in
    the file whatever the live rate was.

    Args:
        lengths: Frames in each scene's demonstration.
        loop: ``all`` or ``each`` (see :func:`loop_frame`).
        frames: How many frames to record, or 0 for one pass over the grid
            (the longest demonstration).

    Returns:
        ``[0, 1, ... n - 1]``; :func:`loop_frame` maps each one onto each
        scene, so a count beyond one pass simply starts the loop again.
    """
    total = int(frames) if int(frames or 0) > 0 else loop_length(lengths, loop)
    return list(range(max(1, total)))


def aim_camera(built: dict, client, camera: int, size=None, settle: float = 0.5):
    """Point one client's camera at the grid (or one scene) and let it land.

    Args:
        built: What :func:`build_grid` returned.
        client: The client that will render.
        camera: Scene index, or -1 for the overview.
        size: The frame ``(width, height)`` the lens is fitted to, or None
            for 16:9.
        settle: Seconds to let the camera update reach the browser before
            the first frame is asked for.

    Returns:
        The ``(position, look_at)`` the camera was put at.
    """
    pose = camera_pose(built, camera)
    aspect = (float(size[0]) / float(size[1])) if size else 16 / 9
    fov = grid_fov(built["offsets"], pose[0], pose[1], aspect) if camera < 0 else None
    with contextlib.suppress(Exception):
        client.camera.position = tuple(float(v) for v in pose[0])
        client.camera.look_at = tuple(float(v) for v in pose[1])
        if fov is not None:
            client.camera.fov = float(fov)
    # The plates turn to this, rather than to whatever the client last
    # reported (a camera write takes a moment to come back).
    built["camera"] = tuple(float(v) for v in pose[0])
    if settle > 0.0:
        time.sleep(settle)
    lens = "" if fov is None else f", {np.degrees(fov):.0f} degree lens"
    print(
        f"[grid] recording camera: at ({pose[0][0]:.2f}, {pose[0][1]:.2f}, "
        f"{pose[0][2]:.2f}) looking at ({pose[1][0]:.2f}, {pose[1][1]:.2f}, "
        f"{pose[1][2]:.2f}){lens}",
        flush=True,
    )
    return pose


def settle_size(size: tuple[int, int]) -> tuple[int, int]:
    """A cheap render size, in the recording's aspect, for the settle loop."""
    width, height = int(size[0]), int(size[1])
    scale = max(1, min(width // 640, height // 360) or 1)
    return (max(160, width // scale), max(90, height // scale))


def wait_for_page(
    client,
    size: tuple[int, int],
    *,
    timeout: float = PAGE_SETTLE_SECONDS,
    tolerance: float = PAGE_SETTLE_TOLERANCE,
    poll: float = 0.5,
) -> bool:
    """Render until the page stops changing, then let the capture start.

    ``get_render`` does not wait for anything the browser is still loading —
    the environment map that makes the sky black, and twenty scenes' worth
    of meshes and label plates all arrive asynchronously. A capture taken
    the moment a client connects therefore shows a white sky and half the
    grid missing (measured: the first render of a twenty-scene page). The
    scene is static here (the playhead has not moved yet), so "loaded" is
    simply "two renders in a row look the same".

    Args:
        client: The browser that renders.
        size: The recording ``(width, height)``; the settle renders are
            taken smaller (:func:`settle_size`) because they are thrown
            away.
        timeout: Give up waiting after this many seconds and record anyway.
        tolerance: Mean absolute channel difference, 0-255, below which two
            renders count as the same picture.
        poll: Seconds between renders.

    Returns:
        True when the page settled, False when it was still changing.
    """
    width, height = settle_size(size)
    previous = None
    started = time.time()
    renders = 0
    while True:
        frame = video.rgb_frame(client.get_render(height, width))
        renders += 1
        if previous is not None and previous.shape == frame.shape:
            delta = float(
                np.abs(frame.astype(np.int16) - previous.astype(np.int16)).mean()
            )
            if delta <= float(tolerance):
                print(
                    f"[grid] page settled after {renders} renders, "
                    f"{time.time() - started:.1f}s (last change {delta:.2f}/255)",
                    flush=True,
                )
                return True
        previous = frame
        if time.time() - started > float(timeout):
            print(
                f"[grid] page still changing after {timeout:g}s; recording "
                "anyway — the first frames may be missing scenery",
                flush=True,
            )
            return False
        time.sleep(poll)


class _FrameSink(Protocol):
    def write(self, frame, /) -> None: ...

    def close(self) -> None: ...


def record_frames(
    built: dict,
    client,
    make_sink: Callable[[tuple[int, int]], _FrameSink],
    *,
    playheads,
    size: tuple[int, int],
    loop: str = LOOP_ALL,
    log_every: int = RECORD_LOG_EVERY,
    timeout: float | None = None,
    camera_at=None,
) -> tuple[int, tuple[int, int]]:
    """Step the playhead, render each frame in the browser, write it out.

    No wall clock anywhere: the playhead moves one position per frame, the
    grid is posed, and only then is the browser asked for a picture. The
    file therefore comes out smooth at ``--record-fps`` however slowly the
    live scene updates.

    The first render decides the real frame size: a headless Chrome renders
    offscreen and returns exactly what it was asked for, but somebody
    else's browser may not, so ``make_sink`` is called with the size that
    actually came back rather than the one that was asked for.

    Args:
        built: What :func:`build_grid` returned.
        client: Anything with ``get_render(height, width, ...)``.
        make_sink: Called once with ``(width, height)``; must return an
            object with ``write(frame)`` and ``close()``.
        playheads: The playhead positions to record, in order (see
            :func:`record_schedule`).
        size: The requested ``(width, height)``.
        loop: ``all`` or ``each``.
        log_every: Print progress every this many frames (0 = never).
        timeout: Seconds to wait for one frame (None = for ever).
        camera_at: Called with the frame number before each render, to put
            the camera somewhere (a path); None leaves the camera alone.

    Returns:
        ``(frames written, the size they were written at)``.

    Raises:
        RuntimeError: The browser changed the frame size mid-recording
            (the window was resized), which no encoder can take.
    """
    width, height = int(size[0]), int(size[1])
    playheads = list(playheads)
    sink: _FrameSink | None = None
    actual = (width, height)
    written = 0
    started = time.time()
    try:
        for index, playhead in enumerate(playheads):
            pose_grid(built, int(playhead), loop)
            if camera_at is not None:
                camera_at(index)
            frame = video.rgb_frame(
                client.get_render(height, width, timeout=timeout)
                if timeout is not None
                else client.get_render(height, width)
            )
            got = (int(frame.shape[1]), int(frame.shape[0]))
            if sink is None:
                actual = got
                if got != (width, height):
                    print(
                        f"[grid] the browser returned {got[0]}x{got[1]}, not "
                        f"the requested {width}x{height}; recording at "
                        f"{got[0]}x{got[1]}",
                        flush=True,
                    )
                sink = make_sink(actual)
            elif got != actual:
                raise RuntimeError(
                    f"frame {index} came back {got[0]}x{got[1]} after "
                    f"{actual[0]}x{actual[1]}: do not resize the browser "
                    "window while recording"
                )
            sink.write(frame)
            written += 1
            if log_every and written % int(log_every) == 0:
                rate = written / max(1e-6, time.time() - started)
                print(
                    f"[grid] recorded {written}/{len(playheads)} frames "
                    f"({rate:.1f} fps captured)",
                    flush=True,
                )
    finally:
        if sink is not None:
            sink.close()
    return written, actual


def record_video(
    built: dict,
    client,
    path,
    *,
    size: tuple[int, int],
    fps: float,
    frames: int = 0,
    loop: str = LOOP_ALL,
    crf: str = video.RECORD_CRF,
    camera_at=None,
) -> tuple[int, tuple[int, int]]:
    """Record the grid to an mp4 through ffmpeg.

    Args:
        built: What :func:`build_grid` returned.
        client: The browser that renders.
        path: Destination file.
        size: Requested ``(width, height)``.
        fps: Frame rate of the file.
        frames: How many frames, or 0 for one pass over the longest demo.
        loop: ``all`` or ``each``.
        crf: x264 quality.

    Returns:
        ``(frames written, the size they were written at)``.
    """
    lengths = [entry["qpos"].shape[0] for entry in built["scenes"]]
    playheads = record_schedule(lengths, loop, frames)
    print(
        f"[grid] recording {len(playheads)} frames at {size[0]}x{size[1]} "
        f"to {path} ({fps:g} fps, {len(playheads) / max(1.0, fps):.1f}s of "
        "video)",
        flush=True,
    )
    started = time.time()
    written, actual = record_frames(
        built,
        client,
        lambda real: video.VideoSink(path, real, fps, crf),
        playheads=playheads,
        size=size,
        loop=loop,
        camera_at=camera_at,
    )
    elapsed = time.time() - started
    print(
        f"[grid] wrote {written} frames at {actual[0]}x{actual[1]} to {path} "
        f"in {elapsed:.1f}s ({written / max(1e-6, elapsed):.1f} fps captured)",
        flush=True,
    )
    return written, actual


def save_screenshot(
    built: dict, client, path, *, size: tuple[int, int], frame: int = 0, loop=LOOP_ALL
) -> tuple[int, int]:
    """Pose the grid at one frame and save a single render as a PNG.

    Args:
        built: What :func:`build_grid` returned.
        client: The browser that renders.
        path: Destination file.
        size: Requested ``(width, height)``.
        frame: The playhead position to capture.
        loop: ``all`` or ``each``.

    Returns:
        The ``(width, height)`` that was saved.
    """
    from PIL import Image

    captured: list = []

    class _Once:
        """A sink that keeps the one frame and saves it on close."""

        def __init__(self, real):
            self.real = real

        def write(self, image) -> None:
            captured.append(np.array(image))

        def close(self) -> None:
            return None

    _, actual = record_frames(
        built,
        client,
        _Once,
        playheads=[int(frame)],
        size=size,
        loop=loop,
        log_every=0,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(captured[0]).save(path)
    print(
        f"[grid] wrote {actual[0]}x{actual[1]} screenshot of frame {frame} to {path}",
        flush=True,
    )
    return actual


def grid_keyframes(built: dict, args, aspect: float = 16 / 9):
    """The camera path a run uses, or None when it flies nothing.

    Args:
        built: What :func:`build_grid` returned.
        args: The parsed command line (``path``, ``path_keyframes``,
            ``duration``).
        aspect: Frame width over height, for the overview keyframe's lens.

    Returns:
        The keyframes, or None for ``--path none`` with no file.
    """
    if args.path_keyframes:
        return load_keyframes(args.path_keyframes)
    if str(args.path) == PATH_FLY:
        entry = built["scenes"][0] if built["scenes"] else None
        anchor_point = None
        if entry is not None and entry.get("pelvis", -1) >= 0:
            pelvis = agent_cells.pelvis_position(entry["data"], entry["pelvis"])
            offset = entry["offset"]
            anchor_point = (
                pelvis[0] + offset[0],
                pelvis[1] + offset[1],
                pelvis[2] + offset[2],
            )
        return fly_keyframes(
            built["layout"], float(args.duration), aspect, anchor_point
        )
    return None


def aim_client(client, position, look_at, fov=None) -> None:
    """Put one client's camera exactly where a path says."""
    with contextlib.suppress(Exception):
        client.camera.position = tuple(float(v) for v in position)
        client.camera.look_at = tuple(float(v) for v in look_at)
        if fov is not None:
            client.camera.fov = float(fov)


def path_camera(client, keys, fps: float):
    """A per-frame camera hook that walks a path at ``fps``.

    Args:
        client: The browser that renders.
        keys: The keyframes.
        fps: Frames per second of the recording, so frame ``n`` is at
            ``n / fps`` seconds of the path.

    Returns:
        A callable for :func:`record_frames`' ``camera_at``.
    """
    rate = max(1e-6, float(fps))

    def pose_at(index: int) -> None:
        position, look_at, fov = sample_path(keys, index / rate)
        aim_client(client, position, look_at, fov)

    return pose_at


def recording_client(built: dict, url: str, size: tuple[int, int], args):
    """Find the browser that will render: somebody's, or one we start.

    A human who already has the page open is used as is; otherwise a
    headless Chrome is started on the page, sized to the recording (the
    window is what caps ``get_render``).

    Args:
        built: What :func:`build_grid` returned.
        url: The viser page.
        size: The recording ``(width, height)``.
        args: The parsed command line (``browser``, ``exit_after_seconds``).

    Returns:
        ``(client, browser process)``; the client is None when nobody came.
    """
    server = built["server"]
    binary = video.browser_binary(args.browser)
    bound = float(args.exit_after_seconds)
    if not binary:
        return video.wait_for_client(server, timeout=bound), None
    grace = (
        min(video.BROWSER_GRACE_SECONDS, bound)
        if bound > 0
        else video.BROWSER_GRACE_SECONDS
    )
    client = video.wait_for_client(server, timeout=grace)
    if client is not None:
        return client, None
    process = video.launch_browser(binary, url, video.browser_window(size))
    if process is None:
        return None, None
    wait = bound if bound > 0 else video.BROWSER_START_SECONDS
    client = video.wait_for_client(server, timeout=wait)
    if client is None:
        video.stop_browser(process)
        return None, None
    return client, process


def run_recording(built: dict, args: GridConfig, url: str = "") -> bool:
    """Get a browser, then write the video and/or the screenshot.

    Args:
        built: What :func:`build_grid` returned.
        args: The parsed command line.
        url: The viser page a headless browser is pointed at.

    Returns:
        True when something was written, False when no browser turned up.
    """
    size = video.parse_record_size(args.record_size)
    fps = float(args.record_fps)
    if args.path_preview:
        size = video.parse_record_size(PREVIEW_SIZE)
        fps = PREVIEW_FPS
        print(
            f"[grid] --path-preview: {size[0]}x{size[1]} at {fps:g} fps, "
            "just to check the motion",
            flush=True,
        )
    aspect = float(size[0]) / float(size[1])
    built["aspect"] = aspect
    camera = camera_choice(args.record_camera, len(built["scenes"]))
    keys = grid_keyframes(built, args, aspect)
    if keys is not None and args.path_dump:
        dump_keyframes(args.path_dump, keys)
    frames = int(args.record_frames)
    if keys is not None and frames <= 0:
        frames = max(1, int(round(path_duration(keys) * fps)))
    client, browser = recording_client(built, url, size, args)
    if client is None:
        return False
    try:
        if keys is None:
            aim_camera(built, client, camera, size)
        else:
            start = sample_path(keys, 0.0)
            aim_client(client, *start)
            print(
                f"[grid] flying {len(keys)} keyframes over "
                f"{path_duration(keys):g}s ({frames} frames at {fps:g} fps), "
                f"peak speed {path_speed(keys):.1f} m/s",
                flush=True,
            )
        pose_grid(built, 0, args.loop)
        wait_for_page(client, size)
        hook = path_camera(client, keys, fps) if keys is not None else None
        if args.screenshot:
            if hook is not None:
                hook(int(args.frame))
            save_screenshot(
                built,
                client,
                args.screenshot,
                size=size,
                frame=int(args.frame),
                loop=args.loop,
            )
        if args.record:
            record_video(
                built,
                client,
                args.record,
                size=size,
                fps=fps,
                frames=frames,
                loop=args.loop,
                camera_at=hook,
            )
    finally:
        video.stop_browser(browser)
    return True


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


@dataclass
class GridConfig:
    """One human demonstration of every published task, side by side.

    A looping viser scene, for screenshots and screen recordings.
    """

    tasks: str = ""
    """Comma-separated task names (default: every task the dataset publishes,
    as bigym-download --list shows them)."""
    columns: int = DEFAULT_COLUMNS
    """Scenes per row (5: 20 tasks read as 4 rows of 5)."""
    clearance: float = DEFAULT_CLEARANCE
    """Metres of floor between neighbouring scenes' measured footprints. The
    grid is packed on what each scene really occupies, so a bare reach-target
    stands as close to its neighbour as a kitchen allows."""
    gap: float = 0.0
    """Override the packing with one uniform pitch, in metres: every scene at
    its origin, GAP apart, whatever its footprint."""
    states: str = ""
    """Build the grid from a file written by --export-states instead of the
    dataset (nothing is fetched from the Hub)."""
    export_states: str = ""
    """Write the chosen demonstrations (full_qpos + seeds) to one compressed
    npz and exit, so a machine without the dataset can render the grid with
    --states."""
    episode: str = DEFAULT_EPISODE
    """Which demonstration of each task: median (the one of median length) or
    a dataset episode index."""
    fps: float = DEFAULT_FPS
    """How fast demo time runs (50 = real time, the rate the demonstrations
    were recorded at). A grid too big to push that fast drops frames rather
    than playing slowly; the measured render rate is logged."""
    loop: Literal["all", "each"] = LOOP_ALL
    """all: every scene restarts together when the longest demo ends, holding
    its last frame meanwhile; each: every scene loops on its own length."""
    labels: bool = True
    """Draw the task name on a plate above each robot's pelvis."""
    tint: bool = False
    """Give each scene's robot a colour from the palette (display only); off
    by default, so the robots keep their real colours."""
    sky: Literal["black", "grey", "transparent"] = "black"
    """Background: black (for slides and video), grey (the environment map,
    dimmed) or transparent (for compositing)."""
    port: int = DEFAULT_PORT
    """viser port."""
    camera: str = CAMERA_OVERVIEW
    """overview (a pose that frames the whole grid from the -x side) or a
    scene index to fly to."""
    hide_gui: bool = True
    """Collapse viser's sidebar and leave only the Playback folder, so a
    recording is clean."""
    exit_after_seconds: float = 0.0
    """Serve for this many seconds and exit (0 = until Ctrl-C); headless smoke
    tests use it, and it also bounds the wait for a browser when recording."""
    record: str = ""
    """Record the grid to this H.264 mp4 and exit. A recording steps the
    playhead one frame at a time and asks the browser for each picture, so the
    file is smooth at --record-fps whatever the live rate is; it starts once a
    client is connected to the printed URL."""
    record_size: str = video.RECORD_SIZE
    """Frame size WxH of the recording and the screenshot. The headless
    browser renders offscreen, so this is not capped by its window; a human's
    browser may return its own size, which is then logged and used."""
    record_fps: float = DEFAULT_FPS
    """Frame rate written into the file (50 plays the demonstrations at their
    recorded speed)."""
    record_frames: int = 0
    """How many frames to record (0: one pass over the grid, the length of the
    longest demonstration)."""
    record_camera: str = CAMERA_OVERVIEW
    """Where the recording camera stands: overview or a scene index."""
    browser: str = video.BROWSER_HEADLESS
    """Who renders the frames: headless starts a headless Chrome/Chromium on
    the page when nobody is connected after 3s, none waits for you to open the
    URL yourself, or the path of a Chrome/Chromium binary."""
    record_then_stay: bool = False
    """Keep serving the live scene after the recording instead of exiting."""
    path: Literal["none", "fly"] = PATH_NONE
    """Fly the camera through the grid: fly walks one scene, both aisles and
    pulls back to the overview; the live viewer follows it too (none: a fixed
    camera)."""
    duration: float = DEFAULT_DURATION
    """Seconds the camera path runs; --record-frames follows from it and
    --record-fps."""
    path_keyframes: str = ""
    """Fly this hand-written path (FILE.json) instead of a preset: a list of
    {"t": seconds, "position": [x,y,z], "look_at": [x,y,z], "fov": degrees}."""
    path_dump: str = ""
    """Write the preset's keyframes to JSON, to edit and feed back in with
    --path-keyframes."""
    path_preview: bool = False
    """Render the path small and slow (480x270 at 5 fps) to check the motion
    before a full capture."""
    screenshot: str = ""
    """Save one --record-size frame as a PNG (see --frame)."""
    frame: int = 0
    """Which playhead position --screenshot captures."""


def parse_args(argv: list[str] | None = None) -> GridConfig:
    """Parse the grid's command line."""
    return tyro.cli(GridConfig, args=argv, prog="scripts/demo_grid.py")


def main() -> None:
    """Open one looping scene per published task on a viser server."""
    args = parse_args()

    wanted = [name.strip() for name in str(args.tasks).split(",") if name.strip()]
    if args.states:
        # Everything comes out of the file: no Hub, no dataset, no network.
        demos = load_states(args.states, wanted or None)
        tasks = [demo.task for demo in demos]
    else:
        from bigym.loco.demos import hub

        tasks = task_names(args.tasks, hub.available_tasks)
        print(
            f"[grid] {len(tasks)} task(s) from {hub.dataset_repo()}: "
            f"{', '.join(tasks)}",
            flush=True,
        )
        demos = read_demos(tasks, args.episode)
    if args.export_states:
        export_states(args.export_states, demos, args.episode)
        return
    camera = camera_choice(args.camera, len(tasks))
    spacing = (
        f"uniform pitch {args.gap:g} m"
        if args.gap and args.gap > 0.0
        else f"clearance {args.clearance:g} m"
    )
    rows, columns = agent_cells.grid_shape(len(tasks), columns_layout(args.columns))
    print(
        f"[grid] {len(tasks)} scenes as {columns} columns x {rows} rows · "
        f"{spacing} · episode {args.episode} · {args.fps:g} fps · "
        f"loop {args.loop} · labels {'on' if args.labels else 'off'} · "
        f"tint {'on' if args.tint else 'off'} · sky {args.sky}",
        flush=True,
    )
    rates = {d.control_seconds for d in demos if d.control_seconds}
    if rates:
        print(
            "[grid] demos recorded at "
            + ", ".join(f"{1.0 / s:g} Hz" for s in sorted(rates))
            + f" · playing at {args.fps:g} fps",
            flush=True,
        )
    started = time.time()
    size = video.parse_record_size(
        PREVIEW_SIZE if args.path_preview else args.record_size
    )
    built = build_grid(
        demos,
        columns=args.columns,
        clearance=args.clearance,
        gap=args.gap,
        port=args.port,
        labels=bool(args.labels),
        tint=bool(args.tint),
        sky=args.sky,
        aspect=float(size[0]) / float(size[1]),
    )
    print(
        f"[grid] built {len(built['scenes'])} scenes in {time.time() - started:.1f}s",
        flush=True,
    )
    url = f"http://localhost:{args.port}"
    print(f"[grid] open {url} (ssh -L {args.port}:localhost:{args.port})", flush=True)
    if args.record or args.screenshot:
        # The recording needs the sidebar out of the picture whatever
        # --hide-gui says for the live scene; the panel is part of the shot.
        if args.hide_gui:
            video.hide_gui_chrome(built["server"])
        wrote = run_recording(built, args, url)
        if not args.record_then_stay:
            close_grid(built)
            if not wrote:
                raise SystemExit("[grid] nothing was recorded: no browser connected")
            return
    run_grid(
        built,
        fps=args.fps,
        loop=args.loop,
        camera=camera,
        hide_gui=bool(args.hide_gui),
        exit_after=float(args.exit_after_seconds),
        keys=grid_keyframes(built, args, built.get("aspect", 16 / 9)),
    )


if __name__ == "__main__":
    main()
