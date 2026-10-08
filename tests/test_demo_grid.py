"""The all-tasks demo grid: the whole benchmark in one looping viser scene.

The fast tests cover what the tool decides before it touches MuJoCo: where
each task stands in the grid, what its plate says, which demonstration of a
task is shown, how the two loop modes map a playhead onto each scene's
frames, and the command line. The slow test builds the real thing — two
cached tasks, two environments in one viser server, each under its own node
prefix — and checks that both scenes are posed from the dataset's stored
``full_qpos`` (not from the reset pose) and that the log says so.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

from bigym.vr.viewer import agent_cells


def load_script():
    """Import ``scripts/demo_grid.py``, which is not part of the package."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "demo_grid.py"
    spec = importlib.util.spec_from_file_location("demo_grid", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered first: its dataclasses look their own module up by name.
    sys.modules["demo_grid"] = module
    spec.loader.exec_module(module)
    return module


demo_grid = load_script()

# Two tasks whose demonstrations are small and are already in the Hub cache
# of any machine that has run the viewer.
TASKS = ("reach_target_single", "move_plate")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_twenty_tasks_fill_four_rows_of_five():
    """The default grid is 5 columns x 4 rows, filled left to right."""
    places = agent_cells.slot_grid(20, demo_grid.columns_layout(5))
    assert places[0] == (0, 0)
    assert places[4] == (0, 4)
    assert places[5] == (1, 0)
    assert places[-1] == (3, 4)
    assert agent_cells.grid_shape(20, demo_grid.columns_layout(5)) == (4, 5)
    note = agent_cells.grid_note(20, 4.0, demo_grid.columns_layout(5), noun="scene")
    assert note == "20 scenes as 5 columns x 4 rows, gap 4 m"


def test_a_column_steps_along_y_and_a_row_back_along_x():
    """Compare mode's placement: +y per column, -x per row, origin first."""
    offsets = agent_cells.grid_offsets(7, 4.0, demo_grid.columns_layout(5))
    assert offsets[0] == (0.0, 0.0, 0.0)
    assert offsets[1] == (0.0, 4.0, 0.0)
    assert offsets[4] == (0.0, 16.0, 0.0)
    assert offsets[5] == (-4.0, 0.0, 0.0)
    assert offsets[6] == (-4.0, 4.0, 0.0)
    # Same helper the compare scene places its slots with.
    assert offsets[6] == agent_cells.slot_offset(1, 1, 4.0)


def test_a_short_grid_never_has_empty_columns():
    """Three tasks in a five-column grid are one row of three."""
    assert agent_cells.grid_shape(3, demo_grid.columns_layout(5)) == (1, 3)
    assert agent_cells.slot_grid(3, demo_grid.columns_layout(5)) == [
        (0, 0),
        (0, 1),
        (0, 2),
    ]
    assert agent_cells.grid_offsets(1, 4.0, demo_grid.columns_layout(5)) == [
        (0.0, 0.0, 0.0)
    ]


def test_packing_leaves_exactly_the_clearance_between_footprints():
    """Scenes are placed on what they measure, not on a uniform pitch."""
    wide = demo_grid.Footprint((-0.9, -2.0), (2.7, 1.3))  # a kitchen
    narrow = demo_grid.Footprint((-0.3, -0.4), (0.6, 0.4))  # a reach target
    layout = demo_grid.pack_layout([wide, narrow, narrow, wide], 2, 0.6)
    # Column 0 is as wide as the kitchen, column 1 as wide as the widest
    # of its two scenes; the floor between them is the clearance.
    assert layout.columns[1][0] - layout.columns[0][1] == pytest.approx(0.6)
    assert layout.rows[0][0] - layout.rows[1][1] == pytest.approx(0.6)
    # Each scene is centred in its cell on its footprint, not its origin.
    for offset, box, (row, column) in zip(
        layout.offsets,
        layout.footprints,
        layout.places,
        strict=True,
    ):
        middle = box.centre
        cell_x = 0.5 * (layout.rows[row][0] + layout.rows[row][1])
        cell_y = 0.5 * (layout.columns[column][0] + layout.columns[column][1])
        assert middle[0] + offset[0] == pytest.approx(cell_x)
        assert middle[1] + offset[1] == pytest.approx(cell_y)
    # A packed grid of kitchens is tighter than a uniform 4 m pitch.
    packed = demo_grid.pack_layout([wide] * 20, 5, 0.6)
    uniform = demo_grid.uniform_layout([wide] * 20, 5, 4.0)
    assert packed.extent[1] < uniform.extent[1]


def test_the_uniform_gap_is_still_available():
    """``--gap`` puts every scene back on its origin, GAP apart."""
    box = demo_grid.Footprint((-0.9, -2.0), (2.7, 1.3))
    layout = demo_grid.uniform_layout([box] * 6, 3, 4.0)
    assert layout.offsets[0] == (0.0, 0.0, 0.0)
    assert layout.offsets[1] == (0.0, 4.0, 0.0)
    assert layout.offsets[3] == (-4.0, 0.0, 0.0)


def test_the_overview_camera_frames_the_whole_grid():
    """It stands off the -x side, left of every scene and above them all."""
    offsets = agent_cells.grid_offsets(
        20, demo_grid.DEFAULT_GAP, demo_grid.columns_layout(5)
    )
    position, look_at = agent_cells.compare_camera_pose(offsets)
    assert position[0] < min(o[0] for o in offsets)
    assert position[1] == pytest.approx(0.5 * (0.0 + 16.0))
    # High enough that the near row cannot hide the far one.
    assert position[2] > 2.0
    assert look_at[1] == pytest.approx(position[1])


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def test_a_task_name_reads_as_a_sentence():
    """``move_plate`` is drawn as ``Move plate``."""
    assert demo_grid.label_text("move_plate") == "Move plate"
    assert demo_grid.label_text("reach_target_single") == "Reach target single"
    assert demo_grid.label_text("") == ""


def test_every_scene_of_a_full_grid_gets_its_own_colour():
    """The palette holds one colour per task of a twenty-task grid."""
    assert len(set(demo_grid.GRID_COLOURS)) == 20
    assert len({demo_grid.scene_colour(i) for i in range(20)}) == 20
    # The tint only ever repaints, never darkens: its peak channel is 1.
    for index in range(20):
        assert max(demo_grid.scene_tint(index)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Which demonstration
# ---------------------------------------------------------------------------


def _episodes(lengths):
    """Fake dataset episodes with the given lengths."""
    from bigym.loco.agent.demo_video import DemoEpisode

    return tuple(
        DemoEpisode(index=i, source_file=f"ep{i}.npz", seed=100 + i, length=length)
        for i, length in enumerate(lengths)
    )


def test_the_median_length_demonstration_is_the_one_shown():
    """``--episode median`` is deterministic and picks a typical demo."""
    from bigym.loco.agent.demo_video import pick_episodes

    episodes = _episodes([90, 10, 50, 70, 30])
    chosen = pick_episodes(episodes, demo_grid.DEFAULT_EPISODE, 1)
    assert [e.length for e in chosen] == [50]
    assert chosen[0].seed == 102


def test_an_explicit_episode_index_is_taken_as_written():
    """``--episode 3`` shows episode 3 of every task."""
    from bigym.loco.agent.demo_video import pick_episodes

    episodes = _episodes([90, 10, 50, 70, 30])
    assert pick_episodes(episodes, "3", 1)[0].index == 3
    with pytest.raises(ValueError):
        pick_episodes(episodes, "9", 1)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def test_loop_all_holds_the_last_frame_and_restarts_together():
    """Every scene waits for the longest demo, then they all start again."""
    short = [demo_grid.loop_frame(p, 5, 10, "all") for p in range(12)]
    assert short == [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 0, 1]
    longest = [demo_grid.loop_frame(p, 10, 10, "all") for p in range(12)]
    assert longest == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1]
    assert demo_grid.loop_length([5, 10, 7], "all") == 10


def test_loop_each_wraps_every_scene_on_its_own_length():
    """No holding: a short demo has played twice before a long one ends."""
    short = [demo_grid.loop_frame(p, 5, 10, "each") for p in range(12)]
    assert short == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]
    assert demo_grid.loop_frame(7, 3, 10, "each") == 1


def test_a_one_frame_demo_never_indexes_past_its_end():
    """Degenerate lengths stay inside the array in both modes."""
    for mode in demo_grid.LOOP_MODES:
        assert demo_grid.loop_frame(99, 1, 10, mode) == 0
        assert demo_grid.loop_frame(0, 0, 0, mode) == 0


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def test_labels_and_the_gui_can_be_turned_off():
    """``--no-labels`` and ``--show-gui`` are the negated options."""
    args = demo_grid.parse_args(["--no-labels", "--no-hide-gui", "--tint"])
    assert args.labels is False
    assert args.hide_gui is False
    assert args.tint is True


def test_tasks_default_to_everything_the_dataset_publishes():
    """``--tasks`` wins; without it the Hub's published list is used."""
    assert demo_grid.task_names("", lambda: ("a", "b")) == ["a", "b"]
    assert demo_grid.task_names("b, a ,b", lambda: ("a",)) == ["b", "a"]
    with pytest.raises(SystemExit):
        demo_grid.task_names("", lambda: ())
    with pytest.raises(SystemExit):
        demo_grid.task_names(" , ", lambda: ("a",))


def test_the_camera_is_the_overview_or_one_scene():
    """``--camera`` takes ``overview`` or an index that exists."""
    assert demo_grid.camera_choice("overview", 3) == -1
    assert demo_grid.camera_choice("2", 3) == 2
    with pytest.raises(SystemExit):
        demo_grid.camera_choice("3", 3)
    with pytest.raises(SystemExit):
        demo_grid.camera_choice("left", 3)


# ---------------------------------------------------------------------------
# The camera path
# ---------------------------------------------------------------------------


def _kitchen_layout(count: int = 20, columns: int = 5, clearance: float = 0.6):
    """A packed grid of identical kitchen-sized footprints."""
    box = demo_grid.Footprint((-0.9, -2.0), (2.7, 1.3))
    return demo_grid.pack_layout([box] * count, columns, clearance)


def test_an_aisle_runs_between_two_rows_of_footprints():
    """The flight's aisles are real gaps, not guesses from the pitch."""
    layout = _kitchen_layout()
    for row in range(len(layout.rows) - 1):
        aisle = layout.aisle(row)
        near, behind = layout.rows[row], layout.rows[row + 1]
        assert behind[1] < aisle < near[0]  # strictly between the two rows
        assert aisle == pytest.approx(0.5 * (near[0] + behind[1]))
    # Even the last row has somewhere to stand, in front of the grid.
    assert layout.aisle(len(layout.rows) - 1) < layout.rows[-1][0]


def test_the_flight_never_leaves_its_aisle():
    """A dolly segment stays on the aisle centre line, to the millimetre."""
    layout = _kitchen_layout()
    keys = demo_grid.fly_keyframes(layout, 30.0, 16 / 9)
    front, back = layout.aisle(0), layout.aisle(len(layout.rows) - 2)
    worst_front = worst_back = 0.0
    for step in range(3001):
        t = step * 0.01
        position, _, _ = demo_grid.sample_path(keys, t)
        if 3.0 <= t <= 14.0:
            worst_front = max(worst_front, abs(position[0] - front))
        if 18.5 <= t <= 22.0:
            worst_back = max(worst_back, abs(position[0] - back))
    assert worst_front < 0.05
    assert worst_back < 0.05


def test_the_flight_keeps_to_its_schedule():
    """The preset's beats land where the storyboard says, at any duration."""
    layout = _kitchen_layout()
    keys = demo_grid.fly_keyframes(layout, 30.0, 16 / 9)
    assert demo_grid.path_duration(keys) == pytest.approx(30.0)
    times = [key.t for key in keys]
    assert times == sorted(times)
    assert times[0] == 0.0
    # The storyboard's beats: the close shot ends at 3 s, the first dolly
    # at 14 s, the return at 22 s, and the pull-back lands before 30.
    assert times[1] == pytest.approx(3.0, abs=0.1)
    assert min(abs(t - 14.0) for t in times) < 0.1
    assert min(abs(t - 22.0) for t in times) < 0.1
    assert times[-2] == pytest.approx(28.0, abs=0.1)
    # It ends parked on the overview, which is where --camera overview is.
    overview = demo_grid.overview_pose(layout, 16 / 9)
    last = demo_grid.sample_path(keys, 30.0)
    assert last[0] == pytest.approx(overview[0])
    assert last[1] == pytest.approx(overview[1])
    assert last[2] == pytest.approx(overview[2])
    # Half the duration is half of every beat.
    short = demo_grid.fly_keyframes(layout, 15.0, 16 / 9)
    assert demo_grid.path_duration(short) == pytest.approx(15.0)
    assert [key.t for key in short] == pytest.approx([t / 2.0 for t in times])


def test_the_path_is_continuous():
    """No jump between frames: the spline never teleports the camera."""
    layout = _kitchen_layout()
    keys = demo_grid.fly_keyframes(layout, 30.0, 16 / 9)
    fastest = demo_grid.path_speed(keys)
    assert 0.1 < fastest < 20.0
    dt = 1.0 / 50.0
    previous = np.asarray(demo_grid.sample_path(keys, 0.0)[0])
    for step in range(1, int(30.0 / dt) + 1):
        here = np.asarray(demo_grid.sample_path(keys, step * dt)[0])
        assert float(np.linalg.norm(here - previous)) <= fastest * dt * 1.05
        previous = here
    # It starts and ends at rest (the eased ends).
    assert demo_grid.sample_path(keys, 0.0)[0] == pytest.approx(
        demo_grid.sample_path(keys, 0.02)[0], abs=0.02
    )
    assert demo_grid.sample_path(keys, 29.98)[0] == pytest.approx(
        demo_grid.sample_path(keys, 30.0)[0], abs=0.02
    )


def test_keyframes_survive_a_json_round_trip(tmp_path):
    """--path-dump writes what --path-keyframes reads."""
    keys = demo_grid.fly_keyframes(_kitchen_layout(), 30.0, 16 / 9)
    path = tmp_path / "flight.json"
    demo_grid.dump_keyframes(path, keys)
    back = demo_grid.load_keyframes(path)
    assert len(back) == len(keys)
    for before, after in zip(keys, back, strict=True):
        assert after.t == pytest.approx(before.t, abs=1e-3)
        assert after.position == pytest.approx(before.position, abs=1e-3)
        assert after.look_at == pytest.approx(before.look_at, abs=1e-3)
        assert after.fov == pytest.approx(before.fov, abs=1e-4)
        assert after.ease == before.ease
    # A hand-written file works too, and degrees are what people type.
    (tmp_path / "hand.json").write_text(
        '[{"t": 0, "position": [0,0,2], "look_at": [1,0,1], "fov": 45},'
        ' {"t": 2, "position": [0,4,2], "look_at": [1,4,1], "fov": 60}]'
    )
    hand = demo_grid.load_keyframes(tmp_path / "hand.json")
    assert len(hand) == 2
    assert hand[0].fov == pytest.approx(np.radians(45))
    assert demo_grid.sample_path(hand, 1.0)[0][1] == pytest.approx(2.0, abs=0.3)
    with pytest.raises(SystemExit):
        demo_grid.load_keyframes(tmp_path / "missing.json")


def test_the_overview_fills_the_frame():
    """The grid's box covers most of the picture, centred, not a band."""
    layout = _kitchen_layout()
    position, look_at, fov = demo_grid.overview_pose(layout, 16 / 9)
    # It looks at the middle of the grid from the -x side, above it.
    assert look_at == pytest.approx(layout.centre)
    assert position[0] < layout.bounds[0][0]
    assert position[2] > 2.0
    # The box covers ~80% of the frame height from there.
    half = layout.half()
    distance = float(np.linalg.norm(np.asarray(position) - np.asarray(look_at)))
    near = distance - half[0] * np.cos(demo_grid.OVERVIEW_TILT)
    upright = half[0] * np.sin(demo_grid.OVERVIEW_TILT) + half[2] * np.cos(
        demo_grid.OVERVIEW_TILT
    )
    covered = 2.0 * np.arctan(upright / near) / fov
    assert 0.55 < covered <= 1.0


# ---------------------------------------------------------------------------
# States export
# ---------------------------------------------------------------------------


def test_states_survive_an_export_and_reload(tmp_path):
    """--export-states holds everything --states needs to draw the grid."""
    demos = [
        demo_grid.GridDemo(
            task="move_plate",
            qpos=np.arange(30, dtype=np.float64).reshape(6, 5),
            seed=87,
            index=28,
            source_file="a.npz",
            count=60,
            control_seconds=0.02,
        ),
        demo_grid.GridDemo(
            task="pick_box",
            qpos=np.zeros((4, 5)),
            seed=145,
            index=36,
            source_file="b.npz",
        ),
    ]
    path = tmp_path / "states.npz"
    demo_grid.export_states(path, demos, "median")
    assert path.exists()

    back = demo_grid.load_states(path)
    assert [d.task for d in back] == ["move_plate", "pick_box"]
    assert [d.seed for d in back] == [87, 145]
    assert [d.index for d in back] == [28, 36]
    assert back[0].source_file == "a.npz"
    assert back[0].control_seconds == pytest.approx(0.02)
    # float32 on disk, and the states come back to seven digits.
    assert back[0].qpos.shape == (6, 5)
    assert back[0].qpos == pytest.approx(demos[0].qpos, rel=1e-6)

    # --tasks picks and orders them, and a missing one is an error.
    picked = demo_grid.load_states(path, ["pick_box"])
    assert [d.task for d in picked] == ["pick_box"]
    with pytest.raises(SystemExit):
        demo_grid.load_states(path, ["dishwasher_close"])
    with pytest.raises(SystemExit):
        demo_grid.load_states(tmp_path / "nothing.npz")


# ---------------------------------------------------------------------------
# Offline recording
# ---------------------------------------------------------------------------


class _FakeServer:
    """Just enough viser server for :func:`demo_grid.pose_grid`."""

    @contextlib.contextmanager
    def atomic(self):
        """The no-op batching context the real server provides."""
        yield


class _FakeScene:
    """Counts the pushes mjviser would send to the browser."""

    def __init__(self):
        self.updates = 0

    def update_from_mjdata(self, data):
        """Record that the scene was pushed."""
        self.updates += 1


class _FakeData:
    """A ``MjData`` stand-in with nothing but ``qpos``."""

    def __init__(self, nq: int):
        self.qpos = np.zeros(nq)


class _FakeMujoco:
    """The ``mujoco`` module, reduced to ``mj_forward``."""

    def __init__(self):
        self.forwards = 0

    def mj_forward(self, model, data):
        """Count a forward call."""
        self.forwards += 1


class _FakeClient:
    """A browser that returns a flat frame of whatever size it feels like."""

    def __init__(self, size=None, channels: int = 3, resize_at: int = -1):
        self.calls: list[tuple[int, int]] = []
        self.size = size
        self.channels = channels
        self.resize_at = resize_at

    def get_render(self, height: int, width: int, **kwargs):
        """Return one frame, as ``ClientHandle.get_render`` would."""
        self.calls.append((height, width))
        shape = self.size or (width, height)
        if 0 <= self.resize_at <= len(self.calls) - 1:
            shape = (shape[0] // 2, shape[1] // 2)
        return np.full((shape[1], shape[0], self.channels), 9, dtype=np.uint8)


class _StubSink:
    """An ffmpeg pipe stand-in that keeps the bytes it was handed."""

    def __init__(self, size):
        self.size = size
        self.data = bytearray()
        self.closed = False

    def write(self, frame) -> None:
        """Append one frame's raw RGB bytes."""
        self.data += np.ascontiguousarray(frame, dtype=np.uint8).tobytes()

    def close(self) -> None:
        """Mark the pipe closed (the real one waits for ffmpeg)."""
        self.closed = True


class _FakeInner:
    """The task hooks run_task_hook calls, doing nothing."""

    class _StepCache:
        def clean(self):
            pass

    _step_cache = _StepCache()

    def _on_step(self):
        pass


def _fake_grid(lengths=(4, 7), nq: int = 5):
    """A ``build_grid`` result made of stubs, poseable without MuJoCo."""
    scenes = []
    for index, length in enumerate(lengths):
        qpos = np.arange(length * nq, dtype=np.float64).reshape(length, nq) + index
        scenes.append(
            {
                "task": f"task_{index}",
                "label": f"Task {index}",
                "qpos": qpos,
                "data": _FakeData(nq),
                "model": None,
                "scene": _FakeScene(),
                "inner": _FakeInner(),
                "targets": [],
                "billboard": None,
                "offset": (0.0, 4.0 * index, 0.0),
                "tolerance": 0.1,
            }
        )
    return {
        "server": _FakeServer(),
        "scenes": scenes,
        "offsets": [entry["offset"] for entry in scenes],
        "mujoco": _FakeMujoco(),
    }


def test_a_recording_is_one_pass_over_the_longest_demonstration():
    """No `--record-frames`: every frame of the longest demo, once."""
    assert demo_grid.record_schedule([5, 10, 7], "all") == list(range(10))
    assert demo_grid.record_schedule([5, 10], "all", 3) == [0, 1, 2]
    # A count beyond one pass simply keeps going; loop_frame wraps it.
    assert demo_grid.record_schedule([4], "all", 6) == list(range(6))
    assert demo_grid.record_schedule([], "all") == [0]


class _LoadingClient:
    """A browser whose picture changes while its assets are still loading."""

    def __init__(self, changes: int = 3, forever: bool = False):
        self.renders = 0
        self.changes = changes
        self.forever = forever

    def get_render(self, height: int, width: int, **kwargs):
        """A flat frame whose value stops moving once the page is loaded."""
        self.renders += 1
        step = self.renders if self.forever else min(self.renders, self.changes)
        # Wraps, so a page that never settles never repeats a picture.
        return np.full((height, width, 3), (17 * step) % 251, dtype=np.uint8)


def test_the_recording_lens_frames_the_compare_slots_not_a_kitchen():
    """compare_box pads slots by a table's reach; grid_fov takes that box."""
    offsets = [(0.0, 0.0, 0.0), (0.0, 2.5, 0.0), (-2.5, 0.0, 0.0), (-2.5, 2.5, 0.0)]
    centre, half = agent_cells.compare_box(offsets)
    assert centre == pytest.approx((-1.25 + 0.4, 1.25, 0.9))
    assert half == pytest.approx((1.25 + 1.3, 1.25 + 1.3, 1.0))
    home = agent_cells.compare_camera_pose(offsets)
    tight = demo_grid.grid_fov(offsets, *home, 16 / 9, box=(centre, half))
    loose = demo_grid.grid_fov(offsets, *home, 16 / 9)
    assert 0.0 < tight < loose


def test_the_overview_lens_narrows_to_fit_the_grid():
    """viser's ~80 degree default leaves the grid in a band; fit it instead."""
    offsets = agent_cells.grid_offsets(
        20, demo_grid.DEFAULT_GAP, demo_grid.columns_layout(5)
    )
    pose = agent_cells.compare_camera_pose(offsets)
    fov = demo_grid.grid_fov(offsets, pose[0], pose[1], 16 / 9)
    assert 0.2 < fov < 0.9  # radians: a long lens, not a fisheye
    # The grid's own box is what it frames: padded, and centred on it.
    centre, half = demo_grid.grid_box(offsets)
    assert centre[1] == pytest.approx(8.0 + demo_grid.SCENE_CENTRE[1])
    assert half[1] > 8.0 and half[0] > 6.0
    # One scene alone is framed more widely than twenty.
    single = agent_cells.grid_offsets(
        1, demo_grid.DEFAULT_GAP, demo_grid.columns_layout(1)
    )
    alone = demo_grid.grid_fov(single, *agent_cells.compare_camera_pose(single), 16 / 9)
    assert alone > fov
    # A taller frame needs a taller lens for the same grid.
    assert demo_grid.grid_fov(offsets, pose[0], pose[1], 4 / 3) > fov


def test_a_label_is_drawn_and_turned_so_its_text_reads():
    """The grid's plates are the compare viewer's: same drawing, same turn."""
    anchor = (0.0, 0.0, 2.0)
    camera = (-5.0, 0.0, 2.0)
    rotation = _quat_matrix(agent_cells.facing_wxyz(anchor, camera))
    # viser's image plane shows its picture on the local -z side with the
    # top along local -y: the picture looks at the camera, text upright.
    assert rotation[:, 2] == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)
    assert rotation[2, 1] == pytest.approx(-1.0, abs=1e-6)

    # ... and that is what one scene's plate is given.
    built = _fake_grid(lengths=(3,))
    billboard = _FakePlate()
    built["scenes"][0]["billboard"] = billboard
    built["scenes"][0]["pelvis"] = -1
    demo_grid.place_label(built["scenes"][0], camera)
    assert billboard.node.wxyz == pytest.approx(
        agent_cells.facing_wxyz(billboard.node.position, camera)
    )
    # With nobody watching, the plate keeps the direction it was built with.
    billboard.node.wxyz = (1.0, 0.0, 0.0, 0.0)
    demo_grid.place_label(built["scenes"][0], None)
    assert billboard.node.wxyz == (1.0, 0.0, 0.0, 0.0)


class _FakeNode:
    """The image node of a plate: a position and an orientation."""

    def __init__(self):
        self.position = (0.0, 0.0, 0.0)
        self.wxyz = (1.0, 0.0, 0.0, 0.0)


class _FakePlate:
    """A :class:`agent_cells.SlotBillboard` stand-in."""

    def __init__(self):
        self.node = _FakeNode()


def _quat_matrix(wxyz) -> np.ndarray:
    """The rotation matrix of a ``(w, x, y, z)`` quaternion."""
    w, x, y, z = (float(v) for v in wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_the_settle_render_is_cheap_and_keeps_the_aspect():
    """Throwaway renders are small; the recording's shape is preserved."""
    assert demo_grid.settle_size((1920, 1080)) == (640, 360)
    assert demo_grid.settle_size((3840, 2160)) == (640, 360)
    assert demo_grid.settle_size((640, 360)) == (640, 360)
    small = demo_grid.settle_size((320, 240))
    assert small[0] / small[1] == pytest.approx(320 / 240)


def test_a_capture_waits_for_the_page_to_stop_changing():
    """Meshes, plates and the sky load asynchronously; two equal renders end it."""
    client = _LoadingClient(changes=3)
    assert demo_grid.wait_for_page(client, (640, 360), poll=0.0) is True
    # Three renders while it loaded, one more that matched the third.
    assert client.renders == 4


def test_a_page_that_never_settles_is_recorded_anyway():
    """A timeout returns False rather than hanging the recording."""
    client = _LoadingClient(forever=True)
    assert demo_grid.wait_for_page(client, (64, 48), timeout=0.01, poll=0.0) is False
    assert client.renders >= 2


def test_recording_steps_the_playhead_and_fills_the_pipe():
    """Every scheduled frame is posed, rendered and written, in order."""
    built = _fake_grid(lengths=(4, 7))
    client = _FakeClient()
    sinks: list[_StubSink] = []
    playheads = demo_grid.record_schedule([4, 7], "all")
    written, size = demo_grid.record_frames(
        built,
        client,
        lambda real: sinks[-1] if sinks else sinks.append(_StubSink(real)) or sinks[-1],
        playheads=playheads,
        size=(64, 48),
        loop="all",
        log_every=0,
    )
    assert written == 7 and size == (64, 48)
    assert client.calls == [(48, 64)] * 7
    # The pipe got exactly W * H * 3 bytes per frame.
    assert len(sinks[-1].data) == 64 * 48 * 3 * 7
    assert sinks[-1].closed
    # Deterministic stepping: the short demo held its last frame while the
    # long one ran on, and both scenes were pushed once per frame.
    short, long = built["scenes"]
    assert short["data"].qpos == pytest.approx(short["qpos"][3])
    assert long["data"].qpos == pytest.approx(long["qpos"][6])
    assert short["scene"].updates == 7 and long["scene"].updates == 7
    assert built["mujoco"].forwards == 14


def test_the_recording_takes_the_size_the_browser_actually_returned():
    """A window smaller than --record-size caps get_render; the file follows."""
    built = _fake_grid(lengths=(3,))
    client = _FakeClient(size=(640, 360))
    sinks: list[_StubSink] = []

    def make_sink(real):
        sinks.append(_StubSink(real))
        return sinks[-1]

    written, size = demo_grid.record_frames(
        built,
        client,
        make_sink,
        playheads=[0, 1, 2],
        size=(3840, 2160),
        loop="all",
        log_every=0,
    )
    assert written == 3
    assert size == (640, 360)
    assert sinks[0].size == (640, 360)
    assert len(sinks[0].data) == 640 * 360 * 3 * 3
    # It still asked for the size it was told to.
    assert client.calls[0] == (2160, 3840)


def test_a_window_resized_mid_recording_is_an_error():
    """No encoder can take a frame size that changes; say so and stop."""
    built = _fake_grid(lengths=(4,))
    client = _FakeClient(size=(64, 48), resize_at=2)
    sinks: list[_StubSink] = []

    def make_sink(real):
        sinks.append(_StubSink(real))
        return sinks[-1]

    with pytest.raises(RuntimeError, match="resize the browser"):
        demo_grid.record_frames(
            built,
            client,
            make_sink,
            playheads=[0, 1, 2, 3],
            size=(64, 48),
            loop="all",
            log_every=0,
        )
    # The pipe is closed even when the recording breaks off.
    assert sinks[0].closed
    assert len(sinks[0].data) == 64 * 48 * 3 * 2


# ---------------------------------------------------------------------------
# The real thing
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_two_task_grid_is_built_hidden_and_posed_from_the_demos(capsys):
    """Two tasks, two environments, both posed from the stored full_qpos."""
    demos = demo_grid.read_demos(TASKS, demo_grid.DEFAULT_EPISODE)
    assert [d.task for d in demos] == list(TASKS)
    assert all(d.qpos.ndim == 2 and d.length > 1 for d in demos)

    built = demo_grid.build_grid(demos, columns=2, port=8794)
    server = built["server"]
    played = False
    try:
        scenes = built["scenes"]
        assert len(scenes) == 2
        names = set(server.scene._handle_from_node_name)
        for i in range(2):
            assert any(n.startswith(f"/bodies/task{i}/") for n in names)
            assert f"/grid/task{i}/label" in names
            frame = server.scene._handle_from_node_name[f"/bodies/task{i}"]
            assert tuple(frame.position) == pytest.approx(built["offsets"][i])
            assert frame.visible
        # Both scenes sit in one row, clear of each other by --clearance.
        layout = built["layout"]
        assert len(layout.rows) == 1 and len(layout.columns) == 2
        assert layout.columns[1][0] - layout.columns[0][1] == pytest.approx(
            demo_grid.DEFAULT_CLEARANCE, abs=1e-6
        )
        # A reach-target scene really is smaller than a kitchen one.
        assert layout.footprints[0].width < layout.footprints[1].width
        # Both scenes hold frame 0 of their own demonstration, which is not
        # the pose reset() left the robot in.
        for entry in scenes:
            stored = entry["demo"].qpos[0]
            assert entry["data"].qpos[: stored.shape[0]] == pytest.approx(stored)
        # The label says the task, drawn as a plate.
        assert scenes[0]["billboard"].kind == "image"
        assert scenes[0]["label"] == "Reach target single"
        assert scenes[1]["label"] == "Move plate"
        # The tint is off by default, so no geom's colour was repainted.
        # (Only alpha moves: the in-model reach target is made invisible
        # before the mesh bake, and the task's own _on_step paints its
        # default colour — alpha included — back on afterwards.)
        for entry in scenes:
            assert np.array_equal(
                entry["model"].geom_rgba[:, :3], entry["geom_rgba"][:, :3]
            )

        log = capsys.readouterr().out
        assert "[grid] building 1/2 reach_target_single" in log
        assert "[grid] building 2/2 move_plate" in log
        assert "[grid] layout: 2 scenes, 1 rows x 2 columns" in log
        # The placement is measured, not assumed.
        assert "footprint" in log and "[grid] extent:" in log
        assert "[grid] column widths (m):" in log
        hidden = [
            i for i, line in enumerate(log.splitlines()) if "built hidden" in line
        ]
        revealed = [
            i
            for i, line in enumerate(log.splitlines())
            if "scenes revealed together" in line
        ]
        assert len(hidden) == 2 and len(revealed) == 1
        assert max(hidden) < revealed[0]
        assert "posed to frame 0 of the demo (not the reset pose)" in log

        # ... and it plays, on one playhead, until the deadline. run_grid
        # owns the shutdown: it stops the server and closes both envs.
        demo_grid.run_grid(built, fps=25.0, exit_after=10.0)
        played = True
        log = capsys.readouterr().out
        assert "[grid] playing 2 scenes at 25 fps, loop=all, longest demo" in log
        assert "[grid] playback" in log and "fps measured" in log
        assert "--exit-after-seconds 10 elapsed" in log
        # The playhead moved: neither scene is still on frame 0.
        moved = [
            not np.allclose(
                entry["data"].qpos[: entry["demo"].qpos.shape[1]],
                entry["demo"].qpos[0],
            )
            for entry in scenes
        ]
        assert all(moved)
    finally:
        if not played:
            server.stop()
            for entry in built["scenes"]:
                entry["env"].close()
