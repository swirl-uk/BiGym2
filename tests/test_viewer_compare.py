"""Compare mode of ``bigym-view``: slot resolution, labels and the scene.

Compare mode puts up to six policies of ONE task side by side on one seed.
The fast tests build cells in ``tmp_path`` and check that a slot resolves to
the right cell, version and stored episode, that a missing episode becomes a
``bigym-agent replay --no-video`` job, and that the sidebar legend and the 3D
billboard say who ran and how well. The slow test builds the real thing:
four environments of one viser server, each under its own node prefix, laid
out as a 2 x 2 grid (columns along +y, rows back along -x), built hidden,
posed to frame 0 and revealed together, with the per-slot tint kept out of
``model.geom_rgba``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bigym.vr.viewer import agent_cells, view_demos
from tests.fixtures.agent_cell import EVAL_SEEDS, TASK, make_cell, write_batch
from tests.fixtures.viser_gui import FakeServer

SEED = EVAL_SEEDS[0]


def two_sessions(tmp_path: Path) -> list:
    """Two sessions of the same task, one with more versions than the other."""
    make_cell(tmp_path / "runs" / "astra_high_s1" / TASK, 8)
    make_cell(tmp_path / "runs" / "astra_high_s2" / TASK, 7)
    return agent_cells.discover_cells([tmp_path / "runs"])


def test_parse_compare_reads_session_and_version_slots():
    """`--compare "s1:v008,s2:7,s3:last"` becomes three (session, version)."""
    assert agent_cells.parse_compare("s1:v008, s2:7 ,s3:last") == [
        ("s1", "v008"),
        ("s2", "7"),
        ("s3", "last"),
    ]
    # Never more than six scenes, whatever was typed.
    assert len(agent_cells.parse_compare("a:1,b:1,c:1,d:1,e:1,f:1,g:1")) == 6
    with pytest.raises(ValueError, match="session>:<version"):
        agent_cells.parse_compare("s1")
    with pytest.raises(ValueError, match="at least one"):
        agent_cells.parse_compare("  ")


def test_sessions_match_by_name_or_a_unique_substring():
    """`s1` is enough when only one session's name contains it."""
    sessions = ["astra_high_s1", "astra_high_s2"]
    assert agent_cells.match_session(sessions, "astra_high_s1") == "astra_high_s1"
    assert agent_cells.match_session(sessions, "s2") == "astra_high_s2"
    assert agent_cells.match_session(sessions, "astra") is None  # ambiguous
    assert agent_cells.match_session(sessions, "s9") is None


def test_versions_resolve_by_number_or_by_last(tmp_path):
    """`v008`, `8`, `last` and `submission` all name a recorded version."""
    cell = make_cell(tmp_path / "runs" / "s1" / TASK, 8)
    assert agent_cells.resolve_version(cell, "v008") == 8
    assert agent_cells.resolve_version(cell, "3") == 3
    assert agent_cells.resolve_version(cell, "last") == 8
    assert agent_cells.resolve_version(cell, "submission") == 8
    assert agent_cells.resolve_version(cell, "") == 8
    assert agent_cells.resolve_version(cell, "newest") is None


def test_a_hidden_seed_needs_no_replay(tmp_path):
    """Evaluation batches hold every hidden seed, so compare reads them."""
    cells = two_sessions(tmp_path)
    request = agent_cells.resolve_compare(
        cells, TASK, [("s1", "v008"), ("s2", "last")], SEED
    )
    assert request.error == ""
    assert request.missing == []
    assert [s.session for s in request.slots] == ["astra_high_s1", "astra_high_s2"]
    assert [s.version for s in request.slots] == [8, 7]
    assert [s.source for s in request.slots] == ["eval/v008", "eval/v007"]
    episodes = [s.episode for s in request.slots]
    assert all(e is not None and Path(e).name == f"seed{SEED}.npz" for e in episodes)
    assert "eval/v008" in agent_cells.compare_status_html(request)


def test_a_seed_nobody_rolled_out_becomes_a_replay_job(tmp_path):
    """A slot with no episode on the seed is listed in `missing`."""
    cells = two_sessions(tmp_path)
    request = agent_cells.resolve_compare(cells, TASK, [("s1", "v003")], 620042)
    assert [s.version for s in request.slots] == [3]
    assert request.missing == request.slots
    assert request.slots[0].episode is None
    assert "needs a replay" in agent_cells.compare_status_html(request)

    job = agent_cells.AgentJob(
        "replay", request.slots[0].cell, 3, str(request.seed), extra=("--no-video",)
    )
    assert job.command[3:] == [
        "replay",
        str(request.slots[0].cell),
        "--version",
        "3",
        "--seeds",
        "620042",
        "--no-video",
    ]


def test_seed_episode_prefers_the_evaluation_then_replays_then_dev(tmp_path):
    """Where a slot's frames come from, best source first."""
    cell = make_cell(tmp_path / "runs" / "s1" / TASK, 8)
    episode = agent_cells.seed_episode(cell, 8, SEED)
    assert episode is not None and episode.parent == cell / "eval/v008/batch"
    # A version with no evaluation falls through to replays, then to dev.
    write_batch(cell / "replays" / "v003" / "batch", (SEED,))
    episode = agent_cells.seed_episode(cell, 3, SEED)
    assert episode is not None and episode.parent == cell / "replays" / "v003" / "batch"
    write_batch(cell / "dev" / "v004_20260921_010203" / "batch", (SEED,))
    episode = agent_cells.seed_episode(cell, 4, SEED)
    assert episode is not None
    assert episode.parent == cell / "dev" / "v004_20260921_010203" / "batch"
    assert agent_cells.seed_episode(cell, 5, SEED) is None
    assert agent_cells.episode_source(cell, cell / "eval/v008/batch/x.npz") == (
        "eval/v008"
    )


def test_a_bad_session_or_version_is_reported_not_raised(tmp_path):
    """Compare says what it could not resolve instead of dying."""
    cells = two_sessions(tmp_path)
    request = agent_cells.resolve_compare(
        cells, TASK, [("s9", "last"), ("s1", "newest")], SEED
    )
    assert request.slots == []
    assert "no session matches 's9'" in request.error
    assert "no version matches 'newest'" in request.error
    assert "no session matches" in agent_cells.compare_status_html(request)


def test_the_slot_label_names_the_session_version_and_scores(tmp_path):
    """`astra_high_s1 · v008 · submission · train 50% · eval 2%`."""
    cells = two_sessions(tmp_path)
    request = agent_cells.resolve_compare(cells, TASK, [("s1", "v008")], SEED)
    slot = request.slots[0]
    assert slot.label == "astra_high_s1 · v008 · submission · train 50% · eval 2%"
    assert slot.version_name == "v008"

    earlier = agent_cells.fill_slot(
        agent_cells.CompareSlot(
            session="astra_high_s1", task=TASK, cell=slot.cell, version=2
        ),
        SEED,
    )
    assert earlier.label == "astra_high_s1 · v002 · no dev episodes · not evaluated"

    loose = tmp_path / "my_policy.py"
    loose.write_text("def act(obs):\n    return 0\n")
    digest = agent_cells.file_digest(loose)[:8]
    from_file = agent_cells.CompareSlot(
        session="astra_high_s1", task=TASK, cell=slot.cell, policy=str(loose)
    )
    assert from_file.version_name == f"policy_{digest}"


def test_the_status_label_says_the_step_and_then_the_outcome():
    """`step 1599/1599 · ✓` / `... · ✗ · timeout · fell`."""
    assert agent_cells.slot_outcome({"success": 1.0, "termination": "success"}) == "✓"
    assert (
        agent_cells.slot_outcome(
            {"success": 0.0, "termination": "timeout", "fell": 1.0}
        )
        == "✗ · timeout · fell"
    )
    assert agent_cells.slot_outcome({}) == ""


def test_the_billboard_hangs_above_the_pelvis_of_its_own_slot():
    """Slot i's anchor is the pelvis plus its grid offset plus ``z_offset``.

    Each slot's environment is modelled at the origin and moved into place
    by a frame; the billboards hang off the scene root, so they have to be
    offset by hand or every one of them marks the first slot.
    """
    pelvis = (0.4, -0.2, 0.78)
    gap = agent_cells.COMPARE_GAP
    offsets = agent_cells.grid_offsets(agent_cells.COMPARE_SLOTS, gap)
    for offset in offsets:
        anchor = agent_cells.billboard_anchor(pelvis, offset)
        assert anchor == pytest.approx(
            (
                pelvis[0] + offset[0],
                pelvis[1] + offset[1],
                pelvis[2] + agent_cells.BILLBOARD_Z,
            )
        )
    high = agent_cells.billboard_anchor(pelvis, (0.0, 0.0, 0.0), z_offset=2.0)
    assert high == pytest.approx((0.4, -0.2, 2.78))


def test_the_billboard_is_a_drawn_plate_not_a_gui_container():
    """viser 1.1's 3D GUI container paints a white card we cannot turn off.

    ``add_3d_gui_container`` takes name/wxyz/position/visible and nothing
    else, its contents are wrapped in a Mantine ``Paper`` whose
    ``background-color: var(--mantine-color-body)`` is the card, and
    ``configure_theme`` carries no CSS. So the plate is drawn here instead
    and added as an RGBA image.
    """
    image = agent_cells.plate_image("s1 · v008", "✗ · timeout", "#73aeff")
    assert image.shape[0] <= agent_cells.PLATE_SIZE[1]
    assert image.shape[1] <= agent_cells.PLATE_SIZE[0]
    assert image.shape[2] == 4
    assert image.dtype == np.uint8
    # The image is the plate itself, opaque, with only the rounded corners
    # clear: a transparent margin would still write depth and blank the
    # semi-transparent floor drawn behind it into a black box.
    h, w = image.shape[:2]
    assert image[0, 0, 3] == 0 and image[-1, -1, 3] == 0
    assert image[h // 2, 0, 3] == 255 and image[h // 2, -1, 3] == 255
    assert image[0, w // 2, 3] == 255 and image[-1, w // 2, 3] == 255
    assert image[h // 2, w // 2, 3] == 255
    # A one-line plate is shorter than a two-line one, and the node it goes
    # into is sized with it, at a width that keeps the text the same size.
    one = agent_cells.plate_image("s1 · v008", "", "#73aeff")
    assert one.shape[0] < h
    full = np.zeros((agent_cells.PLATE_SIZE[1], agent_cells.PLATE_SIZE[0], 4), np.uint8)
    assert agent_cells.plate_extent(full, 0.9) == pytest.approx(
        (0.9, 0.9 * agent_cells.PLATE_SIZE[1] / agent_cells.PLATE_SIZE[0])
    )
    shown_w, shown_h = agent_cells.plate_extent(one, 0.9)
    assert shown_w == pytest.approx(0.9 * one.shape[1] / agent_cells.PLATE_SIZE[0])
    assert shown_h == pytest.approx(shown_w * one.shape[0] / one.shape[1])
    # The stripe is the slot's colour, and there is white text on the plate.
    colours = {tuple(px) for px in image.reshape(-1, 4)}
    assert agent_cells.css_rgb("#73aeff") + (255,) in colours
    assert (255, 255, 255, 255) in colours
    # Nothing at all to say is an empty (fully transparent) canvas.
    assert agent_cells.plate_image("", "", "#73aeff")[..., 3].max() == 0


def test_the_plate_faces_the_side_the_compare_camera_watches_from():
    """A plane cannot billboard, so it is oriented once, looking along -x."""
    wxyz = agent_cells.plate_wxyz()
    w, x, y, z = wxyz
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    # viser pre-rotates the plane half a turn about x: the picture is on the
    # local -z side and its top is along local -y. Those two are what must
    # line up with the facing direction and world up.
    assert rotation @ np.array([0.0, 0.0, -1.0]) == pytest.approx(
        agent_cells.PLATE_FACE
    )
    assert rotation @ np.array([0.0, -1.0, 0.0]) == pytest.approx(agent_cells.PLATE_UP)
    # and the frame stays right-handed (a rotation, not a reflection)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_the_plate_leaves_the_live_step_counter_to_the_legend():
    """An image has to be re-encoded to change, so it carries the outcome."""
    assert agent_cells.plate_status("step 812/1701") == ""
    assert agent_cells.plate_status("step 3/3 · ✗ · timeout") == "✗ · timeout"


def test_the_legend_has_a_row_per_slot_with_a_swatch_and_a_status(tmp_path):
    """The sidebar says which colour is which policy, whatever the 3D does."""
    cells = two_sessions(tmp_path)
    request = agent_cells.resolve_compare(
        cells, TASK, [("s1", "v008"), ("s2", "last")], SEED
    )
    markup = agent_cells.legend_html(request.slots, ["step 12/1701", "✓"])
    assert agent_cells.slot_colour(0) in markup
    assert agent_cells.slot_colour(1) in markup
    # Both sessions are astra_high_*, so the legend drops that prefix.
    assert "s1 · v008 · submission" in markup
    assert "astra_high" not in markup
    assert "step 12/1701" in markup
    assert "building" not in markup
    # A short status list is padded, and the building line sits on top.
    while_building = agent_cells.legend_html(request.slots, [], "building 2/3 …")
    assert "building 2/3 …" in while_building
    assert "s2 · v007" in while_building


def test_auto_puts_three_slots_in_a_row_and_four_in_a_grid():
    """One row stays readable up to three; a fourth folds into 2 x 2."""
    for count in (1, 2, 3):
        assert agent_cells.slot_grid(count) == [(0, i) for i in range(count)]
        assert agent_cells.grid_shape(count) == (1, count)
    assert agent_cells.slot_grid(4) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert agent_cells.grid_shape(4) == (2, 2)
    # "two columns from four" all the way to six: 3 rows of 2, not 2 of 3.
    assert agent_cells.grid_shape(5) == (3, 2) == agent_cells.grid_shape(6)
    assert "4 slots as 2 columns x 2 rows, gap 2.5 m" == agent_cells.grid_note(4, 2.5)
    assert "1 slot as 1 column x 1 row" in agent_cells.grid_note(1, 2.5)


def test_every_layout_the_dropdown_offers():
    """`row` never wraps, `N columns` wraps at N, anything else is auto."""
    assert agent_cells.grid_shape(6, "row") == (1, 6)
    assert agent_cells.grid_shape(6, "3 columns") == (2, 3)
    assert agent_cells.grid_shape(5, "2 columns") == (3, 2)
    assert agent_cells.grid_shape(2, "3 columns") == (1, 2)  # never wider than it is
    assert agent_cells.layout_columns("nonsense", 4) == 2


def test_a_column_steps_along_y_and_a_row_steps_back_along_x():
    """Rows stand behind one another instead of running off the side."""
    gap = agent_cells.COMPARE_GAP
    assert agent_cells.grid_offsets(4, gap) == [
        (0.0, 0.0, 0.0),
        (0.0, gap, 0.0),
        (-gap, 0.0, 0.0),
        (-gap, gap, 0.0),
    ]
    # A single row is exactly the +y line compare mode always drew.
    assert agent_cells.grid_offsets(3, gap) == [(0.0, gap * i, 0.0) for i in range(3)]


def test_the_camera_starts_where_every_slot_is_in_view():
    """Off the -x face, back far enough for the width, up for the depth."""
    gap = agent_cells.COMPARE_GAP
    one_row = agent_cells.grid_offsets(3, gap)
    position, look_at = agent_cells.compare_camera_pose(one_row)
    assert position[0] < min(o[0] for o in one_row)
    assert position[1] == pytest.approx(gap)  # the middle of the row
    assert look_at[1] == pytest.approx(gap)
    single_height = position[2]

    grid = agent_cells.grid_offsets(4, gap)
    position, look_at = agent_cells.compare_camera_pose(grid)
    # Two rows: further out and higher, so the far row clears the near one.
    assert position[0] < min(o[0] for o in grid)
    assert position[2] > single_height + 1.0
    # The line of sight to the far row passes over a 1.7 m robot in the near.
    far = max(o[0] for o in grid)
    near = min(o[0] for o in grid)
    along = (near - position[0]) / (far - position[0])
    assert position[2] + along * (look_at[2] - position[2]) > 1.7


def test_focus_stands_in_front_of_one_slot_and_looks_at_its_pelvis():
    """The focus dropdown flies to the -x side of that robot."""
    offset = (-2.5, 2.5, 0.0)
    pelvis = (0.1, -0.2, 0.76)
    position, look_at = agent_cells.focus_pose(offset, pelvis, distance=2.0)
    assert look_at == pytest.approx((-2.4, 2.3, 0.91))
    assert position == pytest.approx((-4.4, 2.3, 1.35))
    assert position[0] < look_at[0]  # in front of it, not behind


class FakeCamera:
    """A viser client camera that only remembers where it was put."""

    def __init__(self):
        """Start at the origin."""
        self.position = (0.0, 0.0, 0.0)
        self.look_at = (0.0, 0.0, 0.0)


class FakeClient:
    """A connected viser client, for the camera moves."""

    def __init__(self):
        """Give the client its camera."""
        self.camera = FakeCamera()


class ClientServer:
    """A server stand-in whose ``get_clients`` returns fake clients."""

    def __init__(self, count: int = 2):
        """Connect ``count`` clients."""
        self.clients = {i: FakeClient() for i in range(count)}

    def get_clients(self) -> dict:
        """Every connected client, as viser spells it."""
        return self.clients


def test_flying_moves_every_connected_client():
    """One camera move is for everyone watching, not just whoever asked."""
    server = ClientServer(3)
    moved = agent_cells.fly_clients_to(server, (1.0, 2.0, 3.0), (0.0, 0.0, 0.5))
    assert moved == 3
    for client in server.clients.values():
        assert client.camera.position == (1.0, 2.0, 3.0)
        assert client.camera.look_at == (0.0, 0.0, 0.5)
    # A server with no clients yet is not an error.
    assert agent_cells.fly_clients_to(ClientServer(0), (0, 0, 0), (0, 0, 0)) == 0


def test_the_compare_folder_offers_focus_and_hides_the_camera_panels(tmp_path):
    """Focus names every slot; the panels start off and are asked for."""
    cells = two_sessions(tmp_path)
    request = agent_cells.resolve_compare(
        cells, TASK, [("s1", "v008"), ("s2", "last")], SEED
    )
    legend = agent_cells.CompareLegend(FakeServer(), request)
    assert legend.cameras.value is False
    assert legend.focus_dd.options == [
        agent_cells.FOCUS_FREE,
        "slot 0 · s1 v008",
        "slot 1 · s2 v007",
    ]
    # A callback only records the request; the main loop takes it once.
    assert legend.take_focus() is None
    legend.focus_dd.fire("slot 1 · s2 v007")
    assert legend.focus_asked == 1
    assert legend.take_focus() == 1
    assert legend.take_focus() is None
    legend.focus_dd.fire(agent_cells.FOCUS_FREE)
    assert legend.take_focus() is None
    assert agent_cells.grid_note(2, request.gap, request.layout) in (
        legend.header.content
    )


def test_labels_drop_the_prefix_every_session_shares():
    """`astra_high_s2` is `s2` when every slot is an `astra_high_` session."""
    assert (
        agent_cells.session_prefix(["astra_high_s1", "astra_high_s2", "astra_high_s3"])
        == "astra_high_"
    )
    # The cut is at a separator: the raw common prefix would leave "1"/"2".
    assert agent_cells.session_prefix(["s1", "s2"]) == ""
    assert agent_cells.session_prefix(["run-a-1", "run-a-2"]) == "run-a-"
    # Nothing to drop: one session, no shared start, or a name that IS it.
    assert agent_cells.session_prefix(["astra_high_s1"]) == ""
    assert agent_cells.session_prefix(["alpha", "beta"]) == ""
    assert agent_cells.session_prefix(["runs_", "runs_s2"]) == ""


def test_the_compare_labels_drop_the_shared_prefix(tmp_path):
    """Resolution hands the prefix to every slot, and the header names it."""
    cells = two_sessions(tmp_path)
    request = agent_cells.resolve_compare(
        cells, TASK, [("s1", "v008"), ("s2", "last")], SEED
    )
    assert request.prefix == "astra_high_"
    assert [slot.name for slot in request.slots] == ["s1", "s2"]
    assert request.slots[0].label.startswith("s1 · v008 · submission")
    assert "sessions astra_high_*" in agent_cells.compare_header_html(request)
    assert "slot 0 s1 v008" in agent_cells.compare_status_html(request)


def test_a_v001_that_is_still_the_template_says_so(tmp_path):
    """Three identical robots are explained by three v001 templates."""
    cells = two_sessions(tmp_path)
    cell = cells[0].path
    rows = agent_cells.policy_rows(cell)
    # No initial_policy.py at all: nothing is claimed.
    assert agent_cells.is_template_version(cell, rows[0]) is False

    template = "def act(obs):\n    return 0\n"
    (cell / "initial_policy.py").write_text(template)
    (cell / "policies" / "v001" / "policy.py").write_text(template)
    (cell / "policies" / "v002" / "policy.py").write_text(template + "# mine\n")
    assert agent_cells.is_template_version(cell, rows[0]) is True
    assert agent_cells.is_template_version(cell, rows[1]) is False

    slot = agent_cells.fill_slot(
        agent_cells.CompareSlot(session="s1", task=TASK, cell=cell, version=1), SEED
    )
    assert slot.template is True
    assert slot.label.startswith("s1 · v001 (template) · ")


def test_a_score_that_was_never_measured_is_named_not_dashed(tmp_path):
    """`train —` said nothing; `no dev episodes` says what is missing."""
    cells = two_sessions(tmp_path)
    slot = agent_cells.fill_slot(
        agent_cells.CompareSlot(session="s1", task=TASK, cell=cells[0].path, version=2),
        SEED,
    )
    assert slot.label.endswith("· no dev episodes · not evaluated")
    rows = agent_cells.policy_rows(cells[0].path)
    table = agent_cells.policy_table_html(rows)
    assert "no dev episodes" in table and "not evaluated" in table
    # The measured ones are still numbers.
    assert "0.50" in table and "0.02" in table


def test_the_status_line_counts_steps_then_names_the_outcome():
    """`step 1/3` while it plays, `step 3/3 · ✗ · timeout` at the end."""
    built = {
        "qpos": np.zeros((3, 7)),
        "info": {"success": 0.0, "termination": "timeout"},
    }
    assert agent_cells.slot_status_text(built, 0) == "step 1/3"
    assert agent_cells.slot_status_text(built, 2) == "step 3/3 · ✗ · timeout"


def test_the_panel_note_says_how_many_policies_and_how_to_add_one():
    """Three sessions becoming three slots is a default, not a limit."""
    assert "3 policies: one submission per session" in agent_cells.compare_note_html(3)
    assert "up to 6" in agent_cells.compare_note_html(3)
    assert "1 policy:" in agent_cells.compare_note_html(1)


def test_percentages_and_missing_values(tmp_path):
    """Success rates read as whole percentages, unknowns as an empty string."""
    assert agent_cells.percent(0.415) == "42%"
    assert agent_cells.percent(0.0) == "0%"
    assert agent_cells.percent(None) == ""


def test_the_compare_task_comes_from_the_rollout_that_would_open(tmp_path):
    """--compare has no task of its own; --demo-dir picks it."""
    make_cell(tmp_path / "runs" / "s1" / "move_plate", 3)
    make_cell(tmp_path / "runs" / "s2" / "move_plate", 3)
    cells = agent_cells.discover_cells([tmp_path / "runs"])
    batches = view_demos.discover_all([tmp_path / "runs"])
    here = agent_cells.cell_of(cells, batches[0])
    assert here is not None and here.task == "move_plate"
    request = agent_cells.resolve_compare(
        cells, here.task, agent_cells.parse_compare("s1:last,s2:last"), SEED
    )
    assert [s.task for s in request.slots] == ["move_plate", "move_plate"]
    assert request.missing == []


class FakeJob:
    """An ``AgentJob`` stand-in that advances a progress file over ticks.

    Each :meth:`poll` writes the next state of the plan it was given, which
    is what a real subprocess does between two main-loop ticks; the last
    entry also writes the episode the slot was waiting for, unless the job
    is meant to fail.
    """

    def __init__(self, cell: Path, version: int, seed: int, plan, fails: str = ""):
        """Describe one fake rollout.

        Args:
            cell: The cell it writes into.
            version: The policy version it replays.
            seed: The compare seed.
            plan: One ``.progress.json`` state (or None) per tick.
            fails: A message to fail with instead of writing the episode.
        """
        self.cell = Path(cell)
        self.version = int(version)
        self.seed = int(seed)
        self.plan = list(plan)
        self.fails = fails
        self.label = f"v{self.version:03d}"
        self.out_dir = self.cell / "replays" / self.label
        self.log_path = self.out_dir / "job.log"
        self.command = ["replay", str(self.cell), "--version", str(self.version)]
        self.error: str | None = None
        self.started = False
        self.stopped = False
        self.ticks = 0

    def start(self) -> None:
        """Pretend to launch the subprocess."""
        self.started = True
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def poll(self) -> bool:
        """Advance the plan by one tick; True once it has run out."""
        if self.ticks >= len(self.plan):
            if self.fails:
                self.error = self.fails
            elif not self.stopped:
                write_batch(self.out_dir / "batch", (self.seed,))
            return True
        state = self.plan[self.ticks]
        self.ticks += 1
        if state is not None:
            (self.out_dir / agent_cells.PROGRESS_NAME).write_text(json.dumps(state))
        return False

    def stop(self) -> None:
        """Pretend to kill the subprocess."""
        self.stopped = True


def running(step: int, total: int = 3523) -> dict:
    """One ``.progress.json`` state of a single-episode replay."""
    return {
        "seed": SEED,
        "step": step,
        "max_steps": total,
        "episode": 0,
        "episodes": 1,
        "done": False,
        "pid": 4242,
    }


def fake_jobs(monkeypatch, plans: dict, fails: str = "") -> list:
    """Make ``MissingWork`` build :class:`FakeJob` instead of a subprocess."""
    made: list[FakeJob] = []

    def build(slot, seed) -> FakeJob:
        """Stand in for ``agent_cells.missing_job``."""
        job = FakeJob(
            slot.cell, int(slot.version or 0), seed, plans.get(slot.session, ()), fails
        )
        made.append(job)
        return job

    monkeypatch.setattr(agent_cells, "missing_job", build)
    return made


def missing_request(tmp_path, seed: int = 620042):
    """Two sessions whose chosen version has no episode on ``seed``."""
    cells = two_sessions(tmp_path)
    request = agent_cells.resolve_compare(
        cells, TASK, [("s1", "v003"), ("s2", "v003")], seed
    )
    assert len(request.missing) == 2
    return request


def test_missing_rollouts_run_in_parallel_and_never_block(tmp_path, monkeypatch):
    """The jobs start together and are polled a tick at a time."""
    request = missing_request(tmp_path, SEED)
    made = fake_jobs(
        monkeypatch,
        {"astra_high_s1": [running(0), running(812)], "astra_high_s2": [running(25)]},
    )
    work = agent_cells.MissingWork(request)
    work.start()
    assert [j.started for j in made] == [True, True]
    assert len(work.jobs) == 2 and work.done is False

    assert work.poll() is False
    assert work.bars() == [
        (0.0, "s1 · v003 · seed 620003 · 0/3523", False),
        (
            pytest.approx(25 / 3523),
            "s2 · v003 · seed 620003 · 25/3523",
            False,
        ),
    ]
    assert work.poll() is False
    fractions = [row[0] for row in work.bars()]
    labels = [row[1] for row in work.bars()]
    assert labels[0] == "s1 · v003 · seed 620003 · 812/3523"
    assert fractions[0] == pytest.approx(812 / 3523)
    # s2's plan is over, so its job finished on this tick and its bar is full.
    assert work.bars()[1] == (1.0, "s2 · v003 · seed 620003 · done", False)

    assert work.poll() is True
    assert work.error == ""
    assert work.done is True
    # Every slot was re-resolved from the files the jobs wrote.
    assert request.missing == []
    assert [Path(s.episode).name for s in request.slots] == [f"seed{SEED}.npz"] * 2
    assert [s.source for s in request.slots] == ["replays/v003"] * 2


def test_a_job_that_has_written_nothing_yet_animates(tmp_path, monkeypatch):
    """An indeterminate bar beats one stuck at zero with no explanation."""
    request = missing_request(tmp_path, SEED)
    fake_jobs(monkeypatch, {"astra_high_s1": [None, running(5)]})
    work = agent_cells.MissingWork(request)
    work.start()
    work.poll()
    assert work.bars()[0] == (
        0.0,
        "s1 · v003 · seed 620003 · starting…",
        True,
    )


def test_two_slots_on_the_same_policy_share_one_job(tmp_path, monkeypatch):
    """The same episode file is never written twice."""
    cells = two_sessions(tmp_path)
    request = agent_cells.resolve_compare(
        cells, TASK, [("s1", "v003"), ("s1", "v003")], SEED
    )
    made = fake_jobs(monkeypatch, {"astra_high_s1": [running(1)]})
    work = agent_cells.MissingWork(request)
    work.start()
    assert len(made) == 1 and len(work.jobs) == 1
    while not work.poll():
        pass
    assert work.error == "" and request.missing == []


def test_a_failed_job_stops_the_others_and_says_why(tmp_path, monkeypatch):
    """One broken policy ends compare mode instead of hanging it."""
    request = missing_request(tmp_path, SEED)
    made = fake_jobs(
        monkeypatch,
        {"astra_high_s1": [], "astra_high_s2": [running(1)] * 5},
        fails="replay failed (exit 1): RuntimeError: policy v003 crashed",
    )
    work = agent_cells.MissingWork(request)
    work.start()
    assert work.poll() is True
    assert work.error == "replay failed (exit 1): RuntimeError: policy v003 crashed"
    assert made[1].stopped is True
    assert request.missing  # nothing was resolved: the scene is not built


def test_leaving_kills_the_jobs_that_are_still_running(tmp_path, monkeypatch):
    """The Leave button must work while the rollouts run."""
    request = missing_request(tmp_path, SEED)
    plan = [running(1)] * 9
    made = fake_jobs(monkeypatch, {"astra_high_s1": plan, "astra_high_s2": plan})
    work = agent_cells.MissingWork(request)
    work.start()
    work.poll()
    work.stop()
    assert [j.stopped for j in made] == [True] * len(made)
    assert work.pending == set()


def test_a_rollout_that_writes_nothing_is_reported(tmp_path, monkeypatch):
    """A job that exits 0 without the episode is still a reason to stop."""
    request = missing_request(tmp_path, 620042)

    def silent(slot, seed):
        """A job that finishes the first time it is polled, writing nothing."""
        job = FakeJob(slot.cell, int(slot.version or 0), seed, ())
        job.stopped = True  # the plan's "write the episode" step is skipped
        return job

    monkeypatch.setattr(agent_cells, "missing_job", silent)
    work = agent_cells.MissingWork(request)
    work.start()
    while not work.poll():
        pass
    assert "still no episode on seed 620042" in work.error
    assert "s1 v003" in work.error


def test_nothing_missing_is_done_before_the_first_tick(tmp_path):
    """Every slot already stores the seed: no jobs, no bars, no waiting."""
    cells = two_sessions(tmp_path)
    request = agent_cells.resolve_compare(cells, TASK, [("s1", "last")], SEED)
    work = agent_cells.MissingWork(request)
    assert work.start() == []
    assert work.done is True and work.error == "" and work.bars() == []
    assert work.poll() is True


def test_the_compare_folder_draws_one_bar_per_job(tmp_path, monkeypatch, capsys):
    """wait_for_missing keeps the GUI alive: bars, a log line, then the scene."""
    request = missing_request(tmp_path, SEED)
    fake_jobs(
        monkeypatch,
        {"astra_high_s1": [None, running(812)], "astra_high_s2": [running(3000)]},
    )
    server = FakeServer()
    legend = agent_cells.CompareLegend(server, request)
    assert legend.jobs_note.content == ""

    ok = agent_cells.wait_for_missing(
        request, legend, tick=0.0, log_every=0.0, deadline=0.0
    )
    assert ok is True
    assert request.missing == []
    # The bars are HTML in the note node, emptied once the work is over.
    assert legend.jobs_note.content == ""
    log = capsys.readouterr().out
    assert "s1 · v003 · seed 620003 · 812/3523 (23%)" in log
    assert "s1 · v003 · seed 620003 · starting… (0%)" in log


def test_leave_pressed_while_producing_returns_without_a_scene(
    tmp_path, monkeypatch, capsys
):
    """The button the legend already owns works before the scene exists."""
    request = missing_request(tmp_path, SEED)
    made = fake_jobs(monkeypatch, {"astra_high_s1": [running(1)] * 20})
    legend = agent_cells.CompareLegend(FakeServer(), request)
    legend.leave.fire()  # the browser pressed Leave
    assert legend.left is True

    assert agent_cells.wait_for_missing(request, legend, tick=0.0) is False
    assert [j.stopped for j in made] == [True] * len(made)
    assert "Leave pressed" in capsys.readouterr().out


def test_a_failed_job_is_shown_in_red(tmp_path, monkeypatch):
    """The Compare folder says what went wrong instead of going quiet."""
    request = missing_request(tmp_path, SEED)
    fake_jobs(monkeypatch, {}, fails="replay failed (exit 1): boom")
    legend = agent_cells.CompareLegend(FakeServer(), request)
    assert agent_cells.wait_for_missing(request, legend, tick=0.0) is False
    assert "#dc2626" in legend.jobs_note.content
    assert "replay failed (exit 1): boom" in legend.jobs_note.content


def test_the_progress_bars_carry_their_own_label_line():
    """Each job gets a label line and an HTML bar whose width is its fraction."""
    server = FakeServer()
    node = server.gui.add_html("")
    bars = agent_cells.JobProgressBars(node)
    bars.sync(
        [
            (0.23, "astra_high_s2 · v002 · seed 500 · 812/3523", False),
            (0.0, "astra_high_s3 · v001 · seed 500 · starting…", True),
        ],
        note="2 rollout(s) to produce on seed 500",
    )
    assert "2 rollout(s) to produce" in node.content
    assert "812/3523" in node.content and "width:23.0%" in node.content
    assert "starting…" in node.content and "repeating-linear-gradient" in node.content
    # A second tick redraws; the same markup is not re-sent.
    bars.sync([(1.0, "astra_high_s2 · v002 · seed 500 · done", False)])
    assert "width:100.0%" in node.content and "starting…" not in node.content
    bars.clear()
    assert node.content == ""


@pytest.mark.slow
def test_compare_builds_one_prefixed_scene_per_slot(tmp_path, capsys):
    """Two environments in one viser server, offset along +y, each labelled."""
    from bigym.loco import make

    probe = make(TASK)
    try:
        nq = int(probe.inner_env.model.nq)
    finally:
        probe.close()

    for session in ("s1", "s2", "s3", "s4"):
        cell = make_cell(tmp_path / "runs" / session / TASK, 3)
        write_batch(
            cell / "eval" / "v003" / "batch", (SEED, SEED + 1), nq=nq, protocol=True
        )
    cells = agent_cells.discover_cells([tmp_path / "runs"])
    request = agent_cells.resolve_compare(
        cells, TASK, [(s, "last") for s in ("s1", "s2", "s3", "s4")], SEED
    )
    assert request.missing == []
    # Four slots: "auto" folds them into two columns of two rows.
    assert request.offsets() == [
        (0.0, 0.0, 0.0),
        (0.0, agent_cells.COMPARE_GAP, 0.0),
        (-agent_cells.COMPARE_GAP, 0.0, 0.0),
        (-agent_cells.COMPARE_GAP, agent_cells.COMPARE_GAP, 0.0),
    ]

    built = agent_cells.build_compare_scene(
        request,
        port=8791,
        build_env=view_demos.build_env,
        load_metadata=view_demos.load_metadata,
        lighting_prefs=view_demos._LIGHTING_PREFS,
    )
    server = built["server"]
    try:
        assert len(built["slots"]) == 4
        names = set(server.scene._handle_from_node_name)
        for i in range(4):
            assert any(n.startswith(f"/bodies/slot{i}/") for n in names)
            assert f"/compare/slot{i}/billboard" in names
        # Each slot's subtree hangs off a frame that carries its grid place:
        # a column along +y, a row back along -x.
        for i, offset in enumerate(request.offsets()):
            frame = server.scene._handle_from_node_name[f"/bodies/slot{i}"]
            assert tuple(frame.position) == pytest.approx(offset)
        # Only the first slot keeps a ground plane (they are coplanar).
        grids = [n for n in names if n.endswith("/floor/floor")]
        assert len(grids) == 4
        assert server.scene._handle_from_node_name[
            "/fixed_bodies/slot0/floor/floor"
        ].visible
        for i in (1, 2, 3):
            assert not server.scene._handle_from_node_name[
                f"/fixed_bodies/slot{i}/floor/floor"
            ].visible
        # Every slot is built hidden, posed to frame 0 of its own replay and
        # only then revealed -- nobody sees the reset pose, and the scenes
        # arrive together instead of one by one.
        log = capsys.readouterr().out
        hidden = [
            i for i, line in enumerate(log.splitlines()) if "built hidden" in line
        ]
        revealed = [
            i
            for i, line in enumerate(log.splitlines())
            if "scenes revealed together" in line
        ]
        assert len(hidden) == 4 and len(revealed) == 1
        assert max(hidden) < revealed[0]
        assert "posed to frame 0 of the replay" in log
        assert "compare layout auto: 4 slots as 2 columns x 2 rows" in log
        assert "at row 1 column 1 (x -2.50, y +2.50)" in log
        for slot in built["slots"]:
            for frame in slot["frames"]:
                assert frame.visible
            assert slot["billboard"].node.visible
            # A drawn RGBA plate, not viser's white 3D GUI container.
            assert slot["billboard"].kind == "image"
            assert tuple(slot["billboard"].node.wxyz) == pytest.approx(
                agent_cells.plate_wxyz()
            )
            stored = slot["qpos"][0]
            assert slot["data"].qpos[: stored.shape[0]] == pytest.approx(stored)

        # The tint is display only: the model the camera panels render from
        # is exactly what it was before mjviser baked the meshes.
        for slot in built["slots"]:
            assert np.array_equal(slot["model"].geom_rgba, slot["geom_rgba"])
        assert np.array_equal(
            built["slots"][0]["model"].geom_rgba,
            built["slots"][1]["model"].geom_rgba,
        )

        # The billboards sit on their own slot's pelvis, offset into the grid.
        for i, slot in enumerate(built["slots"]):
            pelvis = agent_cells.pelvis_position(slot["data"], slot["pelvis"])
            assert slot["pelvis"] >= 0
            assert slot["billboard"].position == pytest.approx(
                agent_cells.billboard_anchor(pelvis, built["offsets"][i])
            )
        assert built["slots"][0]["slot"].label.startswith("s1 · v003")
        assert "s1 · v003" in built["legend"].body.content
        assert (
            built["legend"]
            .describe()[0]
            .endswith("s1 · v003 · submission · train 50% · eval 2% · step 1/5")
        )
        assert agent_cells.slot_colour(1) in built["legend"].describe()[1]
        assert built["camera_keys"] == ("head",)
    finally:
        server.stop()
        for slot in built["slots"]:
            slot["env"].close()


def test_the_plate_turns_to_face_the_camera():
    """facing_wxyz points the picture at the camera with +z kept up."""
    wxyz = agent_cells.facing_wxyz((0.0, 0.0, 1.9), (3.0, 0.0, 1.9))
    # camera at +x: the picture side (local -z) looks along +x, which is
    # the local +z along -x, and the picture's top (local -y) is world +z.
    w, x, y, z = wxyz
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    assert rot[:, 2] == pytest.approx([-1.0, 0.0, 0.0], abs=1e-9)
    assert rot[:, 1] == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)
    assert np.linalg.det(rot) == pytest.approx(1.0)
    # straight above: still a valid orientation, face pointing up
    w, x, y, z = agent_cells.facing_wxyz((0.0, 0.0, 0.0), (0.0, 0.0, 5.0))
    assert abs(w * w + x * x + y * y + z * z - 1.0) < 1e-9
    # no clients connected -> the default -x facing is kept
    assert agent_cells.viewer_camera_position(object()) is None


def test_the_billboard_faces_the_client_when_one_is_connected():
    """A connected client's camera turns every plate towards it."""

    class _Cam:
        position = (4.0, 1.0, 1.5)

    class _Client:
        camera = _Cam()

    class _Server:
        def get_clients(self):
            return {1: _Client()}

    class _Node:
        position = (0.0, 0.0, 1.9)
        wxyz = agent_cells.plate_wxyz()
        visible = True

    board = agent_cells.SlotBillboard.__new__(agent_cells.SlotBillboard)
    board.server = _Server()
    board.node = _Node()
    board._faced = None
    assert board.face() is True
    assert tuple(board.node.wxyz) == pytest.approx(
        agent_cells.facing_wxyz((0.0, 0.0, 1.9), (4.0, 1.0, 1.5))
    )
    assert board.face() is False  # nothing moved: no update


def test_a_compare_recording_steps_the_playhead_at_the_episodes_rate():
    """One video frame per file frame; the playhead keeps real time."""
    # 50 Hz episodes written at 50 fps: one step per frame, wrapping.
    heads = agent_cells.compare_record_playheads(0.1, 50.0, 50.0, 3)
    assert heads == [0, 1, 2, 0, 1]
    # written at 25 fps: two episode steps per frame, still real time
    assert agent_cells.compare_record_playheads(0.2, 50.0, 25.0, 100) == [0, 2, 4, 6, 8]
    # never empty, never past the end
    assert agent_cells.compare_record_playheads(0.0, 50.0, 50.0, 10) == [0]
    assert max(agent_cells.compare_record_playheads(10, 50, 50, 7)) == 6


def test_a_compare_recording_poses_then_renders_then_writes():
    """record_compare: show(playhead), get_render, write; the sink is sized
    by what really came back and closed at the end."""
    posed = []
    frames = []

    class _Client:
        def get_render(self, height, width):
            # a human's browser may hand back its own size
            return np.full((height // 2, width // 2, 4), 7, np.uint8)

    class _Sink:
        closed = False

        def __init__(self, real):
            self.real = real

        def write(self, frame):
            frames.append(frame)

        def close(self):
            _Sink.closed = True

    made = []

    def make_sink(real):
        sink = _Sink(real)
        made.append(sink)
        return sink

    written, actual = agent_cells.record_compare(
        posed.append,
        _Client(),
        make_sink,
        playheads=[0, 1, 1],
        size=(64, 32),
        log_every=0,
    )
    assert posed == [0, 1, 1]
    assert written == 3 and actual == (32, 16)
    assert made[0].real == (32, 16) and _Sink.closed
    # RGBA renders are written as RGB, contiguous, uint8
    assert frames[0].shape == (16, 32, 3) and frames[0].dtype == np.uint8
    assert frames[0].flags["C_CONTIGUOUS"]


def test_a_compare_recording_refuses_a_resized_browser():
    """A frame of a different size mid-recording is an error, not a corrupt file."""
    sizes = iter([(10, 20), (12, 20)])

    class _Client:
        def get_render(self, height, width):
            h, w = next(sizes)
            return np.zeros((h, w, 3), np.uint8)

    class _Sink:
        def __init__(self, real):
            pass

        def write(self, frame):
            pass

        def close(self):
            pass

    with pytest.raises(RuntimeError, match="resize"):
        agent_cells.record_compare(
            lambda f: None,
            _Client(),
            _Sink,
            playheads=[0, 1],
            size=(20, 10),
            log_every=0,
        )


def test_the_viewer_cli_turns_record_flags_into_a_request(monkeypatch):
    """--record and its companions become run_compare's ``record`` dict."""
    from bigym.vr.viewer import view_demos

    args = view_demos.ViewConfig(
        compare="s1:last",
        record="out.mp4",
        record_seconds=4.0,
        record_size="640x360",
        record_fps=25.0,
        browser="none",
        record_then_stay=True,
        record_view="oblique",
        record_zoom=2.0,
        record_speed=0.5,
        record_label_scale=3.0,
    )
    request = view_demos.record_request(args)
    assert request is not None
    assert request["path"].name == "out.mp4"
    assert request["seconds"] == 4.0 and request["size"] == (640, 360)
    assert request["fps"] == 25.0 and request["browser"] == "none"
    assert request["stay"] is True
    assert request["view"] == "oblique" and request["zoom"] == 2.0
    assert request["speed"] == 0.5 and request["label_scale"] == 3.0
    assert view_demos.record_request(view_demos.ViewConfig()) is None


def test_the_viewer_cli_rejects_values_it_cannot_act_on():
    """ViewConfig refuses a zero zoom, --record without --compare and so on."""
    from bigym.vr.viewer import view_demos

    cases: list[dict[str, Any]] = [
        {"record": "out.mp4"},
        {"compare": "s1:last", "record": "out.mp4", "record_zoom": 0.0},
        {"record_speed": 0.0},
        {"record_label_scale": 0.0},
        {"follow_seconds": 0.0},
        {"exit_after_seconds": -1.0},
        {"episode": -1},
    ]
    for bad in cases:
        with pytest.raises(ValueError):
            view_demos.ViewConfig(**bad)


def test_a_recording_waits_for_the_page_to_finish_loading():
    """settle_render renders until two frames in a row match."""
    frames = iter([1, 2, 3, 3, 3])

    class _Client:
        def get_render(self, height, width):
            return np.full((height, width, 3), next(frames), np.uint8)

    assert agent_cells.settle_render(_Client(), (4, 2)) == 4

    class _Restless:
        def get_render(self, height, width):
            return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    # a page that never settles is recorded anyway once the timeout passes
    assert agent_cells.settle_render(_Restless(), (4, 2), timeout=0.0) >= 1


def test_the_oblique_view_stands_in_front_and_to_one_side_and_fits_the_box():
    """compare_record_pose: overview = home; oblique = +x/+y side, above,
    looking at the grid's centre; box_fov holds every corner in frame."""
    offsets = [(0.0, 0.0, 0.0), (0.0, 2.5, 0.0), (-2.5, 0.0, 0.0), (-2.5, 2.5, 0.0)]
    home = agent_cells.compare_camera_pose(offsets)
    assert agent_cells.compare_record_pose(offsets, "overview", home=home) == home
    position, look_at = agent_cells.compare_record_pose(offsets, "oblique")
    centre, half = agent_cells.compare_box(offsets, agent_cells.COMPARE_PAD_TIGHT)
    assert look_at == pytest.approx(centre)
    # in front of the robots (+x of the grid), to the +y side, above
    assert position[0] > centre[0] and position[1] > centre[1]
    assert position[2] > centre[2]
    with pytest.raises(ValueError):
        agent_cells.compare_record_pose(offsets, "drone")

    fov = agent_cells.box_fov(position, look_at, (centre, half), 16 / 9)
    assert agent_cells.RECORD_FOV_MIN < fov < agent_cells.RECORD_FOV_MAX
    # every corner projects inside the frame at that lens
    eye = np.asarray(position)
    forward = np.asarray(look_at) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corner = np.asarray(centre) + np.asarray(half) * [sx, sy, sz] - eye
                depth = corner @ forward
                assert abs(corner @ up) / depth <= np.tan(fov / 2) + 1e-9
                assert abs(corner @ right) / depth <= np.tan(fov / 2) * 16 / 9 + 1e-9
    # a closer camera needs a wider lens; a tighter pad a narrower one
    nearer = tuple(np.asarray(look_at) + 0.5 * (eye - np.asarray(look_at)))
    assert agent_cells.box_fov(nearer, look_at, (centre, half)) > fov
    assert (
        agent_cells.box_fov(position, look_at, agent_cells.compare_box(offsets)) > fov
    )
    # playback speed scales the playhead step
    assert agent_cells.compare_record_playheads(0.1, 50, 50, 100, speed=2.0) == [
        0,
        2,
        4,
        6,
        8,
    ]
