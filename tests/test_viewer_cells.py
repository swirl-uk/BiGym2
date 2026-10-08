"""The demo viewer on an agent cell: batch discovery, labels, job plumbing.

An agent "cell" is one task x one session of the coding-agent benchmark:
``policies/index.json`` plus ``eval/vNNN/`` and ``replays/vNNN/`` whose
``batch/`` subdirectories are ordinary replay-format demo batches. These
tests build such a directory in ``tmp_path`` (no environment, no rendering)
and check that ``bigym-view`` finds the batches, labels them with the
version's scores, and leaves ordinary demo directories alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bigym.loco.agent.snapshots import version_dir
from bigym.vr.viewer import agent_cells, view_demos
from tests.fixtures.agent_cell import TASK, make_cell, write_batch
from tests.fixtures.viser_gui import FakeServer


@pytest.fixture
def cell(tmp_path) -> Path:
    """A cell with two versions, an evaluation of v001 and a replay of v007."""
    cell = make_cell(
        tmp_path / TASK,
        2,
        evaluated=1,
        eval_success=0.41,
        eval_seeds=(620000, 620001),
    )
    write_batch(cell / "replays" / "v007" / "batch", (620003,), (0.0,))
    return cell


def test_cell_is_detected_and_ordinary_dirs_are_not(cell, tmp_path):
    """policies/index.json is what makes a directory a cell."""
    assert agent_cells.is_agent_cell(cell)
    plain = tmp_path / "demos" / TASK / "20260102_030405"
    write_batch(plain, range(620000, 620003))
    assert not agent_cells.is_agent_cell(plain.parent.parent)


def test_discovery_finds_eval_and_replay_batches(cell):
    """Both run-layout batch dirs are discovered under the cell."""
    found = view_demos.discover_batches(cell)
    assert [b.relative_to(cell).as_posix() for b in found] == [
        "eval/v001/batch",
        "replays/v007/batch",
    ]


def test_labels_carry_version_and_success(cell):
    """eval/replay batches read `eval/v001 · 2 eps · success 0.41`."""
    batches = view_demos.discover_batches(cell)
    labels = view_demos.batch_labels(cell, batches)
    assert labels == ["eval/v001 · 2 eps · success 0.41", "replays/v007 · 1 eps"]


def test_label_without_summary_omits_success(cell):
    """An evaluation still being written has no success figure to show."""
    (cell / "eval" / "v001" / "summary.json").unlink()
    batches = view_demos.discover_batches(cell)
    labels = view_demos.batch_labels(cell, batches)
    assert labels[0] == "eval/v001 · 2 eps"


def test_batch_opened_directly_still_labels_as_a_version(cell):
    """--demo-dir <cell>/eval/v001/batch labels the version, not `.`."""
    batch = cell / "eval" / "v001" / "batch"
    found = view_demos.discover_batches(batch)
    assert found == [batch]
    assert view_demos.batch_labels(batch, found) == ["eval/v001 · 2 eps · success 0.41"]


def test_ordinary_demo_dirs_keep_their_labels(tmp_path):
    """A collector/export directory is labelled exactly as before."""
    root = tmp_path / "demos"
    write_batch(root / TASK / "20260102_030405", range(620000, 620003))
    write_batch(root / "move_plate" / "20260102_040506", range(620000, 620002))
    batches = view_demos.discover_batches(root)
    labels = view_demos.batch_labels(root, batches)
    assert labels == [
        "move_plate/20260102_040506 · 2 eps",
        f"{TASK}/20260102_030405 · 3 eps",
    ]


def test_batch_version_rejects_non_cell_layout(tmp_path):
    """Only `<cell>/eval|replays/vNNN/batch` is treated as a version batch."""
    write_batch(tmp_path / "runs" / "v001" / "batch", (620000,))
    assert agent_cells.batch_version(tmp_path / "runs" / "v001" / "batch") is None
    write_batch(tmp_path / "eval" / "latest" / "batch", (620000,))
    assert agent_cells.batch_version(tmp_path / "eval" / "latest" / "batch") is None


def test_policy_rows_join_the_eval_summary(cell):
    """Policy versions carry train success from the index, eval from summary."""
    rows = agent_cells.policy_rows(cell)
    assert [row["version"] for row in rows] == [1, 2]
    assert rows[0]["trigger"] == "write"
    assert rows[0]["eval_success"] == pytest.approx(0.41)
    assert rows[1]["train_success"] == pytest.approx(0.5)
    assert rows[1]["eval_success"] is None
    assert "v001" in agent_cells.policy_table_html(rows)
    assert "no policies" in agent_cells.policy_table_html([])


def test_version_dir_tolerates_unpadded_names(tmp_path):
    """`eval/v7` and `eval/v007` both resolve to version 7."""
    cell = tmp_path / TASK
    (cell / "eval" / "v7").mkdir(parents=True)
    assert version_dir(cell, 7, "eval").name == "v7"
    assert version_dir(cell, 9, "eval").name == "v009"


def test_pick_batch_by_index_and_substring(tmp_path):
    """--batch takes an index or a unique substring of the label."""
    labels = ["eval/v001 · 2 eps", "replays/v007 · 1 eps"]
    assert view_demos.pick_batch(labels, None) == 0
    assert view_demos.pick_batch(labels, "1") == 1
    assert view_demos.pick_batch(labels, "replays") == 1
    assert view_demos.pick_batch(labels, "v001") == 0
    with pytest.raises(SystemExit):
        view_demos.pick_batch(labels, "nothing")
    with pytest.raises(SystemExit):
        view_demos.pick_batch(labels, "eps")


def test_agent_job_command_and_output_dir(cell):
    """Replay/evaluate run the bigym-agent CLI module with the cell and version."""
    replay = agent_cells.AgentJob("replay", cell, 7, "620000-620004")
    assert replay.command[1:3] == ["-m", agent_cells.AGENT_CLI_MODULE]
    assert replay.command[3:] == [
        "replay",
        str(cell),
        "--version",
        "7",
        "--seeds",
        "620000-620004",
    ]
    assert replay.out_dir == cell / "replays" / "v007"
    assert replay.progress() == "replay v007: 1 episodes written"

    evaluate = agent_cells.AgentJob("evaluate", cell, 1)
    assert evaluate.command[3:] == ["evaluate", str(cell), "--version", "1"]
    assert evaluate.out_dir == cell / "eval" / "v001"
    assert evaluate.progress() == "evaluate v001: 2 episodes scored"


def test_the_progress_label_says_where_a_job_has_got_to():
    """`s2 · v002 · seed 500 · 812/3523`, and what each part is dropped for."""
    one = {"seed": 500, "step": 812, "max_steps": 3523, "episode": 0, "episodes": 1}
    assert agent_cells.progress_label("s2 · v002 · seed 500", one) == (
        "s2 · v002 · seed 500 · 812/3523"
    )
    assert agent_cells.progress_fraction(one) == pytest.approx(812 / 3523)
    # Several episodes: the bar crosses the whole run, not each episode.
    many = {"seed": 620002, "step": 150, "max_steps": 300, "episode": 2, "episodes": 5}
    assert agent_cells.progress_label("replay v007", many) == (
        "replay v007 · 150/300 · 2/5 episodes"
    )
    assert agent_cells.progress_fraction(many) == pytest.approx(0.5)
    # An evaluation reports no steps at all: its episodes run in workers.
    scored = {"seed": 0, "step": 0, "max_steps": 0, "episode": 37, "episodes": 100}
    assert agent_cells.progress_label("evaluate v007", scored) == (
        "evaluate v007 · 37/100 episodes"
    )
    assert agent_cells.progress_fraction(scored) == pytest.approx(0.37)
    # Nothing written yet.
    assert agent_cells.progress_label("evaluate v007", None) == (
        "evaluate v007 · starting…"
    )
    assert agent_cells.progress_fraction(None) == 0.0
    assert "812/3523" in agent_cells.progress_note_html("x · 812/3523")


def test_a_job_reads_the_progress_file_its_subprocess_writes(cell):
    """The bar follows .progress.json, then the files the job finished."""
    job = agent_cells.AgentJob("replay", cell, 7, "620000-620004")
    assert agent_cells.read_progress(job.out_dir) is None
    assert job.progress_state() == (0.0, "replay v007 · starting…", True)

    job.out_dir.mkdir(parents=True, exist_ok=True)
    (job.out_dir / agent_cells.PROGRESS_NAME).write_text(
        json.dumps(
            {
                "seed": 620001,
                "step": 150,
                "max_steps": 300,
                "episode": 1,
                "episodes": 5,
                "done": False,
                "pid": 4242,
            }
        )
    )
    fraction, label, animated = job.progress_state()
    assert fraction == pytest.approx(0.3)
    assert label == "replay v007 · 150/300 · 1/5 episodes"
    assert animated is False

    # An evaluation with no progress file falls back to its scored rows.
    evaluation = agent_cells.AgentJob("evaluate", cell, 1)
    assert evaluation.episodes == agent_cells.DEFAULT_EVAL_EPISODES
    assert evaluation.csv_rows() == 2
    assert evaluation.progress_state() == (
        0.02,
        "evaluate v001 · 2/100 episodes",
        False,
    )
    asked = agent_cells.AgentJob("evaluate", cell, 1, extra=("--episodes", "20"))
    assert asked.episodes == 20
    assert asked.progress_state()[0] == pytest.approx(0.1)


def test_agent_job_reports_a_missing_subcommand(cell):
    """A CLI without the subcommand becomes a readable panel message."""
    job = agent_cells.AgentJob("replay", cell, 1)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    job.log_path.write_text(
        "usage: bigym-agent ...\nbigym-agent: error: argument command: "
        "invalid choice: 'replay'\n"
    )
    message = job.diagnose(2)
    assert "no working 'bigym-agent replay' subcommand" in message
    job.log_path.write_text("bigym-agent: unknown command 'replay'\nusage: ...\n")
    assert "no working 'bigym-agent replay' subcommand" in job.diagnose(2)
    job.log_path.write_text("Traceback ...\nRuntimeError: policy v001 crashed\n")
    assert (
        job.diagnose(1) == "replay failed (exit 1): RuntimeError: policy v001 crashed"
    )


def test_agent_job_start_failure_is_captured(cell):
    """A job that cannot even be launched leaves a message, not an exception."""
    job = agent_cells.AgentJob("replay", cell, 1)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    job.log_path.mkdir()  # opening this for writing fails
    job.start()
    assert job.proc is None
    assert job.error and "cannot write" in job.error
    assert job.poll() is True


@pytest.mark.slow
def test_camera_panels_render_real_pixels():
    """The panels render the env's onboard cameras plus a third-person view."""
    from bigym.loco import make

    env = make(TASK)
    try:
        env.reset(seed=0)
        inner = env.inner_env
        server = FakeServer()
        panels = view_demos.CameraPanels(
            server,
            inner,
            inner.model,
            inner.data,
            ("head", "right_wrist", "left_wrist"),
            (84, 84),
        )
        try:
            panels.update(lookat=(0.0, 0.0, 0.9))
            assert panels.third_label, panels.error
            head = server.gui.images["head"].image
            assert head.shape == (84, 84, 3)
            assert head.dtype == np.uint8
            assert head.any(), "the head camera rendered an all-black frame"
            third = server.gui.images[panels.third_label].image
            assert third.ndim == 3 and third.shape[2] == 3
            assert third.any()
            panels.enabled.value = False
            panels.update()
            assert not server.gui.images["head"].visible
        finally:
            panels.close()
    finally:
        env.close()


def test_dev_batches_are_discovered_and_labelled(cell):
    """`dev/vNNN_<stamp>/batch` reads `dev/v006 · 3 eps · success 0.33`."""
    batch = cell / "dev" / "v006_20260921_010203" / "batch"
    write_batch(batch, range(620000, 620003), (1.0, 0.0, 0.0))
    assert agent_cells.batch_version(batch) == (cell, "dev", 6)
    labels = view_demos.batch_labels(cell, view_demos.discover_batches(cell))
    assert "dev/v006 · 3 eps · success 0.33" in labels
    assert agent_cells.batch_success(batch) == pytest.approx(1 / 3)


def test_policy_file_batches_are_labelled_by_their_digest(cell):
    """`bigym-agent replay --policy` batches live under `policy_<sha8>`."""
    batch = cell / "replays" / "policy_ab12cd34" / "batch"
    write_batch(batch, (620000,))
    assert agent_cells.batch_version(batch) == (cell, "replays", None)
    labels = view_demos.batch_labels(cell, view_demos.discover_batches(cell))
    assert "replays/policy_ab12cd34 · 1 eps" in labels
    # A directory that is not `vNNN` and not a digest is not a version batch.
    write_batch(cell / "replays" / "policy_zz" / "batch", (620000,))
    assert agent_cells.batch_version(cell / "replays" / "policy_zz" / "batch") is None


def test_follow_rescan_picks_up_a_new_batch(cell):
    """A batch written between two rescans appears in the live lists."""
    batches = view_demos.discover_batches(cell)
    labels = view_demos.batch_labels(cell, batches)
    assert view_demos.rescan_into(cell, batches, labels) is False
    before = len(batches)

    dev = cell / "dev" / "v008_20260921_020304" / "batch"
    write_batch(dev, range(620000, 620002), (1.0, 1.0))
    assert view_demos.rescan_into(cell, batches, labels) is True
    assert len(batches) == before + 1
    assert dev in batches
    assert "dev/v008 · 2 eps · success 1.00" in labels
    # The newest episode file is the one the auto-jump follows.
    newest = view_demos.newest_batch(batches)
    assert newest is not None and batches[newest] == dev
    # Episodes added to an existing batch update its label in place.
    write_batch(cell / "replays" / "v007" / "batch", range(620000, 620003), [0.0] * 3)
    view_demos.rescan_into(cell, batches, labels)
    assert "replays/v007 · 4 eps" in labels


def test_episode_option_labels_and_list(tmp_path):
    """Episode options read `620003 ✓ 1599` / `620002 ✗ 1700 timeout`."""
    batch = tmp_path / "dev" / "v001_stamp" / "batch"
    write_batch(batch, (620003, 620002), (1.0, 0.0))
    infos = agent_cells.episode_infos(batch)
    assert view_demos.episode_labels(infos) == [
        "620002 ✗ 1700 timeout",
        "620003 ✓ 1599",
    ]
    listing = view_demos.episode_list_html(infos, current=1)
    assert "620003" in listing and "#16a34a" in listing and "#dc2626" in listing
    assert view_demos.episode_list_html([], 0).count("no episodes") == 1


def test_summary_card_from_run_json(cell):
    """The session card shows harness, budget, verdict, tokens and cost."""
    (cell / "run.json").write_text(
        json.dumps(
            {
                "task": TASK,
                "harness": "codex",
                "model": "gpt-6-astra",
                "effort": "high",
                "interface": "strict",
                "wall_clock_s": 5810.4,
                "budget": {"used": 99651, "cap": 101000},
                "verdict": {"state": "ok", "reason": ""},
                "usage": {
                    "input": 593640,
                    "cached": 24072192,
                    "output": 105312,
                    "reasoning": 76339,
                },
                "cost_usd": 35.2742,
            }
        )
    )
    card = agent_cells.summary_html(agent_cells.read_run(cell))
    for fragment in (
        "codex",
        "gpt-6-astra",
        "high",
        "strict",
        "99,651 / 101,000",
        "ok",
        "1h 36m",
        "0.6M",
        "105k",
        "$35.27",
    ):
        assert fragment in card, fragment
    assert "no run.json" in agent_cells.summary_html({})


def test_summary_card_computes_cost_when_the_run_did_not(tmp_path):
    """Without `cost_usd` the card prices the tokens from the price table."""
    from bigym.loco.agent.transcript import MODEL_PRICES

    run = {
        "model": "gpt-6-astra",
        "usage": {"input": 1_000_000, "cached": 0, "output": 100_000},
    }
    price = MODEL_PRICES["gpt-6-astra"]
    expected = price["input"] + price["output"] * 0.1
    assert agent_cells.token_cost(run["usage"], price) == pytest.approx(expected)
    assert f"${expected:,.2f}" in agent_cells.summary_html(run)
    assert agent_cells.token_cost(None, price) is None


def index_with_provenance(cell: Path) -> None:
    """Add the per-version provenance the snapshot watcher records."""
    path = cell / "policies" / "index.json"
    index = json.loads(path.read_text())
    stamps = [1_790_000_000.0, 1_790_000_600.0]
    for i, entry in enumerate(index["versions"]):
        entry["ts"] = stamps[i]
        entry["command_index"] = 2 * i
        entry["budget_used"] = 1000 * (i + 1)
        entry["tokens"] = {
            "input": 593640 * (i + 1),
            "cached": 0,
            "output": 105312,
            "reasoning": 0,
        }
        entry["elapsed_s"] = 60.0 * (i + 1)
    path.write_text(json.dumps(index))


def test_policy_table_shows_provenance_when_it_is_there(cell):
    """Command index, budget, tokens and cost appear only when recorded."""
    plain = agent_cells.policy_table_html(agent_cells.policy_rows(cell))
    assert "cmd" not in plain and "tokens" not in plain
    assert agent_cells.policy_rows(cell)[0]["ts"] == pytest.approx(
        agent_cells.iso_seconds("2026-01-02T01:00:00+00:00")
    )

    index_with_provenance(cell)
    rows = agent_cells.policy_rows(cell)
    assert rows[0]["command_index"] == 0 and rows[1]["command_index"] == 2
    assert rows[1]["ts"] == pytest.approx(1_790_000_600.0)
    table = agent_cells.policy_table_html(
        rows, {"input": 10.0, "cached": 1.0, "output": 50.0}
    )
    assert "cmd" in table and "budget" in table
    assert "0.6M/105k" in table and "1.2M/105k" in table
    assert "$11.20" in table  # 0.59364M in + 0.105312M out at 10/50 per M
    # Without a price table the cost column degrades to a dash.
    assert "—" in agent_cells.policy_table_html(rows, None)


def test_transcript_slices_between_version_timestamps(cell):
    """The panel shows what the agent did to produce the NEXT version."""
    index_with_provenance(cell)
    records = [
        {"t": 1_790_000_001.0, "kind": "message", "role": "agent", "text": "first"},
        {
            "t": 1_790_000_300.0,
            "kind": "command",
            "role": "agent",
            "command": "ls",
            "exit_code": 0,
        },
        {"t": 1_790_000_900.0, "kind": "message", "role": "agent", "text": "later"},
    ]
    (cell / "transcript.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n{ broken\n"
    )
    assert agent_cells.read_transcript(cell) == records

    rows = agent_cells.policy_rows(cell)
    first = agent_cells.version_records(records, rows, 0)
    assert [r["text"] for r in first if r["kind"] == "message"] == ["first"]
    assert len(first) == 2
    last = agent_cells.version_records(records, rows, 1)
    assert [r["text"] for r in last if "text" in r] == ["later"]
    assert "v002" in agent_cells.transcript_title(rows, 0)
    assert "no later version" in agent_cells.transcript_title(rows, 1)

    body = agent_cells.transcript_html(first)
    assert "first" in body and "ls" in body
    assert "nothing in this slice" in agent_cells.transcript_html([])


def test_transcript_without_timestamps_falls_back_to_command_index(cell):
    """An older transcript is sliced by how many commands had run."""
    index_with_provenance(cell)
    rows = agent_cells.policy_rows(cell)
    records = [
        {"kind": "command", "role": "agent", "command": "a", "exit_code": 0},
        {"kind": "command", "role": "agent", "command": "b", "exit_code": 0},
        {"kind": "message", "role": "agent", "text": "after two commands"},
        {"kind": "command", "role": "agent", "command": "c", "exit_code": 0},
    ]
    first = agent_cells.version_records(records, rows, 0)
    assert [r.get("command") for r in first if r["kind"] == "command"] == ["a", "b"]
    second = agent_cells.version_records(records, rows, 1)
    assert [r.get("text") for r in second if r["kind"] == "message"] == [
        "after two commands"
    ]
    # Long command blocks are clipped to 20 lines in the panel.
    long_command = {"kind": "command", "role": "agent", "command": "\n".join("l" * 40)}
    assert "more lines" in agent_cells.transcript_html([long_command])


def test_transcript_without_any_timestamps_or_index_shows_everything(cell):
    """An old cell with no provenance at all still shows its transcript."""
    rows = agent_cells.policy_rows(cell)
    records = [{"kind": "message", "role": "agent", "text": "hello"}]
    assert agent_cells.version_records(records, rows, 0) == records


def test_code_panel_and_diff(cell):
    """The code panel shows the version's source, its path and a diff."""
    rows = agent_cells.policy_rows(cell)
    first = agent_cells.policy_file(cell, rows[0])
    second = agent_cells.policy_file(cell, rows[1])
    first.write_text("def act(obs):\n    return 0\n")
    second.write_text("def act(obs):\n    return 1\n")
    assert first == cell / "policies" / "v001" / "policy.py"

    code = agent_cells.code_html(first)
    assert "act" in code and "2 lines" in code
    assert "cannot" in agent_cells.code_html(cell / "nope.py") or "No such" in (
        agent_cells.code_html(cell / "nope.py")
    )
    link = agent_cells.vscode_link_html(second)
    assert f"vscode://file{second}" in link

    diff = agent_cells.diff_html(first.read_text(), second.read_text(), "v001", "v002")
    assert "+1 / -1" in diff
    assert "#16a34a" in diff and "#dc2626" in diff
    assert "return 1" in diff and "return 0" in diff
    same = agent_cells.diff_html("x\n", "x\n", "v001", "v002")
    assert "identical" in same


def test_reward_sparkline_svg(tmp_path):
    """The reward strip is an inline SVG with a marker at the frame."""
    svg = view_demos.sparkline_svg([0.0, 0.0, 1.0, 1.0], 2, flags="success · fell")
    assert svg.startswith("<div") and "<svg" in svg and "polyline" in svg
    assert "success · fell" in svg
    assert "reward 1.00 (min 0.00, max 1.00)" in svg
    # The marker stays inside the box for an out-of-range frame.
    assert "<circle" in view_demos.sparkline_svg([1.0], 99)
    assert "no reward" in view_demos.sparkline_svg([], 0)


def test_agent_job_replays_a_loose_policy_file(cell, tmp_path):
    """ "Replay this file" runs `replay --policy` into `replays/policy_<sha8>`."""
    policy = tmp_path / "my_policy.py"
    policy.write_text("def act(obs):\n    return 0\n")
    digest = agent_cells.file_digest(policy)
    job = agent_cells.AgentJob("replay", cell, 0, "620000", policy=str(policy))
    assert job.command[3:] == [
        "replay",
        str(cell),
        "--policy",
        str(policy),
        "--seeds",
        "620000",
    ]
    assert job.out_dir == cell / "replays" / f"policy_{digest[:8]}"
    assert job.label == f"policy_{digest[:8]}"
    assert job.progress() == f"replay policy_{digest[:8]}: 0 episodes written"
    assert agent_cells.file_digest(tmp_path / "missing.py") == ""


def test_files_being_written_are_not_counted(tmp_path):
    """Only finished ``*.npz`` files count as episodes."""
    batch = tmp_path / "dev" / "v001_stamp" / "batch"
    write_batch(batch, (620000,), (1.0,))
    for name in (".w0_1234.npz", "seed620009.npz.tmp", ".seed620009.npz.part"):
        (batch / name).write_bytes(b"half a file")
    assert [p.name for p in agent_cells.episode_files(batch)] == ["seed620000.npz"]
    assert view_demos.batch_labels(tmp_path, view_demos.discover_batches(tmp_path)) == [
        "dev/v001 · 1 eps · success 1.00"
    ]
    # A batch that holds nothing but a file being written is not one yet.
    empty = tmp_path / "dev" / "v002_stamp" / "batch"
    empty.mkdir(parents=True)
    (empty / "metadata.json").write_text("{}")
    (empty / ".w0_1234.npz").write_bytes(b"half a file")
    assert empty not in view_demos.discover_batches(tmp_path)


# --- roots, sessions and the Rollouts tree ----------------------------------


def make_session_root(root: Path, session: str, tasks: tuple[str, ...]) -> Path:
    """A session root: ``<root>/<session>/<task>/`` cells with one eval each."""
    for i, task in enumerate(tasks):
        make_cell(
            root / session / task,
            2,
            eval_success=0.02 + 0.01 * i,
            eval_seeds=(620000, 620001, 620002),
        )
    return root / session


def test_discover_cells_accepts_a_cell_a_root_and_a_parent(tmp_path):
    """One cell, one session root, or a directory of roots — all three work."""
    make_session_root(tmp_path / "runs", "s1", (TASK, "move_plate"))
    make_session_root(tmp_path / "runs", "s2", (TASK,))

    one = agent_cells.discover_cells([tmp_path / "runs" / "s1" / TASK])
    assert [(c.session, c.task) for c in one] == [("s1", TASK)]

    root = agent_cells.discover_cells([tmp_path / "runs" / "s1"])
    assert [(c.session, c.task) for c in root] == [("s1", "move_plate"), ("s1", TASK)]

    parent = agent_cells.discover_cells([tmp_path / "runs"])
    assert [(c.session, c.task) for c in parent] == [
        ("s1", "move_plate"),
        ("s1", TASK),
        ("s2", TASK),
    ]
    assert agent_cells.tasks_of(parent) == ["move_plate", TASK]
    assert agent_cells.sessions_of(parent) == ["s1", "s2"]
    assert agent_cells.sessions_of(parent, "move_plate") == ["s1"]


def test_discover_cells_merges_several_demo_dirs_without_duplicates(tmp_path):
    """--demo-dir r1 r2 is two sessions; the same path twice is still one."""
    make_session_root(tmp_path / "runs", "s1", (TASK,))
    make_session_root(tmp_path / "runs", "s2", (TASK,))
    cells = agent_cells.discover_cells(
        [
            tmp_path / "runs" / "s1",
            tmp_path / "runs" / "s2",
            tmp_path / "runs" / "s1" / TASK,
        ]
    )
    assert [(c.session, c.task) for c in cells] == [("s1", TASK), ("s2", TASK)]
    found = agent_cells.find_cell(cells, TASK, "s2")
    assert found is not None and found.path.name == TASK
    assert agent_cells.find_cell(cells, "nope", "s2") is None


def test_sessions_with_the_same_name_keep_their_parent(tmp_path):
    """Two roots called `nightly` are told apart by the directory above."""
    make_session_root(tmp_path / "a", "nightly", (TASK,))
    make_session_root(tmp_path / "b", "nightly", (TASK,))
    cells = agent_cells.discover_cells([tmp_path / "a", tmp_path / "b"])
    assert sorted(c.session for c in cells) == ["a/nightly", "b/nightly"]


def test_plain_demo_dirs_discover_no_cells(tmp_path):
    """A collector/export tree stays in plain demo-directory mode."""
    write_batch(tmp_path / "demos" / TASK / "20260102_030405", range(620000, 620003))
    assert agent_cells.discover_cells([tmp_path / "demos"]) == []


def test_rollout_labels_say_episodes_and_percentages(cell):
    """Rollout items read `v001 · 2 episodes · 41%`, never `eps`."""
    write_batch(
        cell / "dev" / "v001_20260921_010203" / "batch",
        range(620000, 620003),
        (1.0, 0.0, 0.0),
    )
    batches = view_demos.discover_batches(cell)
    rollouts = agent_cells.rollouts_of(cell, batches)
    labels = {r.kind: agent_cells.rollout_item_label(r) for r in rollouts}
    assert labels["eval"] == "v001 · 2 episodes · 41%"
    assert labels["dev"] == "while policy.py was v001 · 3 episodes · 33%"
    assert labels["replays"] == "v007 · 1 episodes · 0%"
    assert agent_cells.rollout_counts(rollouts) == (
        "evaluation 1 · development 1 · replays 1"
    )


def test_rollouts_are_grouped_by_kind_newest_version_first(cell):
    """Evaluations first, then development, then replays; versions descend."""
    write_batch(cell / "dev" / "v001_20260921_010203" / "batch", (620000,), (1.0,))
    write_batch(cell / "dev" / "v002_20260921_020304" / "batch", (620000,), (0.0,))
    rollouts = agent_cells.rollouts_of(cell, view_demos.discover_batches(cell))
    assert [(r.kind, r.version) for r in rollouts] == [
        ("eval", 1),
        ("dev", 2),
        ("dev", 1),
        ("replays", 7),
    ]
    assert [r.title for r in rollouts][:2] == ["Evaluation", "Development"]


def test_the_submission_evaluation_opens_by_default(cell):
    """The default rollout is the evaluation of the last policy version."""
    write_batch(cell / "eval" / "v002" / "batch", range(620000, 620004))
    (cell / "eval" / "v002" / "summary.json").write_text(
        json.dumps({"version": 2, "success_rate": 0.5})
    )
    rollouts = agent_cells.rollouts_of(cell, view_demos.discover_batches(cell))
    opened, why = agent_cells.default_rollout(rollouts)
    assert opened is not None
    assert opened.version == 2 and opened.kind == "eval" and opened.submission
    assert why == "the evaluation of the submission"
    assert (
        agent_cells.rollout_item_label(opened) == "v002 · 4 episodes · 50% · submission"
    )


def test_default_rollout_falls_back_when_the_submission_is_unscored(cell):
    """Without an evaluation of the submission, the newest evaluation opens."""
    rollouts = agent_cells.rollouts_of(cell, view_demos.discover_batches(cell))
    opened, why = agent_cells.default_rollout(rollouts)
    assert opened is not None
    assert opened.kind == "eval" and opened.version == 1
    assert why == "the newest evaluation (no submission scored yet)"

    only_dev = [r for r in rollouts if r.kind != "eval"]
    opened, why = agent_cells.default_rollout(only_dev)
    assert opened is not None
    assert opened.kind == "replays"
    assert why == "no evaluation yet, so the newest replays"
    assert agent_cells.default_rollout([]) == (None, "this cell has no rollouts yet")


def build_panels(tmp_path):
    """A CellPanels over two sessions, plus the switch/compare requests."""
    make_session_root(tmp_path / "runs", "s1", ("move_plate", TASK))
    make_session_root(tmp_path / "runs", "s2", (TASK,))
    cells = agent_cells.discover_cells([tmp_path / "runs"])
    batches = view_demos.discover_all([tmp_path / "runs"])
    opened, _ = agent_cells.default_rollout(
        agent_cells.rollouts_of(cells[0].path, batches)
    )
    assert opened is not None
    asked: dict = {"switch": [], "compare": []}
    panels = agent_cells.CellPanels(
        FakeServer(),
        cell=cells[0].path,
        cells=cells,
        batches=batches,
        open_batch=opened.batch,
        request_switch=asked["switch"].append,
        request_compare=asked["compare"].append,
        follow=True,
    )
    return panels, cells, batches, asked


def test_cell_panels_offer_the_task_session_and_rollout_choices(tmp_path):
    """Session holds task x session; Rollouts holds kind then item."""
    panels, cells, _, _ = build_panels(tmp_path)
    assert panels.task_dd.options == ["move_plate", TASK]
    assert panels.task_dd.value == "move_plate"
    assert panels.session_dd.options == ["s1"]  # only s1 has move_plate
    assert panels.kind_dd.options == ["Evaluation"]
    assert panels.item_dd.value == "v002 · 3 episodes · 2% · submission"
    assert "the evaluation of the submission" in panels.rollout_note.content
    assert panels.auto_jump is not None and panels.auto_jump.value is False
    assert "opening Evaluation" in "\n".join(panels.describe())


def test_picking_another_session_opens_its_default_rollout(tmp_path):
    """Task / Session are requests; the main loop resolves them in tick()."""
    panels, cells, batches, asked = build_panels(tmp_path)
    panels.task_dd.fire(TASK)
    assert asked["switch"] == []  # callbacks only flag
    panels.tick(0.0)
    assert panels.session_dd.options == ["s1", "s2"]
    wanted = batches[asked["switch"][-1]]
    assert wanted == cells[1].path / "eval" / "v002" / "batch"  # s1/reach_target_single

    panels.session_dd.fire("s2")
    panels.tick(1.0)
    s2 = agent_cells.find_cell(cells, TASK, "s2")
    assert s2 is not None
    assert batches[asked["switch"][-1]] == s2.path / "eval" / "v002" / "batch"


def test_picking_another_rollout_switches_batch(tmp_path):
    """The kind and item dropdowns address batches of the cell on screen."""
    panels, cells, batches, asked = build_panels(tmp_path)
    dev = cells[0].path / "dev" / "v001_20260921_010203" / "batch"
    write_batch(dev, range(620000, 620002), (1.0, 0.0))
    batches[:] = view_demos.discover_all([cells[0].path.parent.parent])
    panels.refresh_rollouts()
    assert panels.kind_dd.options == ["Evaluation", "Development"]
    panels.kind_dd.fire("Development")
    panels.tick(0.0)
    assert panels.item_dd.value == "while policy.py was v001 · 2 episodes · 50%"
    assert batches[asked["switch"][-1]] == dev


def test_the_compare_panel_starts_with_one_slot_per_session(tmp_path):
    """Sessions holding the task become slots, and the note says so."""
    panels, _, _, _ = build_panels(tmp_path)
    panels.task_dd.fire(TASK)
    panels.tick(0.0)  # moving the task re-points Compare at it
    assert [str(r["session"].value) for r in panels.compare.rows] == ["s1", "s2"]
    assert [r["folder"].label for r in panels.compare.rows] == ["slot 0", "slot 1"]
    assert "2 policies: one submission per session" in panels.compare.note.content
    assert f"up to {agent_cells.COMPARE_SLOTS}" in panels.compare.note.content


def test_add_policy_appends_a_slot_and_remove_drops_it(tmp_path):
    """The fourth policy is one button away, and so is dropping one."""
    panels, _, _, _ = build_panels(tmp_path)
    panels.task_dd.fire(TASK)
    panels.tick(0.0)
    panels.compare.add_button.fire()
    assert len(panels.compare.rows) == 2  # callbacks only flag
    panels.tick(1.0)
    assert len(panels.compare.rows) == 3
    assert "3 policies" in panels.compare.note.content

    doomed = panels.compare.rows[1]
    doomed["remove"].fire()
    assert doomed in panels.compare.rows
    panels.tick(2.0)
    assert doomed not in panels.compare.rows
    assert doomed["folder"].removed and doomed["session"].removed
    assert [r["folder"].label for r in panels.compare.rows] == ["slot 0", "slot 1"]


def test_the_panel_never_holds_more_than_six_policies(tmp_path):
    """Six scenes side by side is the most compare mode builds."""
    panels, _, _, _ = build_panels(tmp_path)
    for i in range(8):
        panels.compare.add_button.fire()
        panels.tick(float(i))
    assert len(panels.compare.rows) == 6
    assert "6 policies is the most" in panels.compare.status.content


def test_the_compare_panel_hands_over_a_resolved_request(tmp_path):
    """Compare fills its slots from the sessions that hold this task."""
    panels, cells, _, asked = build_panels(tmp_path)
    panels.task_dd.fire(TASK)
    panels.tick(0.0)  # moving the task re-points Compare at it
    assert panels.compare.rows[0]["version"].options == ["last", "v001", "v002"]
    panels.compare.seed_text.value = "620001"
    panels.compare.button.fire()
    panels.tick(1.0)
    request = asked["compare"][-1]
    assert [s.session for s in request.slots] == ["s1", "s2"]
    assert [s.version for s in request.slots] == [2, 2]
    assert request.seed == 620001 and request.missing == []
    assert all(s.source == "eval/v002" for s in request.slots)


def test_the_layout_dropdown_travels_with_the_request(tmp_path):
    """Compare mode places its slots the way the panel was asked to."""
    panels, _, _, asked = build_panels(tmp_path)
    panels.task_dd.fire(TASK)
    panels.tick(0.0)
    assert panels.compare.layout_dd.options == list(agent_cells.COMPARE_LAYOUTS)
    assert panels.compare.layout_dd.value == agent_cells.LAYOUT_AUTO

    panels.compare.button.fire()
    panels.tick(1.0)
    assert asked["compare"][-1].layout == agent_cells.LAYOUT_AUTO

    panels.compare.layout_dd.value = "2 columns"
    panels.compare.gap_number.value = 3.0
    panels.compare.button.fire()
    panels.tick(2.0)
    request = asked["compare"][-1]
    assert request.layout == "2 columns"
    assert request.offsets() == [(0.0, 0.0, 0.0), (0.0, 3.0, 0.0)]


class PanelJob:
    """An ``AgentJob`` stand-in for the panel's Replay / Evaluate bar."""

    def __init__(self, rows):
        """Take one ``(fraction, label, animated)`` per poll, then finish."""
        self.rows = list(rows)
        self.error = None
        self.kind = "replay"
        self.label = "v002"
        self.polls = 0

    def poll(self) -> bool:
        """True once the planned progress rows have run out."""
        return self.polls >= len(self.rows)

    def progress_state(self):
        """The next planned row."""
        row = self.rows[self.polls]
        self.polls += 1
        return row

    def progress(self) -> str:
        """The finished line the panel prints and shows."""
        return "replay v002: 5 episodes written"


def test_the_replay_button_shows_a_progress_bar_not_just_a_line(tmp_path):
    """A rollout that takes minutes needs a bar that moves while it runs."""
    panels, _, _, _ = build_panels(tmp_path)
    assert panels.job_bar.content == ""

    panels.job = PanelJob(
        [
            (0.0, "replay v002 · starting…", True),
            (0.3, "replay v002 · 150/300 · 1/5 episodes", False),
        ]
    )
    panels.poll_job(1.0)
    assert "repeating-linear-gradient" in panels.job_bar.content  # striped: unknown
    assert "starting…" in panels.job_msg.content

    panels.poll_job(2.0)
    assert "width:30.0%" in panels.job_bar.content
    assert "150/300 · 1/5 episodes" in panels.job_msg.content

    # The job ends: the bar goes away and the buttons come back.
    panels.poll_job(3.0)
    assert panels.job is None
    assert panels.job_bar.content == ""
    assert panels.replay_btn.disabled is False and panels.eval_btn.disabled is False
    assert "5 episodes written" in panels.job_msg.content


def test_a_bad_seed_is_refused_without_a_request(tmp_path):
    """A seed that is not a number never reaches the main loop."""
    panels, _, _, asked = build_panels(tmp_path)
    panels.compare.seed_text.value = "soon"
    panels.compare.button.fire()
    panels.tick(0.0)
    assert asked["compare"] == []
    assert "is not a number" in panels.compare.status.content
