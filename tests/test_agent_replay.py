"""Replay batches and cell-mode evaluation of the coding-agent benchmark.

The fast tests cover the seed parser and the batch metadata builder against a
stand-in environment. The slow ones build the real environment: they replay
the shipped template policy into a batch, open that batch with the demo
viewer's own loader, and run a cell-mode evaluation end to end.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bigym.loco.agent import replay, snapshots
from bigym.loco.agent.batch import (
    PROGRESS_NAME,
    Recorder,
    RunProgress,
    batch_metadata,
    write_episode_npz,
    write_json,
)
from bigym.loco.agent.cli import EnvToolsConfig
from bigym.loco.config import EnvConfig
from bigym.loco.demos.schema import BATCH_FORMAT
from tests.fixtures.agent_cell import make_cell

TASK = "reach_target_single"


def template_policy() -> str:
    """The text of the policy template shipped in the sandbox (it holds still)."""
    return (
        Path(replay.__file__).parent / "templates" / "policy_template.py"
    ).read_text()


@pytest.fixture
def cell(tmp_path) -> Path:
    """A cell whose one version is the shipped template, with its sandbox settings."""
    cell = make_cell(tmp_path / TASK, 1, evaluated=None, policy_text=template_policy())
    (cell / "sandbox_config.json").write_text(
        json.dumps({"task": TASK, "env_tools": dataclasses.asdict(EnvToolsConfig())})
    )
    return cell


class FakeOuter:
    """The fields ``batch_metadata`` reads off the protocol env."""

    task_name = TASK
    config = EnvConfig(
        episode_length=3000,
        enable_all_floating_dof=False,
        reach_tolerance=0.05,
    )
    control_step_seconds = 0.02

    def substrate_fingerprint(self):
        """Return a stand-in fingerprint."""
        return {"task": TASK, "robot_model": "g1_dex1"}

    def episode_fell(self) -> bool:
        """The robot never falls in the stand-in environment."""
        return False

    def episode_succeeded(self) -> bool:
        """The task is never done in the stand-in environment."""
        return False

    def episode_termination(self) -> str:
        """Every stand-in episode runs out of time."""
        return "timeout"


class FakeEnvTools:
    """Enough of EnvTools for the metadata builder and the recorder."""

    task = TASK
    tier = "images"

    def __init__(self):
        """Bind the stand-in env and a tiny simulator state."""
        self.outer = FakeOuter()
        self.data = type("Data", (), {"qpos": np.arange(5.0), "qvel": np.arange(4.0)})()

    def info(self):
        """Return the static interface description."""
        return {"interface": "strict"}

    def close(self) -> None:
        """Release the environment: the stand-in holds nothing."""


class FakeTimestep:
    """What ``EnvTools.step`` hands back: a reward and an end-of-episode flag."""

    def __init__(self, reward: float, last: bool):
        """Store one step's outcome."""
        self.reward = reward
        self._last = last

    def last(self) -> bool:
        """True when the episode ended on this step."""
        return self._last


class SteppingEnvTools(FakeEnvTools):
    """A wrapper that steps instantly, for the progress-file tests.

    It also snapshots ``.progress.json`` after every step, which is the only
    way to see the file as a reader sees it: the values a rollout writes
    while it runs, not just the one it leaves behind.
    """

    def __init__(self, time_limit: int = 60):
        """Bind the fake env and the episode length it reports."""
        super().__init__()
        self.time_limit = int(time_limit)
        self.steps = 0
        self.watched: Path | None = None
        self.seen: list[dict | None] = []

    def watch(self, path: Path) -> None:
        """Snapshot this file after every step."""
        self.watched = Path(path)

    def reset(self, seed):
        """Start a new episode."""
        self.steps = 0

    def observation(self) -> dict:
        """The two keys ``run_episode`` reads off an observation."""
        return {"time_limit": self.time_limit, "fell": False}

    def step(self, raw):
        """Count one step, snapshot the progress file and end on the limit."""
        self.steps += 1
        if self.watched is not None:
            self.seen.append(
                json.loads(self.watched.read_text()) if self.watched.exists() else None
            )
        return FakeTimestep(0.0, self.steps >= self.time_limit)


def write_still_policy(root: Path) -> Path:
    """A policy file that returns a constant action, for the fake env."""
    path = Path(root) / "still.py"
    path.write_text(
        "import numpy as np\n\n\n"
        "class Policy:\n"
        "    def reset(self, obs, tools):\n"
        "        pass\n\n"
        "    def act(self, obs, tools):\n"
        "        return np.zeros(3, dtype=np.float32)\n"
    )
    return path


def test_seed_parser_takes_lists_and_ranges():
    """`a,b` and `a-b` (inclusive), in the order they were written."""
    assert replay.parse_seeds("620000-620004") == [620000 + i for i in range(5)]
    assert replay.parse_seeds("620003,620001") == [620003, 620001]
    assert replay.parse_seeds("1-3,7, 9-10") == [1, 2, 3, 7, 9, 10]
    assert replay.parse_seeds("5") == [5]
    for bad in ("", "  ", "3-1", "abc", "1-x"):
        with pytest.raises(ValueError):
            replay.parse_seeds(bad)


def test_batch_metadata_has_every_key_the_readers_use():
    """The viewer and the re-renderer rebuild the env from these fields."""
    tools: Any = FakeEnvTools()
    metadata = batch_metadata(
        tools, source={"kind": "agent_policy", "cell": "/tmp/cell", "version": 7}
    )
    assert metadata["format"] == BATCH_FORMAT
    assert metadata["control_step_seconds"] == pytest.approx(0.02)
    assert metadata["lowerbody_policy"]["backend"] == "groot_wbc_g1"
    assert metadata["substrate_fingerprint"]["task"] == TASK
    assert metadata["source"]["version"] == 7
    task = metadata["task"]
    for key in (
        "task_name",
        "robot_model",
        "camera_keys",
        "camera_shape",
        "enable_all_floating_dof",
        "action_mode",
        "demo_down_sample_rate",
        "episode_length",
        "success_hold_seconds",
        "reach_tolerance",
        "initialization_profile",
    ):
        assert key in task, key
    assert task["camera_keys"] == ["head", "right_wrist", "left_wrist"]
    assert EnvConfig.from_metadata(metadata) == FakeOuter.config
    # it must survive a round trip through JSON: the viewer reads the file
    assert json.loads(json.dumps(metadata, default=str))["task"]["task_name"] == TASK


def test_recorder_collects_one_more_state_than_actions():
    """T+1 states around T actions, read off the live simulator state."""
    tools: Any = FakeEnvTools()
    recorder = Recorder(tools)
    recorder.capture_state()
    for step in range(3):
        recorder.capture_step(np.full(20, float(step)), 0.5)
        recorder.capture_state()
    arrays = recorder.arrays()
    assert arrays["full_qpos"].shape == (4, 5)
    assert arrays["full_qvel"].shape == (4, 4)
    assert arrays["action"].shape == (3, 20)
    assert arrays["reward"].shape == (3, 1)


def test_write_episode_npz_round_trips(tmp_path):
    """The written keys are the ones a replay batch reader expects."""
    path = write_episode_npz(
        tmp_path / "seed620000.npz",
        full_qpos=np.zeros((4, 7)),
        full_qvel=np.zeros((4, 6)),
        action=np.ones((3, 20), np.float32),
        reward=np.zeros(3),
        seed=620000,
        success=1,
        fell=False,
        termination="success",
    )
    with np.load(path) as episode:
        assert episode["full_qpos"].shape == (4, 7)
        assert episode["action"].shape == (3, 20)
        assert episode["reward"].shape == (3, 1)
        assert int(episode["seed"]) == 620000
        assert float(episode["success"]) == 1.0
        assert int(episode["length"]) == 3
        assert str(episode["termination"]) == "success"


def test_resolve_policy_needs_exactly_one_selector(tmp_path):
    """--version and --policy are alternatives, not options."""
    cell = tmp_path / TASK
    (cell / "policies" / "v001").mkdir(parents=True)
    (cell / "policies" / "v001" / "policy.py").write_text("x = 1\n")
    with pytest.raises(ValueError, match="exactly one"):
        snapshots.resolve_policy(cell, None, None, "replays")
    with pytest.raises(ValueError, match="exactly one"):
        snapshots.resolve_policy(cell, 1, Path("p.py"), "replays")
    path, out_dir, version = snapshots.resolve_policy(cell, 1, None, "replays")
    assert (path, out_dir.name, version) == (
        cell / "policies" / "v001" / "policy.py",
        "v001",
        1,
    )
    loose = tmp_path / "loose.py"
    loose.write_text("x = 1\n")
    _, out_dir, version = snapshots.resolve_policy(cell, None, loose, "replays")
    assert version is None and out_dir.name.startswith("policy_")


def fake_recordings(monkeypatch) -> list[int]:
    """Make ``replay_seeds`` write episodes without an environment.

    The point of the test below is which seeds run, not what a policy does
    with them, so the environment and the episode recorder are stand-ins and
    the whole thing takes milliseconds.

    Args:
        monkeypatch: The pytest fixture.

    Returns:
        The list the fake recorder appends every seed it is asked for to.
    """
    recorded: list[int] = []

    def record(env_tools, policy_path, seed, max_steps=None, progress=None, proc=None):
        """Stand in for ``record_episode``: one two-frame episode."""
        recorded.append(int(seed))
        rec = {
            "seed": int(seed),
            "success": 0,
            "fell": False,
            "length": 1,
            "termination": "timeout",
        }
        arrays = {
            "full_qpos": np.zeros((2, 3), dtype=np.float64),
            "full_qvel": np.zeros((2, 3), dtype=np.float64),
            "action": np.zeros((1, 3), dtype=np.float32),
            "reward": np.zeros((1, 1), dtype=np.float32),
        }
        return rec, arrays

    monkeypatch.setattr(replay, "EnvTools", lambda task, config: FakeEnvTools())
    monkeypatch.setattr(replay, "record_episode", record)
    return recorded


def test_replay_adds_the_missing_seeds_to_an_existing_version(
    cell, capsys, monkeypatch
):
    """A second seed extends the batch instead of being refused.

    ``replays/vNNN`` is added to: seeds it already holds are kept, the
    missing ones run, and ``metadata.json`` (which describes the substrate,
    not the seed set) is left alone.
    """
    recorded = fake_recordings(monkeypatch)
    batch = cell / "replays" / "v001" / "batch"

    argv = [str(cell), "--version", "1", "--no-video", "--seeds"]
    assert replay.main(argv + ["500"]) == 0
    assert recorded == [500]
    metadata = (batch / "metadata.json").read_text()

    # The seed that is already there is kept; only 501 is rolled out.
    assert replay.main(argv + ["500,501"]) == 0
    assert recorded == [500, 501]
    out = capsys.readouterr().out
    assert "seed 500: already recorded" in out
    assert "adding to" in out
    assert sorted(f.name for f in batch.glob("*.npz")) == [
        "seed500.npz",
        "seed501.npz",
    ]
    assert (batch / "metadata.json").read_text() == metadata

    # Nothing missing at all is a no-op, and --force re-records everything.
    assert replay.main(argv + ["500,501"]) == 0
    assert recorded == [500, 501]
    assert replay.main(argv + ["500,501", "--force"]) == 0
    assert recorded == [500, 501, 500, 501]


def test_replay_writes_the_metadata_of_a_new_batch_only(cell, monkeypatch):
    """``replay_seeds`` returns every requested seed, recorded or kept."""
    recorded = fake_recordings(monkeypatch)
    out_dir = cell / "replays" / "v001"
    policy = cell / "policies" / "v001" / "policy.py"

    env_tools = EnvToolsConfig()
    first = replay.replay_seeds(
        TASK, policy, [501, 500], out_dir, env_tools, video=False
    )
    assert [p.name for p in first] == ["seed501.npz", "seed500.npz"]
    assert replay.finished_batch(out_dir) == out_dir / "batch"

    again = replay.replay_seeds(
        TASK, policy, [500, 502], out_dir, env_tools, video=False
    )
    assert [p.name for p in again] == ["seed500.npz", "seed502.npz"]
    assert recorded == [501, 500, 502]


def test_replay_reports_a_missing_version_on_one_line(cell, capsys):
    """A failure is one readable line and a non-zero status, not a traceback."""
    assert replay.main([str(cell), "--version", "9", "--seeds", "620000"]) == 1
    assert "no policy version 9" in capsys.readouterr().err


@pytest.mark.slow
def test_replay_writes_a_batch_the_viewer_opens(cell):
    """Two short episodes of the template policy, read back by bigym-view."""
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from bigym.vr.viewer import agent_cells, view_demos

    assert (
        replay.main(
            [
                str(cell),
                "--version",
                "1",
                "--seeds",
                "620000-620001",
                "--no-video",
                "--max-steps",
                "30",
            ]
        )
        == 0
    )
    batch = cell / "replays" / "v001" / "batch"
    files = sorted(batch.glob("*.npz"))
    assert [f.name for f in files] == ["seed620000.npz", "seed620001.npz"]
    # The progress file the viewer polls is removed when the run finishes,
    # and never sat where a batch scan would find it.
    assert not (cell / "replays" / "v001" / PROGRESS_NAME).exists()
    assert list(batch.glob(".*")) == []

    metadata = json.loads((batch / "metadata.json").read_text())
    assert metadata["source"]["version"] == 1
    assert metadata["source"]["cell"] == str(cell)

    # the viewer's own loader: discovery, labels, metadata, env, episode store
    assert view_demos.discover_batches(cell) == [batch]
    assert agent_cells.batch_version(batch) == (cell, "replays", 1)
    rows = agent_cells.policy_rows(cell)
    assert [row["version"] for row in rows] == [1]
    assert rows[0]["trigger"] == "submission"
    assert rows[0]["eval_success"] is None
    assert view_demos.batch_labels(cell, [batch]) == ["replays/v001 · 2 eps"]

    loaded = view_demos.load_metadata(batch)
    env = view_demos.build_env(loaded)
    try:
        env.reset(seed=0)
        inner = env.inner_env
        model = inner.model
        store = view_demos.DemoStore(env, files)
        assert store.load(0) is True
        assert store.qpos is not None and store.qvel is not None
        assert store.qpos.shape[1] == model.nq
        assert store.qvel.shape[1] == model.nv
        with np.load(files[0]) as episode:
            steps = int(episode["action"].shape[0])
            assert 0 < steps <= 30
            assert episode["full_qpos"].shape == (steps + 1, model.nq)
            assert episode["full_qvel"].shape == (steps + 1, model.nv)
            assert episode["reward"].shape == (steps, 1)
            assert int(episode["seed"]) == 620000
    finally:
        env.close()


@pytest.mark.slow
def test_cell_evaluation_writes_csv_batch_and_summary(cell, capsys):
    """`evaluate <cell> --version N` fills eval/vNNN and refuses a redo."""
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from bigym.loco.agent import evaluate

    argv = [
        str(cell),
        "--version",
        "1",
        "--episodes",
        "2",
        "--jobs",
        "1",
        "--spot-check",
        "0",
        "--no-video",
    ]
    assert evaluate.main(argv) == 0
    out = cell / "eval" / "v001"

    with (out / "episodes.csv").open() as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == list(evaluate.COLUMNS)
    assert len(rows) == 3
    assert [row[1] for row in rows[1:]] == ["0", "1"]
    assert [row[2] for row in rows[1:]] == ["620000", "620001"]

    summary = json.loads((out / "summary.json").read_text())
    assert summary["version"] == 1
    assert summary["task"] == TASK
    assert summary["episodes"] == 2
    assert 0.0 <= float(summary["success_rate"]) <= 1.0
    assert summary["interface"] == "strict"
    assert summary["protocol_violations"] == []
    assert summary["spotcheck_identical"] is None
    assert summary["fingerprint"]["task"] == TASK
    assert summary["time"]
    assert (out / "smoothness.csv").exists()

    batch = out / "batch"
    assert sorted(f.name for f in batch.glob("*.npz")) == [
        "seed620000.npz",
        "seed620001.npz",
    ]
    assert json.loads((batch / "metadata.json").read_text())["source"]["version"] == 1

    capsys.readouterr()
    assert evaluate.main(argv) == 1
    message = capsys.readouterr().err
    assert "already evaluated" in message and str(out / "summary.json") in message
    assert evaluate.main(argv + ["--force"]) == 0


def no_simulator(monkeypatch) -> None:
    """Make building an environment or running a block fail the test."""
    from bigym.loco.agent import evaluate

    def refuse(*args, **kwargs):
        raise AssertionError("a rejected policy must not reach the simulator")

    monkeypatch.setattr(evaluate, "run_block", refuse)
    monkeypatch.setattr(evaluate, "EnvTools", refuse)


def read_rejected_cell(cell: Path, episodes: int) -> dict:
    """Check the rejected record of version 1 and return its summary."""
    from bigym.loco.agent import evaluate, launch, registry
    from bigym.vr.viewer import agent_cells

    out = cell / "eval" / "v001"
    with (out / "episodes.csv").open() as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == list(evaluate.COLUMNS)
    assert [row[2] for row in rows[1:]] == [str(620000 + i) for i in range(episodes)]
    assert {(row[3], row[4], row[6]) for row in rows[1:]} == {("0", "0", "rejected")}
    summary = json.loads((out / "summary.json").read_text())
    assert summary["success_rate"] == 0.0
    assert summary["episodes"] == episodes
    assert summary["version"] == 1
    assert summary["terminations"] == {"rejected": episodes}
    assert summary["batch"] is None and summary["spot_check"] is None
    assert "videos" not in summary
    assert not (out / "batch").exists()
    # The readers of an evaluation take a rejected one as a score of 0.
    assert f"0.00 ({episodes} eps)" in snapshots.policy_table(cell)
    assert agent_cells.eval_success(cell, 1) == 0.0
    state = registry.cell_state(cell.parent, cell.name)
    assert "SCORE 0" in state and "REJECTED" in state
    assert launch.rejection(out / "summary.json") == summary["rejected"]
    return summary


def test_a_policy_that_imports_the_simulator_is_rejected_unrun(
    cell, monkeypatch, capsys
):
    """The static scan rejects the version before any episode: success 0."""
    from bigym.loco.agent import evaluate

    version = cell / "policies" / "v001"
    (version / "planner.py").write_text("import os\n\nfrom mujoco import MjData\n")
    no_simulator(monkeypatch)
    assert evaluate.main([str(cell), "--version", "1", "--episodes", "3"]) == 0
    summary = read_rejected_cell(cell, 3)
    assert summary["rejected"] == "planner.py:3 imports mujoco"
    assert "planner.py:3 imports mujoco" in capsys.readouterr().err


def test_a_policy_refused_at_run_time_is_rejected_too(cell, monkeypatch):
    """A refusal in the policy process mid-block also scores 0 and drops the batch."""
    from bigym.loco.agent import evaluate
    from bigym.loco.agent.policy_process import ForbiddenImport

    no_simulator(monkeypatch)

    def refused(block, **kwargs):
        batch = kwargs["batch_dir"]
        batch.mkdir(parents=True)
        (batch / "seed620000.npz").write_bytes(b"")  # an episode run before it
        kwargs["on_episode"](
            {
                "episode": 0,
                "seed": 620000,
                "success": 1,
                "length": 5,
                "reward": 1.0,
                "termination": "success",
                "fell": False,
            }
        )
        raise ForbiddenImport("policy.py:9 imports bigym")

    monkeypatch.setattr(evaluate, "run_block", refused)
    argv = [str(cell), "--version", "1", "--episodes", "2", "--jobs", "1"]
    assert evaluate.main(argv) == 0
    summary = read_rejected_cell(cell, 2)
    assert summary["rejected"] == "policy.py:9 imports bigym"


def test_the_episode_limit_is_the_time_limit_capped_by_max_steps():
    """`.progress.json` needs a denominator: the env's limit, or a smaller cap."""
    tools: Any = SteppingEnvTools(time_limit=300)
    assert replay.episode_limit(tools) == 300
    assert replay.episode_limit(tools, 30) == 30
    assert replay.episode_limit(tools, 900) == 300  # a bigger cap never applies
    tools.time_limit = 0
    assert replay.episode_limit(tools, 30) == 30
    assert replay.episode_limit(tools) == 0


def test_a_running_episode_writes_its_step_count(tmp_path):
    """`.progress.json` follows the steps, every ``every`` of them."""
    tools: Any = SteppingEnvTools(time_limit=60)
    out = tmp_path / "replays" / "v001"
    progress = RunProgress(out, episodes=2, every=10)
    tools.watch(progress.path)

    rec, _ = replay.record_episode(
        tools, write_still_policy(tmp_path), 620000, None, progress
    )
    assert rec["length"] == 60
    assert progress.path.name == ".progress.json"  # batch scans ignore it
    # Snapshot n is the file as it stood after n steps: one write when the
    # episode starts, then one per ten steps and nothing in between.
    assert tools.seen[0] == {
        "seed": 620000,
        "step": 0,
        "max_steps": 60,
        "episode": 0,
        "episodes": 2,
        "done": False,
        "pid": os.getpid(),
    }
    assert tools.seen[10]["step"] == 10 and tools.seen[10]["done"] is False
    assert tools.seen[29]["step"] == 20
    assert tools.seen[50]["step"] == 50
    # At the end the episode is counted and marked done.
    final = json.loads(progress.path.read_text())
    assert final["step"] == 60 and final["max_steps"] == 60
    assert final["episode"] == 1 and final["episodes"] == 2
    assert final["done"] is True

    progress.clear()
    assert not progress.path.exists()
    progress.clear()  # removing it twice is not an error


def test_max_steps_is_what_the_progress_file_counts_towards(tmp_path):
    """A short smoke run reports its own cap, not the env's time limit."""
    tools: Any = SteppingEnvTools(time_limit=600)
    progress = RunProgress(tmp_path / "replays" / "v001", episodes=1, every=5)
    replay.record_episode(tools, write_still_policy(tmp_path), 620007, 20, progress)
    final = json.loads(progress.path.read_text())
    assert (final["max_steps"], final["step"], final["seed"]) == (20, 20, 620007)
    assert final["episode"] == 1 and final["done"] is True


def test_the_evaluation_hook_counts_finished_episodes(tmp_path):
    """Cell-mode evaluation writes the same file, one update per episode."""
    from bigym.loco.agent import evaluate

    rows: list[dict] = []
    live = type("Live", (), {"append": lambda _self, rec: rows.append(rec)})()
    progress = RunProgress(tmp_path, episodes=100)
    on_episode = evaluate.episode_hook(live, progress)

    on_episode({"seed": 620000, "length": 210})
    first = json.loads(progress.path.read_text())
    assert (first["episode"], first["episodes"]) == (1, 100)
    assert (first["seed"], first["step"], first["max_steps"]) == (620000, 210, 0)
    on_episode({"seed": 620001, "length": 90})
    assert json.loads(progress.path.read_text())["episode"] == 2
    # The CSV writer still sees every record, in order.
    assert [r["seed"] for r in rows] == [620000, 620001]
    assert evaluate.episode_hook(None, None) is None


def test_write_json_survives_concurrent_writers(tmp_path):
    """Several processes writing one metadata.json never trip over a temp name."""
    import multiprocessing as mp

    target = tmp_path / "batch" / "metadata.json"

    def worker(k: int) -> None:
        for i in range(200):
            write_json(target, {"writer": k, "i": i})

    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=worker, args=(k,)) for k in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    assert json.loads(target.read_text())["i"] == 199
    assert not list(target.parent.glob(".*.tmp"))
