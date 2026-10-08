"""Fast tests for the policy version watcher (no simulator, no network).

The watcher turns a live agent sandbox into ``<cell>/policies/``: one
directory per distinct ``policy.py`` content, an index the demo viewer reads
while the session runs, the episodes the server ledger says each version ran,
and what the session had spent when the version was written.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from bigym.loco.agent import snapshots
from bigym.loco.agent.snapshots import PolicyWatcher

POLICY_V1 = "class Policy:\n    def act(self, obs, tools):\n        return 1\n"
POLICY_V2 = "class Policy:\n    def act(self, obs, tools):\n        return 2\n"
POLICY_V3 = "class Policy:\n    def act(self, obs, tools):\n        return 3\n"
HELPER = "VALUE = 1\n"
# What an agent's own policy.py looks like: third-party imports, the helpers
# it wrote, and scratch scripts next to it that it does not import.
POLICY_IMPORTING = (
    "import numpy as np\n"
    "import control\n"
    "from vision import find_plate\n"
    "import scratch_probe as probe  # noqa: F401\n"
    "\n"
    "class Policy:\n"
    "    def act(self, obs, tools):\n"
    "        return control.hold(obs, find_plate(obs))\n"
)
CONTROL = "import numpy as np\nfrom geom import clip\n\n\ndef hold(obs, x):\n    return clip(x)\n"
GEOM = "def clip(x):\n    return x\n"
VISION = "def find_plate(obs):\n    return 0\n"
SCRATCH = "print('a scratch script the policy does not import')\n"
UNUSED = "import geom  # a scratch script that imports a helper\n"


def write_ledger(cell: Path, events: list[dict]) -> Path:
    """Append events to the cell's ledger, as the server does."""
    cell.mkdir(parents=True, exist_ok=True)
    path = cell / "ledger.jsonl"
    with open(path, "a") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
    return path


def episode_end(ts: float, seed: int, success: int, budget_used: int = 0) -> dict:
    """Return one finished episode, in the shape the server logs it."""
    return {
        "ts": float(ts),
        "worker": 0,
        "event": "episode_end",
        "seed": int(seed),
        "success": int(success),
        "length": 100,
        "reward": float(success),
        "termination": "success" if success else "timeout",
        "fell": False,
        "budget_used": int(budget_used),
    }


def make_cell(tmp_path: Path) -> tuple[Path, Path]:
    """Return ``(cell, sandbox)`` of an empty run directory."""
    cell = tmp_path / "reach_target_single"
    sandbox = cell / "sandbox"
    sandbox.mkdir(parents=True)
    return cell, sandbox


def make_watcher(tmp_path: Path, **kw) -> tuple[PolicyWatcher, Path, Path]:
    """Return a watcher over a fresh cell, plus the cell and sandbox paths."""
    cell, sandbox = make_cell(tmp_path)
    watcher = PolicyWatcher(
        sandbox, cell / "policies", task="reach_target_single", **kw
    )
    return watcher, cell, sandbox


def write_run(sandbox: Path, stamp: str, policy_text: str, successes) -> Path:
    """Write what the agent's runner leaves behind for one run."""
    runs = sandbox / "runs"
    runs.mkdir(exist_ok=True)
    (runs / f"{stamp}_policy.py").write_text(policy_text)
    records = {
        "policy_snapshot": f"{stamp}_policy.py",
        "seeds": list(range(len(successes))),
        "episodes": [
            {"seed": i, "success": int(s), "length": 10, "reward": float(s)}
            for i, s in enumerate(successes)
        ],
    }
    path = runs / f"{stamp}.json"
    path.write_text(json.dumps(records))
    return path


def read_index(cell: Path) -> dict:
    """Read the written index.json of a cell."""
    return json.loads((cell / "policies" / "index.json").read_text())


def test_every_new_content_hash_becomes_a_version(tmp_path):
    """Three distinct writes of policy.py are three versions, helpers copied."""
    watcher, cell, sandbox = make_watcher(tmp_path)
    (sandbox / "helper.py").write_text(HELPER)
    (sandbox / "run_episodes.py").write_text("# the harness entry point\n")
    for text in (POLICY_V1, POLICY_V2, POLICY_V3):
        (sandbox / "policy.py").write_text(text)
        assert watcher.poll() is True

    index = read_index(cell)
    assert index["task"] == "reach_target_single"
    versions = index["versions"]
    assert [v["version"] for v in versions] == [1, 2, 3]
    assert [v["dir"] for v in versions] == ["v001", "v002", "v003"]
    assert [v["trigger"] for v in versions] == ["write"] * 3
    assert len({v["sha256"] for v in versions}) == 3
    assert versions[0]["sha256"] == snapshots.sha256_text(POLICY_V1.encode())
    for entry, text in zip(versions, (POLICY_V1, POLICY_V2, POLICY_V3), strict=True):
        body = cell / "policies" / entry["dir"]
        assert (body / "policy.py").read_text() == text
        # helper modules next to policy.py ride along; the harness entry point
        # is not the agent's code
        assert (body / "helper.py").read_text() == HELPER
        assert not (body / "run_episodes.py").exists()
        # ... but this policy imports nothing of its own
        assert entry["helpers"] == []
        assert entry["train_run"] is None
        assert entry["train_success"] is None


def test_rewriting_the_same_content_adds_nothing(tmp_path):
    """A save that changes no byte is not a new version."""
    watcher, cell, sandbox = make_watcher(tmp_path)
    (sandbox / "policy.py").write_text(POLICY_V1)
    assert watcher.poll() is True
    assert watcher.poll() is False
    (sandbox / "policy.py").write_text(POLICY_V1)
    assert watcher.poll() is False
    assert [v["version"] for v in read_index(cell)["versions"]] == [1]


def test_a_run_upgrades_its_version_and_carries_the_score(tmp_path):
    """The runner's snapshot matches a version by hash: trigger run + scores."""
    watcher, cell, sandbox = make_watcher(tmp_path)
    (sandbox / "policy.py").write_text(POLICY_V1)
    watcher.poll()
    (sandbox / "policy.py").write_text(POLICY_V2)
    watcher.poll()
    write_run(sandbox, "20260102_030405", POLICY_V1, [1, 0, 1, 0, 0])
    assert watcher.poll() is True

    versions = read_index(cell)["versions"]
    assert [v["trigger"] for v in versions] == ["run", "write"]
    assert versions[0]["train_run"] == "sandbox/runs/20260102_030405.json"
    assert versions[0]["train_success"] == pytest.approx(0.4)
    assert versions[0]["train_episodes"] == 5
    assert versions[1]["train_run"] is None
    # the same run file is not read twice
    assert watcher.poll() is False


def test_a_run_of_unseen_content_becomes_its_own_version(tmp_path):
    """Edit + run between two polls: the run's snapshot is the version."""
    watcher, cell, sandbox = make_watcher(tmp_path)
    (sandbox / "policy.py").write_text(POLICY_V1)
    watcher.poll()
    # the agent edits and runs before the next poll
    (sandbox / "policy.py").write_text(POLICY_V2)
    write_run(sandbox, "20260102_040506", POLICY_V2, [1, 1])
    watcher.poll()

    versions = read_index(cell)["versions"]
    assert [v["version"] for v in versions] == [1, 2]
    assert versions[1]["trigger"] == "run"
    assert versions[1]["sha256"] == snapshots.sha256_text(POLICY_V2.encode())
    assert versions[1]["train_success"] == pytest.approx(1.0)
    assert (cell / "policies" / "v002" / "policy.py").read_text() == POLICY_V2


def test_finalize_relabels_the_last_version(tmp_path):
    """The content at session end is the submission, not a fourth copy."""
    watcher, cell, sandbox = make_watcher(tmp_path)
    (sandbox / "policy.py").write_text(POLICY_V1)
    watcher.poll()
    (sandbox / "policy.py").write_text(POLICY_V2)
    watcher.poll()
    entry = watcher.finalize()

    versions = read_index(cell)["versions"]
    assert [v["trigger"] for v in versions] == ["write", "submission"]
    assert entry is not None
    assert entry["version"] == 2


def test_finalize_records_content_no_poll_ever_saw(tmp_path):
    """A last-second edit still lands as the submission version."""
    watcher, cell, sandbox = make_watcher(tmp_path)
    (sandbox / "policy.py").write_text(POLICY_V1)
    watcher.poll()
    (sandbox / "policy.py").write_text(POLICY_V3)
    watcher.finalize()

    versions = read_index(cell)["versions"]
    assert [v["version"] for v in versions] == [1, 2]
    assert versions[1]["trigger"] == "submission"
    assert (cell / "policies" / "v002" / "policy.py").read_text() == POLICY_V3


def test_finalize_also_picks_up_the_last_run(tmp_path):
    """Runs written after the last poll are still attached at shutdown."""
    watcher, cell, sandbox = make_watcher(tmp_path)
    (sandbox / "policy.py").write_text(POLICY_V1)
    watcher.poll()
    write_run(sandbox, "20260102_050607", POLICY_V1, [0, 0, 1, 1])
    watcher.finalize()

    entry = read_index(cell)["versions"][0]
    assert entry["trigger"] == "submission"
    assert entry["train_success"] == pytest.approx(0.5)
    assert entry["train_episodes"] == 4


def test_the_index_is_never_seen_half_written(tmp_path):
    """The viewer polls index.json live: every read parses."""
    watcher, cell, sandbox = make_watcher(tmp_path, poll_seconds=0.01)
    (sandbox / "policy.py").write_text(POLICY_V1)
    watcher.start()
    index_path = cell / "policies" / "index.json"
    stop = threading.Event()
    seen: list[int] = []
    errors: list[str] = []

    def reader():
        while not stop.is_set():
            try:
                loaded = json.loads(index_path.read_text())
            except FileNotFoundError:
                continue
            except ValueError as exc:
                errors.append(str(exc))
                continue
            seen.append(len(loaded["versions"]))

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        for i in range(12):
            (sandbox / "policy.py").write_text(f"# version {i}\n{POLICY_V1}")
            time.sleep(0.03)
    finally:
        stop.set()
        thread.join(timeout=5)
        watcher.stop()

    assert errors == []
    assert seen and max(seen) >= 2
    assert len(read_index(cell)["versions"]) == max(seen)


def test_load_version_points_at_the_policy_next_to_its_helpers(tmp_path):
    """load_version(cell, n) is a path episode.load_policy can import."""
    watcher, cell, sandbox = make_watcher(tmp_path)
    (sandbox / "helper.py").write_text(HELPER)
    (sandbox / "policy.py").write_text(POLICY_V1)
    watcher.poll()
    (sandbox / "policy.py").write_text(POLICY_V2)
    watcher.poll()

    path = snapshots.load_version(cell, 2)
    assert path == cell / "policies" / "v002" / "policy.py"
    assert path.read_text() == POLICY_V2
    assert (path.parent / "helper.py").exists()
    with pytest.raises(FileNotFoundError, match="version 9"):
        snapshots.load_version(cell, 9)


def test_the_cell_reports_its_task_and_settings(tmp_path):
    """Task and sandbox settings come from the cell, with sane fallbacks."""
    watcher, cell, sandbox = make_watcher(tmp_path)
    (sandbox / "policy.py").write_text(POLICY_V1)
    watcher.poll()
    assert snapshots.read_sandbox_config(cell) == {}
    assert snapshots.cell_task(cell) == "reach_target_single"
    (cell / "sandbox_config.json").write_text(
        json.dumps({"task": "move_plate", "interface": "strict", "tier": "images"})
    )
    assert snapshots.cell_task(cell) == "move_plate"
    assert snapshots.read_sandbox_config(cell)["tier"] == "images"


def test_the_policies_command_prints_the_table(tmp_path, capsys):
    """`bigym-agent policies <cell>` shows versions, triggers and both scores."""
    watcher, cell, sandbox = make_watcher(tmp_path)
    (sandbox / "policy.py").write_text(POLICY_V1)
    watcher.poll()
    write_run(sandbox, "20260102_060708", POLICY_V1, [1, 0])
    watcher.poll()
    eval_dir = cell / "eval" / "v001"
    eval_dir.mkdir(parents=True)
    (eval_dir / "summary.json").write_text(
        json.dumps({"version": 1, "success_rate": 0.25, "episodes": 100})
    )

    assert snapshots.main([str(cell)]) == 0
    out = capsys.readouterr().out
    assert "v001" in out and "run" in out
    assert "0.50 (2 eps)" in out
    assert "0.25 (100 eps)" in out
    assert snapshots.main([str(tmp_path / "not-a-cell")]) == 2


def test_helpers_are_the_modules_the_policy_imports(tmp_path):
    """Every sibling is copied; only the imported ones are listed."""
    watcher, cell, sandbox = make_watcher(tmp_path)
    for name, text in (
        ("control.py", CONTROL),
        ("geom.py", GEOM),
        ("vision.py", VISION),
        ("scratch_probe.py", SCRATCH),
        ("make_variants.py", UNUSED),
    ):
        (sandbox / name).write_text(text)
    (sandbox / "policy.py").write_text(POLICY_IMPORTING)
    watcher.poll()

    entry = read_index(cell)["versions"][0]
    # control.py and vision.py are imported outright, scratch_probe.py under an
    # alias, geom.py only through control.py; numpy is not a sibling, and
    # make_variants.py is a scratch script nothing imports
    assert entry["helpers"] == [
        "control.py",
        "geom.py",
        "scratch_probe.py",
        "vision.py",
    ]
    body = cell / "policies" / entry["dir"]
    for name in ("control.py", "geom.py", "vision.py", "make_variants.py"):
        assert (body / name).is_file(), f"{name} was not copied"


def test_the_ledger_attributes_episodes_to_the_version_that_ran_them(tmp_path):
    """Episodes belong to the policy.py that was on disk when they ran."""
    watcher, cell, sandbox = make_watcher(tmp_path)
    write_ledger(cell, [{"ts": time.time(), "worker": -1, "event": "server_start"}])
    (sandbox / "policy.py").write_text(POLICY_V1)
    watcher.poll()
    # real episodes are minutes apart; the sleeps only keep the synthetic
    # events from landing in the same microsecond as a snapshot
    time.sleep(0.01)
    write_ledger(
        cell,
        [
            episode_end(time.time(), seed=3, success=1),
            episode_end(time.time(), seed=3, success=0),
            episode_end(time.time(), seed=7, success=0),
        ],
    )
    watcher.poll()
    time.sleep(0.01)
    (sandbox / "policy.py").write_text(POLICY_V2)
    watcher.poll()
    time.sleep(0.01)
    write_ledger(cell, [episode_end(time.time(), seed=11, success=1)])
    watcher.poll()

    first, second = read_index(cell)["versions"]
    assert first["train_episodes"] == 3
    assert first["train_success"] == pytest.approx(1 / 3)
    assert first["train_seeds"] == [3, 7]
    assert second["train_episodes"] == 1
    assert second["train_success"] == pytest.approx(1.0)
    assert second["train_seeds"] == [11]


def test_episodes_before_the_first_version_belong_to_nobody(tmp_path):
    """An episode that ran before any snapshot is not attributed."""
    watcher, cell, sandbox = make_watcher(tmp_path)
    write_ledger(cell, [episode_end(time.time() - 60, seed=1, success=1)])
    (sandbox / "policy.py").write_text(POLICY_V1)
    watcher.poll()
    watcher.poll()

    entry = read_index(cell)["versions"][0]
    assert entry["train_episodes"] == 0
    assert entry["train_success"] is None
    assert entry["train_seeds"] == []


def test_the_ledger_overrules_the_runner_but_keeps_its_link(tmp_path):
    """A runner's own score is replaced by the ledger; train_run survives."""
    watcher, cell, sandbox = make_watcher(tmp_path)
    (sandbox / "policy.py").write_text(POLICY_V1)
    watcher.poll()
    time.sleep(0.01)
    write_run(sandbox, "20260102_030405", POLICY_V1, [1, 1, 1, 1])
    write_ledger(cell, [episode_end(time.time(), seed=5, success=0)])
    watcher.poll()

    entry = read_index(cell)["versions"][0]
    assert entry["train_run"] == "sandbox/runs/20260102_030405.json"
    assert entry["train_episodes"] == 1
    assert entry["train_success"] == pytest.approx(0.0)


# A Codex event stream: two commands and the turn's cumulative token counter.
CODEX_STREAM = [
    {"type": "thread.started", "thread_id": "01a09772-9002-7623-8994-d1ceaafb2d44"},
    {
        "type": "item.completed",
        "item": {
            "id": "item_1",
            "type": "command_execution",
            "command": "/bin/sh -lc 'ls docs'",
            "aggregated_output": "api.md\n",
            "exit_code": 0,
        },
    },
    {
        "type": "item.completed",
        "item": {
            "id": "item_2",
            "type": "command_execution",
            "command": "/bin/sh -lc 'python probe.py'",
            "aggregated_output": "",
            "exit_code": 0,
        },
    },
    {
        "type": "turn.completed",
        "usage": {
            "input_tokens": 1000,
            "cached_input_tokens": 600,
            "output_tokens": 50,
            "reasoning_output_tokens": 30,
        },
    },
]


def write_raw(cell: Path, name: str, lines: list[dict]) -> Path:
    """Write a harness event stream under ``<cell>/raw/``."""
    raw = cell / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / name
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def test_a_version_records_what_the_session_had_spent(tmp_path):
    """Commands, tokens, budget and elapsed time at the moment of the write."""
    started = time.time() - 30
    watcher, cell, sandbox = make_watcher(tmp_path, started=started)
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "budget.json").write_text(json.dumps({"used": 4200, "cap": 101000}))
    write_raw(cell, "codex_events.jsonl", CODEX_STREAM)
    (sandbox / "policy.py").write_text(POLICY_V1)
    watcher.poll()

    entry = read_index(cell)["versions"][0]
    assert entry["command_index"] == 2
    assert entry["tokens"] == {
        "input": 400,
        "cached": 600,
        "output": 50,
        "reasoning": 30,
    }
    assert entry["budget_used"] == 4200
    assert entry["elapsed_s"] >= 30.0
    assert entry["ts"] == pytest.approx(snapshots.parse_time(entry["time"]), abs=1.001)


def test_a_version_records_zeros_when_the_cell_has_no_stream(tmp_path):
    """No raw stream and no budget file is not an error, just zeros."""
    watcher, cell, sandbox = make_watcher(tmp_path)
    (sandbox / "policy.py").write_text(POLICY_V1)
    watcher.poll()

    entry = read_index(cell)["versions"][0]
    assert entry["command_index"] == 0
    assert entry["tokens"] == {"input": 0, "cached": 0, "output": 0, "reasoning": 0}
    assert entry["budget_used"] == 0


def claude_stream(first: str, second: str, result: str) -> list[dict]:
    """A Claude Code stream-json session: two commands and the final usage."""
    return [
        {
            "type": "assistant",
            "message": {
                "id": "msg_01",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "Bash",
                        "input": {"command": "ls -la"},
                    }
                ],
            },
            "timestamp": first,
        },
        {
            "type": "assistant",
            "message": {
                "id": "msg_02",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_02",
                        "name": "Bash",
                        "input": {"command": "./python run_episodes.py"},
                    }
                ],
            },
            "timestamp": second,
        },
        {
            "type": "result",
            "subtype": "success",
            "num_turns": 9,
            "duration_ms": 1000,
            "usage": {
                "input_tokens": 100,
                "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 900,
                "output_tokens": 70,
            },
            "timestamp": result,
        },
    ]


def old_entry(version: int, sha: str, when: str, trigger: str) -> dict:
    """An index entry in the shape written before the derived fields existed."""
    return {
        "version": version,
        "dir": f"v{version:03d}",
        "sha256": sha,
        "time": when,
        "trigger": trigger,
        "helpers": ["control.py", "geom.py", "make_variants.py", "vision.py"],
        "train_run": None,
        "train_success": None,
        "train_episodes": None,
    }


def make_old_cell(tmp_path: Path) -> tuple[Path, list[float]]:
    """Write a cell whose index predates ts, train_seeds and provenance."""
    cell = tmp_path / "move_plate"
    policies = cell / "policies"
    base = time.time() - 600
    times = [base + 60, base + 300]
    for number, text in ((1, POLICY_IMPORTING), (2, POLICY_V2)):
        body = policies / f"v{number:03d}"
        body.mkdir(parents=True)
        (body / "policy.py").write_text(text)
        for name, source in (
            ("control.py", CONTROL),
            ("geom.py", GEOM),
            ("vision.py", VISION),
            ("make_variants.py", UNUSED),
        ):
            (body / name).write_text(source)
    index = {
        "task": "move_plate",
        "versions": [
            old_entry(
                1,
                snapshots.sha256_text(POLICY_IMPORTING.encode()),
                snapshots.stamp_of(times[0]),
                "write",
            ),
            old_entry(
                2,
                snapshots.sha256_text(POLICY_V2.encode()),
                snapshots.stamp_of(times[1]),
                "submission",
            ),
        ],
    }
    (policies / "index.json").write_text(json.dumps(index, indent=1))
    write_ledger(
        cell,
        [
            {"ts": base, "worker": -1, "event": "server_start", "task": "move_plate"},
            episode_end(times[0] + 10, seed=2, success=1, budget_used=500),
            episode_end(times[0] + 20, seed=4, success=0, budget_used=900),
            episode_end(times[1] + 10, seed=6, success=1, budget_used=2000),
        ],
    )
    write_raw(
        cell,
        "claude_stream.jsonl",
        claude_stream(
            snapshots.stamp_of(times[0] - 5),
            snapshots.stamp_of(times[0] + 5),
            snapshots.stamp_of(times[1] - 5),
        ),
    )
    return cell, times


def test_rebuild_upgrades_an_index_that_predates_the_new_fields(tmp_path):
    """--rebuild fills in the derived fields without touching the versions."""
    cell, times = make_old_cell(tmp_path)
    before = {
        path.relative_to(cell): path.read_bytes()
        for path in sorted((cell / "policies").rglob("*.py"))
    }

    assert snapshots.main([str(cell), "--rebuild"]) == 0
    first, second = read_index(cell)["versions"]

    # the recorded facts are untouched
    assert first["sha256"] == snapshots.sha256_text(POLICY_IMPORTING.encode())
    assert [v["trigger"] for v in (first, second)] == ["write", "submission"]
    after = {
        path.relative_to(cell): path.read_bytes()
        for path in sorted((cell / "policies").rglob("*.py"))
    }
    assert after == before

    # the derived ones are rebuilt
    assert first["ts"] == pytest.approx(times[0], abs=1.001)
    assert first["helpers"] == ["control.py", "geom.py", "vision.py"]
    assert second["helpers"] == []
    assert first["train_episodes"] == 2
    assert first["train_success"] == pytest.approx(0.5)
    assert first["train_seeds"] == [2, 4]
    assert second["train_episodes"] == 1
    assert second["train_seeds"] == [6]

    # provenance: the stream's commands up to each version's time, the budget
    # and the elapsed time from the ledger
    assert first["command_index"] == 1
    assert second["command_index"] == 2
    assert first["budget_used"] == 0
    assert second["budget_used"] == 900
    assert first["elapsed_s"] == pytest.approx(times[0] - (times[0] - 60), abs=1.001)
    assert second["elapsed_s"] > first["elapsed_s"]
    # Claude reports its tokens once, at the end of the session: the version
    # written before that point cannot know them.
    assert first["tokens"] == {"input": 0, "cached": 0, "output": 0, "reasoning": 0}
    assert second["tokens"] == {
        "input": 120,
        "cached": 900,
        "output": 70,
        "reasoning": 0,
    }


def test_rebuild_is_idempotent(tmp_path):
    """Rebuilding twice writes the same index."""
    cell, _ = make_old_cell(tmp_path)
    assert snapshots.main([str(cell), "--rebuild"]) == 0
    once = read_index(cell)
    assert snapshots.main([str(cell), "--rebuild"]) == 0
    assert read_index(cell) == once
