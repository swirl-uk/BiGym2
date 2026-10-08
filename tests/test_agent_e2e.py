"""One whole ``bigym-agent run`` session, driven by a scripted stand-in agent.

``--harness custom`` runs any program as the agent, so a session can be exercised
end to end without a model or a container: the program below reads the prompt,
writes ``policy.py`` twice and runs each version on the development seeds,
exactly as an agent would. The test then checks what the session left in the
cell -- the sandbox, the ledger, two policy versions, the transcript, the
verdict and the hidden-seed evaluation of the submission.

The run is real (a MuJoCo environment server, EGL rendering, several hundred
control steps per episode), so it is marked ``slow``::

    MUJOCO_GL=egl uv run pytest tests/test_agent_e2e.py -q --run-slow

The root must be short: the environment server binds unix sockets inside it,
and those paths are limited to 108 bytes, so the test makes its own directory
under ``/tmp`` instead of using pytest's much longer ``tmp_path``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# The scripted agent, written to a file and named by --command. It behaves like
# a (very unimaginative) coding agent: read PROMPT.md, submit the template to
# see what it scores, then submit a policy of its own, running the development
# seeds after each edit. It talks to the session only through the three
# variables the launcher puts in its environment.
AGENT_SOURCE = '''
import json
import os
import subprocess
import sys
from pathlib import Path

# The second submission: hold the pose the episode starts in, and walk one
# joint of the left arm away from it by a small, bounded amount. The action is
# [base (4, or 5 with torso pitch), left arm (7), right arm (7), left gripper,
# right gripper], so the left arm starts 16 entries from the end whatever the
# layout is.
POLICY = """
import numpy as np


class Policy:
    def reset(self, obs, tools):
        self.hold = np.asarray(tools.hold_action(), dtype=np.float32)
        self.step = 0

    def act(self, obs, tools):
        raw = self.hold.copy()
        arm = len(raw) - 16
        raw[arm + 1] += min(self.step, 100) * 0.002
        self.step += 1
        return raw
"""


def run_episodes(sandbox, seeds):
    """Run the current policy.py on these seeds, the way the prompt says to."""
    spec = ",".join(str(s) for s in seeds)
    done = subprocess.run(
        ["./python", "run_episodes.py", "--seeds", spec],
        cwd=str(sandbox),
        capture_output=True,
        text=True,
    )
    print(f"$ ./python run_episodes.py --seeds {spec}")
    print(done.stdout, end="")
    if done.returncode != 0:
        print(done.stderr, end="", file=sys.stderr)
        print(f"run_episodes.py exited {done.returncode}")
    return done.returncode


def main():
    sandbox = Path(os.environ["AGENT_SANDBOX"])
    cell = Path(os.environ["BIGYM_AGENT_CELL"])
    prompt = Path(os.environ["BIGYM_AGENT_PROMPT"]).read_text()
    print(f"read the prompt: {len(prompt.splitlines())} lines from {sandbox.name}")
    print(f"cell: {cell}")
    seeds = json.loads((sandbox / "seeds.json").read_text())[:2]
    print(f"development seeds: {seeds}")
    policy = sandbox / "policy.py"
    # First submission: the template as it was handed over, to see the loop work.
    policy.write_text(policy.read_text())
    run_episodes(sandbox, seeds)
    # Second submission: a policy of its own.
    policy.write_text(POLICY)
    run_episodes(sandbox, seeds)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

TASK = "reach_target_single"


def write_agent(directory: Path) -> Path:
    """Write the scripted agent to a file and return its path.

    Args:
        directory: Where to write ``agent.py``.

    Returns:
        The path of the written program.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "agent.py"
    path.write_text(AGENT_SOURCE)
    return path


def run_cli(*argv: str) -> subprocess.CompletedProcess:
    """Run ``bigym-agent`` in a subprocess and return the finished process.

    Args:
        *argv: The subcommand and its arguments.

    Returns:
        The completed process, with stdout and stderr captured.
    """
    return subprocess.run(
        [sys.executable, "-m", "bigym.loco.agent", *argv],
        capture_output=True,
        text=True,
        env=dict(os.environ, MUJOCO_GL="egl"),
    )


@pytest.mark.slow
def test_custom_agent_session_end_to_end():
    """A scripted agent runs a whole session, and the cell holds every artefact."""
    root = Path(tempfile.mkdtemp(dir="/tmp", prefix="bge-"))
    try:
        agent = write_agent(root / "agent")
        done = run_cli(
            "run",
            "--task",
            TASK,
            "--root",
            str(root),
            "--harness",
            "custom",
            "--command",
            f"{sys.executable} {agent}",
            "--workers",
            "1",
            "--budget-steps",
            "6000",
            "--demo",
            "none",
            "--eval-episodes",
            "2",
            "--spot-check",
            "0",
            "--eval-jobs",
            "1",
        )
        cell = root / TASK
        assert done.returncode == 0, done.stdout + done.stderr

        # the cell layout: the sandbox the agent saw, the server's own files,
        # the transcript, the policy versions and the evaluation
        for name in (
            "run.json",
            "sandbox_config.json",
            "allowed_seeds.json",
            "initial_policy.py",
            "ledger.jsonl",
            "budget.json",
            "server.log",
            "transcript.jsonl",
            "transcript.md",
        ):
            assert (cell / name).is_file(), f"{name} is missing from {cell}"
        for name in ("sandbox", "raw", "policies", "eval"):
            assert (cell / name).is_dir(), f"{name}/ is missing from {cell}"
        for name in ("PROMPT.md", "policy.py", "run_episodes.py", "seeds.json"):
            assert (cell / "sandbox" / name).is_file()
        assert (cell / "sandbox" / "harness" / "runner.py").is_file()
        assert (cell / "sandbox" / "runs").is_dir()

        # the session: a custom agent, no model, the command on record
        run = json.loads((cell / "run.json").read_text())
        assert run["harness"] == "custom"
        assert run["model"] is None
        assert run["command"] == f"{sys.executable} {agent}"
        assert run["isolation"] == "soft"
        assert run["image"] is None
        assert run["exit"] == 0
        assert run["verdict"]["state"] == "ok", run["verdict"]
        assert run["budget"]["used"] > 0
        assert run["usage"] == {"input": 0, "cached": 0, "output": 0, "reasoning": 0}
        assert run["evaluation"]["version"] == 1

        # the two policies the agent wrote: the template it ran, then its own,
        # which is what the session submitted
        index = json.loads((cell / "policies" / "index.json").read_text())
        assert index["task"] == TASK
        versions = index["versions"]
        assert [v["version"] for v in versions] == [0, 1]
        assert [v["trigger"] for v in versions] == ["run", "submission"]
        for entry in versions:
            assert entry["train_run"], entry
            assert entry["train_episodes"] == 2
            assert entry["train_success"] is not None
            assert (cell / "policies" / entry["dir"] / "policy.py").is_file()
        assert versions[0]["sha256"] != versions[1]["sha256"]
        last = index["versions"][-1]["dir"]
        submission = (cell / "policies" / last / "policy.py").read_text()
        assert submission == (cell / "sandbox" / "policy.py").read_text()
        assert "class Policy" in submission

        # the transcript: the command printed what it did, and it was rendered
        transcript = (cell / "transcript.jsonl").read_text().splitlines()
        assert [json.loads(line)["kind"] for line in transcript] == ["output"]
        markdown = (cell / "transcript.md").read_text()
        assert markdown.startswith(f"# {TASK} (custom)")
        assert "development seeds" in markdown
        assert (cell / "raw" / "custom_stdout.txt").is_file()
        assert (cell / "raw" / "custom_stderr.txt").is_file()

        # the hidden-seed evaluation of the submission
        summary = json.loads((cell / "eval" / last / "summary.json").read_text())
        assert summary["version"] == 1
        assert summary["task"] == TASK
        assert summary["episodes"] == 2
        assert 0.0 <= summary["success_rate"] <= 1.0
        assert summary["protocol_violations"] == []
        rows = (cell / "eval" / last / "episodes.csv").read_text().splitlines()
        assert len(rows) == 3  # a header and one row per episode
        batch = cell / "eval" / last / "batch"
        assert (batch / "metadata.json").is_file()
        assert len(list(batch.glob("*.npz"))) == 2

        # the replay the viewer's policy panel runs, on one hidden seed
        replayed = run_cli(
            "replay",
            str(cell),
            "--version",
            str(versions[-1]["version"]),
            "--seeds",
            "620000",
            "--no-video",
        )
        assert replayed.returncode == 0, replayed.stdout + replayed.stderr
        assert (cell / "replays" / last / "batch" / "seed620000.npz").is_file()

        # and the version listing the launcher's report reads
        listed = run_cli("policies", str(cell))
        assert listed.returncode == 0, listed.stdout + listed.stderr
        assert last in listed.stdout
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.slow
def test_a_session_is_scored_in_the_environment_it_ran_against():
    """Non-default environment settings reach the server and the evaluation."""
    from bigym.loco.agent.cli import EnvToolsConfig

    root = Path(tempfile.mkdtemp(dir="/tmp", prefix="bge-"))
    try:
        agent = write_agent(root / "agent")
        done = run_cli(
            "run",
            "--task",
            TASK,
            "--root",
            str(root),
            "--harness",
            "custom",
            "--command",
            f"{sys.executable} {agent}",
            "--workers",
            "1",
            "--budget-steps",
            "6000",
            "--demo",
            "none",
            "--eval-episodes",
            "1",
            "--spot-check",
            "0",
            "--eval-jobs",
            "1",
            "--interface",
            "tools",
            "--no-ik",
            "--slew",
            "--accel",
            "0.02",
            "--lowpass",
            "0.5",
            "--joint-vmax",
            "5.0",
        )
        cell = root / TASK
        assert done.returncode == 0, done.stdout + done.stderr
        expected = EnvToolsConfig(
            interface="tools",
            ik=False,
            slew=True,
            accel=0.02,
            lowpass=0.5,
            joint_vmax=5.0,
        )
        config = json.loads((cell / "sandbox_config.json").read_text())
        assert EnvToolsConfig(**config["env_tools"]) == expected

        ledger = [json.loads(line) for line in (cell / "ledger.jsonl").open()]
        start = next(e for e in ledger if e["event"] == "server_start")
        assert start["interface"] == "tools" and start["ik"] is False
        assert (start["slew"], start["accel"], start["lowpass"]) == (True, 0.02, 0.5)

        summary = json.loads((cell / "eval" / "v001" / "summary.json").read_text())
        assert summary["episodes"] == 1
        assert summary["interface"] == "tools"
        assert summary["ik"] is False and summary["hand_pos"] is True
        assert summary["slew"] == {
            "joint_rad_per_s": 5.0,
            "base_per_s": 0.7,
            "joint_accel_rad_per_step2": 0.02,
            "lowpass_alpha": 0.5,
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)
