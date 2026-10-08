"""The server records the agent's own episodes as replay batches.

Everything the agent runs during a session is written under
``<cell>/dev/vNNN_<stamp>/batch/`` in the format the demo viewer reads, one
directory per policy version, so a session can be watched while it happens.
This test starts a real environment server, drives it through the client the
agent uses, stops it the way the launcher does, and checks what is left on
disk.

The server builds a MuJoCo environment and renders with EGL, so the test is
slow::

    MUJOCO_GL=egl uv run pytest tests/test_agent_dev_recording.py -q --run-slow

The root must be short: the server binds unix sockets inside the sandbox and
those paths are limited to 108 bytes, so the test makes its own directory
under ``/tmp`` rather than using pytest's much longer ``tmp_path``.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

TASK = "reach_target_single"
STEPS = 30
FIRST_SEED = 3
SECOND_SEED = 4
POLICY = "class Policy:\n    def act(self, obs, tools):\n        return None\n"


def start_server(cell: Path, sandbox: Path) -> subprocess.Popen:
    """Start ``bigym-agent serve`` on a sandbox and wait for its worker.

    Args:
        cell: The run directory (the server's ledger directory).
        sandbox: The sandbox the client connects through.

    Returns:
        The running process, with one worker ready.

    Raises:
        RuntimeError: The server died or never announced a worker.
    """
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "bigym.loco.agent",
            "serve",
            "--task",
            TASK,
            "--sandbox",
            str(sandbox),
            "--ledger-dir",
            str(cell),
            "--workers",
            "1",
            "--budget-steps",
            "5000",
            "--reset-cost",
            "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=dict(os.environ, MUJOCO_GL="egl"),
    )
    deadline = time.time() + 600
    lines: list[str] = []
    assert proc.stdout is not None
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        lines.append(line)
        if "ready on" in line:
            return proc
    proc.kill()
    raise RuntimeError("the server never became ready:\n" + "".join(lines))


def run_episodes(sandbox: Path) -> None:
    """Run two short episodes, the second one left unfinished.

    The first is abandoned by resetting the environment under it, which is
    what an agent does when a trial is going nowhere; the second is still
    running when the server is asked to stop.

    Args:
        sandbox: The sandbox holding ``sock/``.
    """
    from bigym.loco.agent.client import connect

    env = connect(sandbox, timeout_s=300)
    try:
        for seed in (FIRST_SEED, SECOND_SEED):
            env.reset(seed)
            hold = env.hold_action()
            for _ in range(STEPS):
                _, _, done, _ = env.step(hold)
                assert not done, "the episode ended sooner than the test expects"
    finally:
        env.close()


def stop_server(proc: subprocess.Popen) -> str:
    """Stop the server the way the launcher does and return its output.

    Args:
        proc: The running server.

    Returns:
        Everything the server printed.
    """
    proc.send_signal(signal.SIGTERM)
    try:
        out = proc.communicate(timeout=60)[0]
    except subprocess.TimeoutExpired:
        proc.kill()
        out = proc.communicate()[0]
    return out or ""


@pytest.fixture(scope="module")
def recorded_cell():
    """Run one short session and yield its cell directory."""
    root = Path(tempfile.mkdtemp(dir="/tmp", prefix="bgd-"))
    cell = root / TASK
    sandbox = cell / "sandbox"
    sandbox.mkdir(parents=True)
    (sandbox / "policy.py").write_text(POLICY)
    proc = start_server(cell, sandbox)
    try:
        run_episodes(sandbox)
    finally:
        output = stop_server(proc)
    try:
        yield cell, output
    finally:
        shutil.rmtree(root, ignore_errors=True)


def batch_of(cell: Path) -> Path:
    """Return the one development batch directory of a cell.

    Args:
        cell: The run directory.

    Returns:
        The ``dev/vNNN_<stamp>/batch`` directory.
    """
    found = sorted((cell / "dev").glob("v*/batch"))
    assert len(found) == 1, f"expected one dev batch, found {found}"
    return found[0]


@pytest.mark.slow
def test_every_episode_lands_in_the_version_that_ran_it(recorded_cell):
    """Both episodes are written under the version the sandbox held."""
    cell, output = recorded_cell
    batch = batch_of(cell)
    # the watcher recorded the sandbox's policy.py as v001 before the first
    # reset, so both episodes belong to it
    assert batch.parent.name.startswith("v001_"), output
    stamp = batch.parent.name.split("_", 1)[1]
    assert len(stamp) == len("20260102_030405")
    # the counter runs over the directory, not the seed: an agent reruns the
    # same seed constantly, and the number says which attempt this was
    files = sorted(p.name for p in batch.glob("*.npz"))
    assert files == [f"seed{FIRST_SEED}_0.npz", f"seed{SECOND_SEED}_1.npz"]
    # nothing half-written is left behind
    assert not list(batch.glob("*.tmp"))
    assert not list((cell / "dev" / ".staging").glob("*.npz"))


@pytest.mark.slow
def test_a_recorded_episode_has_the_arrays_the_viewer_reads(recorded_cell):
    """T+1 states around T actions, the seed, and an abandoned outcome."""
    cell, _ = recorded_cell
    batch = batch_of(cell)
    for seed, number in ((FIRST_SEED, 0), (SECOND_SEED, 1)):
        with np.load(batch / f"seed{seed}_{number}.npz") as data:
            assert data["full_qpos"].shape[0] == STEPS + 1
            assert data["full_qvel"].shape[0] == STEPS + 1
            assert data["full_qpos"].ndim == 2 and data["full_qpos"].shape[1] > 0
            assert data["action"].shape[0] == STEPS
            assert data["reward"].shape == (STEPS, 1)
            assert int(data["seed"]) == seed
            assert int(data["length"]) == STEPS
            assert float(data["success"]) == 0.0
            # the first was reset out from under, the second was still running
            # when the server stopped: both are abandoned
            assert str(data["termination"]) == "abandoned"


@pytest.mark.slow
def test_the_batch_describes_the_environment_it_was_recorded_in(recorded_cell):
    """metadata.json is the replay-batch format, written once per directory."""
    cell, _ = recorded_cell
    batch = batch_of(cell)
    meta = json.loads((batch / "metadata.json").read_text())
    assert meta["format"] == "bigym_replay_npz"
    assert meta["task"]["task_name"]
    assert meta["control_step_seconds"] > 0
    assert meta["substrate_fingerprint"]
    assert meta["source"]["kind"] == "agent_dev"
    assert meta["source"]["version"] == 1
    assert Path(meta["source"]["cell"]).name == TASK


@pytest.mark.slow
def test_the_recording_did_not_disturb_the_session(recorded_cell):
    """The ledger reports exactly the episodes that were run."""
    cell, output = recorded_cell
    events = [
        json.loads(line)
        for line in (cell / "ledger.jsonl").read_text().splitlines()
        if line.strip()
    ]
    resets = [e for e in events if e.get("event") == "reset"]
    assert [e["seed"] for e in resets] == [FIRST_SEED, SECOND_SEED]
    assert not [e for e in events if e.get("event") == "episode_end"]
    assert "[dev]" not in output, output
    index = json.loads((cell / "policies" / "index.json").read_text())
    assert [v["version"] for v in index["versions"]] == [1]
    assert index["versions"][0]["trigger"] == "submission"


@pytest.mark.slow
def test_the_viewer_discovers_a_development_batch(recorded_cell):
    """`bigym-view` treats dev/ like eval/ and replays/."""
    cell, _ = recorded_cell
    from bigym.vr.viewer import agent_cells

    found = agent_cells.batch_version(batch_of(cell))
    assert found is not None
    where, kind, version = found
    assert where == cell
    assert kind == "dev"
    assert version == 1
