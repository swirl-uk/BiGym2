"""The session launcher: dry-run commands, the registry, verdicts and run.json.

Nothing here starts docker, a server or an agent: the dry run builds the very
command lines the session would execute, and the registry tests run against a
sqlite file in a temporary root.
"""

from __future__ import annotations

import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from bigym.loco.agent import cli, containers, gpu, harnesses, launch, registry
from bigym.loco.agent.cli import EnvToolsConfig

CREDENTIAL_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CODEX_API_KEY",
)


@pytest.fixture(autouse=True)
def _no_docker(monkeypatch):
    """Keep every test off docker: no `sg` wrapper, no container listing."""
    monkeypatch.setattr(containers, "docker_argv", lambda argv: list(argv))
    monkeypatch.setattr(containers, "running_containers", lambda *a, **k: set())
    # the dry run lists the docker setup it would do; answer "nothing exists"
    monkeypatch.setattr(containers, "docker_has", lambda *a, **k: False)
    monkeypatch.setattr(containers, "proxy_networks", lambda *a, **k: set())
    # never probe EGL in tests; the detected map is exercised explicitly below
    monkeypatch.setattr(gpu, "detect_egl_map", lambda: {})


@pytest.fixture(autouse=True)
def _no_credentials(monkeypatch, tmp_path):
    """Start every test with no API key, no token and an empty agent home."""
    for var in CREDENTIAL_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("BIGYM_AGENT_CODEX_HOME", raising=False)
    monkeypatch.setenv("BIGYM_AGENT_HOME", str(tmp_path / "agent_home"))


def _run(tmp_path, *extra):
    """Run `bigym-agent run --dry-run` with the given extra arguments."""
    argv = [
        "--task",
        "move_plate",
        "--root",
        str(tmp_path / "root"),
        "--model",
        "a-model",
        "--dry-run",
    ] + list(extra)
    assert launch.main(argv) == 0


def test_dry_run_codex(tmp_path, capsys):
    """The codex session prints its sandbox, serve and container commands."""
    _run(tmp_path, "--harness", "codex", "--effort", "high")
    out = capsys.readouterr().out
    cell = tmp_path / "root" / "move_plate"
    assert f"sandbox --task move_plate --cell {cell}" in out
    assert "--budget-steps 101000 --workers 3 --harness codex --effort high" in out
    assert (
        f"serve --task move_plate --sandbox {cell}/sandbox --ledger-dir {cell}" in out
    )
    assert f"--allowed-seeds {cell}/allowed_seeds.json" in out
    # the default environment needs no flags
    assert "--interface" not in out and "--image-cap" not in out
    assert "--client-root /work" in out
    # the container: the sandbox is the only mount besides the harness home
    assert "docker run --rm -i --name bigym-agent-" in out
    assert f"-v {cell}/sandbox:/work" in out
    assert f"-v {cell}/raw/codex_home:/codex_home" in out
    assert "auth.json:/codex_home/auth.json" in out
    assert "--network bigym-agent-internal" in out
    assert "-e HTTPS_PROXY=http://bigym-agent-proxy:8888" in out
    assert "-e CODEX_HOME=/codex_home" in out
    assert "codex exec --skip-git-repo-check -C /work" in out
    assert "-m a-model -c model_reasoning_effort=high --json" in out
    assert "stdin: PROMPT.md" in out
    assert "MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0" in out
    # the evaluation of the submission, in the fallback (no policies/) form
    assert f"evaluate --task move_plate --policy {cell}/sandbox/policy.py" in out
    # and nothing was written
    assert not cell.exists()


def test_dry_run_claude(tmp_path, capsys):
    """The claude session mounts its config dir and takes the token from a file."""
    _run(tmp_path, "--harness", "claude", "--max-turns", "42")
    out = capsys.readouterr().out
    cell = tmp_path / "root" / "move_plate"
    sessions = tmp_path / "agent_home" / "sessions"
    assert f"--env-file {sessions}/bigym-agent-" in out
    assert "/claude_env" in out
    assert f"-v {cell}/raw/claude_home:/claude_home" in out
    assert "-e CLAUDE_CONFIG_DIR=/claude_home" in out
    assert (
        "claude -p --setting-sources project --settings /work/claude_settings.json"
        in out
    )
    assert "--dangerously-skip-permissions --model a-model --max-turns 42" in out
    assert "--output-format stream-json --verbose" in out
    assert "stdin: PROMPT.md" in out
    assert not (cell / "raw").exists()


def test_dry_run_soft_isolation(tmp_path, capsys):
    """Soft isolation runs the harness on the host with no container at all."""
    _run(tmp_path, "--harness", "claude", "--isolation", "soft")
    out = capsys.readouterr().out
    assert "docker run" not in out
    assert "--permission-mode dontAsk" in out
    assert "--client-root /work" not in out
    assert "--isolation soft" in out


@pytest.mark.parametrize(
    "harness, flags",
    [
        ("codex", "-c project_doc_max_bytes=0 -c skills.max_context_tokens=1"),
        ("claude", "--disable-slash-commands"),
    ],
)
def test_dry_run_soft_keeps_project_files_out(tmp_path, capsys, harness, flags):
    """On the host, the harness skips the project's instruction files and skills."""
    _run(tmp_path, "--harness", harness, "--isolation", "soft")
    out = capsys.readouterr().out
    assert "docker run" not in out
    assert flags in out


def test_dry_run_sessions_write_one_root_each(tmp_path, capsys):
    """--sessions N gives every task N cells, under <root>_s1 ... <root>_sN."""
    _run(tmp_path, "--harness", "codex", "--task", "move_plate", "--sessions", "2")
    out = capsys.readouterr().out
    for k in (1, 2):
        assert f"--cell {tmp_path / f'root_s{k}' / 'move_plate'}" in out
    assert not (tmp_path / "root").exists()


def test_parallel_caps_every_session_of_every_task(tmp_path, monkeypatch):
    """--parallel bounds the sessions running at once over tasks and sessions."""
    import threading
    import time

    running, peak, cells = [0], [0], []
    lock = threading.Lock()

    def fake_session(task, args, cancel=None):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
            cells.append((args.root.name, task))
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        return {"task": task, "exit": 0}

    monkeypatch.setattr(launch, "run_session", fake_session)
    root = tmp_path / "root"
    argv = ["--task", "move_plate", "pick_box", "--root", str(root), "--model", "m"]
    assert launch.main(argv + ["--sessions", "3", "--parallel", "2"]) == 0
    assert peak[0] == 2
    assert sorted(cells) == sorted(
        (f"root_s{k}", task) for k in (1, 2, 3) for task in ("move_plate", "pick_box")
    )
    for k in (1, 2, 3):
        entries = json.loads((tmp_path / f"root_s{k}" / "run_results.json").read_text())
        assert sorted(e["task"] for e in entries) == ["move_plate", "pick_box"]


def test_dry_run_custom(tmp_path, capsys):
    """A custom agent is a shell command with the session in its environment."""
    root = tmp_path / "root"
    assert (
        launch.main(
            [
                "--task",
                "move_plate",
                "--root",
                str(root),
                "--harness",
                "custom",
                "--command",
                "./python agent.py --go",
                "--dry-run",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    cell = root / "move_plate"
    # the cell is built exactly as for a shipped harness, at soft isolation
    assert f"sandbox --task move_plate --cell {cell}" in out
    assert "--isolation soft" in out
    assert f"serve --task move_plate --sandbox {cell}/sandbox" in out
    assert "--client-root /work" not in out
    # the command itself, unquoted, with its three environment variables
    assert "harness: ./python agent.py --go" in out
    assert f"shell in {cell}/sandbox" in out
    assert f"AGENT_SANDBOX={cell}/sandbox" in out
    assert f"BIGYM_AGENT_PROMPT={cell}/sandbox/PROMPT.md" in out
    assert f"BIGYM_AGENT_CELL={cell}" in out
    assert f"stdout: {cell}/raw/custom_stdout.txt" in out
    assert f"stderr: {cell}/raw/custom_stderr.txt" in out
    assert "stdin: none" in out
    # no container, and no version probe of a command we know nothing about
    assert "docker run" not in out
    assert "version:" not in out
    assert not cell.exists()


def test_runs_default_to_a_directory_in_the_current_one(tmp_path, monkeypatch):
    """No --root: runs go to ./bigym-agent-runs, resolved against the cwd."""
    monkeypatch.chdir(tmp_path)
    args = launch.parse_run_args(
        ["--task", "t", "--harness", "custom", "--command", "true"]
    )
    assert args.root == (tmp_path / "bigym-agent-runs").resolve()
    # the same default for the commands that look at a root
    for command in (registry.main_status, registry.main_gc):
        try:
            command([])
        except SystemExit:
            raise AssertionError(f"{command.__name__} still requires --root") from None
    assert (tmp_path / "bigym-agent-runs" / "registry.sqlite").exists()


def test_custom_needs_a_command_but_not_a_model(tmp_path):
    """--command is required for custom; --model is not, and is recorded null."""
    with pytest.raises(SystemExit):
        launch.parse_run_args(
            ["--task", "t", "--root", str(tmp_path), "--harness", "custom"]
        )
    # every other agent still needs a model
    with pytest.raises(SystemExit):
        launch.parse_run_args(["--task", "t", "--root", str(tmp_path)])
    args = launch.parse_run_args(
        [
            "--task",
            "t",
            "--root",
            str(tmp_path),
            "--harness",
            "custom",
            "--command",
            "./agent.sh",
        ]
    )
    assert args.model is None
    assert args.isolation == "soft"
    cell = (tmp_path / "t").resolve()
    spec = harnesses.agent_command("t", cell, args, dry_run=True)
    assert spec.cmd == "./agent.sh"
    assert spec.shell is True
    assert spec.image is None and spec.container is None
    assert spec.env["AGENT_SANDBOX"] == str(cell / "sandbox")
    assert spec.env["BIGYM_AGENT_PROMPT"] == str(cell / "sandbox" / "PROMPT.md")
    assert spec.env["BIGYM_AGENT_CELL"] == str(cell)
    config = launch.run_config("t", cell, args, spec, 0, 0)
    assert config["harness"] == "custom"
    assert config["model"] is None
    assert config["command"] == "./agent.sh"
    assert config["isolation"] == "soft"


def test_custom_refuses_a_container(tmp_path):
    """The benchmark ships no image for someone else's command, and says so."""
    argv = [
        "--task",
        "t",
        "--root",
        str(tmp_path),
        "--harness",
        "custom",
        "--command",
        "./agent.sh",
        "--isolation",
        "container",
    ]
    with pytest.raises(SystemExit):
        launch.parse_run_args(argv)
    # and the refusal stands even when the parser is bypassed
    args = launch.parse_run_args(argv[:-2])
    args.isolation = "container"
    with pytest.raises(RuntimeError, match="use --isolation soft"):
        harnesses.agent_command("t", tmp_path / "t", args, dry_run=True)


def test_env_tools_reach_sandbox_server_and_evaluation(tmp_path, capsys):
    """Every environment setting is passed to all three, the defaults to none."""
    flags = [
        "--interface",
        "tools",
        "--image-cap",
        "640x480",
        "--slew",
        "--accel",
        "None",
        "--lowpass",
        "0.5",
        "--joint-vmax",
        "4.0",
        "--no-ik",
        "--no-hand-pos",
        "--no-pitch",
        "--allow-external-cameras",
        "--tier",
        "privileged",
    ]
    _run(tmp_path, *flags)
    out = capsys.readouterr().out
    lines = [
        line
        for line in out.splitlines()
        if line.lstrip().startswith(("sandbox:", "serve  :", "evaluate:"))
    ]
    assert len(lines) == 3
    for line in lines:
        for flag in flags:
            assert f" {flag}" in line, (flag, line)
    args = launch.parse_run_args(
        ["--task", "t", "--root", str(tmp_path), "--model", "m", *flags]
    )
    assert launch.env_tools_flags(EnvToolsConfig()) == []
    rebuilt = cli.parse_command(
        cli.ServeConfig,
        ["--task", "t", "--sandbox", "s", "--ledger-dir", "l"]
        + launch.env_tools_flags(args.env_tools),
    )
    assert rebuilt.env_tools == args.env_tools


def test_eval_command_prefers_the_last_policy_version(tmp_path):
    """With policies/index.json the cell mode scores the last version."""
    args = launch.parse_run_args(
        ["--task", "t", "--root", str(tmp_path), "--model", "m"]
    )
    cell = tmp_path / "t"
    (cell / "policies").mkdir(parents=True)
    (cell / "policies" / "index.json").write_text(
        json.dumps({"task": "t", "versions": [{"version": 1}, {"version": 7}]})
    )
    command, version = launch.eval_command("t", cell, args)
    assert version == 7
    assert command[-7:] == [
        str(cell),
        "--version",
        "7",
        "--episodes",
        "100",
        "--spot-check",
        "10",
    ]
    # without an index, the sandbox's policy.py is scored directly
    (cell / "policies" / "index.json").unlink()
    command, version = launch.eval_command("t", cell, args)
    assert version is None
    assert "--policy" in command
    assert str(cell / "sandbox" / "policy.py") in command
    assert command[command.index("--episodes") + 1] == "100"
    # --eval-episodes reaches the evaluation in both forms
    args = launch.parse_run_args(
        [
            "--task",
            "t",
            "--root",
            str(tmp_path),
            "--model",
            "m",
            "--eval-episodes",
            "3",
        ]
    )
    fallback = launch.eval_command("t", cell, args)[0]
    assert fallback[fallback.index("--episodes") + 1] == "3"
    (cell / "policies" / "index.json").write_text(
        json.dumps({"task": "t", "versions": [{"version": 2}]})
    )
    assert launch.eval_command("t", cell, args)[0][-4:-2] == ["--episodes", "3"]


def test_egl_map_parsing(monkeypatch):
    """The cuda:egl map is parsed from the flag or the environment."""
    assert gpu.egl_map("0:4,1:5, 2:2 ") == {0: 4, 1: 5, 2: 2}
    assert gpu.egl_map("") == {}
    with pytest.raises(ValueError):
        gpu.egl_map("0-4")
    monkeypatch.setenv("BIGYM_AGENT_EGL_MAP", "3:1")
    assert gpu.egl_map() == {3: 1}
    monkeypatch.delenv("BIGYM_AGENT_EGL_MAP")
    assert gpu.egl_map() == {}


def test_gpu_is_the_nvidia_smi_index(tmp_path, monkeypatch, capsys):
    """--gpu names the GPU as nvidia-smi does; the EGL device follows the map."""
    monkeypatch.setenv("BIGYM_AGENT_EGL_MAP", "0:4,1:5")
    _run(tmp_path, "--harness", "codex", "--gpu", "1")
    out = capsys.readouterr().out
    assert "egl 5 (gpu 1)" in out
    assert "MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=5" in out


def test_several_tasks_go_after_one_task_flag(tmp_path):
    """--task takes a list; there is no comma-separated --tasks."""
    argv = ["--root", str(tmp_path), "--model", "a-model"]
    args = launch.parse_run_args(["--task", "move_plate", "pick_box", *argv])
    assert args.task == ["move_plate", "pick_box"]
    with pytest.raises(SystemExit):
        launch.parse_run_args(["--tasks", "move_plate,pick_box", *argv])


@pytest.mark.parametrize(
    "harness, model",
    [
        ("codex", "claude-opus-4-5"),
        ("codex", "Sonnet"),
        ("claude", "gpt-5.2"),
        ("claude", "o3"),
        ("claude", "gpt-5-codex"),
    ],
)
def test_a_model_of_the_other_vendor_is_refused(tmp_path, harness, model):
    """--harness codex runs OpenAI models and --harness claude Claude models."""
    argv = ["--task", "move_plate", "--root", str(tmp_path), "--harness", harness]
    with pytest.raises(SystemExit):
        launch.parse_run_args([*argv, "--model", model])
    assert not (tmp_path / "move_plate").exists()


@pytest.mark.parametrize(
    "harness, model",
    [("codex", "gpt-5.2"), ("codex", "o3"), ("claude", "claude-opus-4-5")],
)
def test_a_model_of_the_harness_vendor_is_accepted(tmp_path, harness, model):
    argv = ["--task", "move_plate", "--root", str(tmp_path), "--harness", harness]
    assert launch.parse_run_args([*argv, "--model", model]).model == model
    assert harnesses.model_mismatch("custom", "claude-opus-4-5") is None


def test_pick_egl(monkeypatch):
    """The GPU with the most free memory wins, and a full host is refused."""
    monkeypatch.setattr(gpu, "gpu_free_mib", lambda: {0: 1000, 1: 40000})
    monkeypatch.setenv("BIGYM_AGENT_EGL_MAP", "0:4,1:5")
    assert gpu.pick_egl(6000) == (5, 1)
    # no map: the EGL index is the nvidia-smi index
    monkeypatch.delenv("BIGYM_AGENT_EGL_MAP")
    assert gpu.pick_egl(6000) == (1, 1)
    monkeypatch.setattr(gpu, "gpu_free_mib", lambda: {0: 10, 1: 20})
    with pytest.raises(SystemExit, match="MiB free"):
        gpu.pick_egl(6000)
    monkeypatch.setattr(gpu, "gpu_free_mib", dict)
    with pytest.raises(SystemExit, match="no GPU visible"):
        gpu.pick_egl(6000)


def test_registry_refuses_a_second_live_run(tmp_path, capsys):
    """One live session owns a cell; a stale row is closed and reused."""
    root = tmp_path / "root"
    cell = root / "move_plate"
    conn = registry.open_registry(root)
    info = {"container": "bigym-agent-a", "egl": 0, "gpu": 0, "harness": "codex"}
    first = registry.claim(conn, cell, "move_plate", info)
    assert first is not None
    assert registry.claim(conn, cell, "move_plate", dict(info, container="b")) is None
    assert "already owns" in capsys.readouterr().out
    # a different task of the same root is free
    assert registry.claim(conn, root / "pick_box", "pick_box", {"container": "c"})
    # once the row is closed the cell can be claimed again
    registry.close_row(conn, first, "[done]")
    assert registry.claim(conn, cell, "move_plate", {"container": "d"}) is not None


def test_registry_refuses_a_duplicate_container_name(tmp_path, monkeypatch, capsys):
    """A container name already in use is refused before anything starts."""
    monkeypatch.setattr(containers, "running_containers", lambda: {"bigym-agent-taken"})
    conn = registry.open_registry(tmp_path)
    assert (
        registry.claim(conn, tmp_path / "t", "t", {"container": "bigym-agent-taken"})
        is None
    )
    assert "refused: container" in capsys.readouterr().out


def test_kill_stops_one_session_and_leaves_its_run_and_siblings(tmp_path):
    """Killing one cell stops its agent only, and holds the cell until its run ends."""

    def spawn():
        return subprocess.Popen(["sleep", "60"], start_new_session=True)

    run, agent_a, agent_b = spawn(), spawn(), spawn()
    try:
        root = tmp_path / "root"
        conn = registry.open_registry(root)
        rows = {}
        for task, agent in (("move_plate", agent_a), ("pick_box", agent_b)):
            rows[task] = registry.claim(conn, root / task, task, {"container": None})
            conn.execute("UPDATE runs SET pid=? WHERE id=?", (run.pid, rows[task]))
            registry.record_process_group(conn, rows[task], agent.pid)
        conn.commit()
        assert registry.main_kill(["--root", str(root), "move_plate"]) == 0
        assert agent_a.wait(timeout=10) is not None
        assert agent_b.poll() is None
        assert run.poll() is None
        assert registry.stop_requested(conn, rows["move_plate"])
        assert not registry.stop_requested(conn, rows["pick_box"])
        # The run still owns the killed cell until it has wound it down.
        cell = root / "move_plate"
        assert registry.claim(conn, cell, "move_plate", {"container": None}) is None
        registry.close_row(conn, rows["move_plate"])
        assert registry.claim(conn, cell, "move_plate", {"container": None})
        conn.close()
    finally:
        for proc in (run, agent_a, agent_b):
            proc.kill()
            proc.wait()


def test_kill_before_the_agent_starts_stops_it_when_it_does(tmp_path):
    """A cell killed while its sandbox is built never gets a running agent."""
    run = subprocess.Popen(["sleep", "60"], start_new_session=True)
    agent = None
    try:
        root = tmp_path / "root"
        conn = registry.open_registry(root)
        row = registry.claim(
            conn, root / "move_plate", "move_plate", {"container": None}
        )
        conn.execute(
            "UPDATE runs SET pid=?, pgid=? WHERE id=?", (run.pid, run.pid, row)
        )
        conn.commit()
        assert registry.main_kill(["--root", str(root), "move_plate"]) == 0
        assert run.poll() is None
        agent = subprocess.Popen(["sleep", "60"], start_new_session=True)
        launch.cell_process_started(conn, row, agent.pid)
        assert agent.wait(timeout=10) is not None
        conn.close()
    finally:
        for proc in (run, agent):
            if proc is not None:
                proc.kill()
                proc.wait()


def test_kill_before_the_evaluation_starts_stops_it_when_it_does(tmp_path, monkeypatch):
    """A cell killed after its agent ended is not evaluated after all."""
    monkeypatch.setattr(launch, "eval_command", lambda *a: (["sleep", "60"], 1))
    run = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        root = tmp_path / "root"
        cell = root / "move_plate"
        conn = registry.open_registry(root)
        row = registry.claim(conn, cell, "move_plate", {"container": None})
        conn.execute(
            "UPDATE runs SET pid=?, pgid=? WHERE id=?", (run.pid, run.pid, row)
        )
        conn.commit()
        assert registry.main_kill(["--root", str(root), "move_plate"]) == 0
        args = SimpleNamespace(eval_detached=False)
        evaluation = launch.evaluate_submission(
            "move_plate",
            cell,
            args,  # ty: ignore[invalid-argument-type]
            {},
            {},
            on_start=lambda pgid: launch.cell_process_started(conn, row, pgid),
        )
        assert evaluation["status"] == "failed"
        assert evaluation["exit"] != 0
        conn.close()
    finally:
        run.kill()
        run.wait()


RUN_WITH_SLEEPING_AGENTS = """
import os, subprocess, sys, time
from bigym.loco.agent import launch, registry

first = []

def session(task, args, cancel=None):
    if task == "pick_box":
        # Claims only once Ctrl+C has stopped the first cell.
        while not first or registry.alive(first[0]):
            time.sleep(0.05)
    conn = registry.open_registry(args.root)
    row = launch.claim_unless_cancelled(
        conn, args.root / task, task, {"container": None}, cancel
    )
    if row is None:
        return {"task": task, "error": "interrupted"}
    agent = subprocess.Popen(["sleep", "60"], start_new_session=True)
    launch.cell_process_started(conn, row, agent.pid)
    first.append(agent.pid)
    print(task, agent.pid, flush=True)
    agent.wait()
    registry.close_row(conn, row)
    return {"task": task, "exit": agent.returncode}

launch.run_session = session
sys.exit(launch.main([
    "--task", "move_plate", "pick_box", "--parallel", "2",
    "--root", sys.argv[1], "--model", "m",
]))
"""


def test_ctrl_c_on_run_stops_its_cells_and_starts_no_more(tmp_path):
    """Ctrl+C stops the running agents, and a cell claimed afterwards never starts."""
    import signal
    import sys

    run = subprocess.Popen(
        [sys.executable, "-c", RUN_WITH_SLEEPING_AGENTS, str(tmp_path / "root")],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert run.stdout is not None
    task, agent = run.stdout.readline().split()
    assert task == "move_plate"
    try:
        run.send_signal(signal.SIGINT)
        assert run.wait(timeout=60) == 130
        assert not registry.alive(int(agent))
        rest = run.stdout.read().splitlines()
        assert not any(line.startswith("pick_box ") for line in rest)
        assert any("pick_box" in line and "interrupted" in line for line in rest)
    finally:
        run.kill()
        if registry.alive(int(agent)):
            os.kill(int(agent), 9)


def test_status_and_gc(tmp_path, capsys):
    """status lists live rows; gc closes the dead ones."""
    root = tmp_path / "root"
    cell = root / "move_plate"
    cell.mkdir(parents=True)
    (cell / "budget.json").write_text(json.dumps({"used": 42, "cap": 101000}))
    (cell / "run.json").write_text(json.dumps({"verdict": {"state": "ok"}}))
    (cell / "eval" / "v001").mkdir(parents=True)
    (cell / "eval" / "v001" / "summary.json").write_text(
        json.dumps({"version": 1, "success_rate": 0.93})
    )
    conn = registry.open_registry(root)
    row = registry.claim(conn, cell, "move_plate", {"container": None, "egl": 0})
    conn.close()
    assert registry.main_status(["--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "move_plate" in out and "LIVE" in out
    assert "budget 42/101,000" in out and "SCORE 93" in out and "ok" in out
    # a row whose process is gone is closed by gc and then hidden by status
    conn = registry.open_registry(root)
    conn.execute("UPDATE runs SET pid=? WHERE id=?", (2**30, row))
    conn.commit()
    conn.close()
    assert registry.main_gc(["--root", str(root)]) == 0
    assert "closed 1 dead registry rows" in capsys.readouterr().out
    assert registry.main_status(["--root", str(root)]) == 0
    assert "no runs running" in capsys.readouterr().out
    assert registry.main_status(["--root", str(root), "--all"]) == 0
    assert "move_plate" in capsys.readouterr().out


def test_verdict_void_when_the_template_came_back_untouched(tmp_path):
    """An agent that never wrote a policy submits nothing, however it exited."""
    cell = tmp_path / "move_plate"
    (cell / "sandbox").mkdir(parents=True)
    template = "import numpy as np\n\n# TODO: your control logic here\n"
    (cell / "sandbox" / "policy.py").write_text(template)
    (cell / "initial_policy.py").write_text(template)
    assert launch.submission_state(cell, 0, None) == (
        "void",
        "the submission is the untouched template, byte for byte",
    )
    # the marker fallback, for a cell whose builder archived no template
    (cell / "initial_policy.py").unlink()
    assert launch.submission_state(cell, 0, None)[0] == "void"


def test_verdict_ok_and_interrupted(tmp_path):
    """A written policy scores; a non-zero exit scores but is marked."""
    cell = tmp_path / "move_plate"
    (cell / "sandbox").mkdir(parents=True)
    (cell / "initial_policy.py").write_text("template\n")
    (cell / "sandbox" / "policy.py").write_text("a real policy\n")
    assert launch.submission_state(cell, 0, None) == ("ok", "")
    assert launch.submission_state(cell, 1, None) == ("interrupted", "exit=1")
    assert launch.submission_state(cell, -1, None) == ("interrupted", "exit=-1")


def test_verdict_void_without_a_policy_or_a_container(tmp_path):
    """No policy.py, or a container that never started, is void."""
    cell = tmp_path / "move_plate"
    (cell / "sandbox").mkdir(parents=True)
    assert launch.submission_state(cell, 0, None)[0] == "void"
    (cell / "sandbox" / "policy.py").write_text("a real policy\n")
    assert launch.submission_state(cell, 125, "bigym-agent-x") == (
        "void",
        "docker could not start the container (exit 125)",
    )


def test_run_config_records_the_session(tmp_path):
    """run.json carries the session's configuration as plain JSON."""
    args = launch.parse_run_args(
        [
            "--task",
            "move_plate",
            "--root",
            str(tmp_path),
            "--model",
            "a-model",
            "--harness",
            "codex",
        ]
    )
    cell = (tmp_path / "move_plate").resolve()
    spec = harnesses.agent_command("move_plate", cell, args, dry_run=True)
    config = launch.run_config("move_plate", cell, args, spec, 4, 1)
    loaded = json.loads(json.dumps(config))
    assert loaded["schema"] == "bigym-agent/run.json v1"
    assert loaded["task"] == "move_plate"
    assert loaded["cell"] == str(cell)
    assert loaded["harness"] == "codex"
    assert loaded["model"] == "a-model"
    assert loaded["isolation"] == "container"
    assert loaded["image"] == containers.DEFAULT_IMAGES["codex"]
    assert loaded["container_name"].startswith(containers.CONTAINER_PREFIX)
    assert loaded["docker_network"] == containers.DEFAULT_NETWORK
    assert loaded["egl_device"] == 4 and loaded["gpu"] == 1
    assert loaded["auth"] == "subscription"
    for key in (
        "end",
        "exit",
        "verdict",
        "budget",
        "usage",
        "cost_usd",
        "evaluation",
        "resume_session",
    ):
        assert loaded[key] is None


def test_container_names_are_unique_and_docker_safe(tmp_path):
    """The container name carries the root, the task and the millisecond."""
    args = launch.parse_run_args(
        ["--task", "t", "--root", str(tmp_path / "a root"), "--model", "m"]
    )
    first = harnesses.new_container_name("move_plate", tmp_path / "a root" / "t", args)
    assert first.startswith("bigym-agent-")
    assert all(c.isalnum() or c in "_.-" for c in first)
    assert " " not in first


def test_agent_home_follows_the_environment(monkeypatch, tmp_path):
    """Credentials live under $BIGYM_AGENT_HOME, never a hardcoded path."""
    monkeypatch.setenv("BIGYM_AGENT_HOME", str(tmp_path / "creds"))
    assert harnesses.agent_home() == tmp_path / "creds"
    monkeypatch.delenv("BIGYM_AGENT_HOME")
    assert harnesses.agent_home().name == ".bigym-agent"


def test_alive_on_this_process():
    """The liveness probe knows this process is alive and pid 0 is not."""
    assert registry.alive(os.getpid())
    assert not registry.alive(None)
    assert not registry.alive(0)


def _fake_docker(present: set[str], running: set[str]):
    """A ``subprocess.run`` stand-in answering docker inspect / ps queries."""

    def run(argv, capture_output=True, text=True):
        cmd = argv[-1].split() if argv[:2] == ["sg", "docker"] else list(argv)
        code, out = 0, ""
        if cmd[:2] == ["docker", "ps"]:
            out = "\n".join(sorted(running))
        elif cmd[1] in ("network", "image", "container") and cmd[2] == "inspect":
            code = 0 if f"{cmd[1]}:{cmd[3]}" in present else 1
        elif cmd[:2] == ["docker", "inspect"]:
            code = 0 if f"container:{cmd[2]}" in present else 1
            out = "bigym-agent-egress bigym-agent-internal " if code == 0 else ""
        return subprocess.CompletedProcess(argv, code, out, "")

    return run


def test_docker_setup_commands_from_scratch_and_prepared(monkeypatch):
    """A bare host gets networks, proxy and image; a prepared host gets nothing."""
    monkeypatch.undo()  # use the real helpers against the fake docker below
    monkeypatch.setattr(containers, "docker_argv", lambda argv: list(argv))
    cmds = containers.docker_setup_commands(
        "codex",
        containers.DEFAULT_IMAGES["codex"],
        containers.DEFAULT_NETWORK,
        containers.DEFAULT_PROXY,
        run=_fake_docker(set(), set()),
    )
    heads = [c[:3] for c in cmds]
    assert ["docker", "network", "create"] in heads
    assert any(c[:2] == ["docker", "run"] and containers.PROXY_IMAGE in c for c in cmds)
    assert ["docker", "network", "connect"] in heads
    assert cmds[-1][:2] == ["docker", "build"] and "Dockerfile.codex" in cmds[-1][3]
    prepared = {
        f"network:{containers.DEFAULT_NETWORK}",
        f"network:{containers.EGRESS_NETWORK}",
        f"container:{containers.PROXY_CONTAINER}",
        f"image:{containers.DEFAULT_IMAGES['codex']}",
    }
    assert (
        containers.docker_setup_commands(
            "codex",
            containers.DEFAULT_IMAGES["codex"],
            containers.DEFAULT_NETWORK,
            containers.DEFAULT_PROXY,
            run=_fake_docker(prepared, {containers.PROXY_CONTAINER}),
        )
        == []
    )
    # a host running its own proxy and network is never touched
    assert (
        containers.docker_setup_commands(
            "codex", "my-image", "my-net", "http://p:1", run=_fake_docker(set(), set())
        )
        == []
    )


KEY = "sk-test-0123456789abcdefSECRET"


def _session_args(tmp_path, harness, isolation="container"):
    return launch.parse_run_args(
        [
            "--task",
            "move_plate",
            "--root",
            str(tmp_path / "root"),
            "--model",
            "a-model",
            "--harness",
            harness,
            "--isolation",
            isolation,
        ]
    )


def _build(tmp_path, monkeypatch, harness, isolation="container"):
    """The real (not dry-run) command, without probing the CLI version."""
    monkeypatch.setattr(harnesses, "harness_version", lambda *a: ("", []))
    args = _session_args(tmp_path, harness, isolation)
    cell = tmp_path / "root" / "move_plate"
    (cell / "sandbox").mkdir(parents=True, exist_ok=True)
    return cell, harnesses.agent_command("move_plate", cell, args)


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_an_api_key_wins_over_the_subscription(tmp_path, monkeypatch, harness):
    """Variable first, then the key file; no key means the subscription login."""
    env_var, name = harnesses.API_KEYS[harness]
    assert harnesses.api_key(harness) is None
    assert harnesses.auth_mode(harness) == "subscription"
    key_file = harnesses.agent_home() / name
    key_file.parent.mkdir(parents=True)
    key_file.write_text("from-file\n")
    assert harnesses.api_key(harness) == "from-file"
    assert harnesses.api_key_source(harness) == str(key_file)
    assert harnesses.auth_mode(harness) == "api_key"
    monkeypatch.setenv(env_var, "from-env")
    assert harnesses.api_key(harness) == "from-env"
    assert harnesses.api_key_source(harness) == f"${env_var}"
    assert harnesses.auth_mode(harness) == "api_key"
    assert harnesses.auth_mode("custom") is None


def test_auth_report_names_the_mode_and_the_alternative(tmp_path, monkeypatch):
    """The session log says which credential is used and what to set instead."""
    codex = _session_args(tmp_path, "codex")
    line, warning = harnesses.auth_report(codex)
    assert "ChatGPT login" in line and "OPENAI_API_KEY" in line
    assert warning is not None and "codex login" in warning
    claude = _session_args(tmp_path, "claude")
    line, warning = harnesses.auth_report(claude)
    assert warning is not None
    assert "ANTHROPIC_API_KEY" in warning and "claude setup-token" in warning
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", KEY)
    line, warning = harnesses.auth_report(claude)
    assert "OAuth token from $CLAUDE_CODE_OAUTH_TOKEN" in line and warning is None
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    assert harnesses.auth_report(codex) == (
        "codex auth: API key from $OPENAI_API_KEY",
        None,
    )
    assert harnesses.auth_report(claude) == (
        "claude auth: API key from $ANTHROPIC_API_KEY",
        None,
    )
    for args in (codex, claude):
        assert KEY not in " ".join(str(x) for x in harnesses.auth_report(args))


def _assert_private(path, cell):
    """A credential file: owner-only, in the session directory, not in the cell."""
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert path.parent.parent == harnesses.agent_home() / "sessions"
    assert not str(path).startswith(str(cell))


def test_codex_api_key_reaches_the_container_in_a_private_auth_json(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    cell, spec = _build(tmp_path, monkeypatch, "codex")
    assert spec.auth == "api_key"
    argv = " ".join(spec.cmd)
    assert KEY not in argv
    assert spec.secrets is not None
    auth = spec.secrets / "auth.json"
    assert f"-v {auth}:/codex_home/auth.json:ro" in argv
    assert json.loads(auth.read_text()) == {
        "auth_mode": "apikey",
        "OPENAI_API_KEY": KEY,
    }
    _assert_private(auth, cell)
    assert not any(KEY in p.read_text() for p in cell.rglob("*") if p.is_file())


def test_codex_subscription_mounts_the_shared_login(tmp_path, monkeypatch):
    cell, spec = _build(tmp_path, monkeypatch, "codex")
    assert spec.auth == "subscription" and spec.secrets is None
    shared = harnesses.agent_home() / "codex_home" / "auth.json"
    assert f"-v {shared}:/codex_home/auth.json -w" in " ".join(spec.cmd)


@pytest.mark.parametrize(
    "env_var, mode",
    [("ANTHROPIC_API_KEY", "api_key"), ("CLAUDE_CODE_OAUTH_TOKEN", "subscription")],
)
def test_claude_credential_goes_through_a_private_env_file(
    tmp_path, monkeypatch, env_var, mode
):
    monkeypatch.setenv(env_var, KEY)
    cell, spec = _build(tmp_path, monkeypatch, "claude")
    assert spec.auth == mode
    argv = " ".join(spec.cmd)
    assert KEY not in argv
    assert spec.secrets is not None
    env_file = spec.secrets / "claude_env"
    assert f"--env-file {env_file}" in argv
    assert env_file.read_text() == f"{env_var}={KEY}\n"
    _assert_private(env_file, cell)
    assert not any(KEY in p.read_text() for p in cell.rglob("*") if p.is_file())


def test_claude_without_credentials_says_what_to_set(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        _build(tmp_path, monkeypatch, "claude")


@pytest.mark.parametrize(
    "harness, env_var, child_var",
    [
        ("codex", "OPENAI_API_KEY", "CODEX_API_KEY"),
        ("claude", "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    ],
)
def test_host_harness_gets_the_key_from_a_file_in_its_environment(
    tmp_path, monkeypatch, harness, env_var, child_var
):
    key_file = harnesses.agent_home() / harnesses.API_KEYS[harness][1]
    key_file.parent.mkdir(parents=True)
    key_file.write_text(KEY)
    _, spec = _build(tmp_path, monkeypatch, harness, isolation="soft")
    assert spec.auth == "api_key"
    assert spec.env[child_var] == KEY
    assert KEY not in " ".join(spec.cmd)


def test_the_session_credentials_are_removed_after_the_run(tmp_path):
    secrets = harnesses.session_secrets("bigym-agent-test", True)
    harnesses.write_secret_file(secrets / "claude_env", f"ANTHROPIC_API_KEY={KEY}\n")
    cell = tmp_path / "cell"
    (cell / "sandbox").mkdir(parents=True)
    spec = harnesses.AgentCommand(
        ["true"],
        dict(os.environ),
        None,
        cell / "out.txt",
        cell / "err.txt",
        secrets=secrets,
    )
    args = SimpleNamespace(resume=False, session_timeout_s=30)
    assert launch.run_agent(spec, cell, args) == 0  # ty: ignore[invalid-argument-type]
    assert not secrets.exists()


def test_codex_preflight_reports_missing_and_expired(tmp_path):
    """No auth.json, an expired token, and a live token give the right hint."""
    import base64
    import time

    home = tmp_path / "codex_home"
    warning = harnesses.codex_preflight(home)
    assert warning is not None and "codex login" in warning
    home.mkdir()

    def auth(exp):
        payload = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode())
        token = "h." + payload.decode().rstrip("=") + ".s"
        (home / "auth.json").write_text(json.dumps({"tokens": {"access_token": token}}))

    auth(time.time() - 3600)
    warning = harnesses.codex_preflight(home)
    assert warning is not None and "expired" in warning
    auth(time.time() + 3600)
    assert harnesses.codex_preflight(home) is None


def test_egl_map_detected_when_env_unset(monkeypatch):
    """Without $BIGYM_AGENT_EGL_MAP the driver's own map is used."""
    monkeypatch.delenv("BIGYM_AGENT_EGL_MAP", raising=False)
    monkeypatch.setattr(gpu, "detect_egl_map", lambda: {0: 4, 1: 5})
    assert gpu.egl_map() == {0: 4, 1: 5}
    # an explicit (even empty) environment map wins over detection
    monkeypatch.setenv("BIGYM_AGENT_EGL_MAP", "")
    assert gpu.egl_map() == {}


def test_run_results_merge_by_task(tmp_path):
    """Two invocations on one root keep one entry per task, newest first wins."""
    launch.write_run_results(tmp_path, [{"task": "a", "exit": 1, "verdict": "void"}])
    launch.write_run_results(
        tmp_path, [{"task": "a", "exit": 0, "verdict": "ok"}, {"task": "b", "exit": 0}]
    )
    rows = {
        r["task"]: r for r in json.loads((tmp_path / "run_results.json").read_text())
    }
    assert rows["a"]["verdict"] == "ok" and rows["b"]["exit"] == 0
    (tmp_path / "run_results.json").write_text("not json")
    launch.write_run_results(tmp_path, [{"task": "c"}])
    assert [
        r["task"] for r in json.loads((tmp_path / "run_results.json").read_text())
    ] == ["c"]


def test_eval_summary_path_for_cell_and_file_mode(tmp_path):
    """The launcher knows where each evaluation form writes its summary."""
    (tmp_path / "eval" / "v007").mkdir(parents=True)
    assert launch.eval_summary_path(tmp_path, [], 7) == (
        tmp_path / "eval" / "v007" / "summary.json"
    )
    cmd = ["evaluate", "--task", "t", "--out", str(tmp_path / "eval" / "submission")]
    assert launch.eval_summary_path(tmp_path, cmd, None) == (
        tmp_path / "eval" / "submission" / "summary.json"
    )


def test_cell_state_flags_a_failed_evaluation(tmp_path):
    """status shows a cell whose evaluation died as unscored, not as done."""
    cell = tmp_path / "t"
    cell.mkdir()
    (cell / "run.json").write_text(
        json.dumps({"verdict": {"state": "ok"}, "evaluation": {"status": "failed"}})
    )
    assert "EVAL FAILED" in registry.cell_state(tmp_path, "t")
    (cell / "run.json").write_text(
        json.dumps({"verdict": {"state": "ok"}, "evaluation": {"status": "scored"}})
    )
    assert "EVAL FAILED" not in registry.cell_state(tmp_path, "t")
