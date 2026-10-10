"""``bigym-agent run``: one coding-agent session per task.

A *cell* is one task x one session, ``<root>/<task>/``::

    run.json          the session's configuration and result (schema below)
    transcript.jsonl  unified transcript (bigym.loco.agent.transcript)
    transcript.md     its human-readable rendering
    raw/              the harness's own event stream, stderr and home directory
    ledger.jsonl      every reset / episode, written by the environment server
    budget.json       the interaction budget, written by the server
    server.log        the server's own log
    sandbox/          exactly what the agent saw and wrote
    policies/         vNNN/policy.py + index.json (the policy version watcher)
    eval/vNNN/        hidden-seed evaluation of a version

``run`` drives one cell from end to end: build the sandbox
(``bigym-agent sandbox``), start the environment server (``bigym-agent serve``)
and wait for its workers, launch the agent harness (in its container, or on the
host under ``--isolation soft``), wait for it with a wall-clock limit, stop the
server, decide the verdict, write ``run.json`` and the transcript, and score the
submission (``bigym-agent evaluate``) unless ``--no-eval``. Several tasks run
in one command with ``--task a b c --parallel N``.

The pieces live next to this module: the harness command lines and their homes
(:mod:`~bigym.loco.agent.harnesses`), docker (:mod:`~bigym.loco.agent.containers`),
the GPU choice (:mod:`~bigym.loco.agent.gpu`) and the run registry behind
``status``, ``kill`` and ``gc`` (:mod:`~bigym.loco.agent.registry`).

Your own agent
--------------

``--harness custom --command '<shell command>'`` runs any program as the agent.
The cell is built exactly as for the shipped harnesses; the command then runs
on the host (``--isolation soft``; a container is refused, because the
benchmark cannot know what your image contains) with the sandbox as its working
directory and three variables in its environment::

    AGENT_SANDBOX         <cell>/sandbox, the directory the agent works in
    BIGYM_AGENT_PROMPT    <cell>/sandbox/PROMPT.md, the task statement
    BIGYM_AGENT_CELL      <cell>, for anything the agent wants to read there

Its stdout and stderr are captured in ``<cell>/raw/custom_stdout.txt`` and
``<cell>/raw/custom_stderr.txt``, and the stdout becomes the cell's transcript
(:func:`bigym.loco.agent.transcript.from_custom`), so ``bigym-agent report``
works for a custom session too. Nothing reports token usage, so ``usage`` is
empty and ``model`` is null unless ``--model`` was given. Everything after the
agent exits -- the verdict, the transcript, the hidden-seed evaluation -- is the
same as for a shipped harness.

run.json
--------

Written when the session starts and updated when it ends::

    {"schema": "bigym-agent/run.json v1",
     "task": "move_plate", "cell": "<abs>", "root": "<abs>",
     "harness": "codex" | "claude" | "custom",
     "model": "..." | null, "command": "<shell command>" | null,
     "effort": "high",
     "interface": "strict" | "tools", "tier": "images" | "privileged",
     "isolation": "container" | "soft", "image": "<docker image>" | null,
     "container_name": "..." | null, "docker_network": "...", "proxy": "...",
     "harness_version": "codex-cli 0.1.2",
     "auth": "api_key" | "subscription" | null,
     "budget_steps": 101000, "reset_cost": 200, "workers": 3,
     "demo": "video", "image_cap": "84x84",
     "egl_device": 4, "gpu": 0, "session_timeout_s": 10800,
     "resume_session": null,
     "start": "<iso8601>", "end": "<iso8601>", "wall_clock_s": 1234.5,
     "exit": 0,
     "verdict": {"state": "ok" | "interrupted" | "void", "reason": "..."},
     "budget": {"used": 85254, "cap": 101000},
     "usage": {"input": 0, "cached": 0, "output": 0, "reasoning": 0},
     "cost_usd": 1.23, "cost_source": "harness" | "price_table" | "unknown",
     "pricing_date": "...",
     "evaluation": {"version": 7, "command": ["..."]} | null}
"""

from __future__ import annotations

import dataclasses
import json
import os
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from bigym.cli import usage_error

from . import containers, gpu, transcript
from .cli import EnvToolsConfig, ProxyConfig, RunConfig, parse_command
from .harnesses import (
    AgentCommand,
    agent_command,
    agent_home,
    auth_report,
    model_mismatch,
    new_container_name,
)
from .registry import (
    STOP_NOTE,
    claim,
    close_row,
    open_registry,
    read_budget,
    record_process_group,
    stop_cells_of,
    stop_requested,
)
from .snapshots import version_dir

# The benchmark's own commands, run as subprocesses of this interpreter.
SELF = [sys.executable, "-m", "bigym.loco.agent"]
SETUP_LOCK = threading.Lock()


def log(message: str) -> None:
    """Print a timestamped launcher message.

    Args:
        message: The line to print.
    """
    print(f"[run {time.strftime('%H:%M:%S')}] {message}", flush=True)


def command_text(cmd: str | list[str]) -> str:
    """Render a command for printing, whether it is an argv or a shell string.

    Args:
        cmd: An argument vector, or the shell string of ``--harness custom``.

    Returns:
        A single shell-quoted line.
    """
    return cmd if isinstance(cmd, str) else shlex.join(cmd)


def ensure_docker_setup(
    harness: str, image: str | None, network: str, proxy: str, dry_run: bool
) -> list[list[str]]:
    """Bring the docker pieces up (once per process), or list them in a dry run.

    Raises:
        RuntimeError: A setup command failed.
    """
    with SETUP_LOCK:
        cmds = containers.docker_setup_commands(harness, image, network, proxy)
        for cmd in cmds:
            log(("would run: " if dry_run else "docker setup: ") + shlex.join(cmd))
            if dry_run:
                continue
            done = containers.run_docker(cmd)
            if done.returncode != 0:
                raise RuntimeError(
                    f"docker setup failed: {shlex.join(cmd)}\n{done.stderr.strip()}"
                )
    return cmds


def main_proxy(argv: list[str] | ProxyConfig | None = None) -> int:
    """``bigym-agent proxy status|up|down``: the allowlist proxy and its networks.

    Args:
        argv: The parsed settings, or command line arguments (None for
            ``sys.argv[1:]``).

    Returns:
        The process exit status.
    """
    args = parse_command(ProxyConfig, argv)
    if args.action == "down":
        for cmd in (["docker", "rm", "-f", containers.PROXY_CONTAINER],):
            done = containers.run_docker(cmd)
            print(
                shlex.join(cmd),
                "->",
                "ok" if done.returncode == 0 else done.stderr.strip(),
            )
        return 0
    if args.action == "up":
        image = containers.DEFAULT_IMAGES[args.harness] if args.harness else None
        cmds = ensure_docker_setup(
            args.harness or "codex",
            image,
            containers.DEFAULT_NETWORK,
            containers.DEFAULT_PROXY,
            False,
        )
        print("nothing to do" if not cmds else f"{len(cmds)} step(s) done")
    for network in (containers.DEFAULT_NETWORK, containers.EGRESS_NETWORK):
        present = containers.docker_has("network", network)
        print(f"network {network}: {'present' if present else 'missing'}")
    proxy = containers.PROXY_CONTAINER
    running = proxy in containers.running_containers()
    print(
        f"proxy {proxy}: {'running' if running else 'not running'}"
        + (f" on {' '.join(sorted(containers.proxy_networks()))}" if running else "")
    )
    for harness, image in containers.DEFAULT_IMAGES.items():
        present = containers.docker_has("image", image)
        print(f"image {image} ({harness}): {'present' if present else 'missing'}")
    return 0


# ------------------------------------------------------------ session pieces


def env_tools_flags(config: EnvToolsConfig) -> list[str]:
    """The command-line flags that give a subcommand this environment.

    Every setting that differs from its default becomes a flag, so the
    sandbox builder (which records the settings in the cell), the server and
    a file-mode evaluation are all handed the same environment.

    Args:
        config: The session's environment settings.

    Returns:
        The flags, in field order.
    """
    defaults = EnvToolsConfig()
    flags: list[str] = []
    for item in dataclasses.fields(config):
        value = getattr(config, item.name)
        if value == getattr(defaults, item.name):
            continue
        name = item.name.replace("_", "-")
        if isinstance(value, bool):
            flags.append(f"--{name}" if value else f"--no-{name}")
        else:
            flags += [f"--{name}", str(value)]
    return flags


def sandbox_command(task: str, cell: Path, args: RunConfig) -> list[str]:
    """Build the ``bigym-agent sandbox`` command line for a cell.

    Args:
        task: The task name.
        cell: The cell directory.
        args: The parsed ``run`` arguments.

    Returns:
        The command line.
    """
    # The sandbox builder records the harness name and writes the same files
    # either way (the Claude settings file is harmless for other harnesses).
    assert args.isolation is not None
    return (
        SELF
        + [
            "sandbox",
            "--task",
            task,
            "--cell",
            str(cell),
            "--budget-steps",
            str(args.budget_steps),
            "--workers",
            str(args.workers),
            "--harness",
            args.harness,
            "--effort",
            args.effort,
            "--isolation",
            args.isolation,
            "--demo",
            args.demo,
            "--demo-episodes",
            str(args.demo_episodes),
        ]
        + env_tools_flags(args.env_tools)
        + (["--force"] if args.force else [])
    )


def serve_command(task: str, cell: Path, args: RunConfig) -> list[str]:
    """Build the ``bigym-agent serve`` command line for a cell.

    Args:
        task: The task name.
        cell: The cell directory.
        args: The parsed ``run`` arguments.

    Returns:
        The command line.
    """
    return (
        SELF
        + [
            "serve",
            "--task",
            task,
            "--sandbox",
            str(cell / "sandbox"),
            "--ledger-dir",
            str(cell),
            "--workers",
            str(args.workers),
            "--budget-steps",
            str(args.budget_steps),
            "--reset-cost",
            str(args.reset_cost),
            "--allowed-seeds",
            str(cell / "allowed_seeds.json"),
        ]
        + env_tools_flags(args.env_tools)
        + (["--client-root", "/work"] if args.isolation == "container" else [])
    )


# Environment the evaluation subprocess depends on, recorded in run.json.
EVAL_ENV_KEYS = (
    "MUJOCO_GL",
    "MUJOCO_EGL_DEVICE_ID",
    "CUDA_VISIBLE_DEVICES",
    "BIGYM_AGENT_EVAL_JOBS",
)


def eval_summary_path(cell: Path, command: list[str], version: int | None) -> Path:
    """Where the evaluation started by ``command`` writes its summary.json."""
    if version is not None:
        return version_dir(cell, version, "eval") / "summary.json"
    if "--out" in command:
        return Path(command[command.index("--out") + 1]) / "summary.json"
    return cell / "eval" / "submission" / "summary.json"


def eval_command(
    task: str, cell: Path, args: RunConfig
) -> tuple[list[str], int | None]:
    """Build the hidden-seed evaluation command for a finished cell.

    The cell mode (``evaluate <cell> --version N``) scores the last policy
    version the snapshot watcher recorded, in the environment the sandbox
    builder recorded in the cell. A cell without ``policies/index.json`` (no
    watcher, or a session that wrote no version) falls back to scoring the
    sandbox's ``policy.py`` directly, with the environment given as flags.

    Args:
        task: The task name.
        cell: The cell directory.
        args: The parsed ``run`` arguments.

    Returns:
        ``(command, version)``; ``version`` is None for the fallback form.
    """
    index = cell / "policies" / "index.json"
    if index.exists():
        try:
            versions = json.loads(index.read_text()).get("versions") or []
        except (json.JSONDecodeError, OSError):
            versions = []
        if versions:
            version = int(versions[-1]["version"])
            command = SELF + [
                "evaluate",
                str(cell),
                "--version",
                str(version),
                "--episodes",
                str(args.eval_episodes),
                "--spot-check",
                str(args.spot_check),
            ]
            if args.force:
                command += ["--force"]
            return command, version
    model = args.model or "none"
    env_tools = args.env_tools
    label = f"{args.harness}_{model}_{env_tools.tier}_{env_tools.interface}_{args.demo}"
    out = cell / "eval" / "submission"
    return (
        SELF
        + [
            "evaluate",
            "--task",
            task,
            "--policy",
            str(cell / "sandbox" / "policy.py"),
            "--out",
            str(out),
            "--episodes",
            str(args.eval_episodes),
            "--spot-check",
            str(args.spot_check),
            "--jobs",
            str(args.eval_jobs),
            "--label",
            label,
            "--video-dir",
            str(out / "videos"),
            "--video-prefix",
            f"{task}_{args.harness}_",
        ]
        + env_tools_flags(env_tools),
        None,
    )


def submission_state(
    cell: Path, rc: int, container_name: str | None
) -> tuple[str, str]:
    """Decide what a finished session's submission is worth.

    ``"ok"`` evaluates, ``"interrupted"`` evaluates but records that the session
    was cut short, ``"void"`` does not evaluate at all.

    The untouched-template case is the one that matters: the sandbox builder
    always writes a policy template, so a crashed session still leaves a
    valid-looking ``policy.py``, and evaluating it scores the empty shell 0/100,
    indistinguishable from a real failure. The comparison is against the
    archived bytes the builder actually handed over (``initial_policy.py`` in
    the cell), not a reconstruction of them: the template differs per interface
    and per action layout, and a reconstruction silently stops matching.

    Args:
        cell: The cell directory.
        rc: The agent process's exit status.
        container_name: The container the agent ran in, or None.

    Returns:
        ``(state, reason)``.
    """
    submitted = cell / "sandbox" / "policy.py"
    if rc == 125 and container_name is not None:
        return "void", "docker could not start the container (exit 125)"
    if not submitted.exists():
        return "void", "no policy.py in the sandbox"
    initial = cell / "initial_policy.py"
    if initial.exists():
        untouched = submitted.read_bytes() == initial.read_bytes()
    else:
        # Cells whose builder archived no template: fall back to the marker the
        # template's act() carries. A real policy that kept the TODO comment
        # would have to be under 45 lines to be mistaken for one.
        source = submitted.read_text(errors="replace")
        untouched = (
            "# TODO: your control logic here" in source
            and len(source.splitlines()) < 45
        )
    if untouched:
        # An agent that "finished" without touching the template submitted
        # nothing, however it exited.
        return "void", "the submission is the untouched template, byte for byte"
    return ("interrupted", f"exit={rc}") if rc != 0 else ("ok", "")


# -------------------------------------------------------------- the session


def wait_ready(server: subprocess.Popen, log_path: Path, args: RunConfig) -> str | None:
    """Wait for the server's per-worker ``ready`` lines; returns an error or None."""
    start = time.time()
    while True:
        text = log_path.read_text(errors="replace") if log_path.exists() else ""
        if text.count("ready") >= args.workers:
            return None
        if (
            server.poll() is not None
            or time.time() - start > args.server_start_timeout_s
        ):
            return (
                f"server failed to start ({text.count('ready')}/{args.workers} workers "
                f"ready after {time.time() - start:.0f} s)"
            )
        time.sleep(2)


def stop_server(server: subprocess.Popen) -> None:
    """Stop the environment server: SIGTERM, then SIGKILL after 30 s.

    SIGTERM lets the server stop its workers; killing it outright orphans them.
    """
    if server.poll() is not None:
        return
    server.terminate()
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()


def run_agent(
    spec: AgentCommand,
    cell: Path,
    args: RunConfig,
    on_start: Callable[[int], None] | None = None,
) -> int:
    """Run the agent to completion (or to the wall-clock limit); returns its exit.

    ``on_start`` is called with the agent's process group once it is running.
    """
    mode = "a" if args.resume else "w"
    with open(spec.out, mode) as out, open(spec.err, mode) as err:
        try:
            # The agent runs in its own process group so that the wall-clock
            # limit kills the whole tree: killing only the wrapper leaves the
            # harness running for hours.
            proc = subprocess.Popen(
                spec.cmd,
                shell=spec.shell,
                cwd=cell / "sandbox",
                env=spec.env,
                stdin=subprocess.PIPE if spec.stdin is not None else subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                text=spec.stdin is not None,
                start_new_session=True,
            )
            if on_start is not None:
                on_start(proc.pid)
            try:
                proc.communicate(input=spec.stdin, timeout=args.session_timeout_s)
                return proc.returncode
            except subprocess.TimeoutExpired:
                log(
                    f"session hit the {args.session_timeout_s} s wall-clock limit; "
                    "killing its process group"
                )
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                return -1
        finally:
            if spec.container:
                # A timeout or a signal must not leave the container running.
                subprocess.run(
                    containers.docker_argv(["docker", "rm", "-f", spec.container]),
                    capture_output=True,
                )
            if spec.secrets is not None:
                shutil.rmtree(spec.secrets, ignore_errors=True)


def write_run_json(cell: Path, data: dict) -> None:
    """Write the cell's ``run.json``."""
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "run.json").write_text(json.dumps(data, indent=1) + "\n")


def stamp() -> str:
    """The current local time as an ISO-8601 stamp with offset."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def run_config(
    task: str,
    cell: Path,
    args: RunConfig,
    spec: AgentCommand,
    egl: int,
    gpu_index: int,
) -> dict:
    """Build the cell's ``run.json`` as it stands when the session starts.

    The result-side fields (``end``, ``exit``, ``verdict``, ``budget``,
    ``usage``, ``cost_usd``, ``evaluation``) are present and null until the
    session finishes; the module docstring documents the schema.

    Args:
        task: The task name.
        cell: The cell directory.
        args: The parsed ``run`` arguments.
        spec: The agent command.
        egl: The EGL device the session renders on.
        gpu_index: The ``nvidia-smi`` index of that device's GPU.

    Returns:
        The run record.
    """
    container = args.isolation == "container"
    return {
        "schema": "bigym-agent/run.json v1",
        "task": task,
        "cell": str(cell),
        "root": str(args.root),
        "harness": args.harness,
        "model": args.model,
        "command": spec.command,
        "effort": args.effort,
        "service_tier": args.service_tier,
        "interface": args.env_tools.interface,
        "tier": args.env_tools.tier,
        "isolation": args.isolation,
        "image": spec.image,
        "container_name": spec.container,
        "docker_network": args.docker_network if container else None,
        "proxy": args.proxy if container else None,
        "harness_version": spec.version,
        "auth": spec.auth,
        "budget_steps": args.budget_steps,
        "reset_cost": args.reset_cost,
        "workers": args.workers,
        "demo": args.demo,
        "image_cap": args.env_tools.image_cap,
        "egl_device": egl,
        "gpu": gpu_index,
        "session_timeout_s": args.session_timeout_s,
        "resume_session": spec.resume,
        "start": stamp(),
        "end": None,
        "wall_clock_s": None,
        "exit": None,
        "verdict": None,
        "budget": None,
        "usage": None,
        "cost_usd": None,
        "cost_source": None,
        "pricing_date": None,
        "evaluation": None,
    }


def session_device(args: RunConfig) -> tuple[int, int]:
    """Return the session's ``(EGL device, nvidia-smi index)``.

    Args:
        args: The parsed ``run`` arguments.

    Returns:
        The ``--gpu`` (an ``nvidia-smi`` index, mapped to its EGL device), a
        placeholder in a dry run, or the GPU with the most free memory.
    """
    mapping = gpu.egl_map()
    if mapping:
        log(
            "EGL map (nvidia-smi index -> EGL device): "
            + ", ".join(f"{k}:{v}" for k, v in sorted(mapping.items()))
        )
    if args.gpu is not None:
        return mapping.get(args.gpu, args.gpu), args.gpu
    if args.dry_run:
        return 0, 0
    return gpu.pick_egl(args.min_free_mib)


def session_env(args: RunConfig, egl: int, cuda: int) -> dict[str, str]:
    """The environment of the sandbox builder, the server and the evaluation."""
    env = dict(os.environ, MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID=str(egl))
    env["CUDA_VISIBLE_DEVICES"] = str(cuda)
    env["BIGYM_AGENT_EVAL_JOBS"] = str(args.eval_jobs)
    return env


def print_dry_run(
    task: str,
    cell: Path,
    args: RunConfig,
    device: tuple[int, int],
    sandbox_cmd: list[str],
    serve_cmd: list[str],
    spec: AgentCommand,
) -> None:
    """Print every command a session would run."""
    egl, cuda = device
    print(f"{cell.parent.name}/{task}: egl {egl} (gpu {cuda}), cell {cell}")
    print(f"  sandbox: {shlex.join(sandbox_cmd)}")
    print(f"  serve  : {shlex.join(serve_cmd)}")
    if spec.version_cmd:
        print(f"  version: {shlex.join(spec.version_cmd)}")
    print(f"  harness: {command_text(spec.cmd)}")
    if args.harness == "custom":
        print(f"           shell in {cell / 'sandbox'}")
        print(f"           AGENT_SANDBOX={cell / 'sandbox'}")
        print(f"           BIGYM_AGENT_PROMPT={cell / 'sandbox' / 'PROMPT.md'}")
        print(f"           BIGYM_AGENT_CELL={cell}")
    print(f"           stdin: {'PROMPT.md' if spec.stdin else 'none'}")
    print(f"           stdout: {spec.out}  stderr: {spec.err}")
    print(f"           MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID={egl}")
    if args.eval:
        print(f"  evaluate: {shlex.join(eval_command(task, cell, args)[0])}")
    print("  report  : " + shlex.join(SELF + ["report", str(cell)]))


def build_sandbox(
    cell: Path, args: RunConfig, env: dict[str, str], sandbox_cmd: list[str]
) -> None:
    """Build the cell's sandbox, or check that a resumed one is there.

    Raises:
        RuntimeError: The builder failed, or there is nothing to resume.
    """
    if not args.resume:
        # The same environment the server gets: the builder renders the
        # demonstration, so it needs EGL on the session's own device.
        done = subprocess.run(sandbox_cmd, env=env, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(
                f"sandbox build failed ({done.returncode}): "
                f"{(done.stderr or done.stdout).strip()[-400:]}"
            )
    elif not (cell / "sandbox" / "policy.py").exists():
        raise RuntimeError(f"{cell / 'sandbox'} has no policy.py; nothing to resume")


def run_with_server(
    task: str,
    cell: Path,
    args: RunConfig,
    env: dict[str, str],
    serve_cmd: list[str],
    spec: AgentCommand,
    on_start: Callable[[int], None] | None = None,
) -> int:
    """Start the environment server, run the agent against it, stop it.

    Returns:
        The agent's exit status.

    Raises:
        RuntimeError: The server's workers did not come up.
    """
    log(f"{task}: starting the environment server ({args.workers} workers)")
    server_log = cell / "server.log"
    with open(server_log, "a" if args.resume else "w") as handle:
        server = subprocess.Popen(
            serve_cmd, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
    try:
        problem = wait_ready(server, server_log, args)
        if problem:
            raise RuntimeError(problem)
        launched = args.model or command_text(spec.cmd)
        log(f"{task}: server ready, launching {args.harness} ({launched})")
        return run_agent(spec, cell, args, on_start)
    finally:
        stop_server(server)


def record_outcome(
    cell: Path, args: RunConfig, config: dict, rc: int, verdict: dict, started: float
) -> None:
    """Fill the result side of ``run.json`` and write the transcript.

    Args:
        cell: The cell directory.
        args: The parsed ``run`` arguments.
        config: The run record, updated in place.
        rc: The agent's exit status.
        verdict: ``{"state", "reason"}`` of the submission.
        started: When the session started (``time.time()``).
    """
    summary = transcript.write_transcript(cell, args.model)
    config.update(
        {
            "end": stamp(),
            "wall_clock_s": round(time.time() - started, 1),
            "exit": rc,
            "verdict": verdict,
            "budget": read_budget(cell),
            "usage": summary.get("tokens"),
            "cost_usd": summary.get("usd"),
            "cost_source": summary.get("cost_source"),
            "pricing_date": (
                transcript.PRICING_DATE
                if summary.get("cost_source") == "price_table"
                else None
            ),
        }
    )
    write_run_json(cell, config)


def rejection(summary: Path) -> str | None:
    """The ``rejected`` reason an evaluation's summary.json gives, if any."""
    try:
        data = json.loads(summary.read_text())
    except (OSError, ValueError):
        return None
    return data.get("rejected") if isinstance(data, dict) else None


def evaluate_submission(
    task: str,
    cell: Path,
    args: RunConfig,
    env: dict[str, str],
    config: dict,
    on_start: Callable[[int], None] | None = None,
) -> dict:
    """Score the cell's submission and record the evaluation in ``run.json``.

    ``on_start`` is called with the evaluation's process group once it runs.

    Returns:
        The ``evaluation`` block of ``run.json``.
    """
    command, version = eval_command(task, cell, args)
    # The environment the evaluation needs to be re-run by hand
    # (MUJOCO_GL, the EGL device, the job count) is recorded next to
    # the command; copying the command alone fails at GL init.
    evaluation = {
        "version": version,
        "command": command,
        "env": {k: env[k] for k in EVAL_ENV_KEYS if k in env},
        "status": "detached" if args.eval_detached else "running",
        "summary": None,
        "exit": None,
    }
    config["evaluation"] = evaluation
    write_run_json(cell, config)
    log(f"{task}: evaluating the submission ({shlex.join(command)})")
    with open(cell / "eval.log", "w") as handle:
        proc = subprocess.Popen(
            command,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        if on_start is not None:
            on_start(proc.pid)
        if args.eval_detached:
            return evaluation
        returncode = proc.wait()
    summary = eval_summary_path(cell, command, version)
    evaluation["exit"] = returncode
    if returncode == 0 and summary.exists():
        evaluation["status"] = "scored"
        evaluation["summary"] = str(summary)
        rejected = rejection(summary)
        if rejected:
            # Scored 0 because the policy imported the simulator.
            evaluation["status"] = "rejected"
            evaluation["reason"] = rejected
            log(f"{task}: policy rejected, scored 0 ({rejected})")
    else:
        # An evaluation that dies leaves the cell unscored
        # while the session itself was fine; say so loudly
        # and in run.json, so a sweep cannot lose the cell.
        evaluation["status"] = "failed"
        log(
            f"{task}: WARNING evaluation failed (exit "
            f"{returncode}, no {summary.name}); see "
            f"{cell / 'eval.log'}; re-run with "
            f"`bigym-agent evaluate {cell} --version {version}`"
        )
    write_run_json(cell, config)
    return evaluation


def cell_process_started(conn: sqlite3.Connection, row: int | None, pgid: int) -> None:
    """Record the group of a cell's agent or evaluation; stop it if the cell was killed."""
    record_process_group(conn, row, pgid)
    if stop_requested(conn, row):
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def claim_unless_cancelled(
    conn: sqlite3.Connection,
    cell: Path,
    task: str,
    info: dict,
    cancel: threading.Event | None,
) -> int | None:
    """Claim the cell, unless the run was interrupted.

    The interrupt is checked after the claim, so an interrupt either finds
    the row in the registry or is seen here.
    """
    row = claim(conn, cell, task, info)
    if row is not None and cancel is not None and cancel.is_set():
        close_row(conn, row, STOP_NOTE)
        return None
    return row


def run_session(
    task: str, args: RunConfig, cancel: threading.Event | None = None
) -> dict:
    """Run one task's session end to end.

    Args:
        task: The task name.
        args: The parsed ``run`` arguments.
        cancel: Set when the run is interrupted; the session then does not
            start.

    Returns:
        A result dictionary (``task``, and either ``error`` or ``exit``,
        ``verdict`` and ``budget``).
    """
    cell = (args.root / task).resolve()
    device = session_device(args)
    env = session_env(args, *device)

    scored = sorted((cell / "eval").glob("*/summary.json")) if cell.exists() else []
    if scored and not args.force and not args.dry_run:
        print(
            f"refused: {cell.parent.name}/{task} already has a scored evaluation "
            f"({scored[0]}); --force overwrites the cell"
        )
        return {"task": task, "error": "already scored"}

    container = args.isolation == "container"
    if container and args.docker_setup:
        ensure_docker_setup(
            args.harness,
            args.container or containers.DEFAULT_IMAGES.get(args.harness),
            args.docker_network,
            args.proxy,
            args.dry_run,
        )
    line, warning = auth_report(args)
    log(f"{task}: {line}")
    if warning:
        log(f"{task}: WARNING {warning}")

    sandbox_cmd = sandbox_command(task, cell, args)
    serve_cmd = serve_command(task, cell, args)
    name = new_container_name(task, cell, args) if container else None
    spec = agent_command(task, cell, args, dry_run=True, container_name=name)
    if args.dry_run:
        print_dry_run(task, cell, args, device, sandbox_cmd, serve_cmd, spec)
        return {"task": task, "dry_run": True}

    conn = open_registry(args.root)
    row = claim_unless_cancelled(
        conn,
        cell,
        task,
        {
            "container": spec.container,
            "egl": device[0],
            "gpu": device[1],
            "effort": args.effort,
            "image": spec.image,
            "harness": args.harness,
            "model": args.model,
            "cmd": command_text(spec.cmd),
        },
        cancel,
    )
    if row is None:
        conn.close()
        if cancel is not None and cancel.is_set():
            return {"task": task, "error": "interrupted"}
        return {"task": task, "error": "refused by the registry"}
    started = time.time()

    def started_process(pgid: int) -> None:
        cell_process_started(conn, row, pgid)

    config = run_config(task, cell, args, spec, *device)
    try:
        cell.mkdir(parents=True, exist_ok=True)
        write_run_json(cell, config)
        build_sandbox(cell, args, env, sandbox_cmd)
        # Now that the sandbox exists, build the command for real: it reads
        # PROMPT.md, writes the harness home and its credential file, and
        # probes the harness version.
        spec = agent_command(task, cell, args, container_name=name)
        config["harness_version"] = spec.version
        config["resume_session"] = spec.resume
        write_run_json(cell, config)
        rc = run_with_server(
            task,
            cell,
            args,
            env,
            serve_cmd,
            spec,
            on_start=started_process,
        )
        last = cell / "raw" / "codex_home" / "codex_last.txt"
        if last.exists():
            shutil.copy(last, cell / "raw" / "codex_last.txt")
        state, why = submission_state(cell, rc, spec.container)
        killed = stop_requested(conn, row)
        if killed and state != "void":
            state, why = "interrupted", "killed"
        log(f"{task}: session done (exit {rc}), verdict {state} {why}".rstrip())
        verdict = {"state": state, "reason": why}
        record_outcome(cell, args, config, rc, verdict, started)
        result = {
            "task": task,
            "exit": rc,
            "verdict": state,
            "budget": config["budget"],
        }
        if state != "void" and args.eval and not killed:
            evaluation = evaluate_submission(
                task, cell, args, env, config, on_start=started_process
            )
            if stop_requested(conn, row):
                verdict.update(state="interrupted", reason="killed")
                evaluation["status"] = "killed"
                result["verdict"] = "interrupted"
                write_run_json(cell, config)
            result["evaluation"] = evaluation["status"]
        elif state == "void" or killed:
            log(f"{task}: no evaluation, the cell stays unscored ({why})")
        return result
    except Exception as exc:  # the batch keeps going; the cell records why
        config["end"] = stamp()
        config["wall_clock_s"] = round(time.time() - started, 1)
        config["verdict"] = {"state": "void", "reason": f"{type(exc).__name__}: {exc}"}
        write_run_json(cell, config)
        return {"task": task, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        close_row(conn, row)
        conn.close()


# ------------------------------------------------------------------ the CLI


def parse_run_args(argv: list[str] | RunConfig | None = None) -> RunConfig:
    """Parse ``bigym-agent run`` arguments and resolve their defaults.

    Args:
        argv: The parsed settings, or command line arguments (None for
            ``sys.argv[1:]``).

    Returns:
        The settings, with ``root`` and ``codex_home`` resolved and
        ``isolation`` defaulted per harness kind.
    """
    args = parse_command(RunConfig, argv)
    if args.isolation is None:
        # A custom command is an arbitrary host program; only the shipped
        # harnesses have an image to run in.
        args.isolation = "soft" if args.harness == "custom" else "container"
    if args.harness == "custom":
        if not args.command:
            usage_error(
                "--harness custom needs --command '<shell command>'",
                prog="bigym-agent run",
            )
        if args.isolation == "container":
            usage_error(
                "--harness custom runs your command on this host: use "
                "--isolation soft (the benchmark ships no image for it)",
                prog="bigym-agent run",
            )
    elif args.model is None:
        usage_error(
            f"--model is required for --harness {args.harness}",
            prog="bigym-agent run",
        )
    else:
        mismatch = model_mismatch(args.harness, args.model)
        if mismatch:
            usage_error(mismatch, prog="bigym-agent run")
    if args.sessions < 1:
        usage_error("--sessions must be at least 1", prog="bigym-agent run")
    args.root = args.root.expanduser().resolve()
    args.codex_home = (
        args.codex_home
        or Path(os.environ.get("BIGYM_AGENT_CODEX_HOME", agent_home() / "codex_home"))
    ).expanduser()
    return args


def main(argv: list[str] | RunConfig | None = None) -> int:
    """Run ``--sessions`` sessions per task, ``--parallel`` at a time.

    Args:
        argv: The parsed settings, or command line arguments (None for
            ``sys.argv[1:]``).

    Returns:
        The process exit status: 0 when every session ran.
    """
    args = parse_run_args(argv)
    tasks = args.task
    if not tasks:
        print("bigym-agent run: --task is required", file=sys.stderr)
        return 2
    roots = session_roots(args.root, args.sessions)
    results: dict[Path, list[dict]] = {root: [] for root in roots}
    queue = [(root, task) for root in roots for task in tasks]
    lock, cancel = threading.Lock(), threading.Event()

    def worker() -> None:
        while True:
            with lock:
                if not queue:
                    return
                root, task = queue.pop(0)
            try:
                result = run_session(task, dataclasses.replace(args, root=root), cancel)
            except Exception as exc:
                result = {"task": task, "error": f"{type(exc).__name__}: {exc}"}
            with lock:
                results[root].append(result)
                log(f"{root.name}/{task}: {result}")

    threads = [
        threading.Thread(target=worker, daemon=True)
        for _ in range(max(1, min(args.parallel, len(queue))))
    ]
    for thread in threads:
        thread.start()
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        # Agents and evaluations run in their own process groups, out of reach
        # of the terminal's Ctrl+C: stop them as `kill` would, and let each
        # worker record its cell before exiting.
        log("interrupted: stopping the running cells")
        cancel.set()
        with lock:
            queue.clear()
        for root in roots:
            stop_cells_of(root, os.getpid())
        for thread in threads:
            thread.join(timeout=120)
        return 130
    if not args.dry_run:
        for root, entries in results.items():
            write_run_results(root, entries)
    return (
        1 if any("error" in r for entries in results.values() for r in entries) else 0
    )


def session_roots(root: Path, sessions: int) -> list[Path]:
    """The runs root of each session: ``root`` itself, or ``<root>_s1`` ... ``_sN``."""
    if sessions <= 1:
        return [root]
    return [root.with_name(f"{root.name}_s{k}") for k in range(1, sessions + 1)]


def write_run_results(root: Path, results: list[dict]) -> Path:
    """Merge this invocation's results into ``<root>/run_results.json`` by task.

    A root is usually filled by more than one ``run`` invocation (a sweep, then
    a retry of one cell), so the file keeps one entry per task, the newest
    winning, instead of only the last invocation's list. The per-cell
    ``run.json`` and the registry remain the authority.
    """
    path = Path(root) / "run_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, dict] = {}
    try:
        previous = json.loads(path.read_text())
    except (OSError, ValueError):
        previous = []
    for entry in list(previous if isinstance(previous, list) else []) + results:
        if isinstance(entry, dict) and entry.get("task"):
            merged[str(entry["task"])] = entry
    path.write_text(json.dumps(list(merged.values()), indent=1) + "\n")
    return path
