"""In-the-loop evaluation: the model is the policy, one agent session per episode.

    # 1) start the env server with the protocol seeds allowed:
    #    bigym-agent serve --task T --sandbox ROOT/sandbox --ledger-dir ROOT/ledger \
    #        --workers 8 --budget-steps 100000000 --allow-eval-seeds
    # 2) run:
    #    bigym-agent inloop --task T --root ROOT --episodes 100 --workers 8

Each episode i gets its own directory ``ROOT/sandbox/episodes/ep_<i>`` with the
``act`` command bound to worker ``i mod workers``, a fixed prompt, and the
agent harness's isolation settings. The orchestrator resets the worker to seed
``EVAL_SEED_BASE + i``, launches the agent session, and reads the outcome back from the
server.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from bigym.loco.eval.protocol import EVAL_SEED_BASE

from .. import wire
from ..cli import InloopConfig, parse_command
from ..settings import CLAUDE_HOST_FLAGS, CODEX_HOST_FLAGS, claude_settings_soft

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
TEMPLATES = PACKAGE / "templates"

# Where the Codex CLI keeps its own state; kept outside the episode directories so
# a session cannot read another session's history.
CODEX_HOME = Path(
    os.environ.get(
        "BIGYM_AGENT_CODEX_HOME", Path.home() / ".bigym-agent" / "codex_home"
    )
)
ACT_WRAPPER = '#!/bin/sh\nexec {py} "$(dirname "$0")/act.py" "$@"\n'


def project_slug(path: Path) -> str:
    """Return the agent harness's per-project directory name for a path.

    Args:
        path: The working directory of a session.

    Returns:
        The slug the harness derives from it.
    """
    return "".join(ch if ch.isalnum() else "-" for ch in str(path))


def clear_auto_memory(cwd: Path) -> None:
    """Drop the auto-memory the agent harness would reload for this directory.

    Every episode must start from the same state, so nothing the model wrote
    about an earlier episode may survive into the next one.

    Args:
        cwd: The working directory of the session.
    """
    mem = Path.home() / ".claude" / "projects" / project_slug(cwd) / "memory"
    if mem.exists():
        shutil.rmtree(mem, ignore_errors=True)


# USD per 1M tokens (input, cached input, output): the provider's list price on the
# run date, used only to report what a block of episodes cost.
OPENAI_PRICES = {"gpt-6-astra": (10.0, 1.0, 50.0)}


def codex_cost_usd(usage: dict | None, model: str) -> float | None:
    """Price one session's token usage.

    Args:
        usage: The token counts reported by the CLI.
        model: The model name, which must be in :data:`OPENAI_PRICES`.

    Returns:
        The cost in USD, or None when no price is known.
    """
    if not usage or model not in OPENAI_PRICES:
        return None
    pi, pc, po = OPENAI_PRICES[model]
    cached = float(usage.get("cached_input_tokens") or 0)
    uncached = max(0.0, float(usage.get("input_tokens") or 0) - cached)
    return (
        uncached * pi + cached * pc + float(usage.get("output_tokens") or 0) * po
    ) / 1e6


def parse_codex_events(text: str, ep: Path) -> dict:
    """Reduce a Codex ``exec --json`` event stream to the fields we record.

    Args:
        text: The JSONL event stream.
        ep: The episode directory (holds the final message file).

    Returns:
        The same fields the other CLI's result JSON provides (cost is None:
        tokens only).
    """
    usage, thread, items, last_msg, error = None, None, 0, "", False
    for line in text.splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = d.get("type", "")
        if t == "thread.started":
            thread = d.get("thread_id") or d.get("id")
        elif t == "item.completed":
            items += 1
            it = d.get("item", {})
            if it.get("type") == "agent_message":
                last_msg = it.get("text", "") or last_msg
        elif t == "turn.completed":
            usage = d.get("usage") or usage
        elif t in ("error", "turn.failed"):
            error = True
    try:
        last_msg = (ep / "codex_last.txt").read_text() or last_msg
    except FileNotFoundError:
        pass
    return {
        "is_error": error,
        "num_turns": items,
        "total_cost_usd": None,
        "usage": usage,
        "session_id": thread,
        "result": last_msg,
    }


def rpc(sock_path: str, msg: dict) -> dict:
    """Send one request to a server worker and return the reply.

    Args:
        sock_path: Path of the worker's unix socket.
        msg: The request.

    Returns:
        The reply dictionary.

    Raises:
        RuntimeError: The server refused the request.
    """
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path)
    try:
        wire.send(s, msg)
        r = wire.recv(s)
    finally:
        s.close()
    if not r.get("ok"):
        raise RuntimeError(r.get("error"))
    return r


def wilson(k: int, n: int, z: float = 1.96):
    """Return the Wilson score interval for k successes out of n.

    Args:
        k: Successes.
        n: Episodes.
        z: Normal quantile (1.96 = 95%).

    Returns:
        ``(low, high)``.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def make_episode_dir(
    root: Path,
    i: int,
    worker: int,
    task: str,
    max_commands: int,
    tier: str = "images",
    smooth: bool = False,
    task_doc: Path | None = None,
) -> Path:
    """Build one episode's directory: the act command, the prompt, the settings.

    Args:
        root: The run root (holds ``sandbox/``).
        i: Episode index.
        worker: Worker index this episode is bound to.
        task: Task name.
        max_commands: Per-episode command cap.
        tier: Observation tier.
        smooth: Ramp commanded targets at the demo-collection slew limits.
        task_doc: A task document to hand the model, or None for a one-liner.

    Returns:
        The episode directory.
    """
    ep = root / "sandbox" / "episodes" / f"ep_{i:03d}"
    if ep.exists():
        shutil.rmtree(ep)
    (ep / "frames").mkdir(parents=True)
    (ep / "harness").mkdir()
    (ep / ".claude").mkdir()
    shutil.copy(PACKAGE / "wire.py", ep / "harness" / "wire.py")
    shutil.copy(HERE / "act.py", ep / "act.py")
    (ep / "act").write_text(ACT_WRAPPER.format(py=sys.executable))
    os.chmod(ep / "act", 0o755)
    if task_doc is not None:
        shutil.copy(task_doc, ep / "task.md")
    else:
        (ep / "task.md").write_text(f"# {task}\n\nTask: {task.replace('_', ' ')}.\n")
    prompt = (
        (TEMPLATES / "PROMPT_INLOOP.md")
        .read_text()
        .format(task=task, max_commands=max_commands)
    )
    if tier == "images":
        prompt += (TEMPLATES / "PROMPT_INLOOP_IMAGES.md").read_text()
    (ep / "PROMPT.md").write_text(prompt)
    (ep / "episode.json").write_text(
        json.dumps(
            {
                "socket": str(root / "sandbox" / "sock" / f"w{worker}.sock"),
                "episode_rel": f"episodes/ep_{i:03d}",
                "max_commands": max_commands,
                "worker": worker,
                "tier": tier,
                "smooth": smooth,
            }
        )
    )
    settings = claude_settings_soft(ep)
    # The model acts only through ./act: no file it writes can reach the environment.
    settings["permissions"]["deny"] += ["Edit", "Write"]
    settings["permissions"]["allow"] = [
        a for a in settings["permissions"]["allow"] if a != "Bash(cd:*)"
    ]
    (ep / ".claude" / "settings.json").write_text(json.dumps(settings, indent=1))
    (ep / "claude_settings.json").write_text(json.dumps(settings, indent=1))
    return ep


def run_episode(root: Path, i: int, worker: int, args) -> dict:
    """Run one episode end to end and return its record.

    Args:
        root: The run root.
        i: Episode index.
        worker: Worker index this episode is bound to.
        args: The parsed command line.

    Returns:
        The episode record.
    """
    ep = make_episode_dir(
        root,
        i,
        worker,
        args.task,
        args.max_commands,
        args.tier,
        args.smooth,
        args.task_doc,
    )
    sock = str(root / "sandbox" / "sock" / f"w{worker}.sock")
    seed = EVAL_SEED_BASE + args.seed_offset + i
    rpc(sock, {"op": "reset", "seed": seed, "max_commands": args.max_commands})
    clear_auto_memory(ep)
    t0 = time.time()
    if args.harness == "codex":
        cmd = [
            "codex", "exec", "--skip-git-repo-check", "-C", str(ep),
            "--dangerously-bypass-approvals-and-sandbox", "-m", args.model,
            "-c", f"model_reasoning_effort={args.effort}", *CODEX_HOST_FLAGS, "--json",
            "-o", str(ep / "codex_last.txt"), (ep / "PROMPT.md").read_text(),
        ]  # fmt: skip
        env = dict(os.environ, CODEX_HOME=str(CODEX_HOME))
        out = subprocess.run(
            cmd,
            cwd=ep,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=args.episode_timeout_s,
            env=env,
        )
        (ep / "codex_events.jsonl").write_text(out.stdout)
        (ep / "codex_stderr.txt").write_text(out.stderr)
        res = parse_codex_events(out.stdout, ep)
        res["total_cost_usd"] = codex_cost_usd(res.get("usage"), args.model)
    else:
        cmd = [
            "claude", "-p", (ep / "PROMPT.md").read_text(), "--setting-sources", "project",
            "--settings", str(ep / "claude_settings.json"), "--permission-mode", "dontAsk",
            "--allowedTools", "Read,Bash,Glob,Grep", *CLAUDE_HOST_FLAGS, "--model", args.model,
            "--max-turns", str(args.max_turns), "--output-format", "json",
        ]  # fmt: skip
        env = dict(os.environ, CLAUDE_CODE_DISABLE_AUTO_MEMORY="1")
        out = subprocess.run(
            cmd,
            cwd=ep,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=args.episode_timeout_s,
            env=env,
        )
        (ep / "claude_out.json").write_text(out.stdout)
        (ep / "claude_stderr.txt").write_text(out.stderr)
        try:
            res = json.loads(out.stdout)
        except json.JSONDecodeError:
            res = {"is_error": True, "result": out.stdout[-500:]}
    st = rpc(sock, {"op": "status"})
    finished = (
        not st["episode_active"]
        and st.get("last_result")
        and st["last_result"].get("seed") == seed
    )
    if finished:
        lr = st["last_result"]
        success, length, term = int(lr["success"]), int(lr["length"]), lr["termination"]
    else:
        success = 0
        length = int(st.get("length") or 0)
        term = "agent_stopped" if st["episode_active"] else "unknown"
        if st["episode_active"]:
            rpc(sock, {"op": "abandon"})
    counter = ep / "commands_used.txt"
    n_cmd = int(counter.read_text()) if counter.exists() else 0
    rec = {
        "episode": i,
        "seed": seed,
        "worker": worker,
        "success": success,
        "length": length,
        "termination": term,
        "commands": n_cmd,
        "turns": res.get("num_turns"),
        "cost_usd": res.get("total_cost_usd"),
        "is_error": bool(res.get("is_error")),
        "session_id": res.get("session_id"),
        "wall_s": round(time.time() - t0, 1),
        "usage": res.get("usage"),
        "final_text": str(res.get("result") or "")[:600],
    }
    (ep / "result.json").write_text(json.dumps(rec, indent=1))
    return rec


def main(argv: list[str] | InloopConfig | None = None):
    """Run a block of in-the-loop episodes and write the ledger.

    Args:
        argv: The parsed settings, or command line arguments (None for
            ``sys.argv[1:]``).
    """
    args = parse_command(InloopConfig, argv)
    root = args.root.resolve()
    (root / "ledger").mkdir(parents=True, exist_ok=True)
    socks = sorted((root / "sandbox" / "sock").glob("w*.sock"))
    if len(socks) < args.workers:
        sys.exit(
            f"need {args.workers} server workers, found {len(socks)} sockets in "
            f"{root / 'sandbox' / 'sock'}"
        )
    episodes = list(range(args.first_episode, args.first_episode + args.episodes))
    results: dict[int, dict] = {}
    lock = threading.Lock()
    queue = list(episodes)
    version = subprocess.run(
        ["codex" if args.harness == "codex" else "claude", "--version"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    (root / "ledger" / f"{args.label}_meta.txt").write_text(
        f"start {time.strftime('%Y-%m-%dT%H:%M:%S')}\nharness {args.harness}\n"
        f"model {args.model}\neffort {args.effort}\ntier {args.tier}\n"
        f"smooth {args.smooth}\nmax_commands {args.max_commands}\n"
        f"max_turns {args.max_turns}\nworkers {args.workers}\nversion {version}\n"
    )

    def worker_loop(k: int):
        while True:
            with lock:
                if not queue:
                    return
                i = queue.pop(0)
            try:
                rec = run_episode(root, i, k, args)
            except Exception as exc:  # keep the pool alive
                rec = {
                    "episode": i,
                    "seed": EVAL_SEED_BASE + args.seed_offset + i,
                    "worker": k,
                    "success": 0,
                    "length": 0,
                    "termination": "orchestrator_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            with lock:
                results[i] = rec
                done = len(results)
                ok = sum(r["success"] for r in results.values())
                cost = sum((r.get("cost_usd") or 0.0) for r in results.values())
                print(
                    f"[inloop] ep {i} seed {rec['seed']} worker {k}: "
                    f"success={rec['success']} len={rec['length']} "
                    f"end={rec['termination']} cmds={rec.get('commands')} "
                    f"turns={rec.get('turns')} ${rec.get('cost_usd') or 0:.2f}  "
                    f"| {ok}/{done} done, ${cost:.2f} so far",
                    flush=True,
                )
                _write(root, args, results)

    threads = [
        threading.Thread(target=worker_loop, args=(k,), daemon=True)
        for k in range(args.workers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    _write(root, args, results, final=True)


def _write(root: Path, args, results: dict, final: bool = False):
    recs = [results[i] for i in sorted(results)]
    with open(root / "ledger" / f"{args.label}_episodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "episode",
                "seed",
                "success",
                "length",
                "termination",
                "commands",
                "turns",
                "cost_usd",
                "wall_s",
                "is_error",
            ]
        )
        for r in recs:
            w.writerow(
                [
                    r["episode"],
                    r["seed"],
                    r["success"],
                    r["length"],
                    r["termination"],
                    r.get("commands"),
                    r.get("turns"),
                    r.get("cost_usd"),
                    r.get("wall_s"),
                    r.get("is_error"),
                ]
            )
    n = len(recs)
    k = sum(r["success"] for r in recs)
    lo, hi = wilson(k, n)
    token_keys = (
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
    )
    summary = {
        "task": args.task,
        "harness": args.harness,
        "model": args.model,
        "tier": args.tier,
        "smooth": args.smooth,
        "label": args.label,
        "episodes_done": n,
        "successes": k,
        "success_rate": (k / n) if n else None,
        "wilson95": [lo, hi],
        "cost_usd_total": sum((r.get("cost_usd") or 0.0) for r in recs),
        "tokens_total": (
            {
                key: sum(((r.get("usage") or {}).get(key) or 0) for r in recs)
                for key in token_keys
            }
            if args.harness == "codex"
            else None
        ),
        "mean_commands": (sum((r.get("commands") or 0) for r in recs) / n)
        if n
        else None,
        "mean_wall_s": (sum((r.get("wall_s") or 0) for r in recs) / n) if n else None,
        "terminations": {
            t: sum(r["termination"] == t for r in recs)
            for t in {r["termination"] for r in recs}
        },
        "errors": sum(bool(r.get("is_error")) for r in recs),
        "final": final,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (root / "ledger" / f"{args.label}_summary.json").write_text(
        json.dumps(summary, indent=1)
    )


if __name__ == "__main__":
    main()
