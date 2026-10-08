"""Replay batches: an agent policy rolled out into the demo viewer's format.

    bigym-agent replay <cell> --version 7 --seeds 620000-620004

Writes one ``seed<S>.npz`` per episode plus a ``metadata.json`` describing the
environment they were produced in, which is exactly the "replay batch" layout
``bigym-view`` reads (``bigym/vr/viewer/view_demos.py``) and
``bigym.loco.demos.rerender`` re-renders from. Each episode stores the full
physical state of every frame (``full_qpos``/``full_qvel``, ``T+1`` states
around ``T`` actions), so the rollout can be replayed and re-rendered without
running the policy again.

The batch is written by :mod:`bigym.loco.agent.batch`, whose recording
leaves the rollout bit-identical to an unrecorded one. As in the evaluator,
the policy runs in its own interpreter
(:class:`~bigym.loco.agent.policy_process.PolicyProcess`), never next to the
simulator it is scored on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from bigym.loco.eval.protocol import is_success

from .batch import (
    RecordingEnv,
    RunProgress,
    batch_metadata,
    batch_source,
    write_json,
    write_record,
)
from .cli import (
    EnvToolsConfig,
    ReplayConfig,
    limit_blas_threads,
    parse_command,
)
from .envtools import EnvTools
from .episode import run_episode
from .evaluate import record_seeds
from .policy_process import PolicyProcess
from .snapshots import cell_task, read_env_tools, resolve_policy


def parse_seeds(spec: str) -> list[int]:
    """Expand a seed specification such as ``620000,620004`` or ``620000-620004``.

    Args:
        spec: Comma-separated seeds and inclusive ``a-b`` ranges.

    Returns:
        The seeds in the order they were written.

    Raises:
        ValueError: The specification is empty or malformed.
    """
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part[1:]:
                first, _, last = part[1:].partition("-")
                lo, hi = int(part[0] + first), int(last)
                if hi < lo:
                    raise ValueError(f"empty seed range {part!r}")
                out.extend(range(lo, hi + 1))
            else:
                out.append(int(part))
        except ValueError as exc:
            raise ValueError(f"bad seed specification {spec!r}: {exc}") from exc
    if not out:
        raise ValueError(f"no seeds in {spec!r}")
    return out


def episode_limit(env_tools: EnvTools, max_steps: int | None = None) -> int:
    """Return how many steps one episode may run.

    Args:
        env_tools: The wrapper holding the environment.
        max_steps: A ``--max-steps`` cap, or None.

    Returns:
        The environment's own time limit, lowered by ``max_steps`` when that
        is smaller; 0 when neither is known.
    """
    limit = int(env_tools.time_limit)
    cap = int(max_steps or 0)
    if cap > 0:
        limit = min(limit, cap) if limit > 0 else cap
    return max(0, limit)


def record_episode(
    env_tools: EnvTools,
    policy_path,
    seed: int,
    max_steps: int | None = None,
    progress: RunProgress | None = None,
    policy_process: PolicyProcess | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Run one episode with a fresh policy instance and record it.

    Args:
        env_tools: The wrapper holding the environment.
        policy_path: Path of the policy file.
        seed: Episode seed.
        max_steps: Step cap, or None for the episode's own time limit.
        progress: A :class:`RunProgress` to report the step count to.
        policy_process: The process to run the policy in; None starts one
            for this episode.

    Returns:
        ``(record, arrays)``: the per-episode record and the batch arrays.
    """
    if policy_process is None:
        with PolicyProcess() as proc:
            return record_episode(
                env_tools, policy_path, seed, max_steps, progress, proc
            )
    if progress is not None:
        progress.begin(int(seed), episode_limit(env_tools, max_steps))
    env = RecordingEnv(env_tools, progress=progress)
    policy = policy_process.load(policy_path)
    rec = run_episode(env, policy, int(seed), max_steps)
    rec["success"] = int(is_success(env_tools.outer))
    rec["termination"] = env_tools.outer.episode_termination()
    rec["fell"] = bool(env_tools.outer.episode_fell())
    if progress is not None:
        progress.finish(int(seed), int(rec["length"]))
    return rec, env.recorder.arrays()


def replay_seeds(
    task: str,
    policy_path: Path,
    seeds,
    out_dir: Path,
    env_tools: EnvToolsConfig,
    *,
    source: dict | None = None,
    video: bool = True,
    max_steps: int | None = None,
    force: bool = False,
) -> list[Path]:
    """Roll a policy out on given seeds and write a replay batch.

    Each episode's ``.npz`` is written as that episode finishes, so a reader
    (the viewer's job panel) can count them for progress; while an episode
    runs, ``<out_dir>/.progress.json`` says which step it is on (see
    :class:`RunProgress`), and it is removed when the run finishes cleanly.

    A version directory is **added to**, not replaced: a seed whose
    ``seed<S>.npz`` is already in the batch is kept (``seed S: already
    recorded``) and only the missing ones are rolled out, so asking for one
    more seed of a version that has a replay costs one episode rather than
    a refusal. ``metadata.json`` describes the substrate, not the seed set,
    so it is written only when it is absent. ``force`` re-records every seed
    asked for and rewrites the metadata.

    Args:
        task: Task name.
        policy_path: Path of the policy file.
        seeds: Seeds to replay.
        out_dir: ``<cell>/replays/vNNN``; ``batch/`` and ``videos/`` go here.
        env_tools: The environment the cell's session ran against.
        source: The ``source`` block for the metadata.
        video: Also record one mp4 per seed.
        max_steps: Step cap per episode (tests), or None for the time limit.
        force: Re-record seeds the batch already holds.

    Returns:
        The episode file of every requested seed, in the order asked for —
        the ones just recorded and the ones that were already there.
    """
    out_dir = Path(out_dir)
    batch_dir = out_dir / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = batch_dir / "metadata.json"
    seeds = [int(s) for s in seeds]
    made: dict[int, Path] = {}
    todo: list[int] = []
    for seed in seeds:
        path = batch_dir / f"seed{seed}.npz"
        if path.exists() and not force:
            made[seed] = path
            print(f"seed {seed}: already recorded -> {path}", flush=True)
        else:
            todo.append(seed)
    if not todo and metadata_path.exists():
        return [made[s] for s in seeds if s in made]
    tools = EnvTools(task, env_tools)
    progress = RunProgress(out_dir, episodes=len(todo))
    try:
        if force or not metadata_path.exists():
            write_json(metadata_path, batch_metadata(tools, source=source or {}))
        with PolicyProcess() as proc:
            for seed in todo:
                rec, arrays = record_episode(
                    tools, policy_path, int(seed), max_steps, progress, proc
                )
                made[int(rec["seed"])] = write_record(batch_dir, rec, arrays)
                print(
                    f"seed {rec['seed']}: success={rec['success']} "
                    f"length={rec['length']} end={rec['termination']} "
                    f"-> {made[int(rec['seed'])]}",
                    flush=True,
                )
        if video and todo:
            record_seeds(
                tools,
                policy_path,
                todo,
                out_dir / "videos",
                max_steps=max_steps,
            )
        progress.clear()
    finally:
        tools.close()
    return [made[s] for s in seeds if s in made]


def finished_batch(out_dir: Path) -> Path | None:
    """Return the batch directory of an existing result, or None.

    A replay adds seeds to an existing batch, so a batch found here holds
    episodes to keep.

    Args:
        out_dir: The version's output directory.

    Returns:
        ``<out_dir>/batch`` when it already holds a finished batch.
    """
    batch_dir = Path(out_dir) / "batch"
    if (batch_dir / "metadata.json").exists() and any(batch_dir.glob("*.npz")):
        return batch_dir
    return None


def main(argv: list[str] | ReplayConfig | None = None):
    """Replay one policy version and write ``<cell>/replays/vNNN/``.

    An existing version directory is added to: seeds already in its batch
    are kept and only the missing ones are rolled out, so a second compare
    on a new seed extends the replay it finds instead of failing. ``--force``
    re-records every seed asked for.

    Args:
        argv: The parsed settings, or command line arguments (None for
            ``sys.argv[1:]``).

    Returns:
        The process exit status.
    """
    limit_blas_threads()
    args = parse_command(ReplayConfig, argv)
    try:
        cell = args.cell.expanduser().resolve()
        if not cell.is_dir():
            raise FileNotFoundError(f"no such cell directory: {cell}")
        seeds = parse_seeds(args.seeds)
        policy_path, out_dir, version = resolve_policy(
            cell, args.version, args.policy, "replays"
        )
        if version is not None:
            # A recorded version directory is the archive of what the agent
            # wrote; importing its helpers must not leave __pycache__ there.
            sys.dont_write_bytecode = True
        existing = finished_batch(out_dir)
        if existing is not None and not args.force:
            print(
                f"replay: adding to {existing}, which already holds "
                f"{len(list(existing.glob('*.npz')))} episode(s); seeds it has "
                "are kept (pass --force to re-record them)",
                flush=True,
            )
        task = args.task or cell_task(cell)
        if not task:
            raise ValueError(
                f"cannot tell which task {cell} ran: no task in sandbox_config.json "
                "or policies/index.json (pass --task)"
            )
        written = replay_seeds(
            task,
            policy_path,
            seeds,
            out_dir,
            read_env_tools(cell),
            source=batch_source(cell, policy_path, version),
            video=args.video,
            max_steps=args.max_steps,
            force=args.force,
        )
    except Exception as exc:
        print(f"replay failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"replay: {len(written)} episodes -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
