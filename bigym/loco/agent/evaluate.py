"""Evaluation of a submitted policy.py on the hidden protocol seeds.

    bigym-agent evaluate <cell> --version 7 [--episodes 100] [--jobs 3]
    bigym-agent evaluate --task reach_target_single \
        --policy <sandbox>/policy.py --out <dir> [--episodes 100] \
        [--seed-offset 0] [--repeat 2]

The first form scores one recorded policy version of a run directory (a
"cell": one task x one session) in the environment the session ran against,
recorded whole in ``<cell>/sandbox_config.json``. It writes
``<cell>/eval/vNNN/`` and refuses to replace a finished evaluation unless
``--force`` is given. ``episodes.csv``
grows a row per finished episode so a reader can follow the progress, and
every episode is also written to ``batch/`` as a replay batch the demo viewer
opens. The second form scores any policy file into any directory.

Runs the real environment in this process with the protocol reset
(deterministic controller state) on seeds ``EVAL_SEED_BASE + offset + i``
(``bigym.loco.eval.protocol``), and writes episodes.csv
(the same columns as the trainers' evaluation log) plus summary.json with the
substrate fingerprint. ``--repeat 2`` runs the block twice and reports whether
the per-episode records are identical.

The policy itself never runs in this process: every episode, the spot-check
and the videos drive it in a separate interpreter
(:class:`~bigym.loco.agent.policy_process.PolicyProcess`) that holds no
simulator, so it can neither read nor rewrite the environment it is scored on.

A policy that imports the simulator (``bigym``, ``mujoco``, ...) is rejected:
found by a scan of its directory before any episode, or by the policy process
refusing the import, it scores 0 on every episode of the block and
summary.json names the import in ``rejected``.

``bigym-agent record`` (:func:`main_record`) reuses the same machinery to
record given seeds to mp4 without scoring a block.
"""

from __future__ import annotations

import csv
import dataclasses
import functools
import hashlib
import json
import multiprocessing as mp
import queue as queue_module
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bigym.cli import usage_error
from bigym.loco.eval.protocol import EVAL_SEED_BASE, is_success, protocol_violations

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
    EvaluateConfig,
    RecordConfig,
    limit_blas_threads,
    parse_command,
)
from .envtools import EnvTools, LocalEnv
from .episode import Tools, run_episode
from .policy_process import (
    ForbiddenImport,
    PolicyProcess,
    forbidden_imports,
)
from .snapshots import cell_task, now_stamp, read_env_tools, resolve_policy

COLUMNS = (
    "frame",
    "episode",
    "seed",
    "success",
    "length",
    "reward",
    "termination",
    "fell",
)


def record_seeds(
    tools: EnvTools,
    policy_path,
    seeds,
    out_dir: Path,
    prefix: str = "",
    views=None,
    suffixes=None,
    max_steps: int | None = None,
):
    """Record the policy on the given seeds to mp4.

    The recording views are ours only: nothing here is policy-facing.

    Args:
        tools: The wrapper holding the environment.
        policy_path: Path of the policy file.
        seeds: Seeds to record.
        out_dir: Directory for the mp4 files.
        prefix: File name prefix.
        views: Recording view pair, or None for the per-task default.
        suffixes: Per-seed file name suffixes.
        max_steps: Step cap per episode (smoke runs), or None for the
            episode's own time limit.

    Returns:
        The list of written paths.
    """
    env = LocalEnv(tools)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = []
    with PolicyProcess() as proc:
        for k, seed in enumerate(seeds):
            suffix = suffixes[k] if suffixes else ""
            out = out_dir / f"{prefix}seed{seed}{suffix}.mp4"
            policy = proc.load(policy_path)
            obs = env.reset(
                int(seed)
            )  # reset first so recording starts from the settled state
            tools.start_recording(out, views=views)
            t = Tools(env)
            policy.reset(obs, t)
            length, done = 0, False
            while not done and (max_steps is None or length < max_steps):
                obs, _, done, _ = env.step(policy.act(obs, t))
                length += 1
            tools.stop_recording()
            success = int(is_success(tools.outer))
            print(
                f"seed {seed}: success={success} length={length} -> {out}", flush=True
            )
            outs.append(out)
    return outs


def pick_video_seeds(episode_records):
    """Pick one success and one failure from the hidden-seed records.

    Two of the same kind are returned when the other does not exist (a 0%
    policy gets two failure videos, a 100% policy two success videos).

    Args:
        episode_records: Per-episode records.

    Returns:
        ``(seeds, suffixes)``.
    """
    ok = [r["seed"] for r in episode_records if r["success"]]
    bad = [r["seed"] for r in episode_records if not r["success"]]
    if ok and bad:
        return [ok[0], bad[0]], ["_success", "_fail"]
    if ok:
        return ok[:2], ["_success"] * len(ok[:2])
    return bad[:2], ["_fail"] * len(bad[:2])


@dataclass(frozen=True)
class Block:
    """A block of hidden-seed episodes: the policy, its environment, the seeds."""

    task: str
    """Task name."""
    policy: Path
    """The policy file."""
    env_tools: EnvToolsConfig
    """The environment the policy is scored in."""
    episodes: int
    """Number of episodes; episode ``i`` runs on :meth:`seed` ``(i)``."""
    seed_offset: int = 0
    """Offset added to the seed base."""
    seed_base: int = EVAL_SEED_BASE
    """First seed of the block before the offset."""

    def seed(self, episode: int) -> int:
        """Return the seed of one episode of the block.

        Args:
            episode: Episode index within the block.

        Returns:
            ``seed_base + seed_offset + episode``.
        """
        return self.seed_base + self.seed_offset + episode


def run_indices(
    block: Block,
    indices: list[int],
    tag: str = "",
    *,
    verbose: bool = True,
    batch_dir: Path | None = None,
    source: dict | None = None,
    on_episode=None,
    queue=None,
):
    """Run the given episodes of a block sequentially in this process.

    With ``batch_dir`` every episode is also written there as a replay-batch
    ``.npz`` (full state per frame, raw actions, reward), and each finished
    record is handed to ``on_episode`` or put on ``queue`` so the caller can
    report progress while the block runs.

    Args:
        block: The block the episodes belong to.
        indices: Episode indices within the block.
        tag: Label of this process in the per-episode lines.
        verbose: Print a line per episode.
        batch_dir: Write every episode there as a replay batch.
        source: The ``source`` block of that batch's metadata.
        on_episode: Called with each finished record.
        queue: Put each finished record there instead (a worker process).

    Returns:
        ``(episode_records, fingerprint, violations)``.
    """
    tools = EnvTools(block.task, block.env_tools)
    env = LocalEnv(tools)
    if batch_dir is not None:
        env = RecordingEnv(tools)
        batch_dir = Path(batch_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)
        if not (batch_dir / "metadata.json").exists():
            write_json(
                batch_dir / "metadata.json", batch_metadata(tools, source=source or {})
            )
    episode_records = []
    t0 = time.time()
    cmds = []
    orig_slew = tools.apply_slew

    def record_slew(raw):  # smoothness: record the arm command actually sent
        out = orig_slew(raw)
        arm = np.asarray(out, dtype=np.float32)[
            tools.n_base : tools.n_base + len(tools.limb_names)
        ]
        cmds.append(arm.copy())
        return out

    tools.apply_slew = record_slew  # ty: ignore[invalid-assignment]
    with PolicyProcess() as proc:
        for n, i in enumerate(indices):
            seed = block.seed(i)
            policy = proc.load(block.policy)
            cmds.clear()
            record = run_episode(env, policy, seed)
            record["success"] = int(is_success(tools.outer))
            record["termination"] = tools.outer.episode_termination()
            record["fell"] = bool(tools.outer.episode_fell())
            record["episode"] = i
            if len(cmds) > 2:
                v = np.diff(np.array(cmds), axis=0)
                record["reversals_per_s"] = float(
                    ((v[1:] * v[:-1]) < -1e-8).sum() / (len(v) / 50.0)
                )
                record["mean_abs_dq"] = float(np.abs(v).mean())
                record["p95_max_dq"] = float(np.percentile(np.abs(v).max(1), 95))
            episode_records.append(record)
            if batch_dir is not None:
                assert isinstance(env, RecordingEnv)
                write_record(batch_dir, record, env.recorder.arrays())
            if queue is not None:
                queue.put(record)
            elif on_episode is not None:
                on_episode(record)
            if verbose:
                sr = sum(r["success"] for r in episode_records) / len(episode_records)
                print(
                    f"[eval{tag}] ep {n + 1}/{len(indices)} seed={seed} "
                    f"success={record['success']} len={record['length']} "
                    f"end={record['termination']} running_sr={sr:.2%}  "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
    fingerprint = dict(tools.env.substrate_fingerprint())
    violations = protocol_violations(tools.env)
    tools.close()
    return episode_records, fingerprint, violations


def _drain(progress, on_episode):
    """Hand every record a worker has finished so far to ``on_episode``."""
    if progress is None or on_episode is None:
        return
    while True:
        try:
            on_episode(progress.get_nowait())
        except queue_module.Empty:
            return


def run_block(
    block: Block,
    *,
    verbose: bool = True,
    jobs: int = 1,
    batch_dir: Path | None = None,
    source: dict | None = None,
    on_episode=None,
):
    """Run one evaluation block, optionally split over ``jobs`` processes.

    Episodes are independent (the deterministic reset restores the controller
    state), so the merged records are identical to a sequential run.

    Args:
        block: The block to run.
        verbose: Print a line per episode.
        jobs: Processes to split the block over (~2 GB RAM each).
        batch_dir: Write every episode there as a replay batch.
        source: The ``source`` block of that batch's metadata.
        on_episode: Called with each finished record, in the order the
            episodes finish (the caller's progress report).

    Returns:
        ``(episode_records, fingerprint, violations)``.

    Raises:
        RuntimeError: The workers disagree on the substrate fingerprint.
    """
    jobs = max(1, min(int(jobs), block.episodes))
    if jobs == 1:
        return run_indices(
            block,
            list(range(block.episodes)),
            verbose=verbose,
            batch_dir=batch_dir,
            source=source,
            on_episode=on_episode,
        )
    chunks = [list(range(block.episodes))[k::jobs] for k in range(jobs)]
    ctx = mp.get_context("fork")
    # A manager queue (not a plain one) because pool arguments are pickled.
    manager = ctx.Manager() if on_episode is not None else None
    progress = manager.Queue() if manager is not None else None
    work = functools.partial(
        run_indices,
        block,
        verbose=verbose,
        batch_dir=batch_dir,
        source=source,
        queue=progress,
    )
    try:
        with ctx.Pool(jobs) as pool:
            pending = pool.starmap_async(
                work, [(chunk, f" w{k}") for k, chunk in enumerate(chunks)]
            )
            while not pending.ready():
                _drain(progress, on_episode)
                pending.wait(0.2)
            parts = pending.get()
        _drain(progress, on_episode)
    finally:
        if manager is not None:
            manager.shutdown()
    episode_records = sorted(
        (r for part in parts for r in part[0]), key=lambda r: r["episode"]
    )
    fingerprints = [part[1] for part in parts]
    if any(fingerprint != fingerprints[0] for fingerprint in fingerprints[1:]):
        raise RuntimeError("substrate fingerprint differs between evaluation workers")
    return episode_records, fingerprints[0], parts[0][2]


def rejected_records(block: Block) -> list[dict]:
    """Return the records of a rejected policy: every episode failed, none run.

    Args:
        block: The block the policy was to be scored on.

    Returns:
        One record per episode, with success 0 and termination ``rejected``.
    """
    return [
        {
            "episode": i,
            "seed": block.seed(i),
            "success": 0,
            "length": 0,
            "reward": 0.0,
            "termination": "rejected",
            "fell": False,
        }
        for i in range(block.episodes)
    ]


def csv_row(record, step_label) -> list:
    """Return one ``episodes.csv`` row for a per-episode record.

    Args:
        record: The per-episode record.
        step_label: Value of the ``frame`` column (the trainers' step label).

    Returns:
        The row, in :data:`COLUMNS` order.
    """
    return [
        step_label,
        record["episode"],
        record["seed"],
        record["success"],
        record["length"],
        f"{record['reward']:.6f}",
        record["termination"],
        int(record["fell"]),
    ]


def write_csv(path: Path, episode_records, step_label):
    """Write the per-episode records as CSV.

    Args:
        path: Destination file.
        episode_records: Per-episode records.
        step_label: Value of the ``frame`` column (the trainers' step label).
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in episode_records:
            w.writerow(csv_row(r, step_label))


class IncrementalCsv:
    """``episodes.csv`` written a row at a time, as the episodes finish.

    A reader (the viewer's job panel) counts the rows for progress, so the
    file is flushed after every row. The rows arrive in completion order;
    the caller rewrites the file in episode order when the block is done.
    """

    def __init__(self, path: Path, step_label):
        """Open the file and write its header.

        Args:
            path: Destination file.
            step_label: Value of the ``frame`` column.
        """
        self.path = Path(path)
        self.step_label = step_label
        self.rows = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "w", newline="")
        self.writer = csv.writer(self.handle)
        self.writer.writerow(COLUMNS)
        self.handle.flush()

    def append(self, record) -> None:
        """Append one finished episode and flush.

        Args:
            record: The per-episode record.
        """
        self.writer.writerow(csv_row(record, self.step_label))
        self.handle.flush()
        self.rows += 1

    def close(self) -> None:
        """Close the file."""
        self.handle.close()


def prepare_cell(args: EvaluateConfig):
    """Resolve a cell run into the file-mode arguments.

    Args:
        args: The parsed settings, updated in place.

    Returns:
        ``(cell, version, source)``: the cell directory, the policy version
        (None for ``--policy``) and the batch metadata's source block.

    Raises:
        FileNotFoundError: The cell does not exist.
        FileExistsError: That version is already evaluated and no --force.
        ValueError: The cell does not say which task it ran.
    """
    assert args.cell is not None
    cell = args.cell.expanduser().resolve()
    if not cell.is_dir():
        raise FileNotFoundError(f"no such cell directory: {cell}")
    policy_path, out_dir, version = resolve_policy(
        cell, args.version, args.policy, "eval"
    )
    if version is not None:
        # A recorded version directory is the archive of what the agent wrote;
        # importing its helpers must not leave __pycache__ there.
        sys.dont_write_bytecode = True
    summary = out_dir / "summary.json"
    if summary.exists() and not args.force:
        raise FileExistsError(
            f"already evaluated: {summary}; pass --force to replace it"
        )
    task = args.task or cell_task(cell)
    if not task:
        raise ValueError(
            f"cannot tell which task {cell} ran: no task in sandbox_config.json "
            "or policies/index.json (pass --task)"
        )
    args.task = task
    args.policy = policy_path
    args.out = out_dir
    args.env_tools = read_env_tools(cell)
    if args.video_dir is None and args.video:
        args.video_dir = out_dir / "videos"
    return cell, version, batch_source(cell, policy_path, version)


def episode_hook(live, watch):
    """Combine the two per-episode reporters into one ``on_episode``.

    Args:
        live: The :class:`IncrementalCsv` of the scored block, or None.
        watch: The ``batch.RunProgress`` of a cell run, or None.

    Returns:
        A callable for ``run_block(on_episode=...)``, or None when neither
        reporter is there (the block then runs with no progress queue, just
        as it did before).
    """
    if live is None and watch is None:
        return None

    def on_episode(record) -> None:
        """Record one finished episode: a CSV row and a progress update."""
        if live is not None:
            live.append(record)
        if watch is not None:
            watch.finish(record.get("seed"), record.get("length"))

    return on_episode


def score_blocks(
    args: EvaluateConfig,
    block: Block,
    out: Path,
    batch_dir: Path | None,
    source: dict | None,
    watch,
):
    """Run the block ``--repeat`` times and write its ``episodes*.csv``.

    The first block is the scored one: it feeds ``episodes.csv`` row by row
    (a reader follows the progress), ``watch`` and the replay batch. A policy
    that imports the simulator is not run at all, or stops being run once the
    policy process refuses the import, and scores 0 on every episode.

    Args:
        args: The parsed settings.
        block: The block to score.
        out: The output directory.
        batch_dir: Where the scored block's episodes go as a replay batch.
        source: The ``source`` block of that batch's metadata.
        watch: The cell run's ``batch.RunProgress``, or None.

    Returns:
        ``(blocks, fingerprint, violations, rejected)``: the records of each
        run of the block, and why the policy was rejected (None when it ran).
    """
    hits = forbidden_imports(block.policy)
    rejected = "; ".join(hits) or None
    blocks, fingerprint, violations = [], None, None
    for k in range(args.repeat if rejected is None else 0):
        live = (
            IncrementalCsv(out / "episodes.csv", args.label)
            if watch is not None and k == 0
            else None
        )
        try:
            episode_records, fingerprint, violations = run_block(
                block,
                jobs=args.jobs,
                batch_dir=batch_dir if k == 0 else None,
                source=source,
                on_episode=episode_hook(live, watch if k == 0 else None),
            )
        except ForbiddenImport as exc:
            rejected = str(exc)
            break
        finally:
            if live is not None:
                live.close()
        write_csv(
            out / ("episodes.csv" if k == 0 else f"episodes_repeat{k}.csv"),
            episode_records,
            args.label,
        )
        blocks.append(episode_records)
    if rejected is not None:
        print(f"evaluate: policy rejected, scored 0: {rejected}", file=sys.stderr)
        blocks = [rejected_records(block)]
        write_csv(out / "episodes.csv", blocks[0], args.label)
        if batch_dir is not None:
            # Episodes run before the refusal are not part of the record.
            shutil.rmtree(batch_dir, ignore_errors=True)
    return blocks, fingerprint, violations, rejected


def write_smoothness(path: Path, episode_records) -> dict:
    """Write ``smoothness.csv`` and return the block's mean of each measure.

    Args:
        path: Destination file.
        episode_records: The scored block's records.

    Returns:
        ``{"reversals_per_s", "mean_abs_dq", "p95_max_dq"}`` means.
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("episode", "seed", "reversals_per_s", "mean_abs_dq", "p95_max_dq"))
        for r in episode_records:
            w.writerow(
                [
                    r["episode"],
                    r["seed"],
                    f"{r.get('reversals_per_s', float('nan')):.3f}",
                    f"{r.get('mean_abs_dq', float('nan')):.5f}",
                    f"{r.get('p95_max_dq', float('nan')):.5f}",
                ]
            )
    return {
        k: float(np.nanmean([r.get(k, np.nan) for r in episode_records]))
        for k in ("reversals_per_s", "mean_abs_dq", "p95_max_dq")
    }


def spot_check(args: EvaluateConfig, block: Block, scored, out: Path) -> dict:
    """Rerun the first ``--spot-check`` episodes and compare them.

    Args:
        args: The parsed settings.
        block: The scored block.
        scored: Its records.
        out: The output directory.

    Returns:
        The ``spot_check`` block of the summary.
    """
    n = min(args.spot_check, block.episodes)
    episode_records, _, _ = run_block(
        dataclasses.replace(block, episodes=n),
        verbose=False,
        jobs=min(args.jobs, n),
    )
    write_csv(out / "episodes_spotcheck.csv", episode_records, args.label)
    first = scored[:n]
    return {
        "episodes": n,
        "success_identical": all(
            a["success"] == b["success"]
            for a, b in zip(first, episode_records, strict=True)
        ),
        "length_diffs": sum(
            a["length"] != b["length"]
            for a, b in zip(first, episode_records, strict=True)
        ),
        "max_abs_length_diff": max(
            (
                abs(a["length"] - b["length"])
                for a, b in zip(first, episode_records, strict=True)
            ),
            default=0,
        ),
        "bit_identical": all(
            (a["success"], a["length"], round(a["reward"], 6))
            == (b["success"], b["length"], round(b["reward"], 6))
            for a, b in zip(first, episode_records, strict=True)
        ),
    }


def repeat_identical(blocks) -> bool | None:
    """Return whether every run of the block gave the same records.

    Args:
        blocks: The records of each run.

    Returns:
        None for a single run.
    """
    if len(blocks) < 2:
        return None

    def key(r):
        return (r["seed"], r["success"], r["length"], round(r["reward"], 6))

    return all(
        [key(a) for a in blocks[0]] == [key(b) for b in blk] for blk in blocks[1:]
    )


def evaluation_summary(args: EvaluateConfig, block: Block, blocks, **results) -> dict:
    """Return ``summary.json`` of a file-mode evaluation.

    Args:
        args: The parsed settings.
        block: The scored block.
        blocks: The records of each run of the block.
        **results: ``spot_check``, ``smoothness``, ``substrate_fingerprint``,
            ``protocol_violations`` and ``rejected``.

    Returns:
        The summary.
    """
    env_tools = block.env_tools
    scored = blocks[0]
    return {
        "task": block.task,
        "policy": str(block.policy.resolve()),
        "label": args.label,
        "tier": env_tools.tier,
        "episodes": block.episodes,
        "seed_offset": block.seed_offset,
        "seed_base": block.seed_base,
        "success_rate": float(np.mean([r["success"] for r in scored])),
        "mean_length": float(np.mean([r["length"] for r in scored])),
        "fell_rate": float(np.mean([r["fell"] for r in scored])),
        "terminations": {
            t: sum(r["termination"] == t for r in scored)
            for t in {r["termination"] for r in scored}
        },
        "repeat_identical": repeat_identical(blocks),
        "spot_check": results["spot_check"],
        "smoothness": results["smoothness"],
        "onboard_only": not env_tools.allow_external_cameras,
        "substrate_fingerprint": results["substrate_fingerprint"],
        "protocol_violations": results["protocol_violations"],
        "interface": env_tools.interface,
        "hand_pos": env_tools.interface == "tools" and env_tools.hand_pos,
        "ik": env_tools.interface == "tools" and env_tools.ik,
        "calibration": env_tools.interface == "tools",
        "image_cap": env_tools.image_cap,
        "external_view": False,
        "slew": (
            {
                "joint_rad_per_s": env_tools.joint_vmax,
                "base_per_s": 0.7,
                "joint_accel_rad_per_step2": env_tools.accel,
                "lowpass_alpha": env_tools.lowpass,
            }
            if env_tools.slew
            else None
        ),
        "eval_jobs": args.jobs,
        "pitch_layout": "task-config" if env_tools.pitch else "forced-20dim",
        "policy_sha256": hashlib.sha256(block.policy.read_bytes()).hexdigest(),
        "rejected": results["rejected"],
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def record_videos(
    args: EvaluateConfig, block: Block, scored, video_dir: Path
) -> list[Path]:
    """Record one success and one failure of the scored block to mp4.

    Args:
        args: The parsed settings.
        block: The scored block.
        scored: Its records.
        video_dir: Where the videos go.

    Returns:
        The written paths.
    """
    seeds, suffixes = pick_video_seeds(scored)
    tools = EnvTools(block.task, block.env_tools)
    try:
        return record_seeds(
            tools,
            block.policy,
            seeds,
            video_dir,
            args.video_prefix,
            views=args.video_views.split(",") if args.video_views else None,
            suffixes=suffixes,
        )
    finally:
        tools.close()


def write_summary(path: Path, summary: dict) -> None:
    """Write ``summary.json``.

    Args:
        path: Destination file.
        summary: The summary.
    """
    path.write_text(json.dumps(summary, indent=1, default=str))


def run_evaluation(args: EvaluateConfig, cell=None, version=None, source=None):
    """Score a policy on a block of hidden seeds and write the report.

    Args:
        args: The parsed arguments (file mode; a cell has been resolved into
            them by :func:`prepare_cell`).
        cell: The cell directory, or None in file mode.
        version: The policy version, when one was given.
        source: The batch metadata's source block, in cell mode.
    """
    assert args.task is not None and args.policy is not None and args.out is not None
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    block = Block(
        args.task,
        args.policy,
        args.env_tools,
        args.episodes,
        args.seed_offset,
        args.seed_base,
    )
    batch_dir = (out / "batch") if cell is not None else None
    # The same ``.progress.json`` a replay writes, in the version's eval
    # directory: the episodes are split over worker processes, so it counts
    # finished episodes rather than steps (see batch.RunProgress).
    watch = None
    if cell is not None:
        watch = RunProgress(out, episodes=int(args.episodes))
        watch.write()
    blocks, fingerprint, violations, rejected = score_blocks(
        args, block, out, batch_dir, source, watch
    )
    smooth = write_smoothness(out / "smoothness.csv", blocks[0])
    spot = None
    if args.spot_check > 0 and args.repeat == 1 and rejected is None:
        spot = spot_check(args, block, blocks[0], out)
    summary = evaluation_summary(
        args,
        block,
        blocks,
        spot_check=spot,
        smoothness=smooth,
        substrate_fingerprint=fingerprint,
        protocol_violations=violations,
        rejected=rejected,
    )
    if cell is not None:
        # The names the run layout's readers use, next to the file-mode fields.
        summary.update(
            {
                "cell": str(cell),
                "version": version,
                "fingerprint": fingerprint,
                "spotcheck_identical": (
                    None if spot is None else bool(spot["success_identical"])
                ),
                "batch": None if rejected is not None else str(batch_dir),
                "time": now_stamp(),
            }
        )
    write_summary(out / "summary.json", summary)
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ("substrate_fingerprint", "fingerprint")
            },
            indent=1,
            default=str,
        )
    )
    if args.video_dir is not None and rejected is None:
        videos = record_videos(args, block, blocks[0], args.video_dir)
        summary["videos"] = [str(path) for path in videos]
        write_summary(out / "summary.json", summary)
    if watch is not None:
        watch.clear()


def main(argv: list[str] | EvaluateConfig | None = None):
    """Score a policy (of a cell, or any file) and write the report.

    Args:
        argv: The parsed settings, or command line arguments (None for
            ``sys.argv[1:]``).

    Returns:
        The process exit status.
    """
    limit_blas_threads()
    args = parse_command(EvaluateConfig, argv)
    if args.cell is None and not (args.task and args.policy and args.out):
        usage_error(
            "without a cell, --task, --policy and --out are all required",
            prog="bigym-agent evaluate",
        )
    if args.cell is not None and args.env_tools != EnvToolsConfig():
        usage_error(
            "a cell is scored in the environment its session ran against "
            "(env_tools in its sandbox_config.json); the environment flags "
            "apply without a cell",
            prog="bigym-agent evaluate",
        )
    cell = version = source = None
    try:
        if args.cell is not None:
            cell, version, source = prepare_cell(args)
        run_evaluation(args, cell, version, source)
    except Exception as exc:
        print(f"evaluate failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


def main_record(argv: list[str] | RecordConfig | None = None):
    """Record a policy on the given seeds to mp4.

    Args:
        argv: The parsed settings, or command line arguments (None for
            ``sys.argv[1:]``).
    """
    limit_blas_threads()
    args = parse_command(RecordConfig, argv)
    tools = EnvTools(args.task, args.env_tools)
    try:
        record_seeds(
            tools,
            args.policy,
            [int(s) for s in args.seeds.split(",")],
            args.out,
            args.prefix,
            views=args.views.split(",") if args.views else None,
        )
    finally:
        tools.close()


if __name__ == "__main__":
    main()
