"""Coding-agent cells and replay-format batches, written by hand.

A cell is one task x one session of the coding-agent benchmark:
``policies/index.json`` with a ``vNNN`` directory per version, and
``eval/vNNN/``, ``replays/vNNN/`` or ``dev/vNNN_<stamp>/`` directories whose
``batch/`` holds ordinary replay-format episodes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from bigym.loco.demos.schema import BATCH_FORMAT
from bigym.loco.tasks import task_config

TASK = "reach_target_single"
EVAL_SEEDS = (620003, 620004)


def batch_metadata(task: str = TASK, protocol: bool = False) -> dict:
    """Replay-batch metadata; ``protocol`` writes the official env's blocks."""
    if protocol:
        config = task_config(task).override(camera_keys=("head",))
        return {
            "format": BATCH_FORMAT,
            "control_step_seconds": 0.02,
            **config.to_metadata(task),
        }
    return {
        "format": BATCH_FORMAT,
        "control_step_seconds": 0.02,
        "task": {
            "task_name": task,
            "robot_model": "g1_dex1",
            "camera_keys": ["head"],
            "camera_shape": [84, 84],
        },
        "lowerbody_policy": None,
    }


def write_batch(
    batch_dir: Path,
    seeds: Iterable[int],
    successes: Iterable[float] | None = None,
    *,
    task: str = TASK,
    nq: int = 7,
    protocol: bool = False,
) -> None:
    """Write one replay-format episode per seed with its stored outcome.

    Args:
        batch_dir: The ``batch`` directory; episodes already there are kept.
        seeds: One episode per seed.
        successes: Each episode's success; by default only the first succeeded.
        task: The task the metadata names.
        nq: Width of ``full_qpos``.
        protocol: Write the official environment's metadata blocks.
    """
    seeds = list(seeds)
    if successes is None:
        successes = [1.0] + [0.0] * (len(seeds) - 1)
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "metadata.json").write_text(json.dumps(batch_metadata(task, protocol)))
    rng = np.random.default_rng(0)
    for seed, success in zip(seeds, successes, strict=True):
        np.savez(
            batch_dir / f"seed{seed}.npz",
            full_qpos=rng.standard_normal((5, nq)).astype(np.float64),
            action=rng.standard_normal((5, 3)).astype(np.float32),
            reward=np.zeros((5, 1), np.float32),
            seed=np.int64(seed),
            success=np.float32(success),
            fell=np.float32(0.0),
            length=np.int64(1599 if success else 1700),
            termination=np.str_("success" if success else "timeout"),
        )


def make_cell(
    cell: Path,
    versions: int,
    *,
    evaluated: int | None = -1,
    eval_success: float = 0.02,
    eval_seeds: Iterable[int] = EVAL_SEEDS,
    policy_text: str | None = None,
) -> Path:
    """Write a cell named after its task, with ``versions`` policy versions.

    The last version is the submission (train success 0.5), every earlier one
    a write.

    Args:
        cell: The cell directory, ``<...>/<task>``.
        versions: How many policy versions the index lists.
        evaluated: The version with an evaluation (``-1``: the last one;
            ``None``: no evaluation).
        eval_success: The evaluation's success rate.
        eval_seeds: One evaluated episode per seed; the first one succeeded.
        policy_text: When given, every version's ``policy.py``.

    Returns:
        ``cell``.
    """
    entries = []
    for version in range(1, versions + 1):
        directory = cell / "policies" / f"v{version:03d}"
        directory.mkdir(parents=True)
        digest = f"{version:064x}"
        if policy_text is not None:
            (directory / "policy.py").write_text(policy_text)
            digest = hashlib.sha256(policy_text.encode()).hexdigest()
        last = version == versions
        entries.append(
            {
                "version": version,
                "dir": directory.name,
                "sha256": digest,
                "time": f"2026-01-02T0{version}:00:00+00:00",
                "trigger": "submission" if last else "write",
                "helpers": [],
                "train_run": None,
                "train_success": 0.5 if last else None,
                "train_episodes": None,
            }
        )
    (cell / "policies" / "index.json").write_text(
        json.dumps({"task": cell.name, "versions": entries})
    )
    if evaluated is None:
        return cell
    if evaluated == -1:
        evaluated = versions
    eval_dir = cell / "eval" / f"v{evaluated:03d}"
    eval_dir.mkdir(parents=True)
    (eval_dir / "summary.json").write_text(
        json.dumps({"version": evaluated, "success_rate": eval_success})
    )
    eval_seeds = list(eval_seeds)
    rows = ["frame,episode,seed,success,length,reward,termination,fell"]
    for episode, seed in enumerate(eval_seeds):
        if episode == 0:
            rows.append(f"0,0,{seed},1,1599,1.0,success,0")
        else:
            rows.append(f"0,{episode},{seed},0,1700,0.0,timeout,0")
    (eval_dir / "episodes.csv").write_text("\n".join(rows) + "\n")
    write_batch(eval_dir / "batch", eval_seeds, task=cell.name)
    return cell
