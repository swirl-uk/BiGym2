"""Trim the terminal success hold of a replay-format demo batch.

An episode succeeds once the task predicate has held for
``success_hold_seconds``; the recorded episode ends on that frame with the
single terminal reward (1) and discount (0). The collector records with a
longer hold than training and evaluation use, so a raw batch has to be
re-timed before it matches the env it trains: ``trim`` drops the surplus
terminal hold and moves the terminal marker to the new final frame (raw 3 s
hold -> 1 s training view). It writes a NEW directory and never modifies the
source batch. Usage::

    python -m bigym.loco.demos.success_hold trim --demo-dir <batch>
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import tyro

# Episode-level arrays: their leading axis is not time.
EPISODE_LEVEL_PREFIXES = ("init_", "lb_state.")
EPISODE_LEVEL_KEYS = frozenset({"seed", "pre_engage_steps"})


@dataclass(frozen=True)
class EpisodePlan:
    """One episode of a batch: source file, output name and frame counts."""

    source: Path
    output_name: str
    old_frames: int
    new_frames: int


@dataclass(frozen=True)
class BatchPlan:
    """A validated re-timing of a whole batch."""

    source: Path
    output: Path | None
    metadata: dict
    source_hold_seconds: float
    target_hold_seconds: float
    control_step_seconds: float
    # Frames removed per episode.
    frame_delta: int
    episodes: tuple[EpisodePlan, ...]


def is_per_step(key: str, value: np.ndarray, frames: int) -> bool:
    """Whether array ``key`` has one entry per frame of a ``frames``-long episode."""
    if key in EPISODE_LEVEL_KEYS or key.startswith(EPISODE_LEVEL_PREFIXES):
        return False
    return value.ndim >= 1 and value.shape[0] == frames


def control_step_seconds(metadata: dict) -> float:
    """Seconds per recorded frame, from the batch metadata."""
    step_seconds = metadata.get("control_step_seconds")
    if step_seconds is None:
        downsample = (metadata.get("task") or {}).get("demo_down_sample_rate")
        if downsample is None:
            raise ValueError(
                "metadata.json lacks control_step_seconds and "
                "task.demo_down_sample_rate"
            )
        step_seconds = float(downsample) / 500.0
    step_seconds = float(step_seconds)
    if step_seconds <= 0.0:
        raise ValueError(f"control_step_seconds must be positive, got {step_seconds}")
    return step_seconds


def recorded_hold_seconds(metadata: dict) -> float:
    """The hold the batch's terminal marker reflects."""
    task = metadata.get("task") or {}
    if "success_hold_seconds" not in task:
        raise ValueError(
            "metadata.json lacks task.success_hold_seconds; pass the source "
            "hold explicitly"
        )
    return float(task["success_hold_seconds"])


def training_hold_seconds(metadata: dict) -> float:
    """The hold training and evaluation use for this batch (default 1.0)."""
    task = metadata.get("task") or {}
    return float(task.get("training_success_hold_seconds", 1.0))


def default_output(source: Path, target_hold_seconds: float) -> Path:
    """``<source>_hold<target>s`` next to the source batch."""
    label = f"{target_hold_seconds:g}".replace(".", "p")
    source = Path(source)
    return source.with_name(f"{source.name}_hold{label}s")


def _frames_for(seconds: float, step_seconds: float, what: str) -> int:
    frames_float = seconds / step_seconds
    frames = int(round(frames_float))
    if not np.isclose(frames_float, frames, atol=1e-6, rtol=0.0):
        raise ValueError(
            f"{what} {seconds:g}s is not an integer number of {step_seconds:g}s "
            f"control frames ({frames_float:g})"
        )
    if frames <= 0:
        raise ValueError(f"{what} resolves to zero frames")
    return frames


def _terminal_vector(episode: Any, key: str, frames: int, source: Path) -> np.ndarray:
    if key not in episode:
        raise ValueError(f"{source.name}: missing required field '{key}'")
    value = np.asarray(episode[key])
    if value.shape[0] != frames or value.size != frames:
        raise ValueError(
            f"{source.name}: {key} must contain one scalar per frame, "
            f"got shape {value.shape} for {frames} frames"
        )
    return value.reshape(-1)


def _check_terminal_marker(episode: Any, frames: int, source: Path) -> None:
    reward = _terminal_vector(episode, "reward", frames, source)
    discount = _terminal_vector(episode, "discount", frames, source)
    if not (np.isclose(reward[-1], 1.0) and np.allclose(reward[:-1], 0.0)):
        raise ValueError(f"{source.name}: expected one reward=1 on the final frame")
    if not (np.isclose(discount[-1], 0.0) and np.allclose(discount[:-1], 1.0)):
        raise ValueError(f"{source.name}: expected discount=1 then final discount=0")


def _output_episode_name(source: Path, old_frames: int, new_frames: int) -> str:
    parts = source.stem.split("_")
    if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
        raise ValueError(
            f"{source.name}: expected replay filename <timestamp>_<index>_<length>.npz"
        )
    if int(parts[2]) != old_frames - 1:
        raise ValueError(
            f"{source.name}: filename says {parts[2]} transitions but the "
            f"episode contains {old_frames - 1}"
        )
    return f"{parts[0]}_{parts[1]}_{new_frames - 1}.npz"


def _load_metadata(source: Path) -> dict:
    if not source.is_dir():
        raise FileNotFoundError(f"demo directory does not exist: {source}")
    path = source / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing metadata.json under {source}")
    return json.loads(path.read_text(encoding="utf-8"))


def _plan(
    demo_dir: Path,
    out_dir: Path | None,
    *,
    source_hold: float,
    target_hold: float,
    frame_delta: int,
    metadata: dict,
    step_seconds: float,
) -> BatchPlan:
    source = Path(demo_dir).expanduser().resolve()
    destination = None if out_dir is None else Path(out_dir).expanduser().resolve()
    if destination == source:
        raise ValueError("output directory must differ from the source batch")
    if destination is not None and destination.exists():
        raise FileExistsError(f"refusing to overwrite existing {destination}")
    paths = sorted(source.glob("*.npz"))
    if not paths:
        raise ValueError(f"no replay-format .npz episodes found under {source}")
    episodes: list[EpisodePlan] = []
    names: set[str] = set()
    for path in paths:
        with np.load(path) as episode:
            if "action" not in episode:
                raise ValueError(f"{path.name}: missing required field 'action'")
            action = np.asarray(episode["action"])
            if action.ndim < 1:
                raise ValueError(f"{path.name}: action has no time axis")
            old_frames = int(action.shape[0])
            new_frames = old_frames - frame_delta
            if new_frames < 2:
                raise ValueError(
                    f"{path.name}: {old_frames} frames cannot lose {frame_delta}; "
                    "at least two frames must remain"
                )
            _check_terminal_marker(episode, old_frames, path)
        name = _output_episode_name(path, old_frames, new_frames)
        if name in names:
            raise ValueError(f"multiple source episodes map to {name}")
        names.add(name)
        episodes.append(EpisodePlan(path, name, old_frames, new_frames))
    return BatchPlan(
        source=source,
        output=destination,
        metadata=metadata,
        source_hold_seconds=source_hold,
        target_hold_seconds=target_hold,
        control_step_seconds=step_seconds,
        frame_delta=frame_delta,
        episodes=tuple(episodes),
    )


def plan_trim(
    demo_dir: Path,
    out_dir: Path | None,
    *,
    target_hold_seconds: float | None = None,
    source_hold_seconds: float | None = None,
) -> BatchPlan:
    """Validate a trim of every episode; nothing is written.

    Args:
        demo_dir: The raw batch.
        out_dir: Where the training view goes (None for a dry run).
        target_hold_seconds: The training hold; default the batch's
            ``task.training_success_hold_seconds``, else 1.0.
        source_hold_seconds: The recorded hold, for metadata that lacks it.

    Returns:
        The plan :func:`write_batch` executes.
    """
    source = Path(demo_dir).expanduser().resolve()
    metadata = _load_metadata(source)
    source_hold = (
        recorded_hold_seconds(metadata)
        if source_hold_seconds is None
        else float(source_hold_seconds)
    )
    target_hold = (
        training_hold_seconds(metadata)
        if target_hold_seconds is None
        else float(target_hold_seconds)
    )
    if target_hold < 0.0:
        raise ValueError(f"target hold must be non-negative, got {target_hold}")
    if source_hold <= target_hold:
        raise ValueError(
            f"source terminal hold is {source_hold:g}s; it must be longer than "
            f"the {target_hold:g}s target to trim"
        )
    step_seconds = control_step_seconds(metadata)
    frames = _frames_for(source_hold - target_hold, step_seconds, "hold difference")
    return _plan(
        source,
        out_dir,
        source_hold=source_hold,
        target_hold=target_hold,
        frame_delta=frames,
        metadata=metadata,
        step_seconds=step_seconds,
    )


def _retime_episode(plan: BatchPlan, episode: EpisodePlan, output: Path) -> None:
    with np.load(episode.source) as data:
        values = {key: np.asarray(data[key]) for key in data.files}
    out: dict[str, np.ndarray] = {}
    for key, value in values.items():
        if not is_per_step(key, value, episode.old_frames):
            out[key] = value
            continue
        result = np.array(value[: episode.new_frames], copy=True)
        if key == "reward":
            result.fill(0.0)
            result.reshape(-1)[-1] = 1.0
        elif key == "discount":
            result.fill(1.0)
            result.reshape(-1)[-1] = 0.0
        out[key] = result
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **out)  # ty: ignore[invalid-argument-type]
    tmp.replace(output)


def retimed_metadata(plan: BatchPlan) -> dict:
    """The source metadata with the hold fields set to the new terminal timing."""
    metadata = dict(plan.metadata)
    target = plan.target_hold_seconds
    task = dict(metadata.get("task") or {})
    task["success_hold_seconds"] = target
    task["training_success_hold_seconds"] = target
    if "env_config" in metadata:
        metadata["env_config"] = dict(
            metadata["env_config"], success_hold_seconds=target
        )
    task["collect_success_hold_seconds"] = plan.source_hold_seconds
    metadata["success_hold_trim"] = {
        "source_batch": str(plan.source),
        "source_success_hold_seconds": plan.source_hold_seconds,
        "target_success_hold_seconds": target,
        "trim_seconds": plan.source_hold_seconds - target,
        "trim_frames": plan.frame_delta,
        "control_step_seconds": plan.control_step_seconds,
        "method": (
            "drop the surplus terminal hold; keep one terminal reward; move "
            "discount=0 to the new final frame"
        ),
        "date": str(date.today()),
    }
    metadata["task"] = task
    return metadata


def write_batch(plan: BatchPlan) -> Path:
    """Write the re-timed batch atomically (a temporary sibling, then rename).

    Args:
        plan: From :func:`plan_trim`, with an output.

    Returns:
        The output directory.
    """
    if plan.output is None:
        raise ValueError("an output directory is required")
    tmp = plan.output.with_name(f".{plan.output.name}.tmp-{os.getpid()}")
    if tmp.exists():
        raise FileExistsError(f"temporary output already exists: {tmp}")
    tmp.mkdir(parents=True)
    try:
        for episode in plan.episodes:
            _retime_episode(plan, episode, tmp / episode.output_name)
        (tmp / "metadata.json").write_text(
            json.dumps(retimed_metadata(plan), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for extra in plan.source.glob("*.json"):
            if extra.name != "metadata.json":
                shutil.copy2(extra, tmp / extra.name)
        tmp.replace(plan.output)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return plan.output


def trim_batch(
    demo_dir: Path,
    out_dir: Path | None = None,
    *,
    target_hold_seconds: float | None = None,
    source_hold_seconds: float | None = None,
) -> Path:
    """Write the training view of a raw batch; see :func:`plan_trim`.

    Returns:
        The output directory (default ``<demo_dir>_hold<target>s``).
    """
    plan = plan_trim(
        demo_dir,
        None,
        target_hold_seconds=target_hold_seconds,
        source_hold_seconds=source_hold_seconds,
    )
    output = out_dir or default_output(plan.source, plan.target_hold_seconds)
    return write_batch(_with_output(plan, output))


def _with_output(plan: BatchPlan, output: Path) -> BatchPlan:
    destination = Path(output).expanduser().resolve()
    if destination == plan.source:
        raise ValueError("output directory must differ from the source batch")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing {destination}")
    return BatchPlan(
        source=plan.source,
        output=destination,
        metadata=plan.metadata,
        source_hold_seconds=plan.source_hold_seconds,
        target_hold_seconds=plan.target_hold_seconds,
        control_step_seconds=plan.control_step_seconds,
        frame_delta=plan.frame_delta,
        episodes=plan.episodes,
    )


def _summary(plan: BatchPlan, verb: str) -> str:
    old = [e.old_frames - 1 for e in plan.episodes]
    new = [e.new_frames - 1 for e in plan.episodes]
    return (
        f"{len(plan.episodes)} episodes {verb}: {min(old)}-{max(old)} -> "
        f"{min(new)}-{max(new)} transitions "
        f"({plan.source_hold_seconds:g}s -> {plan.target_hold_seconds:g}s hold)"
    )


@dataclass
class TrimConfig:
    """Drop the surplus terminal hold."""

    demo_dir: Path
    """Replay-format batch to re-time."""
    out_dir: Path | None = None
    """Output batch; default: <demo-dir>_hold<target>s."""
    target_hold_seconds: float | None = None
    """Default: task.training_success_hold_seconds, else 1.0."""
    source_hold_seconds: float | None = None
    """The recorded hold, when metadata.json lacks it."""
    dry_run: bool = False
    """Validate without writing."""


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: ``trim`` one batch."""
    args = tyro.extras.subcommand_cli_from_dict(
        {"trim": TrimConfig}, args=argv, description=__doc__
    )
    plan = plan_trim(
        args.demo_dir,
        None,
        target_hold_seconds=args.target_hold_seconds,
        source_hold_seconds=args.source_hold_seconds,
    )
    if args.dry_run:
        print(_summary(plan, "validated"))
        return
    output = args.out_dir or default_output(plan.source, plan.target_hold_seconds)
    write_batch(_with_output(plan, output))
    print(_summary(plan, "trimmed"))
    print(
        f"wrote {Path(output).expanduser().resolve()}; source unchanged: {plan.source}"
    )


if __name__ == "__main__":
    main()
