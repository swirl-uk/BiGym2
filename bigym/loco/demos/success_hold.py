"""Cut a raw replay-format demo batch to its training view.

An episode succeeds once the task predicate has held for
``success_hold_seconds``; the recorded episode ends on that frame with the
single terminal reward (1) and discount (0). The collector records with a
longer hold than training and evaluation use, so the raw batch is cut before
it trains: each episode is replayed from its engage snapshot on the task's
official env and ends on the control step where that env latches success. A
predicate that held long enough, broke and restarted before the collection
hold completed makes that step fall anywhere before the raw end, so no fixed
trim finds it. The cut goes to a NEW directory and the source batch is never
modified. Usage::

    python -m bigym.loco.demos.success_hold --demo-dir <batch>
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import numpy as np
import tyro

from bigym.loco import make

if TYPE_CHECKING:
    import mujoco

    from bigym.loco.env import BiGym

# Episode-level arrays: their leading axis is not time.
EPISODE_LEVEL_PREFIXES = ("init_", "lb_state.")
EPISODE_LEVEL_KEYS = frozenset({"seed", "pre_engage_steps"})


def restore_engage_state(
    env: BiGym, episode: Mapping[str, np.ndarray]
) -> mujoco.MjData:
    """Reset ``env`` on the episode's seed and write its engage snapshot.

    The seed places what qpos does not hold (racks, reach targets); the
    snapshot sets the simulator and the lower-body controller.

    Returns:
        The env's data, at the snapshot.
    """
    env.reset(seed=int(np.asarray(episode["seed"]).reshape(-1)[0]))
    simulation = env.inner_env.simulation
    data = simulation.data
    for name in ("qpos", "qvel", "ctrl", "act", "qacc_warmstart"):
        if f"init_{name}" in episode:
            getattr(data, name)[:] = episode[f"init_{name}"]
    simulation.forward()
    data.qacc_warmstart[:] = episode["init_qacc_warmstart"]
    env.set_lowerbody_state(
        {
            key.removeprefix("lb_state."): np.asarray(value)
            for key, value in episode.items()
            if key.startswith("lb_state.")
        }
    )
    return data


def latch_step(env: BiGym, episode: Mapping[str, np.ndarray]) -> int:
    """The row of ``episode`` on which ``env`` latches success replaying it.

    Restores the engage snapshot, then steps the recorded actions (row t
    holds the action that produced row t), checking that every step lands
    on the recorded ``full_qpos``. ``env`` must already map normalized
    actions through the episode's ``action_stats`` (``set_action_stats``).

    Raises:
        ValueError: The replay leaves the recording, or the env ends the
            episode or the actions run out before success latches.
    """
    data = restore_engage_state(env, episode)
    actions = np.asarray(episode["action"], dtype=np.float32)
    full_qpos = np.asarray(episode["full_qpos"])
    for step in range(1, len(actions)):
        time_step = env.step(actions[step])
        if not np.array_equal(data.qpos, full_qpos[step]):
            raise ValueError(f"the replay leaves the recorded full_qpos at row {step}")
        if env.episode_succeeded():
            return step
        if time_step.last():
            raise ValueError(
                f"the env ended the episode at row {step} without success "
                f"({env.episode_termination()})"
            )
    raise ValueError(
        f"success does not latch within the {len(actions) - 1} recorded steps"
    )


def load_metadata(demo_dir: Path) -> dict:
    """The batch's ``metadata.json``."""
    return json.loads((Path(demo_dir) / "metadata.json").read_text(encoding="utf-8"))


def latch_steps(demo_dir: Path, task: str | None = None) -> tuple[dict[str, int], dict]:
    """Replay every episode of a batch on the task's official env (no cameras).

    Args:
        demo_dir: Replay-format batch with ``full_qpos`` and engage snapshots.
        task: Default: the batch's ``task.task_name``.

    Returns:
        The latch row of each episode by file name, and the env's
        ``substrate_fingerprint()``.
    """
    metadata = load_metadata(demo_dir)
    env = make(task or metadata["task"]["task_name"], camera_keys=()).bigym
    # The recorded actions are normalized to the collector's envelope, not to
    # the env's own bounds.
    stats = metadata["action_stats"]
    env.set_action_stats(
        np.asarray(stats["min"], dtype=np.float32).reshape(-1),
        np.asarray(stats["max"], dtype=np.float32).reshape(-1),
    )
    try:
        steps = {}
        for path in sorted(Path(demo_dir).glob("*.npz")):
            with np.load(path) as data:
                episode = {key: data[key] for key in data.files if key != "rgb_obs"}
            steps[path.name] = latch_step(env, episode)
        return steps, env.substrate_fingerprint()
    finally:
        env.close()


def default_output(source: Path, hold_seconds: float) -> Path:
    """``<source>_hold<hold>s`` next to the source batch."""
    label = f"{hold_seconds:g}".replace(".", "p")
    return source.with_name(f"{source.name}_hold{label}s")


def cut_name(source: Path, frames: int) -> str:
    """``<timestamp>_<index>_<transitions>.npz`` for ``source`` cut to ``frames``."""
    parts = source.stem.split("_")
    if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
        raise ValueError(
            f"{source.name}: expected replay filename <timestamp>_<index>_<length>.npz"
        )
    return f"{parts[0]}_{parts[1]}_{frames - 1}.npz"


def cut_episode(source: Path, frames: int, output: Path) -> int:
    """Write ``source`` cut to its first ``frames`` frames, ending in success.

    Per-step arrays are cut; the single reward=1 and discount=0 move to the
    new final frame. Episode-level arrays are copied whole.

    Returns:
        The number of frames dropped.
    """
    with np.load(source) as data:
        values = {key: np.asarray(data[key]) for key in data.files}
    old_frames = int(values["action"].shape[0])
    if not 2 <= frames <= old_frames:
        raise ValueError(f"{source.name}: cannot cut {old_frames} frames to {frames}")
    out = {}
    for key, value in values.items():
        episode_level = key in EPISODE_LEVEL_KEYS or key.startswith(
            EPISODE_LEVEL_PREFIXES
        )
        if episode_level or value.ndim == 0 or value.shape[0] != old_frames:
            out[key] = value
            continue
        out[key] = np.array(value[:frames], copy=True)
        if key == "reward":
            out[key].fill(0.0)
            out[key].reshape(-1)[-1] = 1.0
        elif key == "discount":
            out[key].fill(1.0)
            out[key].reshape(-1)[-1] = 0.0
    with output.open("wb") as f:
        np.savez_compressed(f, **out)  # ty: ignore[invalid-argument-type]
    return old_frames - frames


def cut_batch(
    demo_dir: Path, out_dir: Path, steps: Mapping[str, int], fingerprint: dict
) -> Path:
    """Write a batch whose every episode ends on its latch row.

    Written to a temporary sibling, then renamed; the source batch's other
    ``*.json`` files are copied along.

    Args:
        demo_dir: The source batch.
        out_dir: The output directory; must not exist.
        steps: The latch row of each episode by file name (:func:`latch_steps`).
        fingerprint: ``substrate_fingerprint()`` of the env that latched.

    Returns:
        The output directory.
    """
    source = Path(demo_dir).expanduser().resolve()
    output = Path(out_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing {output}")
    names = sorted(path.name for path in source.glob("*.npz"))
    if not names or names != sorted(steps):
        raise ValueError(f"latch rows {sorted(steps)} do not match {source}'s episodes")
    metadata = load_metadata(source)
    tmp = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    tmp.mkdir(parents=True)
    try:
        trim_frames = {}
        for name in names:
            frames = steps[name] + 1
            cut = cut_name(source / name, frames)
            trim_frames[cut] = cut_episode(source / name, frames, tmp / cut)
        (tmp / "metadata.json").write_text(
            json.dumps(
                cut_metadata(metadata, source, trim_frames, fingerprint),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        for extra in source.glob("*.json"):
            if extra.name != "metadata.json":
                shutil.copy2(extra, tmp / extra.name)
        tmp.replace(output)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return output


def cut_metadata(
    metadata: dict, source: Path, trim_frames: dict[str, int], fingerprint: dict
) -> dict:
    """The source metadata at the env's hold, with the cut recorded."""
    task = dict(metadata["task"])
    source_hold = float(task["success_hold_seconds"])
    target = float(fingerprint["success_hold_seconds"])
    task.update(
        success_hold_seconds=target,
        training_success_hold_seconds=target,
        collect_success_hold_seconds=source_hold,
    )
    metadata = dict(
        metadata,
        task=task,
        env_config=dict(metadata["env_config"], success_hold_seconds=target),
    )
    metadata["success_hold_trim"] = {
        "source_batch": str(source),
        "source_success_hold_seconds": source_hold,
        "target_success_hold_seconds": target,
        "trim_frames": trim_frames,
        "control_step_seconds": metadata["control_step_seconds"],
        "method": (
            "replay each episode from its engage snapshot on the task's official "
            "env; end it on the control step that env latches success; keep one "
            "terminal reward; move discount=0 to the new final frame"
        ),
        "substrate_fingerprint": fingerprint,
        "date": str(date.today()),
    }
    return metadata


def latch_batch(demo_dir: Path, out_dir: Path | None = None) -> Path:
    """Replay and cut a raw batch; see :func:`latch_steps` and :func:`cut_batch`.

    Returns:
        The output directory (default ``<demo_dir>_hold<env hold>s``).
    """
    source = Path(demo_dir).expanduser().resolve()
    steps, fingerprint = latch_steps(source)
    output = out_dir or default_output(source, fingerprint["success_hold_seconds"])
    return cut_batch(source, output, steps, fingerprint)


@dataclass
class LatchConfig:
    """End each episode on the step the task's official env latches success."""

    demo_dir: Path
    """Replay-format batch recorded with a longer hold than the env's."""
    out_dir: Path | None = None
    """Output batch; default: <demo-dir>_hold<env hold>s."""


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: cut one batch."""
    args = tyro.cli(LatchConfig, args=argv, description=__doc__)
    output = latch_batch(args.demo_dir, args.out_dir)
    trim = load_metadata(output)["success_hold_trim"]
    cut = trim["trim_frames"].values()
    print(
        f"{len(cut)} episodes latch at the {trim['target_success_hold_seconds']:g}s "
        f"hold: {min(cut)}-{max(cut)} frames cut; wrote {output}"
    )


if __name__ == "__main__":
    main()
