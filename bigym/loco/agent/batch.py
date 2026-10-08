"""Replay batches: the episode files and metadata the demo viewer reads.

A batch is a directory of ``seed<S>.npz`` episodes plus a ``metadata.json``
describing the environment they were produced in: the layout ``bigym-view``
reads (``bigym/vr/viewer/view_demos.py``) and ``bigym.loco.demos.rerender``
re-renders from. Each episode stores the full physical state of every frame
(``full_qpos``/``full_qvel``, ``T+1`` states around ``T`` actions), so it can
be replayed and re-rendered without running the policy again. ``bigym-agent
replay``, the evaluator and the server's development recording all write
their batches here.

The recording does not change the environment wrapper the policy runs
against: :class:`Recorder` reads ``env_tools.data`` after every step, and
:class:`RecordingEnv` is the evaluator's own ``LocalEnv`` with the two hooks
added, so a recorded rollout is bit-identical to an unrecorded one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from bigym.loco.demos.schema import BATCH_FORMAT
from bigym.loco.fingerprint import package_versions

from .envtools import EnvTools, LocalEnv
from .snapshots import sha256_text

# Where a running rollout says how far it has got, so a reader (the viewer's
# progress bars) can follow an episode that takes minutes. It sits in the
# run's output directory next to ``batch/``, and is dot-prefixed so batch and
# episode scans never list it.
PROGRESS_NAME = ".progress.json"
# One write per this many steps: often enough for a bar that moves, rare
# enough that the rollout is not doing file I/O instead of physics.
PROGRESS_EVERY = 25


def batch_metadata(env_tools: EnvTools, *, source: dict) -> dict:
    """Describe the environment a batch was produced in.

    The keys are the ones the readers need: every reader rebuilds the env
    with ``EnvConfig.from_metadata`` (``env_config``, with the flat ``task``
    and ``lowerbody_policy`` blocks alongside), and playback prefers
    ``control_step_seconds`` over the env's own step length. Everything is
    read from the live environment, not from a config file.

    Args:
        env_tools: The wrapper holding the environment being recorded.
        source: What produced the batch (``kind``, ``cell``, ``version``,
            ``policy_sha256``, ...).

    Returns:
        The metadata dict, ready to be written as ``metadata.json``.
    """
    outer = env_tools.outer
    blocks = outer.config.to_metadata(outer.task_name)
    return {
        "format": BATCH_FORMAT,
        "producer": "bigym.loco.agent.replay",
        "package_versions": package_versions(),
        "task": blocks["task"],
        "control_step_seconds": float(outer.control_step_seconds),
        "lowerbody_policy": blocks["lowerbody_policy"],
        "env_config": blocks["env_config"],
        "substrate_fingerprint": dict(outer.substrate_fingerprint()),
        "interface": env_tools.info()["interface"],
        "tier": env_tools.tier,
        "source": dict(source),
    }


def write_json(path: Path, payload: dict) -> Path:
    """Write JSON atomically (a reader polls these files while we write).

    Args:
        path: Destination file.
        payload: The object to serialise.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The temp name carries the pid: several evaluation workers write the
    # same batch metadata, and a shared temp name lets one worker's
    # os.replace consume the file another is about to move (the second
    # replace then fails). Dot-prefixed so batch scans never list it.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    os.replace(tmp, path)
    return path


class RunProgress:
    """``<out_dir>/.progress.json``: how far a running rollout has got.

    One small JSON object, rewritten atomically every
    :data:`PROGRESS_EVERY` steps and at the end of every episode::

        {"seed": 620000, "step": 812, "max_steps": 3523,
         "episode": 2, "episodes": 5, "done": false, "pid": 31337}

    ``episode`` counts the episodes that have *finished*, so it reads as
    "2 of 5 done"; ``done`` is true between episodes (the episode the other
    fields describe has ended). An evaluation splits its episodes over
    worker processes and cannot cheaply say where each one is, so it leaves
    ``step``/``max_steps`` at 0 and only counts episodes.

    The file is removed when the run finishes cleanly; a file left behind
    belongs to a run that died, which is what its ``pid`` is for.
    """

    def __init__(self, out_dir: Path, episodes: int = 0, every: int = PROGRESS_EVERY):
        """Bind the writer to one run's output directory.

        Args:
            out_dir: The run directory (``<cell>/replays/vNNN``, ``eval/vNNN``).
            episodes: How many episodes the whole run will produce.
            every: Write one update per this many steps.
        """
        self.path = Path(out_dir) / PROGRESS_NAME
        self.episodes = int(episodes)
        self.every = max(1, int(every))
        self.finished = 0
        self.seed = 0
        self.max_steps = 0
        self.step = 0

    def state(self, done: bool = False) -> dict:
        """Return the object that is written to the file."""
        return {
            "seed": int(self.seed),
            "step": int(self.step),
            "max_steps": int(self.max_steps),
            "episode": int(self.finished),
            "episodes": int(self.episodes),
            "done": bool(done),
            "pid": os.getpid(),
        }

    def write(self, done: bool = False) -> None:
        """Write the current state; a failure never stops the rollout."""
        try:
            write_json(self.path, self.state(done))
        except OSError:
            pass

    def begin(self, seed: int, max_steps: int = 0) -> None:
        """An episode is starting on ``seed``, capped at ``max_steps``."""
        self.seed = int(seed)
        self.max_steps = int(max_steps or 0)
        self.step = 0
        self.write()

    def advance(self, step: int) -> None:
        """The running episode reached ``step`` (written every ``every``)."""
        self.step = int(step)
        if self.step % self.every == 0:
            self.write()

    def finish(self, seed: int | None = None, step: int | None = None) -> None:
        """An episode has ended: count it and write ``done``."""
        if seed is not None:
            self.seed = int(seed)
        if step is not None:
            self.step = int(step)
        self.finished += 1
        self.write(done=True)

    def clear(self) -> None:
        """Remove the file: this run is over."""
        try:
            self.path.unlink()
        except OSError:
            pass


def write_episode_npz(
    path: Path,
    *,
    full_qpos,
    full_qvel,
    action,
    reward,
    seed: int,
    success: int = 0,
    fell: bool = False,
    length: int | None = None,
    termination: str = "",
) -> Path:
    """Write one episode of a replay batch.

    Args:
        path: Destination ``.npz``.
        full_qpos: ``[T+1, nq]`` state of every frame, the reset state first.
        full_qvel: ``[T+1, nv]`` velocities of the same frames.
        action: ``[T, action_dim]`` raw physical actions.
        reward: ``[T, 1]`` per-step reward.
        seed: The episode's seed.
        success: 1 when the episode succeeded.
        fell: Whether the robot fell.
        length: Number of steps, or None for ``len(action)``.
        termination: The env's termination label.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    action = np.asarray(action, dtype=np.float32)
    # Staged under a dot-prefixed, per-process name and moved into place, so
    # a viewer polling the batch never opens a half-written episode.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as handle:
        _savez(
            handle,
            full_qpos,
            full_qvel,
            action,
            reward,
            seed,
            success,
            fell,
            length,
            termination,
        )
    os.replace(tmp, path)
    return path


def _savez(
    handle,
    full_qpos,
    full_qvel,
    action,
    reward,
    seed,
    success,
    fell,
    length,
    termination,
):
    """Write the episode arrays into an open binary file."""
    np.savez(
        handle,
        full_qpos=np.asarray(full_qpos, dtype=np.float64),
        full_qvel=np.asarray(full_qvel, dtype=np.float64),
        action=action,
        reward=np.asarray(reward, dtype=np.float32).reshape(-1, 1),
        seed=np.int64(int(seed)),
        success=np.float32(float(success)),
        fell=np.float32(float(bool(fell))),
        length=np.int64(int(len(action) if length is None else length)),
        termination=np.str_(str(termination)),
    )


class Recorder:
    """Collect the full physical state along one rollout.

    The environment wrapper is untouched: ``EnvTools`` exposes ``model`` and
    ``data``, and this reads ``data.qpos``/``data.qvel`` after each step. The
    state list holds ``T+1`` entries (the state the episode reset to, then
    one per step) around ``T`` actions and rewards.
    """

    def __init__(self, env_tools: EnvTools):
        """Bind the recorder to a wrapper.

        Args:
            env_tools: The wrapper holding the environment.
        """
        self.tools = env_tools
        self.qpos: list[np.ndarray] = []
        self.qvel: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []

    def reset(self) -> None:
        """Drop everything recorded so far."""
        self.qpos.clear()
        self.qvel.clear()
        self.actions.clear()
        self.rewards.clear()

    def capture_state(self) -> None:
        """Append the simulator's current state."""
        data = self.tools.data
        self.qpos.append(np.array(data.qpos, dtype=np.float64))
        self.qvel.append(np.array(data.qvel, dtype=np.float64))

    def capture_step(self, action, reward: float) -> None:
        """Append one action and the reward it earned.

        Args:
            action: The raw physical action the policy returned.
            reward: The reward of that step.
        """
        self.actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
        self.rewards.append(float(reward))

    def arrays(self) -> dict[str, np.ndarray]:
        """Return the recorded episode as the batch's arrays."""
        return {
            "full_qpos": np.asarray(self.qpos, dtype=np.float64),
            "full_qvel": np.asarray(self.qvel, dtype=np.float64),
            "action": np.asarray(self.actions, dtype=np.float32),
            "reward": np.asarray(self.rewards, dtype=np.float32).reshape(-1, 1),
        }


class RecordingEnv(LocalEnv):
    """The evaluator's ``LocalEnv`` with a :class:`Recorder` attached."""

    def __init__(
        self,
        tools: EnvTools,
        recorder: Recorder | None = None,
        progress: RunProgress | None = None,
    ):
        """Wrap a wrapper and record every episode run through it.

        Args:
            tools: The wrapper holding the environment.
            recorder: An existing recorder, or None for a fresh one.
            progress: Where to report the step count, or None for silence.
        """
        super().__init__(tools)
        self.recorder = recorder if recorder is not None else Recorder(tools)
        self.progress = progress
        self.steps = 0

    def reset(self, seed):
        """Reset the env and start a new recording.

        Args:
            seed: Episode seed.

        Returns:
            The first observation.
        """
        obs = super().reset(seed)
        self.recorder.reset()
        self.recorder.capture_state()
        self.steps = 0
        return obs

    def step(self, raw):
        """Apply one physical action and record the state it led to.

        Args:
            raw: The physical action.

        Returns:
            ``(obs, reward, done, info)``.
        """
        obs, reward, done, info = super().step(raw)
        self.recorder.capture_state()
        self.recorder.capture_step(raw, reward)
        self.steps += 1
        if self.progress is not None:
            self.progress.advance(self.steps)
        return obs, reward, done, info


def write_record(batch_dir: Path, record: dict, arrays: dict[str, np.ndarray]) -> Path:
    """Write one finished episode into a batch directory.

    Args:
        batch_dir: The batch directory.
        record: The per-episode record.
        arrays: The recorded arrays.

    Returns:
        The path written.
    """
    return write_episode_npz(
        Path(batch_dir) / f"seed{int(record['seed'])}.npz",
        full_qpos=arrays["full_qpos"],
        full_qvel=arrays["full_qvel"],
        action=arrays["action"],
        reward=arrays["reward"],
        seed=int(record["seed"]),
        success=int(record["success"]),
        fell=bool(record["fell"]),
        length=int(record["length"]),
        termination=str(record.get("termination") or ""),
    )


def batch_source(
    cell: Path, policy_path: Path, version: int | None, kind: str = "agent_policy"
) -> dict:
    """Describe where a batch's policy came from.

    Args:
        cell: The cell directory.
        policy_path: The policy file that was run.
        version: The policy version, or None for a loose file.
        kind: The source kind recorded in the metadata.

    Returns:
        The ``source`` block of the batch metadata.
    """
    return {
        "kind": kind,
        "cell": str(Path(cell).resolve()),
        "version": version,
        "policy": str(policy_path),
        "policy_sha256": sha256_text(Path(policy_path).read_bytes()),
    }
