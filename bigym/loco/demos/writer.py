"""Replay-format episode storage (the native-demo npz writer).

Demos are benchmark assets, so the code that WRITES them lives next to the
schema (bigym.loco.demos); a training codebase's replay-buffer sampler stays
training-side.

Behavior contract (frozen): filenames ``<ts>_<idx>_<len>.npz``, atomic
write-then-rename via np.savez_compressed, success = final reward ~= 1.0,
relabeling sets demo=1 on successful episodes. Schema validation is
deliberately OFF here — older batches predate the schema and training
rollout dumps reuse save_episode; enable it per call site, not globally.
"""

from __future__ import annotations

import datetime
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from bigym.loco.demos import io as demo_io


def episode_len(episode):
    """Transitions in an episode (frames minus the dummy first one)."""
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, path):
    """Atomic npz write (schema validation off)."""
    demo_io.save_episode(episode, path, validate=False)


def load_episode(path):
    """Load one episode npz (delegates to :mod:`bigym.loco.demos.io`)."""
    return demo_io.load_episode(path)


class ReplayBufferStorage:
    """Writes episodes to a replay directory as ``<ts>_<idx>_<len>.npz``."""

    def __init__(
        self,
        data_specs,
        replay_dir,
        use_relabeling,
        is_demo_buffer=False,
        async_write=False,
    ):
        """Prepare the replay dir and count the episodes already on disk."""
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        self._use_relabeling = use_relabeling
        self._is_demo_buffer = is_demo_buffer
        # Off-thread npz compression+write for interactive collection (a
        # ~0.5-1s compressed write would freeze the VR loop at every save).
        # Single worker keeps writes ordered; success validation and episode
        # counters stay synchronous, only the disk write is deferred. Callers
        # MUST call flush() before relying on the files (and before exit).
        self._async_write = bool(async_write)
        self._write_executor = None
        self._write_futures = []
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        """Number of transitions stored in the replay directory."""
        return self._num_transitions

    def add(self, time_step):
        """Append one timestep, storing the episode once it ends."""
        for spec in self._data_specs:
            value = time_step[spec.name]
            # Remove frame stacking
            if spec.name == "low_dim_obs":
                low_dim = spec.shape[0]
                value = value[..., -low_dim:]
            elif spec.name == "rgb_obs":
                rgb_dim = spec.shape[1]
                value = value[:, -rgb_dim:]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype, (
                spec.name,
                spec.shape,
                value.shape,
                spec.dtype,
                value.dtype,
            )
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            if self._use_relabeling:
                episode = self._relabel_episode(episode)
            if self._is_demo_buffer:
                # If this is demo replay buffer, save only when it's successful
                if self._check_if_successful(episode):
                    self._store_episode(episode)
            else:
                self._store_episode(episode)

    def add_episode(
        self,
        episode,
        *,
        require_success=False,
        demo_value=None,
        is_expert_value=None,
    ):
        """Store a complete episode, returning False when it is rejected."""
        episode = {k: np.array(v, copy=True) for k, v in episode.items()}
        if require_success and not self._check_if_successful(episode):
            return False
        if self._use_relabeling:
            episode = self._relabel_episode(episode)
        if demo_value is not None:
            episode["demo"] = np.full_like(episode["demo"], float(demo_value))
        if is_expert_value is not None:
            template = episode.get("is_expert", episode["demo"])
            episode["is_expert"] = np.full_like(template, float(is_expert_value))
        if self._is_demo_buffer and not self._check_if_successful(episode):
            return False
        self._store_episode(episode)
        return True

    def _relabel_episode(self, episode):
        if self._check_if_successful(episode):
            episode["demo"] = np.ones_like(episode["demo"])
        return episode

    def _check_if_successful(self, episode):
        reward = episode["reward"]
        return np.isclose(reward[-1], 1.0)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for path in self._replay_dir.glob("*.npz"):
            _, _, eps_len = path.stem.split("_")
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        file_name = f"{ts}_{eps_idx}_{eps_len}.npz"
        if not self._async_write:
            save_episode(episode, self._replay_dir / file_name)
            return
        if self._write_executor is None:
            self._write_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="episode-writer"
            )
        # Surface earlier write failures loudly instead of dropping demos
        # silently mid-session.
        for fut in [f for f in self._write_futures if f.done()]:
            self._write_futures.remove(fut)
            exc = fut.exception()
            if exc is not None:
                raise RuntimeError("async episode write FAILED") from exc
        self._write_futures.append(
            self._write_executor.submit(
                self._write_and_report, episode, self._replay_dir / file_name
            )
        )

    @staticmethod
    def _write_and_report(episode, path):
        # Runs on the writer thread. Print failures immediately: a future's
        # exception is invisible until retrieved, and on Ctrl-C the flush()
        # that would surface it may never run.
        try:
            save_episode(episode, path)
        except BaseException:
            print(f"\n[!!!] async episode write FAILED for {path}:", flush=True)
            traceback.print_exc()
            raise

    def flush(self):
        """Wait for pending async writes; raise if any failed."""
        futures, self._write_futures = self._write_futures, []
        for fut in futures:
            exc = fut.exception()
            if exc is not None:
                raise RuntimeError("async episode write FAILED") from exc
        if self._write_executor is not None:
            self._write_executor.shutdown(wait=True)
            self._write_executor = None
