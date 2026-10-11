"""One episode of a demo batch or a dataset task, loaded into the replay env."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

import numpy as np

from bigym.loco import EnvConfig, make
from bigym.loco.demos.kinematic import run_task_hook
from bigym.vr.viewer.episodes import episode_infos


def episode_labels(infos: list[dict]) -> list[str]:
    """Dropdown labels such as ``620003 ✓ 1599`` / ``620002 ✗ 1700 timeout``."""
    labels = []
    for i, info in enumerate(infos):
        seed = info["seed"]
        head = str(seed) if seed is not None else info["name"]
        mark = "·"
        if info["success"] is not None:
            mark = "✓" if info["success"] > 0 else "✗"
        parts = [head, mark]
        if info["length"] is not None:
            parts.append(str(info["length"]))
        end = str(info["termination"] or "")
        if end and end != "success":
            parts.append(end)
        label = " ".join(parts)
        # viser dropdowns key on the option string, so equal labels (an old
        # batch without seeds) must still differ.
        labels.append(f"{label}  [{i}]" if label in labels else label)
    return labels


def load_metadata(demo_dir: Path) -> dict:
    """The batch's ``metadata.json``, at its root or one folder down."""
    for path in sorted(demo_dir.glob("metadata.json")) + sorted(
        demo_dir.glob("*/metadata.json")
    ):
        with path.open() as f:
            return json.load(f)
    raise FileNotFoundError(f"No metadata.json under {demo_dir}")


def build_env(metadata: dict):
    """A ``make()`` env built the way the batch's metadata says it was recorded."""
    return make(metadata["task"]["task_name"], EnvConfig.from_metadata(metadata))


class Store(Protocol):
    """The episodes a frontend plays: a demo batch or one dataset task.

    load() must run on the main thread: env.reset renders camera obs on the
    main thread's EGL context, and the main loop is the only mutator of
    (model, data).
    """

    files: list[Path]
    """One path per episode; the frontends show ``.name`` and use ``.stem``."""
    ep: int
    """The loaded episode, -1 before the first load."""
    info: dict
    """Per-episode scalars (success, fell, length, termination) of the loaded
    episode, as far as the source stores them."""
    qvel: np.ndarray | None
    """``(T, nv)`` full qvel of the loaded episode, when stored."""
    cmd: np.ndarray | None
    """``(T, 3)`` lower-body command of the loaded episode, when stored."""

    @property
    def qpos(self) -> np.ndarray:
        """``(T, nq)`` full qpos of the loaded episode."""
        ...

    @property
    def rewards(self) -> np.ndarray:
        """Per-frame reward of the loaded episode."""
        ...

    def load(self, idx: int) -> bool:
        """Load episode ``idx`` and reseed the env; False if already loaded."""
        ...

    def episode_labels(self) -> list[str]:
        """One episode-dropdown label per file."""
        ...

    def episode_infos(self) -> list[dict]:
        """Per-episode seed, outcome and length, for the episode list."""
        ...

    def after_forward(self, frame: int) -> None:
        """Run once a frame is posed and forwarded."""
        ...


class DemoStore:
    """Episode loading + reset-with-seed for stored-qpos demo batches."""

    def __init__(self, env, files: list[Path]):
        """Bind the store to an env and its list of episode .npz files."""
        self.env = env
        self.files = files
        self.ep = -1
        self._qpos: np.ndarray | None = None
        self._rewards: np.ndarray | None = None
        self.qvel: np.ndarray | None = None
        self.cmd: np.ndarray | None = None
        self.info: dict = {}

    @property
    def qpos(self) -> np.ndarray:
        """``(T, nq)`` full qpos of the loaded episode."""
        assert self._qpos is not None, "no episode loaded"
        return self._qpos

    @property
    def rewards(self) -> np.ndarray:
        """Per-frame reward of the loaded episode."""
        assert self._rewards is not None, "no episode loaded"
        return self._rewards

    def load(self, idx: int) -> bool:
        """Load episode ``idx`` and reseed the env; False if already loaded."""
        idx = int(np.clip(idx, 0, len(self.files) - 1))
        if idx == self.ep:
            return False
        ep = np.load(self.files[idx])
        if "seed" in ep.files:
            self.env.reset(seed=int(np.asarray(ep["seed"]).reshape(-1)[0]))
        self._qpos = np.asarray(ep["full_qpos"], dtype=np.float64)
        self._rewards = np.asarray(ep["reward"]).reshape(-1)
        # The velocity arrows need both; agent rollouts store no command.
        self.qvel = np.asarray(ep["full_qvel"]) if "full_qvel" in ep.files else None
        self.cmd = (
            np.asarray(ep["lowerbody_command"])
            if "lowerbody_command" in ep.files
            else None
        )
        self.info = {
            key: (
                str(np.asarray(ep[key]).reshape(-1)[0])
                if key == "termination"
                else float(np.asarray(ep[key]).reshape(-1)[0])
            )
            for key in ("success", "fell", "length", "termination")
            if key in ep.files
        }
        self.ep = idx
        return True

    def episode_labels(self) -> list[str]:
        """Seed + outcome when the batch stores them, else the file name.

        The shared timestamp prefix eats the narrow dropdown width, so a
        batch without seeds shows the distinctive tail of each name, which
        already carries ``<episode>_<steps>``.
        """
        infos = episode_infos(self.files[0].parent, self.files)
        if any(info["seed"] is not None for info in infos):
            return episode_labels(infos)
        names = [f.name for f in self.files]
        prefix = len(os.path.commonprefix(names)) if len(names) > 1 else 0
        return [name[prefix:] for name in names]

    def episode_infos(self) -> list[dict]:
        """Per-episode seed, outcome and length read from the batch."""
        return episode_infos(self.files[0].parent.resolve(), self.files)

    def after_forward(self, frame: int) -> None:
        """Nothing: a batch episode is fully posed by its stored qpos."""


class DatasetStore:
    """A :class:`Store` over one task of a dataset.

    The episodes come from a LeRobot task export (the published dataset or
    a local copy) through :class:`bigym.loco.agent.demo_video.TaskDemos`,
    which reads only the state columns, never a camera PNG. Each load resets
    the env on the episode's seed; ``after_forward`` runs the task's
    ``_on_step`` hook once a frame is posed.
    """

    COLUMNS = ("full_qpos", "full_qvel", "reward", "lowerbody_command")

    def __init__(self, env, inner, demos):
        """Bind the store to a replay env and a task's :class:`TaskDemos`."""
        self.env = env
        self.inner = inner
        self.demos = demos
        self.episodes = list(demos.episodes)
        # Stand-in paths: the frontends show .name and use .stem for
        # capture file names; nothing is read from them.
        self.files = [
            demos.dir / f"{e.index:03d}_{Path(e.source_file).stem}"
            for e in self.episodes
        ]
        self.labels = [
            f"{e.index:02d} · seed {e.seed} · {e.length}" for e in self.episodes
        ]
        self.infos = [
            {
                "name": path.name,
                "seed": e.seed,
                "success": None,
                "length": e.length,
                "termination": "",
                "fell": None,
            }
            for path, e in zip(self.files, self.episodes, strict=True)
        ]
        self.ep = -1
        self._qpos: np.ndarray | None = None
        self._rewards: np.ndarray | None = None
        self.qvel: np.ndarray | None = None
        self.cmd: np.ndarray | None = None
        self.info: dict = {}

    @property
    def qpos(self) -> np.ndarray:
        """``(T, nq)`` full qpos of the loaded episode."""
        assert self._qpos is not None, "no episode loaded"
        return self._qpos

    @property
    def rewards(self) -> np.ndarray:
        """Per-frame reward of the loaded episode."""
        assert self._rewards is not None, "no episode loaded"
        return self._rewards

    def load(self, idx: int) -> bool:
        """Load episode ``idx`` and reseed the env; False if already loaded."""
        idx = int(np.clip(idx, 0, len(self.episodes) - 1))
        if idx == self.ep:
            return False
        episode = self.episodes[idx]
        arrays = self.demos.columns(episode, self.COLUMNS)
        self.env.reset(seed=episode.seed)
        self._qpos = np.asarray(arrays["full_qpos"], dtype=np.float64)
        steps = self._qpos.shape[0]
        self._rewards = np.asarray(
            arrays.get("reward", np.zeros(steps)), dtype=float
        ).reshape(-1)
        self.qvel = arrays.get("full_qvel")
        self.cmd = arrays.get("lowerbody_command")
        self.info = {"length": float(steps)}
        self.ep = idx
        return True

    def episode_labels(self) -> list[str]:
        """``<index> · seed <seed> · <length>`` per episode."""
        return self.labels

    def episode_infos(self) -> list[dict]:
        """Seed and length per episode; the dataset stores no outcome."""
        return self.infos

    def after_forward(self, frame: int) -> None:
        """Run the task's step hook on the posed frame (target highlight)."""
        run_task_hook(self.inner)
