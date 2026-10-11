"""The episode files of a replay-format batch and the scalars each one stores."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def is_episode_file(name: str) -> bool:
    """True for a finished episode file.

    Writers produce an episode under a temporary name and rename (or
    hard-link) it into place, so a name that is hidden or carries a ``.tmp``
    / ``.part`` marker is a file still being written, not an episode.
    """
    if not name.endswith(".npz") or name.startswith("."):
        return False
    return ".tmp" not in name and ".part" not in name


def episode_files(batch: Path) -> list[Path]:
    """The batch's episode files, in the order the viewer shows them."""
    return sorted(
        path for path in Path(batch).glob("*.npz") if is_episode_file(path.name)
    )


def episode_info(path: Path) -> dict:
    """Read one episode's scalars (seed, success, length, termination)."""
    info: dict = {
        "name": path.name,
        "seed": None,
        "success": None,
        "length": None,
        "termination": "",
        "fell": None,
    }
    try:
        with np.load(path, allow_pickle=False) as ep:
            keys = set(ep.files)
            if "seed" in keys:
                info["seed"] = int(np.asarray(ep["seed"]).reshape(-1)[0])
            if "success" in keys:
                info["success"] = float(np.asarray(ep["success"]).reshape(-1)[0])
            if "fell" in keys:
                info["fell"] = float(np.asarray(ep["fell"]).reshape(-1)[0])
            if "length" in keys:
                info["length"] = int(np.asarray(ep["length"]).reshape(-1)[0])
            elif "reward" in keys:
                info["length"] = int(np.asarray(ep["reward"]).reshape(-1).shape[0])
            if "termination" in keys:
                info["termination"] = str(np.asarray(ep["termination"]).reshape(-1)[0])
    except (OSError, ValueError, KeyError, EOFError):
        pass
    return info


# Per-episode scalars are read once per batch and re-read only when the set
# of .npz files changes (a running evaluation adds one file per episode).
_EPISODE_CACHE: dict[str, tuple[tuple[str, ...], list[dict]]] = {}


def episode_infos(batch: Path, files: list[Path] | None = None) -> list[dict]:
    """Scalars of every episode in a batch, cached until its files change."""
    batch = Path(batch)
    paths = list(files) if files is not None else episode_files(batch)
    signature = tuple(p.name for p in paths)
    cached = _EPISODE_CACHE.get(str(batch))
    if cached is not None and cached[0] == signature:
        return cached[1]
    infos = [episode_info(p) for p in paths]
    _EPISODE_CACHE[str(batch)] = (signature, infos)
    return infos


def batch_success(batch: Path) -> float | None:
    """Mean of the stored per-episode ``success`` values, when they exist."""
    values = [i["success"] for i in episode_infos(batch) if i["success"] is not None]
    return float(np.mean(values)) if values else None
