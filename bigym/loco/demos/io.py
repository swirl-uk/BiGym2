"""npz + metadata.json IO for native demos (schema: bigym.loco.demos.schema)."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from bigym.loco.demos.schema import validate_episode


def load_episode(path: Path) -> dict[str, np.ndarray]:
    """Load one episode npz into a plain dict (all arrays materialized)."""
    with Path(path).open("rb") as f:
        episode = np.load(f)
        return {k: episode[k] for k in episode.keys()}


def save_episode(
    episode: Mapping[str, np.ndarray], path: Path, *, validate: bool = True
) -> Path:
    """Atomically save one episode npz (write-then-rename)."""
    path = Path(path)
    if validate:
        issues = validate_episode(episode)
        if issues:
            raise ValueError(f"Refusing to save invalid episode {path.name}: {issues}")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **episode)  # ty: ignore[invalid-argument-type]
    tmp = path.with_suffix(".npz.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("wb") as f:
        f.write(buffer.getbuffer())
    tmp.rename(path)
    return path


def load_metadata(demo_dir: Path) -> dict[str, Any] | None:
    """Load the collection dir's metadata.json (None when absent)."""
    meta_path = Path(demo_dir) / "metadata.json"
    if not meta_path.exists():
        return None
    with meta_path.open("r") as f:
        return json.load(f)
