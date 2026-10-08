"""Read a LeRobot v3 lossless task export back into replay-format episodes.

This is the reader half of :mod:`bigym.loco.demos.lerobot_export`. It reads
the on-disk format directly (pyarrow + PNG decode); the ``lerobot`` package
is never imported, so it works on every Python the core package supports.

Only lossless PNG-mode datasets are accepted: video-mode cameras are lossy
and would silently change training inputs.

Alignment: the export (``meta/alignment.json``, version 2) stores
transition-scoped features shifted so LeRobot frame k pairs obs[k] with the
action executed FROM it; this reader undoes the shift (replay index 0 comes
back from the ``first_transition`` sidecar, the repeated final frame is
dropped), so the episodes come back exactly as the collector wrote them:
row t holds the observation at step t together with the action, reward and
discount of the transition that PRODUCED it (row 0 is the reset row with a
zero action).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from bigym.loco.action_representation import (
    ABSOLUTE,
    convert_episode_action_representation,
    validate_action_representation,
)
from bigym.loco.demos.lerobot_export import ACTION_ALIGNMENT_VERSION

INDEX_FEATURES = ("timestamp", "frame_index", "episode_index", "index", "task_index")
IMAGE_PREFIX = "observation.images."


def load_task_metadata(task_dir: Path) -> dict[str, Any]:
    """Return the collector metadata stored with a task export (root ``metadata.json``)."""
    return json.loads((Path(task_dir) / "metadata.json").read_text())


def load_info(task_dir: Path) -> dict[str, Any]:
    """Return the LeRobot ``meta/info.json`` of a task export."""
    return json.loads((Path(task_dir) / "meta" / "info.json").read_text())


def load_transition_keys(task_dir: Path) -> list[str]:
    """Return the export's transition-scoped column names (``meta/alignment.json``).

    Raises:
        RuntimeError: The export uses another action alignment version.
    """
    alignment = json.loads((Path(task_dir) / "meta" / "alignment.json").read_text())
    version = int(alignment["action_alignment_version"])
    if version != ACTION_ALIGNMENT_VERSION:
        raise RuntimeError(
            f"{task_dir}: action alignment v{version}; this reader reads "
            f"v{ACTION_ALIGNMENT_VERSION}"
        )
    return list(alignment["transition_keys"])


def undo_transition_shift(
    episode: dict[str, np.ndarray],
    first_transition: dict[str, Any],
    transition_keys: list[str],
    where: str = "",
) -> None:
    """Shift an exported episode's transition columns back, in place.

    LeRobot frame k stores the transition executed FROM obs[k]; the
    collector's row t holds the one that PRODUCED obs[t]. Row 0 comes back
    from the ``first_transition`` sidecar and the repeated final frame is
    dropped. Keys the episode does not carry are skipped.

    Args:
        episode: Column name to ``[T, ...]`` array, as read from the export.
        first_transition: The episode's ``first_transition`` entry of
            ``meta/episode_init_states.json``.
        transition_keys: The export's transition-scoped column names.
        where: What to name in the error message.
    """
    for k in transition_keys:
        if k not in episode:
            continue
        rec = first_transition.get(k)
        if rec is None:
            raise RuntimeError(
                f"{where}: alignment v2 needs first_transition[{k!r}] in "
                "episode_init_states.json"
            )
        first_row = np.asarray(rec["data"], dtype=rec["dtype"]).reshape(rec["shape"])
        episode[k] = np.concatenate([first_row[None], episode[k][:-1]], axis=0)


def load_episodes(
    task_dir: Path,
    max_episodes: int = -1,
    *,
    action_representation: str = ABSOLUTE,
    upper_delta_scale_rad: float | None = None,
) -> list[tuple[str, dict[str, np.ndarray]]]:
    """Reconstruct replay-format episode dicts from one task's LeRobot export.

    Returns ``(source_name, episode)`` pairs in dataset order. Each episode
    maps feature names to ``[T, ...]`` arrays: ``rgb_obs`` ``[T, cams, 3, H, W]``
    uint8 (camera order from the collector metadata), ``low_dim_obs``
    ``[T, D]``, ``action`` ``[T, A]`` (normalized outer action), ``reward``,
    ``discount``, ``demo``, ``is_expert``, ``event_progress`` ``[T, 1]`` plus
    any per-step extras the collector stored.

    With the default ``action_representation="absolute"`` the arrays are the
    collector's bit for bit. ``"upper_delta"`` derives a training view in
    memory; the dataset is never modified.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    task_dir = Path(task_dir)
    info = load_info(task_dir)
    version = str(info.get("codebase_version", ""))
    if not version.startswith("v3"):
        raise RuntimeError(
            f"LeRobot dataset {task_dir} is format {version or 'unknown'}; this "
            "reader is pinned to v3 (bigym-export-lerobot writes v3)"
        )
    features = info["features"]
    if any(v.get("dtype") == "video" for v in features.values()):
        raise RuntimeError(
            f"LeRobot dataset {task_dir} stores cameras as lossy video; demos "
            "must be the lossless PNG-mode export (bigym-export-lerobot "
            "without --videos)"
        )
    transition_keys = load_transition_keys(task_dir)
    metadata = load_task_metadata(task_dir)
    cams = list(metadata["task"]["camera_keys"])
    action_representation = validate_action_representation(action_representation)

    init_states_path = task_dir / "meta" / "episode_init_states.json"
    init_states = (
        json.loads(init_states_path.read_text())["episodes"]
        if init_states_path.exists()
        else {}
    )

    data_paths = sorted(task_dir.glob("data/chunk-*/file-*.parquet"))
    if not data_paths:
        raise RuntimeError(f"No data parquet files under {task_dir}")
    table = pa.concat_tables([pq.read_table(path) for path in data_paths])
    ep_index = table.column("episode_index").to_numpy()

    episodes: list[tuple[str, dict[str, np.ndarray]]] = []
    for ep in np.unique(ep_index):
        if max_episodes > 0 and len(episodes) >= max_episodes:
            break
        rows = table.slice(
            int(np.searchsorted(ep_index, ep)), int((ep_index == ep).sum())
        )
        num_rows = rows.num_rows
        episode: dict[str, np.ndarray] = {}
        rgb = None
        for name, spec in features.items():
            if name in INDEX_FEATURES:
                continue
            if name.startswith(IMAGE_PREFIX):
                cam_index = cams.index(name[len(IMAGE_PREFIX) :])
                col = rows.column(name).to_pylist()
                if rgb is None:
                    channels, height, width = spec["shape"]
                    rgb = np.empty(
                        (num_rows, len(cams), channels, height, width), dtype=np.uint8
                    )
                for t in range(num_rows):
                    img = np.asarray(Image.open(io.BytesIO(col[t]["bytes"])))
                    rgb[t, cam_index] = img.transpose(2, 0, 1)
                continue
            key = {"observation.state": "low_dim_obs"}.get(name, name)
            arr = np.stack([np.asarray(v) for v in rows.column(name).to_pylist()])
            episode[key] = arr.astype(spec["dtype"]).reshape((num_rows, *spec["shape"]))
        episode["rgb_obs"] = rgb  # ty: ignore[invalid-assignment]
        ep_extra = init_states.get(str(int(ep)), {})
        undo_transition_shift(
            episode,
            ep_extra.get("first_transition", {}),
            transition_keys,
            where=f"{task_dir} episode {int(ep)}",
        )
        for k, rec in ep_extra.get("arrays", {}).items():
            episode[k] = np.asarray(rec["data"], dtype=rec["dtype"]).reshape(
                rec["shape"]
            )
        episode = convert_episode_action_representation(
            episode,
            metadata,
            action_representation=action_representation,
            upper_delta_scale_rad=upper_delta_scale_rad,
        )
        episodes.append((ep_extra.get("source_file", f"episode_{int(ep)}"), episode))
    return episodes
