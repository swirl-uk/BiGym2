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


def episode_rows(task_dir: Path) -> dict[int, tuple[Path, int, int]]:
    """Map each episode index of an export to ``(parquet file, first row, row count)``."""
    import pyarrow.parquet as pq

    paths = sorted(Path(task_dir).glob("data/chunk-*/file-*.parquet"))
    if not paths:
        raise RuntimeError(f"No data parquet files under {task_dir}")
    rows: dict[int, tuple[Path, int, int]] = {}
    for path in paths:
        index = pq.read_table(path, columns=["episode_index"]).column(0).to_numpy()
        for episode in np.unique(index):
            start = int(np.searchsorted(index, episode))
            rows[int(episode)] = (path, start, int((index == episode).sum()))
    return rows


class PngFrames:
    """One episode's camera frames, kept as PNG bytes and decoded when indexed.

    It stands in for the ``rgb_obs`` array of an episode, ``[T, cams, 3, H, W]``
    uint8, in a fraction of the memory. ``frames[t]`` decodes row ``t``;
    ``np.asarray(frames)`` decodes them all. :meth:`stacked` gives the
    frame-stacked view an env observes.

    The bytes of each camera sit in one contiguous array, so worker processes
    forked from the process that loaded them share them instead of copying.
    """

    def __init__(
        self,
        data: list[np.ndarray],
        offsets: list[np.ndarray],
        frame_shape: tuple[int, ...],
        stack: int = 1,
    ):
        """Wrap per-camera PNG bytes.

        Args:
            data: Per camera, the PNG files back to back, uint8.
            offsets: Per camera, ``T + 1`` int64 offsets into its ``data``;
                row ``t`` is ``data[offsets[t]:offsets[t + 1]]``.
            frame_shape: ``(3, H, W)`` of one decoded frame.
            stack: Frames stacked along the channel axis, oldest first, the
                first row padding the start.
        """
        self.data, self.offsets = data, offsets
        self.frame_shape = tuple(frame_shape)
        self.stack = int(stack)
        channels, height, width = self.frame_shape
        length = len(offsets[0]) - 1 if offsets else 0
        self.shape = (length, len(data), channels * self.stack, height, width)
        self.dtype = np.dtype(np.uint8)
        self.ndim = len(self.shape)

    def stacked(self, stack: int) -> "PngFrames":
        """The same frames, ``stack`` consecutive rows to an observation."""
        return PngFrames(self.data, self.offsets, self.frame_shape, stack)

    def __len__(self) -> int:
        """Number of rows."""
        return self.shape[0]

    def decode(self, row: int) -> np.ndarray:
        """Decode one recorded row, ``[cams, 3, H, W]``, without stacking."""
        from PIL import Image

        out = np.empty((len(self.data), *self.frame_shape), dtype=np.uint8)
        for cam, (data, offsets) in enumerate(
            zip(self.data, self.offsets, strict=True)
        ):
            png = data[offsets[row] : offsets[row + 1]]
            out[cam] = np.asarray(Image.open(io.BytesIO(png.tobytes()))).transpose(
                2, 0, 1
            )
        return out

    def __getitem__(self, index):
        """Index like the decoded array, decoding only the rows it selects.

        The first index picks rows (an integer, a slice or a 1-D integer
        array); any further indices apply to the decoded result, so
        ``frames[t, 0]`` is camera 0 of row ``t``. After an array of rows,
        only integers and slices may follow. Other indices raise IndexError.
        """
        rest: tuple = ()
        if isinstance(index, tuple):
            if not index:
                return np.asarray(self)
            index, rest = index[0], index[1:]
        if any(i is None or i is Ellipsis for i in (index, *rest)) or isinstance(
            index, (bool, np.bool_)
        ):
            raise IndexError("PngFrames takes integers, slices and integer arrays")
        if isinstance(index, (int, np.integer)):
            out = self._row(int(index))
            return out[rest] if rest else out
        if isinstance(index, slice):
            rows = range(*index.indices(len(self)))
        else:
            rows = np.asarray(index)
            if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
                raise IndexError(
                    "PngFrames picks rows with an integer, a slice or a 1-D "
                    f"integer array, not {rows.dtype} of shape {rows.shape}"
                )
            if not all(
                isinstance(i, slice)
                or (isinstance(i, (int, np.integer)) and not isinstance(i, bool))
                for i in rest
            ):
                raise IndexError(
                    "PngFrames takes only integers and slices after an array of rows"
                )
        out = np.empty((len(rows), *self.shape[1:]), dtype=np.uint8)
        for i, t in enumerate(rows):
            out[i] = self._row(int(t))
        return out[(slice(None), *rest)] if rest else out

    def _row(self, t: int) -> np.ndarray:
        """Decode and stack one row, ``[cams, 3 * stack, H, W]``."""
        if t < 0:
            t += len(self)
        if not 0 <= t < len(self):
            raise IndexError(f"row {t} of {len(self)}")
        rows = [max(t - (self.stack - 1 - j), 0) for j in range(self.stack)]
        decoded = {row: self.decode(row) for row in set(rows)}
        if self.stack == 1:
            return decoded[t]
        return np.concatenate([decoded[row] for row in rows], axis=1)

    def __array__(self, dtype=None, copy=None):
        """Decode every row."""
        out = np.empty(self.shape, dtype=np.uint8)
        for t in range(len(self)):
            out[t] = self._row(t)
        return out if dtype is None else out.astype(dtype)


def _png_bytes(column) -> tuple[np.ndarray, np.ndarray]:
    """An image column's PNG bytes as ``(data, offsets)``, copied out of the table."""
    import pyarrow as pa

    struct = column.combine_chunks()
    binary = struct.flatten()[struct.type.get_field_index("bytes")]
    binary = binary.cast(pa.large_binary())
    _, offsets, data = binary.buffers()
    offsets = np.frombuffer(offsets, dtype=np.int64)
    offsets = offsets[binary.offset : binary.offset + len(binary) + 1]
    data = np.frombuffer(data, dtype=np.uint8)[offsets[0] : offsets[-1]].copy()
    return data, offsets - offsets[0]


def load_episodes(
    task_dir: Path,
    max_episodes: int = -1,
    *,
    action_representation: str = ABSOLUTE,
    upper_delta_scale_rad: float | None = None,
    decode_images: bool = True,
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
    memory; the dataset is never modified. With ``decode_images=False``,
    ``rgb_obs`` is a :class:`PngFrames` that decodes rows when indexed.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

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

    rows = episode_rows(task_dir)
    wanted = sorted(rows)[: max_episodes if max_episodes > 0 else None]
    episodes: list[tuple[str, dict[str, np.ndarray]]] = []
    # Episodes are stored in order, so each file is read once.
    path, table = None, pa.table({})
    for ep in wanted:
        if rows[ep][0] != path:
            path, table = rows[ep][0], pa.table({})  # free the previous file first
            table = pq.read_table(path)
        start, num_rows = rows[ep][1:]
        ep_rows = table.slice(start, num_rows)
        episode: dict[str, np.ndarray] = {}
        png: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        frame_shape = None
        for name, spec in features.items():
            if name in INDEX_FEATURES:
                continue
            if name.startswith(IMAGE_PREFIX):
                png[cams.index(name[len(IMAGE_PREFIX) :])] = _png_bytes(
                    ep_rows.column(name)
                )
                frame_shape = tuple(spec["shape"])
                continue
            key = {"observation.state": "low_dim_obs"}.get(name, name)
            arr = np.stack([np.asarray(v) for v in ep_rows.column(name).to_pylist()])
            episode[key] = arr.astype(spec["dtype"]).reshape((num_rows, *spec["shape"]))
        if frame_shape is not None:
            frames = PngFrames(
                [png[i][0] for i in range(len(png))],
                [png[i][1] for i in range(len(png))],
                frame_shape,
            )
            episode["rgb_obs"] = np.asarray(frames) if decode_images else frames  # ty: ignore[invalid-assignment]
        else:
            episode["rgb_obs"] = None  # ty: ignore[invalid-assignment]
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
