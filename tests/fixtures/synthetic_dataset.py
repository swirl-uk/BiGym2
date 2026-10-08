"""A tiny LeRobot v3 task export, laid out the way ``bigym-export-lerobot`` writes it.

Episodes are replay-format dicts (row t holds obs[t] and the transition that
produced it). The export is lossless PNG mode with action alignment v2: frame
k carries obs[k] plus the transition executed FROM it, the final frame repeats
the last transition, and replay row 0 of every transition column goes to the
``first_transition`` sidecar.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from bigym.loco.demos.schema import BATCH_FORMAT

CAMERAS = ("head", "right_wrist", "left_wrist")
TRANSITION_KEYS = (
    "action",
    "reward",
    "discount",
    "demo",
    "is_expert",
    "event_progress",
)
INDEX_FEATURES = ("timestamp", "frame_index", "episode_index", "index", "task_index")
IMAGE_SIZE = 8


def make_episode(
    rng: np.random.Generator,
    length: int,
    action_dim: int,
    state_dim: int,
    nq: int | None = None,
) -> dict[str, Any]:
    """One replay-format episode of random cameras, state and actions.

    Args:
        rng: Source of the random values.
        length: Number of rows.
        action_dim: Width of ``action``.
        state_dim: Width of ``low_dim_obs``.
        nq: When given, also the ``raw_outer_action`` and ``full_qpos``
            columns a sandbox reads.
    """
    reward = np.zeros((length, 1), np.float32)
    reward[-1] = 1.0
    discount = np.ones((length, 1), np.float32)
    discount[-1] = 0.0
    size = (length, len(CAMERAS), 3, IMAGE_SIZE, IMAGE_SIZE)
    episode = {
        "rgb_obs": rng.integers(0, 256, size, dtype=np.uint8),
        "low_dim_obs": rng.standard_normal((length, state_dim)).astype(np.float32),
        "action": rng.uniform(-1, 1, (length, action_dim)).astype(np.float32),
    }
    if nq is not None:
        episode["raw_outer_action"] = rng.uniform(-1, 1, (length, action_dim)).astype(
            np.float32
        )
        episode["full_qpos"] = rng.standard_normal((length, nq)).astype(np.float64)
    episode.update(
        reward=reward,
        discount=discount,
        demo=np.ones((length, 1), np.float32),
        is_expert=np.ones((length, 1), np.float32),
        event_progress=np.linspace(0, 1, length, dtype=np.float32)[:, None],
    )
    return episode


def png_bytes(chw: np.ndarray) -> bytes:
    """PNG-encode one channel-first uint8 image."""
    buffer = io.BytesIO()
    Image.fromarray(chw.transpose(1, 2, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def write_export(
    root: Path,
    episodes: list[dict],
    task_name: str = "reach_target_single",
    task_fields: dict | None = None,
) -> Path:
    """Write ``episodes`` as one task's export under ``root`` and return it.

    Every column the episodes carry is written: ``rgb_obs`` as one PNG image
    column per camera, ``low_dim_obs`` as ``observation.state``, the
    transition keys shifted by one frame, anything else per frame as is. An
    episode's integer ``seed`` goes to the per-episode sidecar.

    Args:
        root: The task directory to write.
        episodes: Replay-format episodes, each with a ``seed``.
        task_name: The task the metadata names.
        task_fields: More settings for the metadata's ``task`` block.
    """
    first = episodes[0]
    columns = [key for key in first if key not in ("rgb_obs", "low_dim_obs", "seed")]
    transition_keys = [key for key in TRANSITION_KEYS if key in first]
    cameras = CAMERAS if "rgb_obs" in first else ()

    features: dict[str, dict] = {}
    if "low_dim_obs" in first:
        state = first["low_dim_obs"]
        features["observation.state"] = {
            "dtype": str(state.dtype),
            "shape": list(state.shape[1:]),
        }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "image",
            "shape": list(first["rgb_obs"].shape[2:]),
        }
    for key in columns:
        features[key] = {
            "dtype": str(first[key].dtype),
            "shape": list(first[key].shape[1:]),
        }
    for key in INDEX_FEATURES:
        features[key] = {
            "dtype": "float32" if key == "timestamp" else "int64",
            "shape": [1],
        }

    rows: dict[str, list] = {name: [] for name in features}
    init_states: dict[str, dict] = {}
    global_index = 0
    for episode_index, episode in enumerate(episodes):
        length = len(episode[columns[0]])
        for k in range(length):
            transition_row = min(k + 1, length - 1)
            if "low_dim_obs" in episode:
                rows["observation.state"].append(episode["low_dim_obs"][k].tolist())
            for cam_index, cam in enumerate(cameras):
                image = png_bytes(episode["rgb_obs"][k, cam_index])
                rows[f"observation.images.{cam}"].append({"bytes": image, "path": None})
            for key in columns:
                row = transition_row if key in transition_keys else k
                rows[key].append(episode[key][row].tolist())
            rows["timestamp"].append(k / 50.0)
            rows["frame_index"].append(k)
            rows["episode_index"].append(episode_index)
            rows["index"].append(global_index)
            rows["task_index"].append(0)
            global_index += 1
        init_states[str(episode_index)] = {
            "source_file": f"episode_{episode_index}.npz",
            "first_transition": {
                key: {
                    "dtype": str(episode[key].dtype),
                    "shape": list(episode[key][0].shape),
                    "data": episode[key][0].tolist(),
                }
                for key in transition_keys
            },
            "arrays": {
                "seed": {"dtype": "int64", "shape": [], "data": int(episode["seed"])}
            },
        }

    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(exist_ok=True)
    pq.write_table(pa.table(rows), root / "data" / "chunk-000" / "file-000.parquet")
    info = {
        "codebase_version": "v3.0",
        "fps": 50,
        "total_episodes": len(episodes),
        "robot_type": "g1_dex1",
        "features": features,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "alignment.json").write_text(
        json.dumps({"action_alignment_version": 2, "transition_keys": transition_keys})
    )
    (root / "meta" / "episode_init_states.json").write_text(
        json.dumps({"episodes": init_states})
    )
    metadata: dict = {
        "format": BATCH_FORMAT,
        "control_step_seconds": 0.02,
        "task": {
            "task_name": task_name,
            "robot_model": "g1_dex1",
            "camera_keys": list(cameras),
            "camera_shape": [IMAGE_SIZE, IMAGE_SIZE],
            **(task_fields or {}),
        },
    }
    if "action" in first:
        action_dim = first["action"].shape[1]
        metadata["action_stats"] = {
            "min": [-1.0] * action_dim,
            "max": [1.0] * action_dim,
        }
    (root / "metadata.json").write_text(json.dumps(metadata))
    return root
