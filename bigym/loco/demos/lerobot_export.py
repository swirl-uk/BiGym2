# /// script
# requires-python = ">=3.12"
# dependencies = ["lerobot[dataset]>=0.6", "numpy", "pyyaml", "tyro>=0.9"]
# ///
# Layer LeRobot onto the project env without changing its lock:
# uv run --no-sync --with "lerobot[dataset]>=0.6" bigym-export-lerobot ...
"""Export a replay-format npz demo batch to a LeRobot v3 dataset.

One-copy policy: LeRobot is the distributed demo format. This converter
carries EVERYTHING the npz holds — cameras, low-dim
obs, outer action, per-step fullbody extras — as first-class LeRobot features,
and the per-episode engage snapshots (init_qpos/init_qvel/init_ctrl/
init_qacc_warmstart, lb_state.*, seed, pre_engage_steps) as a sidecar
`meta/episode_init_states.json` in the same dataset, so deterministic replay
and arbitrary-resolution re-rendering stay possible from this single copy.

Usage:
  uv run --no-sync --with "lerobot[dataset]>=0.6" bigym-export-lerobot \
      --demo-dir <batch-dir> \
      --repo-id  YOUR_HF_USER/bigym-g1-move-plate-hold1s \
      --root     <output-dir> \
      --task-text "Move the plate from one dish rack to the other." \
      [--push]   # requires `hf auth login`; dataset is created private
"""

from __future__ import annotations

import copy
import json
import pathlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import yaml

PER_STEP_EXTRAS = [
    # Diagnostic only: seconds since collector start. Older batches lack the
    # key -- see the presence filter below.
    ("wall_clock", "float64"),
    ("full_qpos", "float64"),
    ("full_qvel", "float64"),
    ("raw_outer_action", "float32"),
    ("expanded_action", "float32"),
    ("lowerbody_action", "float32"),
    ("leg_joint_targets", "float32"),
    ("torso_target", "float32"),
    ("lowerbody_command", "float32"),
    ("height_command", "float32"),
    ("reward", "float32"),
    ("event_progress", "float32"),
    ("discount", "float32"),
    ("demo", "float32"),
    ("is_expert", "float32"),
]
EPISODE_LEVEL_PREFIXES = ("init_", "lb_state.")
EPISODE_LEVEL_KEYS = ("seed", "pre_engage_steps")

# Alignment v2. Source npz convention: index t holds the state
# AFTER executing action[t] (index 0 is the pre-engage slot). LeRobot frames
# must pair an observation with the action executed FROM it, so every
# transition-scoped feature is stored shifted: frame k <- npz index k+1.
# The final frame repeats index T-1 ("keep holding"; dropped on
# reconstruction) and npz index 0 goes to the episode_init_states sidecar
# as first_transition. State-scoped extras (full_qpos/full_qvel) stay at k.
ACTION_ALIGNMENT_VERSION = 2
TRANSITION_KEYS = [
    "action",
    "raw_outer_action",
    "expanded_action",
    "lowerbody_action",
    "leg_joint_targets",
    "torso_target",
    "lowerbody_command",
    "height_command",
    "reward",
    "event_progress",
    "discount",
    "demo",
    "is_expert",
]


def sanitize_published_metadata(meta: dict) -> dict:
    """Drop collector-machine absolute paths from the published metadata.

    ``metadata.json`` records where the collector's checkouts and batches sat
    on disk. That is useful locally -- you can walk to the source batch -- but
    a published copy carries someone's home directory to every downloader and
    tells them nothing: what identifies the controller is the
    ``lowerbody_weights`` sha256 recorded beside it, which is machine
    independent. Training and evaluation never read these fields; the demo
    viewer does, and falls back to a local checkout when the recorded path is
    absent.

    ``collection.operator`` and ``host`` (and the same two keys inside
    ``sessions``) are dropped: a login name and a hostname identify a
    machine, not a demonstration.
    """
    out = copy.deepcopy(meta)
    # Launch-configuration leftovers that describe nothing about the data:
    # unused checkpoint paths, the settings block of a backend that never ran
    # (present in the raw source batches), config placeholders, local batch
    # directories, and the machine and account the collector ran under.
    dropped_keys = ("checkpoint_path", "mjlab_g1")

    def walk(node, key=""):
        if isinstance(node, dict):
            if key == "lowerbody_policy":
                for k in dropped_keys:
                    node.pop(k, None)
            if key == "substrate_fingerprint":
                node.pop("lowerbody_checkpoint", None)
            if key in ("collection", "sessions_item"):
                node.pop("host", None)
                node.pop("operator", None)
            for k, v in list(node.items()):
                if isinstance(v, str):
                    if k == "repo_root":
                        node[k] = "<groot-checkout>"
                    elif k in ("source_batch", "source_dir", "demo_dir"):
                        node[k] = pathlib.PurePath(v).name
                    elif "${hydra:" in v:
                        node[k] = pathlib.PurePath(v).name
                    elif re.match(r"^/(home|data|mnt)/", v):
                        node[k] = "<local-path>/" + pathlib.PurePath(v).name
                elif isinstance(v, list) and k in ("sources", "sessions"):
                    if k == "sources":
                        node[k] = [
                            "/".join(pathlib.PurePath(x).parts[-2:])
                            if isinstance(x, str)
                            else x
                            for x in v
                        ]
                    else:
                        for x in v:
                            walk(x, "sessions_item")
                else:
                    walk(v, k)
        elif isinstance(node, list):
            for v in node:
                walk(v, key)

    walk(out)
    return out


@dataclass
class ExportConfig:
    """bigym-export-lerobot settings."""

    demo_dir: str
    """Replay-format npz batch to export."""
    repo_id: str
    """LeRobot repo id, e.g. YOUR_HF_USER/bigym-g1-move-plate."""
    root: str
    """Output directory of the dataset."""
    task_text: str | None = None
    """Language instruction override; default resolves the demo's task_name in
    bigym/loco/demos/task_instructions.yaml (`_g1` suffix stripped)."""
    push: bool = False
    """Push the dataset to the Hub (created private; needs `hf auth login`)."""
    max_episodes: int = -1
    """Export at most this many episodes; -1 exports all."""
    fps: int | None = None
    """Override the LeRobot timestamp rate; only needed for batches whose
    metadata lacks control_step_seconds."""
    videos: bool = False
    """Encode cameras as mp4 (LOSSY, for visualization variants only); default
    is lossless embedded PNG, the canonical one-copy format."""


def main():
    """Export replay-format demos using command-line options."""
    # bigym-collect runs this file as a standalone script (its inline metadata
    # above), so nothing here may import bigym.
    args = tyro.cli(ExportConfig, description=__doc__)

    from lerobot.datasets.lerobot_dataset import (  # ty: ignore[unresolved-import]
        LeRobotDataset,
    )

    src = Path(args.demo_dir)
    meta = json.loads((src / "metadata.json").read_text())
    task_text = args.task_text
    if task_text is None:
        table_path = Path(__file__).with_name("task_instructions.yaml")
        table = yaml.safe_load(table_path.read_text())
        task_name = str(meta.get("task", {}).get("task_name", ""))
        base = task_name[:-3] if task_name.endswith("_g1") else task_name
        task_text = table.get(task_name) or table.get(base)
        if not task_text:
            raise SystemExit(
                f"no instruction for task '{task_name}' in {table_path}; "
                "add an entry or pass --task-text"
            )
        print(f"instruction[{task_name}]: {task_text}")
    # LeRobot's timestamp column is fps-synthesized, so a wrong fps silently
    # stretches the whole time axis. Never default it quietly.
    step_s = meta.get("control_step_seconds")
    if args.fps is not None:
        fps = int(args.fps)
        print(f"fps overridden to {fps} (metadata says {step_s})")
    elif step_s is None:
        fps = 50
        print(
            "WARNING: metadata has no control_step_seconds; assuming 50 Hz. "
            "If the batch was recorded at another rate, pass --fps or every "
            "LeRobot timestamp is stretched."
        )
    else:
        fps = int(round(1.0 / float(step_s)))
    cams = meta["task"]["camera_keys"]
    paths = sorted(src.glob("*.npz"))
    if args.max_episodes > 0:
        paths = paths[: args.max_episodes]
    probe = np.load(paths[0])
    n_cam, c, h, w = probe["rgb_obs"].shape[1:]
    assert n_cam == len(cams), (n_cam, cams)

    # Per-step extras are versioned: a batch only carries the ones its
    # collector knew about. Intersect over EVERY file, not just the first --
    # merged batches can mix collector eras, and probing only paths[0] would
    # KeyError partway through a long conversion.
    common = set(probe.files)
    for path in paths[1:]:
        with np.load(path) as z:
            common &= set(z.files)
    extras = [(k, dt) for k, dt in PER_STEP_EXTRAS if k in common]
    absent = [k for k, _ in PER_STEP_EXTRAS if k not in common]
    if absent:
        print(
            f"note: batch lacks per-step {absent} in at least one episode; "
            f"exporting without them"
        )

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (probe["low_dim_obs"].shape[1],),
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (probe["action"].shape[1],),
            "names": None,
        },
    }
    for cam in cams:
        features[f"observation.images.{cam}"] = {
            "dtype": "video" if args.videos else "image",
            "shape": (c, h, w),
            "names": ["channels", "height", "width"],
        }
    for key, dtype in extras:
        features[key.replace(".", "_")] = {
            "dtype": dtype,
            "shape": (probe[key].shape[1],),
            "names": None,
        }

    root = Path(args.root)
    if root.exists():
        raise SystemExit(f"refusing to overwrite existing {root}")
    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type=str(meta.get("action_semantics", {}).get("robot_model", "g1_dex1")),
        use_videos=bool(args.videos),
        # async PNG encoding; without this add_frame encodes synchronously
        # and dominates conversion wall-clock (~10 min per 60-episode batch)
        image_writer_threads=4 * len(cams),
    )

    init_states = {}
    for ep_idx, path in enumerate(paths):
        d = np.load(path)
        # Materialize every per-step array ONCE. `np.load` on an .npz returns a
        # lazy NpzFile: each `d[key]` re-reads and re-inflates the whole member.
        # Indexing inside the frame loop would re-inflate rgb_obs several
        # times per frame; hoisting it out is a ~100x speedup.
        rgb_obs = d["rgb_obs"]
        low_dim_obs = d["low_dim_obs"]
        action = d["action"]
        step_arrays = {key: d[key] for key, _ in extras}
        T = action.shape[0]
        for t in range(T):
            tt = min(t + 1, T - 1)  # transition executed FROM this frame
            frame = {
                "observation.state": low_dim_obs[t].astype(np.float32),
                "action": action[tt].astype(np.float32),
                "task": task_text,
            }
            for i, cam in enumerate(cams):
                frame[f"observation.images.{cam}"] = rgb_obs[t, i]
            for key, dtype in extras:
                src_t = tt if key in TRANSITION_KEYS else t
                frame[key.replace(".", "_")] = step_arrays[key][src_t].astype(dtype)
            ds.add_frame(frame)
        ds.save_episode()
        # dtype/shape recorded explicitly: JSON alone cannot round-trip numpy
        # dtypes, and the loader must reconstruct these arrays bit-exactly.
        init_states[str(ep_idx)] = {
            "source_file": path.name,
            "arrays": {
                k: {
                    "dtype": str(np.asarray(d[k]).dtype),
                    "shape": list(np.asarray(d[k]).shape),
                    "data": np.asarray(d[k]).tolist(),
                }
                for k in d.files
                if k in EPISODE_LEVEL_KEYS or k.startswith(EPISODE_LEVEL_PREFIXES)
            },
            # npz index 0 of every shifted feature; needed to reconstruct the
            # source episode bit-exactly (see TRANSITION_KEYS note above).
            "first_transition": {
                k: {
                    "dtype": str(np.asarray(d[k]).dtype),
                    "shape": list(np.asarray(d[k][0]).shape),
                    "data": np.asarray(d[k][0]).tolist(),
                }
                for k in TRANSITION_KEYS
                if k in d.files
            },
        }
        print(f"episode {ep_idx}: {T} frames <- {path.name}")

    ds.finalize()
    (root / "meta" / "alignment.json").write_text(
        json.dumps(
            {
                "action_alignment_version": ACTION_ALIGNMENT_VERSION,
                "transition_keys": TRANSITION_KEYS,
                "doc": (
                    "Frame k pairs the frame's observation with the "
                    "transition executed FROM it (source npz index k+1) for "
                    "every transition_keys feature; the final frame repeats "
                    "the last real transition and source index 0 lives in "
                    "episode_init_states.json first_transition. Datasets "
                    "without this file are v1: features stored at source "
                    "index k verbatim (one-frame hindsight misalignment for "
                    "standard LeRobot consumers)."
                ),
            }
        )
    )
    (root / "meta" / "episode_init_states.json").write_text(
        json.dumps(
            {
                "doc": (
                    "Per-episode engage snapshots for deterministic replay "
                    "(replay-format schema; see source metadata.json). "
                    "Sidecar: not part of the LeRobot schema."
                ),
                "episodes": init_states,
            }
        )
    )
    # Root-level copy keeps validate_native_demo_launch working unchanged on
    # LeRobot demo dirs (it globs <dir>/metadata.json). Both copies are
    # published, so both drop the collector's absolute paths.
    published_meta = sanitize_published_metadata(
        json.loads((src / "metadata.json").read_text())
    )
    blob = json.dumps(published_meta, indent=1, ensure_ascii=False)
    (root / "metadata.json").write_text(blob)
    (root / "meta" / "source_metadata.json").write_text(blob)
    print(f"\nwrote LeRobot dataset: {root} ({len(paths)} episodes, fps={fps})")
    if args.push:
        ds.push_to_hub(private=True)
        print(f"pushed to hub: {args.repo_id} (private)")


if __name__ == "__main__":
    main()
