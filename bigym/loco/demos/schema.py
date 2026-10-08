"""Native demo schema v1 (frozen).

A *native demo* is one teleoperated episode recorded against the
controller-in-the-loop env (bigym.loco.env), stored as one ``.npz`` per
episode plus one ``metadata.json`` per collection directory. This module
freezes the contract those files satisfy so training/replay code can
validate instead of assuming.

Episode npz keys
----------------

Per-step arrays, first axis = outer control steps (T):

- ``rgb_obs``    (T, num_cameras, 3, H, W) uint8
- ``low_dim_obs`` (T, low_dim) float32 — raw (unstacked) layout, see
  ``low_dim_component_slices()`` of the env for the component map
- ``action``     (T, action_dim) float32 — the NORMALIZED outer action in
  [-1, 1] (commands + upper-body targets + grippers), exactly what the
  collector fed env.step(). Decode to physical units via the per-dim
  ``action_stats`` min/max recorded in metadata.json (the collection
  envelope; ``env.get_demos()`` adopts it at load). The RAW per-step values
  live in the diagnostic ``raw_outer_action`` array, not here.
- ``reward`` / ``discount`` / ``demo`` / ``is_expert`` (T, 1) float32
- ``event_progress`` (T, 1) float32 — optional

Per-episode scalars / snapshots:

- ``seed``             (1,) int64 — env reset seed
- ``pre_engage_steps`` (1,) int64 — steps between reset and VR engage
- engage-state snapshot: ``init_qpos``,
  ``init_qvel``, ``init_ctrl``, ``init_qacc_warmstart`` (and ``init_act``
  when present) — MuJoCo state at the engage moment
- ``lb_state.*`` — the flattened lower-body state snapshot
  (env.get_lowerbody_state(): ``ctrl.*`` controller fields per
  LowerBodyController.get_state(), ``env.*`` env-side fields). Restoring
  MuJoCo state + ``lb_state.*`` reproduces the episode bit-exactly on the
  Newton-pinned scenes.

metadata.json
-------------

Written once per collection dir by the collector. Required keys:

- ``format``: ``"bigym_replay_npz"``
- ``pipeline_version``: collector pipeline version (string, e.g.
  ``"2026-08-26-success-hold-training-view-v1"``)
- ``task``: the flat env settings (must include ``episode_length`` and
  ``demo_down_sample_rate`` — training MUST match these), plus the
  collection and training success holds
- ``lowerbody_policy``: the lower-body controller settings, flat
- optional ``env_config``: the full ``EnvConfig`` the batch was recorded in;
  ``EnvConfig.from_metadata`` reads it, and needs it
- ``reset_semantics``: init keyframe / pelvis z / warmup steps
- ``action_semantics``: robot model, backend, base_action_mode, stick map
- optional ``action_stats``: per-dim min/max for [-1,1] rescaling
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

NATIVE_DEMO_SCHEMA_VERSION = 1

REQUIRED_STEP_KEYS: tuple[str, ...] = (
    "rgb_obs",
    "low_dim_obs",
    "action",
    "reward",
    "discount",
    "demo",
    "is_expert",
)

ENGAGE_SNAPSHOT_KEYS: tuple[str, ...] = (
    "init_qpos",
    "init_qvel",
    "init_ctrl",
    "init_qacc_warmstart",
)

# The "full state per frame plus action" npz layout that demo batches use.
BATCH_FORMAT = "bigym_replay_npz"

REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "format",
    "pipeline_version",
    "task",
    "lowerbody_policy",
    "reset_semantics",
    "action_semantics",
)


def validate_episode(episode: Mapping[str, np.ndarray]) -> list[str]:
    """Return a list of schema violations (empty = valid)."""
    issues: list[str] = []
    for key in REQUIRED_STEP_KEYS:
        if key not in episode:
            issues.append(f"missing per-step key {key!r}")
    if issues:
        return issues

    action = np.asarray(episode["action"])
    if action.ndim != 2:
        return [f"action must be (T, action_dim); got shape {action.shape}"]
    if action.dtype != np.float32:
        issues.append(f"action must be float32; got {action.dtype}")

    steps = int(action.shape[0])
    for key in REQUIRED_STEP_KEYS:
        arr = np.asarray(episode[key])
        if arr.ndim == 0 or arr.shape[0] != steps:
            issues.append(
                f"{key!r} has shape {arr.shape}, expected {steps} steps (from action)"
            )
    rgb = np.asarray(episode["rgb_obs"])
    if rgb.dtype != np.uint8 or rgb.ndim != 5:
        issues.append(
            f"rgb_obs must be uint8 (T, cams, 3, H, W); got {rgb.dtype} {rgb.shape}"
        )
    low_dim = np.asarray(episode["low_dim_obs"])
    if low_dim.ndim != 2 or low_dim.dtype != np.float32:
        issues.append(
            f"low_dim_obs must be float32 (T, low_dim); "
            f"got {low_dim.dtype} {low_dim.shape}"
        )
    for key in ("reward", "discount", "demo", "is_expert"):
        arr = np.asarray(episode[key])
        if arr.ndim != 2 or arr.shape[1] != 1:
            issues.append(f"{key!r} must be (T, 1); got {arr.shape}")
        elif arr.dtype != np.float32:
            issues.append(f"{key!r} must be float32; got {arr.dtype}")
    for key in ("seed", "pre_engage_steps"):
        if key not in episode:
            issues.append(f"missing per-episode key {key!r}")

    has_snapshot = [key for key in ENGAGE_SNAPSHOT_KEYS if key in episode]
    if has_snapshot and len(has_snapshot) != len(ENGAGE_SNAPSHOT_KEYS):
        missing = set(ENGAGE_SNAPSHOT_KEYS) - set(has_snapshot)
        issues.append(f"partial engage snapshot: missing {sorted(missing)}")
    if any(key.startswith("lb_state.") for key in episode) and not has_snapshot:
        issues.append("lb_state.* present but MuJoCo engage snapshot missing")
    return issues


def validate_metadata(metadata: Mapping[str, Any]) -> list[str]:
    """Return a list of metadata violations (empty = valid)."""
    issues = [
        f"missing metadata key {key!r}"
        for key in REQUIRED_METADATA_KEYS
        if key not in metadata
    ]
    task = metadata.get("task")
    if isinstance(task, Mapping):
        for key in ("episode_length", "demo_down_sample_rate"):
            if key not in task:
                issues.append(f"metadata task block missing {key!r}")
    elif task is not None:
        issues.append("metadata 'task' must be a mapping (task preset)")
    return issues
