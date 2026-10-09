"""A task's published demonstrations, checked against and shaped for an env.

``BiGym.load_demo_episodes`` / ``get_demos`` call these: :func:`load_task_episodes`
fetches and decodes a task's episodes and returns the action statistics the
env adopts, :func:`check_compatibility` refuses a dataset recorded under a
different embodiment or layout, and :func:`episode_to_timesteps` turns one
stacked episode into :class:`~bigym.loco.timestep.ExtendedTimeStep` rows.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from dm_env import StepType

from bigym.loco.timestep import ExtendedTimeStep


def load_task_episodes(
    task_name: str,
    num_demos: int,
    *,
    action_representation: str,
    upper_delta_scale_rad: Optional[float],
    outer_action_low: Optional[np.ndarray],
    outer_action_high: Optional[np.ndarray],
    check_compatibility: Callable[[dict[str, Any], dict[str, Any]], None],
) -> tuple[list[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Return ``(episodes, action_stats)`` for ``task_name`` (fetched on demand).

    ``check_compatibility(metadata, info)`` runs before any episode is
    decoded. The dataset's ``action_stats`` must lie inside the env's outer
    action bounds. ``num_demos < 0`` loads every episode.
    """
    # Deferred: huggingface_hub and the export reader load only when an env
    # reads demonstrations.
    from bigym.loco.demos import dataset, hub

    task_dir = hub.task_dir(task_name)
    metadata = dataset.load_task_metadata(task_dir)
    check_compatibility(metadata, dataset.load_info(task_dir))
    episodes = dataset.load_episodes(
        task_dir,
        max_episodes=int(num_demos),
        action_representation=action_representation,
        upper_delta_scale_rad=upper_delta_scale_rad,
    )
    stats = metadata.get("action_stats") or {}
    action_min = np.asarray(stats.get("min"), dtype=np.float32).reshape(-1)
    action_max = np.asarray(stats.get("max"), dtype=np.float32).reshape(-1)
    if outer_action_low is not None and outer_action_high is not None:
        low = np.asarray(outer_action_low, dtype=np.float32)
        high = np.asarray(outer_action_high, dtype=np.float32)
        if action_min.shape != low.shape or not (
            np.all(action_min >= low - 1e-6) and np.all(action_max <= high + 1e-6)
        ):
            raise ValueError(
                f"demo action_stats of task {task_name!r} fall outside "
                "this env's outer action bounds; the dataset was recorded "
                "on a different action layout or command clip"
            )
    return [episode for _, episode in episodes], {"min": action_min, "max": action_max}


def check_compatibility(
    metadata: dict[str, Any],
    info: dict[str, Any],
    *,
    task_name: str,
    robot_model: str,
    lowerbody_backend: str,
    demo_down_sample_rate: int,
    camera_keys: tuple[str, ...],
    camera_shape: tuple[int, ...],
    low_dim_dim: int,
    action_dim: int,
) -> None:
    """Raise ValueError listing every way the dataset differs from the env."""
    task = metadata.get("task", {})
    semantics = metadata.get("action_semantics", {})
    problems = []

    def expect(label, got, want):
        if got != want:
            problems.append(f"{label}: dataset {got!r} vs env {want!r}")

    expect(
        "robot_model",
        str(info.get("robot_type", task.get("robot_model"))),
        robot_model,
    )
    expect(
        "lowerbody backend",
        str(semantics.get("lowerbody_backend")),
        lowerbody_backend,
    )
    expect(
        "demo_down_sample_rate",
        int(task.get("demo_down_sample_rate", -1)),
        int(demo_down_sample_rate),
    )
    expect("camera_keys", list(task.get("camera_keys", [])), list(camera_keys))
    expect(
        "camera_shape",
        tuple(task.get("camera_shape", ())),
        tuple(camera_shape),
    )
    features = info.get("features", {})
    state = features.get("observation.state", {}).get("shape", [None])[0]
    expect("low_dim_obs dim", state, low_dim_dim)
    action = features.get("action", {}).get("shape", [None])[0]
    expect("action dim", action, action_dim)
    if problems:
        raise ValueError(
            f"demonstrations for task {task_name!r} do not match this "
            "env:\n  - " + "\n  - ".join(problems) + "\nBuild the env with "
            "bigym.loco.make / make_gym (the official configuration the demos "
            "were recorded under), or re-render the dataset for another "
            "camera setup with bigym-rerender-lerobot."
        )


def renormalize_actions(
    action: np.ndarray,
    from_stats: dict[str, np.ndarray],
    to_stats: dict[str, np.ndarray],
    normalized: np.ndarray,
) -> np.ndarray:
    """Re-express ``[-1, 1]`` actions from one action envelope in another.

    Only the ``normalized`` slots (a boolean mask) are converted. Row 0, the
    reset row that no step executes, is clipped instead of checked.
    """
    action = np.asarray(action, dtype=np.float32)
    from_min, from_max, to_min, to_max = (
        np.asarray(stats[key], dtype=np.float64)
        for stats, key in (
            (from_stats, "min"),
            (from_stats, "max"),
            (to_stats, "min"),
            (to_stats, "max"),
        )
    )
    raw = (action.astype(np.float64) + 1.0) / 2.0 * (from_max - from_min + 1e-8)
    moved = (raw + from_min - to_min) / (to_max - to_min + 1e-8) * 2.0 - 1.0
    moved = moved[:, normalized]
    if np.any(np.abs(moved[1:]) > 1.0 + 1e-5):
        raise ValueError(
            "the demonstrations command raw actions outside this env's "
            "action_stats; widen them with set_action_stats"
        )
    out = action.copy()
    out[:, normalized] = np.clip(moved, -1.0, 1.0)
    return out


def low_dim_obs_stats(episodes: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Per-dimension mean/std of the episodes' raw low-dim observations."""
    low_dim = np.concatenate(
        [np.asarray(ep["low_dim_obs"], dtype=np.float32) for ep in episodes], axis=0
    )
    return {"mean": np.mean(low_dim, 0), "std": np.std(low_dim, 0)}


def episode_to_timesteps(
    episode: dict[str, np.ndarray],
    rgb: np.ndarray,
    low_dim: np.ndarray,
    successful: bool,
) -> list[ExtendedTimeStep]:
    """One episode as ExtendedTimeSteps, from its stacked observations.

    ``rgb``/``low_dim`` are the env-stacked observations
    (``BiGym.demo_observations``); ``successful`` fills the ``demo`` flag
    when the episode carries no ``demo`` column.
    """
    actions = np.asarray(episode["action"], dtype=np.float32)
    rewards = np.asarray(episode["reward"], dtype=np.float32).reshape(-1)
    discounts = np.asarray(episode["discount"], dtype=np.float32).reshape(-1)
    length = len(actions)

    def column(key, default):
        if key in episode:
            return np.asarray(episode[key], dtype=np.float32).reshape(-1)
        return np.full((length,), default, dtype=np.float32)

    demo_flags = column("demo", float(successful))
    is_expert = column("is_expert", 1.0)
    progress = column("event_progress", np.nan)
    timesteps = []
    for t in range(length):
        if t == 0:
            step_type = StepType.FIRST
        elif t == length - 1:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID
        timesteps.append(
            ExtendedTimeStep(
                step_type=step_type,
                reward=float(rewards[t]),
                discount=float(discounts[t]),
                rgb_obs=rgb[t],
                low_dim_obs=low_dim[t],
                action=actions[t],
                demo=float(demo_flags[t]),
                is_expert=float(is_expert[t]),
                event_progress=float(progress[t]),
            )
        )
    return timesteps
