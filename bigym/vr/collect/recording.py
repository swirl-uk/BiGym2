"""Action stats, data specs, and episode assembly for the replay format."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from dm_env import specs

from bigym.action_modes import PelvisDof
from bigym.loco.timestep import ExtendedTimeStepWrapper
from bigym.vr.collect.config import CollectConfig


def _collector_action_stats(
    env: ExtendedTimeStepWrapper, config: CollectConfig
) -> tuple[np.ndarray, np.ndarray]:
    outer = env.bigym
    layout = outer.wholebody_action_layout()
    action_min = layout["action_low"]
    action_max = layout["action_high"]
    base_dofs = list(outer.outer_action_floating_dofs)
    base_idx = {dof: i for i, dof in enumerate(base_dofs)}

    if PelvisDof.X in base_idx:
        i = int(base_idx[PelvisDof.X])
        action_min[i] = -float(config.base_vx_scale)
        action_max[i] = float(config.base_vx_scale)
    if PelvisDof.Y in base_idx:
        i = int(base_idx[PelvisDof.Y])
        action_min[i] = -float(config.base_vy_scale)
        action_max[i] = float(config.base_vy_scale)
    if PelvisDof.RZ in base_idx:
        i = int(base_idx[PelvisDof.RZ])
        if config.yaw_mode == "base":
            bound = float(config.base_wz_scale)
        else:
            # Keep a tiny non-zero range so raw zero maps to normalized zero.
            bound = 1e-6
        action_min[i] = -bound
        action_max[i] = bound
    return action_min, action_max


def set_collector_action_stats(
    env: ExtendedTimeStepWrapper, config: CollectConfig
) -> None:
    """Normalize actions over the collector's stick envelope, not the env bounds."""
    env.bigym.set_action_stats(*_collector_action_stats(env, config))


def episode_data_specs(env: ExtendedTimeStepWrapper, *, store_event_progress: bool):
    """The per-step fields of a stored episode, as dm_env specs."""
    items = [
        env.rgb_raw_observation_spec(),
        env.low_dim_raw_observation_spec(),
        env.action_spec(),
        specs.Array((1,), np.float32, "reward"),
        specs.Array((1,), np.float32, "discount"),
        specs.Array((1,), np.float32, "demo"),
        specs.Array((1,), np.float32, "is_expert"),
    ]
    if store_event_progress:
        items.append(specs.Array((1,), np.float32, "event_progress"))
    return tuple(items)


def episode_from_time_steps(time_steps: list[Any], data_specs) -> dict[str, np.ndarray]:
    """Stack an attempt's time steps into one episode dict (unstacked frames)."""
    episode = defaultdict(list)
    for time_step in time_steps:
        for spec in data_specs:
            value = time_step[spec.name]
            if spec.name == "low_dim_obs":
                low_dim = spec.shape[0]
                value = value[..., -low_dim:]
            elif spec.name == "rgb_obs":
                rgb_dim = spec.shape[1]
                value = value[:, -rgb_dim:]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            value = np.asarray(value, dtype=spec.dtype)
            if value.shape != spec.shape:
                raise ValueError(
                    f"{spec.name} shape mismatch: expected {spec.shape}, "
                    f"got {value.shape}"
                )
            episode[spec.name].append(value)

    out = {
        spec.name: np.asarray(episode[spec.name], dtype=spec.dtype)
        for spec in data_specs
    }
    out["demo"] = np.ones_like(out["demo"], dtype=np.float32)
    out["is_expert"] = np.ones_like(out["is_expert"], dtype=np.float32)
    return out
