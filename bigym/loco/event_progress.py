"""Event progress: a [0, 1] task-progress label and the reward shaping on it.

The label is defined for the plate tasks (``EVENT_METRICS``): hand near the
plate, plate held, plate lifted, plate carried to the target. Other tasks
report NaN. :class:`EventProgress` is owned by :class:`~bigym.loco.env.BiGym`;
shaping is a training aid that never enters the success criterion.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from bigym.bigym_env import BiGymEnv
from bigym.loco.config import EnvConfig


def plate_event_metrics(env: Any) -> dict[str, float]:
    """Plate height, nearest-hand and nearest-target distances, and holding.

    Args:
        env: A plate task (``plates``; ``rack_target`` for move_plate).
    """
    plate = env.plates[0]
    plate_pos = np.asarray(plate.get_position(), dtype=np.float32).reshape(-1)
    robot = env.robot
    hand_distances = [
        float(
            np.linalg.norm(
                np.asarray(robot.get_hand_pos(side), dtype=np.float32).reshape(-1)
                - plate_pos
            )
        )
        for side in robot.grippers
    ]
    holding = any(
        bool(robot.is_gripper_holding_object(plate, side)) for side in robot.grippers
    )
    metrics = {"plate_z": float(plate_pos[2])}
    if hand_distances:
        metrics["hand_dist"] = float(min(hand_distances))
    metrics["holding"] = float(holding)

    # move_plate's destination rack; the dishwasher plate tasks have none.
    rack_target = getattr(env, "rack_target", None)
    if rack_target is not None and rack_target.sites:
        metrics["target_dist"] = float(
            min(
                np.linalg.norm(
                    np.asarray(env.data.bind(site).xpos, dtype=np.float32).reshape(-1)
                    - plate_pos
                )
                for site in rack_target.sites
            )
        )
    return metrics


# Task -> the metrics its progress label is computed from.
EVENT_METRICS: dict[str, Callable[[BiGymEnv], dict[str, float]]] = {
    task: plate_event_metrics
    for task in (
        "move_plate",
        "move_two_plates",
        "dishwasher_load_plates",
        "dishwasher_unload_plates",
        "dishwasher_unload_plates_long",
    )
}


class EventProgress:
    """Per-episode progress label (max-latched) and its shaping reward."""

    def __init__(self, task_name: str, config: EnvConfig):
        """Keep the env's progress and shaping settings for ``task_name``."""
        self.config = config
        self._metrics = EVENT_METRICS.get(task_name)
        self.initial_plate_z: Optional[float] = None
        self.initial_target_dist: Optional[float] = None
        self.held_once = False
        self.max = 0.0

    def metrics(self, env: BiGymEnv) -> dict[str, float]:
        """The task's event metrics; {} for tasks without a progress label."""
        if self._metrics is None:
            return {}
        return self._metrics(env)

    def reset(self, env: BiGymEnv) -> None:
        """Clear the episode's progress and capture its starting metrics."""
        self.initial_plate_z = None
        self.initial_target_dist = None
        self.max = 0.0
        # Lift/carry tiers are gated on having actually held the plate this
        # episode: post-reset settling can shrink plate-to-target distance by
        # a few cm, which would otherwise award free carry credit at step 0.
        self.held_once = False
        metrics = self.metrics(env)
        plate_z = metrics.get("plate_z")
        target_dist = metrics.get("target_dist")
        if plate_z is not None and np.isfinite(plate_z):
            self.initial_plate_z = float(plate_z)
        if target_dist is not None and np.isfinite(target_dist):
            self.initial_target_dist = float(target_dist)

    def update(self, env: BiGymEnv, succeeded: bool) -> np.float32:
        """Progress after a step; 1 when the task succeeded on it.

        NaN when progress is disabled or undefined for the task.
        """
        if not self.config.event_progress_enabled:
            return np.float32(np.nan)

        metrics = self.metrics(env)
        if not metrics:
            return np.float32(np.nan)

        if succeeded:
            self.max = 1.0
            return np.float32(1.0)

        progress = 0.0
        hand_dist = metrics.get("hand_dist")
        if hand_dist is not None and np.isfinite(hand_dist):
            # Early shaping label: hand approaching the plate, capped below grasp.
            progress = max(
                progress, 0.20 * float(np.clip((0.30 - hand_dist) / 0.30, 0.0, 1.0))
            )

        holding = float(metrics.get("holding", 0.0)) > 0.5
        if holding and not self.held_once:
            self.held_once = True
            # Re-capture lift/carry baselines at first grasp: transport
            # progress is measured from where the plate was when picked up,
            # so pre-grasp settling drift cannot award carry credit.
            plate_z_now = metrics.get("plate_z")
            if plate_z_now is not None and np.isfinite(plate_z_now):
                self.initial_plate_z = float(plate_z_now)
            target_dist_now = metrics.get("target_dist")
            if target_dist_now is not None and np.isfinite(target_dist_now):
                self.initial_target_dist = float(target_dist_now)
        if holding:
            progress = max(progress, 0.35)

        plate_z = metrics.get("plate_z")
        if (
            self.held_once
            and plate_z is not None
            and np.isfinite(plate_z)
            and self.initial_plate_z is not None
        ):
            lift_score = float(
                np.clip((plate_z - self.initial_plate_z) / 0.10, 0.0, 1.0)
            )
            if lift_score > 0.0:
                progress = max(progress, 0.35 + 0.25 * lift_score)

        target_dist = metrics.get("target_dist")
        if (
            self.held_once
            and target_dist is not None
            and np.isfinite(target_dist)
            and self.initial_target_dist is not None
        ):
            start = max(self.initial_target_dist, 0.06)
            goal = 0.05
            denom = max(start - goal, 1e-6)
            target_score = float(np.clip((start - target_dist) / denom, 0.0, 1.0))
            if target_score > 0.0:
                progress = max(progress, 0.60 + 0.35 * target_score)

        self.max = max(self.max, float(progress))
        return np.float32(self.max)

    def shaping(self, previous_progress: float, event_progress: np.float32) -> float:
        """Shaping reward for moving from ``previous_progress`` to ``event_progress``."""
        if not np.isfinite(event_progress):
            return 0.0

        prev_progress = (
            0.0 if not np.isfinite(previous_progress) else float(previous_progress)
        )
        curr_progress = float(event_progress)
        progress_delta = max(0.0, curr_progress - prev_progress)
        shaping_reward = self.config.event_reward_progress_scale * progress_delta

        # The label is float32 (0.35 rounds to 0.34999999), so the tier
        # crossings are tested in float32 on both sides.
        prev32 = np.float32(prev_progress)
        bonuses = (
            (0.35, self.config.event_reward_holding_bonus),
            (0.60, self.config.event_reward_lift_bonus),
            (0.85, self.config.event_reward_target_bonus),
        )
        for tier, bonus in bonuses:
            if prev32 < np.float32(tier) <= event_progress:
                shaping_reward += bonus
        return float(shaping_reward)
