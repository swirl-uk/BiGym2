"""Collection settings and the collector's environment configuration.

The environment is the task's official configuration (``bigym.loco.make``)
with three collection-only changes: a longer success hold, an episode cap
that never binds, and event progress recorded alongside each step.
Everything else here configures the operator's session, not the env.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import tyro

from bigym.loco.config import EnvConfig
from bigym.loco.objref import path_name
from bigym.loco.tasks import task_config

# The collection cap in recorded steps when the task budget is shorter:
# 2 min at 50 Hz. Training budgets are derived from the demos afterwards
# (bigym.loco.tasks.BUDGET_RULE), so collection must never be cut short.
COLLECT_MIN_OUTER_STEPS = 6000


@dataclass(frozen=True)
class CollectConfig:
    """One VR collection session (bigym-collect); each field is a flag."""

    task: str
    """Task name, e.g. move_plate."""
    out_dir: Path | None = None
    """Batch directory for the .npz episodes; None writes
    ./bigym_demos/<task>/<timestamp>. An existing batch is resumed."""
    episodes: int = 60
    """Stop after this many saved demos (60 is the published batch size)."""
    seed: int = 1
    """Reset seed of the first attempt; each attempt adds one."""
    collect_success_hold_seconds: float = 3.0
    """How long the success predicate must hold before an attempt succeeds.
    It is longer than the training hold, so every demo ends with a stay-still
    tail, and the training view is cut back to the env's hold."""
    episode_steps: int | None = None
    """Episode cap in recorded steps; None uses the task budget, but at
    least 6000 steps (2 min at 50 Hz)."""
    episode_seconds: float | None = None
    """Episode cap in seconds, instead of episode_steps."""
    max_steps: int | None = None
    """Optional cap on steps per attempt; cannot extend the episode cap."""
    keep_empty_session: bool = False
    """Keep the batch dir when no demo was saved."""
    keep_black_rgb: bool = False
    """Save episodes whose rgb_obs is all zero instead of rejecting them."""
    store_event_progress: bool = True
    """Record event_progress per step."""
    store_fullbody: bool = True
    """Record full-body diagnostics (qpos, qvel, controller commands) per
    step next to the training action."""
    yaw_mode: Literal["base", "none"] = "base"
    """Right stick X: base turns the robot (controller wz), none ignores it."""
    base_vx_scale: float = 0.35
    """Left stick Y to forward velocity (m/s)."""
    base_vy_scale: float = 0.25
    """Left stick X to lateral velocity (m/s)."""
    base_wz_scale: float = 0.5
    """Right stick X to turn rate (rad/s)."""
    base_z_scale: float = 0.004
    """Right stick Y to height-command change per step (m)."""
    height_cmd_max: float = 0.80
    """Cap on the integrated height command (m); above ~0.8 the controller
    locks the knees and stops stepping. 0 disables the cap."""
    pitch_rate: float = 0.8
    """Torso pitch change per second at full stick in pitch mode (rad/s)."""
    stick_deadzone: float = 0.15
    """Thumbstick deadzone."""
    base_cmd_slew: float | None = 0.7
    """Max change per second of the velocity commands; None maps the stick
    directly. Recorded actions carry the slewed commands."""
    recenter_height_offset: float = 0.0
    """Extra z offset when aligning the headset to the robot head camera."""
    vr_space_mode: Literal["follow_head", "fixed"] = "follow_head"
    """follow_head keeps the view attached to the robot head as it walks;
    fixed keeps the recentered world frame."""
    settle_view: Literal["curtain", "live"] = "curtain"
    """Headset view during the reset warmup: a still dark frame, or the
    moving scene for debugging."""
    hud: Literal["minimal", "full", "off"] = "minimal"
    """In-headset stats overlay."""
    resolution: Literal["lq", "mq", "hq"] = "lq"
    """Headset render resolution."""
    mujoco_gl: Literal["glfw", "egl", "osmesa"] = "glfw"
    """MUJOCO_GL, set before MuJoCo is imported."""
    openxr_log_level: Literal["trace", "debug", "info", "warn", "error"] = "error"
    """OpenXR runtime log level."""
    spectator: Literal["none", "mujoco", "viser", "both"] = "none"
    """Desktop views of the session: a MuJoCo window, a viser web page, or
    both. A MuJoCo window shares the headset stream's GPU and can stutter it."""
    spectator_hz: float = 15.0
    """viser update rate."""
    spectator_port: int = 8080
    """viser HTTP port."""
    spectator_gpu: int | None = None
    """EGL device for the viser camera renders (multi-GPU machines)."""
    frame_spike_ms: float = 20.0
    """Frames whose work exceeds this are logged to frame_timing.jsonl;
    0 disables timing."""
    operator: str | None = None
    """Who is collecting (default: $BIGYM_OPERATOR, then the login name)."""
    export_lerobot: bool = False
    """After the session, write the training view and export it as a
    LeRobot v3 dataset to <training view>_lerobot."""
    task_text: str | None = None
    """Language instruction for the LeRobot export; None looks the task up
    in bigym/loco/demos/task_instructions.yaml."""


def parse_cli(argv: Sequence[str] | None = None) -> CollectConfig:
    """Parse ``bigym-collect`` flags into a :class:`CollectConfig`."""
    args = list(sys.argv[1:] if argv is None else argv)
    return tyro.cli(CollectConfig, args=args, prog="bigym-collect")


def collection_env_config(config: CollectConfig) -> tuple[EnvConfig, EnvConfig]:
    """The env a session records in, and the configuration it trains for.

    Returns:
        ``(collection, training)``: training is the task's official
        configuration; collection differs from it only in the success hold,
        the episode cap and event progress.
    """
    training = task_config(config.task)
    dsr = int(training.demo_down_sample_rate)
    outer_hz = 500.0 / dsr
    if config.episode_steps is not None and config.episode_seconds is not None:
        raise ValueError("pass only one of episode_steps and episode_seconds")
    if config.episode_steps is not None:
        steps = int(config.episode_steps)
    elif config.episode_seconds is not None:
        steps = int(round(float(config.episode_seconds) * outer_hz))
    else:
        steps = max(int(training.episode_length or 0) // dsr, COLLECT_MIN_OUTER_STEPS)
    if steps <= 0:
        raise ValueError(f"episode steps must be positive, got {steps}")
    collection = training.override(
        episode_length=steps * dsr,
        success_hold_seconds=float(config.collect_success_hold_seconds),
        event_progress_enabled=bool(config.store_event_progress),
    )
    return collection, training


def default_out_dir(task: str) -> Path:
    """``./bigym_demos/<path_name(task)>/<timestamp>``."""
    return Path.cwd() / "bigym_demos" / path_name(task) / time.strftime("%Y%m%d_%H%M%S")
