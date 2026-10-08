"""bigym-collect settings: CollectConfig, its command line and the collection env."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from bigym.loco.tasks import task_config
from bigym.vr.collect.config import (
    COLLECT_MIN_OUTER_STEPS,
    CollectConfig,
    collection_env_config,
    parse_cli,
)


def test_collection_env_departs_from_the_official_one_only_where_it_must():
    collection, training = collection_env_config(CollectConfig(task="pick_box"))
    assert training == task_config("pick_box")
    assert collection.differences(training) == {
        "episode_length": (44000, COLLECT_MIN_OUTER_STEPS * 10),
        "success_hold_seconds": (1.0, 3.0),
        "event_progress_enabled": (False, True),
    }
    # A long budget is kept as the cap.
    collection, _ = collection_env_config(CollectConfig(task="stack_blocks"))
    assert collection.episode_length == 87500


def test_explicit_episode_caps():
    collection, _ = collection_env_config(
        CollectConfig(task="move_plate", episode_seconds=30.0)
    )
    assert collection.episode_length == 1500 * 10
    collection, _ = collection_env_config(
        CollectConfig(task="move_plate", episode_steps=100)
    )
    assert collection.episode_length == 1000
    with pytest.raises(ValueError, match="only one"):
        collection_env_config(
            CollectConfig(task="move_plate", episode_steps=1, episode_seconds=1.0)
        )


def test_command_line_flags_set_the_session():
    config = parse_cli(
        [
            "--task",
            "flip_cup",
            "--episodes",
            "5",
            "--base-cmd-slew",
            "0.5",
            "--out-dir",
            "batches/flip_cup",
            "--no-store-fullbody",
        ]
    )
    assert config == CollectConfig(
        task="flip_cup",
        episodes=5,
        base_cmd_slew=0.5,
        out_dir=Path("batches/flip_cup"),
        store_fullbody=False,
    )
    assert parse_cli(["--task", "move_plate"]) == CollectConfig(task="move_plate")
    assert parse_cli(
        [
            "--task",
            "my_lab.tasks:SPEC",
            "--export-lerobot",
            "--task-text",
            "Move the plate.",
        ]
    ) == CollectConfig(
        task="my_lab.tasks:SPEC", export_lerobot=True, task_text="Move the plate."
    )
    with pytest.raises(SystemExit):
        parse_cli(["--task", "move_plate", "--config", "session"])


def test_help_lists_the_collection_settings():
    out = subprocess.run(
        [sys.executable, "-m", "bigym.vr.collect", "--help"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for flag in (
        "--task",
        "--collect-success-hold-seconds",
        "--base-cmd-slew",
        "--yaw-mode",
    ):
        assert flag in out
