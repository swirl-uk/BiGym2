"""EnvConfig resolution, task registry and the metadata round trip (no simulation)."""

from __future__ import annotations

import dataclasses
import json
import typing
from typing import Any

import pytest

from bigym.loco.adapters import resolve_backend_name
from bigym.loco.config import (
    ControllerConfig,
    EnvConfig,
    WholeBodyConfig,
    affects_results,
    conformance_violations,
    resolve_config,
)
from bigym.loco.tasks import (
    DATA_DERIVED_BUDGET_TASKS,
    TASKS,
    all_task_names,
    budget_provenance,
    task_config,
    task_spec,
)
from bigym.robots.configs import ROBOT_MODELS

NO_PITCH = {
    "drawer_top_close",
    "drawer_top_open",
    "move_plate",
    "reach_target_dual",
    "reach_target_multi_modal",
    "reach_target_single",
}
REACH = {"reach_target_dual", "reach_target_multi_modal", "reach_target_single"}
TOP_DRAWER = {"drawer_top_close", "drawer_top_open"}

# The flat task and lower-body blocks of a demonstration batch, as the
# collector wrote them (unused keys included).
FLAT_BLOCKS = {
    "task": {
        "action_mode": "absolute",
        "camera_keys": ["head", "right_wrist", "left_wrist"],
        "camera_shape": [84, 84],
        "collect_success_hold_seconds": 3.0,
        "control_pelvis": True,
        "demo_down_sample_rate": 10,
        "enable_all_floating_dof": True,
        "episode_length": 60000,
        "initialization_profile": None,
        "lowerbody_policy": {"route_pelvis_rz_to_torso": True},
        "normalize_low_dim_obs": True,
        "num_demos": 60,
        "reach_tolerance": 0.05,
        "render_mode": "rgb_array",
        "robot_model": "g1_dex1",
        "state_keys": [
            "proprioception",
            "proprioception_grippers",
            "proprioception_floating_base",
        ],
        "success_hold_seconds": 1.0,
        "task_name": "reach_target_dual",
        "training_success_hold_seconds": 1.0,
    },
    "lowerbody_policy": {
        "action_clip": 5.0,
        "backend": "groot_wbc_g1",
        "base_action_mode": "lowerbody_cmd",
        "cmd_clip": 1.0,
        "default_height_cmd": 0.98,
        "enabled": True,
        "groot_wbc_g1": {
            "default_height_cmd": 0.74,
            "passive_base_tilt": True,
            "repo_root": "<groot-checkout>",
        },
        "init_foot_backward_offset": 0.0,
        "init_pelvis_z": 0.92,
        "init_stance": "keyframe",
        "reset_warmup_steps": 200,
        "route_pelvis_rz_to_torso": False,
        "support_base_with_legs": True,
        "use_height_cmd": True,
        "wz_clip": 1.0,
    },
}


def test_defaults_are_the_official_values():
    config = EnvConfig()
    assert config.robot_model == "g1_dex1"
    assert config.demo_down_sample_rate == 10
    assert config.success_hold_seconds == 1.0
    assert config.reach_tolerance is None
    assert config.initialization_profile == "upstream"
    assert config.enable_all_floating_dof is True
    controller = config.controller
    assert controller is not None
    assert controller.backend == "groot_wbc_g1"
    assert controller.base_action_mode == "lowerbody_cmd"
    assert controller.reset_warmup_steps == 200
    assert controller.init_stance == "keyframe"
    assert controller.deterministic_reset is True
    assert controller.pitch_command is True


def test_registry_holds_every_task_with_its_exceptions():
    assert len(TASKS) == 40
    assert all_task_names() == tuple(sorted(TASKS))
    for name in TASKS:
        config = task_config(name)
        assert config.episode_length == task_spec(name).episode_length >= 2000
        assert config.controller is not None
        assert config.controller.pitch_command is (name not in NO_PITCH)
        assert config.reach_tolerance == (0.05 if name in REACH else None)
        assert config.initialization_profile == (
            "g1_id_v1" if name in TOP_DRAWER else "upstream"
        )
        diff = set(config.differences(EnvConfig(episode_length=None)))
        expected = {"episode_length"}
        expected |= {"controller.pitch_command"} if name in NO_PITCH else set()
        expected |= {"reach_tolerance"} if name in REACH else set()
        expected |= {"initialization_profile"} if name in TOP_DRAWER else set()
        assert diff == expected, name
    assert task_config("stack_blocks").episode_length == 87500
    assert task_config("drawers_close_all").episode_length == 5000


def test_budget_provenance():
    assert len(DATA_DERIVED_BUDGET_TASKS) == 20
    assert budget_provenance("pick_box") == "data_derived"
    assert budget_provenance("store_box") == "upstream_placeholder"


def test_unknown_names_are_rejected_clearly():
    with pytest.raises(KeyError, match="unknown task 'move_plate_h1'.*move_plate"):
        task_config("move_plate_h1")
    with pytest.raises(ValueError, match="Unknown lowerbody backend 'homie'"):
        resolve_backend_name("homie")
    with pytest.raises(ValueError, match="unknown EnvConfig field 'episode_len'"):
        EnvConfig().override(episode_len=10)
    with pytest.raises(ValueError, match=r"pass controller=\{'cmd_clip'"):
        EnvConfig().override(cmd_clip=0.5)
    with pytest.raises(ValueError, match="unknown ControllerConfig field"):
        EnvConfig().override(controller={"pitch": False})


def test_override_merges_nested_configs_and_coerces_values():
    config = EnvConfig().override(
        controller={"reset_warmup_steps": 0},
        camera_keys=["head"],
        camera_shape=[96.0, 128],
        success_hold_seconds=2,
    )
    assert config.controller == ControllerConfig(reset_warmup_steps=0)
    assert config.camera_keys == ("head",)
    assert config.camera_shape == (96, 128)
    assert isinstance(config.success_hold_seconds, float)
    assert EnvConfig().override(controller=None).controller is None
    assert EnvConfig().override(wholebody={}).wholebody == WholeBodyConfig()
    # Merging into an absent controller starts from the defaults.
    floating = EnvConfig(controller=None)
    restored = floating.override(controller={"pitch_command": False})
    assert restored.controller == ControllerConfig(pitch_command=False)


def test_resolve_config_applies_overrides_over_the_official_config():
    config = resolve_config(
        "move_plate",
        None,
        {
            "camera_keys": ["head"],
            "episode_length": 4000,
            "controller": {"reset_warmup_steps": 0},
            "frame_stack": 2,
        },
    )
    official = task_config("move_plate")
    assert config.differences(official) == {
        "episode_length": (17000, 4000),
        "controller.reset_warmup_steps": (200, 0),
        "camera_keys": (official.camera_keys, ("head",)),
        "frame_stack": (1, 2),
    }
    # The official per-task differences survive a controller override.
    assert config.controller is not None
    assert config.controller.pitch_command is False
    assert resolve_config("move_plate", None, {"controller": None}).controller is None
    with pytest.raises(TypeError, match="EnvConfig or None"):
        not_a_config: Any = "official"
        resolve_config("move_plate", not_a_config)


def test_resolve_config_fills_the_budget():
    assert resolve_config("pick_box").episode_length == 44000
    assert resolve_config("pick_box", EnvConfig()).episode_length == 44000
    assert resolve_config("pick_box", None, {"episode_length": 10}).episode_length == 10
    with pytest.raises(TypeError, match="EnvConfig"):
        not_a_config: Any = {"episode_length": 10}
        resolve_config("pick_box", not_a_config)


def test_conformance_exempts_presentation_fields_only():
    official = task_config("flip_cup")
    exempt = official.override(
        render_mode="human",
        frame_stack=3,
        normalize_low_dim_obs=True,
        event_progress_enabled=True,
        action_representation="upper_delta",
        upper_delta_scale_rad=0.1,
    )
    assert conformance_violations(exempt, official) == []
    changed = official.override(
        camera_keys=("head",), controller={"deterministic_reset": False}
    )
    violations = conformance_violations(changed, official)
    assert [v.split(":")[0] for v in violations] == [
        "controller.deterministic_reset",
        "camera_keys",
    ]
    assert not affects_results("frame_stack")
    assert affects_results("camera_keys")
    assert affects_results("event_reward_shaping_enabled")


@pytest.mark.parametrize("name", sorted(TASKS))
def test_metadata_round_trip_of_every_official_config(name):
    config = task_config(name)
    blocks = json.loads(json.dumps(config.to_metadata(name)))
    assert blocks["task"]["task_name"] == name
    assert EnvConfig.from_metadata(blocks) == config


def test_metadata_round_trip_of_research_configs():
    for config in (
        task_config("move_plate").override(controller=None),
        task_config("pick_box").override(
            controller={
                "height_cmd_max": 0.8,
                "pitch_cmd_min": -0.1,
                "init_pelvis_z": 0.72,
                "reset_warmup_steps": None,
                "model_path": "a.onnx,b.onnx",
            },
            camera_keys=(),
            wholebody={"preserve_leg_proprio": True},
        ),
    ):
        blocks = json.loads(json.dumps(config.to_metadata("t")))
        assert EnvConfig.from_metadata(blocks) == config
    floating = task_config("move_plate").override(controller=None)
    assert floating.to_metadata("move_plate")["lowerbody_policy"] == {}


def test_metadata_without_env_config_is_rejected():
    with pytest.raises(ValueError, match="env_config"):
        EnvConfig.from_metadata(FLAT_BLOCKS)


def test_env_config_block_wins_over_the_flat_blocks():
    config = task_config("flip_cup").override(frame_stack=2)
    metadata = dict(FLAT_BLOCKS, env_config=config.to_dict())
    assert EnvConfig.from_metadata(metadata) == config


def test_configs_are_frozen_and_hashable():
    config = EnvConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.robot_model = "h1"  # ty: ignore[invalid-assignment]
    assert hash(config) == hash(EnvConfig())


def test_fixed_choices_are_validated_on_construction():
    with pytest.raises(ValueError, match="EnvConfig.action_mode must be one of"):
        EnvConfig(action_mode="relative")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="EnvConfig.initialization_profile"):
        EnvConfig().override(initialization_profile="foo")
    with pytest.raises(ValueError, match="ControllerConfig.base_action_mode"):
        EnvConfig().override(controller={"base_action_mode": "foo"})
    with pytest.raises(ValueError, match="WholeBodyConfig.mode"):
        WholeBodyConfig(mode="residual_leg")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="success_hold_seconds must be >= 0"):
        EnvConfig(success_hold_seconds=-1.0)
    with pytest.raises(ValueError, match="requires action_mode='absolute'"):
        EnvConfig(action_mode="delta", action_representation="upper_delta")


def test_robot_model_choices_are_the_robot_registry():
    choices = typing.get_args(typing.get_type_hints(EnvConfig)["robot_model"])
    assert set(choices) == set(ROBOT_MODELS)
