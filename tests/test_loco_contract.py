"""Unit tests for the bigym.loco contract layer (pure functions only)."""

import numpy as np
import pytest

from bigym.loco.base import (
    LowerBodyBase,
    mujoco_basename,
    quat_rotate_inverse_wxyz,
    yaw_from_qpos,
    yaw_from_quat_wxyz,
)
from bigym.loco.command import CommandField, CommandKind, CommandSpec, velocity_spec


def test_command_field_clip():
    f = CommandField("vx", "m/s", -0.8, 1.2)
    assert f.clip(2.0) == 1.2
    assert f.clip(-2.0) == -0.8
    assert f.clip(0.3) == 0.3


def test_velocity_spec_layout():
    spec = velocity_spec(
        vx=(-0.8, 1.2),
        vy=(-0.5, 0.5),
        wz=(-0.8, 0.8),
        height=(0.28, 0.78),
        height_rate=0.3,
        torso_pitch=(-0.2, 0.45),
        torso_pitch_rate=0.6,
    )
    assert spec.kind is CommandKind.VELOCITY
    assert spec.dim == 5
    assert spec.names == ("vx", "vy", "wz", "height", "torso_pitch")
    assert spec.field("height").rate == 0.3
    assert spec.field("vx").rate is None
    assert spec.has("torso_pitch")
    assert not spec.has("ee_pose")
    with pytest.raises(KeyError):
        spec.field("nope")


def test_command_spec_rejects_duplicates():
    with pytest.raises(ValueError):
        CommandSpec(
            kind=CommandKind.VELOCITY,
            fields=(
                CommandField("vx", "m/s", -1, 1),
                CommandField("vx", "m/s", -1, 1),
            ),
        )


def test_mujoco_basename():
    assert mujoco_basename("g1_29dof_with_dex1_1/left_knee_joint") == "left_knee_joint"
    assert mujoco_basename("left_knee_joint") == "left_knee_joint"
    assert mujoco_basename(None) == ""
    assert mujoco_basename("") == ""


def test_quat_rotate_inverse_identity():
    v = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    out = quat_rotate_inverse_wxyz(q, v)
    np.testing.assert_allclose(out, v, atol=1e-7)
    assert out.dtype == np.float32


def test_quat_rotate_inverse_ninety_deg_roll():
    # 90deg roll about +x: world -z maps to body -y under inverse rotation.
    q = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0], dtype=np.float64)
    out = quat_rotate_inverse_wxyz(q, np.array([0.0, 0.0, -1.0]))
    np.testing.assert_allclose(out, [0.0, -1.0, 0.0], atol=1e-6)


def test_yaw_helpers():
    yaw = 0.7
    q = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    assert yaw_from_quat_wxyz(q) == pytest.approx(yaw, abs=1e-7)
    qpos_free = np.concatenate([[0, 0, 0.8], q])
    assert yaw_from_qpos(qpos_free) == pytest.approx(yaw, abs=1e-7)
    # slide/hinge floating base stack: yaw is the last entry
    assert yaw_from_qpos(np.array([1.0, 2.0, 0.9, 0.3])) == pytest.approx(0.3)
    assert yaw_from_qpos(np.array([])) == 0.0


class _FakeController(LowerBodyBase):
    STATEFUL = {
        "cmd": "command",
        "height_cmd": "height_command",
        "last_action": "last_action",
        "engaged": "engaged",
    }

    def __init__(self):
        self.command = np.zeros(3, dtype=np.float32)
        self.height_command = 0.78
        self.last_action: np.ndarray = np.zeros(4, dtype=np.float32)
        self.engaged = False
        self.velocity_clip = 1.0
        self.yaw_rate_clip = 0.5


def test_declarative_state_roundtrip():
    ctrl = _FakeController()
    ctrl.command = np.array([0.3, -0.1, 0.2], dtype=np.float32)
    ctrl.height_command = 0.6
    ctrl.last_action = np.array([1, 2, 3, 4], dtype=np.float32)
    ctrl.engaged = True

    snap = ctrl.get_state()
    assert set(snap) == {"cmd", "height_cmd", "last_action", "engaged"}
    assert snap["height_cmd"].shape == ()

    other = _FakeController()
    other.set_state(snap)
    np.testing.assert_array_equal(other.command, ctrl.command)
    assert other.height_command == pytest.approx(0.6)
    np.testing.assert_array_equal(other.last_action, ctrl.last_action)
    assert other.engaged is True
    assert isinstance(other.height_command, float)


def test_declarative_state_missing_key_is_loud():
    ctrl = _FakeController()
    snap = ctrl.get_state()
    del snap["last_action"]
    with pytest.raises(KeyError):
        _FakeController().set_state(snap)


def test_default_set_command_clips():
    ctrl = _FakeController()
    ctrl.set_command(2.0, -2.0, 2.0, height=0.5)
    np.testing.assert_allclose(ctrl.command, [1.0, -1.0, 0.5])
    assert ctrl.get_height_command() == pytest.approx(0.5)
    np.testing.assert_array_equal(ctrl.get_command(), ctrl.command)
