"""Lower-body backends and a task as another package would define them.

Public bigym names only: tests/test_extension_api.py checks the imports and
reaches these objects by registration and by ``"module:ATTR"`` strings.
docs/extending.md includes parts of this file.
"""

import numpy as np

from bigym.envs.move_plates import MovePlateG1
from bigym.loco import BackendBinding, LowerBodyBase, velocity_spec
from bigym.loco.adapters.groot_wbc import (
    G1_GROOT_WBC_JOINT_NAMES,
    G1_GROOT_WBC_JOINT_PD,
    GROOT_WBC_G1,
    GrootWbcG1Config,
    GrootWbcG1Controller,
)
from bigym.loco.base import quat_rotate_inverse_wxyz
from bigym.loco.tasks import TaskSpec
from bigym.robots.configs.g1 import G1Dex1


class ReusedGrootBinding(BackendBinding):
    """The GR00T-WBC controller under another backend name."""

    robot_models = ("g1_dex1",)

    def robot_cls(self, robot_cls, config):
        """The G1 Dex1 with the GR00T-WBC joints actuated."""
        return GROOT_WBC_G1.robot_cls(robot_cls, config)

    def build_controller(self, env, config, *, control_dt):
        """The shipped GR00T-WBC policy pair."""
        return GrootWbcG1Controller(
            env=env,
            config=GrootWbcG1Config(
                control_dt=control_dt, enable_pitch_cmd=config.pitch_command
            ),
        )

    def configure_model(self, env):
        """The G1 contact solver options."""
        GROOT_WBC_G1.configure_model(env)


BINDING = ReusedGrootBinding()


# Hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll (left, then
# right), then waist yaw, roll, pitch: the order of G1_GROOT_WBC_JOINT_NAMES.
STANDING_POSE = np.array(
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2 + [0.0, 0.0, 0.0], dtype=np.float32
)


class HoldPoseController(LowerBodyBase):
    """Holds the legs and waist at a standing pose and ignores the command."""

    STATEFUL = {"cmd": "command", "height_cmd": "height_command"}

    def __init__(self, env, *, control_dt):
        joints = G1_GROOT_WBC_JOINT_NAMES
        self._env = env
        self.control_dt = control_dt
        self.controlled_joints = joints
        self.command_spec = velocity_spec(
            vx=(-1.0, 1.0), vy=(-1.0, 1.0), wz=(-1.0, 1.0), height=(0.4, 1.0)
        )
        self.velocity_clip = self.yaw_rate_clip = 1.0
        self.qpos_addresses, self.dof_addresses = self.build_joint_addresses(joints)
        low, high = self.build_joint_ranges(joints)
        self.controlled_range_low, self.controlled_range_high = low, high
        _, self.base_dof_addresses = self.build_joint_addresses(
            ("pelvis_x", "pelvis_y", "pelvis_z")
        )
        orientation = self.find_sensor("orientation", dim=4)
        gyro = self.find_sensor("angular-velocity", dim=3)
        assert orientation is not None and gyro is not None, "no pelvis IMU"
        self.orientation_sensor_address, self.gyro_sensor_address = orientation, gyro
        self.last_action = np.zeros(len(STANDING_POSE), dtype=np.float32)
        self.reset()

    def reset(self):
        """Clear the command and put the joints in the standing pose."""
        self.command = np.zeros(3, dtype=np.float32)
        self.height_command = 0.74
        self.apply_pose(
            self.qpos_addresses,
            self.dof_addresses,
            STANDING_POSE,
            G1_GROOT_WBC_JOINT_NAMES,
        )

    def step(self):
        """The standing pose, in controlled_joints order."""
        return STANDING_POSE.copy()

    def get_base_obs(self):
        """Pelvis linear velocity, angular velocity and gravity, pelvis frame."""
        data = self._env.data
        quat = data.sensordata[
            self.orientation_sensor_address : self.orientation_sensor_address + 4
        ]
        lin_vel = quat_rotate_inverse_wxyz(quat, data.qvel[self.base_dof_addresses])
        ang_vel = data.sensordata[
            self.gyro_sensor_address : self.gyro_sensor_address + 3
        ]
        gravity = quat_rotate_inverse_wxyz(quat, np.array([0.0, 0.0, -1.0]))
        return lin_vel, ang_vel.astype(np.float32), gravity


class HoldPoseBinding(BackendBinding):
    """HoldPoseController on the G1 Dex1, legs and waist on the GR00T-WBC gains."""

    robot_models = ("g1_dex1",)

    def robot_cls(self, robot_cls, config):
        """The G1 Dex1 with the controlled joints actuated."""
        assert issubclass(robot_cls, G1Dex1)
        return robot_cls.variant(
            actuated=G1_GROOT_WBC_JOINT_NAMES, joint_pd=G1_GROOT_WBC_JOINT_PD
        )

    def build_controller(self, env, config, *, control_dt):
        """A HoldPoseController stepping at the env's control rate."""
        return HoldPoseController(env, control_dt=control_dt)


HOLD_POSE = HoldPoseBinding()


class PreciseMovePlateG1(MovePlateG1):
    """move_plate with half the distance tolerance at the target rack."""

    _SUCCESSFUL_DIST = 0.025


TASK = TaskSpec(
    PreciseMovePlateG1,
    episode_length=17000,
    overrides={"controller": {"pitch_command": False}},
)
