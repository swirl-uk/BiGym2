"""NVIDIA GR00T Whole-Body Control (decoupled / Gear WBC) G1 backend.

Wraps the official frozen ``G1GearWbcPolicy`` (Balance + Walk ONNX pair)
behind the BiGym lower-body contract. The policy runtime is vendored in
``bigym.loco.adapters._groot`` (see its PROVENANCE.md) and the two ONNX
weight files ship as package assets, byte-identical to the official
``sim2mujoco/resources/robots/g1/policy`` files, so the backend needs only
onnxruntime. ``GROOT_WBC_G1`` binds the controller to the G1 Dex1 robot.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from bigym.loco.adapters.binding import BackendBinding
from bigym.loco.base import (
    LowerBodyBase,
    yaw_from_quat_wxyz,
)
from bigym.loco.base import (
    quat_rotate_inverse_wxyz as _quat_rotate_inverse_wxyz,
)
from bigym.loco.command import velocity_spec

if TYPE_CHECKING:
    from bigym.bigym_env import BiGymEnv
    from bigym.loco.config import ControllerConfig
    from bigym.robots.robot import Robot

G1_GROOT_WBC_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)

G1_GROOT_WBC_BODY_JOINT_NAMES: tuple[str, ...] = G1_GROOT_WBC_JOINT_NAMES + (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_GROOT_WBC_JOINT_PD: dict[str, tuple[float, float]] = {
    name: (kp, kd)
    for name, kp, kd in zip(
        G1_GROOT_WBC_JOINT_NAMES,
        (150, 150, 150, 200, 40, 40, 150, 150, 150, 200, 40, 40, 250, 250, 250),
        (2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5),
        strict=True,
    )
}

# The official MuJoCo deployment asset applies this motor inertia to every
# controlled joint. Without it the same PD targets are under-damped in BiGym.
G1_GROOT_WBC_JOINT_ARMATURE: dict[str, float] = {
    name: 0.01 for name in G1_GROOT_WBC_JOINT_NAMES
}

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
DEFAULT_GROOT_MODEL_PATHS: tuple[str, str] = (
    str(ASSETS_DIR / "GR00T-WholeBodyControl-Balance.onnx"),
    str(ASSETS_DIR / "GR00T-WholeBodyControl-Walk.onnx"),
)


@dataclass(frozen=True)
class GrootWbcG1Config:
    """Launch config for the frozen GR00T-WBC G1 policy pair."""

    # Ignored: the runtime is vendored. Still accepted because recorded
    # configs carry it.
    repo_root: Optional[Path] = None
    control_dt: float = 0.02
    default_height_cmd: float = 0.74
    cmd_clip: float = 1.0
    wz_clip: float = 1.0
    # Comma-separated "balance,walk" ONNX pair (absolute paths, or paths
    # relative to the current directory).
    model_path: str = ",".join(DEFAULT_GROOT_MODEL_PATHS)
    # Torso pitch command: the official policy consumes a torso
    # roll/pitch/yaw command (obs[4:7]). Opt-in: enabling it adds the
    # PelvisDof.RY slot to the outer action (+1 dim, recorded in demo
    # metadata / substrate fingerprint, launch validation blocks mixing).
    # + = lean forward. Bounds are the benchmark clip, not a training range
    # (the official repo does not publish one); the robot stays balanced
    # across the range at low and default height. At 0.8 standing, the
    # shoulders move ~15 cm forward and ~8 cm down.
    enable_pitch_cmd: bool = False
    pitch_cmd_min: float = -0.2
    # Waist pitch saturates at ~30 deg for commands >= 0.8; above that the
    # pelvis tilts instead, so 0.8 is the useful max.
    pitch_cmd_max: float = 0.8


class G1JointGroups:
    """The official policy only needs these two index groups."""

    @staticmethod
    def get_joint_group_indices(group: str) -> np.ndarray:
        """Indices of the ``body`` (29) or ``lower_body`` (15) joints."""
        if group == "body":
            return np.arange(29, dtype=np.int64)
        if group == "lower_body":
            return np.arange(15, dtype=np.int64)
        raise KeyError(group)


def policy_obs_dim(policy: Any) -> int:
    """Per-step obs dim from the official config (num_obs / history_len).

    516 / 6 = 86 for the packaged Balance/Walk policies; deriving it keeps
    the snapshot format in lockstep with the vendored runtime.
    """
    return int(policy.config["num_obs"]) // int(policy.config["obs_history_len"])


def restore_policy_state(policy: Any, state: dict[str, np.ndarray]) -> None:
    """Write the ``policy_*`` entries of a controller snapshot into a policy.

    Args:
        policy: An official ``G1GearWbcPolicy``.
        state: A :meth:`GrootWbcG1Controller.get_state` snapshot.
    """
    policy.action = (
        np.asarray(state["policy_action"], dtype=np.float32).reshape(15).copy()
    )
    policy.target_dof_pos = (
        np.asarray(state["policy_target_dof_pos"], dtype=np.float32).reshape(15).copy()
    )
    policy.cmd = np.asarray(state["policy_cmd"], dtype=np.float32).reshape(3).copy()
    policy.height_cmd = float(np.asarray(state["policy_height_cmd"]).reshape(-1)[0])
    rpy = np.asarray(state["policy_rpy_cmd"], dtype=np.float32).reshape(3)
    policy.roll_cmd, policy.pitch_cmd, policy.yaw_cmd = map(float, rpy)
    policy.gait_indices = (
        np.asarray(state["policy_gait_indices"], dtype=np.float32).reshape(1).copy()
    )
    history = np.asarray(state["policy_obs_history"], dtype=np.float32).reshape(
        -1, policy_obs_dim(policy)
    )
    policy.obs_history = collections.deque(
        (row.copy() for row in history),
        maxlen=int(policy.config["obs_history_len"]),
    )


class GrootWbcG1Controller(LowerBodyBase):
    """Expose the official frozen GR00T-WBC policy through the loco contract."""

    # Measured settle: 178-180 steps. The official configuration
    # (ControllerConfig.reset_warmup_steps) pins 200.
    recommended_reset_warmup_steps = 180

    def _set_pitch_command(self, pitch_cmd: float) -> None:
        """Torso pitch (rad, + = lean forward) into the official policy's pitch_cmd."""
        if not self._pitch_enabled:
            return
        value = float(
            np.clip(pitch_cmd, self._config.pitch_cmd_min, self._config.pitch_cmd_max)
        )
        self._pitch_cmd = value
        self.policy.pitch_cmd = value

    def weight_files(self) -> tuple[str, ...]:
        """Absolute paths of the Balance/Walk ONNX files that were loaded."""
        return tuple(
            str(Path(part.strip()).expanduser().resolve())
            for part in str(self._config.model_path).split(",")
            if part.strip()
        )

    def __init__(self, env: Any, config: GrootWbcG1Config) -> None:
        """Build the official policy and resolve joint/IMU addresses.

        Loads the Balance/Walk ONNX pair named by ``config.model_path``,
        puts the policy in teleop-command mode, and caches the body and
        lower-body qpos/qvel addresses, the joint ranges and the default
        joint targets.
        """
        self._env = env
        self._config = config
        self.control_dt = float(config.control_dt)
        self.velocity_clip = float(config.cmd_clip)
        self.yaw_rate_clip = float(config.wz_clip)
        self.controlled_joints = G1_GROOT_WBC_JOINT_NAMES
        # Command bounds: benchmark-level clips; the official teleop stack
        # feeds the same [vx, vy, wz] + height channels. The official repo
        # does not publish training command ranges, so the height range
        # uses the env default (0.4, 1.0). The env derives its Z-slot bounds
        # from this spec, so changing it here changes the outer action space
        # (= substrate bump).
        self.command_spec = velocity_spec(
            vx=(-float(config.cmd_clip), float(config.cmd_clip)),
            vy=(-float(config.cmd_clip), float(config.cmd_clip)),
            wz=(-float(config.wz_clip), float(config.wz_clip)),
            height=(0.4, 1.0),
            torso_pitch=(
                (float(config.pitch_cmd_min), float(config.pitch_cmd_max))
                if config.enable_pitch_cmd
                else None
            ),
        )
        self._pitch_enabled = bool(config.enable_pitch_cmd)
        self._pitch_cmd = 0.0
        # Deferred: the backend registry imports this module with bigym.loco,
        # and onnxruntime loads only when a controller is built.
        from bigym.loco.adapters._groot import POLICY_CONFIG_PATH, G1GearWbcPolicy

        for path in self.weight_files():
            if not Path(path).is_file():
                raise FileNotFoundError(f"Missing GR00T-WBC ONNX weights: {path}")
        self.policy = G1GearWbcPolicy(
            robot_model=G1JointGroups(),
            config=str(POLICY_CONFIG_PATH),
            model_path=",".join(self.weight_files()),
        )
        self.policy.use_policy_action = True
        self.policy.set_use_teleop_policy_cmd(True)

        self.body_qpos_addresses, self.body_dof_addresses = self.build_joint_addresses(
            G1_GROOT_WBC_BODY_JOINT_NAMES
        )
        (
            self.controlled_qpos_addresses,
            self.controlled_dof_addresses,
        ) = self.build_joint_addresses(G1_GROOT_WBC_JOINT_NAMES)
        # The BiGym G1 base uses slide/hinge joints, NOT a free joint.
        # The x/y/z/yaw stack gains passive roll/pitch when leg support and
        # passive_base_tilt are enabled (the GR00T protocol defaults).
        # Read orientation/angular velocity from the pelvis IMU sensors,
        # not a free-joint qpos slice — the base has no free joint to read.
        orientation_sensor_adr = self.find_sensor("orientation", dim=4)
        gyro_sensor_adr = self.find_sensor("angular-velocity", dim=3)
        if orientation_sensor_adr is None or gyro_sensor_adr is None:
            raise ValueError(
                "groot_wbc_g1 requires pelvis 'orientation' (framequat) and "
                "'angular-velocity' (gyro) sensors in the robot model."
            )
        self.orientation_sensor_address = orientation_sensor_adr
        self.gyro_sensor_address = gyro_sensor_adr
        self.base_qpos_addresses, self.base_dof_addresses = self.build_joint_addresses(
            ("pelvis_x", "pelvis_y", "pelvis_z")
        )
        (
            self.controlled_range_low,
            self.controlled_range_high,
        ) = self.build_joint_ranges(G1_GROOT_WBC_JOINT_NAMES)
        self._default_targets = np.asarray(
            self.policy.config["default_angles"], dtype=np.float32
        ).reshape(15)
        self.command = np.zeros((3,), dtype=np.float32)
        self.height_command = float(config.default_height_cmd)
        self.last_action = np.zeros((15,), dtype=np.float32)
        self.last_targets = self._default_targets.copy()
        self.reset()

    def reset(self) -> None:
        """Zero the commands, reset the official policy and pose the joints.

        The controlled joints are written to the policy's ``default_angles``
        (with matching actuator targets), so an episode starts at the
        policy's own anchor pose.
        """
        self.command[:] = 0.0
        self.height_command = float(self._config.default_height_cmd)
        self._pitch_cmd = 0.0
        self.last_action[:] = 0.0
        self.policy.reset()
        # The official reset() does not touch the torso orientation command.
        self.policy.roll_cmd = 0.0
        self.policy.pitch_cmd = 0.0
        self.policy.yaw_cmd = 0.0
        self.policy.use_policy_action = True
        self.policy.set_use_teleop_policy_cmd(True)
        self.apply_pose(
            self.controlled_qpos_addresses,
            self.controlled_dof_addresses,
            self._default_targets,
            G1_GROOT_WBC_JOINT_NAMES,
        )
        self.last_targets = self._default_targets.copy()

    # set_command / get_command / getters: base clip-on-set semantics.

    def get_base_obs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return base linear velocity, angular velocity and gravity.

        Linear velocity is the world pelvis velocity rotated into the yaw
        frame; the angular velocity and the projected gravity vector come
        from the pelvis IMU gyro and framequat.
        """
        pose, velocity = self._read_floating_base()
        quat = pose[3:7]
        yaw = yaw_from_quat_wxyz(quat)
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        base_lin = np.asarray(
            [
                c * velocity[0] + s * velocity[1],
                -s * velocity[0] + c * velocity[1],
                velocity[2],
            ],
            dtype=np.float32,
        )
        gravity = _quat_rotate_inverse_wxyz(
            quat, np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
        )
        return base_lin, velocity[3:6].copy(), gravity

    def step(self) -> np.ndarray:
        """Run one policy step and return the 15 joint position targets.

        Feeds body joint state and the floating-base pose/velocity to the
        official policy together with the current height and navigate
        commands; non-finite output falls back to the default angles, and
        the targets are clipped to the joint ranges.
        """
        data = self._env.data
        pose, velocity = self._read_floating_base()
        observation = {
            "q": data.qpos[self.body_qpos_addresses].astype(np.float32, copy=True),
            "dq": data.qvel[self.body_dof_addresses].astype(np.float32, copy=True),
            "floating_base_pose": pose,
            "floating_base_vel": velocity,
        }
        self.policy.set_observation(observation)
        result = self.policy.get_action(
            arms_target_pose=None,
            base_height_command=np.asarray([self.height_command], dtype=np.float32),
            # The official get_action() copies this into roll/pitch/yaw_cmd
            # every call, so the torso pitch must be re-sent here rather than
            # only set in _set_pitch_command.
            torso_orientation_rpy=np.asarray(
                [0.0, self._pitch_cmd, 0.0], dtype=np.float32
            ),
            interpolated_navigate_cmd=self.command.copy(),
        )
        targets = np.asarray(result["body_action"][0], dtype=np.float32).reshape(15)
        if not np.all(np.isfinite(targets)):
            targets = self._default_targets.copy()
        self.last_action = np.asarray(self.policy.action, dtype=np.float32).reshape(15)
        self.last_targets = np.clip(
            targets, self.controlled_range_low, self.controlled_range_high
        )
        return self.last_targets.astype(np.float32, copy=True)

    def get_state(self) -> dict[str, np.ndarray]:
        """Snapshot the commands plus the official policy's mutable state.

        Includes the last action, target dof positions, command, height and
        rpy commands, gait indices and the stacked observation history, so
        a restored episode reproduces the history-dependent output.
        """
        history = list(self.policy.obs_history)
        if history:
            obs_history = np.stack(history).astype(np.float32, copy=False)
        else:
            obs_history = np.zeros((0, policy_obs_dim(self.policy)), dtype=np.float32)
        return {
            "cmd": self.command.copy(),
            "height_cmd": np.asarray(self.height_command, dtype=np.float32),
            "pitch_cmd": np.asarray(self._pitch_cmd, dtype=np.float32),
            "last_action": self.last_action.copy(),
            "last_targets": self.last_targets.copy(),
            "policy_action": np.asarray(self.policy.action, dtype=np.float32).copy(),
            "policy_target_dof_pos": np.asarray(
                self.policy.target_dof_pos, dtype=np.float32
            ).copy(),
            "policy_cmd": np.asarray(self.policy.cmd, dtype=np.float32).copy(),
            "policy_height_cmd": np.asarray(self.policy.height_cmd, dtype=np.float32),
            "policy_rpy_cmd": np.asarray(
                [self.policy.roll_cmd, self.policy.pitch_cmd, self.policy.yaw_cmd],
                dtype=np.float32,
            ),
            "policy_gait_indices": np.asarray(
                self.policy.gait_indices, dtype=np.float32
            )
            .reshape(1)
            .copy(),
            "policy_obs_history": obs_history,
        }

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        """Restore a snapshot produced by get_state() into the policy.

        Every key is required; the observation history is rebuilt as a
        deque with the official ``obs_history_len`` maxlen.
        """
        self.command = np.asarray(state["cmd"], dtype=np.float32).reshape(3).copy()
        self.height_command = float(np.asarray(state["height_cmd"]).reshape(-1)[0])
        # Absent in snapshots recorded without the pitch channel (pitch was 0).
        self._pitch_cmd = float(np.asarray(state.get("pitch_cmd", 0.0)).reshape(-1)[0])
        self.last_action = (
            np.asarray(state["last_action"], dtype=np.float32).reshape(15).copy()
        )
        self.last_targets = (
            np.asarray(state["last_targets"], dtype=np.float32).reshape(15).copy()
        )
        restore_policy_state(self.policy, state)

    def _read_floating_base(self) -> tuple[np.ndarray, np.ndarray]:
        """Free-joint-convention (pose[7], vel[6]) view of the BiGym base.

        Position/linear velocity come from the pelvis_x/y/z slide joints
        (world frame); orientation and angular velocity come from the pelvis
        IMU sensors (framequat wxyz + gyro, pelvis frame) — the policy only
        consumes the quaternion and the angular velocity.
        """
        data = self._env.data
        pos = data.qpos[self.base_qpos_addresses].astype(np.float32, copy=True)
        lin_vel = data.qvel[self.base_dof_addresses].astype(np.float32, copy=True)
        quat = data.sensordata[
            self.orientation_sensor_address : self.orientation_sensor_address + 4
        ].astype(np.float32, copy=True)
        ang_vel = data.sensordata[
            self.gyro_sensor_address : self.gyro_sensor_address + 3
        ].astype(np.float32, copy=True)
        pose = np.concatenate([pos, quat])
        velocity = np.concatenate([lin_vel, ang_vel])
        return pose, velocity


class GrootWbcG1Binding(BackendBinding):
    """GR00T-WBC on the G1 Dex1: its 15 joints actuated with the training PD."""

    robot_models = ("g1_dex1",)
    supports_passive_base_tilt = True

    def robot_cls(
        self, robot_cls: type[Robot], config: ControllerConfig
    ) -> type[Robot]:
        """The G1 Dex1 with the backend's joints actuated, PD and armature."""
        # The robot config loads with the env, not with bigym.loco.
        from bigym.robots.configs.g1 import G1_PASSIVE_TILT_DOFS, G1Dex1

        assert issubclass(robot_cls, G1Dex1)
        return robot_cls.variant(
            actuated=G1_GROOT_WBC_JOINT_NAMES,
            joint_pd=G1_GROOT_WBC_JOINT_PD,
            joint_armature=G1_GROOT_WBC_JOINT_ARMATURE,
            passive_dofs=G1_PASSIVE_TILT_DOFS if config.passive_base_tilt else None,
        )

    def build_controller(
        self, env: BiGymEnv, config: ControllerConfig, *, control_dt: float
    ) -> GrootWbcG1Controller:
        """The frozen policy pair with the env's command clips and height."""
        # Unset pitch bounds keep the backend's own range, which the RY
        # slot bounds then follow (LowerBody.resolve_command_bounds).
        pitch_bounds: dict[str, Any] = {
            key: float(getattr(config, key))
            for key in ("pitch_cmd_min", "pitch_cmd_max")
            if getattr(config, key) is not None
        }
        cfg = GrootWbcG1Config(
            control_dt=control_dt,
            default_height_cmd=float(config.default_height_cmd),
            cmd_clip=float(config.cmd_clip),
            wz_clip=float(config.wz_clip),
            model_path=str(config.model_path or ",".join(DEFAULT_GROOT_MODEL_PATHS)),
            enable_pitch_cmd=bool(config.pitch_command),
            **pitch_bounds,
        )
        return GrootWbcG1Controller(env=env, config=cfg)

    def configure_model(self, env: BiGymEnv) -> None:
        """Pin the G1 contact solver options."""
        from bigym.robots.configs.g1 import pin_contact_solver_options

        pin_contact_solver_options(env.model)


GROOT_WBC_G1 = GrootWbcG1Binding()
