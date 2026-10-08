"""Unitree G1 robot configs for BiGym wrappers."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Mapping, Optional

import mujoco
import numpy as np

from bigym.action_modes import PelvisDof
from bigym.const import ASSETS_PATH, HandSide
from bigym.robots.config import (
    ArmConfig,
    FloatingBaseConfig,
    FullBodyConfig,
    GripperConfig,
    RobotConfig,
)
from bigym.robots.gripper import Gripper
from bigym.robots.robot import Robot
from bigym.utils.dof import Dof

# The G1 Dex1-1 MJCF + meshes ship as package assets, from the official
# Unitree G1 description (see envs/xmls/g1/LICENSE). BIGYM_G1_DEX1_MJCF
# stays as a dev override. Keep the MJCF model name stable: scene element
# names are scoped by it.
G1_DEX1_MODEL = Path(
    os.environ.get(
        "BIGYM_G1_DEX1_MJCF",
        str(ASSETS_PATH / "g1" / "g1_29dof_with_dex1_1.xml"),
    )
).expanduser()


G1_LEG_JOINT_NAMES: tuple[str, ...] = (
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
)

G1_WAIST_JOINT_NAMES: tuple[str, ...] = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)

G1_LEFT_ARM_JOINT_NAMES: tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)

G1_RIGHT_ARM_JOINT_NAMES: tuple[str, ...] = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_ACTUATORS = {
    **{name: False for name in G1_LEG_JOINT_NAMES},
    **{name: False for name in G1_WAIST_JOINT_NAMES},
    **{name: True for name in G1_LEFT_ARM_JOINT_NAMES},
    **{name: True for name in G1_RIGHT_ARM_JOINT_NAMES},
}

# Robot-level arm PD: shoulders/elbows follow the real G1 onboard controller
# table (unitree_rl-style deploy stacks); wrists use (20, 2), NVIDIA Decoupled
# WBC's real-hardware teleop setting (the onboard (10/5, 0.5) lags under VR
# teleop). Arm gains stay a robot constant across lower-body backends, which
# only observe arm qpos/qvel. The armature values are the physical motor
# constants (5020 / 4010 types).
G1_ARM_PD: dict[str, tuple[float, float]] = {
    name: gains
    for side in ("left", "right")
    for name, gains in {
        f"{side}_shoulder_pitch_joint": (150.0, 4.0),
        f"{side}_shoulder_roll_joint": (150.0, 4.0),
        f"{side}_shoulder_yaw_joint": (150.0, 4.0),
        f"{side}_elbow_joint": (100.0, 1.0),
        f"{side}_wrist_roll_joint": (20.0, 2.0),
        f"{side}_wrist_pitch_joint": (20.0, 2.0),
        f"{side}_wrist_yaw_joint": (20.0, 2.0),
    }.items()
}

G1_ARM_ARMATURE: dict[str, float] = {
    name: (0.00425 if ("wrist_pitch" in name or "wrist_yaw" in name) else 0.003609725)
    for name in G1_ARM_PD
}

G1_DEX1_LEFT_HAND_JOINT_NAMES: tuple[str, ...] = (
    "left_dex1_finger_joint_1",
    "left_dex1_finger_joint_2",
)

G1_DEX1_RIGHT_HAND_JOINT_NAMES: tuple[str, ...] = (
    "right_dex1_finger_joint_1",
    "right_dex1_finger_joint_2",
)

# Dex1-1 finger travel (both joints share it): +0.0245 m = fully open,
# -0.02 m = fully closed. Measured on the generated MJCF: the fingers sit on
# opposing slide axes, so this 0.0445 m per-finger travel is the official
# 90 mm stroke. On hardware one M4010 motor drives both fingers; the model's
# two joints are always commanded together (one scalar per hand).
G1_DEX1_QPOS_OPEN = 0.0245
G1_DEX1_QPOS_CLOSED = -0.02

# Dex1-1 finger PD (prismatic). kp=2000 saturates the URDF 20 N effort cap
# at 1 cm of position error; kd~=critical for the 0.087 kg finger
# (2*sqrt(kp*m)~=26). Full-stroke close takes ~0.25 s (the 0.2 m/s hardware
# rate, via the gripper slew limit) with no hold chatter, and a blocked
# squeeze saturates the +-20 N cap.
G1_DEX1_HAND_PD: dict[str, tuple[float, float]] = {
    name: (2000.0, 30.0)
    for name in (G1_DEX1_LEFT_HAND_JOINT_NAMES + G1_DEX1_RIGHT_HAND_JOINT_NAMES)
}

G1_LEFT_ARM = ArmConfig(
    site="left_end_effector",
    links=[name.replace("_joint", "_link") for name in G1_LEFT_ARM_JOINT_NAMES],
)
G1_RIGHT_ARM = ArmConfig(
    site="right_end_effector",
    links=[name.replace("_joint", "_link") for name in G1_RIGHT_ARM_JOINT_NAMES],
)

G1_DEX1_LEFT_GRIPPER = GripperConfig(
    actuators=list(G1_DEX1_LEFT_HAND_JOINT_NAMES),
    range=np.array([0.0, 1.0], dtype=np.float32),
    body="left_wrist_yaw_link",
    pad_bodies=[
        "left_dex1_finger_link_1",
        "left_dex1_finger_link_2",
    ],
    discrete=False,
)
G1_DEX1_RIGHT_GRIPPER = GripperConfig(
    actuators=list(G1_DEX1_RIGHT_HAND_JOINT_NAMES),
    range=np.array([0.0, 1.0], dtype=np.float32),
    body="right_wrist_yaw_link",
    pad_bodies=[
        "right_dex1_finger_link_1",
        "right_dex1_finger_link_2",
    ],
    discrete=False,
)

G1_FLOATING_BASE = FloatingBaseConfig(
    dofs={
        PelvisDof.X: Dof(
            joint_type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=(1, 0, 0),
            stiffness=1e4,
        ),
        PelvisDof.Y: Dof(
            joint_type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=(0, 1, 0),
            stiffness=1e4,
        ),
        PelvisDof.Z: Dof(
            joint_type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=(0, 0, 1),
            joint_range=(0.35, 1.20),
            action_range=(0.35, 1.20),
            stiffness=1e6,
        ),
        PelvisDof.RZ: Dof(
            joint_type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=(0, 0, 1),
            stiffness=1e4,
        ),
    },
    delta_range_position=(-0.01, 0.01),
    delta_range_rotation=(-0.05, 0.05),
    offset_position=np.array([0.0, 0.0, 0.75]),
    reset_state=np.zeros(len(G1_LEFT_ARM_JOINT_NAMES) + len(G1_RIGHT_ARM_JOINT_NAMES)),
)

G1_PASSIVE_TILT_DOFS = {
    "pelvis_rx": Dof(
        joint_type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=(1, 0, 0),
    ),
    "pelvis_ry": Dof(
        joint_type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=(0, 1, 0),
    ),
}

G1_FULL_BODY = FullBodyConfig(
    offset_position=np.array([0.0, 0.0, 0.75]),
    reset_state=np.zeros(
        len(G1_LEG_JOINT_NAMES)
        + len(G1_WAIST_JOINT_NAMES)
        + len(G1_LEFT_ARM_JOINT_NAMES)
        + len(G1_RIGHT_ARM_JOINT_NAMES)
    ),
)

# The floating-base G1 Dex1: legs and waist unactuated. A lower-body backend
# builds the robot from its own copy (G1Dex1.variant).
G1_DEX1_CONFIG = RobotConfig(
    model=G1_DEX1_MODEL,
    delta_range=(-0.1, 0.1),
    position_kp=150,
    pelvis_body="pelvis",
    full_body=G1_FULL_BODY,
    floating_base=G1_FLOATING_BASE,
    gripper=G1_DEX1_RIGHT_GRIPPER,
    arms={HandSide.LEFT: G1_LEFT_ARM, HandSide.RIGHT: G1_RIGHT_ARM},
    actuators=G1_ACTUATORS,
    cameras=["head", "left_wrist", "right_wrist"],
)


class G1Dex1(Robot):
    """Unitree G1 with Dex1-1 parallel two-finger grippers.

    The MJCF ships as a bigym package asset (envs/xmls/g1/). BiGym adds its
    own end-effector sites and camera elements at load time so the existing
    task observation interface remains `head`, `left_wrist`, and
    `right_wrist`.
    """

    # Finger PD table and the end-effector site offset from wrist_yaw_link:
    # the pinch center between the finger pads, measured on the generated
    # MJCF (collision pad centers sit at x~=0.148 in the wrist_yaw frame).
    _HAND_PD: dict[str, tuple[float, float]] = G1_DEX1_HAND_PD
    _EE_SITE_POS: tuple[float, float, float] = (0.148, 0.0, 0.0)

    _CONFIG: RobotConfig = G1_DEX1_CONFIG
    # Per-joint (kp, kd) applied at MJCF load time on top of BiGym's actuator
    # rebuild. BiGym replaces every `motor` actuator with a `position`
    # actuator using one uniform `position_kp` and critical joint damping;
    # walking policies are trained against specific per-joint PD gains
    # (critical damping on a hip is ~10-20x the RL deploy value and kills leg
    # swing). A lower-body backend sets its training gains via variant().
    _JOINT_PD_OVERRIDES: Mapping[str, tuple[float, float]] = {}
    # Per-joint armature (motor reflected inertia) applied alongside the PD
    # overrides. The G1 source MJCF only sets armature on 2 of 29 joints;
    # backends trained with full motor models (GR00T-WBC) must inject their
    # armature table or the closed-loop joint dynamics differ from training.
    _JOINT_ARMATURE_OVERRIDES: Mapping[str, float] = {}

    @property
    def config(self) -> RobotConfig:
        """Get robot config."""
        return self._CONFIG

    @classmethod
    def variant(
        cls,
        *,
        actuated: tuple[str, ...] = (),
        joint_pd: Optional[Mapping[str, tuple[float, float]]] = None,
        joint_armature: Optional[Mapping[str, float]] = None,
        passive_dofs: Optional[Mapping[str, Dof]] = None,
    ) -> type[G1Dex1]:
        """A subclass built from its own copy of the config (one per backend).

        Args:
            actuated: Joints of ``G1_ACTUATORS`` the backend drives; they keep
                their actuators in floating-base mode.
            joint_pd: Per-joint (kp, kd) applied over the arm and hand tables.
            joint_armature: Per-joint armature applied with ``joint_pd``.
            passive_dofs: Unactuated floating-base joints to add (e.g.
                ``G1_PASSIVE_TILT_DOFS``).
        """
        base = cls._CONFIG
        actuators = dict(base.actuators)
        for name in actuated:
            if name in actuators:
                actuators[name] = True
        floating_base = dataclasses.replace(
            base.floating_base,
            passive_dofs={**base.floating_base.passive_dofs, **(passive_dofs or {})},
        )
        return type(
            cls.__name__,
            (cls,),
            {
                "_CONFIG": dataclasses.replace(
                    base, actuators=actuators, floating_base=floating_base
                ),
                "_JOINT_PD_OVERRIDES": {**cls._JOINT_PD_OVERRIDES, **(joint_pd or {})},
                "_JOINT_ARMATURE_OVERRIDES": {
                    **cls._JOINT_ARMATURE_OVERRIDES,
                    **(joint_armature or {}),
                },
            },
        )

    def _get_grippers(self, model: mujoco.MjSpec) -> dict[HandSide, Gripper]:
        return {
            side: G1Dex1Gripper(
                side, self._wrist_sites[side], config, self.simulation, model
            )
            for side, config in (
                (HandSide.LEFT, G1_DEX1_LEFT_GRIPPER),
                (HandSide.RIGHT, G1_DEX1_RIGHT_GRIPPER),
            )
        }

    def _on_loaded(self, model: mujoco.MjSpec):
        # Joint actuator-force limits follow autolimits, not the MJCF default class.
        model.default.joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_AUTO
        for joint in model.joints:
            joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_AUTO

        # The robot MJCF carries its own ground plane below the scene floor;
        # it never touches anything but shows as a second floor in viewers.
        ground = model.geom("ground")
        if ground is not None:
            model.delete(ground)

        self._ensure_site(
            model, "left_wrist_yaw_link", "left_end_effector", pos=self._EE_SITE_POS
        )
        self._ensure_site(
            model, "right_wrist_yaw_link", "right_end_effector", pos=self._EE_SITE_POS
        )

        self._ensure_camera(
            model,
            "torso_link",
            "head",
            pos=(0.12, 0.0, 0.37),
            euler=(0.0, -0.52, -1.57),
        )
        self._ensure_camera(
            model,
            "left_wrist_yaw_link",
            "left_wrist",
            pos=(0.12, 0.0, 0.04),
            euler=(0.0, -0.87, -1.57),
        )
        self._ensure_camera(
            model,
            "right_wrist_yaw_link",
            "right_wrist",
            pos=(0.12, 0.0, 0.04),
            euler=(0.0, -0.87, -1.57),
        )

        super()._on_loaded(model)
        self._apply_joint_pd_overrides(model)

    @classmethod
    def _apply_joint_pd_overrides(cls, model: mujoco.MjSpec) -> None:
        """Set per-joint PD gains for hands and backend-controlled joints.

        Runs after ``Robot._on_loaded`` so it overrides both the freshly
        created position actuators (uniform kp + critical damping) and the
        finger actuators from the source MJCF. Damping goes on the joint,
        not actuator kv: joint damping is integrated implicitly under the
        Euler integrator and stays stable on very light finger links at a
        500 Hz physics step, actuator kv does not. The source MJCF ships arm
        actuators with their own kv (11-25), which would add to the joint
        damping and overdamp the wrists ~10x, so kv is zeroed.
        """
        overrides: dict[str, tuple[float, float]] = dict(cls._HAND_PD)
        overrides.update(G1_ARM_PD)
        overrides.update(cls._JOINT_PD_OVERRIDES)
        armature_overrides = dict(G1_ARM_ARMATURE)
        armature_overrides.update(cls._JOINT_ARMATURE_OVERRIDES)
        for actuator in model.actuators:
            joint = model.joint(actuator.target)
            if joint.name in armature_overrides:
                joint.armature = float(armature_overrides[joint.name])
            if joint.name not in overrides:
                continue
            kp, kd = overrides[joint.name]
            actuator.gainprm[0] = kp
            actuator.biasprm[1] = -kp
            # Zeroes kv; a source kv compiles to -0.0, as in the published models.
            actuator.biasprm[2] *= 0
            joint.damping[0] = float(kd)
            if joint.name in cls._HAND_PD:
                joint.armature = 0.001

    @staticmethod
    def _ensure_site(
        model: mujoco.MjSpec,
        body_name: str,
        site_name: str,
        *,
        pos: tuple[float, float, float],
    ) -> None:
        if model.site(site_name) is not None:
            return
        # group=5 keeps the marker out of every render (viewer default shows
        # site groups 0-2): the Dex1 pinch-center site floats between the
        # finger pads and would otherwise show up as a 1 cm gray ball in
        # cameras and VR. Sites stay fully functional (IK target / obs)
        # regardless of group.
        model.body(body_name).add_site(
            name=site_name, size=(0.01, 0.01, 0.01), pos=pos, group=5
        )

    @staticmethod
    def _ensure_camera(
        model: mujoco.MjSpec,
        body_name: str,
        camera_name: str,
        *,
        pos: tuple[float, float, float],
        euler: tuple[float, float, float],
    ) -> None:
        if model.camera(camera_name) is not None:
            return
        camera = model.body(body_name).add_camera(name=camera_name, fovy=60, pos=pos)
        camera.alt.type = mujoco.mjtOrientation.mjORIENTATION_EULER
        camera.alt.euler = euler


def pin_contact_solver_options(model: mujoco.MjModel) -> None:
    """Pin the G1 scene physics options explicitly (all G1 backends).

    cone=elliptic, impratio=10 are what the G1 grasp tuning was validated
    under; noslip=2 stops stance-foot slip. solver=Newton overrides the
    robot XML's PGS (which the scene inherits from the robot): PGS
    warmstarts from variable-size efc/contact internals that cannot be
    reseeded from a demo npz, breaking bit-exact replay restore, while
    Newton's only cross-step memory is qacc_warmstart.
    """
    model.opt.solver = int(mujoco.mjtSolver.mjSOL_NEWTON)
    model.opt.cone = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
    model.opt.impratio = 10.0
    model.opt.noslip_iterations = 2


class G1Dex1Gripper(Gripper):
    """One-scalar Dex1-1 parallel gripper: 0 = open, 1 = closed.

    Continuous open/close semantics. Both finger joints always receive the
    same target, mirroring the single M4010 motor on hardware. Targets are
    slew-limited so a step in the command does not become a contact-time
    torque impulse.
    """

    # Per-control-step target change in meters. The real Dex1-1 finger
    # velocity limit is 0.2 m/s (URDF); at the 50 Hz control rate that is
    # 0.004 m/step, closing the full 0.0445 m travel in ~0.22 s.
    _QPOS_RATE_PER_STEP = 0.004

    _target_qpos: Optional[dict[str, float]] = None
    _target_qpos_time: Optional[float] = None

    def set_control(self, ctrl: float):
        """Drive both fingers toward ``ctrl`` in [0, 1], slew-limited."""
        ctrl = float(np.clip(ctrl, 0.0, 1.0))
        desired = float(
            np.interp(ctrl, (0.0, 1.0), (G1_DEX1_QPOS_OPEN, G1_DEX1_QPOS_CLOSED))
        )
        for actuator in self._actuators:
            target = self._slew_limited_joint_target(actuator.name, desired)
            ctrlrange = np.asarray(
                self.simulation.model.bind(actuator).ctrlrange, dtype=np.float32
            )
            target = float(np.clip(target, ctrlrange[0], ctrlrange[1]))
            self.simulation.data.bind(actuator).ctrl = target

    def reset(self) -> None:
        """Drop the slew-limiter targets so the next command re-seeds them.

        ``_slew_limited_joint_target`` also re-seeds when physics time runs
        backwards, but an episode that ends at t=0 (e.g. reset right after
        reset, or ``check_env``'s seed-determinism probe) leaves ``time``
        equal, so the previous episode's targets would leak into the new one.
        """
        self._target_qpos = None
        self._target_qpos_time = None

    def _slew_limited_joint_target(self, name: str, target_cmd: float) -> float:
        now = float(self.simulation.data.time)
        targets = self._target_qpos
        last_time = self._target_qpos_time
        if targets is None or last_time is None or now < last_time:
            targets = {
                joint.name: float(self.simulation.data.bind(joint).qpos.item())
                for joint in self._actuated_joints
            }
        current = float(targets.get(name, target_cmd))
        step = float(
            np.clip(
                target_cmd - current,
                -self._QPOS_RATE_PER_STEP,
                self._QPOS_RATE_PER_STEP,
            )
        )
        target = float(current + step)
        targets[name] = target
        self._target_qpos = targets
        self._target_qpos_time = now
        return target

    @property
    def qpos(self) -> float:
        """Current opening as a scalar: 0 = open, 1 = closed."""
        if not self._actuated_joints:
            return 0.0
        positions = [
            float(self.simulation.data.bind(joint).qpos.item())
            for joint in self._actuated_joints
        ]
        scalar = float(
            np.interp(
                float(np.mean(positions)),
                (G1_DEX1_QPOS_CLOSED, G1_DEX1_QPOS_OPEN),
                (1.0, 0.0),
            )
        )
        return float(np.round(scalar, decimals=self._ROUND_DECIMALS))
