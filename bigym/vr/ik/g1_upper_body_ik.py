"""G1 upper-body IK solver for VR demo collection (mink-based)."""

import mujoco
import numpy as np

from bigym import scene
from bigym.bigym_env import BiGymEnv
from bigym.const import HandSide
from bigym.robots.configs.g1 import (
    G1_LEFT_ARM_JOINT_NAMES,
    G1_RIGHT_ARM_JOINT_NAMES,
    G1_WAIST_JOINT_NAMES,
)
from bigym.vr.ik.mink_upper_body_ik import MinkUpperBodyIK, Pose  # noqa: F401

# Redundancy rest pose per 7-DoF arm and per-joint posture weights
# (shoulder pitch/roll/yaw, elbow, wrist r/p/y): relaxed shoulder_roll.
# shoulder_yaw/wrist_yaw are the softest joints but anchored at 0.5: a
# near-free null-space direction amplifies controller tracking noise into
# arm wobble along the yaw-roll redundancy circle.
REST_POSE_LEFT = np.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
REST_POSE_RIGHT = np.array([0.0, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
POSTURE_WEIGHT = np.array([4.0, 3.0, 0.5, 3.0, 1.0, 1.0, 0.5])

# Posture damping and continuous solver state follow NVIDIA's Pink G1 teleop.
# Five inner solves, not Pink's three: in Mink three leaves materially more
# one-frame target error.
POSTURE_LM_DAMPING = 1.0
SOLVE_ITERS = 5


class G1UpperBodyIK(MinkUpperBodyIK):
    """6-DoF pose IK for the two 7-DoF G1 arms.

    Hand/finger joints are intentionally not part of this IK model; the
    collector still writes them through the two BiGym gripper scalar
    actions. Element names come from the env's robot namespace (the arm
    chain and the end-effector site names are what the solver relies on).
    """

    def __init__(self, env: BiGymEnv):
        """Strip an arm-only model out of the live env's spec and init the solver."""
        robot = env.robot
        spec = env.spec.copy()
        frame = spec.body(robot.body.name)
        for element in (
            *spec.keys,
            *spec.actuators,
            *spec.sensors,
            *spec.tendons,
            *spec.equalities,
            *spec.excludes,
            *spec.pairs,
        ):
            spec.delete(element)
        world = spec.worldbody
        for element in (*world.geoms, *world.sites, *world.cameras, *world.lights):
            spec.delete(element)
        for body in list(world.bodies):
            if body.name != frame.name:
                spec.delete(body)

        arm_joint_names = [
            robot.namespace + name
            for name in (*G1_LEFT_ARM_JOINT_NAMES, *G1_RIGHT_ARM_JOINT_NAMES)
        ]
        # Waist joints stay in the model as FIXED joints (measured angles
        # written in per solve): the torso-pitch command bends the waist and
        # moves the shoulders, which an upright stripped model would miss.
        waist_joint_names = [robot.namespace + name for name in G1_WAIST_JOINT_NAMES]
        kept = {*arm_joint_names, *waist_joint_names}
        # The base joints go too: the pelvis is placed in the world per solve.
        for joint in scene.descendants(frame, "joints"):
            if joint.name not in kept:
                spec.delete(joint)
        for geom in spec.geoms:
            geom.contype = 0
            geom.conaffinity = 0

        arms = robot.config.arms
        self.pelvis = spec.body(robot.pelvis.name)
        self.arm_joints = [spec.joint(name) for name in arm_joint_names]
        self.left_arm_site = spec.site(robot.namespace + arms[HandSide.LEFT].site)
        self.right_arm_site = spec.site(robot.namespace + arms[HandSide.RIGHT].site)
        self.fixed_joint_names = G1_WAIST_JOINT_NAMES
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self._init_solver(
            rest_pose=np.concatenate((REST_POSE_LEFT, REST_POSE_RIGHT)),
            posture_weight=np.concatenate((POSTURE_WEIGHT, POSTURE_WEIGHT)),
            posture_lm_damping=POSTURE_LM_DAMPING,
            solve_iters=SOLVE_ITERS,
            measured_resync_threshold=None,
            fixed_joints=[spec.joint(name) for name in waist_joint_names],
        )
