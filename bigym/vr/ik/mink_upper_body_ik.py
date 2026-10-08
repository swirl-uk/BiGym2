"""Shared dual-arm upper-body IK core based on mink (QP differential IK).

Robot-specific subclasses (G1) build a stripped arm-only model from the
live env's spec and then call :meth:`_init_solver`. Each hand is a
full 6-DoF FrameTask; redundancy (or the position/orientation trade-off on
under-actuated arms) is resolved by the task weights and a weak rest-posture
task, with joint limits enforced as QP constraints. Costs follow NVIDIA's
Pink G1 teleop IK: position 8, orientation 2, posture 0.01 with per-joint
weights.
"""

from dataclasses import dataclass, field

import mink
import mujoco
import numpy as np
from pyquaternion import Quaternion

from bigym.utils.physics_utils import get_quaternion

POSITION_COST = 8.0
ORIENTATION_COST = 2.0
FRAME_LM_DAMPING = 3.0
POSTURE_COST = 0.01
QP_SOLVER = "daqp"
SOLVE_DT = 0.05
SOLVE_ITERS = 5


@dataclass
class Pose:
    """Pose represented by np.ndarray and Quaternion."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: Quaternion = Quaternion()


class MinkUpperBodyIK:
    """Dual-arm 6-DoF pose IK over a stripped arm-only model.

    Subclass __init__ must set `model`, `data`, `pelvis`,
    `arm_joints` (left arm joints then right arm joints, chain order),
    `left_arm_site`, `right_arm_site`, then call `_init_solver`.
    `fixed_joint_names` lists the unprefixed names of the fixed (measured,
    not solved) joints.
    """

    model: mujoco.MjModel
    data: mujoco.MjData
    pelvis: mujoco.MjsBody
    arm_joints: list[mujoco.MjsJoint]
    left_arm_site: mujoco.MjsSite
    right_arm_site: mujoco.MjsSite
    fixed_joint_names: tuple[str, ...] = ()

    def _init_solver(
        self,
        rest_pose: np.ndarray,
        posture_weight: np.ndarray,
        position_cost: float = POSITION_COST,
        orientation_cost: float = ORIENTATION_COST,
        posture_lm_damping: float = 0.0,
        solve_iters: int = SOLVE_ITERS,
        measured_resync_threshold: float | None = 0.5,
        fixed_joints: list[mujoco.MjsJoint] | None = None,
    ) -> None:
        # Joints kept in the stripped model but NOT solved for: their measured
        # angles are written in before every solve (the GR00T torso-pitch
        # command moves the shoulders by up to 15 cm; without this the solver
        # places the hands for an upright torso).
        model = self.model
        self.fixed_joints = list(fixed_joints or [])
        fixed_ids = [model.bind(joint).id for joint in self.fixed_joints]
        self._fixed_qpos_adr = model.jnt_qposadr[fixed_ids].astype(np.int64)
        self._fixed_dof_adr = model.jnt_dofadr[fixed_ids].astype(np.int64)
        arm_ids = [model.bind(joint).id for joint in self.arm_joints]
        self._limits_low = model.jnt_range[arm_ids, 0].copy()
        self._limits_high = model.jnt_range[arm_ids, 1].copy()
        self._num_joints = len(self.arm_joints)
        self._qpos_adr = model.jnt_qposadr[arm_ids].astype(np.int64)

        self._configuration = mink.Configuration(model)
        self._left_task = mink.FrameTask(
            frame_name=self.left_arm_site.name,
            frame_type="site",
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=FRAME_LM_DAMPING,
        )
        self._right_task = mink.FrameTask(
            frame_name=self.right_arm_site.name,
            frame_type="site",
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=FRAME_LM_DAMPING,
        )
        posture_cost = np.zeros(model.nv)
        posture_cost[self._qpos_adr] = POSTURE_COST * np.asarray(posture_weight)
        self._posture_task = mink.PostureTask(
            model,
            cost=posture_cost,
            lm_damping=posture_lm_damping,
        )
        q_rest = np.zeros(model.nq)
        q_rest[self._qpos_adr] = np.asarray(rest_pose)
        self._posture_task.set_target(q_rest)

        self._tasks = [self._left_task, self._right_task, self._posture_task]
        self._limits = [mink.ConfigurationLimit(model)]
        self._solve_iters = int(solve_iters)
        self._measured_resync_threshold = measured_resync_threshold
        self._last_solution: np.ndarray | None = None

    def reset_seed(self) -> None:
        """Drop the internal warm start (call when tracking (re-)engages)."""
        self._last_solution = None

    def seed(
        self,
        pelvis_pose: Pose,
        qpos_arm: np.ndarray,
        fixed_qpos: np.ndarray | None = None,
    ) -> None:
        """Pose the solver's model before reading wrist poses from it.

        The arms start from the previous solution when there is one, else
        from ``qpos_arm`` (left then right). ``fixed_qpos`` sets the fixed
        joints and locks them in the posture task, so the plan cannot lean
        on them (``solve`` only zeroes their velocity afterwards).
        """
        self.set_pelvis_pose(pelvis_pose)
        start = self._last_solution if self._last_solution is not None else qpos_arm
        q_full = np.zeros(self._configuration.model.nq)
        q_full[self._qpos_adr] = np.clip(start, self._limits_low, self._limits_high)
        if fixed_qpos is not None and len(self._fixed_qpos_adr):
            q_full[self._fixed_qpos_adr] = fixed_qpos
            target_q = np.array(self._posture_task.target_q, dtype=np.float64).copy()
            target_q[self._fixed_qpos_adr] = fixed_qpos
            self._posture_task.set_target(target_q)
            self._posture_task.cost[self._fixed_dof_adr] = 1e3
        self._configuration.update(q_full)

    def wrist_pose(self, side: str) -> Pose:
        """World pose of the ``left``/``right`` wrist site in the solver's model."""
        task = self._left_task if side == "left" else self._right_task
        transform = self._configuration.get_transform_frame_to_world(
            task.frame_name, "site"
        )
        return Pose(
            np.asarray(transform.translation(), dtype=np.float64),
            Quaternion(np.asarray(transform.rotation().wxyz, dtype=np.float64)),
        )

    def set_orientation_cost(self, side: str, cost: float) -> None:
        """Weight of the ``left``/``right`` wrist orientation target (0 frees it)."""
        task = self._left_task if side == "left" else self._right_task
        task.set_orientation_cost(cost)

    def solve(
        self,
        pelvis_pose: Pose,
        qpos_arm_left: np.ndarray,
        qpos_arm_right: np.ndarray,
        target_pose_left: Pose,
        target_pose_right: Pose,
        fixed_qpos: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve both arms toward their targets; returns the arm joint angles.

        ``fixed_qpos`` are the measured angles of the fixed (non-solved)
        joints declared at init, in that order; None keeps them at zero.
        """
        # The stripped model is shared with the mink Configuration, so the
        # pelvis pose written here is picked up by its next update().
        self.set_pelvis_pose(pelvis_pose)

        qpos = np.concatenate((qpos_arm_left, qpos_arm_right)).astype(np.float64)
        qpos = np.clip(qpos, self._limits_low, self._limits_high)
        # Seed from our own previous solution, not the measured joints: the
        # measured arm always lags between the last two answers, and seeding
        # from it lets the solver alternate between adjacent null-space
        # configurations every frame (25 Hz arm chatter during holds). Re-sync
        # to the measured state when it
        # diverges from the last answer (external contact, missed frames).
        if self._last_solution is not None:
            resync_threshold = self._measured_resync_threshold
            if (
                resync_threshold is None
                or np.abs(self._last_solution - qpos).max() < resync_threshold
            ):
                qpos = self._last_solution
        q_full = np.zeros(self._configuration.model.nq)
        q_full[self._qpos_adr] = qpos
        if fixed_qpos is not None and len(self._fixed_qpos_adr):
            q_full[self._fixed_qpos_adr] = np.asarray(fixed_qpos, dtype=np.float64)
        self._configuration.update(q_full)

        for task, target in (
            (self._left_task, target_pose_left),
            (self._right_task, target_pose_right),
        ):
            task.set_target(
                mink.SE3.from_rotation_and_translation(
                    mink.SO3(np.asarray(target.orientation.elements, dtype=np.float64)),
                    np.asarray(target.position, dtype=np.float64),
                )
            )

        for _ in range(self._solve_iters):
            velocity = mink.solve_ik(
                self._configuration,
                self._tasks,
                dt=SOLVE_DT,
                solver=QP_SOLVER,
                damping=1e-8,
                limits=self._limits,
            )
            if len(self._fixed_dof_adr):
                velocity[self._fixed_dof_adr] = 0.0
            self._configuration.integrate_inplace(velocity, SOLVE_DT)

        solution = self._configuration.q[self._qpos_adr]
        if solution.shape != (self._num_joints,):
            raise RuntimeError(
                f"IK produced {solution.shape}, expected ({self._num_joints},)."
            )
        self._last_solution = np.asarray(solution, dtype=np.float64).copy()
        return solution.astype(np.float32, copy=False)

    def set_pelvis_pose(self, pelvis_pose: Pose) -> None:
        """Place the solver model's pelvis, which is fixed to the world."""
        bound_pelvis = self.model.bind(self.pelvis)
        bound_pelvis.pos = pelvis_pose.position
        bound_pelvis.quat = pelvis_pose.orientation.elements

    def site_quaternion(self, site: mujoco.MjsSite) -> Quaternion:
        """Orientation of a site of the solver's model, as last forwarded."""
        return Quaternion(get_quaternion(self.data, site))
