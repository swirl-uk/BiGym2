"""Controller-pose -> robot-action mapping (arms IK, base velocity commands)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
from pyquaternion import Quaternion

from bigym.action_modes import PelvisDof
from bigym.utils.physics_utils import get_quaternion
from bigym.vr.collect.constants import (
    G1_ARM_ENGAGE_SMOOTH_SECONDS,
    G1_ARM_MAX_JOINT_SPEED,
    G1_LEFT_ARM_NAMES,
    G1_RIGHT_ARM_NAMES,
    G1_ROBOT_MODELS,
    GROOT_WBC_MIN_WALK_SPEED,
)
from bigym.vr.viewer import Side

if TYPE_CHECKING:
    from bigym.vr.ik.mink_upper_body_ik import Pose


def _shape_planar_velocity_command(
    vx: float,
    vy: float,
    *,
    lowerbody_backend: str,
) -> tuple[float, float]:
    """Keep GROOT's non-zero planar commands out of its Balance/Walk gap."""
    vx = float(vx)
    vy = float(vy)
    if str(lowerbody_backend) != "groot_wbc_g1":
        return vx, vy

    speed = float(np.hypot(vx, vy))
    if speed == 0.0 or speed >= GROOT_WBC_MIN_WALK_SPEED:
        return vx, vy
    gain = GROOT_WBC_MIN_WALK_SPEED / speed
    return vx * gain, vy * gain


def _condition_arm_joint_targets(
    solution: np.ndarray,
    *,
    previous_command: np.ndarray,
    engage_start: np.ndarray,
    engage_elapsed: float,
    engage_duration: float,
    max_joint_speed: float | None,
    control_dt: float,
) -> np.ndarray:
    """Blend in a fresh IK branch, then reject implausible joint jumps."""
    target = np.asarray(solution, dtype=np.float64)
    previous = np.asarray(previous_command, dtype=np.float64)
    start = np.asarray(engage_start, dtype=np.float64)

    if engage_duration > 0.0:
        alpha = float(np.clip(engage_elapsed / engage_duration, 0.0, 1.0))
        target = start + alpha * (target - start)
    if max_joint_speed is not None:
        max_delta = float(max_joint_speed) * float(control_dt)
        target = previous + np.clip(target - previous, -max_delta, max_delta)
    return target.astype(np.float32)


def set_posef_position(pose: Any, value: np.ndarray) -> None:
    """Write a 3-vector into an OpenXR ``Posef``'s position."""
    pose.position.x = float(value[0])
    pose.position.y = float(value[1])
    pose.position.z = float(value[2])


def wrap_angle(angle: float) -> float:
    """``angle`` in rad, wrapped to ``[-pi, pi)``."""
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def planar_space_transform(
    hmd_position: np.ndarray,
    hmd_forward: np.ndarray,
    robot_head_position: np.ndarray,
    robot_yaw: float,
) -> tuple[float, np.ndarray]:
    """Map the current HMD position/heading onto the robot head/heading."""
    from bigym.vr.viewer.pyopenxr_to_mujoco_converter import rotate_vector_yaw

    hmd_yaw = float(np.arctan2(hmd_forward[1], hmd_forward[0]))
    yaw_offset = wrap_angle(float(robot_yaw) - hmd_yaw)
    translation = np.asarray(robot_head_position, dtype=np.float64) - rotate_vector_yaw(
        np.asarray(hmd_position, dtype=np.float64), yaw_offset
    )
    return yaw_offset, translation


def _engaged_target_position(
    ctrl_pos: np.ndarray,
    ref_ee: np.ndarray,
    ref_ctrl: np.ndarray,
    yaw_delta: float,
) -> np.ndarray:
    """Engaged-hand target with the clutch offset carried in the heading frame.

    The engage-time hand-vs-controller mismatch (ref_ee - ref_ctrl) is a
    body-attached quantity: when follow_head rotates the VR space with the
    robot (stick turn-in-place), it must rotate too; as a fixed world vector
    it would leave an (I - R(yaw_delta)) @ (ref_ee - ref_ctrl) hand offset
    after any turn. The orientation path needs no correction: its space-yaw factors
    cancel algebraically (mapped quats carry R(yaw) as a left factor on
    both sides of the delta product).
    """
    from bigym.vr.viewer.pyopenxr_to_mujoco_converter import rotate_vector_yaw

    return np.asarray(ctrl_pos, dtype=np.float64) + rotate_vector_yaw(
        np.asarray(ref_ee, dtype=np.float64) - np.asarray(ref_ctrl, dtype=np.float64),
        float(yaw_delta),
    )


class VRActionMapper:
    """Map VR controller state to G1 actions: arm IK and base commands."""

    def __init__(
        self,
        env: Any,
        *,
        robot_model: str,
        lowerbody_backend: str,
        yaw_mode: str,
        base_vx_scale: float,
        base_vy_scale: float,
        base_wz_scale: float,
        base_z_scale: float,
        height_cmd: float,
        height_cmd_max: float | None = None,
        pitch_rate: float = 0.8,
        stick_deadzone: float,
        base_cmd_slew: float | None = None,
    ) -> None:
        """Bind to a live env and cache the arm IK, stick scales and limits."""
        self._env = env
        self._outer = env.bigym
        self._inner = self._outer.inner_env
        self._robot_model = str(robot_model)
        self._lowerbody_backend = str(lowerbody_backend)
        # mink 6-DoF pose IK; orientation targets are deltas from the engage
        # reference, so engage continuity needs no static wrist offsets
        # (which also froze hand orientation).
        # Both G1 hand variants share the arm chain and end-effector sites
        # (the Dex1-1 XML is a build-time graft below the wrist), so one
        # solver serves them; it derives body-name prefixes from the env.
        if self._robot_model not in G1_ROBOT_MODELS:
            raise ValueError(
                f"unsupported robot_model {self._robot_model!r}; expected one of "
                f"{G1_ROBOT_MODELS}"
            )
        from bigym.vr.ik.g1_upper_body_ik import G1UpperBodyIK

        self._ik_cls = G1UpperBodyIK
        self._left_arm_names = G1_LEFT_ARM_NAMES
        self._right_arm_names = G1_RIGHT_ARM_NAMES
        self._arm_names = self._left_arm_names + self._right_arm_names
        self._yaw_mode = yaw_mode
        self._base_vx_scale = float(base_vx_scale)
        self._base_vy_scale = float(base_vy_scale)
        self._base_wz_scale = float(base_wz_scale)
        self._base_z_scale = float(base_z_scale)
        self._height_cmd_default = float(height_cmd)
        # Collector-side cap on the integrated height (None = the env's raw
        # bound). GR00T's spec allows 1.0 m, but above ~0.8 the knees lock and
        # the policy stops stepping (turns become a torso twist).
        self._height_cmd_cap = None if height_cmd_max is None else float(height_cmd_max)
        self._height_cmd = self._height_cmd_default
        self.last_height = self._height_cmd_default
        self._stick_deadzone = float(stick_deadzone)
        # Optional command slew (units/s); None maps the stick directly.
        # Height and pitch are integrated from the stick below and are not
        # slewed; the adapter only clips them.
        self._base_cmd_slew = None if base_cmd_slew is None else float(base_cmd_slew)
        self._cmd_dt = float(env.control_step_seconds)
        self._last_wz_cmd = 0.0
        self.arm_engage_smooth_seconds = G1_ARM_ENGAGE_SMOOTH_SECONDS
        self.arm_max_joint_speed = G1_ARM_MAX_JOINT_SPEED
        self._arm_engage_started_at: float | None = None
        self._arm_engage_start: np.ndarray | None = None
        # Torso pitch (PelvisDof.RY slot, only present when the backend
        # enables it, as groot_wbc_g1 does). Height and pitch are
        # integrated and hold their value when the stick is released. Right
        # stick Y is always height; holding the left grip enters pitch mode,
        # in which the left stick Y leans the torso.
        self._pitch_cmd = 0.0
        self.pitch_mode = False
        # Pitch integrates at pitch_rate rad/s of full stick (default 0.8:
        # the whole GR00T range in ~1 s; the official teleop steps 10 deg per
        # key press).
        self._pitch_scale = float(pitch_rate) * self._cmd_dt
        self.last_pitch = 0.0
        self._ik = self._ik_cls(self._inner)

        self._raw_low = np.asarray(self._outer.action_stats["min"], dtype=np.float32)
        self._raw_high = np.asarray(self._outer.action_stats["max"], dtype=np.float32)

        self._outer_base_dofs = list(self._outer.outer_action_floating_dofs)
        self._outer_base_idx = {dof: i for i, dof in enumerate(self._outer_base_dofs)}
        self.pitch_slot = PelvisDof.RY in self._outer_base_idx
        self._outer_limb_names = tuple(self._outer.outer_limb_actuator_names)
        self._outer_limb_idx = {
            name: i for i, name in enumerate(self._outer_limb_names)
        }
        self._full_limb_idx = dict(self._outer.limb_name_to_full_index)

        missing = [name for name in self._arm_names if name not in self._full_limb_idx]
        if missing:
            raise RuntimeError(
                f"Cannot locate {self._robot_model} arm actuators for VR IK: {missing}"
            )

        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_yaw = 0.0
        # Until engaged, arms stay pinned at the hold pose instead of
        # tracking the real controllers, so every attempt starts identically.
        self.engaged = False
        # No height lock: the axis-dominance gate below keeps a turning thumb
        # out of the height channel. The attribute stays False for the HUD.
        self.height_locked = False
        # Captured on the first engaged frame; hand targets are deltas from it
        # so engaging never jumps regardless of where the real hands are.
        self._engage_ref: dict[str, Any] | None = None
        self._hand_button_cmd = {
            "left": {"trigger": 0.0, "grip": 0.0},
            "right": {"trigger": 0.0, "grip": 0.0},
        }

    def reset(self) -> None:
        """Rebuild the IK solver and clear engage state and cached commands."""
        self._ik = self._ik_cls(self._inner)
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_yaw = 0.0
        self._last_wz_cmd = 0.0
        self._height_cmd = self._height_cmd_default
        self.last_height = self._height_cmd_default
        self._pitch_cmd = 0.0
        self.pitch_mode = False
        self.last_pitch = 0.0
        self.engaged = False
        self._engage_ref = None
        self._arm_engage_started_at = None
        self._arm_engage_start = None
        self._hand_button_cmd = {
            "left": {"trigger": 0.0, "grip": 0.0},
            "right": {"trigger": 0.0, "grip": 0.0},
        }

    def get_action(
        self, context: Any, space_offset: Any, space_yaw_offset: float = 0.0
    ) -> np.ndarray:
        """Map the current controller state to one normalized env action."""
        from bigym.vr.ik.g1_upper_body_ik import Pose
        from bigym.vr.viewer.control_profiles.control_profile import controller_pose

        raw = self._outer.raw_hold_action().astype(np.float32, copy=True)

        left_state = context.input.state[Side.LEFT]
        right_state = context.input.state[Side.RIGHT]
        if not (left_state.is_active and right_state.is_active) or not self.engaged:
            # Also re-anchor after tracking loss so re-acquire never jumps.
            self._engage_ref = None
            self._arm_engage_started_at = None
            self._arm_engage_start = None
            self.last_vx = 0.0
            self.last_vy = 0.0
            self.last_yaw = 0.0
            self.last_height = self._height_cmd
            if self.pitch_slot:
                # Hold the current pitch while disengaged (the hold action
                # defaults the RY slot to 0, which would straighten the torso).
                raw[self._outer_base_idx[PelvisDof.RY]] = self._pitch_cmd
            action = self._outer.normalize_action(raw)
            return np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)

        vx = self._stick(left_state.thumbstick_y) * self._base_vx_scale
        # Stick +x is rightward but body-frame vy>0 means leftward, so negate.
        vy = -self._stick(left_state.thumbstick_x) * self._base_vy_scale
        # Same convention mismatch as vy: positive yaw/wz turns left (CCW).
        yaw = -self._stick(right_state.thumbstick_x)
        # Right stick Y integrates height; torso pitch has its own mode
        # (see below) driven by the left stick.
        # G1-Dex1 (one-scalar gripper on the trigger): HOLD the LEFT
        # grip button for PITCH mode (a stick click would jolt the walk
        # command). While held, the left stick Y leans the
        # torso and walking is suspended; releasing keeps the integrated
        # lean and returns the stick to walking. Hysteresis-latched like
        # the trigger so a half-squeezed grip does not chatter.
        if self.pitch_slot:
            held = (
                self._latched_hand_button("left", "grip", left_state.grip_value) > 0.5
            )
            if held and not self.pitch_mode:
                left_state.vibration = True
            self.pitch_mode = held
        stick_y = self._stick(right_state.thumbstick_y)
        # Axis-dominance gate on the right stick: the stick is either a turn
        # (X) or a height/pitch push (Y), never both, so a diagonal thumb
        # during a turn cannot integrate an unintended squat.
        if abs(right_state.thumbstick_y) < abs(right_state.thumbstick_x):
            stick_y = 0.0
        else:
            yaw = 0.0
        z_rate = stick_y
        pitch_rate = 0.0
        if self.pitch_mode:
            # Left stick Y -> torso pitch (dominant axis only); no walking
            # while leaning.
            left_y = self._stick(left_state.thumbstick_y)
            if abs(left_state.thumbstick_y) < abs(left_state.thumbstick_x):
                left_y = 0.0
            pitch_rate = left_y
            vx, vy = 0.0, 0.0

        self._write_base_command(
            raw, vx=vx, vy=vy, yaw=yaw, z_rate=z_rate, pitch_rate=pitch_rate
        )

        l_pos, l_quat = controller_pose(
            context, Side.LEFT, space_offset, yaw_offset=space_yaw_offset
        )
        r_pos, r_quat = controller_pose(
            context, Side.RIGHT, space_offset, yaw_offset=space_yaw_offset
        )
        l_pos = np.asarray(l_pos, dtype=np.float64)
        r_pos = np.asarray(r_pos, dtype=np.float64)

        just_anchored = self._engage_ref is None
        if just_anchored:
            ee = self._current_ee_positions()
            self._engage_ref = {
                "l_ctrl": l_pos.copy(),
                "r_ctrl": r_pos.copy(),
                "l_ctrl_quat": Quaternion(l_quat),
                "r_ctrl_quat": Quaternion(r_quat),
                "l_ee": ee["left"],
                "r_ee": ee["right"],
                "l_ee_quat": ee["left_quat"],
                "r_ee_quat": ee["right_quat"],
                "space_yaw": float(space_yaw_offset),
            }
            self._ik.reset_seed()
            self._arm_engage_start = self._arm_command_from_raw(raw)
            self._arm_engage_started_at = time.monotonic()
        ref = self._engage_ref
        yaw_delta = wrap_angle(float(space_yaw_offset) - float(ref["space_yaw"]))
        l_target_pos = _engaged_target_position(
            l_pos, ref["l_ee"], ref["l_ctrl"], yaw_delta
        )
        r_target_pos = _engaged_target_position(
            r_pos, ref["r_ee"], ref["r_ctrl"], yaw_delta
        )
        # World-frame controller rotation since engage, applied on top of
        # the hand pose captured at engage: at the engage moment the target
        # equals the current pose, so tracking starts with zero orientation
        # error and rotating the controller rotates the hand.
        l_target_quat = (Quaternion(l_quat) * ref["l_ctrl_quat"].inverse) * ref[
            "l_ee_quat"
        ]
        r_target_quat = (Quaternion(r_quat) * ref["r_ctrl_quat"].inverse) * ref[
            "r_ee_quat"
        ]

        solution = self._ik.solve(
            pelvis_pose=self._pelvis_pose(),
            qpos_arm_left=self._arm_qpos(self._left_arm_names),
            qpos_arm_right=self._arm_qpos(self._right_arm_names),
            target_pose_left=Pose(l_target_pos, l_target_quat),
            target_pose_right=Pose(r_target_pos, r_target_quat),
            fixed_qpos=self._fixed_qpos(),
        )
        previous_arm_command = self._arm_command_from_raw(raw)
        engage_start = (
            previous_arm_command
            if self._arm_engage_start is None
            else self._arm_engage_start
        )
        engage_elapsed = (
            0.0
            if just_anchored or self._arm_engage_started_at is None
            else max(0.0, time.monotonic() - self._arm_engage_started_at)
        )
        conditioned_solution = _condition_arm_joint_targets(
            solution,
            previous_command=previous_arm_command,
            engage_start=engage_start,
            engage_elapsed=engage_elapsed,
            engage_duration=self.arm_engage_smooth_seconds,
            max_joint_speed=self.arm_max_joint_speed,
            control_dt=self._cmd_dt,
        )
        self._write_arm_solution(raw, conditioned_solution)

        raw[-2] = self._gripper_from_vr_buttons(
            "left",
            left_state.trigger_value,
            left_state.grip_value,
        )
        raw[-1] = self._gripper_from_vr_buttons(
            "right",
            right_state.trigger_value,
            right_state.grip_value,
        )
        raw = np.clip(raw, self._raw_low, self._raw_high)
        action = self._outer.normalize_action(raw)
        return np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)

    def reanchor_hands(self) -> None:
        """Keep engaged hands continuous after a manual VR-space recenter."""
        self._engage_ref = None
        self._arm_engage_started_at = None
        self._arm_engage_start = None

    def _stick(self, value: float) -> float:
        value = float(np.clip(value, -1.0, 1.0))
        return value if abs(value) >= self._stick_deadzone else 0.0

    def _gripper_from_vr_buttons(
        self,
        side: str,
        trigger_value: float,
        grip_value: float,
    ) -> float:
        # Hysteresis latch instead of round(trigger): an analog trigger held
        # near 0.5 makes a rounded command flip 0/1 every frame, which would
        # sweep the fingers open/close continuously (chatter).
        trigger = self._latched_hand_button(side, "trigger", trigger_value)
        # The G1 Dex1-1 is a one-scalar open/close gripper: the latched
        # trigger IS the command (the grip button drives pitch mode instead).
        return trigger

    def _latched_hand_button(self, side: str, name: str, value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        cmd = self._hand_button_cmd[side][name]
        if cmd < 0.5:
            if value > 0.6:
                cmd = 1.0
        elif value < 0.4:
            cmd = 0.0
        self._hand_button_cmd[side][name] = cmd
        return cmd

    def _current_ee_positions(self) -> dict[str, Any]:
        """World-frame hand site positions of the robot's current arm pose.

        Runs forward kinematics on the IK solver's internal model, posed
        like the robot.
        """
        ik = self._ik
        ik.data.bind(ik.arm_joints).qpos = np.concatenate(
            (
                self._arm_qpos(self._left_arm_names),
                self._arm_qpos(self._right_arm_names),
            )
        )
        fixed = self._fixed_qpos()
        if fixed is not None:
            ik.data.bind(ik.fixed_joints).qpos = fixed
        ik.set_pelvis_pose(self._pelvis_pose())
        mujoco.mj_forward(ik.model, ik.data)

        return {
            "left": ik.data.bind(ik.left_arm_site).xpos.copy(),
            "right": ik.data.bind(ik.right_arm_site).xpos.copy(),
            "left_quat": ik.site_quaternion(ik.left_arm_site),
            "right_quat": ik.site_quaternion(ik.right_arm_site),
        }

    def _pelvis_pose(self) -> Pose:
        """World pose of the robot's pelvis, as an IK ``Pose``."""
        from bigym.vr.ik.mink_upper_body_ik import Pose

        data = self._inner.data
        pelvis = self._inner.robot.pelvis
        return Pose(
            data.bind(pelvis).xpos.copy(), Quaternion(get_quaternion(data, pelvis))
        )

    def _write_base_command(
        self,
        raw: np.ndarray,
        *,
        vx: float,
        vy: float,
        yaw: float,
        z_rate: float = 0.0,
        pitch_rate: float = 0.0,
    ) -> None:
        desired_vx = float(vx)
        desired_vy = float(vy)
        desired_planar_moving = bool(np.hypot(desired_vx, desired_vy) > 0.0)
        if self._base_cmd_slew is not None:
            # Ramp-limit velocity commands toward the stick value so engage /
            # stick steps become smooth transitions (recorded as executed).
            dmax = self._base_cmd_slew * self._cmd_dt
            vx = float(np.clip(vx, self.last_vx - dmax, self.last_vx + dmax))
            vy = float(np.clip(vy, self.last_vy - dmax, self.last_vy + dmax))
        if desired_planar_moving:
            # Apply after optional slew so an executed non-zero command never
            # falls into GROOT's Balance/Walk gap. A direction reversal can
            # make the component-wise ramp land exactly on zero; use the new
            # desired direction for that one boundary frame.
            if self._lowerbody_backend == "groot_wbc_g1" and np.hypot(vx, vy) == 0.0:
                vx, vy = desired_vx, desired_vy
            vx, vy = _shape_planar_velocity_command(
                vx,
                vy,
                lowerbody_backend=self._lowerbody_backend,
            )
        elif (
            self._lowerbody_backend == "groot_wbc_g1"
            and np.hypot(vx, vy) < GROOT_WBC_MIN_WALK_SPEED
        ):
            # A slewed stop stays in Walk until the last short remainder,
            # then snaps to exact zero/Balance instead of traversing the gap.
            vx, vy = 0.0, 0.0
        if PelvisDof.X in self._outer_base_idx:
            raw[self._outer_base_idx[PelvisDof.X]] = float(vx)
        if PelvisDof.Y in self._outer_base_idx:
            raw[self._outer_base_idx[PelvisDof.Y]] = float(vy)
        if PelvisDof.Z in self._outer_base_idx:
            # Command-mode Z slot is the ABSOLUTE height command (not a delta).
            # Integrate the stick so releasing it holds the current height,
            # then clamp to the raw height bounds [height_cmd_min, _max].
            z_idx = self._outer_base_idx[PelvisDof.Z]
            z_high = float(self._raw_high[z_idx])
            if self._height_cmd_cap is not None:
                z_high = min(z_high, self._height_cmd_cap)
            self._height_cmd = float(
                np.clip(
                    self._height_cmd + float(z_rate) * self._base_z_scale,
                    float(self._raw_low[z_idx]),
                    z_high,
                )
            )
            raw[z_idx] = self._height_cmd
        if self.pitch_slot:
            # RY slot: ABSOLUTE torso-pitch command, same integrate-
            # and-hold semantics as height.
            ry_idx = self._outer_base_idx[PelvisDof.RY]
            self._pitch_cmd = float(
                np.clip(
                    self._pitch_cmd + float(pitch_rate) * self._pitch_scale,
                    float(self._raw_low[ry_idx]),
                    float(self._raw_high[ry_idx]),
                )
            )
            raw[ry_idx] = self._pitch_cmd
            self.last_pitch = self._pitch_cmd
        if PelvisDof.RZ in self._outer_base_idx:
            rz_idx = self._outer_base_idx[PelvisDof.RZ]
            if self._yaw_mode == "base":
                wz_cmd = float(yaw * self._base_wz_scale)
                if self._base_cmd_slew is not None:
                    dmax = self._base_cmd_slew * self._cmd_dt
                    wz_cmd = float(
                        np.clip(
                            wz_cmd,
                            self._last_wz_cmd - dmax,
                            self._last_wz_cmd + dmax,
                        )
                    )
                self._last_wz_cmd = wz_cmd
                raw[rz_idx] = wz_cmd
            else:
                raw[rz_idx] = 0.0
        self.last_vx = float(vx)
        self.last_vy = float(vy)
        self.last_yaw = float(yaw)
        self.last_height = float(self._height_cmd)

    # Waist joints whose MEASURED angle is fed to the IK; the others stay at
    # zero in the solver. Only the commanded pitch is synced: while turning
    # in place GR00T rocks the waist at step frequency (roll +-5 deg, yaw
    # +-2 deg), and syncing that would make the arms counter-move every step - a visible
    # tremor of the gripper. Letting the hands ride the rocking (zero-mean)
    # keeps them smooth; the slow, commanded pitch is what moves the
    # shoulders by centimetres and must be tracked.
    _WAIST_SYNC_JOINTS = ("waist_pitch_joint",)

    def _fixed_qpos(self) -> np.ndarray | None:
        """Angles for the IK's fixed (waist) joints: measured pitch, zero roll/yaw."""
        names = self._ik.fixed_joint_names
        if not names or any(n not in self._full_limb_idx for n in names):
            return None
        measured = self._arm_qpos(names).astype(np.float64)
        keep = np.array([n in self._WAIST_SYNC_JOINTS for n in names])
        return np.where(keep, measured, 0.0)

    def _arm_qpos(self, names: tuple[str, ...]) -> np.ndarray:
        robot = self._inner.robot
        base = robot.floating_base
        base_dofs = int(base.dof_amount) if base is not None else 0
        qpos = np.asarray(robot.qpos_actuated, dtype=np.float32)
        return np.asarray(
            [qpos[base_dofs + int(self._full_limb_idx[name])] for name in names],
            dtype=np.float32,
        )

    def _arm_command_from_raw(self, raw: np.ndarray) -> np.ndarray:
        base_dim = len(self._outer_base_dofs)
        return np.asarray(
            [raw[base_dim + self._outer_limb_idx[name]] for name in self._arm_names],
            dtype=np.float32,
        )

    def _write_arm_solution(self, raw: np.ndarray, solution: np.ndarray) -> None:
        base_dim = len(self._outer_base_dofs)
        for name, value in zip(self._arm_names, solution, strict=True):
            outer_i = self._outer_limb_idx.get(name)
            if outer_i is None:
                continue
            raw[base_dim + outer_i] = float(value)
