"""Shared environment helpers for the coding-agent benchmark.

Used by the env server (agent side, through a socket) and by the evaluator
(our side, in-process). Both build observations and convert actions with the
same code so a policy behaves identically in development and in evaluation.

Physical action layout (20 dims, ``raw``):
    [0] vx m/s body frame   [1] vy m/s   [2] pelvis height m (abs)   [3] wz rad/s
    [4:11]  left arm joint targets rad  (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)
    [11:18] right arm joint targets rad (same order)
    [18] left gripper 0=open 1=closed   [19] right gripper

Tasks whose protocol row enables the torso-pitch command insert a ``pitch``
slot after ``wz`` and the layout is 21 dims; :func:`task_pitch_enabled`
answers which layout a task uses.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from bigym.const import HandSide
from bigym.loco.eval.protocol import EVAL_SEED_RESERVED
from bigym.utils.physics_utils import get_quaternion

from .cli import EnvToolsConfig
from .geometry import pixel_to_ray

# The task and env modules are imported where they are used so that importing
# this module stays light, as importing bigym.loco does.

# The hidden evaluation block, offsets included: the server refuses these seeds
# to the agent, and the evaluator draws from them.
EVAL_SEED_LO, EVAL_SEED_HI = EVAL_SEED_RESERVED
CAMERA_KEYS = ("head", "right_wrist", "left_wrist")
BASE_SLOTS = ("vx", "vy", "height", "wz")
ARM_JOINTS = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)
FIXTURE_TASKS = (
    "drawer_top_open",
    "drawer_top_close",
    "dishwasher_close",
    "dishwasher_load_cups",
    "pick_box",
    "dishwasher_open",
    "drawers_open_all",
    "drawers_close_all",
)
# Tasks with a privileged (object-state) observation. Any protocol task works
# in the images tier, which is what the benchmark itself uses.
TASKS = ("reach_target_single", "move_plate")


def task_pitch_enabled(task: str) -> bool:
    """Whether the task's official configuration enables the torso-pitch command.

    The official configuration decides the action layout: with
    ``controller.pitch_command`` the outer action carries a torso-pitch command
    (21 dims, 56-dim proprioception), otherwise 20/50. The agent must get the
    same layout as the learned baselines and the demonstrations of that task.

    Args:
        task: Task name.

    Returns:
        True when the task is on the 21-dim layout.
    """
    from bigym.loco.tasks import task_config

    controller = task_config(task).controller
    return controller is not None and controller.pitch_command


def make_env(task: str, pitch: bool | None = None):
    """Build the benchmark environment for a task.

    Args:
        task: Task name.
        pitch: None follows the official configuration (the benchmark layout).
            True or False overrides the torso-pitch command, which is only
            useful to replay a submission written against the other layout.

    Returns:
        The controller-in-the-loop env from :func:`bigym.loco.make`.
    """
    from bigym.loco import make

    if pitch is None:
        return make(task)
    return make(task, controller={"pitch_command": bool(pitch)})


def _basename(name: str) -> str:
    return name.split("/")[-1]


class EnvTools:
    """Wrap a protocol env: physical actions in, structured observations out."""

    def __init__(self, task: str, config: EnvToolsConfig, env=None):
        """Build the wrapper (and the env itself unless one is passed in).

        Args:
            task: Task name.
            config: The environment settings; the interface decides which
                tools exist at all (see the class notes below).
            env: An existing protocol env, or None to build one.

        Raises:
            ValueError: A privileged tier asked for a task with no privileged
                observation.
        """
        self.config = config
        self.image_cap = config.image_size
        # Interface presets:
        #   strict (default): the policy signs the same I/O contract as the learned
        #     baselines - the three 84x84 cameras plus the raw low_dim_obs vector in,
        #     the physical action out. No named state fields, no wrist positions, no
        #     camera_info / pixel_to_ray, no IK. The robot model and the camera
        #     calibration are things the baselines never receive either.
        #   tools: named state, wrist positions, calibration and IK on top, reported
        #     as a separately labelled row so the gain from the geometry stack is
        #     visible instead of being folded into the headline number.
        self.interface = config.interface
        self.named_state = config.interface == "tools"
        self.hand_pos = self.named_state and config.hand_pos
        self.allow_ik = self.named_state and config.ik
        self.calibration = self.named_state
        if config.tier == "privileged" and task not in TASKS:
            raise ValueError(
                f"no privileged observation implemented for {task!r}; "
                f"use tier='images' (known: {TASKS})"
            )
        self.task = task
        self.tier = config.tier
        self.env = (
            env if env is not None else make_env(task, None if config.pitch else False)
        )
        self.outer = self.env.bigym
        self.inner = self.outer.inner_env
        self.model, self.data = self.inner.model, self.inner.data
        self.robot = self.inner.robot
        from bigym.action_modes import PelvisDof

        self._HandSide = HandSide
        self._base_idx = {
            dof: i for i, dof in enumerate(self.outer.outer_action_floating_dofs)
        }
        self._slot = {
            "vx": self._base_idx[PelvisDof.X],
            "vy": self._base_idx[PelvisDof.Y],
            "height": self._base_idx[PelvisDof.Z],
            "wz": self._base_idx[PelvisDof.RZ],
        }
        if PelvisDof.RY in self._base_idx:  # torso-pitch command slot (21-dim layout)
            self._slot["pitch"] = self._base_idx[PelvisDof.RY]
        self.n_base = len(self._base_idx)
        self.limb_names = [
            _basename(n) for n in (self.outer.outer_limb_actuator_names or ())
        ]
        self.action_dim = self.n_base + len(self.limb_names) + len(self.robot.grippers)
        self.time_limit = (
            self.outer.config.episode_length // self.outer.config.demo_down_sample_rate
        )
        # Scripted policies speak physical units: map over the true bounds.
        layout = self.outer.wholebody_action_layout()
        low, high = layout["action_low"], layout["action_high"]
        self.outer.set_action_stats(low, high)
        self.raw_low, self.raw_high = low, high
        self.control_dt = 0.02
        self._ik = None
        self._limb_full_idx = self.outer.limb_name_to_full_index or {}
        self._renderer_ready = False
        self._img_renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._sites = {}
        self.step_count = 0
        self.last_timestep = None
        # Command rate limiter, OFF by default: the learned baselines' actions reach the
        # position controller unfiltered (no slew anywhere in bigym.loco for the G1
        # adapter), so the agent's do too. When on, the numbers are the teleoperator's
        # (arm joints 6 rad/s, base command slew 0.7/s, both recorded in the demo
        # metadata) plus an acceleration cap. It is a flag so that submissions developed
        # with the limiter on can be replayed.
        self.slew = config.slew
        self.onboard_only = not config.allow_external_cameras
        self.slew_joint_rad_s = float(config.joint_vmax)
        self.slew_base_per_s = 0.7
        slew_limit = np.full(self.action_dim, np.inf, dtype=np.float32)
        slew_limit[: self.n_base] = self.slew_base_per_s * self.control_dt
        slew_limit[self.n_base : self.n_base + len(self.limb_names)] = (
            self.slew_joint_rad_s * self.control_dt
        )
        self.slew_limit = slew_limit
        # Arm joint targets are also acceleration-limited so a target jump becomes a
        # trapezoidal velocity profile instead of full-speed-then-stop. 0.03 rad/step^2
        # is the 99th percentile of the human teleoperator's commands in the
        # demonstrations; None restores pure velocity capping.
        self.slew_accel_rad_step2 = (
            None if config.accel is None else float(config.accel)
        )
        acc = np.full(self.action_dim, np.inf, dtype=np.float32)
        if self.slew_accel_rad_step2 is not None:
            acc[self.n_base : self.n_base + len(self.limb_names)] = (
                self.slew_accel_rad_step2
            )
        self._slew_acc = acc
        # Optional first-order low-pass on the arm targets (alpha per step) before the
        # caps: removes the high-frequency dithering of scripts that re-target every
        # step from noisy perception. None = off.
        self.slew_lowpass_alpha = (
            None if config.lowpass is None else float(config.lowpass)
        )
        self._lp = None
        self._prev_cmd = None
        self._prev_vel = None

    # ------------------------------------------------------------------ state
    def _joint_qpos(self, basenames) -> np.ndarray:
        base = self.robot.floating_base
        base_dofs = int(base.dof_amount) if base is not None else 0
        qpos = np.asarray(self.robot.qpos_actuated, dtype=np.float32)
        full = {_basename(k): v for k, v in self._limb_full_idx.items()}
        return np.asarray(
            [qpos[base_dofs + int(full[n])] for n in basenames], dtype=np.float32
        )

    def arm_qpos(self, side: str) -> np.ndarray:
        """Return the 7 measured arm joint angles of one side.

        Args:
            side: ``left`` or ``right``.

        Returns:
            The joint angles in rad, in the action's order.
        """
        return self._joint_qpos([f"{side}_{j}_joint" for j in ARM_JOINTS])

    def _hand(self, side: str):
        return self.robot.grippers[
            self._HandSide.LEFT if side == "left" else self._HandSide.RIGHT
        ]

    def pelvis_pose(self):
        """Return the pelvis position (3,) and quaternion (4,) in the world frame."""
        pelvis = self.robot.pelvis
        return (
            self.data.bind(pelvis).xpos.copy(),
            get_quaternion(self.data, pelvis),
        )

    def observation(self) -> dict[str, Any]:
        """Return the structured, read-only view of the state: numbers, no handles."""
        pos, quat = self.pelvis_pose()
        yaw = float(quat_to_yaw(quat))
        ts = self.last_timestep
        obs = {
            "t": int(self.step_count),
            "time_limit": self.time_limit,
            "fell": bool(self.outer.episode_fell()),
            "low_dim_obs": None
            if ts is None
            else np.asarray(ts.low_dim_obs, dtype=np.float32),
        }
        if (
            self.named_state
        ):  # tools interface: the same quantities under readable names
            obs.update(
                {
                    "base_pos": pos.astype(np.float32),
                    "base_yaw": yaw,
                    "base_quat": quat.astype(np.float32),
                    "left_arm_qpos": self.arm_qpos("left"),
                    "right_arm_qpos": self.arm_qpos("right"),
                    "left_gripper": float(np.asarray(self._hand("left").qpos).item()),
                    "right_gripper": float(np.asarray(self._hand("right").qpos).item()),
                }
            )
        if self.hand_pos:
            obs["left_hand_pos"] = np.asarray(
                self._hand("left").wrist_position, dtype=np.float32
            )
            obs["right_hand_pos"] = np.asarray(
                self._hand("right").wrist_position, dtype=np.float32
            )
        if self.tier == "privileged":
            obs.update(self._task_obs())
        return obs

    def _task_obs(self) -> dict[str, Any]:
        inner = self.inner
        if self.task == "reach_target_single":
            return {
                "target_pos": np.asarray(
                    inner.targets[0].get_position(), dtype=np.float32
                )
            }
        if self.task == "move_plate":
            plate = inner.plates[0]
            up = np.asarray(
                _rotate(plate.get_quaternion(), np.array([0.0, 0.0, 1.0])),
                dtype=np.float32,
            )
            return {
                "plate_pos": np.asarray(plate.get_position(), dtype=np.float32),
                "plate_quat": np.asarray(plate.get_quaternion(), dtype=np.float32),
                "plate_up_axis": up,
                "rack_start_pos": np.asarray(
                    inner.rack_start.get_position(), dtype=np.float32
                ),
                "rack_target_pos": np.asarray(
                    inner.rack_target.get_position(), dtype=np.float32
                ),
                "rack_start_sites": np.asarray(
                    [self.data.bind(s).xpos for s in inner.rack_start.sites],
                    dtype=np.float32,
                ),
                "rack_target_sites": np.asarray(
                    [self.data.bind(s).xpos for s in inner.rack_target.sites],
                    dtype=np.float32,
                ),
                "rack_target_site_quats": np.asarray(
                    [get_quaternion(self.data, s) for s in inner.rack_target.sites],
                    dtype=np.float32,
                ),
            }
        return {}

    # ---------------------------------------------------------------- actions
    def hold_action(self) -> np.ndarray:
        """Return the physical action that holds the current pose, zero velocity."""
        raw = self.outer.raw_hold_action().astype(np.float32)
        raw[self._slot["vx"]] = 0.0
        raw[self._slot["vy"]] = 0.0
        raw[self._slot["wz"]] = 0.0
        return raw

    def to_normalized(self, raw: np.ndarray) -> np.ndarray:
        """Convert a physical action into the env's normalised action.

        Args:
            raw: The physical action.

        Returns:
            The normalised action, clipped to [-1, 1].

        Raises:
            ValueError: Wrong shape, or a non-finite entry.
        """
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape != (self.action_dim,):
            raise ValueError(
                f"raw action must have shape ({self.action_dim},), got {raw.shape}"
            )
        if not np.all(np.isfinite(raw)):
            raise ValueError("raw action contains NaN/inf")
        return np.clip(self.outer.normalize_action(raw), -1.0, 1.0).astype(np.float32)

    def reset(self, seed: int):
        """Reset the env for one episode.

        Args:
            seed: Episode seed.

        Returns:
            The first timestep.
        """
        self.step_count = 0
        self.last_timestep = self.env.reset(seed=int(seed))
        self._prev_cmd = self.hold_action() if self.slew else None
        self._prev_vel = (
            np.zeros(self.action_dim, dtype=np.float32) if self.slew else None
        )
        self._lp = None
        if self._ik is not None:
            self._ik.reset_seed()
        return self.last_timestep

    def apply_slew(self, raw: np.ndarray) -> np.ndarray:
        """Rate-limit the physical command against the previous one.

        Args:
            raw: The requested physical action.

        Returns:
            The command actually sent (``raw`` itself when slew is off).
        """
        raw = np.asarray(raw, dtype=np.float32)
        if not self.slew or self._prev_cmd is None or raw.shape != self._prev_cmd.shape:
            return raw
        if self.slew_lowpass_alpha is not None:
            sl = slice(self.n_base, self.n_base + len(self.limb_names))
            if self._lp is None:
                self._lp = raw.copy()
            self._lp[sl] = self._lp[sl] + self.slew_lowpass_alpha * (
                raw[sl] - self._lp[sl]
            )
            raw = raw.copy()
            raw[sl] = self._lp[sl]
        delta = raw - self._prev_cmd
        v_allow = self.slew_limit.copy()
        fin = np.isfinite(self._slew_acc)
        # Brake early enough to stop exactly on the target with per-step decrements of a:
        # the largest v whose braking series v + (v-a) + ... fits in |delta| is
        # v = (-a + sqrt(a^2 + 8 a |delta|)) / 2.
        a = self._slew_acc[fin]
        v_allow[fin] = np.minimum(
            v_allow[fin], (-a + np.sqrt(a * a + 8.0 * a * np.abs(delta[fin]))) / 2.0
        )
        v = np.clip(delta, -v_allow, v_allow)
        if self._prev_vel is not None:
            v = np.clip(
                v, self._prev_vel - self._slew_acc, self._prev_vel + self._slew_acc
            )
        self._prev_vel = v.astype(np.float32)
        self._prev_cmd = (self._prev_cmd + v).astype(np.float32)
        return self._prev_cmd

    def slew_params(self):
        """Return the rate limiter's parameters, or None when it is off."""
        if not self.slew:
            return None
        return {
            "joint_rad_per_s": self.slew_joint_rad_s,
            "base_per_s": self.slew_base_per_s,
            "dt": self.control_dt,
            "joint_accel_rad_per_step2": self.slew_accel_rad_step2,
            "lowpass_alpha": self.slew_lowpass_alpha,
        }

    def step(self, raw: np.ndarray):
        """Apply one physical action.

        Args:
            raw: The physical action.

        Returns:
            The resulting timestep.
        """
        ts = self.env.step(self.to_normalized(self.apply_slew(raw)))
        self.step_count += 1
        self.last_timestep = ts
        if (
            self._recorder_process is not None
            and self.step_count % self._record_every == 0
        ):
            self._rec_frame()
        return ts

    # -------------------------------------------------------------- recording
    _recorder_process: subprocess.Popen[bytes] | None = None
    _record_every = 2
    _rec_views = None

    def start_recording(self, path, every: int = 2, fps: int = 25, views=None):
        """Record two views side by side to an H.264 mp4 (via ffmpeg).

        Args:
            path: Destination mp4.
            every: Record one frame per this many control steps.
            fps: Frame rate of the mp4.
            views: Overrides the per-task default view pair.
        """
        self._rec_views = tuple(views) if views else None
        self.stop_recording()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        w, h = 640, 480
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{2 * w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path),
        ]  # fmt: skip
        self._recorder_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self._record_every = int(every)
        self._rec_path = path
        self._rec_frame()

    def _rec_frame(self):
        try:
            views = self._rec_views or (
                ("rec_third", "rec_side")
                if self.task in FIXTURE_TASKS
                else ("third_person", "front")
            )
            img = np.concatenate([self.render(views[0]), self.render(views[1])], axis=1)
            assert (
                self._recorder_process is not None
                and self._recorder_process.stdin is not None
            )
            self._recorder_process.stdin.write(
                np.ascontiguousarray(img, dtype=np.uint8).tobytes()
            )
        except Exception:
            self.stop_recording()

    def stop_recording(self):
        """Close a running recording, if any."""
        if self._recorder_process is not None:
            try:
                assert self._recorder_process.stdin is not None
                self._recorder_process.stdin.close()
                self._recorder_process.wait(timeout=60)
            except Exception:
                pass
            self._recorder_process = None

    # --------------------------------------------------------------------- IK
    def ik(
        self,
        left_pos=None,
        left_quat=None,
        right_pos=None,
        right_quat=None,
        iters: int = 10,
    ) -> np.ndarray:
        """Solve arm joints so the wrist sites reach world-frame targets.

        Args:
            left_pos: Left wrist target, or None to keep its current position.
            left_quat: Left wrist orientation (w, x, y, z), or None for free.
            right_pos: Right wrist target, or None to keep its current position.
            right_quat: Right wrist orientation (w, x, y, z), or None for free.
            iters: Differential-IK passes, so a far target converges in one call.

        Returns:
            14 joint angles (left 7, right 7) for the action's arm slots.

        Raises:
            RuntimeError: The interface withholds the solver.
        """
        if not self.allow_ik:
            raise RuntimeError(
                "tools.ik is not available under this interface: drive the arms in joint "
                "space (raw arm slots are absolute joint targets in rad; low_dim_obs gives "
                "the measured angles)"
            )
        from bigym.vr.ik.mink_upper_body_ik import Pose  # imports mink

        if self._ik is None:
            from bigym.vr.ik.g1_upper_body_ik import G1UpperBodyIK

            self._ik = G1UpperBodyIK(self.inner)
        ik = self._ik
        pelvis_position, pelvis_quaternion = self.pelvis_pose()
        pelvis = Pose(pelvis_position, Quaternion(pelvis_quaternion))
        names = tuple(ik.fixed_joint_names)
        fixed = None
        if names:
            measured = self._joint_qpos(list(names)).astype(np.float64)
            fixed = np.where(
                np.array([n == "waist_pitch_joint" for n in names]), measured, 0.0
            )
        q_left, q_right = self.arm_qpos("left"), self.arm_qpos("right")
        want = {
            "left": (
                None if left_pos is None else np.asarray(left_pos, dtype=np.float64),
                None
                if left_quat is None
                else Quaternion(np.asarray(left_quat, dtype=np.float64)),
            ),
            "right": (
                None if right_pos is None else np.asarray(right_pos, dtype=np.float64),
                None
                if right_quat is None
                else Quaternion(np.asarray(right_quat, dtype=np.float64)),
            ),
        }
        # Seed the IK model so its wrist poses are meaningful before the first solve.
        ik.seed(pelvis, np.concatenate((q_left, q_right)).astype(np.float64), fixed)
        for side in ("left", "right"):
            ik.set_orientation_cost(side, 0.0 if want[side][1] is None else 2.0)
        real_pos = {
            "left": np.asarray(self._hand("left").wrist_position, dtype=np.float64),
            "right": np.asarray(self._hand("right").wrist_position, dtype=np.float64),
        }

        solution = None
        for _ in range(max(1, int(iters))):
            targets = {}
            for side in ("left", "right"):
                pos, quat = want[side]
                current_quaternion = ik.wrist_pose(side).orientation
                # an unspecified side is anchored at its real (measured) wrist position,
                # orientation free
                targets[side] = Pose(
                    real_pos[side] if pos is None else pos,
                    current_quaternion if quat is None else quat,
                )
            solution = ik.solve(
                pelvis_pose=pelvis,
                qpos_arm_left=q_left,
                qpos_arm_right=q_right,
                target_pose_left=targets["left"],
                target_pose_right=targets["right"],
                fixed_qpos=fixed,
            )
        return np.asarray(solution, dtype=np.float32)

    # ----------------------------------------------------------------- render
    def render(
        self, camera: str = "third_person", width: int = 640, height: int = 480
    ) -> np.ndarray:
        """Return an RGB uint8 image.

        Args:
            camera: head | right_wrist | left_wrist (robot-mounted) |
                third_person | front (agent-facing, fixed definition) |
                rec_third | rec_side | both_tables (recording-only views that
                keep the hands and a fixture in front of the robot in frame).
            width: Image width for the free cameras.
            height: Image height for the free cameras.

        Returns:
            A ``(height, width, 3)`` uint8 array.
        """
        if camera in CAMERA_KEYS and self.last_timestep is not None:
            idx = CAMERA_KEYS.index(camera)
            img = np.asarray(self.last_timestep.rgb_obs[idx])  # (3, 84, 84)
            return np.transpose(img, (1, 2, 0)).copy()
        inner = self.inner
        rend = inner.mujoco_renderer
        rend.width, rend.height = int(width), int(height)
        self.model.vis.global_.offwidth = max(
            int(self.model.vis.global_.offwidth), int(width)
        )
        self.model.vis.global_.offheight = max(
            int(self.model.vis.global_.offheight), int(height)
        )
        inner.camera_id = -1
        viewer = rend.get_viewer("rgb_array")
        viewer.vopt = mujoco.MjvOption()
        cam = mujoco.MjvCamera()
        pos, quat = self.pelvis_pose()
        yaw = float(quat_to_yaw(quat))
        yaw_deg = float(np.degrees(yaw))
        if camera == "both_tables":
            # World-fixed, not pelvis-relative: on the two-table layouts every
            # robot-following view cuts the far table out of frame. Recording only: it
            # is in no policy-facing camera whitelist (the server's camera lists, and
            # render_png refuses anything outside CAMERA_KEYS under onboard_only).
            cam.lookat[:] = [1.05, 0.0, 1.0]
            cam.distance, cam.elevation, cam.azimuth = 4.4, -24.0, 150.0
        elif camera in ("rec_third", "rec_side"):
            # over the shoulder from behind-left, and from the left side at hand height
            ahead = np.array([np.cos(yaw), np.sin(yaw)]) * 0.55
            cam.lookat[:] = [pos[0] + ahead[0], pos[1] + ahead[1], 0.85]
            if camera == "rec_side":
                cam.distance, cam.elevation, cam.azimuth = 1.7, -8.0, yaw_deg + 90.0
            else:
                cam.distance, cam.elevation, cam.azimuth = 2.3, -28.0, yaw_deg + 35.0
        else:
            cam.lookat[:] = [pos[0], pos[1], 0.7]
            if camera == "front":
                cam.distance, cam.elevation, cam.azimuth = (
                    1.6,
                    -12.0,
                    yaw_deg + 180.0 - 40.0,
                )
            else:
                cam.distance, cam.elevation, cam.azimuth = (
                    2.6,
                    -20.0,
                    yaw_deg + 180.0 + 35.0,
                )
        viewer.cam = cam
        return np.asarray(inner.render()).copy()

    def _cam_id(self, name: str) -> int:
        return int(self.inner._cameras_map[name][0])

    def image(
        self, camera: str = "head", width: int = 84, height: int = 84
    ) -> np.ndarray:
        """Return an RGB uint8 image at the requested resolution.

        Args:
            camera: A robot-mounted camera (head/left_wrist/right_wrist), or a
                free camera (third_person/front) when onboard_only is off.
            width: Requested width.
            height: Requested height.

        Returns:
            A ``(height, width, 3)`` uint8 array.

        Raises:
            ValueError: Above the resolution cap, or a camera the policy may
                not use.
        """
        # The cap is the policy-facing resolution ceiling, not a property of the camera:
        # a MuJoCo camera carries only a vertical fovy (60 deg here) and takes its aspect
        # ratio from the render buffer, so there is no "native" size. 84x84 is what the
        # learned baselines are rendered at; a 4:3 cap would also widen the HORIZONTAL
        # field of view by 25% (60.0 -> 75.2 deg), a square cap does not.
        cw, ch = self.image_cap
        width, height = int(width), int(height)
        if width > cw or height > ch or width < 1 or height < 1:
            # Refuse rather than clamp: a silently shrunk image fed to pixel_to_ray with
            # the requested width/height gives wrong rays with no error.
            raise ValueError(
                f"requested {width}x{height}; the largest image the policy may request "
                f"is {cw}x{ch}"
            )
        if camera not in CAMERA_KEYS:
            if self.onboard_only:
                raise ValueError(
                    f"camera {camera!r} is not mounted on the robot; policies may use "
                    f"{CAMERA_KEYS} only"
                )
            return self.render(camera, width=width, height=height)
        key = (height, width)
        cache = self._img_renderers
        if key not in cache:
            self.model.vis.global_.offwidth = max(
                int(self.model.vis.global_.offwidth), width
            )
            self.model.vis.global_.offheight = max(
                int(self.model.vis.global_.offheight), height
            )
            cache[key] = mujoco.Renderer(self.model, height=height, width=width)
        r = cache[key]
        r.update_scene(self.data, self._cam_id(camera))
        return np.asarray(r.render()).copy()

    def image_png(
        self, camera: str = "head", width: int = 84, height: int = 84
    ) -> bytes:
        """Return :meth:`image` encoded as a PNG.

        Args:
            camera: Camera name.
            width: Requested width.
            height: Requested height.

        Returns:
            The PNG bytes.
        """
        buf = io.BytesIO()
        Image.fromarray(self.image(camera, width, height)).save(buf, format="PNG")
        return buf.getvalue()

    def camera_info(self) -> dict[str, Any]:
        """Return the calibration of every camera.

        Reports the vertical field of view (deg) and the current world pose
        (position, 3x3 rotation whose columns are the camera x=right, y=up,
        z=backward axes, MuJoCo convention: the camera looks along -z).

        Returns:
            One entry per camera.

        Raises:
            RuntimeError: The interface withholds the calibration.
        """
        if not self.calibration:
            raise RuntimeError(
                "camera calibration is not available under this interface: the policy "
                "sees the three cameras as pixels only, like the learned baselines"
            )
        out = {}
        for name in CAMERA_KEYS:
            cid = self._cam_id(name)
            out[name] = {
                "fovy_deg": float(self.model.cam_fovy[cid]),
                "pos": np.asarray(self.data.cam_xpos[cid], dtype=np.float32),
                "rot": np.asarray(self.data.cam_xmat[cid], dtype=np.float32).reshape(
                    3, 3
                ),
                "default_resolution": [84, 84],
                "mounted_on": "robot",
            }
        # free cameras: defined relative to the pelvis (see render()); report their pose
        for name in () if self.onboard_only else ("third_person", "front"):
            self.render(name)  # updates viewer.cam
            viewer = self.inner.mujoco_renderer.get_viewer("rgb_array")
            cam = mujoco.MjvCamera()
            cam.type, cam.lookat, cam.distance, cam.azimuth, cam.elevation = (
                viewer.cam.type,
                viewer.cam.lookat,
                viewer.cam.distance,
                viewer.cam.azimuth,
                viewer.cam.elevation,
            )
            az, el = np.radians(cam.azimuth), np.radians(cam.elevation)
            forward = np.array(
                [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
            )
            pos = np.asarray(cam.lookat) - cam.distance * forward
            out[name] = {
                "fovy_deg": float(self.model.vis.global_.fovy),
                "pos": pos.astype(np.float32),
                "lookat": np.asarray(cam.lookat, dtype=np.float32),
                "distance": float(cam.distance),
                "azimuth_deg": float(cam.azimuth),
                "elevation_deg": float(cam.elevation),
                "default_resolution": [640, 480],
                "mounted_on": (
                    "free camera that follows the pelvis (lookat = pelvis xy, z 0.7; "
                    "azimuth relative to base yaw)"
                ),
            }
        return out

    def render_png(self, camera: str = "third_person", **kw) -> bytes:
        """Return a policy-facing PNG render.

        Args:
            camera: Camera name; robot-mounted only under the onboard-only
                interface.
            **kw: Forwarded to :meth:`render`.

        Returns:
            The PNG bytes.

        Raises:
            ValueError: A camera the policy may not use.
        """
        if self.onboard_only and camera not in CAMERA_KEYS:
            raise ValueError(
                f"camera {camera!r} is not mounted on the robot; policies may use "
                f"{CAMERA_KEYS} only"
            )
        return self.view_png(camera, **kw)

    def view_png(self, camera: str = "third_person", **kw) -> bytes:
        """Return an unrestricted render, for our own recordings only.

        Not reachable from the sandbox: the server has no camera op that
        reaches it. An outside view of the scene during development is scene
        information the learned baselines never get.

        Args:
            camera: Camera name.
            **kw: Forwarded to :meth:`render`.

        Returns:
            The PNG bytes.
        """
        img = self.render(camera, **kw)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        return buf.getvalue()

    def info(self) -> dict[str, Any]:
        """Return the static description of the interface handed to the policy."""
        return {
            "task": self.task,
            "action_dim": self.action_dim,
            "base_slots": sorted(self._slot, key=lambda k: self._slot[k]),
            "pitch_enabled": "pitch" in self._slot,
            "arm_slice": [self.n_base, self.n_base + len(self.limb_names)],
            "gripper_slice": [self.n_base + len(self.limb_names), self.action_dim],
            "limb_names": list(self.limb_names),
            "time_limit": self.time_limit,
            "control_dt": self.control_dt,
            "height_range": [0.4, 1.0],
            "raw_low": self.raw_low,
            "raw_high": self.raw_high,
            "vx_vy_range": [-1.0, 1.0],
            "wz_range": [-1.0, 1.0],
            "min_walk_speed": 0.055,
            "tier": self.tier,
            "interface": self.interface,
            "hand_pos": self.hand_pos,
            "ik": self.allow_ik,
            "calibration": self.calibration,
            "image_cap": list(self.image_cap),
            "slew": self.slew_params(),
        }

    def close(self):
        """Close the wrapped environment."""
        self.env.close()


class LocalEnv:
    """The client's ``Env`` interface, backed by an in-process EnvTools."""

    def __init__(self, tools: EnvTools):
        """Wrap an :class:`~bigym.loco.agent.envtools.EnvTools`.

        Args:
            tools: The wrapper holding the environment.
        """
        self.t = tools
        self.info = tools.info()

    def reset(self, seed):
        """Reset the env and return the first observation.

        Args:
            seed: Episode seed.

        Returns:
            The observation dict.
        """
        self.t.reset(int(seed))
        return self.t.observation()

    def step(self, raw):
        """Apply one physical action.

        Args:
            raw: The physical action.

        Returns:
            ``(obs, reward, done, info)``.
        """
        ts = self.t.step(np.asarray(raw, dtype=np.float32))
        done = bool(ts.last())
        info = {"termination": None}
        if done:
            info["termination"] = "pending"  # filled by the caller from the env
        return self.t.observation(), float(ts.reward or 0.0), done, info

    def hold_action(self):
        """Return the action that holds the current pose."""
        return self.t.hold_action()

    def ik(self, **kw):
        """Return the arm joint targets reaching world-frame wrist targets."""
        return self.t.ik(**kw)

    def image(self, camera="head", width=84, height=84):
        """Return an RGB uint8 array from a robot-mounted camera.

        Args:
            camera: Camera name.
            width: Requested width.
            height: Requested height.

        Returns:
            A ``(height, width, 3)`` uint8 array.
        """
        return self.t.image(camera, width, height)

    def camera_info(self):
        """Return the calibration of every camera."""
        return self.t.camera_info()

    def pixel_to_ray(self, camera, u, v, width, height):
        """Return the world-frame ray through a pixel of a rendered image.

        Args:
            camera: Camera the image came from.
            u: Pixel column.
            v: Pixel row.
            width: Width the image was rendered at.
            height: Height the image was rendered at.

        Returns:
            ``(origin, unit direction)``.
        """
        return pixel_to_ray(self.t.camera_info()[camera], u, v, width, height)

    def render(self, camera="head", path=None):
        """Save a PNG from one of the robot's cameras.

        Args:
            camera: Camera name.
            path: Destination, or None to do nothing.

        Returns:
            The path written, or None.
        """
        if path is None:
            return None
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(self.t.render_png(camera))
        return str(path)


def quat_to_yaw(q) -> float:
    """Return the yaw angle (rad) of a (w, x, y, z) quaternion.

    Args:
        q: Quaternion in (w, x, y, z) order.

    Returns:
        The yaw in rad, 0 along +x.
    """
    w, x, y, z = [float(v) for v in q]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _rotate(q, v):
    return Quaternion(np.asarray(q, dtype=np.float64)).rotate(
        np.asarray(v, dtype=np.float64)
    )
