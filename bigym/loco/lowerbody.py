"""The lower-body controller in the loop: commands, reset settling and state.

:class:`LowerBody` is owned by :class:`~bigym.loco.env.BiGym`. It reads the
:class:`~bigym.loco.config.ControllerConfig`, builds the backend's controller
through the backend binding (:mod:`bigym.loco.adapters`), turns the outer
action's base slots into controller commands and merges the controller's
joint targets into the inner action. The outer action layout lives on the
env, so the methods that map between the outer and the inner action take
the env as their first argument.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Optional

import mujoco
import numpy as np

from bigym.action_modes import PelvisDof
from bigym.bigym_env import CONTROL_FREQUENCY_MAX, BiGymEnv
from bigym.loco.adapters import backend_binding, resolve_backend_name
from bigym.loco.config import ControllerConfig
from bigym.loco.controller import LowerBodyController

if TYPE_CHECKING:
    from bigym.loco.env import BiGym
    from bigym.robots.robot import Robot


def _or_default(value: Optional[float], default: float) -> float:
    return float(default if value is None else value)


def _mujoco_basename(full_identifier: str | None) -> str:
    if not full_identifier:
        return ""
    return full_identifier.split("/")[-1]


class LowerBody:
    """Controller settings, the controller, and the command state between steps."""

    def __init__(self, controller: Optional[ControllerConfig], *, robot_model: str):
        """Read and validate the controller settings (None: floating base).

        Args:
            controller: The env's ``ControllerConfig``; None builds no
                controller, and the base slots are pelvis position targets.
            robot_model: The env's robot, checked against the backend.
        """
        self.controller: Optional[LowerBodyController] = None
        # Last [vx, vy, wz] sent to the controller; snapshotted by
        # get_state() under the env.* keys.
        self.prev_cmd = np.zeros((3,), dtype=np.float32)
        self.last_control_info: dict[str, Any] = {}

        # A floating-base env still names a backend for validation; no adapter
        # module is imported without a controller.
        self.enabled = controller is not None
        self.config = controller or ControllerConfig()
        self.backend = resolve_backend_name(self.config.backend)
        self.binding = backend_binding(self.backend)
        if robot_model not in self.binding.robot_models:
            raise ValueError(
                f"controller.backend={self.backend} currently "
                f"requires robot_model={'/'.join(self.binding.robot_models)}"
            )
        # Height/pitch bounds left as None are filled from the backend's
        # declared training range (command_spec) after the controller is
        # built; see resolve_command_bounds. These bounds feed the outer
        # action space, so the VR collector's stick clamp inherits them too.
        self.height_cmd_min = _or_default(self.config.height_cmd_min, 0.4)
        self.height_cmd_max = _or_default(self.config.height_cmd_max, 1.0)
        self.pitch_cmd_min = _or_default(self.config.pitch_cmd_min, -0.2)
        self.pitch_cmd_max = _or_default(self.config.pitch_cmd_max, 0.45)
        # Deferred reset warmup (VR collector only): when enabled via
        # BiGym.set_deferred_reset_warmup(), reset() leaves the settle warmup
        # pending and the caller drains it a few steps per XR frame with
        # BiGym.run_reset_warmup_steps(): same step sequence, no 1 s freeze.
        self.defer_reset_warmup = False
        self.pending_reset_warmup = 0
        self.deferred_warmup_hold_action = None
        self.canonical_state: Optional[dict[str, Any]] = None
        # None = fill from the backend's recommended value once the
        # controller is built. 0 is a known-bad setting: demos start at the
        # post-settle engage moment, so eval must settle too.
        warmup = self.config.reset_warmup_steps if self.enabled else None
        self.reset_warmup_steps = max(int(warmup), 0) if warmup is not None else None

    @property
    def base_action_mode(self) -> str:
        """What the outer base slots carry; pelvis position targets without a controller."""
        return self.config.base_action_mode if self.enabled else "legacy_delta"

    @property
    def pitch_cmd_enabled(self) -> bool:
        """Whether the outer RY slot carries the absolute waist-pitch target (rad)."""
        return self.enabled and self.config.pitch_command

    @property
    def deterministic_reset(self) -> bool:
        """Whether every reset restores the controller state of the first reset.

        The controller keeps internal history across reset() (obs history,
        gait phase, mode), so without it a reused env would start an episode
        up to 1e-1 rad away from a fresh env under the same seed.
        """
        return self.enabled and self.config.deterministic_reset

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def robot_cls(self, robot_cls: type[Robot]) -> type[Robot]:
        """The robot class the task env is built with (the backend's variant)."""
        if not self.enabled:
            return robot_cls
        return self.binding.robot_cls(robot_cls, self.config)

    def build_controller(self, env: BiGymEnv, *, control_dt: float) -> None:
        """Build the controller on the freshly built task env."""
        if not self.enabled:
            return
        self.controller = self.binding.build_controller(
            env, self.config, control_dt=control_dt
        )
        self.binding.configure_model(env)
        disable_skyhook(
            env, kp=self.config.skyhook_kp, damping=self.config.skyhook_damping
        )

    def resolve_command_bounds(self) -> None:
        """Fill unset height/pitch bounds from the backend's command spec.

        The values come from the backend's declared training ranges
        (``LowerBodyController.command_spec``).

        Precedence: explicitly configured values always win — the outer
        action-space bounds define how recorded demo actions denormalize,
        so they must never move under an existing config. Only keys the
        config leaves unset are taken from the spec; the env defaults remain
        the last fallback (controller disabled / field absent from the spec).

        Twist (vx/vy/wz) slots intentionally stay on the symmetric
        cmd_clip/wz_clip scalars: switching them to per-channel spec bounds
        would change action semantics (a substrate bump).
        """
        if self.controller is None:
            return
        spec = self.controller.command_spec
        if spec.has("height"):
            field = spec.field("height")
            if self.config.height_cmd_min is None:
                self.height_cmd_min = float(field.low)
            if self.config.height_cmd_max is None:
                self.height_cmd_max = float(field.high)
        if spec.has("torso_pitch"):
            field = spec.field("torso_pitch")
            if self.config.pitch_cmd_min is None:
                self.pitch_cmd_min = float(field.low)
            if self.config.pitch_cmd_max is None:
                self.pitch_cmd_max = float(field.high)

    def resolve_reset_warmup(self) -> None:
        """Fill a non-configured reset_warmup_steps from the backend.

        Each adapter declares ``recommended_reset_warmup_steps`` (measured
        settle time; re-measure when the checkpoint changes). Explicit config
        wins verbatim. 0 is a valid explicit choice only for callers that
        manage settling themselves.
        """
        if self.controller is None or self.reset_warmup_steps is not None:
            return
        # Not part of LowerBodyController: a controller without it settles 0 steps.
        recommended = getattr(self.controller, "recommended_reset_warmup_steps", 0)
        self.reset_warmup_steps = max(int(recommended), 0)

    def effective_reset_warmup_steps(self) -> Optional[int]:
        """Settle steps run at reset (None without a controller)."""
        if self.controller is None or self.reset_warmup_steps is None:
            return None
        return self.reset_warmup_steps

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, env: BiGymEnv) -> None:
        """Re-pose the robot and reset the controller after the task env's reset."""
        assert self.controller is not None
        # A forward pass between the inner reset and the init-height overlay:
        # it refreshes qacc_warmstart before the overlay forwards again, and
        # the recorded reset trajectories bake that in.
        mujoco.mj_forward(env.model, env.data)
        set_pelvis_height(env, float(self.config.init_pelvis_z))
        self.controller.reset()
        if self.canonical_state is None:
            self.canonical_state = copy.deepcopy(self.controller.get_state())
        elif self.deterministic_reset:
            self.controller.set_state(copy.deepcopy(self.canonical_state))
        self.prev_cmd[:] = 0.0

    def run_reset_warmup(self, env: BiGym, steps: int) -> None:
        """Hold the robot still for ``steps`` control steps so it settles."""
        if self.controller is None or steps <= 0:
            return

        raw_hold_action = env.raw_hold_action()
        for _ in range(int(steps)):
            self.warmup_step(env, raw_hold_action)
        # Warmup steps must never pre-fill the success-hold window (e.g. a
        # target randomized within tolerance of the resting hand).
        env.inner_env.reset_success_hold()

    def warmup_step(self, env: BiGym, raw_hold_action) -> None:
        """One settle step with the hold action (no observation rendering)."""
        self.update_command(env, raw_hold_action)
        expanded_action = self.expand_action(env, raw_hold_action)
        expanded_action = np.clip(
            expanded_action,
            env.inner_env.action_space.low,
            env.inner_env.action_space.high,
        ).astype(raw_hold_action.dtype, copy=False)
        # fast=True skips per-step observation assembly (3 camera renders,
        # ~half the step cost); nothing reads warmup observations and the
        # success-hold counter still updates before the fast return (then
        # reset_success_hold() zeroes it anyway). The explicit mj_forward
        # reproduces the one physical side effect of the render path:
        # update_scene's internal mj_forward overwrites qacc_warmstart, and
        # recorded trajectories bake that in.
        env.inner_env.step(expanded_action, fast=True)
        env.inner_env._on_reset_warmup_step()
        mujoco.mj_forward(env.inner_env.model, env.inner_env.data)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def uses_command_base_actions(self, env: BiGym) -> bool:
        """True when the outer base slots carry controller commands."""
        return (
            self.base_action_mode == "lowerbody_cmd"
            and env.config.control_pelvis
            and bool(env.outer_action_floating_dofs)
        )

    def _extract_pitch_cmd(
        self, base_action: np.ndarray, dof_to_idx: dict
    ) -> float | None:
        """Torso-pitch command from the RY slot (None when disabled)."""
        if not self.pitch_cmd_enabled or PelvisDof.RY not in dof_to_idx:
            return None
        return float(
            np.clip(
                base_action[dof_to_idx[PelvisDof.RY]],
                self.pitch_cmd_min,
                self.pitch_cmd_max,
            )
        )

    def raw_action_to_command(
        self, env: BiGym, action: np.ndarray
    ) -> tuple[np.ndarray, float, float | None] | None:
        """``([vx, vy, wz], height, pitch)`` from a raw outer action, or None."""
        if self.controller is None:
            return None

        floating_base = env.inner_env.robot.floating_base
        if floating_base is None:
            return None

        dt = float(env.config.demo_down_sample_rate) / float(CONTROL_FREQUENCY_MAX)
        if dt <= 0:
            return None

        dof_amount = int(len(env.outer_action_floating_dofs))
        base_action = action[:dof_amount]
        dof_to_idx = {dof: i for i, dof in enumerate(env.outer_action_floating_dofs)}
        if PelvisDof.X not in dof_to_idx or PelvisDof.Y not in dof_to_idx:
            return None

        if self.uses_command_base_actions(env):
            vx_body = float(
                np.clip(
                    base_action[dof_to_idx[PelvisDof.X]],
                    -self.config.cmd_clip,
                    self.config.cmd_clip,
                )
            )
            vy_body = float(
                np.clip(
                    base_action[dof_to_idx[PelvisDof.Y]],
                    -self.config.cmd_clip,
                    self.config.cmd_clip,
                )
            )
            if PelvisDof.RZ in dof_to_idx:
                wz = float(
                    np.clip(
                        base_action[dof_to_idx[PelvisDof.RZ]],
                        -self.config.wz_clip,
                        self.config.wz_clip,
                    )
                )
            else:
                wz = 0.0
            if self.config.use_height_cmd and PelvisDof.Z in dof_to_idx:
                height_cmd = float(
                    np.clip(
                        base_action[dof_to_idx[PelvisDof.Z]],
                        self.height_cmd_min,
                        self.height_cmd_max,
                    )
                )
            else:
                height_cmd = self.config.default_height_cmd
            return (
                np.asarray([vx_body, vy_body, wz], dtype=np.float32),
                float(height_cmd),
                self._extract_pitch_cmd(base_action, dof_to_idx),
            )

        vx_world = float(base_action[dof_to_idx[PelvisDof.X]]) / dt
        vy_world = float(base_action[dof_to_idx[PelvisDof.Y]]) / dt
        if PelvisDof.RZ in dof_to_idx:
            wz = float(base_action[dof_to_idx[PelvisDof.RZ]]) / dt
        else:
            wz = 0.0

        yaw = yaw_from_floating_qpos(np.asarray(floating_base.qpos, dtype=np.float32))
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        vx_body = c * vx_world + s * vy_world
        vy_body = -s * vx_world + c * vy_world

        if self.config.use_height_cmd and PelvisDof.Z in dof_to_idx:
            inner_z_idx = None
            if env.inner_floating_dofs is not None:
                for i, dof in enumerate(env.inner_floating_dofs):
                    if dof == PelvisDof.Z:
                        inner_z_idx = i
                        break
            current_z = (
                float(floating_base.qpos[int(inner_z_idx)])
                if inner_z_idx is not None
                else 0.0
            )
            target_z = current_z + float(base_action[dof_to_idx[PelvisDof.Z]])
            height_cmd = float(
                np.clip(
                    target_z,
                    self.height_cmd_min,
                    self.height_cmd_max,
                )
            )
        else:
            height_cmd = self.config.default_height_cmd

        return (
            np.asarray([vx_body, vy_body, wz], dtype=np.float32),
            float(height_cmd),
            self._extract_pitch_cmd(base_action, dof_to_idx),
        )

    def set_command(
        self,
        desired_cmd: np.ndarray,
        *,
        height_cmd: float,
        pitch_cmd: float | None = None,
    ) -> None:
        """Latch a command on the controller and record it."""
        if self.controller is None:
            return

        cmd = np.asarray(desired_cmd, dtype=np.float32).reshape(3).copy()
        # Keyword names follow the command_spec field names.
        set_command_kwargs = {"height": height_cmd}
        if pitch_cmd is not None:
            # The RY slot only exists when the backend enabled pitch.
            set_command_kwargs["torso_pitch"] = pitch_cmd
        self.controller.set_command(
            float(cmd[0]),
            float(cmd[1]),
            float(cmd[2]),
            **set_command_kwargs,
        )
        self.prev_cmd = cmd.copy()

    def update_command(self, env: BiGym, action: np.ndarray) -> None:
        """Send the command carried by a raw outer action, if any."""
        command = self.raw_action_to_command(env, action)
        if command is None:
            return
        desired_cmd, height_cmd, pitch_cmd = command
        self.set_command(
            desired_cmd,
            height_cmd=height_cmd,
            pitch_cmd=pitch_cmd,
        )

    def expand_action(self, env: BiGym, raw_action_outer: np.ndarray) -> np.ndarray:
        """The inner action: outer slots mapped in, controller targets merged."""
        base = env.inner_env.robot.floating_base
        inner_base_dof_amount = int(base.dof_amount) if base is not None else 0
        outer_base_dof_amount = int(len(env.outer_action_floating_dofs))
        gripper_count = int(len(env.inner_env.robot.grippers))
        leg_joint_names = env.leg_joint_names()

        full = np.zeros(env.inner_env.action_space.shape, dtype=raw_action_outer.dtype)
        base_action_for_env = raw_action_outer[:outer_base_dof_amount]
        if self.uses_command_base_actions(env):
            # The controller consumes the base slots as commands.
            base_action_for_env = np.zeros_like(base_action_for_env)
        if (
            inner_base_dof_amount
            and env.outer_action_base_to_inner_base_idx is not None
            and env.outer_action_base_to_inner_base_idx.size
        ):
            for outer_i, inner_i in enumerate(env.outer_action_base_to_inner_base_idx):
                if int(inner_i) < 0:
                    # Command-only dof (RY): consumed by the lower-body
                    # backend, nothing to forward to the inner env.
                    continue
                full[int(inner_i)] = base_action_for_env[outer_i]
        if gripper_count:
            full[-gripper_count:] = raw_action_outer[-gripper_count:]

        limb_start = outer_base_dof_amount
        limb_end = (
            raw_action_outer.shape[0] - gripper_count
            if gripper_count
            else raw_action_outer.shape[0]
        )
        outer_limb = raw_action_outer[limb_start:limb_end]
        if env.outer_limb_actuator_names and env.limb_name_to_full_index:
            for i, name in enumerate(env.outer_limb_actuator_names):
                full_idx = env.limb_name_to_full_index[name]
                full[inner_base_dof_amount + full_idx] = outer_limb[i]

        if self.controller is not None:
            leg_targets = self.controller.step()
            leg_action = self.controller.get_last_action()
            assert env.limb_name_to_full_index is not None
            for j, joint_name in enumerate(leg_joint_names):
                full_idx = env.limb_name_to_full_index[joint_name]
                action_idx = inner_base_dof_amount + full_idx
                low = float(env.inner_env.action_space.low[action_idx])
                high = float(env.inner_env.action_space.high[action_idx])
                full[action_idx] = float(np.clip(leg_targets[j], low, high))

            command = self.controller.get_command()
            self.last_control_info = {
                "lowerbody_action": np.asarray(leg_action, dtype=np.float32).copy(),
                "leg_joint_targets": np.asarray(leg_targets, dtype=np.float32).copy(),
                "leg_joint_names": np.asarray(leg_joint_names, dtype=object),
                # The demo schema has a torso-target column; the G1 has no
                # separate torso joint, so it is NaN.
                "torso_target": np.asarray([np.nan], dtype=np.float32),
                "lowerbody_command": np.asarray(command, dtype=np.float32).copy(),
                "height_command": np.asarray(
                    [float(self.controller.get_height_command())],
                    dtype=np.float32,
                ),
                "raw_outer_action": np.asarray(
                    raw_action_outer, dtype=np.float32
                ).copy(),
                "expanded_action": np.asarray(full, dtype=np.float32).copy(),
            }

        return full

    # ------------------------------------------------------------------
    # Snapshot / restore (BiGym.get_lowerbody_state / set_lowerbody_state)
    # ------------------------------------------------------------------

    def get_state(self, env: BiGym) -> dict[str, np.ndarray]:
        """Controller fields under ``ctrl.*``, env-side command state under ``env.*``."""
        if self.controller is None:
            return {}
        state = {
            f"ctrl.{key}": np.asarray(value)
            for key, value in self.controller.get_state().items()
        }
        # Three keys holding the same last base command; the names are kept
        # so snapshots stay byte-identical to recorded ones.
        for key in (
            "env.adapter_prev_desired_cmd",
            "env.adapter_prev_corrected_cmd",
            "env.waypoint_prev_cmd",
        ):
            state[key] = self.prev_cmd.astype(np.float32, copy=True)
        if (
            env.upper_delta_accumulator is not None
            and env.upper_delta_accumulator.initialized
        ):
            state["env.upper_delta_target"] = env.upper_delta_accumulator.target
        return state

    def set_state(self, env: BiGym, state: dict[str, np.ndarray]) -> None:
        """Restore a snapshot produced by get_state()."""
        if self.controller is None:
            raise RuntimeError(
                "set_lowerbody_state() requires an enabled lower-body backend."
            )
        ctrl_state = {
            key[len("ctrl.") :]: value
            for key, value in state.items()
            if key.startswith("ctrl.")
        }
        if ctrl_state:
            self.controller.set_state(ctrl_state)
        for key in (
            "env.adapter_prev_desired_cmd",
            "env.adapter_prev_corrected_cmd",
            "env.waypoint_prev_cmd",
        ):
            if key in state:
                self.prev_cmd[:] = np.asarray(state[key], dtype=np.float32).reshape(3)
                break
        if env.upper_delta_accumulator is not None:
            if "env.upper_delta_target" in state:
                env.upper_delta_accumulator.set_state(state["env.upper_delta_target"])
            else:
                env.reset_action_representation_state()


def yaw_from_floating_qpos(qpos: np.ndarray) -> float:
    """Base yaw from floating-base qpos (a wxyz quaternion at [3:7], else the last entry)."""
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.size >= 7:
        w, x, y, z = float(qpos[3]), float(qpos[4]), float(qpos[5]), float(qpos[6])
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))
    return float(qpos[-1]) if qpos.size else 0.0


def disable_skyhook(env: BiGymEnv, *, kp: float, damping: float) -> None:
    """Set the floating-base actuators to a ``kp`` spring and its joints' damping."""
    floating_base = env.robot.floating_base
    if floating_base is None:
        return

    kp = float(kp)
    damping = float(damping)

    model = env.model
    target_names = {dof.value for dof in env.action_mode.floating_dofs}

    for aid in range(int(model.nu)):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        if _mujoco_basename(act_name) not in target_names:
            continue
        model.actuator_gainprm[aid, :] = 0.0
        model.actuator_biasprm[aid, :] = 0.0
        model.actuator_gainprm[aid, 0] = kp
        model.actuator_biasprm[aid, 1] = -kp

    for jid in range(int(model.njnt)):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if _mujoco_basename(joint_name) not in target_names:
            continue
        dofadr = int(model.jnt_dofadr[jid])
        model.dof_damping[dofadr] = damping


def set_pelvis_height(env: BiGymEnv, target_z: float) -> None:
    """Place the pelvis Z joint (and its actuator target) at ``target_z``."""
    floating_base = env.robot.floating_base
    if floating_base is None:
        return

    model = env.model
    data = env.data

    z_qadr = None
    z_dofadr = None
    for jid in range(int(model.njnt)):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if _mujoco_basename(joint_name) != PelvisDof.Z.value:
            continue

        if int(model.jnt_limited[jid]):
            low = float(model.jnt_range[jid, 0])
            high = float(model.jnt_range[jid, 1])
            target_z = float(np.clip(target_z, min(low, high), max(low, high)))

        z_qadr = int(model.jnt_qposadr[jid])
        z_dofadr = int(model.jnt_dofadr[jid])
        break

    if z_qadr is None:
        return

    data.qpos[z_qadr] = target_z
    if z_dofadr is not None and 0 <= z_dofadr < int(model.nv):
        data.qvel[z_dofadr] = 0.0
        data.qacc[z_dofadr] = 0.0

    for aid in range(int(model.nu)):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        if _mujoco_basename(act_name) != PelvisDof.Z.value:
            continue

        ctrl = target_z
        if int(model.actuator_ctrllimited[aid]):
            low = float(model.actuator_ctrlrange[aid, 0])
            high = float(model.actuator_ctrlrange[aid, 1])
            ctrl = float(np.clip(ctrl, min(low, high), max(low, high)))
        data.ctrl[aid] = ctrl
        break

    mujoco.mj_forward(model, data)
