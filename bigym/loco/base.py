"""LowerBodyBase — the fat base class implementing the shared pipeline.

Every concrete backend deals with the same plumbing: resolving joint
addresses by name, reading the floating base and IMU-style quantities,
applying an anchor pose at reset, clipping commands to training ranges, and
snapshotting mutable state for bit-exact demo replay. This class implements
all of it so a new backend only provides:

- a policy loader (``__init__``),
- ``step()`` (build obs -> run policy -> joint position targets),
- declarations: ``controlled_joints``, ``command_spec``, and either the
  declarative ``STATEFUL`` mapping or custom ``get_state``/``set_state``.

Where an adapter's behavior deviates from the base defaults (e.g. a
controller that stores raw commands and clips on read), the adapter
overrides the base method rather than the base absorbing the quirk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import mujoco
import numpy as np

from bigym.loco.command import CommandSpec
from bigym.loco.controller import OutputSpec


def mujoco_basename(full_identifier: str | None) -> str:
    """Strip the attachment namespace from a scene element name."""
    if not full_identifier:
        return ""
    return full_identifier.split("/")[-1]


def quat_rotate_inverse_wxyz(q_wxyz: np.ndarray, v_xyz: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by the inverse of quaternion ``q`` (wxyz order)."""
    w, x, y, z = (
        float(q_wxyz[0]),
        float(q_wxyz[1]),
        float(q_wxyz[2]),
        float(q_wxyz[3]),
    )
    rx, ry, rz = -x, -y, -z

    tx = 2.0 * (ry * v_xyz[2] - rz * v_xyz[1])
    ty = 2.0 * (rz * v_xyz[0] - rx * v_xyz[2])
    tz = 2.0 * (rx * v_xyz[1] - ry * v_xyz[0])

    out_x = v_xyz[0] + w * tx + (ry * tz - rz * ty)
    out_y = v_xyz[1] + w * ty + (rz * tx - rx * tz)
    out_z = v_xyz[2] + w * tz + (rx * ty - ry * tx)
    return np.array([out_x, out_y, out_z], dtype=np.float32)


def yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    """Yaw angle in radians from a wxyz quaternion."""
    w, x, y, z = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def yaw_from_qpos(qpos: np.ndarray) -> float:
    """Yaw from a floating-base qpos: free joint (>=7) or slide/hinge stack."""
    if qpos.size >= 7:
        return yaw_from_quat_wxyz(qpos[3:7])
    if qpos.size:
        return float(qpos[-1])
    return 0.0


class LowerBodyBase:
    """Shared pipeline for lower-body backends.

    Subclasses set in ``__init__``: ``controlled_joints``, ``command_spec``,
    ``control_dt``, ``controlled_range_low``/``controlled_range_high`` (from
    :meth:`build_joint_ranges`), ``velocity_clip``/``yaw_rate_clip``, and the
    latched ``command``, ``height_command`` and ``last_action``.
    """

    # Declarative replay state: snapshot key -> attribute name. The base
    # get_state()/set_state() stack and restore them, so a forgotten field is
    # a loud KeyError instead of a silent divergence.
    STATEFUL: dict[str, str] = {}

    # Post-reset settle steps the env runs before handing control to the
    # agent when the config does not set reset_warmup_steps. Measured per
    # checkpoint; adapters override. Demos are recorded from the post-settle
    # engage moment, so eval must settle too — 0 is only correct for callers
    # that manage settling themselves.
    recommended_reset_warmup_steps: int = 0

    def weight_files(self) -> tuple[str, ...]:
        """Absolute paths of the policy weight files this controller loaded.

        Used by ``substrate_fingerprint()`` to record a SHA-256 per weight
        file, so a leaderboard record pins the exact lower-body policy.
        Adapters override; the default (no files) is only right for
        analytic controllers.
        """
        return ()

    _env: Any
    _pelvis_body_id: Optional[int] = None
    #: Joints whose position targets this backend owns.
    controlled_joints: tuple[str, ...]
    #: The backend's typed command declaration and its bounds.
    command_spec: CommandSpec
    #: Seconds between ``step()`` calls.
    control_dt: float
    #: Symmetric clip on ``vx``/``vy``; 0 disables it.
    velocity_clip: float
    #: Symmetric clip on ``wz``; 0 disables it.
    yaw_rate_clip: float
    #: Lower joint limits of ``controlled_joints``.
    controlled_range_low: np.ndarray
    #: Upper joint limits of ``controlled_joints``.
    controlled_range_high: np.ndarray
    #: The latched ``[vx, vy, wz]`` command.
    command: np.ndarray
    #: The latched height command in meters.
    height_command: float
    #: The policy's most recent raw action.
    last_action: np.ndarray

    if TYPE_CHECKING:

        def step(self) -> np.ndarray:
            """Advance the policy one control step; backends implement it."""
            ...

    # ------------------------------------------------------------------
    # Contract properties
    # ------------------------------------------------------------------

    @property
    def output_spec(self) -> OutputSpec:
        """Names and bounds of the joint targets produced by ``step()``."""
        return OutputSpec(
            joint_names=self.controlled_joints,
            low=np.asarray(self.controlled_range_low, dtype=np.float32).copy(),
            high=np.asarray(self.controlled_range_high, dtype=np.float32).copy(),
        )

    def get_command(self) -> np.ndarray:
        """A copy of the latched twist command ``[vx, vy, wz]``."""
        return self.command.astype(np.float32, copy=True)

    def get_height_command(self) -> float:
        """The latched height command in meters."""
        return float(self.height_command)

    def get_last_action(self) -> np.ndarray:
        """A copy of the lower-body policy's most recent raw action."""
        return self.last_action.astype(np.float32, copy=True)

    # ------------------------------------------------------------------
    # Failure detection
    # ------------------------------------------------------------------

    # Upright projected gravity is [0, 0, -1]; proj_g_z above this means the
    # base tilted past ~53 degrees — backend-independent (fallen is fallen).
    _FAILED_MAX_PROJ_G_Z = -0.6
    # Height backstop: a commanded squat at the bottom of the backend's
    # training height range is NOT a failure, so the threshold derives from
    # command_spec (height.low - margin) rather than one fixed constant.
    # The margin absorbs pelvis-vs-commanded-height tracking slack; an
    # actually fallen robot is far below it and also trips the tilt check.
    _FAILED_HEIGHT_MARGIN = 0.10
    # Fallback for specs without a height field (twist-only backends).
    _FAILED_MIN_BASE_HEIGHT = 0.35

    def _failed_min_base_height(self) -> float:
        spec = self.command_spec
        if spec.has("height"):
            return float(spec.field("height").low) - self._FAILED_HEIGHT_MARGIN
        return self._FAILED_MIN_BASE_HEIGHT

    def is_failed(self) -> bool:
        """Fallen / tipped-over detection from base height and tilt."""
        _, _, proj_g = self.get_base_obs()
        if float(proj_g[2]) > self._FAILED_MAX_PROJ_G_Z:
            return True
        pelvis_z = self._pelvis_height()
        if pelvis_z is not None and pelvis_z < self._failed_min_base_height():
            return True
        return False

    def _pelvis_height(self) -> Optional[float]:
        pelvis_body_id = self._pelvis_body_id
        if pelvis_body_id is None:
            model = self._env.model
            pelvis_body_id = -1
            for candidate in range(int(model.nbody)):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, candidate)
                if mujoco_basename(name) == "pelvis":
                    pelvis_body_id = int(candidate)
                    break
            self._pelvis_body_id = pelvis_body_id
        if pelvis_body_id < 0:
            return None
        return float(self._env.data.xpos[pelvis_body_id][2])

    # ------------------------------------------------------------------
    # Command handling
    # ------------------------------------------------------------------

    def _clip_twist(
        self, cmd_vx: float, cmd_vy: float, cmd_wz: float
    ) -> tuple[float, float, float]:
        cmd_clip = float(self.velocity_clip)
        wz_clip = float(self.yaw_rate_clip)
        vx = (
            float(np.clip(cmd_vx, -cmd_clip, cmd_clip))
            if cmd_clip > 0
            else float(cmd_vx)
        )
        vy = (
            float(np.clip(cmd_vy, -cmd_clip, cmd_clip))
            if cmd_clip > 0
            else float(cmd_vy)
        )
        wz = float(np.clip(cmd_wz, -wz_clip, wz_clip)) if wz_clip > 0 else float(cmd_wz)
        return vx, vy, wz

    def set_command(
        self,
        cmd_vx: float,
        cmd_vy: float,
        cmd_wz: float,
        *,
        height: Optional[float] = None,
        torso_pitch: Optional[float] = None,
    ) -> None:
        """Default clip-on-set semantics (groot_wbc family).

        Keyword names match the ``command_spec`` field names.
        """
        vx, vy, wz = self._clip_twist(cmd_vx, cmd_vy, cmd_wz)
        self.command = np.asarray([vx, vy, wz], dtype=np.float32)
        if height is not None:
            self.height_command = float(height)
        if torso_pitch is not None:
            self._set_pitch_command(float(torso_pitch))

    def _set_pitch_command(self, pitch_cmd: float) -> None:
        """Hook for backends with a torso-pitch channel; default: ignore."""

    # ------------------------------------------------------------------
    # Joint / sensor resolution
    # ------------------------------------------------------------------

    def _joint_name_to_id(self) -> dict[str, int]:
        model = self._env.model
        out: dict[str, int] = {}
        for jid in range(int(model.njnt)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            out[mujoco_basename(name)] = jid
        return out

    def build_joint_addresses(
        self, joint_names: tuple[str, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        """The qpos and dof addresses of ``joint_names``, in order.

        Raises:
            ValueError: A joint is missing from the model.
        """
        model = self._env.model
        joint_name_to_id = self._joint_name_to_id()
        qposadr = []
        dofadr = []
        for joint_name in joint_names:
            if joint_name not in joint_name_to_id:
                raise ValueError(
                    f"Missing joint '{joint_name}' required by {type(self).__name__}."
                )
            jid = int(joint_name_to_id[joint_name])
            qposadr.append(int(model.jnt_qposadr[jid]))
            dofadr.append(int(model.jnt_dofadr[jid]))
        return np.asarray(qposadr, dtype=np.int32), np.asarray(dofadr, dtype=np.int32)

    def build_joint_ranges(
        self, joint_names: tuple[str, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        """The lower and upper limits of ``joint_names``; unlimited joints get +-inf."""
        model = self._env.model
        joint_name_to_id = self._joint_name_to_id()
        lows = []
        highs = []
        for joint_name in joint_names:
            jid = int(joint_name_to_id[joint_name])
            if int(model.jnt_limited[jid]):
                lows.append(float(model.jnt_range[jid, 0]))
                highs.append(float(model.jnt_range[jid, 1]))
            else:
                lows.append(-np.inf)
                highs.append(np.inf)
        return np.asarray(lows, dtype=np.float32), np.asarray(highs, dtype=np.float32)

    def find_sensor(self, sensor_name: str, *, dim: int) -> Optional[int]:
        """The ``sensordata`` address of the ``dim``-wide sensor ``sensor_name``, or None."""
        model = self._env.model
        for sid in range(int(model.nsensor)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sid)
            if mujoco_basename(name) != sensor_name:
                continue
            if int(model.sensor_dim[sid]) != int(dim):
                continue
            return int(model.sensor_adr[sid])
        return None

    # ------------------------------------------------------------------
    # Reset pose application
    # ------------------------------------------------------------------

    def apply_pose(
        self,
        qpos_addresses: np.ndarray,
        dof_addresses: np.ndarray,
        targets: np.ndarray,
        joint_names: tuple[str, ...],
    ) -> None:
        """Write a joint pose + matching actuator targets and re-forward."""
        model = self._env.model
        data = self._env.data

        for i, qadr in enumerate(qpos_addresses):
            data.qpos[int(qadr)] = float(targets[i])
        for adr in dof_addresses:
            adr = int(adr)
            if 0 <= adr < int(model.nv):
                data.qvel[adr] = 0.0
                data.qacc[adr] = 0.0

        target_by_joint = {
            name: float(targets[i]) for i, name in enumerate(joint_names)
        }
        for aid in range(int(model.nu)):
            act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
            joint_name = mujoco_basename(act_name)
            if joint_name not in target_by_joint:
                continue
            target = target_by_joint[joint_name]
            if int(model.actuator_ctrllimited[aid]):
                low = float(model.actuator_ctrlrange[aid, 0])
                high = float(model.actuator_ctrlrange[aid, 1])
                target = float(np.clip(target, min(low, high), max(low, high)))
            data.ctrl[aid] = target

        mujoco.mj_forward(model, data)

    # ------------------------------------------------------------------
    # Declarative replay state
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, np.ndarray]:
        """Snapshot every attribute named in ``STATEFUL``.

        Scalars become 0-d float32 arrays, arrays are float32 copies —
        matching the recorded snapshot format so demo npz files stay
        interchangeable.
        """
        state: dict[str, np.ndarray] = {}
        for key, attribute in self.STATEFUL.items():
            value = getattr(self, attribute)
            if isinstance(value, np.ndarray):
                state[key] = value.astype(np.float32, copy=True)
            else:
                state[key] = np.asarray(value, dtype=np.float32)
        return state

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        """Restore a snapshot; missing keys fall back to _state_default().

        Restoration follows ``STATEFUL`` order, an ordering contract adapters
        may rely on: a ``_state_default`` for a derived field (e.g. a slewed
        height that equals the raw command) must come after the field it
        derives from.
        """
        for key, attribute in self.STATEFUL.items():
            if key in state:
                raw = state[key]
            else:
                raw = self._state_default(key)
            current = getattr(self, attribute)
            if isinstance(current, np.ndarray):
                setattr(
                    self,
                    attribute,
                    np.asarray(raw, dtype=np.float32)
                    .reshape(np.asarray(current).shape)
                    .copy(),
                )
            elif isinstance(current, bool):
                setattr(self, attribute, bool(np.asarray(raw).reshape(-1)[0]))
            else:
                setattr(
                    self,
                    attribute,
                    float(np.asarray(raw, dtype=np.float64).reshape(-1)[0]),
                )

    def _state_default(self, key: str):
        """Value for a snapshot key the snapshot lacks.

        Default: a KeyError, since a silently defaulted stateful field makes
        replay diverge. Adapters override it for keys whose absence has one
        meaning (e.g. a snapshot without a pitch channel restores pitch 0).
        """
        raise KeyError(
            f"Snapshot is missing stateful key {key!r} required by "
            f"{type(self).__name__}."
        )

    # ------------------------------------------------------------------
    # Base observation (adapter-specific frames; must be provided)
    # ------------------------------------------------------------------

    def get_base_obs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (base_lin_vel, base_ang_vel, projected_gravity), base frame."""
        raise NotImplementedError
