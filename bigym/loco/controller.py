"""The lower-body controller contract (thin required core).

``LowerBodyController`` is the *entire* surface the BiGym env integration
relies on. Anything else a concrete adapter exposes is an implementation
detail. New backends normally subclass :class:`bigym.loco.base.LowerBodyBase`
(the fat base class that absorbs joint addressing, reset pose application,
failure detection and declarative replay state) and provide a policy loader,
an obs builder and a policy step — a few hundred lines for a real backend —
but any object satisfying this protocol plugs in.

Contract summary:

- ``controlled_joints``   joints whose position targets this backend owns.
- ``command_spec``        typed command declaration (see loco.command).
- ``control_dt``          seconds between ``step()`` calls (1/control_hz).
- ``output_spec``         names + bounds of the produced targets.
- ``reset()``             re-anchor to the backend's init pose, clear state.
- ``set_command(...)``    latch the current command (VELOCITY kind channels).
- ``step()``              run the policy once -> joint position targets, in
                          ``controlled_joints`` order.
- ``is_failed()``         has the robot fallen / left the recoverable region.
- ``get_state()/set_state()``  every mutable field that shapes future
                          targets, for bit-exact mid-episode save/restore
                          (demo replay). MuJoCo qpos/qvel/ctrl/qacc_warmstart
                          are snapshotted by the caller; this covers the
                          controller-side remainder (obs histories, last
                          actions, rate-limiter anchors, gait clocks...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from bigym.loco.command import CommandSpec


@dataclass(frozen=True)
class OutputSpec:
    """Names and bounds of the joint position targets a backend produces."""

    joint_names: tuple[str, ...]
    low: np.ndarray
    high: np.ndarray

    @property
    def dim(self) -> int:
        """Number of joints in this output spec."""
        return len(self.joint_names)


@runtime_checkable
class LowerBodyController(Protocol):
    """Thin required core every lower-body backend implements."""

    @property
    def controlled_joints(self) -> tuple[str, ...]:
        """Joints whose position targets this backend owns (incl. waist)."""
        ...

    @property
    def command_spec(self) -> CommandSpec:
        """Typed command declaration and its bounds."""
        ...

    @property
    def control_dt(self) -> float:
        """Seconds between step() calls."""
        ...

    @property
    def output_spec(self) -> OutputSpec:
        """Names/bounds of the produced targets."""
        ...

    def reset(self) -> None:
        """Re-apply the anchor pose and clear mutable state."""
        ...

    def set_command(
        self,
        cmd_vx: float,
        cmd_vy: float,
        cmd_wz: float,
        *,
        height: Optional[float] = None,
        torso_pitch: Optional[float] = None,
    ) -> None:
        """Latch the VELOCITY-kind command channels.

        Keyword names match the ``command_spec`` field names (``height`` /
        ``torso_pitch``). Channels absent from ``command_spec`` are ignored.
        Non-velocity command kinds (EE_POSE / MOTION_REF) will extend this
        surface as a pure addition; velocity backends stay untouched.
        """
        ...

    def step(self) -> np.ndarray:
        """Run the policy once; return targets in controlled_joints order."""
        ...

    def is_failed(self) -> bool:
        """True when the base has fallen / tipped beyond recovery."""
        ...

    def get_state(self) -> dict[str, np.ndarray]:
        """Snapshot every mutable field that shapes future targets."""
        ...

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        """Restore a snapshot produced by get_state()."""
        ...

    # ------------------------------------------------------------------
    # Introspection the env integration also relies on. LowerBodyBase
    # implements all of these (get_base_obs excepted — frame conventions
    # are backend-specific), so subclassing the base satisfies the whole
    # protocol; a from-scratch implementation must provide them too.
    # ------------------------------------------------------------------

    def get_base_obs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(base_lin_vel, base_ang_vel, projected_gravity) in the base frame."""
        ...

    def get_command(self) -> np.ndarray:
        """Current [vx, vy, wz] command (clipped view)."""
        ...

    def get_height_command(self) -> float:
        """Current height command in meters."""
        ...

    def get_last_action(self) -> np.ndarray:
        """Raw policy action from the most recent step()."""
        ...
