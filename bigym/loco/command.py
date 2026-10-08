"""Typed lower-body command declarations.

A lower-body backend consumes a *command* every control step. Every shipped
backend is velocity-conditioned (``VELOCITY``): the command is a flat vector
of scalar channels such as ``vx / vy / wz / height / torso_pitch``. The
``kind`` tag leaves room for other kinds of controller (SE3 end-effector
targets, reference-motion trackers) without changing the velocity backends.

- ``CommandSpec`` is data: adapters declare their spec. Its bounds are the
  commands the backend accepts; GR00T-WBC publishes no training ranges, so
  its adapter declares benchmark clips.
- ``height`` / ``torso_pitch``: the outer action-space bounds derive from the
  spec unless the config sets them (``LowerBody.resolve_command_bounds``;
  the config wins, so recorded demo action spaces do not move).
- ``vx`` / ``vy`` / ``wz``: the env clips them with the symmetric
  ``cmd_clip`` / ``wz_clip``; the spec bounds are not read. Per-channel spec
  bounds would change the action semantics (a substrate bump).
- ``rate``: an optional slew limit in units/s, ``None`` for a policy trained
  on step commands. It is declarative: adapters own their slew logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CommandKind(Enum):
    """What a lower-body command vector means."""

    VELOCITY = "velocity"
    """Flat scalar channels, e.g. [vx, vy, wz, height, torso_pitch...].

    ``groot_wbc_g1`` is this kind.
    """

    EE_POSE = "ee_pose"
    """Reserved for SE3 hand/foot targets of whole-body controllers."""

    MOTION_REF = "motion_ref"
    """Reserved for reference-motion trajectories (SONIC-style trackers)."""


@dataclass(frozen=True)
class CommandField:
    """One scalar channel of a VELOCITY command."""

    name: str
    unit: str
    low: float
    high: float
    rate: float | None = None
    """Slew limit in unit/s; None = step commands (no slew) by training."""

    def clip(self, value: float) -> float:
        """Clip a value into this field's range (endpoint order-agnostic)."""
        lo, hi = (
            (self.low, self.high) if self.low <= self.high else (self.high, self.low)
        )
        return min(max(float(value), lo), hi)


@dataclass(frozen=True)
class CommandSpec:
    """A backend's full command declaration."""

    kind: CommandKind
    fields: tuple[CommandField, ...]

    def __post_init__(self) -> None:
        """Reject a spec with duplicate field names."""
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate command field names: {names}")

    @property
    def dim(self) -> int:
        """Number of scalar channels in the command vector."""
        return len(self.fields)

    @property
    def names(self) -> tuple[str, ...]:
        """Field names in command-vector order."""
        return tuple(f.name for f in self.fields)

    def field(self, name: str) -> CommandField:
        """The field with this name; raises KeyError if it is not declared."""
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"No command field named {name!r}; spec has {self.names}")

    def has(self, name: str) -> bool:
        """Whether the spec declares a field with this name."""
        return any(f.name == name for f in self.fields)


def velocity_spec(
    *,
    vx: tuple[float, float],
    vy: tuple[float, float],
    wz: tuple[float, float],
    height: tuple[float, float] | None = None,
    height_rate: float | None = None,
    torso_pitch: tuple[float, float] | None = None,
    torso_pitch_rate: float | None = None,
) -> CommandSpec:
    """Build the common twist(+height)(+torso_pitch) VELOCITY spec."""
    fields = [
        CommandField("vx", "m/s", vx[0], vx[1]),
        CommandField("vy", "m/s", vy[0], vy[1]),
        CommandField("wz", "rad/s", wz[0], wz[1]),
    ]
    if height is not None:
        fields.append(CommandField("height", "m", height[0], height[1], height_rate))
    if torso_pitch is not None:
        fields.append(
            CommandField(
                "torso_pitch", "rad", torso_pitch[0], torso_pitch[1], torso_pitch_rate
            )
        )
    return CommandSpec(kind=CommandKind.VELOCITY, fields=tuple(fields))
