"""Action-representation adapters for controller-in-the-loop environments.

The real-leg outer action is heterogeneous::

    [base command, position-controlled outer limbs, grippers]

``upper_delta`` changes only the position-controlled limb slice. Base slots
retain their command semantics (velocity/height/yaw or legacy pelvis delta),
and grippers remain absolute. The inner BiGym environment continues to use
absolute joint targets, so frozen lower-body targets are never accumulated.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

ABSOLUTE = "absolute"
UPPER_DELTA = "upper_delta"
ACTION_REPRESENTATIONS = (ABSOLUTE, UPPER_DELTA)
PRESERVED_BASE_MODES = (
    "legacy_delta",
    "lowerbody_cmd",
)


def validate_action_representation(value: str) -> str:
    """Lower-case and validate an action-representation name."""
    representation = str(value).lower()
    if representation not in ACTION_REPRESENTATIONS:
        raise ValueError(
            f"action_representation must be one of {ACTION_REPRESENTATIONS}, "
            f"got {value!r}"
        )
    return representation


def validate_upper_delta_scale_rad(value: float | None) -> float:
    """Validate the upper-delta scale and return it as a positive float."""
    if value is None:
        raise ValueError(
            "upper_delta_scale_rad is required when action_representation='upper_delta'"
        )
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"upper_delta_scale_rad must be finite and > 0, got {value!r}")
    return scale


def upper_delta_scale_from_metadata(
    metadata: Mapping[str, Any],
    explicit_scale_rad: float | None = None,
) -> float:
    """Resolve the per-step upper delta envelope in radians."""
    if explicit_scale_rad is not None:
        return validate_upper_delta_scale_rad(explicit_scale_rad)
    conditioning = (metadata.get("action_semantics") or {}).get(
        "upperbody_ik_conditioning"
    ) or {}
    max_speed = conditioning.get("max_joint_speed_rad_s")
    step_seconds = metadata.get("control_step_seconds")
    if max_speed is None or step_seconds is None or not np.isscalar(max_speed):
        raise ValueError(
            "cannot derive upper_delta scale: metadata needs scalar "
            "action_semantics.upperbody_ik_conditioning."
            "max_joint_speed_rad_s and control_step_seconds; pass "
            "upper_delta_scale_rad explicitly"
        )
    return validate_upper_delta_scale_rad(
        float(max_speed) * float(step_seconds)  # ty: ignore[invalid-argument-type]
    )


def mixed_action_slices(
    metadata: Mapping[str, Any], action_dim: int
) -> tuple[slice, slice, slice]:
    """Return base, upper-limb and gripper slices from native metadata."""
    base_mode = str(
        (metadata.get("action_semantics") or {}).get("base_action_mode", "")
    )
    if base_mode not in PRESERVED_BASE_MODES:
        raise ValueError(
            "upper_delta requires a known preserved base representation; "
            f"got action_semantics.base_action_mode={base_mode!r}"
        )
    base_dim = len(metadata.get("outer_action_floating_dofs") or ())
    upper_dim = len(metadata.get("outer_limb_actuator_names") or ())
    gripper_dim = int(action_dim) - base_dim - upper_dim
    if base_dim < 0 or upper_dim <= 0 or gripper_dim < 0:
        raise ValueError(
            "metadata does not describe a valid mixed outer action: "
            f"action_dim={action_dim}, base={base_dim}, upper={upper_dim}, "
            f"remaining gripper={gripper_dim}"
        )
    return (
        slice(0, base_dim),
        slice(base_dim, base_dim + upper_dim),
        slice(base_dim + upper_dim, int(action_dim)),
    )


def convert_episode_action_representation(
    episode: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    action_representation: str = ABSOLUTE,
    upper_delta_scale_rad: float | None = None,
) -> dict[str, np.ndarray]:
    """Return an in-memory episode action view; never mutate source data.

    Replay row 0 is the reset/dummy row. Its absolute upper target initializes
    the sequence, so its derived delta is zero and row 1 is measured from it.
    """
    representation = validate_action_representation(action_representation)
    if representation == ABSOLUTE:
        return episode if isinstance(episode, dict) else dict(episode)

    action = np.asarray(episode.get("action"), dtype=np.float32)
    if action.ndim != 2 or action.shape[0] == 0:
        raise ValueError(
            f"upper_delta expects a non-empty action array [T,D], got {action.shape}"
        )
    base, upper, gripper = mixed_action_slices(metadata, action.shape[1])
    stats = metadata.get("action_stats") or {}
    action_min = np.asarray(stats.get("min"), dtype=np.float32)
    action_max = np.asarray(stats.get("max"), dtype=np.float32)
    if action_min.shape != (action.shape[1],) or action_max.shape != (action.shape[1],):
        raise ValueError(
            f"metadata action_stats min/max must both be [{action.shape[1]}], "
            f"got {action_min.shape} and {action_max.shape}"
        )
    scale = upper_delta_scale_from_metadata(metadata, upper_delta_scale_rad)

    raw = ((action + 1.0) * 0.5 * (action_max - action_min + 1e-8) + action_min).astype(
        np.float32
    )
    raw_delta = np.zeros_like(raw[:, upper])
    raw_delta[1:] = raw[1:, upper] - raw[:-1, upper]
    normalized_delta = raw_delta / scale
    max_abs = float(np.max(np.abs(normalized_delta), initial=0.0))
    if max_abs > 1.0 + 1e-5:
        raise ValueError(
            "upper_delta exceeds its normalized [-1, 1] envelope "
            f"(max_abs={max_abs:.6g}, scale={scale:.6g} rad); pass a larger "
            "upper_delta_scale_rad explicitly"
        )

    converted = action.copy()
    converted[:, upper] = np.clip(normalized_delta, -1.0, 1.0)
    converted[:, base] = action[:, base]
    converted[:, gripper] = action[:, gripper]
    result = dict(episode)
    result["action"] = converted
    return result


class UpperDeltaAccumulator:
    """Stateful normalized upper-delta to absolute-target adapter."""

    def __init__(self, scale_rad: float | None):
        """Store the delta scale in radians; the target starts unset."""
        self.scale_rad = validate_upper_delta_scale_rad(scale_rad)
        self._target: np.ndarray | None = None

    @property
    def initialized(self) -> bool:
        """Whether ``reset`` has supplied an absolute target."""
        return self._target is not None

    @property
    def target(self) -> np.ndarray:
        """A copy of the current absolute joint target; raises before reset."""
        if self._target is None:
            raise RuntimeError("upper_delta accumulator has not been reset")
        return self._target.copy()

    def reset(self, absolute_target: np.ndarray) -> None:
        """Seed the accumulator with a finite 1D absolute target."""
        target = np.asarray(absolute_target, dtype=np.float32)
        if target.ndim != 1 or not np.all(np.isfinite(target)):
            raise ValueError(
                "upper_delta reset target must be a finite 1D array, "
                f"got shape={target.shape}"
            )
        self._target = target.copy()

    def set_state(self, absolute_target: np.ndarray) -> None:
        """Alias of :meth:`reset`, for controller-style state restore."""
        self.reset(absolute_target)

    def apply(
        self,
        normalized_delta: np.ndarray,
        absolute_low: np.ndarray,
        absolute_high: np.ndarray,
    ) -> np.ndarray:
        """Accumulate one normalized delta and return the clipped target."""
        if self._target is None:
            raise RuntimeError("upper_delta accumulator has not been reset")
        delta = np.asarray(normalized_delta, dtype=np.float32)
        low = np.asarray(absolute_low, dtype=np.float32)
        high = np.asarray(absolute_high, dtype=np.float32)
        if (
            delta.shape != self._target.shape
            or low.shape != delta.shape
            or high.shape != delta.shape
        ):
            raise ValueError(
                "upper_delta accumulator shape mismatch: "
                f"target={self._target.shape}, delta={delta.shape}, "
                f"low={low.shape}, high={high.shape}"
            )
        if not np.all(np.isfinite(delta)) or np.any(np.abs(delta) > 1.0 + 1e-6):
            raise ValueError("normalized upper delta must be finite and in [-1, 1]")
        self._target = np.clip(
            self._target + np.clip(delta, -1.0, 1.0) * self.scale_rad,
            low,
            high,
        ).astype(np.float32)
        return self._target.copy()


__all__ = [
    "ABSOLUTE",
    "UPPER_DELTA",
    "ACTION_REPRESENTATIONS",
    "UpperDeltaAccumulator",
    "convert_episode_action_representation",
    "mixed_action_slices",
    "upper_delta_scale_from_metadata",
    "validate_action_representation",
    "validate_upper_delta_scale_rad",
]
