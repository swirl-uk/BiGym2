"""dm_env-style transition records and the wrapper ``make`` returns.

:class:`TimeStep` is what :meth:`bigym.loco.env.BiGym.reset` and ``step``
return; :class:`ExtendedTimeStep` adds the action that produced it (the
demo and replay format), and :class:`ExtendedTimeStepWrapper` turns the
former into the latter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from dm_env import StepType

if TYPE_CHECKING:
    from bigym.loco.env import BiGym


class TimeStep(NamedTuple):
    """Transition record returned by BiGym.step()/reset() (no action)."""

    step_type: Any
    reward: Any
    discount: Any
    rgb_obs: Any
    low_dim_obs: Any
    demo: Any
    is_expert: Any
    event_progress: Any

    def first(self):
        """Return True on the first TimeStep of an episode."""
        return self.step_type == StepType.FIRST

    def mid(self):
        """Return True on a mid-episode TimeStep."""
        return self.step_type == StepType.MID

    def last(self):
        """Return True on the final TimeStep of an episode."""
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        """Return a field by name, or by tuple index for non-string keys."""
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ExtendedTimeStep(NamedTuple):
    """TimeStep plus the action that produced it (demo/replay format)."""

    step_type: Any
    reward: Any
    discount: Any
    rgb_obs: Any
    low_dim_obs: Any
    action: Any
    demo: Any
    is_expert: Any
    event_progress: Any

    def first(self):
        """Return True on the first TimeStep of an episode."""
        return self.step_type == StepType.FIRST

    def mid(self):
        """Return True on a mid-episode TimeStep."""
        return self.step_type == StepType.MID

    def last(self):
        """Return True on the final TimeStep of an episode."""
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        """Return a field by name, or by tuple index for non-string keys."""
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ExtendedTimeStepWrapper:
    """Wrap a BiGym env so reset()/step() return an ExtendedTimeStep.

    The action that produced the step is carried alongside the
    observation; reset() reports a zero action instead. ``bigym`` is the
    wrapped :class:`~bigym.loco.env.BiGym`; any other attribute is looked
    up on it.
    """

    def __init__(self, env: BiGym):
        """Store the environment to wrap."""
        self.bigym = env

    def reset(self, **kwargs):
        """Reset the wrapped env and return the ExtendedTimeStep (zero action)."""
        time_step = self.bigym.reset(**kwargs)
        return self._augment_time_step(time_step)

    def step(self, action):
        """Step the wrapped env and return the ExtendedTimeStep for the action."""
        time_step = self.bigym.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            rgb_obs=time_step.rgb_obs,
            low_dim_obs=time_step.low_dim_obs,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward,
            discount=time_step.discount,
            demo=time_step.demo,
            is_expert=time_step.is_expert,
            event_progress=time_step.event_progress,
        )

    def low_dim_observation_spec(self):
        """Return the wrapped env's frame-stacked low-dim observation spec."""
        return self.bigym.low_dim_observation_spec()

    def rgb_observation_spec(self):
        """Return the wrapped env's frame-stacked RGB observation spec."""
        return self.bigym.rgb_observation_spec()

    def low_dim_raw_observation_spec(self):
        """Return the wrapped env's unstacked low-dim observation spec."""
        return self.bigym.low_dim_raw_observation_spec()

    def low_dim_component_slices(self):
        """Return the wrapped env's per-state-key low-dim slices."""
        return self.bigym.low_dim_component_slices()

    def rgb_raw_observation_spec(self):
        """Return the wrapped env's unstacked RGB observation spec."""
        return self.bigym.rgb_raw_observation_spec()

    def action_spec(self):
        """Return the wrapped env's action spec."""
        return self.bigym.action_spec()

    def run_reset_warmup_steps(self, max_steps: int):
        """Forward the deferred-warmup drain to the wrapped env.

        The completion TimeStep is augmented exactly like reset() does
        (zero action); None is passed through while the drain is
        incomplete.
        """
        time_step = self.bigym.run_reset_warmup_steps(max_steps)
        if time_step is None:
            return None
        return self._augment_time_step(time_step)

    def __getattr__(self, name):
        """Delegate unknown attribute lookups to the wrapped env."""
        return getattr(self.bigym, name)
