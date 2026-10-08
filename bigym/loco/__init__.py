"""bigym.loco — locomotion-in-the-loop public API (BiGym 2.0).

Layered as:

- ``bigym.loco.command`` / ``controller`` / ``base``: the backend contract
  (typed commands, the thin controller protocol, the fat shared base class).
- ``bigym.loco.adapters``: the backend registry and the v1 backend —
  ``groot_wbc_g1`` (NVIDIA GR00T-WBC).
  The GR00T-WBC runtime is vendored and runs on onnxruntime; the core
  stays torch-free.
- ``bigym.loco.env``: the controller-in-the-loop environment
  (gym loop + make + get_demos + specs is the stable public surface),
  assembled from ``lowerbody`` (the controller in the loop),
  ``event_progress`` (progress label and shaping) and ``fingerprint``
  (substrate identity).
- ``bigym.loco.demos``: native demo schema v1 + npz/metadata IO.
- ``bigym.loco.eval``: the frozen evaluation protocol.
- ``bigym.loco.config``: :class:`EnvConfig`, every setting an env is built
  from; its defaults are the official benchmark values.
- ``bigym.loco.tasks``: the task registry (class, budget and official-config
  differences per task).

Public factory::

    from bigym.loco import make
    env = make("move_plate")                      # the official configuration
    env = make("move_plate", camera_keys=("head",))

Controllers and tasks from your own package: :class:`BackendBinding` with
:func:`register_backend`, and :func:`register_task`; or name the object
directly as ``"pkg.module:ATTR"``.
"""

from bigym.loco.action_representation import (
    ABSOLUTE,
    ACTION_REPRESENTATIONS,
    UPPER_DELTA,
    UpperDeltaAccumulator,
    convert_episode_action_representation,
)
from bigym.loco.adapters import BackendBinding, register_backend
from bigym.loco.base import LowerBodyBase
from bigym.loco.command import (
    CommandField,
    CommandKind,
    CommandSpec,
    velocity_spec,
)
from bigym.loco.config import ControllerConfig, EnvConfig, WholeBodyConfig
from bigym.loco.controller import LowerBodyController, OutputSpec


def make(task_name, config=None, **overrides):
    """Create a BiGym env; with no overrides, the official configuration.

    ``config`` is None or an :class:`EnvConfig`; keyword ``overrides`` are
    EnvConfig fields applied last. See
    :func:`bigym.loco.env.make`.
    """
    from bigym.loco.env import make as _make

    return _make(task_name, config, **overrides)


def make_gym(task_name, config=None, **overrides):
    """:func:`make` behind the gymnasium API (see :mod:`bigym.loco.gym_adapter`)."""
    from bigym.loco.gym_adapter import make_gym as _make_gym

    return _make_gym(task_name, config, **overrides)


def register_task(name, spec):
    """Register a :class:`~bigym.loco.tasks.TaskSpec` so ``make(name)`` builds it.

    A name can be registered once; registering it again (a built-in task
    included) raises ValueError. ``make("pkg.module:ATTR")`` builds a
    TaskSpec without registering it.
    """
    from bigym.loco.tasks import register_task as _register_task

    _register_task(name, spec)


__all__ = [
    "CommandField",
    "CommandKind",
    "CommandSpec",
    "velocity_spec",
    "LowerBodyController",
    "OutputSpec",
    "LowerBodyBase",
    "ABSOLUTE",
    "UPPER_DELTA",
    "ACTION_REPRESENTATIONS",
    "UpperDeltaAccumulator",
    "convert_episode_action_representation",
    "ControllerConfig",
    "EnvConfig",
    "WholeBodyConfig",
    "make",
    "make_gym",
    "BackendBinding",
    "register_backend",
    "register_task",
]
