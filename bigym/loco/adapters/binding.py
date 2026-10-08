"""What a lower-body backend provides to put its controller in the env's loop.

Each backend is a :class:`BackendBinding` registered under its name with
:func:`bigym.loco.register_backend` (the built-in ``groot_wbc_g1`` included),
or named by a ``"pkg.module:ATTR"`` string. Robot facts that hold for every
backend stay in the robot config (``bigym/robots/configs``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bigym.bigym_env import BiGymEnv
    from bigym.loco.config import ControllerConfig
    from bigym.loco.controller import LowerBodyController
    from bigym.robots.robot import Robot


class BackendBinding:
    """A lower-body backend's hooks into env construction.

    The env calls the hooks in this order:
    :meth:`robot_cls` while building the task env, then
    :meth:`build_controller` and :meth:`configure_model` on the built env.
    ``env`` is the task's :class:`~bigym.bigym_env.BiGymEnv`; a controller
    may rely on ``env.model``, ``env.data``, ``env.robot`` and
    ``env.action_space``. ``config`` is the env's
    :class:`~bigym.loco.config.ControllerConfig`: a backend reads the fields
    that apply to it and may ignore the rest.

    Subclass it and set :attr:`robot_models`; override
    :meth:`build_controller`. The controller implements
    :class:`~bigym.loco.controller.LowerBodyController`, most easily by
    subclassing :class:`~bigym.loco.base.LowerBodyBase`.
    """

    #: ``EnvConfig.robot_model`` names the backend drives.
    robot_models: tuple[str, ...] = ()
    #: True when the backend can run with the pelvis roll/pitch left passive
    #: (``ControllerConfig.passive_base_tilt``).
    supports_passive_base_tilt: bool = False

    def robot_cls(
        self, robot_cls: type[Robot], config: ControllerConfig
    ) -> type[Robot]:
        """The robot class to build the task env with.

        ``robot_cls`` is the robot model's floating-base class. Return it
        (the default), or a variant with the joints the controller drives
        actuated and their PD gains set (``G1Dex1.variant``).
        """
        return robot_cls

    def build_controller(
        self, env: BiGymEnv, config: ControllerConfig, *, control_dt: float
    ) -> LowerBodyController:
        """Build the controller on the freshly built task env.

        ``control_dt`` is the env's control step in seconds.
        """
        raise NotImplementedError

    def configure_model(self, env: BiGymEnv) -> None:
        """Adjust the compiled model once the controller is built (default: none)."""
