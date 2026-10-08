"""The scene's spec, its compiled model and data, advanced with warnings checked."""

from __future__ import annotations

import mujoco
import numpy as np

WARNING_NAMES = np.array(list(mujoco.mjtWarning.__members__))


class PhysicsError(RuntimeError):
    """MuJoCo raised a warning while the simulation was advanced."""


class Simulation:
    """A scene built as one ``MjSpec``, compiled once into a model and its data.

    Robots and props add themselves to ``spec`` and keep their elements; after
    :meth:`compile` those elements index the arrays directly, e.g.
    ``simulation.data.bind(body).xpos`` or ``simulation.model.bind(geom).rgba``.
    ``step`` and ``forward`` raise :class:`PhysicsError` when MuJoCo raises a
    new warning, e.g. on a diverging state.
    """

    model: mujoco.MjModel
    data: mujoco.MjData

    def __init__(self, spec: mujoco.MjSpec):
        """Start a scene from ``spec``."""
        self.spec = spec

    def compile(self) -> None:
        """Compile the spec and reset the data."""
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.reset()

    def step(self, substeps: int = 1) -> None:
        """Advance ``substeps`` physics steps."""
        warnings = self.data.warning.number.copy()
        mujoco.mj_step(self.model, self.data, substeps)
        self.raise_on_new_warnings(warnings)

    def forward(self) -> None:
        """Recompute every quantity derived from the current state."""
        warnings = self.data.warning.number.copy()
        mujoco.mj_forward(self.model, self.data)
        self.raise_on_new_warnings(warnings)

    def reset(self) -> None:
        """Reset the data to the model's initial state, forwarded without actuation."""
        mujoco.mj_resetData(self.model, self.data)
        flags = self.model.opt.disableflags
        self.model.opt.disableflags = flags | mujoco.mjtDisableBit.mjDSBL_ACTUATION
        try:
            self.forward()
        finally:
            self.model.opt.disableflags = flags

    def raise_on_new_warnings(self, before: np.ndarray) -> None:
        """Raise :class:`PhysicsError` if any warning count grew since ``before``."""
        raised = self.data.warning.number > before
        if raised.any():
            names = ", ".join(WARNING_NAMES[: len(raised)][raised])
            raise PhysicsError(f"Physics state is invalid. Warning(s) raised: {names}")
