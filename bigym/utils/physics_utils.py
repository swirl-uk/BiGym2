"""Physics utils."""

from __future__ import annotations

import math
from typing import Any, Iterable, Union

import mujoco
import numpy as np

from bigym.simulation import Simulation

# Contacts farther apart than this do not count as a collision.
COLLISION_MARGIN = 1e-8

Colliders = Union[mujoco.MjsGeom, Iterable[mujoco.MjsGeom], Any]


def set_position_servo(actuator: mujoco.MjsActuator, kp: float) -> None:
    """Make ``actuator`` a position servo with stiffness ``kp`` and no velocity gain."""
    actuator.set_to_position(kp=kp)
    # set_to_position stores -kv, a -0.0 here; the published models hold +0.0.
    actuator.biasprm[2] = 0.0


def critical_damping(
    stiffness: float, joint: mujoco.MjsJoint, model: mujoco.MjModel
) -> float:
    """Damping that critically damps ``joint`` under ``stiffness`` at the initial pose."""
    joint_mass = model.dof_M0[model.bind(joint).dofadr[0]]
    return 2 * math.sqrt(joint_mass * stiffness)


def is_target_reached(
    simulation: Simulation,
    actuator: mujoco.MjsActuator,
    joint: mujoco.MjsJoint,
    tolerance: float,
) -> bool:
    """Whether ``joint`` is within ``tolerance`` of the target of its position actuator."""
    qpos = float(simulation.data.bind(joint).qpos.item())
    ctrl = float(simulation.data.bind(actuator).ctrl.item())
    if np.any(actuator.ctrlrange):
        ctrl = np.clip(ctrl, *actuator.ctrlrange)
    return np.abs(qpos - ctrl) <= tolerance


def set_joint_position(
    simulation: Simulation,
    joint: mujoco.MjsJoint,
    value: float,
    normalized: bool = False,
):
    """Set the position of ``joint``, optionally as a fraction of its range, at rest."""
    if normalized:
        value = np.interp(np.clip(value, 0, 1), [0, 1], joint.range)
    bound = simulation.data.bind(joint)
    bound.qpos = value
    bound.qvel *= 0
    bound.qacc *= 0


def get_joint_position(
    simulation: Simulation, joint: mujoco.MjsJoint, normalized: bool = False
) -> float:
    """Position of ``joint``, optionally normalized by its range."""
    value = float(simulation.data.bind(joint).qpos.item())
    if not normalized:
        return value
    lower, upper = joint.range
    # lower + upper, not the span: the published success thresholds use this scale.
    return (value - lower) / (lower + upper)


def set_body_position(
    model: mujoco.MjModel, body: mujoco.MjsBody, position: np.ndarray
):
    """Move ``body`` relative to its parent in the compiled model."""
    bound = model.bind(body)
    bound.pos = position
    # The pose no longer comes from the compiler, so its frame flag is dropped.
    bound.sameframe = 0


def set_body_quaternion(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body: mujoco.MjsBody,
    quaternion: np.ndarray,
):
    """Rotate ``body`` relative to its parent in the compiled model.

    The body's world rotation in ``data`` follows at once, so reads before the
    next forward pass see it.
    """
    bound = model.bind(body)
    bound.quat = quaternion
    bound.sameframe = 0
    mujoco.mju_quat2Mat(data.bind(body).xmat, np.asarray(quaternion, dtype=np.float64))


def get_quaternion(
    data: mujoco.MjData, element: mujoco.MjsBody | mujoco.MjsSite
) -> np.ndarray:
    """World rotation of a body or site as a quaternion (w, x, y, z)."""
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, data.bind(element).xmat)
    return quaternion


def zyz_euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) of intrinsic z-y-z Euler angles (alpha, beta, gamma)."""
    alpha, beta, gamma = euler
    return np.array(
        [
            np.cos(beta / 2) * np.cos((alpha + gamma) / 2),
            -np.sin(beta / 2) * np.sin((alpha - gamma) / 2),
            np.sin(beta / 2) * np.cos((alpha - gamma) / 2),
            np.cos(beta / 2) * np.sin((alpha + gamma) / 2),
        ]
    )


def get_colliders(obj: Colliders) -> Iterable[mujoco.MjsGeom]:
    """Collider geoms of a geom, a collection of geoms or a prop."""
    if isinstance(obj, mujoco.MjsGeom):
        return [obj]
    if isinstance(obj, Iterable):
        return obj
    return obj.colliders


def has_collided_collections(
    simulation: Simulation, colliders_1: Colliders, colliders_2: Colliders
) -> bool:
    """Whether any geom of ``colliders_1`` touches any geom of ``colliders_2``."""
    ids_1 = [geom.id for geom in get_colliders(colliders_1)]
    ids_2 = [geom.id for geom in get_colliders(colliders_2)]
    contact = simulation.data.contact
    touching = ~(contact.dist > COLLISION_MARGIN)
    geom_1 = contact.geom1[touching]
    geom_2 = contact.geom2[touching]
    return bool(
        np.any(
            (np.isin(geom_1, ids_1) & np.isin(geom_2, ids_2))
            | (np.isin(geom_2, ids_1) & np.isin(geom_1, ids_2))
        )
    )
