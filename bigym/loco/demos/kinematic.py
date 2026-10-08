"""Kinematic replay of stored simulator states.

Every demo player in BiGym shows a recorded episode the same way: build the
task's environment, reset it on the episode's seed (props and reach targets
land where the demonstration found them), then per frame write the stored
``full_qpos``, call ``mj_forward`` and run the task's ``_on_step`` hook, which
paints scene state such as the reach-target highlight. No physics is
integrated and no lower-body policy runs. ``bigym-view``, the all-tasks grid
(``scripts/demo_grid.py``), the camera re-renderer
(:class:`bigym.loco.demos.rerender.BatchRenderer`) and ``bigym-agent
demo-video`` all go through these helpers.
"""

from __future__ import annotations

import mujoco
import numpy as np

from bigym.loco import make


def replay_env(task: str):
    """Build the official environment of a task, with no cameras.

    ``make(task)`` is the benchmark's own construction (embodiment,
    lower-body backend, warmup, ``success_hold_seconds`` and
    ``reach_tolerance``), so the props land where a demonstration's seed puts
    them and the success predicate that paints the reach-target highlight is
    the real one. ``camera_keys=()`` skips the onboard renderers: a player
    that draws the scene itself never reads a camera observation.

    Args:
        task: The task name.

    Returns:
        The environment.
    """
    return make(task, camera_keys=())


def run_task_hook(inner) -> None:
    """Run the task's ``_on_step`` the way a real ``step()`` does.

    ``step()`` runs ``_on_step`` before the observation is taken, and some
    tasks paint scene state there (the reach-target highlight). The step
    cache is cleaned first, or ``_success()`` answers from the previous
    frame.

    Args:
        inner: The raw ``BiGymEnv``.
    """
    inner._step_cache.clean()
    inner._on_step()


def pose(inner, qpos, qvel=None) -> None:
    """Pose the scene at one stored state: write it, forward, run the hook.

    Args:
        inner: The raw ``BiGymEnv``.
        qpos: One frame of ``full_qpos`` (a prefix of ``data.qpos``).
        qvel: One frame of ``full_qvel``, or None to leave velocities alone.
    """
    model, data = inner.model, inner.data
    qpos = np.asarray(qpos, dtype=np.float64)
    data.qpos[: qpos.shape[0]] = qpos
    if qvel is not None:
        qvel = np.asarray(qvel, dtype=np.float64)
        data.qvel[: qvel.shape[0]] = qvel
    mujoco.mj_forward(model, data)
    run_task_hook(inner)
