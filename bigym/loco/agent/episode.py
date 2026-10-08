"""One episode of a scripted policy, shared by the agent's runner and the evaluator.

A policy is a class named ``Policy`` in a single file with two methods::

    reset(obs, tools)       called once at the start of every episode
    act(obs, tools) -> raw  called every control step, returns the action

Running the identical loop on both sides is what makes a policy behave the
same during development and during hidden-seed evaluation.

This module is copied into an agent sandbox as ``harness/episode.py``, so it
must keep working when imported as a top-level module.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np


class Tools:
    """What a policy may call besides reading observations (no reset/step)."""

    def __init__(self, env):
        """Bind the tools to one environment handle.

        Args:
            env: A client ``Env`` or the evaluator's in-process ``LocalEnv``.
        """
        self._env = env
        self.info = env.info

    def hold_action(self):
        """Return the physical action that holds the current pose."""
        return self._env.hold_action()

    def ik(self, **kw):
        """Return arm joint targets reaching world-frame wrist targets."""
        return self._env.ik(**kw)

    def render(self, camera="head", path=None):
        """Save a PNG from one of the robot's cameras and return its path."""
        return self._env.render(camera, path)

    def image(self, camera="head", width=84, height=84):
        """Return an RGB uint8 array from a robot-mounted camera."""
        return self._env.image(camera, width, height)

    def camera_info(self):
        """Return the calibration of every camera."""
        return self._env.camera_info()

    def pixel_to_ray(self, camera, u, v, width, height):
        """Return the world-frame ray through a pixel of a rendered image."""
        return self._env.pixel_to_ray(camera, u, v, width, height)


def load_policy(path: Path):
    """Import ``path`` and instantiate the ``Policy`` class it defines.

    Args:
        path: Path to the policy file.

    Returns:
        A fresh ``Policy`` instance.

    Raises:
        AttributeError: The file defines no ``Policy`` class.
    """
    path = Path(path).resolve()
    if str(path.parent) not in sys.path:  # helper modules next to policy.py are allowed
        # Appended, never inserted at 0: at position 0 a file left beside policy.py
        # would shadow any module the evaluator imports lazily after this point.
        # Last place still resolves genuinely new names, which is all a helper needs.
        sys.path.append(str(path.parent))
    spec = importlib.util.spec_from_file_location(
        f"agent_policy_{abs(hash(str(path)))}", path
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if not hasattr(mod, "Policy"):
        raise AttributeError(
            f"{path} must define a class Policy with reset(obs, tools) and act(obs, tools)"
        )
    return mod.Policy()


def run_episode(env, policy, seed: int, max_steps: int | None = None) -> dict:
    """Run one episode and return its record.

    Args:
        env: A client ``Env`` or the evaluator's in-process ``LocalEnv``.
        policy: An object with ``reset(obs, tools)`` and ``act(obs, tools)``.
        seed: Seed handed to ``env.reset``.
        max_steps: Step cap; defaults to the episode's own time limit.

    Returns:
        A per-episode record (seed, success, length, reward, termination,
        fell, wall_s, final_obs).
    """
    tools = Tools(env)
    obs = env.reset(seed)
    policy.reset(obs, tools)
    total, length, done, info = 0.0, 0, False, {}
    t0 = time.time()
    limit = max_steps or int(obs.get("time_limit") or 10**9)
    while not done and length < limit:
        raw = np.asarray(policy.act(obs, tools), dtype=np.float32).reshape(-1)
        obs, r, done, info = env.step(raw)
        total += r
        length += 1
    fell = bool(obs.get("fell", False))
    # Scoring rule (protocol v1): an episode in which the robot falls is a failure,
    # whatever reward the task itself handed out.
    success = total >= 0.25 and not fell
    return {
        "seed": int(seed),
        "success": int(success),
        "length": int(length),
        "reward": float(total),
        "termination": (
            ("fell" if fell and total >= 0.25 else info.get("termination"))
            or ("timeout" if not done else "unknown")
        ),
        "fell": fell,
        "wall_s": round(time.time() - t0, 2),
        "final_obs": {
            k: (
                v.tolist()
                if isinstance(v, np.ndarray) and v.size <= 16
                else (v if isinstance(v, (int, float, bool)) else None)
            )
            for k, v in obs.items()
            if k != "low_dim_obs"
        },
    }
