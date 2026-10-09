"""Gymnasium adapter for the controller-in-the-loop BiGym 2.0 environment.

``bigym.loco.make`` returns an ``ExtendedTimeStepWrapper``
whose ``reset``/``step`` yield a dm_env-style :class:`ExtendedTimeStep`
namedtuple (``step_type`` / ``discount`` / separate ``rgb_obs`` +
``low_dim_obs`` fields). External RL/IL libraries (Stable-Baselines3,
CleanRL, LeRobot's gym wrappers) speak the gymnasium 5-tuple instead. This
module is the thin, lossless translation between the two::

    from bigym.loco import make_gym

    env = make_gym("move_plate")            # official env, gymnasium API
    obs, info = env.reset(seed=620000)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

Nothing about the substrate changes: the adapter owns no physics, no
reward shaping and no episode bookkeeping of its own. Everything it reports
comes from the wrapped env, ``info["success"]`` included
(:func:`bigym.loco.eval.is_success`).

Action space: ``env.action_space``, the agent-facing ``Box(-1, 1)`` that
the env de-normalizes onto ``env.action_stats`` (``action_spec()`` carries
no bounds). Actions are cast to float32 and clipped into it: the env rejects
values outside ``[-1, 1]``, and squashed-Gaussian policies emit
``1.0000001``.

Observation space: a ``Dict`` of ``rgb`` (uint8, ``(num_cameras,
3 * frame_stack, H, W)``, camera-major, newest frame last) and ``state``
(float32, ``(low_dim * frame_stack,)``; the component offsets are in
``metadata["bigym.low_dim_component_slices"]``).

Episode end: ``terminated = last and discount == 0`` (the task terminal, or
a physics error, which the env ends with zero reward); ``truncated = last
and not terminated`` (the episode budget).

Seeding: ``reset(seed=s)`` pins the episode as the eval protocol does;
``reset()`` draws the episode seed from the adapter's ``np_random``, so an
episode chain is reproducible from one initial seed (the gymnasium
contract ``check_env`` verifies).

Substrate fingerprint: computed once at construction and carried in
``metadata["bigym.substrate"]`` and the reset ``info["substrate"]``. The
env pins ``compiled_model_sha256`` before its first reset, so task
randomization does not change it.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium
import numpy as np
from gymnasium import spaces

from bigym.loco.config import EnvConfig
from bigym.loco.env import make
from bigym.loco.eval.protocol import is_success

__all__ = ["GymnasiumEnv", "make_gym"]


class GymnasiumEnv(gymnasium.Env):
    """``gymnasium.Env`` view of a ``bigym.loco`` controller-in-the-loop env.

    Args:
        env: the object returned by :func:`bigym.loco.make` (an
            ``ExtendedTimeStepWrapper``).
            The adapter does not take ownership of construction, so a caller
            that already built an env — with custom action stats, say —
            can wrap it directly.
        render_camera: camera key used by :meth:`render`; falls back to the
            wrapped env's own free-camera render when that camera is not in
            the observation set.

    The wrapped env stays reachable as ``.bigym_env``, and unknown public
    attributes are forwarded to it, so ``action_spec``,
    ``low_dim_component_slices`` and friends remain available.
    :meth:`get_demos` is the gymnasium-shaped view of the demonstrations.
    """

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 50}
    action_space: spaces.Box

    def __init__(self, env: Any, render_camera: str = "head"):
        """Build the gymnasium spaces and metadata from the wrapped env."""
        self.bigym_env = env
        self.render_mode = "rgb_array"

        rgb_spec = env.rgb_observation_spec()
        state_spec = env.low_dim_observation_spec()
        self.observation_space = spaces.Dict(
            {
                "rgb": spaces.Box(
                    low=0, high=255, shape=tuple(rgb_spec.shape), dtype=np.uint8
                ),
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=tuple(state_spec.shape),
                    dtype=np.float32,
                ),
            }
        )
        # See the module docstring: env.action_space is the OUTER normalized
        # space; action_spec() carries no bounds.
        inner_action_space = env.action_space
        self.action_space = spaces.Box(
            low=np.asarray(inner_action_space.low, dtype=np.float32),
            high=np.asarray(inner_action_space.high, dtype=np.float32),
            shape=tuple(inner_action_space.shape),
            dtype=np.float32,
        )

        # Frozen for the lifetime of the env, and expensive (hashes the
        # compiled model), so it is computed exactly once.
        self._substrate = env.substrate_fingerprint()
        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": int(round(1.0 / env.control_step_seconds)),
            "bigym.substrate": self._substrate,
            "bigym.action_layout": env.wholebody_action_layout(),
            "bigym.low_dim_component_slices": env.low_dim_component_slices(),
        }

        camera_keys = env.config.camera_keys
        self._render_camera_index = (
            camera_keys.index(render_camera) if render_camera in camera_keys else None
        )
        self._last_rgb: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # gymnasium.Env API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Start a seeded episode and return ``(observation, info)``."""
        super().reset(seed=seed)
        if seed is None:
            # Keep the episode chain reproducible from the adapter's own RNG
            # rather than the env's global-numpy fallback (see module docstring).
            episode_seed = int(self.np_random.integers(2**32))
        else:
            episode_seed = int(seed)

        kwargs: dict[str, Any] = {"seed": episode_seed}
        if options is not None:
            kwargs["options"] = options
        time_step = self.bigym_env.reset(**kwargs)
        obs = self._observation(time_step)
        info = self._info(time_step, terminal=False)
        info["substrate"] = self._substrate
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Clip the action into the space, step the env, return the 5-tuple."""
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        # denormalize_action asserts |a| <= 1; a squashed policy can round
        # a hair outside, and an AssertionError deep in the env is a poor way
        # to report that.
        action = np.clip(action, self.action_space.low, self.action_space.high)

        time_step = self.bigym_env.step(action)
        reward = float(time_step.reward)
        last = bool(time_step.last())
        terminated = bool(last and float(time_step.discount) == 0.0)
        truncated = bool(last and not terminated)

        obs = self._observation(time_step)
        info = self._info(time_step, terminal=terminated or truncated)
        return obs, reward, terminated, truncated, info

    def get_demos(
        self, num_demos: int = -1, only_successful: bool = False
    ) -> list[dict[str, Any]]:
        """Return this task's demonstrations as gymnasium-shaped trajectories.

        The task's demonstrations are downloaded from the Hugging Face Hub on
        first use (see ``bigym.loco.demos.hub``; ``bigym-download`` fetches
        ahead of time). Each trajectory is a dict::

            {
              "obs":        {"rgb": uint8 [T + 1, cams, 3 * stack, H, W],
                             "state": float32 [T + 1, D * stack]},
              "action":     float32 [T, A]      # the env's normalized action space
              "reward":     float32 [T],
              "terminated": bool [T],
              "truncated":  bool [T],
              "success":    bool,
            }

        so transition ``i`` is ``obs[i] --action[i]--> obs[i + 1]`` with
        ``reward[i]``, exactly what ``step`` returns. Observations are
        stacked/normalized like the env's own, and actions are normalized
        over ``env.action_stats``, so ``action`` values and this env's
        ``action_space`` mean the same thing. ``num_demos < 0`` loads every
        episode; ``only_successful`` keeps the successful ones.
        """
        inner = self.bigym_env
        episodes = inner.load_training_episodes(num_demos)
        trajectories: list[dict[str, Any]] = []
        # Free each raw episode once converted.
        episodes.reverse()
        while episodes:
            episode = episodes.pop()
            success = bool(inner.demo_is_successful(episode))
            if only_successful and not success:
                continue
            rgb, state = inner.demo_observations(episode)
            actions = np.asarray(episode["action"], dtype=np.float32)
            rewards = np.asarray(episode["reward"], dtype=np.float32).reshape(-1)
            discounts = np.asarray(episode["discount"], dtype=np.float32).reshape(-1)
            length = len(actions) - 1  # row 0 is the reset row
            if length < 1:
                continue
            terminated = np.zeros(length, dtype=bool)
            truncated = np.zeros(length, dtype=bool)
            terminated[-1] = bool(discounts[-1] == 0.0)
            truncated[-1] = not terminated[-1]
            trajectories.append(
                {
                    "obs": {"rgb": rgb, "state": state},
                    "action": actions[1:],
                    "reward": rewards[1:],
                    "terminated": terminated,
                    "truncated": truncated,
                    "success": success,
                }
            )
        if not trajectories:
            raise RuntimeError("No demonstrations available after filtering")
        return trajectories

    def render(self) -> Optional[np.ndarray]:
        """The head-camera frame as ``(H, W, 3)`` uint8, when available.

        Falls back to the wrapped env's own free-camera render (a larger
        viewport image) when the requested camera is not part of the
        observation set, or before the first reset.
        """
        if self._render_camera_index is not None and self._last_rgb is not None:
            # rgb_obs is (num_cameras, 3 * frame_stack, H, W) with the newest
            # frame in the trailing channels.
            frame = self._last_rgb[self._render_camera_index][-3:]
            return np.ascontiguousarray(np.transpose(frame, (1, 2, 0)))
        return self.bigym_env.render()

    def close(self) -> None:
        """Close the wrapped env."""
        self.bigym_env.close()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _observation(self, time_step: Any) -> dict[str, np.ndarray]:
        rgb = np.asarray(time_step.rgb_obs, dtype=np.uint8)
        self._last_rgb = rgb
        return {
            "rgb": rgb,
            "state": np.asarray(time_step.low_dim_obs, dtype=np.float32),
        }

    def _info(self, time_step: Any, *, terminal: bool) -> dict[str, Any]:
        progress = float(time_step.event_progress)
        fell = bool(self.bigym_env.episode_fell())
        success = is_success(self.bigym_env)
        info: dict[str, Any] = {
            "success": success,
            "fell": fell,
            # NaN when event progress is disabled; None keeps the info dict
            # comparable (NaN != NaN breaks gymnasium's data_equivalence).
            "event_progress": progress if np.isfinite(progress) else None,
            "discount": float(time_step.discount),
        }
        if terminal:
            info["termination"] = self.bigym_env.episode_termination()
            info["truncation_reason"] = self.bigym_env.last_truncation_reason()
        return info

    def __getattr__(self, name: str) -> Any:
        """Forward unknown public attributes to the wrapped env."""
        if name.startswith("_"):
            raise AttributeError(name)
        inner = self.__dict__.get("bigym_env")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)


def make_gym(
    task_name: str,
    config: EnvConfig | None = None,
    **overrides: Any,
) -> GymnasiumEnv:
    """Build a BiGym 2.0 env and return it behind the gymnasium API.

    Args:
        task_name: registered task (``bigym.loco.tasks.TASKS``) or a
            ``"pkg.module:ATTR"`` reference to a ``TaskSpec``.
        config: None for the official configuration or an ``EnvConfig``, as
            for :func:`bigym.loco.make`.
        **overrides: EnvConfig fields applied last.

    Returns:
        A :class:`GymnasiumEnv`.
    """
    return GymnasiumEnv(make(task_name, config, **overrides))
