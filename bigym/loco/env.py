"""Controller-in-the-loop BiGym environment (the bigym.loco public env).

Wraps a BiGym task env with a lower-body backend (bigym.loco.adapters): the
outer action space carries base *commands* (velocity/height/torso-pitch)
plus upper-body joint targets, and the backend closes the loop on the legs.
Also provides the research-facing dm_env-style TimeStep interface, frame
stacking, demonstration loading/rescaling, and the public spec surface
(action_spec / low_dim_raw_observation_spec / rgb_raw_observation_spec /
low_dim_component_slices).

:class:`BiGym` assembles the parts: the outer action/observation layout
lives here; the controller in the loop is :mod:`bigym.loco.lowerbody`,
event progress and its shaping :mod:`bigym.loco.event_progress`, the
substrate record :mod:`bigym.loco.fingerprint` and demo decoding
:mod:`bigym.loco.demos.episodes`.

Public factory: :func:`make` (also re-exported as ``bigym.loco.make``),
which builds an env from a :class:`~bigym.loco.config.EnvConfig`.
"""

import dataclasses
from collections import deque
from typing import Any, Dict, Optional, Union, cast

import numpy as np
from dm_env import StepType, specs
from gymnasium import spaces

from bigym.action_modes import DEFAULT_DOFS, JointPositionActionMode, PelvisDof
from bigym.bigym_env import CONTROL_FREQUENCY_MAX
from bigym.loco import fingerprint
from bigym.loco.action_representation import ABSOLUTE, UpperDeltaAccumulator
from bigym.loco.config import EnvConfig, resolve_config
from bigym.loco.demos import episodes as demo_episodes
from bigym.loco.event_progress import EventProgress
from bigym.loco.fingerprint import SUBSTRATE_VERSION
from bigym.loco.lowerbody import LowerBody, yaw_from_floating_qpos
from bigym.loco.tasks import (
    check_task_initialization_profile,
    resolve_task_name,
    task_spec,
)
from bigym.loco.timestep import ExtendedTimeStep, ExtendedTimeStepWrapper, TimeStep
from bigym.robots.configs import ROBOT_MODELS
from bigym.utils.observation_config import CameraConfig, ObservationConfig

__all__ = [
    "SUBSTRATE_VERSION",
    "BiGym",
    "ExtendedTimeStep",
    "ExtendedTimeStepWrapper",
    "TimeStep",
    "make",
]


class BiGym:
    """Controller-in-the-loop BiGym env with a dm_env-style TimeStep API.

    Wraps a BiGym task env and, when configured, a lower-body backend:
    the outer action carries base commands plus upper-body joint
    targets. Adds frame stacking, the low-dim/RGB/action spec surface,
    demonstration loading and demo action rescaling.
    """

    def __init__(
        self,
        task_name: str,
        config: Optional[EnvConfig] = None,
        *,
        episode_seed_rng: Optional[np.random.Generator] = None,
    ):
        """Build the task env, lower-body controller, frame stacks and specs.

        Args:
            task_name: A registered task (``bigym.loco.tasks.TASKS``) or a
                ``"pkg.module:ATTR"`` reference to a ``TaskSpec``.
            config: The settings to build from; None is the task's official
                configuration. A None ``episode_length`` takes the task's
                budget.
            episode_seed_rng: Generator that draws per-episode seeds when
                ``reset`` gets no explicit seed.
        """
        self.task_name = resolve_task_name(task_name)
        spec = task_spec(self.task_name)
        self.official_config = spec.config()
        if config is None:
            config = self.official_config
        episode_length = config.episode_length
        if episode_length is None:
            episode_length = spec.episode_length
            config = dataclasses.replace(config, episode_length=episode_length)
        self.config = config
        self._time_limit = episode_length // config.demo_down_sample_rate
        check_task_initialization_profile(self.task_name, config.initialization_profile)
        self._episode_seed_rng = episode_seed_rng
        self._event_progress = EventProgress(self.task_name, config)
        self.lowerbody = LowerBody(config.controller, robot_model=config.robot_model)
        # Fall record: the controller's is_failed() is polled after every
        # step and latched per episode. Record-only: a fallen robot still
        # runs to the time limit, but the episode is distinguishable from a
        # slow failure.
        self._episode_fell = False
        self._episode_succeeded = False
        self._last_truncation: Optional[str] = None
        # Identity of the compiled model BEFORE any reset: task randomization
        # writes body poses into mjModel at reset, so hashing live would give
        # a per-episode nonce instead of a substrate identity.
        self._compiled_model_sha256: Optional[str] = None
        self._outer_floating_dofs = (
            [PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]
            if self.config.enable_all_floating_dof
            else list(DEFAULT_DOFS)
        )
        if self.lowerbody.pitch_cmd_enabled:
            # Command-only slot: no inner floating-base counterpart (the
            # waist_pitch drive lives inside the lower-body backend).
            self._outer_floating_dofs.append(PelvisDof.RY)
        self.outer_action_floating_dofs = (
            list(self._outer_floating_dofs) if self.config.control_pelvis else []
        )
        self._outer_action_dim = None
        self._outer_action_low = None
        self._outer_action_high = None
        self._outer_base_to_inner_base_idx = None
        self.outer_action_base_to_inner_base_idx = None
        self.inner_floating_dofs = None
        self._proprio_full_qpos_dim = None
        self._proprio_keep_qpos_idx = None
        self.limb_name_to_full_index: dict[str, int] = {}
        self.outer_limb_actuator_names: tuple[str, ...] = ()
        self.upper_delta_accumulator = None

        self.build_inner_env()
        self._initialize_action_representation()
        self._initialize_frame_stack()
        self._construct_action_and_observation_spaces()

    def low_dim_observation_spec(self) -> specs.Array:
        """Return the dm_env spec of the frame-stacked low-dim observation."""
        shape = self.low_dim_observation_space.shape
        spec = specs.Array(shape, np.float32, "low_dim_obs")
        return spec

    def low_dim_raw_observation_spec(self) -> specs.Array:
        """Return the dm_env spec of one unstacked low-dim observation."""
        shape = self.low_dim_raw_observation_space.shape
        spec = specs.Array(shape, np.float32, "low_dim_obs")
        return spec

    def low_dim_component_slices(self) -> dict[str, tuple[int, int]]:
        """Return each state key's [start, stop) slice into the low-dim vector."""
        offset = 0
        slices: dict[str, tuple[int, int]] = {}
        for state_key in self.config.state_keys:
            if (
                state_key == "proprioception"
                and self._proprio_keep_qpos_idx is not None
                and self._proprio_full_qpos_dim is not None
            ):
                dim = int(self._proprio_keep_qpos_idx.shape[0]) * 2
            elif (
                state_key
                in (
                    "proprioception_floating_base",
                    "proprioception_floating_base_actions",
                )
                and self._outer_base_to_inner_base_idx is not None
            ):
                # Outer base dofs without an inner base joint (the RY torso-
                # pitch command slot) map to -1 and are not observed; the
                # observation itself counts only mapped entries (see the
                # low-dim spec), so the slice must too.
                dim = int((self._outer_base_to_inner_base_idx >= 0).sum())
            else:
                state_space = cast(
                    spaces.Box, self.inner_env.observation_space[state_key]
                )
                dim = int(state_space.shape[-1])
            slices[state_key] = (offset, offset + dim)
            offset += dim
        return slices

    def rgb_observation_spec(self) -> specs.Array:
        """Return the dm_env spec of the frame-stacked RGB observation."""
        shape = self.rgb_observation_space.shape
        spec = specs.Array(shape, np.uint8, "rgb_obs")
        return spec

    def rgb_raw_observation_spec(self) -> specs.Array:
        """Return the dm_env spec of one unstacked RGB observation."""
        shape = self.rgb_raw_observation_space.shape
        spec = specs.Array(shape, np.uint8, "rgb_obs")
        return spec

    def action_spec(self) -> specs.Array:
        """Return the dm_env spec of the normalized outer action."""
        shape = self.action_space.shape
        spec = specs.Array(shape, np.float32, "action")
        return spec

    def step(self, action):
        """Advance one outer control step and return the resulting TimeStep.

        Denormalizes the action, refreshes the lower-body command, clips the
        expanded action to the inner action space, and then applies the fall
        latch, event-progress shaping and the episode time limit.
        """
        raw_action = self.denormalize_action(action)
        self.lowerbody.last_control_info = {}
        self.lowerbody.update_command(self, raw_action)
        if self.lowerbody.controller is None and raw_action.shape[0] == int(
            self.inner_env.action_space.shape[0]
        ):
            expanded_action = raw_action
        else:
            expanded_action = self.lowerbody.expand_action(self, raw_action)
        # Guard against tiny float overflows (e.g. 0.34000003 vs 0.34) that can
        # fail strict bounds checks.
        expanded_action = np.clip(
            expanded_action,
            self.inner_env.action_space.low,
            self.inner_env.action_space.high,
        ).astype(raw_action.dtype, copy=False)

        bigym_obs, reward, terminated, truncated, info = self.inner_env.step(
            expanded_action
        )
        inner_truncated = bool(truncated)
        self._update_fall_latch()
        if inner_truncated:
            # BiGym uses truncation for unhealthy MuJoCo states. Treat these as
            # terminal failures so physics explosions cannot become successes.
            reward = 0.0
            terminated = True
        succeeded = not inner_truncated and self.inner_env.success
        self._episode_succeeded = self._episode_succeeded or succeeded
        obs = self._extract_obs(bigym_obs)
        self._step_counter += 1
        previous_event_progress = self._event_progress.max
        progress = self._event_progress.update(self.inner_env, succeeded)
        if (
            not inner_truncated
            and self.config.event_reward_shaping_enabled
            and not succeeded
        ):
            reward = float(reward) + self._event_progress.shaping(
                previous_event_progress, progress
            )

        # Timelimit
        time_limit_truncated = self._step_counter >= self._time_limit
        truncated = bool(inner_truncated or time_limit_truncated)
        self._last_truncation = (
            "physics_error"
            if inner_truncated
            else ("time_limit" if time_limit_truncated else None)
        )

        # Handle bootstrap
        if terminated or truncated:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID
        discount = float(1 - terminated)

        return TimeStep(
            rgb_obs=obs["rgb_obs"],
            low_dim_obs=obs["low_dim_obs"],
            step_type=step_type,
            reward=reward,
            discount=discount,
            demo=0.0,
            is_expert=0.0,
            event_progress=progress,
        )

    def get_episode_seed_state(self):
        """Return the inner env's isolated episode-seed chain state, if any."""
        return self.inner_env.get_episode_seed_state()

    # ------------------------------------------------------------------
    # Episode outcome record: success and fall latches, truncation reason
    # ------------------------------------------------------------------

    def _update_fall_latch(self) -> None:
        controller = self.lowerbody.controller
        if controller is not None and not self._episode_fell:
            self._episode_fell = bool(controller.is_failed())

    def episode_fell(self) -> bool:
        """True once the lower-body controller reported a fall this episode."""
        return bool(self._episode_fell)

    def episode_succeeded(self) -> bool:
        """True once the task reported success this episode.

        The task's success is its held success (``success_hold_seconds``),
        the condition that ends the episode with the task reward. A step the
        env ends for a physics error does not count.
        """
        return bool(self._episode_succeeded)

    def last_truncation_reason(self) -> Optional[str]:
        """``"physics_error"`` / ``"time_limit"`` for the last step, else None."""
        return self._last_truncation

    def episode_termination(self) -> str:
        """One label per finished episode for eval records.

        ``fell`` (controller fall latch, whatever the task reported),
        ``success`` (:func:`bigym.loco.eval.protocol.is_success`),
        ``physics_error`` (MuJoCo divergence), ``timeout`` (episode budget),
        ``terminated`` (task-side terminal without success).
        """
        if self._episode_fell:
            return "fell"
        if self._episode_succeeded:
            return "success"
        if self._last_truncation == "physics_error":
            return "physics_error"
        if self._last_truncation == "time_limit":
            return "timeout"
        return "terminated"

    def set_episode_seed_state(self, state) -> None:
        """Restore an episode-seed chain state on the inner env."""
        self.inner_env.set_episode_seed_state(state)

    def reset(self, **kwargs):
        # Clear deques used for frame stacking
        """Reset the episode and return its first TimeStep.

        Clears the frame stacks, resets the inner env and the lower-body
        controller, applies the init pose/height overrides, and then runs
        the settle warmup (or leaves it pending in deferred mode).
        """
        self._clear_frame_stacks()

        lowerbody = self.lowerbody
        lowerbody.pending_reset_warmup = 0
        lowerbody.deferred_warmup_hold_action = None
        self._episode_fell = False
        self._episode_succeeded = False
        self._last_truncation = None
        if self._compiled_model_sha256 is None:
            self._compiled_model_sha256 = fingerprint.compiled_model_sha256(
                self.inner_env.model
            )
        bigym_obs, info = self.inner_env.reset(**kwargs)
        if lowerbody.controller is not None:
            lowerbody.reset(self.inner_env)
            warmup_steps = lowerbody.reset_warmup_steps
            if lowerbody.defer_reset_warmup and warmup_steps and int(warmup_steps) > 0:
                # Leave the settle warmup pending; the TimeStep returned
                # below is a pre-settle placeholder the caller replaces with
                # the one run_reset_warmup_steps() returns on completion.
                lowerbody.pending_reset_warmup = int(warmup_steps)
                lowerbody.deferred_warmup_hold_action = self.raw_hold_action()
            else:
                assert warmup_steps is not None
                lowerbody.run_reset_warmup(self, warmup_steps)
                self._update_fall_latch()
            bigym_obs = self.inner_env.get_observation()
        self.reset_action_representation_state()
        return self._build_reset_timestep(bigym_obs)

    def _build_reset_timestep(self, bigym_obs) -> TimeStep:
        obs = self._extract_obs(bigym_obs)
        self._step_counter = 0
        self._event_progress.reset(self.inner_env)
        progress = self._event_progress.update(self.inner_env, False)

        return TimeStep(
            rgb_obs=obs["rgb_obs"],
            low_dim_obs=obs["low_dim_obs"],
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
            demo=0.0,
            is_expert=0.0,
            event_progress=progress,
        )

    def set_deferred_reset_warmup(self, enabled: bool) -> None:
        """Enable or disable the VR-collector deferred reset warmup.

        When enabled, reset() leaves the lower-body settle warmup pending
        instead of blocking on it; drain it with run_reset_warmup_steps().
        Training and eval never enable this.
        """
        self.lowerbody.defer_reset_warmup = bool(enabled)

    @property
    def pending_reset_warmup_steps(self) -> int:
        """Deferred warmup steps still to run (0 outside deferred resets)."""
        return int(self.lowerbody.pending_reset_warmup)

    def run_reset_warmup_steps(self, max_steps: int):
        """Drain up to max_steps of a deferred reset warmup.

        Returns the post-settle reset TimeStep when the drain completes on
        this call (the caller must replace its placeholder reset TimeStep
        with it); None while steps remain or no warmup is pending.
        """
        lowerbody = self.lowerbody
        if lowerbody.pending_reset_warmup <= 0:
            return None
        n = min(int(max_steps), lowerbody.pending_reset_warmup)
        for _ in range(n):
            lowerbody.warmup_step(self, lowerbody.deferred_warmup_hold_action)
        lowerbody.pending_reset_warmup -= n
        if lowerbody.pending_reset_warmup > 0:
            return None
        lowerbody.deferred_warmup_hold_action = None
        # Same bookkeeping tail as the blocking warmup path.
        self.inner_env.reset_success_hold()
        bigym_obs = self.inner_env.get_observation()
        self.reset_action_representation_state()
        # Frame-stack deques must hold only post-settle frames; drop the
        # placeholder reset()'s pre-settle entries.
        self._clear_frame_stacks()
        return self._build_reset_timestep(bigym_obs)

    def render(self) -> Union[None, np.ndarray]:
        """Render the inner env in the configured render mode."""
        return self.inner_env.render()

    # ------------------------------------------------------------------
    # Demonstrations (Hugging Face Hub, LeRobot v3 lossless exports)
    # ------------------------------------------------------------------

    def load_demo_episodes(self, num_demos: int = -1) -> list[dict[str, np.ndarray]]:
        """Return this task's recorded episodes in replay format (fetched on demand).

        The task's folder of the demonstration dataset is downloaded on
        first use (``bigym.loco.demos.hub``; ``bigym-download`` pre-fetches),
        checked against this env's embodiment, backend, control rate, camera
        set/resolution and observation/action layout, and decoded
        (``bigym.loco.demos.dataset``). Row ``t`` of an episode holds the
        observation at step ``t`` together with the action, reward and
        discount of the transition that produced it; row 0 is the reset row.

        Side effects: the env adopts the dataset's ``action_stats`` (the
        collector's outer-action envelope), so ``[-1, 1]`` actions learned
        from these demos de-normalize exactly as they were recorded; with
        ``normalize_low_dim_obs`` on, it also takes its low-dim mean/std from
        these episodes. ``num_demos < 0`` loads every episode.
        """
        episodes, self._action_stats = demo_episodes.load_task_episodes(
            self.task_name,
            num_demos,
            action_representation=self.config.action_representation,
            upper_delta_scale_rad=self.config.upper_delta_scale_rad,
            outer_action_low=self._outer_action_low,
            outer_action_high=self._outer_action_high,
            check_compatibility=self._check_demo_compatibility,
        )
        if self.config.normalize_low_dim_obs:
            self._low_dim_obs_stats = self.extract_low_dim_obs_stats(episodes)
        return episodes

    def _check_demo_compatibility(
        self, metadata: dict[str, Any], info: dict[str, Any]
    ) -> None:
        demo_episodes.check_compatibility(
            metadata,
            info,
            task_name=self.task_name,
            robot_model=self.config.robot_model,
            lowerbody_backend=self.lowerbody.backend,
            demo_down_sample_rate=self.config.demo_down_sample_rate,
            camera_keys=self.config.camera_keys,
            camera_shape=self.config.camera_shape,
            low_dim_dim=int(self.low_dim_raw_observation_spec().shape[0]),
            action_dim=int(self.action_space.shape[0]),
        )

    def get_demos(self, num_demos: int = -1, only_successful: bool = False):
        """Return demonstrations as lists of :class:`ExtendedTimeStep`, one per episode.

        Observations are frame-stacked and (if enabled) low-dim-normalized
        exactly like the env's own ``reset``/``step`` output, so a demo
        timestep and a live timestep are interchangeable in a replay buffer.
        When low-dim normalization is on, its mean/std are set from these
        demos first. ``only_successful`` drops episodes without a success
        reward. ``num_demos < 0`` loads every episode.
        """
        episodes = self.load_demo_episodes(num_demos)
        demos = []
        num_successful = 0
        for episode in episodes:
            timesteps, successful = self._episode_to_timesteps(episode)
            num_successful += int(successful)
            if successful or not only_successful:
                demos.append(timesteps)
        print(f"Number of successful demos: {num_successful}/{len(episodes)}")
        if only_successful:
            print(f"Using successful demos only: {len(demos)}/{len(episodes)}")
        if not demos:
            raise RuntimeError("No demonstrations available after filtering")
        return demos

    @staticmethod
    def demo_is_successful(episode: dict[str, np.ndarray]) -> bool:
        """A recorded episode counts as successful when it earned reward."""
        return bool(np.asarray(episode["reward"], dtype=np.float32).sum() > 0.0)

    def extract_low_dim_obs_stats(self, episodes: list[dict[str, np.ndarray]]):
        """Return per-dimension mean/std of the demos' raw low-dim observations."""
        return demo_episodes.low_dim_obs_stats(episodes)

    def demo_observations(
        self, episode: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stack (and normalize) one episode's frames the way the env observes them.

        Returns ``(rgb, low_dim)`` with shapes ``[T, cams, 3 * frame_stack, H, W]``
        and ``[T, D * frame_stack]``. The env's own frame-stack deques are
        reset first, as at ``reset()``.
        """
        rgb_frames = np.asarray(episode["rgb_obs"], dtype=np.uint8)
        low_dims = np.asarray(episode["low_dim_obs"], dtype=np.float32)
        self._clear_frame_stacks()
        rgb_out, low_out = [], []
        for t in range(len(low_dims)):
            obs = self._stack_observation(rgb_frames[t], low_dims[t])
            rgb_out.append(obs["rgb_obs"])
            low_out.append(obs["low_dim_obs"])
        return np.stack(rgb_out), np.stack(low_out)

    def _stack_observation(
        self, rgb_frames, low_dim_obs: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Normalize and frame-stack one unstacked observation.

        Args:
            rgb_frames: One ``[3, H, W]`` frame per camera, in ``camera_keys``
                order.
            low_dim_obs: The unstacked low-dim vector.
        """
        out: Dict[str, np.ndarray] = {}
        low_dim_obs = np.asarray(low_dim_obs, dtype=np.float32)
        if self.config.normalize_low_dim_obs:
            mean, std = self._low_dim_obs_stats["mean"], self._low_dim_obs_stats["std"]
            low_dim_obs = (low_dim_obs - mean) / (std + 1e-8)
        if len(self._low_dim_obses) == 0:
            for _ in range(self.config.frame_stack):
                self._low_dim_obses.append(low_dim_obs)
        else:
            self._low_dim_obses.append(low_dim_obs)
        out["low_dim_obs"] = np.concatenate(list(self._low_dim_obses), axis=0)
        if not self.config.camera_keys:
            out["rgb_obs"] = np.zeros(
                (0, 3 * self.config.frame_stack, *self.config.camera_shape),
                dtype=np.uint8,
            )
        else:
            for index, camera_key in enumerate(self.config.camera_keys):
                pixels = np.asarray(rgb_frames[index], dtype=np.uint8).copy()
                if len(self._frames[camera_key]) == 0:
                    for _ in range(self.config.frame_stack):
                        self._frames[camera_key].append(pixels)
                else:
                    self._frames[camera_key].append(pixels)
            out["rgb_obs"] = np.stack(
                [
                    np.concatenate(list(self._frames[camera_key]), axis=0)
                    for camera_key in self.config.camera_keys
                ],
                0,
            )
        return out

    def _clear_frame_stacks(self) -> None:
        self._low_dim_obses.clear()
        for frames in self._frames.values():
            frames.clear()

    def _episode_to_timesteps(
        self, episode: dict[str, np.ndarray]
    ) -> tuple[list[ExtendedTimeStep], bool]:
        successful = self.demo_is_successful(episode)
        rgb, low_dim = self.demo_observations(episode)
        return (
            demo_episodes.episode_to_timesteps(episode, rgb, low_dim, successful),
            successful,
        )

    def close(self) -> None:
        """Close the inner BiGym environment."""
        self.inner_env.close()

    def build_inner_env(self) -> None:
        """Build the task env and the lower-body controller, then the action layout."""
        bigym_class = task_spec(self.task_name).env_cls
        robot_cls = self.lowerbody.robot_cls(ROBOT_MODELS[self.config.robot_model])
        camera_configs = [
            CameraConfig(
                name=camera_name,
                rgb=True,
                depth=False,
                resolution=self.config.camera_shape,
            )
            for camera_name in self.config.camera_keys
        ]

        use_all_floating_dofs = bool(
            self.config.enable_all_floating_dof or self.lowerbody.enabled
        )

        if use_all_floating_dofs:
            action_mode = JointPositionActionMode(
                absolute=self.config.action_mode == "absolute",
                floating_base=True,
                floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
            )
        else:
            action_mode = JointPositionActionMode(
                absolute=self.config.action_mode == "absolute",
                floating_base=True,
            )

        self.inner_env = bigym_class(
            render_mode=self.config.render_mode,
            action_mode=action_mode,
            observation_config=ObservationConfig(
                cameras=camera_configs,
                proprioception=True,
                privileged_information=False,
            ),
            control_frequency=CONTROL_FREQUENCY_MAX
            // self.config.demo_down_sample_rate,
            success_hold_seconds=self.config.success_hold_seconds,
            reach_tolerance=self.config.reach_tolerance,
            robot_cls=robot_cls,
            episode_seed_rng=self._episode_seed_rng,
            initialization_profile=self.config.initialization_profile,
        )
        self.lowerbody.build_controller(
            self.inner_env,
            control_dt=float(self.config.demo_down_sample_rate)
            / float(CONTROL_FREQUENCY_MAX),
        )

        # Episode length counter
        self._step_counter = 0
        # Must run after controller construction (needs command_spec) and
        # before the outer action bounds are computed below.
        self.lowerbody.resolve_command_bounds()
        self.lowerbody.resolve_reset_warmup()
        self._setup_outer_action_and_observation_adapters()

    def raw_hold_action(self) -> np.ndarray:
        """The raw outer action that holds the current pose.

        Zero base velocity, the default height command, the arms' current
        joint targets and the grippers' current positions.
        """
        action_dim = (
            int(self._outer_action_dim)
            if self._outer_action_dim is not None
            else int(self.inner_env.action_space.shape[0])
        )
        raw = np.zeros((action_dim,), dtype=np.float32)
        outer_base_dofs = int(len(self.outer_action_floating_dofs))
        lowerbody = self.lowerbody

        if (
            lowerbody.config.use_height_cmd
            and lowerbody.uses_command_base_actions(self)
            and outer_base_dofs > 0
        ):
            dof_to_idx = {
                dof: i for i, dof in enumerate(self.outer_action_floating_dofs)
            }
            z_idx = dof_to_idx.get(PelvisDof.Z)
            if z_idx is not None and z_idx < action_dim:
                raw[z_idx] = float(
                    np.clip(
                        lowerbody.config.default_height_cmd,
                        lowerbody.height_cmd_min,
                        lowerbody.height_cmd_max,
                    )
                )

        robot = self.inner_env.robot
        if self.config.action_mode == "absolute":
            for i, name in enumerate(self.outer_limb_actuator_names):
                actuator = robot.limb_actuators[self.limb_name_to_full_index[name]]
                ctrl = self.inner_env.data.bind(actuator).ctrl
                raw[outer_base_dofs + i] = float(np.asarray(ctrl).squeeze())

        if robot.grippers:
            raw[-len(robot.grippers) :] = np.asarray(
                [
                    float(np.asarray(gripper.qpos).item())
                    for gripper in robot.grippers.values()
                ],
                dtype=np.float32,
            )
        return raw

    def _outer_upper_action_slice(self) -> slice:
        base_dim = int(len(self.outer_action_floating_dofs))
        upper_dim = len(self.outer_limb_actuator_names)
        return slice(base_dim, base_dim + upper_dim)

    def _initialize_action_representation(self) -> None:
        if self.config.action_representation == ABSOLUTE:
            return
        upper = self._outer_upper_action_slice()
        if upper.stop <= upper.start:
            raise ValueError(
                "action_representation='upper_delta' requires at least one "
                "position-controlled outer limb"
            )
        self.upper_delta_accumulator = UpperDeltaAccumulator(
            self.config.upper_delta_scale_rad
        )

    def reset_action_representation_state(self) -> None:
        """Restart the upper-delta targets from the hold action (absolute: no-op)."""
        if self.upper_delta_accumulator is None:
            return
        upper = self._outer_upper_action_slice()
        hold = self.raw_hold_action()
        self.upper_delta_accumulator.reset(hold[upper])

    def _apply_action_representation(
        self, normalized_action: np.ndarray, raw_action: np.ndarray
    ) -> np.ndarray:
        if self.upper_delta_accumulator is None:
            return raw_action
        upper = self._outer_upper_action_slice()
        converted = np.asarray(raw_action).copy()
        assert self._outer_action_low is not None
        assert self._outer_action_high is not None
        converted[upper] = self.upper_delta_accumulator.apply(
            np.asarray(normalized_action)[upper],
            self._outer_action_low[upper],
            self._outer_action_high[upper],
        )
        return converted.astype(raw_action.dtype, copy=False)

    def leg_joint_names(self) -> tuple[str, ...]:
        """The joints the lower-body controller drives (none for a floating base)."""
        if self.lowerbody.controller is not None:
            return tuple(self.lowerbody.controller.controlled_joints)
        return ()

    def floating_base_yaw(self) -> float:
        """The robot base's heading in rad (0 without a floating base)."""
        floating_base = self.inner_env.robot.floating_base
        if floating_base is None:
            return 0.0
        return yaw_from_floating_qpos(np.asarray(floating_base.qpos))

    def get_lowerbody_state(self) -> dict[str, np.ndarray]:
        """Snapshot the mutable lower-body control state beyond qpos/qvel.

        Restoring MuJoCo qpos/qvel alone does not reproduce a mid-episode
        moment: the controller keeps observation histories, last actions and
        rate-limiter anchors, and the env keeps the previous base command
        and the upper-delta target. VR demo collection snapshots this
        at the engage moment so replay can restore the full control state.
        Keys are namespaced ("ctrl." for controller fields, "env." for
        env-side fields) so they can be stored flat in a demo npz.

        Deliberately NOT covered: the train-time reward-shaping progress
        state (:class:`~bigym.loco.event_progress.EventProgress`). Restoring
        mid-episode therefore reproduces dynamics and success bit-exactly
        but may diverge in *shaped* reward bookkeeping — acceptable because
        shaping is a train-only aid, never part of the frozen success
        criterion.
        """
        return self.lowerbody.get_state(self)

    def set_lowerbody_state(self, state: dict[str, np.ndarray]) -> None:
        """Restore a snapshot produced by get_lowerbody_state()."""
        self.lowerbody.set_state(self, state)

    def _setup_outer_action_and_observation_adapters(self) -> None:
        base = self.inner_env.robot.floating_base
        self.inner_floating_dofs = (
            list(self.inner_env.action_mode.floating_dofs) if base is not None else []
        )
        outer_floating_dofs = (
            list(self._outer_floating_dofs) if base is not None else []
        )
        outer_action_floating_dofs = (
            list(self.outer_action_floating_dofs) if base is not None else []
        )
        outer_base_dof_amount = int(len(outer_action_floating_dofs))
        gripper_count = int(len(self.inner_env.robot.grippers))

        if base is not None:
            inner_name_to_index = {
                dof.value: i for i, dof in enumerate(self.inner_floating_dofs)
            }
            # -1 = command-only outer dof with no inner counterpart (RY):
            # consumers skip it when indexing inner vectors.
            self._outer_base_to_inner_base_idx = np.asarray(
                [inner_name_to_index.get(dof.value, -1) for dof in outer_floating_dofs],
                dtype=np.int64,
            )
            self.outer_action_base_to_inner_base_idx = np.asarray(
                [
                    inner_name_to_index.get(dof.value, -1)
                    for dof in outer_action_floating_dofs
                ],
                dtype=np.int64,
            )
        else:
            self._outer_base_to_inner_base_idx = None
            self.outer_action_base_to_inner_base_idx = None

        robot = self.inner_env.robot
        limb_names = [
            a.name.removeprefix(robot.namespace) for a in robot.limb_actuators
        ]
        self.limb_name_to_full_index = {n: i for i, n in enumerate(limb_names)}

        excluded_limb_names = set()
        excluded_joint_names = set()
        if self.lowerbody.controller is not None:
            controlled_joint_names = set(self.leg_joint_names())

            excluded_limb_names |= controlled_joint_names
            if not (
                self.config.wholebody is not None
                and self.config.wholebody.preserve_leg_proprio
            ):
                excluded_joint_names |= controlled_joint_names
            if self.lowerbody.pitch_cmd_enabled:
                # The outer policy commands the torso pitch, so it must observe
                # the waist it is bending: keep the backend-owned waist
                # joints in proprioception (+qpos/qvel per joint).
                excluded_joint_names -= {
                    n for n in controlled_joint_names if "waist" in n
                }

        if base is not None:
            inner_base_joint_names = {dof.value for dof in self.inner_floating_dofs}
            outer_base_joint_names = {dof.value for dof in outer_floating_dofs}
            excluded_joint_names |= inner_base_joint_names - outer_base_joint_names

        self.outer_limb_actuator_names = tuple(
            n for n in limb_names if n not in excluded_limb_names
        )

        self._outer_action_dim = int(
            outer_base_dof_amount + len(self.outer_limb_actuator_names) + gripper_count
        )
        self._outer_action_low = self._compute_outer_action_bounds(is_low=True)
        self._outer_action_high = self._compute_outer_action_bounds(is_low=False)

        joint_names = [j.name.removeprefix(robot.namespace) for j in robot.joints]
        self._proprio_full_qpos_dim = int(len(joint_names))

        if excluded_joint_names and joint_names:
            self._proprio_keep_qpos_idx = np.asarray(
                [i for i, n in enumerate(joint_names) if n not in excluded_joint_names],
                dtype=np.int64,
            )
        else:
            self._proprio_keep_qpos_idx = None

    def _compute_outer_action_bounds(self, *, is_low: bool) -> np.ndarray:
        base = self.inner_env.robot.floating_base
        inner_base_dof_amount = int(base.dof_amount) if base is not None else 0
        gripper_count = int(len(self.inner_env.robot.grippers))
        lowerbody = self.lowerbody

        bounds = []
        space_arr = (
            self.inner_env.action_space.low
            if is_low
            else self.inner_env.action_space.high
        )
        if self.outer_action_base_to_inner_base_idx is not None:
            for outer_i, inner_idx in enumerate(
                self.outer_action_base_to_inner_base_idx
            ):
                dof = self.outer_action_floating_dofs[outer_i]
                if lowerbody.uses_command_base_actions(self):
                    if dof in (PelvisDof.X, PelvisDof.Y):
                        bound = (
                            -lowerbody.config.cmd_clip
                            if is_low
                            else lowerbody.config.cmd_clip
                        )
                    elif dof == PelvisDof.RZ:
                        bound = (
                            -lowerbody.config.wz_clip
                            if is_low
                            else lowerbody.config.wz_clip
                        )
                    elif dof == PelvisDof.Z:
                        bound = (
                            lowerbody.height_cmd_min
                            if is_low
                            else lowerbody.height_cmd_max
                        )
                    elif dof == PelvisDof.RY:
                        bound = (
                            lowerbody.pitch_cmd_min
                            if is_low
                            else lowerbody.pitch_cmd_max
                        )
                    else:
                        bound = float(space_arr[int(inner_idx)])
                else:
                    bound = float(space_arr[int(inner_idx)])
                bounds.append(np.asarray([bound], dtype=np.float32))

        for name in self.outer_limb_actuator_names:
            full_idx = self.limb_name_to_full_index[name]
            bounds.append(np.asarray([space_arr[inner_base_dof_amount + full_idx]]))

        if gripper_count:
            bounds.append(space_arr[-gripper_count:])
        return np.concatenate(bounds).astype(np.float32, copy=False)

    def _initialize_frame_stack(self):
        # Create deques for frame stacking
        self._low_dim_obses = deque([], maxlen=self.config.frame_stack)
        self._frames = {
            camera_key: deque([], maxlen=self.config.frame_stack)
            for camera_key in self.config.camera_keys
        }

    def _construct_action_and_observation_spaces(self):
        # Setup action/observation spaces
        action_dim = (
            int(self._outer_action_dim)
            if self._outer_action_dim is not None
            else int(self.inner_env.action_space.shape[0])
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,))

        # Compute dimension of low_dim_obs
        low_dim = 0
        for state_key in self.config.state_keys:
            if (
                state_key == "proprioception"
                and self._proprio_keep_qpos_idx is not None
                and self._proprio_full_qpos_dim is not None
            ):
                low_dim += int(self._proprio_keep_qpos_idx.shape[0]) * 2
            elif (
                state_key
                in (
                    "proprioception_floating_base",
                    "proprioception_floating_base_actions",
                )
                and self._outer_base_to_inner_base_idx is not None
            ):
                low_dim += int((self._outer_base_to_inner_base_idx >= 0).sum())
            else:
                state_space = cast(
                    spaces.Box, self.inner_env.observation_space[state_key]
                )
                low_dim += state_space.shape[-1]
        self.low_dim_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(low_dim * self.config.frame_stack,),
            dtype=np.float32,
        )
        self.low_dim_raw_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(low_dim,), dtype=np.float32
        )  # without frame stacking
        # Identity normalization until get_demos() sets the demos' statistics.
        self._low_dim_obs_stats = {
            "mean": np.zeros((low_dim,), dtype=np.float32),
            "std": np.ones((low_dim,), dtype=np.float32),
        }
        self.rgb_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                len(self.config.camera_keys),
                3 * self.config.frame_stack,
                *self.config.camera_shape,
            ),
            dtype=np.uint8,
        )
        self.rgb_raw_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(len(self.config.camera_keys), 3, *self.config.camera_shape),
            dtype=np.uint8,
        )  # without frame stacking

        # Set default action stats, which will be overridden by demonstration
        # action stats
        # Required for a case we don't use demonstrations
        action_min = -np.ones(self.action_space.shape, dtype=self.action_space.dtype)
        action_max = np.ones(self.action_space.shape, dtype=self.action_space.dtype)
        gripper_count = int(len(self.inner_env.robot.grippers))
        if gripper_count:
            action_min[-gripper_count:] = 0
            action_max[-gripper_count:] = 1
        self._action_stats = {"min": action_min, "max": action_max}

    def _extract_low_dim_from_obs(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        for key in self.config.state_keys:
            value = obs[key]
            if key == "proprioception":
                value = self._maybe_crop_proprioception(value)
            elif key in (
                "proprioception_floating_base",
                "proprioception_floating_base_actions",
            ):
                value = self._maybe_crop_floating_base(value)
            parts.append(value)
        return np.hstack(parts, dtype=np.float32)

    @property
    def action_stats(self) -> dict[str, np.ndarray]:
        """``{"min", "max"}``: the raw values the normalized ``-1`` and ``1`` map to."""
        return self._action_stats

    def set_action_stats(self, action_min: np.ndarray, action_max: np.ndarray) -> None:
        """Set the raw envelope that normalized actions map onto."""
        self._action_stats = {"min": action_min, "max": action_max}

    def denormalize_action(self, action):
        """Convert a [-1, 1] action to raw joint space using action stats.

        Under ``upper_delta`` the arm slots are integrated onto the current
        targets, so this advances the accumulator.
        """
        action = np.asarray(action)
        assert (max(action) <= 1) and (min(action) >= -1)
        action_min, action_max = self._action_stats["min"], self._action_stats["max"]
        new_action = (action + 1) / 2.0  # to [0, 1]
        new_action = new_action * (action_max - action_min + 1e-8) + action_min
        new_action = new_action.astype(action.dtype, copy=False)
        return self._apply_action_representation(action, new_action)

    def normalize_action(self, action):
        """Convert a raw joint-space action to [-1, 1] using action stats."""
        action_min, action_max = self._action_stats["min"], self._action_stats["max"]
        new_action = (action - action_min) / (action_max - action_min + 1e-8)
        new_action = new_action * 2 - 1  # to [-1, 1]
        return new_action.astype(action.dtype, copy=False)

    def _extract_obs(self, obs) -> Dict[str, np.ndarray]:
        return self._stack_observation(
            [obs[f"rgb_{camera_key}"] for camera_key in self.config.camera_keys],
            self._extract_low_dim_from_obs(obs),
        )

    def _maybe_crop_proprioception(self, proprio: np.ndarray) -> np.ndarray:
        if self._proprio_keep_qpos_idx is None or self._proprio_full_qpos_dim is None:
            return proprio
        if proprio.ndim != 1:
            return proprio
        if proprio.shape[0] % 2 != 0:
            return proprio
        qpos_dim = int(proprio.shape[0] // 2)
        if qpos_dim != int(self._proprio_full_qpos_dim):
            return proprio

        qpos = proprio[:qpos_dim]
        qvel = proprio[qpos_dim:]
        keep = self._proprio_keep_qpos_idx
        qpos = qpos[keep]
        qvel = qvel[keep]
        return np.concatenate([qpos, qvel]).astype(np.float32, copy=False)

    def _maybe_crop_floating_base(self, value: np.ndarray) -> np.ndarray:
        if self._outer_base_to_inner_base_idx is None:
            return value
        # Command-only dofs (RY) have no inner floating-base entry (-1).
        idx = self._outer_base_to_inner_base_idx
        idx = idx[idx >= 0]
        if value.ndim != 1:
            return value
        if value.shape[0] < int(idx.max()) + 1:
            return value
        return value[idx].astype(np.float32, copy=False)

    def get_last_lowerbody_control_info(self) -> dict[str, Any]:
        """Return a copy of the diagnostics recorded during the last step."""
        out: dict[str, Any] = {}
        for key, value in self.lowerbody.last_control_info.items():
            if isinstance(value, np.ndarray):
                out[key] = value.copy()
            else:
                out[key] = value
        return out

    @property
    def config_overrides(self) -> dict[str, Any]:
        """``{dotted field: value}`` wherever ``config`` departs from ``official_config``."""
        return {
            name: value
            for name, (_, value) in self.config.differences(
                self.official_config
            ).items()
        }

    @property
    def upper_delta_scale_rad(self) -> Optional[float]:
        """Radians per unit of normalized upper delta (None for absolute actions)."""
        if self.upper_delta_accumulator is None:
            return None
        return self.upper_delta_accumulator.scale_rad

    @property
    def control_step_seconds(self) -> float:
        """True sim seconds per outer control step (from the compiled model)."""
        return float(self.inner_env.control_step_seconds)

    @property
    def compiled_model_sha256(self) -> str:
        """SHA-256 of the compiled model as built, before a reset randomized it."""
        if self._compiled_model_sha256 is None:
            return fingerprint.compiled_model_sha256(self.inner_env.model)
        return self._compiled_model_sha256

    def substrate_fingerprint(self) -> dict[str, Any]:
        """Identity of the frozen substrate this env instance realizes.

        Stamp this into run/eval records so results can be grouped by
        comparable substrate (see SUBSTRATE_VERSION).
        """
        return fingerprint.substrate_fingerprint(self)

    def describe_model(self) -> dict[str, Any]:
        """What is actually simulated: compiled options, gains, armature, backend.

        The numbers here are read from the COMPILED MjModel (not from config
        tables), so they are what the physics ran with. Complements
        ``substrate_fingerprint()`` (identity) with the human-readable values
        a paper's appendix needs.
        """
        return fingerprint.describe_model(self)

    def wholebody_action_layout(self) -> dict[str, Any]:
        """Return the outer action layout used by whole-body control."""
        gripper_count = int(len(self.inner_env.robot.grippers))
        assert self._outer_action_dim is not None
        return {
            "action_dim": int(self._outer_action_dim),
            "base_dim": int(len(self.outer_action_floating_dofs)),
            "limb_names": self.outer_limb_actuator_names,
            "gripper_count": gripper_count,
            "leg_joint_names": tuple(self.leg_joint_names()),
            "action_low": np.asarray(self._outer_action_low, dtype=np.float32).copy(),
            "action_high": np.asarray(self._outer_action_high, dtype=np.float32).copy(),
            "action_representation": str(self.config.action_representation),
            "upper_delta_scale_rad": self.upper_delta_scale_rad,
        }

    def __del__(
        self,
    ) -> None:
        # Guard against partially-constructed instances (__init__ raised
        # before self.inner_env existed).
        """Close the environment when the instance is garbage collected."""
        if getattr(self, "inner_env", None) is not None:
            self.close()


def make(
    task_name: str,
    config: Optional[EnvConfig] = None,
    *,
    episode_seed_rng: Optional[np.random.Generator] = None,
    **overrides: Any,
) -> ExtendedTimeStepWrapper:
    """Create a BiGym environment; with no arguments, the official one.

    Example::

        from bigym.loco import make

        env = make("move_plate")                      # official configuration
        env = make("move_plate", camera_keys=("head",))
        env = make("move_plate", controller=None)     # floating base, no legs
        ts = env.reset(seed=620000)
        ts = env.step(env.action_space.sample())
        env.close()

    Args:
        task_name: A registered task (``bigym.loco.tasks.TASKS``) or a
            ``"pkg.module:ATTR"`` reference to a ``TaskSpec``.
        config: None for the task's official configuration, or an
            :class:`~bigym.loco.config.EnvConfig` to build from as is.
        episode_seed_rng: Generator that draws per-episode seeds when
            ``reset`` gets no explicit seed.
        **overrides: EnvConfig fields applied last; an unknown name raises
            ValueError. ``controller`` takes None, a ControllerConfig or a
            dict of its fields merged into the current one.

    Returns:
        The env behind the ExtendedTimeStep interface. ``env.config`` is
        the resolved configuration and ``env.config_overrides`` the fields
        that differ from the official one; ``env.substrate_fingerprint()``
        stamps the physics, controller and layout.
    """
    resolved = resolve_config(task_name, config, overrides)
    return ExtendedTimeStepWrapper(
        BiGym(task_name, resolved, episode_seed_rng=episode_seed_rng)
    )
