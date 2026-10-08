"""Core BiGym env functionality."""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Optional, Type, Union

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from bigym import scene
from bigym.action_modes import ActionMode
from bigym.bigym_renderer import BiGymRenderer
from bigym.const import WORLD_MODEL
from bigym.envs.props.preset import Preset
from bigym.robots.configs.g1 import G1Dex1
from bigym.robots.robot import Robot
from bigym.simulation import Simulation
from bigym.utils.callables_cache import CallablesCache
from bigym.utils.env_health import EnvHealth
from bigym.utils.observation_config import ObservationConfig

CONTROL_FREQUENCY_MAX = 500
CONTROL_FREQUENCY_MIN = 20

PHYSICS_DT = 0.002


def control_sub_steps(control_frequency: float, compiled_timestep: float) -> int:
    """Physics substeps per control step at the COMPILED model timestep.

    Attached robot XMLs can override the scene's ``<option timestep>`` (the G1
    ships 0.001 s against the nominal ``PHYSICS_DT``), so the substep count,
    and with it the floating-base delta bounds ``ActionMode.action_space``
    scales by, must be derived from the compiled timestep. Everything that
    rebuilds an action space outside a live env (demo decimation, clipping,
    absolute-to-delta conversion) goes through this same function.
    """
    return int(np.round(1.0 / (float(control_frequency) * float(compiled_timestep))))


MAX_DISTANCE_FROM_ORIGIN = 10
SPARSE_REWARD_FACTOR = 1


class BiGymEnv(gym.Env):
    """Core BiGym environment which loads in common robot across all tasks."""

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 1 / PHYSICS_DT,
    }

    _ENV_CAMERAS = ["external"]

    _MODEL_PATH: Path = WORLD_MODEL
    _PRESET_PATH: Optional[Path] = None
    _FLOOR = "floor"

    DEFAULT_ROBOT = G1Dex1

    action_space: spaces.Box
    observation_space: spaces.Dict

    RESET_ROBOT_POS = np.array([0, 0, 0])
    RESET_ROBOT_QUAT = np.array([1, 0, 0, 0])

    def __init__(
        self,
        action_mode: ActionMode,
        observation_config: ObservationConfig | None = None,
        render_mode: Optional[str] = None,
        start_seed: Optional[int] = None,
        control_frequency: int = CONTROL_FREQUENCY_MAX,
        robot_cls: Optional[Type[Robot]] = None,
        success_hold_seconds: float = 0.0,
        reach_tolerance: Optional[float] = None,
        episode_seed_rng: Optional[np.random.Generator] = None,
        initialization_profile: str = "upstream",
    ):
        """Init.

        :param action_mode: The action mode of the robot. Use this to configure how
            you plan to control the robot. E.g. joint position, delta ee pose, ect.
        :param observation_config: Observations configuration. Use this to configure
            collected data.
        :param render_mode: The render mode for mujoco. Options are
            "human", "rgb_array" or "depth_array". If None, the default render mode
            will be used.
        :param start_seed: The seed to start the environment with. If None, a random
            seed will be used.
        :param control_frequency: Control loop frequency, 500 Hz by default.
        :param robot_cls: Environment robot class override.
        :param success_hold_seconds: If > 0, the task success predicate must stay
            true for this many consecutive seconds (converted to control steps)
            before ``success`` is reported — filters transient "swing-through"
            goal states. 0 (default) checks the predicate instantaneously.
        :param reach_tolerance: Optional override for the reach-family success
            radius in meters (distance from the gripper's end-effector site to
            the target center). None (default) keeps each task's class
            constant (0.1 for the stock reach tasks). Ignored by tasks
            that do not use a reach tolerance.
        :param episode_seed_rng: Optional per-environment RNG for the episode seed
            chain. None uses the global numpy RNG.
        :param initialization_profile: Versioned task reset distribution. The
            default ``"upstream"`` is the original BiGym distribution; task
            subclasses may opt into additional profiles.
        """
        # Tracks physics simulation stability
        self._env_health = EnvHealth()
        # Caches results valid for one environment step
        self._step_cache = CallablesCache()

        self._observation_config = observation_config or ObservationConfig()
        self.action_mode = action_mode

        self._episode_seed_rng = episode_seed_rng
        self._initialization_profile = str(initialization_profile)
        if start_seed is None:
            if self._episode_seed_rng is None:
                start_seed = np.random.randint(2**32)
            else:
                start_seed = int(self._episode_seed_rng.integers(2**32))
        if not isinstance(start_seed, int):
            raise ValueError("Expected start_seed to be an integer.")
        self._next_seed = start_seed
        self._current_seed = None

        assert CONTROL_FREQUENCY_MIN <= control_frequency <= CONTROL_FREQUENCY_MAX, (
            f"Control frequency must be in "
            f"{CONTROL_FREQUENCY_MIN}-{CONTROL_FREQUENCY_MAX} range."
        )
        self._control_frequency = control_frequency
        self._sub_steps_count = int(
            np.round(CONTROL_FREQUENCY_MAX / self._control_frequency)
        )

        self._success_hold_seconds = float(success_hold_seconds)
        if self._success_hold_seconds < 0.0:
            raise ValueError(
                f"success_hold_seconds must be >= 0, got {success_hold_seconds}"
            )
        self._success_hold_counter = 0

        self._reach_tolerance = (
            None if reach_tolerance is None else float(reach_tolerance)
        )
        if self._reach_tolerance is not None and self._reach_tolerance <= 0.0:
            raise ValueError(f"reach_tolerance must be > 0, got {reach_tolerance}")

        self.simulation = Simulation(scene.load(self._MODEL_PATH))
        self.simulation.spec.option.timestep = PHYSICS_DT
        self._robot = (robot_cls or self.DEFAULT_ROBOT)(
            self.action_mode, self.simulation
        )
        self._preset = Preset(self.simulation, self._PRESET_PATH)
        bodies = len(self.simulation.spec.bodies)
        self._initialize_env()
        # The published scenes were compiled before the last preset prop was
        # placed, unless the task added bodies after the preset.
        placed_after_compile = len(self.simulation.spec.bodies) == bodies
        if not placed_after_compile:
            self._preset.place_last_prop(compiled=False)
        self.simulation.compile()
        if placed_after_compile:
            self._preset.place_last_prop(compiled=True)
        self._floor = self.simulation.spec.geom(self._FLOOR)

        # Attached robot XMLs can override <option> attributes of the scene
        # (the G1 XML ships timestep=0.001). Derive the substep count from the
        # COMPILED timestep so control_frequency means true Hz, which the
        # 50 Hz-trained lower-body policy depends on.
        compiled_timestep = float(self.simulation.model.opt.timestep)
        true_sub_steps = control_sub_steps(self._control_frequency, compiled_timestep)
        if true_sub_steps != self._sub_steps_count:
            print(
                f"[bigym] compiled timestep {compiled_timestep:g}s != nominal "
                f"{PHYSICS_DT:g}s: control substeps {self._sub_steps_count} -> "
                f"{true_sub_steps} to keep {self._control_frequency} Hz true"
            )
            self._sub_steps_count = true_sub_steps
        self._control_step_seconds = compiled_timestep * float(self._sub_steps_count)
        self._success_hold_steps = int(
            np.round(self._success_hold_seconds / self._control_step_seconds)
        )
        if self._success_hold_seconds > 0.0 and self._success_hold_steps < 1:
            raise ValueError(
                f"success_hold_seconds={success_hold_seconds} rounds to 0 control "
                f"steps at {1.0 / self._control_step_seconds:.1f} Hz"
            )

        self.action_space: spaces.Box = self.action_mode.action_space(
            action_scale=self._sub_steps_count, seed=self._next_seed
        )
        self._action: np.ndarray = np.zeros_like(self.action_space.low)

        self.observation_space = self.get_observation_space()

        assert self.metadata["render_modes"] == [
            "human",
            "rgb_array",
            "depth_array",
        ], self.metadata["render_modes"]

        self.render_mode = render_mode

        # Validate cameras configuration
        available_cameras = set(self._ENV_CAMERAS + self._robot.config.cameras)
        for camera_config in self._observation_config.cameras:
            assert camera_config.name in available_cameras

        # Mapping original camera names to full identifiers
        self._cameras_map = self._initialize_cameras()

        self.mujoco_renderer: Optional[BiGymRenderer] = None
        self.obs_renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        # None = not decided yet, False = strip path not applicable, else renderer
        self._strip_renderer_state: Any = None
        self._strip_buffer: Optional[np.ndarray] = None
        self._initialize_renderers()

    @property
    def task_name(self) -> str:
        """Returns the class name of the environment."""
        return self.__class__.__name__

    @property
    def _use_pixels(self):
        """Returns True if the environment uses pixels."""
        return len(self.observation_config.cameras) > 0

    @property
    def seed(self) -> Optional[int]:
        """Initial seed of the environment."""
        return self._current_seed

    @property
    def initialization_profile(self) -> str:
        """Versioned task reset distribution used by this environment."""
        return self._initialization_profile

    @property
    def success(self) -> bool:
        """Check if current step is successful.

        With ``success_hold_seconds > 0`` this reports the HELD success (the
        instantaneous predicate has stayed true for the required consecutive
        control steps); otherwise the instantaneous predicate.
        """
        if self._success_hold_steps > 0:
            return self._success_hold_counter >= self._success_hold_steps
        return bool(self._step_cache.get(self._success))

    @property
    def success_hold_seconds(self) -> float:
        """Configured success-hold duration in seconds (0 = instantaneous)."""
        return self._success_hold_seconds

    @property
    def success_hold_steps(self) -> int:
        """Success-hold duration in control steps (0 = instantaneous)."""
        return self._success_hold_steps

    @property
    def control_step_seconds(self) -> float:
        """True sim seconds per control step (compiled timestep x substeps).

        Can differ from 1/control_frequency when an attached robot XML
        overrides the scene timestep (the G1 model ships 0.001).
        """
        return self._control_step_seconds

    def reset_success_hold(self) -> None:
        """Zero the success-hold counter.

        Wrappers that step the env during reset (lower-body warmup) call this
        afterwards so warmup steps can never pre-fill the hold window (e.g. a
        reach target randomized within tolerance of the resting hand).
        """
        self._success_hold_counter = 0

    @property
    def fail(self) -> bool:
        """Check if current step is successful."""
        return bool(self._step_cache.get(self._fail))

    @property
    def reward(self) -> float:
        """Get current step reward."""
        return float(self._step_cache.get(self._reward))

    @property
    def terminate(self) -> bool:
        """Get current step termination condition."""
        return bool(self.success or self.fail)

    @property
    def truncate(self) -> bool:
        """Get current step truncation condition."""
        return bool(not self.is_healthy)

    @property
    def is_healthy(self) -> bool:
        """Checks if the simulation is currently healthy."""
        return bool(self._env_health.is_healthy)

    @property
    def observation_config(self) -> ObservationConfig:
        """Get the observation configuration."""
        return self._observation_config

    @property
    def control_frequency(self):
        """Control frequency of the environment."""
        return self._control_frequency

    @property
    def robot(self) -> Robot:
        """Get robot."""
        return self._robot

    @property
    def floor(self) -> mujoco.MjsGeom:
        """Get environment floor."""
        return self._floor

    @property
    def spec(self) -> mujoco.MjSpec:
        """The scene spec the model was compiled from."""
        return self.simulation.spec

    @property
    def model(self) -> mujoco.MjModel:
        """The compiled scene model."""
        return self.simulation.model

    @property
    def data(self) -> mujoco.MjData:
        """The simulation data."""
        return self.simulation.data

    @property
    def action(self) -> np.ndarray:
        """Get last executed action."""
        return self._action.copy()

    def _initialize_renderers(self):
        self._close_renderers()
        self.mujoco_renderer: BiGymRenderer = BiGymRenderer(self.model, self.data)
        for camera_config in self._observation_config.cameras:
            resolution = camera_config.resolution
            if resolution in self.obs_renderers:
                continue
            self.obs_renderers[resolution] = mujoco.Renderer(
                self.model, resolution[0], resolution[1]
            )

    def _initialize_cameras(self) -> dict[str, tuple[int, mujoco.MjsCamera]]:
        cameras = {name: self.spec.camera(name) for name in self._ENV_CAMERAS}
        cameras.update(
            zip(self._robot.config.cameras, self._robot.cameras, strict=True)
        )
        for camera_config in self._observation_config.cameras:
            camera = self.model.bind(cameras[camera_config.name])
            if camera_config.pos is not None:
                camera.pos = camera_config.pos
            if camera_config.quat is not None:
                camera.quat = camera_config.quat
        return {
            name: (self.model.bind(camera).id, camera)
            for name, camera in cameras.items()
        }

    def _close_renderers(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
        for renderer in self.obs_renderers.values():
            renderer.close()
        self.mujoco_renderer = None
        self.obs_renderers.clear()
        strip = getattr(self, "_strip_renderer_state", None)
        if strip:
            strip.close()
        self._strip_renderer_state = None

    def _initialize_env(self):
        """Can be overwritten to add task specific items to scene."""
        pass

    def get_observation_space(self) -> spaces.Dict:
        """Get observation space."""
        obs_dict: dict[str, spaces.Space] = {}
        if self._observation_config.proprioception:
            obs_dict = {
                "proprioception": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(self._robot.qpos) + len(self._robot.qvel),),
                    dtype=np.float32,
                ),
                "proprioception_grippers": spaces.Box(
                    low=0,
                    high=1,
                    shape=(len(self.robot.qpos_grippers),),
                    dtype=np.float32,
                ),
            }
            if self.robot.floating_base:
                obs_dict.update(
                    {
                        "proprioception_floating_base": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(len(self.robot.floating_base.qpos),),
                            dtype=np.float32,
                        ),
                        "proprioception_floating_base_actions": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(
                                len(self.robot.floating_base.get_accumulated_actions),
                            ),
                            dtype=np.float32,
                        ),
                    }
                )
        if self._use_pixels:
            for camera in self.observation_config.cameras:
                if camera.rgb:
                    obs_dict[f"rgb_{camera.name}"] = spaces.Box(
                        low=0, high=255, shape=(3, *camera.resolution), dtype=np.uint8
                    )
                if camera.depth:
                    obs_dict[f"depth_{camera.name}"] = spaces.Box(
                        # todo: check if this is the correct range
                        low=0,
                        high=1,
                        shape=camera.resolution,
                        dtype=np.float32,
                    )
        if self._observation_config.privileged_information:
            obs_dict.update(self._get_task_privileged_obs_space())
        return spaces.Dict(obs_dict)

    def _get_task_privileged_obs_space(self) -> dict[str, Any]:
        """Get the task privileged observation space."""
        return {}

    def get_observation(self) -> dict[str, np.ndarray]:
        """Get the observation."""
        obs = {}
        if self._observation_config.proprioception:
            obs |= self._get_proprioception_obs()
        if self._use_pixels:
            obs |= self._get_visual_obs()
        if self._observation_config.privileged_information:
            obs |= self._get_task_privileged_obs()
        return obs

    def _get_task_info(self) -> dict[str, Any]:
        """Get the task info dict."""
        return {}

    def get_info(self) -> dict[str, Any]:
        """Get info dict."""
        info = self._get_task_info()
        info.update({"task_success": float(self.success)})
        return info

    def _get_proprioception_obs(self) -> dict[str, Any]:
        obs = {
            "proprioception": np.concatenate(
                [self._robot.qpos, self._robot.qvel]
            ).astype(np.float32),
            "proprioception_grippers": np.array(self.robot.qpos_grippers).astype(
                np.float32
            ),
        }
        if self.robot.floating_base:
            obs["proprioception_floating_base"] = np.array(
                self.robot.floating_base.qpos
            ).astype(np.float32)
            obs["proprioception_floating_base_actions"] = np.array(
                self.robot.floating_base.get_accumulated_actions
            ).astype(np.float32)
        return obs

    def _strip_renderer(self) -> Optional[mujoco.Renderer]:
        """One wide offscreen renderer for all RGB cameras (lazy).

        Every camera is drawn into its own viewport of a single ``H x (W * N)``
        buffer and the pixels come back with ONE ``mjr_readPixels`` instead of
        one per camera. ``glReadPixels`` is a full GPU sync, so N render/read
        pairs serialise the GPU work (~25% slower per control tick with three
        84x84 cameras).

        OPT-IN (``BIGYM_RENDER_STRIP=1``): the strip is NOT bit-identical to
        the per-camera path. Viewport-offset rasterisation changes a few dozen
        edge pixels per image, within the per-camera path's own
        frame-to-frame nondeterminism, but it is a different draw of the same
        scene. Applies only when every camera is RGB-only at one resolution
        and the model's offscreen buffer is wide enough.
        """
        if self._strip_renderer_state is not None:
            return self._strip_renderer_state or None
        cameras = list(self._observation_config.cameras)
        usable = (
            os.environ.get("BIGYM_RENDER_STRIP", "0") == "1"
            and len(cameras) > 1
            and all(c.rgb and not c.depth for c in cameras)
            and len({tuple(c.resolution) for c in cameras}) == 1
        )
        if usable:
            height, width = cameras[0].resolution
            model = self.model
            usable = (
                width * len(cameras) <= model.vis.global_.offwidth
                and height <= model.vis.global_.offheight
            )
        if not usable:
            self._strip_renderer_state = False
            return None
        height, width = cameras[0].resolution
        renderer = mujoco.Renderer(self.model, height, width * len(cameras))
        self._strip_renderer_state = renderer
        self._strip_buffer = np.empty((height, width * len(cameras), 3), np.uint8)
        return renderer

    def _get_visual_obs_strip(self, renderer: mujoco.Renderer) -> dict[str, Any]:
        cameras = self._observation_config.cameras
        height, width = cameras[0].resolution
        # Set together with the strip renderer in _strip_renderer().
        strip_buffer = self._strip_buffer
        assert strip_buffer is not None
        # Renderer's GL/scene/context attributes are private, absent from stubs.
        if renderer._gl_context:  # ty: ignore[unresolved-attribute]
            renderer._gl_context.make_current()  # ty: ignore[unresolved-attribute]
        for index, camera_config in enumerate(cameras):
            renderer.update_scene(self.data, self._cameras_map[camera_config.name][0])
            mujoco.mjr_render(
                mujoco.MjrRect(index * width, 0, width, height),
                renderer._scene,  # ty: ignore[unresolved-attribute]
                renderer._mjr_context,  # ty: ignore[unresolved-attribute]
            )
        mujoco.mjr_readPixels(
            strip_buffer,
            None,
            mujoco.MjrRect(0, 0, width * len(cameras), height),
            renderer._mjr_context,  # ty: ignore[unresolved-attribute]
        )
        image = np.flipud(strip_buffer)
        return {
            f"rgb_{camera_config.name}": np.ascontiguousarray(
                np.moveaxis(image[:, index * width : (index + 1) * width], -1, 0)
            )
            for index, camera_config in enumerate(cameras)
        }

    def _get_visual_obs(self) -> dict[str, Any]:
        """Get the visual observation."""
        strip = self._strip_renderer()
        if strip is not None:
            return self._get_visual_obs_strip(strip)
        obs = {}
        for camera_config in self._observation_config.cameras:
            obs_renderer = self.obs_renderers[camera_config.resolution]
            obs_renderer.update_scene(
                self.data, self._cameras_map[camera_config.name][0]
            )
            if camera_config.rgb:
                rgb = obs_renderer.render()
                obs[f"rgb_{camera_config.name}"] = np.moveaxis(rgb, -1, 0)
            if camera_config.depth:
                obs_renderer.enable_depth_rendering()
                obs[f"depth_{camera_config.name}"] = obs_renderer.render()
                obs_renderer.disable_depth_rendering()
        return obs

    def _get_task_privileged_obs(self) -> dict[str, Any]:
        """Get the task privileged observation."""
        return {}

    def _update_seed(self, override_seed=None):
        """Update the seed for the environment.

        Args:
            override_seed: If not None, the next seed will be set to this value.
        """
        if override_seed is not None:
            if not isinstance(override_seed, int):
                logging.warning(
                    "Expected override_seed to be an integer. Casting to int."
                )
                override_seed = int(override_seed)
            self._next_seed = override_seed
            self.action_space = self.action_mode.action_space(
                action_scale=self._sub_steps_count, seed=override_seed
            )
        self._current_seed = self._next_seed
        assert self._current_seed is not None
        if self._episode_seed_rng is None:
            self._next_seed = np.random.randint(2**32)
        else:
            self._next_seed = int(self._episode_seed_rng.integers(2**32))
        np.random.seed(self._current_seed)

    def get_episode_seed_state(self) -> Optional[dict[str, Any]]:
        """Return the isolated episode-seed chain state, if enabled."""
        if self._episode_seed_rng is None:
            return None
        return {
            "current_seed": self._current_seed,
            "next_seed": self._next_seed,
            "rng_state": copy.deepcopy(self._episode_seed_rng.bit_generator.state),
        }

    def set_episode_seed_state(self, state: dict[str, Any]) -> None:
        """Restore a state returned by :meth:`get_episode_seed_state`."""
        if self._episode_seed_rng is None:
            raise RuntimeError("Episode-seed isolation is not enabled.")
        self._current_seed = state["current_seed"]
        self._next_seed = state["next_seed"]
        self._episode_seed_rng.bit_generator.state = copy.deepcopy(state["rng_state"])

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment.

        Args:
           seed: If not None, the environment will be reset with this seed.
           options: Additional information to specify how the environment is reset
            (optional, depending on the specific environment).
        """
        self._env_health.reset()
        self._update_seed(override_seed=seed)
        self.simulation.reset()
        self._success_hold_counter = 0
        self._action = np.zeros_like(self._action)
        reset_pos, reset_quat = self._sample_reset_robot_pose()
        self._robot.reset(reset_pos, reset_quat)
        self._on_reset()
        self.simulation.forward()
        return self.get_observation(), self.get_info()

    def _sample_reset_robot_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample the robot spawn after the episode RNG has been seeded.

        The base implementation exactly preserves upstream's fixed pose.
        Task variants can override this hook for a versioned initialization
        profile without reimplementing the reset lifecycle.
        """
        return (
            np.asarray(self.RESET_ROBOT_POS, dtype=np.float64).copy(),
            np.asarray(self.RESET_ROBOT_QUAT, dtype=np.float64).copy(),
        )

    def _on_reset(self):
        """Custom environment reset behaviour."""
        pass

    def _on_reset_warmup_step(self):
        """Task hook after each outer-controller reset warmup step.

        Most task state should simply evolve under physics during controller
        settling. Tasks whose sampled initial state must not be displaced by
        the robot before the episode begins can hold it here.
        """
        pass

    def _on_step(self):
        """Custom environment behaviour after stepping."""
        pass

    def render(self):
        """Renders a frame of the simulation."""
        assert self.mujoco_renderer is not None
        return self.mujoco_renderer.render(self.render_mode)

    def step(
        self, action: np.ndarray, fast: bool = False
    ) -> tuple[Any, float, bool, bool, dict]:
        """Step the environment.

        Args:
            action: Action to take.
            fast: If True, perform the environment step without processing observations
                and return default values. Useful when performance is crucial,
                but observations are not required, e.g., demo collection in VR.

        Returns:
            tuple: (observation, reward, terminated, truncated, info).
        """
        self._step_cache.clean()
        self._step_mujoco_simulation(action)
        self._on_step()
        if self._success_hold_steps > 0:
            # Tick exactly once per env step, via the step cache. The counter
            # must NOT live inside _success(): tasks may call _success()
            # directly (e.g. reach-target _on_step highlighting bypasses the
            # cache), which would double-count a stateful predicate.
            if bool(self._step_cache.get(self._success)):
                self._success_hold_counter += 1
            else:
                self._success_hold_counter = 0
        self._action = action
        if fast:
            return {}, 0, False, False, {}
        else:
            return (
                self.get_observation(),
                self.reward,
                self.terminate,
                self.truncate,
                self.get_info(),
            )

    def _step_mujoco_simulation(self, action):
        """Step the mujoco simulation."""
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Action shape mismatch: "
                f"expected {self.action_space.shape}, but got {action.shape}."
            )
        if np.any(action < self.action_space.low) or np.any(
            action > self.action_space.high
        ):
            clipped_action = np.clip(
                action, self.action_space.low, self.action_space.high
            )
            raise ValueError(
                f"Action {action} is out of the action space bounds. "
                f"Overhead: {action - clipped_action}"
            )
        with self._env_health.track():
            # Substeps after the first are pure physics: the action mode
            # applies ctrl once per control step (its own step() ends with one
            # physics step), so the remainder batches into a single
            # mj_step(nstep=N-1) call. No mj_rnePostConstraint: the cfrc_*/cacc
            # diagnostics it fills are not read anywhere (MuJoCo still computes
            # them internally when a sensor needs them).
            self.action_mode.step(action)
            remaining = self._sub_steps_count - 1
            if remaining > 0:
                self.simulation.step(remaining)
            # mj_step leaves kinematics, contacts and sensors at the state
            # before its last substep; recompute them so task checks,
            # observations and the lower-body controller read the current
            # state.
            self.simulation.forward()

    def _success(self) -> bool:
        """Check if the episode is successful."""
        return False

    def _fail(self) -> Union[bool, np.bool_]:
        """Check if the episode is failed."""
        floating_base = self._robot.floating_base
        if floating_base:
            pelvis_position = np.zeros(3)
            for i, actuator in enumerate(floating_base.position_actuators):
                if actuator:
                    joint = floating_base.joint(actuator)
                    pelvis_position[i] = self.data.bind(joint).qpos.item()
                else:
                    pelvis_position[i] = self.model.bind(self._robot.pelvis).pos[i]
        else:
            pelvis_position = self.data.bind(self._robot.pelvis).xpos.copy()
        return np.linalg.norm(pelvis_position) > MAX_DISTANCE_FROM_ORIGIN

    def _reward(self) -> float:
        """Get current episode reward."""
        return float(self.success) * SPARSE_REWARD_FACTOR

    def close(self):
        """Close environment."""
        self._close_renderers()
