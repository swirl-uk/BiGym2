"""Environment configuration: one frozen dataclass whose defaults are official.

:class:`EnvConfig` holds every setting ``bigym.loco.make`` builds an env from.
Its defaults are the official benchmark values; each task states only how it
departs from them, next to its registration (:data:`bigym.loco.tasks.TASKS`).
``make(task)`` therefore builds the official env, and ``make(task, **kw)``
changes named fields on top of it.

The env records the config it was built from (``env.config``) and the fields
that differ from the task's official config (``env.config_overrides``).
:func:`conformance_violations` reads those to decide whether a result is
official; fields marked ``affects_results=False`` (rendering, policy-side
observation and action views) never make a run unofficial.

Metadata round trip: :meth:`EnvConfig.to_metadata` writes the config into
demo, replay and run metadata (``env_config``) together with the flat
``task`` / ``lowerbody_policy`` blocks that readers take single fields from
(task name, cameras), and :meth:`EnvConfig.from_metadata` reads
``env_config`` back into a config.
"""

from __future__ import annotations

import dataclasses
import functools
import typing
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from bigym.loco.action_representation import validate_upper_delta_scale_rad

# The camera and proprioception sets the demonstrations were recorded with.
DEFAULT_CAMERA_KEYS: tuple[str, ...] = ("head", "right_wrist", "left_wrist")
DEFAULT_STATE_KEYS: tuple[str, ...] = (
    "proprioception",
    "proprioception_grippers",
    "proprioception_floating_base",
)


def _exempt(default: Any) -> Any:
    """A field whose value never makes a run unofficial."""
    return field(default=default, metadata={"affects_results": False})


@functools.cache
def _literal_choices(config_type: Any) -> dict[str, tuple[Any, ...]]:
    """``{field: choices}`` for every ``Literal`` field of a config dataclass."""
    hints = typing.get_type_hints(config_type)
    return {
        f.name: typing.get_args(hints[f.name])
        for f in dataclasses.fields(config_type)
        if typing.get_origin(hints[f.name]) is Literal
    }


def _check_choices(config: Any) -> None:
    """Raise ValueError for a ``Literal`` field set outside its choices."""
    for name, choices in _literal_choices(type(config)).items():
        value = getattr(config, name)
        if value not in choices:
            raise ValueError(
                f"{type(config).__name__}.{name} must be one of {choices}, "
                f"got {value!r}"
            )


@dataclass(frozen=True)
class ControllerConfig:
    """The lower-body controller in the loop (GR00T-WBC on the G1).

    ``None`` command bounds mean "the ranges the backend declares".
    """

    # A registered backend name (bigym.loco.register_backend), or a
    # "pkg.module:ATTR" reference to a BackendBinding.
    backend: str = "groot_wbc_g1"
    # Base action slots carry velocity/height commands ("lowerbody_cmd") or
    # pelvis position deltas ("legacy_delta").
    base_action_mode: Literal["lowerbody_cmd", "legacy_delta"] = "lowerbody_cmd"
    # Torso-pitch command in the RY base slot (adds one action dimension).
    pitch_command: bool = True
    # Control steps the controller settles before the agent engages; the
    # measured settle is 178-180 steps. None = the backend's recommendation.
    reset_warmup_steps: int | None = 200
    # Restore the controller's canonical state at every reset, so reset(seed)
    # is a function of the seed alone.
    deterministic_reset: bool = True
    # Reset stance: the backend's keyframe pose (the only one measured).
    init_stance: Literal["keyframe"] = "keyframe"
    # Leave pelvis roll/pitch passive so the legs carry the base.
    passive_base_tilt: bool = True
    default_height_cmd: float = 0.74
    # Pelvis height at reset, before the warmup settles the robot.
    init_pelvis_z: float = 0.74
    height_cmd_min: float | None = None
    height_cmd_max: float | None = None
    pitch_cmd_min: float | None = None
    pitch_cmd_max: float | None = None
    default_pitch_cmd: float = 0.0
    cmd_clip: float = 1.0
    wz_clip: float = 1.0
    use_height_cmd: bool = True
    skyhook_kp: float = 0.0
    skyhook_damping: float = 0.0
    # Comma-separated "balance,walk" ONNX pair; None = the shipped weights.
    model_path: str | None = None

    def __post_init__(self) -> None:
        """Validate the fixed choices."""
        _check_choices(self)


@dataclass(frozen=True)
class WholeBodyConfig:
    """Outer action over the leg joints too (research modes, off by default)."""

    mode: Literal["hierarchical_current"] = "hierarchical_current"
    preserve_leg_proprio: bool = False

    def __post_init__(self) -> None:
        """Validate the fixed choices."""
        _check_choices(self)


@dataclass(frozen=True)
class EnvConfig:
    """Every setting an env is built from; the defaults are the official values.

    ``controller=None`` builds a floating-base env with no lower-body
    controller (fast, but not the benchmark substrate).
    ``episode_length=None`` takes the task's registered budget.
    """

    # A bigym.robots.configs.ROBOT_MODELS name.
    robot_model: Literal["g1_dex1"] = "g1_dex1"
    controller: ControllerConfig | None = field(default_factory=ControllerConfig)
    # Budget in env steps (outer steps = episode_length / demo_down_sample_rate).
    episode_length: int | None = None
    # Env steps per outer control step: 10 = 50 Hz.
    demo_down_sample_rate: int = 10
    # The success predicate must hold this long before the episode succeeds.
    success_hold_seconds: float = 1.0
    # Reach-task success radius in metres; None = the task class constant.
    reach_tolerance: float | None = None
    # The task reset distribution; "g1_id_v1" exists for the drawer and
    # wall-cupboard tasks only.
    initialization_profile: Literal["upstream", "g1_id_v1"] = "upstream"
    # Outer base slots [x, y, z, rz]; False keeps [x, y, rz].
    enable_all_floating_dof: bool = True
    control_pelvis: bool = True
    # Joint targets are absolute positions or per-step deltas.
    action_mode: Literal["absolute", "delta"] = "absolute"
    camera_keys: tuple[str, ...] = DEFAULT_CAMERA_KEYS
    camera_shape: tuple[int, int] = (84, 84)
    state_keys: tuple[str, ...] = DEFAULT_STATE_KEYS
    event_reward_shaping_enabled: bool = False
    event_reward_progress_scale: float = 0.0
    event_reward_holding_bonus: float = 0.0
    event_reward_lift_bonus: float = 0.0
    event_reward_target_bonus: float = 0.0
    wholebody: WholeBodyConfig | None = None
    # Policy-side views of the same observations and actions, and rendering.
    frame_stack: int = _exempt(1)
    normalize_low_dim_obs: bool = _exempt(False)
    action_representation: Literal["absolute", "upper_delta"] = _exempt("absolute")
    upper_delta_scale_rad: float | None = _exempt(None)
    event_progress_enabled: bool = _exempt(False)
    render_mode: str = _exempt("rgb_array")

    def __post_init__(self) -> None:
        """Validate the fixed choices and the settings that depend on each other."""
        _check_choices(self)
        if self.success_hold_seconds < 0.0:
            raise ValueError(
                f"success_hold_seconds must be >= 0, got {self.success_hold_seconds}"
            )
        if self.action_representation == "upper_delta":
            if self.action_mode != "absolute":
                raise ValueError(
                    "action_representation='upper_delta' requires "
                    "action_mode='absolute'; the inner environment must keep "
                    "absolute joint targets for frozen lower-body control"
                )
            validate_upper_delta_scale_rad(self.upper_delta_scale_rad)

    def override(
        self, overrides: Mapping[str, Any] | None = None, /, **kwargs: Any
    ) -> EnvConfig:
        """Return a copy with the named fields replaced.

        Keys are EnvConfig field names; an unknown key raises ValueError.
        ``controller`` / ``wholebody`` accept None, a config instance, or a
        mapping of its fields merged into the current one.
        """
        merged = dict(overrides or {})
        merged.update(kwargs)
        changes = {}
        for key, value in merged.items():
            if key not in ENV_FIELDS:
                raise ValueError(_unknown_field_message(key))
            changes[key] = _coerce(self, key, value)
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form (tuples as lists, nested configs as dicts)."""
        return _jsonable(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EnvConfig:
        """Inverse of :meth:`to_dict`; missing keys keep their defaults."""
        return cls().override(data)

    def differences(self, other: EnvConfig) -> dict[str, tuple[Any, Any]]:
        """``{dotted field: (other's value, this value)}`` for every differing field."""
        return _differences(self, other, "")

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> EnvConfig:
        """The config an env was built from, read back from its ``env_config`` block."""
        env_config = metadata.get("env_config")
        if env_config is None:
            raise ValueError(
                "metadata has no env_config block; re-download the dataset "
                "or re-record the batch with this version of BiGym"
            )
        return cls.from_dict(env_config)

    def to_metadata(self, task_name: str) -> dict[str, Any]:
        """``task``, ``lowerbody_policy`` and ``env_config`` metadata blocks.

        ``env_config`` is what :meth:`from_metadata` reads; the two flat
        blocks repeat the settings for readers that take single fields.
        """
        task = {"task_name": str(task_name)}
        for key in LEGACY_TASK_KEYS:
            task[key] = getattr(self, key)
        return _jsonable(
            {
                "task": task,
                "lowerbody_policy": _controller_to_legacy(self.controller),
                "env_config": self.to_dict(),
            }
        )


ENV_FIELDS = {f.name: f for f in dataclasses.fields(EnvConfig)}
NESTED = {"controller": ControllerConfig, "wholebody": WholeBodyConfig}

# Flat ``task`` metadata keys, in the order the collector has written them.
LEGACY_TASK_KEYS: tuple[str, ...] = (
    "robot_model",
    "enable_all_floating_dof",
    "control_pelvis",
    "action_mode",
    "demo_down_sample_rate",
    "episode_length",
    "camera_shape",
    "camera_keys",
    "state_keys",
    "render_mode",
    "normalize_low_dim_obs",
    "success_hold_seconds",
    "reach_tolerance",
    "initialization_profile",
)


def affects_results(name: str) -> bool:
    """Whether changing top-level field ``name`` makes a run unofficial."""
    return bool(ENV_FIELDS[name].metadata.get("affects_results", True))


def conformance_violations(config: EnvConfig, official: EnvConfig) -> list[str]:
    """Result-affecting fields where ``config`` departs from ``official``.

    An empty list means the env is the official one for its task, up to
    fields that cannot change a result (see :func:`affects_results`).
    """
    return [
        f"{name}: official {want!r}, env {got!r}"
        for name, (want, got) in config.differences(official).items()
        if affects_results(name.split(".", 1)[0])
    ]


def resolve_config(
    task_name: str,
    config: EnvConfig | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> EnvConfig:
    """The config ``make`` builds ``task_name`` from.

    ``config`` is None (the task's official config) or an :class:`EnvConfig`
    used as the base. ``overrides`` go on top. A None ``episode_length`` becomes the
    task's budget.
    """
    from bigym.loco.tasks import task_spec

    spec = task_spec(task_name)
    if config is None:
        base = spec.config()
    elif isinstance(config, EnvConfig):
        base = config
    else:
        raise TypeError(
            f"config must be an EnvConfig or None, got {type(config).__name__}"
        )
    resolved = base.override(overrides or {})
    if resolved.episode_length is None:
        resolved = dataclasses.replace(resolved, episode_length=spec.episode_length)
    return resolved


def _unknown_field_message(key: str) -> str:
    for nested, cls in NESTED.items():
        if key in {f.name for f in dataclasses.fields(cls)}:
            return f"{key!r} is a {cls.__name__} field; pass {nested}={{{key!r}: ...}}"
    return f"unknown EnvConfig field {key!r}; known fields: {', '.join(ENV_FIELDS)}"


def _coerce(config: EnvConfig, key: str, value: Any) -> Any:
    if key in NESTED:
        cls = NESTED[key]
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            current = getattr(config, key) or cls()
            known = {f.name: f for f in dataclasses.fields(cls)}
            unknown = sorted(set(value) - set(known))
            if unknown:
                raise ValueError(
                    f"unknown {cls.__name__} field(s) {unknown}; known fields: "
                    + ", ".join(known)
                )
            return dataclasses.replace(
                current,
                **{k: _coerce_scalar(known[k], v) for k, v in value.items()},
            )
        raise TypeError(
            f"{key} must be None, a {cls.__name__} or a mapping of its fields, "
            f"got {type(value).__name__}"
        )
    return _coerce_scalar(ENV_FIELDS[key], value)


def _coerce_scalar(spec: dataclasses.Field, value: Any) -> Any:
    kind = str(spec.type)
    if value is None:
        return None
    if kind.startswith("tuple["):
        if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
            raise TypeError(f"{spec.name} must be a sequence, got {value!r}")
        items = tuple(value)
        return tuple(int(v) for v in items) if "int" in kind else items
    if (
        kind.startswith("float")
        and isinstance(value, int)
        and not isinstance(value, bool)
    ):
        return float(value)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _differences(a: Any, b: Any, prefix: str) -> dict[str, tuple[Any, Any]]:
    out: dict[str, tuple[Any, Any]] = {}
    for f in dataclasses.fields(a):
        mine, theirs = getattr(a, f.name), getattr(b, f.name)
        name = f"{prefix}{f.name}"
        if (
            dataclasses.is_dataclass(mine)
            and dataclasses.is_dataclass(theirs)
            and type(mine) is type(theirs)
        ):
            out.update(_differences(mine, theirs, f"{name}."))
        elif mine != theirs:
            out[name] = (theirs, mine)
    return out


def _controller_to_legacy(controller: ControllerConfig | None) -> dict[str, Any]:
    """The flat ``lowerbody_policy`` block written beside ``env_config``."""
    if controller is None:
        return {}
    sub: dict[str, Any] = {
        "enable_pitch_cmd": controller.pitch_command,
        "passive_base_tilt": controller.passive_base_tilt,
        "default_height_cmd": controller.default_height_cmd,
        "default_pitch_cmd": controller.default_pitch_cmd,
    }
    for key in ("init_pelvis_z", "pitch_cmd_min", "pitch_cmd_max", "model_path"):
        if getattr(controller, key) is not None:
            sub[key] = getattr(controller, key)
    policy: dict[str, Any] = {
        "enabled": True,
        "backend": controller.backend,
        "base_action_mode": controller.base_action_mode,
        "reset_warmup_steps": controller.reset_warmup_steps,
        "deterministic_reset": controller.deterministic_reset,
        "init_stance": controller.init_stance,
        "support_base_with_legs": True,
        "cmd_clip": controller.cmd_clip,
        "wz_clip": controller.wz_clip,
        "use_height_cmd": controller.use_height_cmd,
        "skyhook_kp": controller.skyhook_kp,
        "skyhook_damping": controller.skyhook_damping,
        controller.backend: sub,
    }
    for key in ("height_cmd_min", "height_cmd_max"):
        if getattr(controller, key) is not None:
            policy[key] = getattr(controller, key)
    return policy
