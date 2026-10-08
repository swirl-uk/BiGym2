"""CI smoke test for the bigym.loco environment factory.

Every task in :data:`bigym.loco.tasks.TASK_MAP` must construct, reset and
take one zero action, and the observations it hands back must match the
shapes/dtypes its own specs advertise. The sweep uses the floating-base
research env (``make(task, controller=None)``), so it needs no controller
and runs anywhere the MJCF assets do.

One further test builds the official env, with the GR00T lower-body
controller in the loop, and asserts the task registry is realised; it skips unless the policy runtime is reachable
(vendored in ``bigym.loco.adapters._groot``, always available).
"""

import os
from typing import Any

import numpy as np
import pytest

from bigym.loco import make
from bigym.loco.eval import protocol_violations
from bigym.loco.tasks import TASK_MAP

# All names in TASK_MAP are canonical (unsuffixed) G1 tasks.
G1_TASKS = tuple(sorted(TASK_MAP))


def _rendering_available() -> bool:
    """Return True when MuJoCo can open an offscreen GL context here."""
    if not os.environ.get("MUJOCO_GL"):
        return False
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><geom size='0.1'/></worldbody></mujoco>"
        )
        renderer = mujoco.Renderer(model, height=16, width=16)
        renderer.close()
        return True
    except Exception:  # any GL failure means "no rendering here"
        return False


RENDERING = _rendering_available()
NO_RENDER_REASON = (
    "no offscreen GL context (MUJOCO_GL unset or the backend failed to "
    "initialise); RGB observations are not exercised"
)


def _make_env(task_name: str):
    """Build a floating-base env (no controller) for ``task_name``, tiny budget."""
    kwargs: dict[str, Any] = dict(controller=None, episode_length=10)
    if not RENDERING:
        # Keep the smoke test useful on a runner with no GL: the low-dim
        # half of the observation is still fully exercised.
        kwargs["camera_keys"] = ()
    return make(task_name, **kwargs)


def _assert_specs_match_observation(env, time_step):
    """Assert the returned observation matches the env's advertised specs."""
    low_dim_spec = env.low_dim_observation_spec()
    low_dim = np.asarray(time_step.low_dim_obs)
    assert low_dim.shape == tuple(low_dim_spec.shape)
    assert low_dim.dtype == low_dim_spec.dtype
    assert np.all(np.isfinite(low_dim))

    if not RENDERING:
        return
    rgb_spec = env.rgb_observation_spec()
    rgb = np.asarray(time_step.rgb_obs)
    assert rgb.shape == tuple(rgb_spec.shape)
    assert rgb.dtype == rgb_spec.dtype
    assert rgb.shape[0] > 0, "expected at least one camera in the RGB stack"


def _smoke_one_task(task_name: str):
    """Construct, reset and single-step ``task_name``, checking every spec."""
    env = _make_env(task_name)
    try:
        action_spec = env.action_spec()
        assert len(action_spec.shape) == 1 and action_spec.shape[0] > 0

        time_step = env.reset(seed=0)
        _assert_specs_match_observation(env, time_step)

        zero_action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        time_step = env.step(zero_action)
        _assert_specs_match_observation(env, time_step)
    finally:
        env.close()


@pytest.mark.parametrize("task_name", G1_TASKS)
def test_canonical_g1_task_constructs_resets_and_steps(task_name):
    """Every canonical (G1) task builds and survives one zero action."""
    _smoke_one_task(task_name)


def test_rgb_observations_are_exercised():
    """Fail loudly only if rendering silently stopped being tested."""
    if not RENDERING:
        pytest.skip(NO_RENDER_REASON)
    env = _make_env(G1_TASKS[0])
    try:
        rgb = np.asarray(env.reset(seed=0).rgb_obs)
        assert rgb.shape == tuple(env.rgb_observation_spec().shape)
        assert rgb.dtype == np.uint8
    finally:
        env.close()


def test_floating_base_env_is_not_official():
    """A floating-base env records the override and is flagged."""
    env = _make_env("reach_target_dual")
    try:
        assert env.config.controller is None
        assert env.config_overrides["controller"] is None
        assert env.substrate_fingerprint()["lowerbody_backend"] is None
        assert any(v.startswith("controller:") for v in protocol_violations(env))
    finally:
        env.close()


def test_groot_wbc_official_env_is_official():
    """make(task) with GR00T in the loop is the official configuration."""
    env = make("reach_target_dual")
    try:
        action_spec = env.action_spec()
        env.reset(seed=0)
        zero_action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        for _ in range(3):
            env.step(zero_action)

        fingerprint = env.substrate_fingerprint()
        assert fingerprint["lowerbody_backend"] == "groot_wbc_g1"
        assert env.config_overrides == {}
        assert protocol_violations(env) == []
    finally:
        env.close()


def test_pick_box_g1_box_stays_on_platform_through_reset_warmup():
    """The pick_box box must survive the controller settle at every reset seed.

    A resting box-on-box contact is solver-unstable on the deferred-warmup
    path and can launch the box metres away. The box spawns at its resting
    height on the low side table; every seed must settle where it spawned.
    """
    # No camera observations are read, so skip rendering (runs without GL).
    env = make("pick_box", camera_keys=())
    try:
        env.set_deferred_reset_warmup(True)
        outer = env.bigym
        inner = outer.inner_env
        data = inner.data
        dof = int(inner.model.bind(inner.box.freejoint).dofadr[0])
        for seed in range(1, 13):
            env.reset(seed=seed)
            peak_speed = 0.0
            while outer.pending_reset_warmup_steps > 0:
                outer.run_reset_warmup_steps(4)
                peak_speed = max(
                    peak_speed, float(np.linalg.norm(data.qvel[dof : dof + 3]))
                )
            box = inner.box.get_position()
            assert peak_speed < 1.0, (seed, peak_speed)
            assert box[2] == pytest.approx(0.625, abs=0.004), (seed, box)
            assert abs(box[0] - 0.70) < 0.06 and abs(box[1] + 1.0) < 0.06, (seed, box)
    finally:
        env.close()
