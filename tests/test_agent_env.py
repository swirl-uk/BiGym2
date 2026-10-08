"""Slow tests for the coding-agent benchmark against the real environment.

These build the benchmark substrate (GR00T lower body in the loop) and render
with EGL, so they are marked slow: run with ``MUJOCO_GL=egl pytest --run-slow``.
"""

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

pytestmark = pytest.mark.slow

TASK = "reach_target_single"


@pytest.fixture(scope="module")
def strict_tools():
    """Build the benchmark env for TASK under the strict interface."""
    from bigym.loco.agent import EnvTools, EnvToolsConfig

    tools = EnvTools(TASK, EnvToolsConfig())
    tools.reset(0)
    yield tools
    tools.close()


@pytest.fixture(scope="module")
def tools_tools():
    """Build the benchmark env for TASK under the tools interface."""
    from bigym.loco.agent import EnvTools, EnvToolsConfig

    tools = EnvTools(TASK, EnvToolsConfig(interface="tools"))
    tools.reset(0)
    yield tools
    tools.close()


def test_action_dim_matches_the_protocol_layout(strict_tools):
    from bigym.loco.agent import task_pitch_enabled

    pitch = task_pitch_enabled(TASK)
    assert strict_tools.action_dim == (21 if pitch else 20)
    info = strict_tools.info()
    assert info["action_dim"] == strict_tools.action_dim
    assert info["pitch_enabled"] is pitch
    assert info["interface"] == "strict"
    assert info["ik"] is False
    assert info["calibration"] is False
    assert info["hand_pos"] is False


def test_hold_action_shape_and_zero_base_velocity(strict_tools):
    raw = strict_tools.hold_action()
    assert raw.shape == (strict_tools.action_dim,)
    assert np.isfinite(raw).all()
    slot = strict_tools._slot
    assert raw[slot["vx"]] == 0.0
    assert raw[slot["vy"]] == 0.0
    assert raw[slot["wz"]] == 0.0


def test_strict_observation_is_the_baselines_contract(strict_tools):
    obs = strict_tools.observation()
    for key in ("t", "time_limit", "fell", "low_dim_obs"):
        assert key in obs, key
    assert obs["low_dim_obs"].shape == (56 if strict_tools.action_dim == 21 else 50,)
    for withheld in (
        "base_pos",
        "base_yaw",
        "base_quat",
        "left_arm_qpos",
        "right_arm_qpos",
        "left_gripper",
        "right_gripper",
        "left_hand_pos",
        "right_hand_pos",
    ):
        assert withheld not in obs, withheld


def test_strict_interface_withholds_ik_and_calibration(strict_tools):
    with pytest.raises(RuntimeError, match="not available under this interface"):
        strict_tools.ik(left_pos=[0.3, 0.2, 0.9])
    with pytest.raises(RuntimeError, match="not available under this interface"):
        strict_tools.camera_info()


def test_tools_interface_adds_the_named_state(tools_tools):
    obs = tools_tools.observation()
    for key in ("base_pos", "base_yaw", "base_quat", "left_hand_pos", "right_hand_pos"):
        assert key in obs, key
    assert np.asarray(obs["base_pos"]).shape == (3,)
    assert np.asarray(obs["left_hand_pos"]).shape == (3,)
    cams = tools_tools.camera_info()
    assert set(cams) == {"head", "left_wrist", "right_wrist"}
    assert np.asarray(cams["head"]["rot"]).shape == (3, 3)


def test_image_returns_the_capped_resolution(strict_tools):
    img = strict_tools.image("head", 84, 84)
    assert img.shape == (84, 84, 3)
    assert img.dtype == np.uint8


def test_image_above_the_cap_is_refused(strict_tools):
    with pytest.raises(ValueError, match="largest image"):
        strict_tools.image("head", 640, 480)
    with pytest.raises(ValueError, match="not mounted on the robot"):
        strict_tools.image("third_person", 84, 84)


def test_step_accepts_a_physical_action_and_advances(strict_tools):
    strict_tools.reset(1)
    before = strict_tools.step_count
    strict_tools.step(strict_tools.hold_action())
    assert strict_tools.step_count == before + 1
    with pytest.raises(ValueError, match="shape"):
        strict_tools.step(np.zeros(3, dtype=np.float32))


def test_run_block_scores_the_template_policy(tmp_path):
    """Two hidden-seed episodes of the shipped template policy (it holds still)."""
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from bigym.loco.agent import EnvToolsConfig, evaluate

    template = Path(evaluate.__file__).parent / "templates" / "policy_template.py"
    policy = tmp_path / "policy.py"
    policy.write_text(template.read_text())

    recs, fingerprint, violations = evaluate.run_block(
        evaluate.Block(TASK, policy, EnvToolsConfig(), episodes=2), verbose=False
    )
    assert violations == []
    assert fingerprint.get("task") == TASK
    assert [r["episode"] for r in recs] == [0, 1]
    assert [r["seed"] for r in recs] == [
        evaluate.EVAL_SEED_BASE,
        evaluate.EVAL_SEED_BASE + 1,
    ]
    for r in recs:
        assert set(r) >= {"seed", "success", "length", "reward", "termination", "fell"}
        assert r["success"] in (0, 1)
        assert r["length"] > 0
        assert isinstance(r["termination"], str)

    out = tmp_path / "eval"
    out.mkdir()
    evaluate.write_csv(out / "episodes.csv", recs, "template")
    rows = (out / "episodes.csv").read_text().splitlines()
    assert rows[0] == ",".join(evaluate.COLUMNS)
    assert len(rows) == 3


TAMPER_POLICY = """
import gc

import numpy as np


class Policy:
    def reset(self, obs, tools):
        self.hold = np.asarray(tools.hold_action(), dtype=np.float32)
        for o in gc.get_objects():
            try:
                if not isinstance(o, type) and callable(
                    getattr(type(o), "_success", None)
                ):
                    o._success = lambda *a, **k: True
            except Exception:
                pass

    def act(self, obs, tools):
        return self.hold
"""


def test_an_evaluated_policy_cannot_patch_the_task_it_is_scored_on(tmp_path):
    """Patching ``_success`` through gc would score 1.0 next to the simulator.

    The policy runs in its own interpreter, which holds no environment: the
    patch finds nothing, and holding still on the hidden seed fails.
    """
    from bigym.loco.agent import EnvToolsConfig, evaluate

    policy = tmp_path / "policy.py"
    policy.write_text(TAMPER_POLICY)
    recs, _, violations = evaluate.run_block(
        evaluate.Block(TASK, policy, EnvToolsConfig(), episodes=1), verbose=False
    )
    assert violations == []
    assert recs[0]["success"] == 0
    assert recs[0]["reward"] == 0.0
    assert recs[0]["termination"] != "success"
