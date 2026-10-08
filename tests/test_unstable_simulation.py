from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from bigym.action_modes import JointPositionActionMode
from bigym.bigym_env import BiGymEnv
from bigym.envs.reach_target import ReachTarget
from bigym.utils.env_health import (
    EnvHealth,
    UnstableSimulationError,
    UnstableSimulationWarning,
)

# A base-yaw delta far outside any sane command, applied every step, drives
# the floating-base spring into a NaN/huge-acceleration physics error. The G1
# base (1e4 N m/rad, 1 ms timestep) shrugs off a 180 deg/step delta, so the
# magnitudes are sized to blow it up within a handful of steps.
BASE_DELTA_RZ = 1e4
UNSTABLE_ACTION_SPACE_MULTIPLIER = 1e6


def run_simulation(
    env: BiGymEnv, unstable: bool
) -> tuple[Any, float, bool, bool, dict]:
    timestep = env.reset()
    if unstable:
        # Widen the bounds from the env's own (unscaled) space each call, so
        # repeated unstable runs on one env do not compound the multiplier
        # past float32 range.
        base_space = env.action_mode.action_space(env._sub_steps_count)
        env.action_space.low[:] = base_space.low * UNSTABLE_ACTION_SPACE_MULTIPLIER
        env.action_space.high[:] = base_space.high * UNSTABLE_ACTION_SPACE_MULTIPLIER
    for _ in range(100):
        action = np.zeros_like(env.action_space.sample())
        if unstable:
            action[2] = BASE_DELTA_RZ
        timestep = env.step(action)
        if not env.is_healthy:
            break
    return timestep


def test_unstable_simulation_warns():
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=False),
    )
    with pytest.warns(UnstableSimulationWarning):
        run_simulation(env, unstable=True)


def test_unstable_simulation_is_truncated():
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=False),
    )
    with pytest.warns(UnstableSimulationWarning):
        _, _, _, truncate, _ = run_simulation(env, unstable=True)
        assert truncate


def test_unstable_simulation_consecutive_warnings():
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=False),
    )
    with pytest.raises(UnstableSimulationError):
        # Run unstable simulation multiple times to cause UnstableSimulationError
        for _ in range(EnvHealth.CONSECUTIVE_WARNINGS_THRESHOLD):
            with pytest.warns(UnstableSimulationWarning):
                run_simulation(env, unstable=True)


def test_unstable_simulation_non_consecutive_warnings():
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=False),
    )
    # Run unstable simulation 1 time less than the CONSECUTIVE_WARNINGS_THRESHOLD
    for _ in range(EnvHealth.CONSECUTIVE_WARNINGS_THRESHOLD - 1):
        with pytest.warns(UnstableSimulationWarning):
            run_simulation(env, unstable=True)

    # Run stable simulation to reset counter
    run_simulation(env, unstable=False)
    env.reset()

    # Running unstable simulation now would not cause exception
    with pytest.warns(UnstableSimulationWarning):
        run_simulation(env, unstable=True)
        env.reset()

    env.close()


def test_gripper_model_is_stable():
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True)
    )
    env.reset()

    # The finger joints are the most unstable part of the model (the upstream
    # Robotiq 2F-85 fingertips needed a Menagerie fix for this). Read them
    # through the robot's grippers so the test follows the default robot:
    # ``Gripper.qpos`` is the actuated finger qpos mapped onto the gripper's
    # control range, and the gripper action is held at the open end of that
    # range while the arms take random actions.
    grippers = list(env.robot.grippers.values())
    assert grippers, "the default robot has no grippers to check"
    open_ctrl = [float(gripper.range[0]) for gripper in grippers]
    # Acceptable "jitter" of the fingers is 10% of the control range.
    tolerances = [float(np.ptp(gripper.range)) * 0.1 for gripper in grippers]
    # Let the fingers travel from the reset pose to the commanded open pose
    # first; the check is about jitter while the arms move, not that travel.
    settle_action = np.zeros_like(env.action_space.sample())
    settle_action[-len(grippers) :] = open_ctrl
    for _ in range(50):
        env.step(settle_action)
    for _ in range(100):
        action = env.action_space.sample()
        action[-len(grippers) :] = open_ctrl
        env.step(action)
        for gripper, target, tolerance in zip(
            grippers, open_ctrl, tolerances, strict=True
        ):
            assert_allclose(gripper.qpos, target, rtol=0, atol=tolerance)
