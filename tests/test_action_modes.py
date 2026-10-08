import numpy as np
import pytest

from bigym.action_modes import (
    ActionMode,
    JointPositionActionMode,
    PelvisDof,
)
from bigym.bigym_env import BiGymEnv
from bigym.envs.reach_target import ReachTarget
from bigym.robots.floating_base import RobotFloatingBase

# ``block_until_reached`` settles for a bounded time and, as its docstring
# says, guarantees nothing under collisions. The default G1Dex1 rests its feet
# and knees on the floor in the floating-base env, so friction holds a small
# steady-state offset against the base springs and a downward Z command is
# simply blocked. The tests below therefore (a) assert a residual band on the
# base instead of ``is_target_reached`` and (b) lift the base clear of the
# floor before driving it in all four DOFs.
BASE_RESIDUAL_LINEAR = 0.01
BASE_RESIDUAL_ANGULAR = 0.05


def _base_residuals(env: BiGymEnv) -> tuple[list[float], list[float]]:
    """Absolute |ctrl - qpos| per floating-base position / rotation actuator."""
    floating_base = env.robot.floating_base
    assert floating_base is not None

    def residual(actuator) -> float:
        return abs(
            float(env.data.bind(actuator).ctrl.item())
            - float(env.data.bind(floating_base.joint(actuator)).qpos.item())
        )

    linear = [residual(a) for a in floating_base.position_actuators if a]
    angular = [residual(a) for a in floating_base.rotation_actuators if a]
    return linear, angular


def _assert_base_settled(env: BiGymEnv) -> None:
    linear, angular = _base_residuals(env)
    assert max(linear, default=0.0) < BASE_RESIDUAL_LINEAR, linear
    assert max(angular, default=0.0) < BASE_RESIDUAL_ANGULAR, angular


def test_floating_base():
    rng = np.random.default_rng(0)
    env: BiGymEnv = ReachTarget(
        action_mode=JointPositionActionMode(
            floating_base=True,
            floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
            block_until_reached=True,
        )
    )
    env.reset()
    # Lift the base until the legs hang clear of the floor.
    lift = np.zeros_like(env.action_space.sample())
    lift[2] = RobotFloatingBase.DELTA_RANGE_POS[1]
    for _ in range(15):
        env.step(lift)
    for _ in range(100):
        ctrl = np.zeros_like(env.action_space.sample())
        ctrl[0:3] = rng.uniform(*RobotFloatingBase.DELTA_RANGE_POS)
        ctrl[3] = rng.uniform(*RobotFloatingBase.DELTA_RANGE_ROT)
        env.step(ctrl)
        _assert_base_settled(env)


@pytest.mark.parametrize(
    "action_mode",
    [
        JointPositionActionMode(floating_base=True, absolute=False),
        JointPositionActionMode(floating_base=True, absolute=True),
    ],
)
class TestActionModeStability:
    ENV_STEPS_COUNT = 100
    UNSTABLE_ACTION_MULTIPLIER = 50

    def run_simulation(self, env: BiGymEnv, action: np.ndarray):
        env.reset()
        for _ in range(self.ENV_STEPS_COUNT):
            env.step(action)

    def test_action_space_is_stable(self, action_mode: ActionMode):
        env: BiGymEnv = ReachTarget(action_mode=action_mode)
        low = env.action_space.low
        high = env.action_space.high
        self.run_simulation(env, low)
        self.run_simulation(env, high)

    def test_action_space_raises_when_beyond_bounds(self, action_mode: ActionMode):
        env: BiGymEnv = ReachTarget(action_mode=action_mode)
        low = env.action_space.low
        high = env.action_space.high
        with pytest.raises(ValueError):
            self.run_simulation(env, low * self.UNSTABLE_ACTION_MULTIPLIER)
        with pytest.raises(ValueError):
            self.run_simulation(env, high * self.UNSTABLE_ACTION_MULTIPLIER)
