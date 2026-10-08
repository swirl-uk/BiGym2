"""The GR00T-WBC lower body keeps the G1 standing on a zero command.

The official env (controller in the loop) is reset and then held for
:data:`STAND_SECONDS` of simulated time on the hold action: zero base
velocity, the default height command and the arms where reset left them.
That is the command a policy sends while it thinks, and the state every
episode starts from; a controller that cannot hold it makes every task
unsolvable. The robot must not trip the controller's fall latch, the
pelvis must stay near standing height and the feet must stay put.

The bounds are loose on purpose (measured: pelvis height
0.747-0.748 m, horizontal drift 6.5 mm, tilt 0.42 deg): this test is for
"the robot stopped standing". Subtler controller damage -- feeding it zero
joint velocities, say, which still stands (drift 9 mm, tilt 0.67 deg) --
is ``tests/test_substrate_golden.py``'s to catch.
"""

import warnings

import mujoco
import numpy as np

from bigym.loco import make

STAND_SECONDS = 5.0
MIN_PELVIS_HEIGHT = 0.70  # m
MAX_PELVIS_DRIFT = 0.05  # m, horizontal, from the post-reset pose
MAX_TILT = np.deg2rad(5.0)  # pelvis z-axis away from vertical


def _pelvis_body(model) -> int:
    for body in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        if name.rsplit("/", 1)[-1] == "pelvis":
            return body
    raise KeyError("pelvis")


def test_the_robot_stands_on_a_zero_command():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = make("reach_target_dual", camera_keys=())
    try:
        outer = env.bigym
        inner = outer.inner_env
        model, data = inner.model, inner.data
        pelvis = _pelvis_body(model)

        env.reset(seed=0)
        assert not outer.episode_fell(), "fell during the reset warmup"
        hold = np.clip(outer.normalize_action(outer.raw_hold_action()), -1, 1).astype(
            np.float32
        )
        start = data.xpos[pelvis].copy()

        heights, drifts, tilts = [], [], []
        steps = int(round(STAND_SECONDS / outer.control_step_seconds))
        for _ in range(steps):
            time_step = env.step(hold)
            assert not time_step.last(), "the episode ended while standing still"
            heights.append(float(data.xpos[pelvis][2]))
            drifts.append(float(np.linalg.norm(data.xpos[pelvis][:2] - start[:2])))
            # xmat row-major: element 8 is the world-z component of body z.
            tilts.append(float(np.arccos(np.clip(data.xmat[pelvis][8], -1, 1))))

        assert not outer.episode_fell()
        assert outer.episode_termination() != "fell"
        assert min(heights) > MIN_PELVIS_HEIGHT, min(heights)
        assert max(drifts) < MAX_PELVIS_DRIFT, max(drifts)
        assert max(tilts) < MAX_TILT, np.rad2deg(max(tilts))
    finally:
        env.close()
