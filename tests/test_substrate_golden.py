"""Golden-trajectory regression test for the benchmark substrate.

The official env (G1 + GR00T-WBC lower body in the loop) is reset on a fixed
seed and driven by a fixed, scripted action sequence: zero base velocity at
the default height, both arms swinging through a slow sinusoid. After
:data:`OUTER_STEPS` control steps a compact signature of the state (pelvis
pose, a few leg/waist/arm joints, and the box for ``pick_box``) must match
the golden values below to :data:`TOLERANCE`.

Anything that changes the closed loop shows up here: MuJoCo options or
model edits, the controller's observation (e.g. feeding it the wrong joint
velocities), its command plumbing, the action normalisation, the reset and
warmup path. The fast suite otherwise only checks that these things run.

Tolerance: the signature is bit-identical across Python/onnxruntime
versions and the CI container. A 1e-4 relative noise on every controller
output (a generous stand-in for CPU kernel differences) moves it by at most
1.2e-4; zero joint velocities into the controller move it by 5e-3 to 7e-3.
TOLERANCE sits between the two.

When the substrate changes ON PURPOSE, bump ``SUBSTRATE_VERSION`` in
``bigym/loco/fingerprint.py`` and regenerate the goldens on a Linux machine with::

    MUJOCO_GL=egl python -m tests.test_substrate_golden

then paste the printed ``GOLDEN_SUBSTRATE_VERSION`` and ``GOLDEN`` over the
ones below. A golden mismatch WITHOUT an intended substrate change is a
regression, not a reason to regenerate.
"""

import functools
import warnings

import mujoco
import numpy as np
import pytest

from bigym.loco import make

# The substrate the goldens were recorded on (bigym.loco.fingerprint.SUBSTRATE_VERSION).
GOLDEN_SUBSTRATE_VERSION = "bigym2-mj381-v1"
RESET_SEED = 7
OUTER_STEPS = 150  # 3 s at the 50 Hz control rate
TOLERANCE = 5e-4  # metres / radians

# Joint basenames (the MJCF prefixes them with the robot name).
SIGNATURE_JOINTS = (
    "pelvis_x",
    "pelvis_y",
    "pelvis_z",
    "pelvis_rx",
    "pelvis_ry",
    "pelvis_rz",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "right_elbow_joint",
)

# reach_target_dual: the 20-dim action layout, no props in qpos.
# pick_box: the 21-dim layout (torso-pitch slot) and a free-floating box.
GOLDEN = {
    "pick_box": {
        "pelvis_x": -0.067884,
        "pelvis_y": -0.011498,
        "pelvis_z": 0.750364,
        "pelvis_rx": 0.007726,
        "pelvis_ry": -0.018700,
        "pelvis_rz": -0.028046,
        "left_hip_pitch_joint": -0.371347,
        "left_knee_joint": 0.703105,
        "right_knee_joint": 0.716761,
        "right_ankle_pitch_joint": -0.307391,
        "waist_yaw_joint": -0.015233,
        "waist_pitch_joint": 0.147261,
        "left_shoulder_pitch_joint": -0.122446,
        "right_elbow_joint": -0.163475,
        "box_x": 0.674419,
        "box_y": -0.983234,
        "box_z": 0.625075,
    },
    "reach_target_dual": {
        "pelvis_x": -0.063604,
        "pelvis_y": -0.011825,
        "pelvis_z": 0.749969,
        "pelvis_rx": 0.007210,
        "pelvis_ry": 0.004776,
        "pelvis_rz": -0.020754,
        "left_hip_pitch_joint": -0.395802,
        "left_knee_joint": 0.708518,
        "right_knee_joint": 0.720943,
        "right_ankle_pitch_joint": -0.314344,
        "waist_yaw_joint": -0.011786,
        "waist_pitch_joint": 0.038889,
        "left_shoulder_pitch_joint": -0.123395,
        "right_elbow_joint": -0.163732,
    },
}


def scripted_action(step: int, hold: np.ndarray, base_dim: int) -> np.ndarray:
    """The fixed action at ``step``: hold the base, swing the 14 arm joints.

    The swing ramps in over the first two seconds so the controller is not
    kicked, then each joint follows a 1 Hz sinusoid with its own phase. On a
    layout with the torso-pitch slot (base ``x, y, z, rz, ry``) the torso
    also leans forward by up to 0.15 rad, so the pitch command is exercised.
    """
    action = hold.copy()
    arms = slice(base_dim, base_dim + 14)
    ramp = min(step / 100.0, 1.0)
    phase = 2.0 * np.pi * step / 50.0 + np.arange(14)
    action[arms] = hold[arms] + 0.4 * ramp * np.sin(phase)
    if base_dim == 5:
        action[4] = hold[4] + 0.15 * ramp
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _qpos_address(model, basename: str) -> int:
    for joint in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
        if name.rsplit("/", 1)[-1] == basename:
            return int(model.jnt_qposadr[joint])
    raise KeyError(basename)


@functools.lru_cache(maxsize=None)
def rollout(task_name: str) -> tuple[dict[str, float], dict]:
    """Run the scripted rollout once; return its signature and fingerprint."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = make(task_name, camera_keys=())
    try:
        outer = env.bigym
        inner = outer.inner_env
        model, data = inner.model, inner.data
        env.reset(seed=RESET_SEED)
        # Zero base velocity, default height, current arm and gripper targets.
        hold = np.clip(outer.normalize_action(outer.raw_hold_action()), -1, 1)
        base_dim = outer.wholebody_action_layout()["base_dim"]
        for step in range(OUTER_STEPS):
            env.step(scripted_action(step, hold, base_dim))
        signature = {
            name: float(data.qpos[_qpos_address(model, name)])
            for name in SIGNATURE_JOINTS
        }
        if task_name == "pick_box":
            box = np.asarray(inner.box.get_position(), dtype=float)
            signature.update(box_x=box[0], box_y=box[1], box_z=box[2])
        return signature, env.substrate_fingerprint()
    finally:
        env.close()


@pytest.mark.parametrize("task_name", sorted(GOLDEN))
def test_a_scripted_rollout_lands_on_the_golden_state(task_name):
    signature, fingerprint = rollout(task_name)
    golden = GOLDEN[task_name]
    assert sorted(signature) == sorted(golden)
    off = {
        name: (round(signature[name], 6), golden[name])
        for name in golden
        if abs(signature[name] - golden[name]) > TOLERANCE
    }
    assert not off, (
        f"{task_name}: (got, golden) beyond {TOLERANCE}: {off}. Goldens were "
        f"recorded on {GOLDEN_SUBSTRATE_VERSION}; this env is "
        f"{fingerprint['substrate_version']} on MuJoCo "
        f"{fingerprint['mujoco_version']} (see the module docstring)."
    )


def test_the_fingerprint_names_the_substrate_and_the_mujoco_build():
    _signature, fingerprint = rollout("reach_target_dual")
    assert fingerprint["substrate_version"] == GOLDEN_SUBSTRATE_VERSION
    assert fingerprint["mujoco_version"] == mujoco.__version__
    assert fingerprint["task"] == "reach_target_dual"
    assert fingerprint["lowerbody_backend"] == "groot_wbc_g1"


if __name__ == "__main__":
    # Regeneration helper: print goldens to paste over the ones above.
    from bigym.loco.env import SUBSTRATE_VERSION

    print(f"GOLDEN_SUBSTRATE_VERSION = {SUBSTRATE_VERSION!r}")
    print("GOLDEN = {")
    for task in sorted(GOLDEN):
        print(f'    "{task}": {{')
        for name, value in rollout(task)[0].items():
            print(f'        "{name}": {value:.6f},')
        print("    },")
    print("}")
