"""Collector constants."""

from __future__ import annotations

DEMO_PIPELINE_VERSION = "2026-08-26-success-hold-training-view-v1"

# The official GROOT-WBC wrapper selects Balance below 0.05 m/s command norm
# and Walk otherwise. Keep a small margin above that boundary so leaving the
# VR stick deadzone can never produce a non-zero command that still runs the
# Balance network. This shaping is GROOT-specific: it exists because of the
# Balance/Walk switch, not as a generic stick deadzone.
GROOT_WBC_POLICY_WALK_THRESHOLD = 0.05
GROOT_WBC_MIN_WALK_SPEED = 0.055

# The first IK command is interpolated over one second from the current
# joints; arm speed is capped at the 6 rad/s of NVIDIA's G1 safety monitor.
# The cap is loose for ordinary tracking and only catches rare
# redundant-branch IK jumps (up to 20 rad/s).
G1_ARM_ENGAGE_SMOOTH_SECONDS = 1.0
G1_ARM_MAX_JOINT_SPEED = 6.0


# G1 backends share the outer-action contract (lowerbody_cmd base mode, a G1
# robot, no torso-yaw target); their settings come from the env's
# ControllerConfig.
G1_LOWERBODY_BACKENDS: tuple[str, ...] = ("groot_wbc_g1",)
# G1 robot models: the 29-dof body with Dex1-1 parallel grippers and a
# one-scalar-per-hand action layout.
G1_ROBOT_MODELS: tuple[str, ...] = ("g1_dex1",)


G1_LEFT_ARM_NAMES: tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
G1_RIGHT_ARM_NAMES: tuple[str, ...] = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
