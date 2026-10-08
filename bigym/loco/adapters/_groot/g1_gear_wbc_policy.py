"""G1 Gear-WBC policy runtime (NVIDIA GR00T-WholeBodyControl, vendored).

Origin: ``decoupled_wbc/control/policy/g1_gear_wbc_policy.py`` at upstream
commit ``021df739f0b36e514399f0030e3a195683a46383`` (Apache-2.0). Edits
relative to the original are listed in ``PROVENANCE.md``: the ``torch``
tensors became numpy arrays (the network runs in onnxruntime either way),
the ONNX paths are taken as given instead of being joined onto the
upstream resources directory, sessions are pinned to one thread, and the
keyboard teleop handler was dropped. The observation layout, the history
buffer, the Balance/Walk switch and every numeric constant are unchanged.
"""

from __future__ import annotations

import collections
from typing import Any, Dict, Optional

import numpy as np
import onnxruntime as ort

from bigym.loco.adapters._groot.gear_wbc_utils import (
    get_gravity_orientation,
    load_config,
)
from bigym.loco.adapters.onnx_session import single_thread_session_options


class G1GearWbcPolicy:
    """Simple G1 robot policy using OpenGearWbc trained neural network."""

    def __init__(self, robot_model, config: str, model_path: str):
        """Load the config and the ``balance,walk`` ONNX pair.

        Args:
            robot_model: object with ``get_joint_group_indices("body" |
                "lower_body")``.
            config: path to the gear_wbc YAML configuration file.
            model_path: comma-separated absolute paths of the Balance and
                Walk ONNX files.
        """
        self.config, self.LEGGED_GYM_ROOT_DIR = load_config(config)
        self.robot_model = robot_model
        self.use_teleop_policy_cmd = False

        model_path_1, model_path_2 = model_path.split(",")
        self.policy_1 = self.load_onnx_policy(model_path_1.strip())
        self.policy_2 = self.load_onnx_policy(model_path_2.strip())

        # Initialize observation history buffer
        self.observation = None
        self.obs_tensor: Optional[np.ndarray] = None
        self.obs_history = collections.deque(maxlen=self.config["obs_history_len"])
        self.obs_buffer = np.zeros(self.config["num_obs"], dtype=np.float32)
        self.counter = 0

        # Initialize state variables
        self.use_policy_action = False
        self.action = np.zeros(self.config["num_actions"], dtype=np.float32)
        self.target_dof_pos = self.config["default_angles"].copy()
        self.cmd = self.config["cmd_init"].copy()
        self.height_cmd = self.config["height_cmd"]
        self.freq_cmd = self.config["freq_cmd"]
        self.roll_cmd = self.config["rpy_cmd"][0]
        self.pitch_cmd = self.config["rpy_cmd"][1]
        self.yaw_cmd = self.config["rpy_cmd"][2]
        self.gait_indices = np.zeros((1,), dtype=np.float32)

    def load_onnx_policy(self, model_path: str):
        """Return ``obs (1, num_obs) float32 -> action (1, num_actions)``."""
        model = ort.InferenceSession(
            model_path, sess_options=single_thread_session_options()
        )
        input_name = model.get_inputs()[0].name

        def run_inference(input_tensor: np.ndarray) -> np.ndarray:
            return model.run(None, {input_name: input_tensor})[0]

        return run_inference

    def compute_observation(
        self, observation: Dict[str, Any]
    ) -> tuple[np.ndarray, int]:
        """Compute the observation vector from current state."""
        # Gait clock: advanced every call, carried in the state snapshot; the
        # clock inputs derived from it upstream never entered the observation.
        self.gait_indices = np.remainder(
            self.gait_indices + 0.02 * self.freq_cmd, 1.0
        ).astype(np.float32, copy=False)

        body_indices = self.robot_model.get_joint_group_indices("body")
        body_indices = [idx for idx in body_indices]

        n_joints = len(body_indices)

        # Extract joint data
        qj = observation["q"][body_indices].copy()
        dqj = observation["dq"][body_indices].copy()

        # Extract floating base data
        quat = observation["floating_base_pose"][3:7].copy()  # quaternion
        omega = observation["floating_base_vel"][3:6].copy()  # angular velocity

        # Handle default angles padding
        if len(self.config["default_angles"]) < n_joints:
            padded_defaults = np.zeros(n_joints, dtype=np.float32)
            padded_defaults[: len(self.config["default_angles"])] = self.config[
                "default_angles"
            ]
        else:
            padded_defaults = self.config["default_angles"][:n_joints]

        # Scale the values
        qj_scaled = (qj - padded_defaults) * self.config["dof_pos_scale"]
        dqj_scaled = dqj * self.config["dof_vel_scale"]
        gravity_orientation = get_gravity_orientation(quat)
        omega_scaled = omega * self.config["ang_vel_scale"]

        # Calculate single observation dimension
        single_obs_dim = (
            86  # 3 + 1 + 3 + 3 + 3 + n_joints + n_joints + 15, n_joints = 29
        )

        # Create single observation
        single_obs = np.zeros(single_obs_dim, dtype=np.float32)
        single_obs[0:3] = self.cmd[:3] * self.config["cmd_scale"]
        single_obs[3:4] = np.array([self.height_cmd])
        single_obs[4:7] = np.array([self.roll_cmd, self.pitch_cmd, self.yaw_cmd])
        single_obs[7:10] = omega_scaled
        single_obs[10:13] = gravity_orientation
        single_obs[13 : 13 + n_joints] = qj_scaled
        single_obs[13 + n_joints : 13 + 2 * n_joints] = dqj_scaled
        single_obs[13 + 2 * n_joints : 13 + 2 * n_joints + 15] = self.action
        return single_obs, single_obs_dim

    def set_observation(self, observation: Dict[str, Any]):
        """Update the policy's current observation of the environment.

        Args:
            observation: dict with ``q``, ``dq``, ``floating_base_pose`` and
                ``floating_base_vel`` for the current state.
        """
        # Extract the single observation
        self.observation = observation
        single_obs, single_obs_dim = self.compute_observation(observation)

        # Add current observation to history
        self.obs_history.append(single_obs)

        # Fill history with zeros if not enough observations yet
        while len(self.obs_history) < self.config["obs_history_len"]:
            self.obs_history.appendleft(np.zeros_like(single_obs))

        # Construct full observation with history
        single_obs_dim = len(single_obs)
        for i, hist_obs in enumerate(self.obs_history):
            start_idx = i * single_obs_dim
            end_idx = start_idx + single_obs_dim
            self.obs_buffer[start_idx:end_idx] = hist_obs

        # Batch of one for the network
        self.obs_tensor = self.obs_buffer[None, :]

        assert self.obs_tensor.shape[1] == self.config["num_obs"]

    def reset(self):
        """Reset hook of the upstream ``Policy`` base class (a no-op there too)."""

    def close(self):
        """Close hook of the upstream ``Policy`` base class (a no-op there too)."""

    def set_use_teleop_policy_cmd(self, use_teleop_policy_cmd: bool):
        """Enable/disable command input from ``get_action`` arguments."""
        self.use_teleop_policy_cmd = use_teleop_policy_cmd
        # Safety: When teleop is disabled, reset navigation to stop
        if not use_teleop_policy_cmd:
            self.nav_cmd = self.config["cmd_init"].copy()  # Reset to safe default

    def set_goal(self, goal: Dict[str, Any]):
        """Set the goal for the policy.

        Args:
            goal: Dictionary containing the goal for the policy
        """
        if "toggle_policy_action" in goal:
            if goal["toggle_policy_action"]:
                self.use_policy_action = not self.use_policy_action

    def get_action(
        self,
        time: Optional[float] = None,
        arms_target_pose: Optional[np.ndarray] = None,
        base_height_command: Optional[np.ndarray] = None,
        torso_orientation_rpy: Optional[np.ndarray] = None,
        interpolated_navigate_cmd: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute and return the next action based on current observation.

        Args:
            time: Optional "monotonic time" for time-dependent policies (unused)

        Returns:
            Dictionary containing the action to be executed
        """
        if self.obs_tensor is None:
            raise ValueError("No observation set. Call set_observation() first.")

        if base_height_command is not None and self.use_teleop_policy_cmd:
            self.height_cmd = (
                base_height_command[0]
                if isinstance(base_height_command, list)
                else base_height_command
            )

        if interpolated_navigate_cmd is not None and self.use_teleop_policy_cmd:
            self.cmd = interpolated_navigate_cmd

        if torso_orientation_rpy is not None and self.use_teleop_policy_cmd:
            self.roll_cmd = torso_orientation_rpy[0]
            self.pitch_cmd = torso_orientation_rpy[1]
            self.yaw_cmd = torso_orientation_rpy[2]

        # Select appropriate policy based on command magnitude
        if np.linalg.norm(self.cmd) < 0.05:
            # Use standing policy for small commands
            policy = self.policy_1
        else:
            # Use walking policy for movement commands
            policy = self.policy_2

        self.action = policy(self.obs_tensor).squeeze()

        # Transform action to target_dof_pos
        if self.use_policy_action:
            cmd_q = (
                self.action * self.config["action_scale"]
                + self.config["default_angles"]
            )
        else:
            cmd_q = self.observation["q"][
                self.robot_model.get_joint_group_indices("lower_body")
            ]

        cmd_dq = np.zeros(self.config["num_actions"])
        cmd_tau = np.zeros(self.config["num_actions"])

        return {"body_action": (cmd_q, cmd_dq, cmd_tau)}
