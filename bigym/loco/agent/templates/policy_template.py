"""Your policy: fill in reset() and act().

Keep this file self-contained (helper modules next to it are fine). Available
libraries: numpy, scipy, OpenCV (cv2), pillow and the Python standard library;
nothing else is installed.

Contract
  reset(obs, tools)      called once at the start of every episode
  act(obs, tools) -> raw called every control step (50 Hz); must return 20 floats

  raw = [vx, vy, height, wz, left arm (7), right arm (7), left gripper, right gripper]
  See docs/api.md for units, ranges and the observation keys.
  tools.hold_action()    -> raw action that holds the current pose
  tools.ik(left_pos=..., right_pos=..., ...) -> 14 arm joint targets
  tools.image(camera, w, h) -> RGB array; tools.camera_info(); tools.pixel_to_ray(camera, u, v, w, h)
  tools.render(camera, path) -> save a PNG for your own inspection

Determinism: do not use wall-clock time; if you need randomness, seed it from
obs at reset (e.g. np.random.default_rng(int(obs["t"]) + 1)).
"""

import cv2  # noqa: F401  OpenCV: colour spaces, inRange, connectedComponents, contours
import numpy as np
import scipy  # noqa: F401  scipy.ndimage (label, filters), scipy.optimize, scipy.signal


class Policy:
    """The control policy that is scored on the hidden seeds."""

    def reset(self, obs, tools):
        """Prepare for one episode.

        Args:
            obs: The first observation.
            tools: The tools described in docs/api.md.
        """
        self.hold = np.asarray(tools.hold_action(), dtype=np.float32)

    def act(self, obs, tools):
        """Return the action for this control step.

        Args:
            obs: The current observation.
            tools: The tools described in docs/api.md.

        Returns:
            The physical action.
        """
        raw = self.hold.copy()
        # TODO: your control logic here
        return raw
