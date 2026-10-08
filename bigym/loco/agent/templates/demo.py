"""Read the human demonstration videos (development only, not available at evaluation).

    from harness.demo import Demo
    d = Demo()                       # finds demo_<camera>.mp4 + demo.json in the sandbox
    d.cameras                        # ('head', 'left_wrist', 'right_wrist')
    d.n_frames                       # frames per camera (all cameras are synchronised)
    im = d.frame("head", k)          # uint8 RGB (height, width, 3), same layout as tools.image()
    ims = d.frames("left_wrist")     # uint8 RGB (n_frames, height, width, 3)
    d.step_of(k)                     # control step (50 Hz) at which frame k was recorded
    d.frame_at_step("head", 300)     # the frame closest to control step 300

Frames come back as RGB like the live cameras; OpenCV's own ``VideoCapture``
would give BGR.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SANDBOX = Path(__file__).resolve().parent.parent


class Demo:
    """The demonstration videos of this sandbox, decoded on demand."""

    def __init__(self, sandbox: Path | None = None):
        """Locate the demonstration videos and their metadata.

        Args:
            sandbox: Sandbox directory, or None for this file's grandparent.

        Raises:
            FileNotFoundError: The sandbox holds no demonstration video.
        """
        self.sandbox = Path(sandbox or SANDBOX)
        meta = self.sandbox / "demo.json"
        self.meta = json.loads(meta.read_text()) if meta.exists() else {}
        # Control steps between consecutive frames (the loop runs at 50 Hz).
        self.every = int(self.meta.get("every", 2))
        self.fps = int(self.meta.get("fps", 25))
        files = sorted(self.sandbox.glob("demo_*.mp4")) or sorted(
            self.sandbox.glob("demo.mp4")
        )
        if not files:
            raise FileNotFoundError(f"no demo_*.mp4 in {self.sandbox}")
        self._files = {
            (f.stem[len("demo_") :] if f.stem.startswith("demo_") else "all"): f
            for f in files
        }
        self.cameras = tuple(self._files)
        self._cache: dict[str, np.ndarray] = {}
        self.n_frames = self._count(self.cameras[0])

    def _count(self, camera: str) -> int:
        """Return the frame count a container reports for one camera."""
        import cv2

        capture = cv2.VideoCapture(str(self._files[camera]))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        return count

    def frames(self, camera: str) -> np.ndarray:
        """Return every frame of one camera, decoded once and cached.

        Args:
            camera: Camera name (see ``cameras``).

        Returns:
            uint8 RGB ``(n_frames, height, width, 3)``.
        """
        import cv2

        if camera not in self._cache:
            capture = cv2.VideoCapture(str(self._files[camera]))
            out = []
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            capture.release()
            self._cache[camera] = np.stack(out)
            self.n_frames = len(out)
        return self._cache[camera]

    def frame(self, camera: str, k: int) -> np.ndarray:
        """Return frame ``k`` (0-based) of one camera.

        Args:
            camera: Camera name.
            k: Frame index.

        Returns:
            uint8 RGB ``(height, width, 3)``.
        """
        return self.frames(camera)[int(k)]

    def step_of(self, k: int) -> int:
        """Return the control step at which frame ``k`` was recorded.

        Args:
            k: Frame index.

        Returns:
            The control step (50 Hz, 0 = the first step after reset).
        """
        return int(k) * self.every

    def frame_at_step(self, camera: str, step: int) -> np.ndarray:
        """Return the frame recorded closest to a control step.

        Args:
            camera: Camera name.
            step: Control step.

        Returns:
            uint8 RGB ``(height, width, 3)``.
        """
        k = int(round(int(step) / self.every))
        return self.frame(camera, max(0, min(k, self.n_frames - 1)))
