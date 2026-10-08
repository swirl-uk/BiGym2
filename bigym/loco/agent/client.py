"""Agent-side client for the environment server (readable by the agent).

    from harness.client import connect
    with connect() as env:
        obs = env.reset(seed=3)
        raw = env.hold_action()
        obs, reward, done, info = env.step(raw)

All arrays are plain Python lists / numpy arrays; there is no access to the
simulator itself. The interaction budget is counted by the server.

This module is copied into an agent sandbox as ``harness/client.py``, so it
must keep working when imported as a top-level module.
"""

from __future__ import annotations

import fcntl
import os
import socket
from pathlib import Path

import numpy as np

try:  # inside the package
    from . import wire
except ImportError:  # copied into a sandbox next to harness/wire.py
    import wire  # ty: ignore[unresolved-import]

SANDBOX = Path(os.environ.get("AGENT_SANDBOX", Path(__file__).resolve().parent.parent))


class ServerError(RuntimeError):
    """The server refused a request (budget, seed, camera, bad action)."""


class Env:
    """One claimed environment worker, spoken to over a unix socket."""

    def __init__(self, sock: socket.socket, lock_file):
        """Wrap a connected socket and the lock that reserves the worker.

        Args:
            sock: Socket connected to a worker.
            lock_file: Open file holding the worker's flock, or None.
        """
        self._sock = sock
        self._lock = lock_file
        self.info = self._call({"op": "info"})["info"]
        self.action_dim = int(self.info["action_dim"])

    # -- context management: releases the worker slot
    def __enter__(self):
        """Return self; the worker is already claimed."""
        return self

    def __exit__(self, *exc):
        """Close the socket and release the worker slot."""
        self.close()

    def close(self):
        """Close the socket and release the worker lock."""
        try:
            self._sock.close()
        finally:
            if self._lock is not None:
                fcntl.flock(self._lock, fcntl.LOCK_UN)
                self._lock.close()
                self._lock = None

    def _call(self, msg: dict) -> dict:
        wire.send(self._sock, msg)
        reply = wire.recv(self._sock)
        if not reply.get("ok"):
            raise ServerError(reply.get("error", "unknown server error"))
        return reply

    @staticmethod
    def _obs(o: dict) -> dict:
        return {
            k: (np.asarray(v, dtype=np.float32) if isinstance(v, list) else v)
            for k, v in o.items()
        }

    # -- environment interface
    def reset(self, seed: int) -> dict:
        """Start an episode with a development seed.

        Args:
            seed: Development seed (see docs/api.md for the allowed range; the
                evaluation block is refused).

        Returns:
            The first observation.
        """
        return self._obs(self._call({"op": "reset", "seed": int(seed)})["obs"])

    def step(self, raw_action) -> tuple[dict, float, bool, dict]:
        """Apply one physical action.

        Args:
            raw_action: The physical action (20 or 21 floats, see info()).

        Returns:
            ``(obs, reward, done, info)``.
        """
        a = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        r = self._call({"op": "step", "action": a.tolist()})
        info = {
            "termination": r.get("termination"),
            "budget_used": r.get("budget_used"),
        }
        return self._obs(r["obs"]), float(r["reward"]), bool(r["done"]), info

    def hold_action(self) -> np.ndarray:
        """Return the action that keeps the current targets with zero base velocity."""
        return np.asarray(self._call({"op": "hold_action"})["action"], dtype=np.float32)

    def ik(self, left_pos=None, left_quat=None, right_pos=None, right_quat=None):
        """Return the 14 arm joint targets reaching world-frame wrist targets.

        Args:
            left_pos: World-frame left wrist target, or None to keep it.
            left_quat: Left wrist orientation (w, x, y, z), or None for free.
            right_pos: World-frame right wrist target, or None to keep it.
            right_quat: Right wrist orientation (w, x, y, z), or None for free.

        Returns:
            14 joint angles (left 7, right 7); see docs/api.md.
        """
        msg = {"op": "ik"}
        for k, v in (
            ("left_pos", left_pos),
            ("left_quat", left_quat),
            ("right_pos", right_pos),
            ("right_quat", right_quat),
        ):
            if v is not None:
                msg[k] = np.asarray(v, dtype=np.float64).tolist()
        return np.asarray(self._call(msg)["arm_qpos"], dtype=np.float32)

    def render(self, camera: str = "head", path: str | None = None) -> str:
        """Save a PNG from one of the robot's own cameras under the sandbox.

        The three robot-mounted cameras (head, left_wrist, right_wrist) are the
        only view of the scene there is.

        Args:
            camera: Camera name.
            path: Sandbox-relative destination, or None for a generated one.

        Returns:
            The path of the written PNG.
        """
        msg = {"op": "render", "camera": camera}
        if path is not None:
            msg["path"] = str(path)
        return self._call(msg)["path"]

    def image(
        self, camera: str = "head", width: int = 84, height: int = 84
    ) -> np.ndarray:
        """Return an RGB uint8 array from a robot-mounted camera.

        Args:
            camera: head, left_wrist or right_wrist.
            width: Requested width (capped by the server).
            height: Requested height (capped by the server).

        Returns:
            A ``(height, width, 3)`` uint8 array. Costs no budget.
        """
        import base64
        import io

        from PIL import Image

        r = self._call(
            {
                "op": "image",
                "camera": camera,
                "width": int(width),
                "height": int(height),
            }
        )
        return np.asarray(
            Image.open(io.BytesIO(base64.b64decode(r["png_b64"]))).convert("RGB")
        )

    def pixel_to_ray(self, camera: str, u: float, v: float, width: int, height: int):
        """Return the world-frame ray through a pixel of a rendered image.

        Args:
            camera: Camera the image came from.
            u: Pixel column.
            v: Pixel row.
            width: Width the image was rendered at.
            height: Height the image was rendered at.

        Returns:
            ``(origin, unit direction)``; see docs/api.md.
        """
        try:  # inside the package
            from .geometry import pixel_to_ray
        except ImportError:  # copied into a sandbox next to harness/geometry.py
            from geometry import pixel_to_ray  # ty: ignore[unresolved-import]

        return pixel_to_ray(self.camera_info()[camera], u, v, width, height)

    def camera_info(self) -> dict:
        """Return the calibration of every camera (fovy_deg, world pos and rot)."""
        cams = self._call({"op": "camera_info"})["cameras"]
        return {
            k: {
                kk: (np.asarray(vv, dtype=np.float32) if isinstance(vv, list) else vv)
                for kk, vv in v.items()
            }
            for k, v in cams.items()
        }

    def budget(self) -> dict:
        """Return ``{'used', 'cap', 'remaining'}`` environment steps."""
        return self._call({"op": "budget"})["budget"]


def connect(sandbox: Path | None = None, timeout_s: float = 600.0) -> Env:
    """Claim a free environment worker (blocks until one is free).

    Args:
        sandbox: Sandbox directory holding ``sock/``; defaults to
            ``$AGENT_SANDBOX`` or this file's grandparent.
        timeout_s: How long to wait for a free worker.

    Returns:
        A connected :class:`Env`.

    Raises:
        RuntimeError: No server sockets exist.
        TimeoutError: Every worker stayed busy.
    """
    import time

    sandbox = Path(sandbox or SANDBOX)
    sock_dir = sandbox / "sock"
    socks = sorted(sock_dir.glob("w*.sock"))
    if not socks:
        raise RuntimeError(
            f"no environment server sockets in {sock_dir}; is the server running?"
        )
    t0 = time.time()
    while True:
        for sp in socks:
            lock_path = sock_dir / (sp.stem + ".lock")
            lf = open(lock_path, "a+")
            try:
                fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lf.close()
                continue
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                # sock/ may be a symlink to a short directory (108-byte
                # unix socket path limit): connect on the resolved path.
                s.connect(os.path.realpath(sp))
            except OSError:
                fcntl.flock(lf, fcntl.LOCK_UN)
                lf.close()
                continue
            return Env(s, lf)
        if time.time() - t0 > timeout_s:
            raise TimeoutError("no free environment worker")
        time.sleep(0.5)
