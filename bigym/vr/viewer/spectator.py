"""Desktop spectator views for VR demo collection.

The VR collector renders only into the headset; everything on the desk is
blind. This module adds live spectator views of the same physics, meant to be
synced from the collection loop (same thread that steps the env):

- ``MujocoGuiSpectator``: a native interactive MuJoCo window
  (``mujoco.viewer.launch_passive``) running in a SEPARATE PROCESS on a
  saved copy of the model, fed qpos over shared memory (seqlock, ~30 Hz).
  Full GUI: orbit/zoom, contact visualization (the child recomputes
  kinematics + collision), and camera cycling with the ``[`` / ``]`` keys —
  including the robot's fixed cameras (``head`` = the collector's
  first-person view, plus the wrists). Out-of-process is REQUIRED, not an
  optimization: the OpenXR session's GL context provider is GLFW-based
  (xr_context.py), and GLFW's X11 error handler is process-global — a
  second in-process GLFW user (launch_passive's thread) asserts with
  `_glfwGrabErrorHandlerX11: _glfw.x11.errorHandler == NULL` as soon as
  both touch X11.
- ``ViserSpectator``: a web viewer (default http://localhost:8080) running in
  a separate process, with a live 3D mirror, simultaneous head/wrist/follow
  camera streams, and session status. Camera images use a MuJoCo renderer in
  that child process while the main canvas remains Viser. Each stream has one
  WebUI checkbox; disabled streams are hidden and skipped. The collector only
  snapshots state to shared memory.

Both are strictly read-only observers: they never mutate the live qpos/ctrl
and drop to a disabled no-op state on failure (no display, closed window)
instead of killing the collection session. ``make_spectators`` builds the
requested set from a mode string; ``SpectatorSet.sync()`` is the single
per-step hook.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import Any, Callable

import mujoco
import numpy as np
import tyro

from bigym.vr.viewer.mjviser_compat import find_dynamic_body_ids, open_viser_scene


def _find_camera_id(model: Any, name: str) -> int:
    """Resolve a camera by bare name, tolerating MJCF scope prefixes."""
    for cam_id in range(model.ncam):
        full = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id) or ""
        if full == name or full.split("/")[-1] == name:
            return cam_id
    return -1


class MujocoGuiSpectator:
    """Interactive MuJoCo GUI window in a child process (see module doc)."""

    _MAX_SYNC_HZ = 30.0
    _HEADER = 2  # int64 slots: [seq, stop]

    def __init__(self, inner_env: Any) -> None:
        """Save the model, map the shared qpos buffer and spawn the child."""
        self._proc = None
        self._shm = None
        self._last_sync = 0.0
        self._model_path: str | None = None
        model, data = inner_env.model, inner_env.data
        self._live_data = data
        nq = int(model.nq)
        try:
            self._model_path = os.path.join(
                tempfile.mkdtemp(prefix="bigym_spectator_"), "scene.mjb"
            )
            mujoco.mj_saveModel(model, self._model_path, None)
            self._shm = shared_memory.SharedMemory(
                create=True, size=8 * (self._HEADER + nq)
            )
            self._head = np.ndarray(
                (self._HEADER,), dtype=np.int64, buffer=self._shm.buf
            )
            self._qpos = np.ndarray(
                (nq,), dtype=np.float64, buffer=self._shm.buf, offset=8 * self._HEADER
            )
            self._head[:] = 0
            self._qpos[:] = data.qpos
            lookat = ",".join(
                f"{v:.3f}" for v in data.bind(inner_env.robot.pelvis).xpos
            )
            self._proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "bigym.vr.viewer.spectator",
                    "--model",
                    self._model_path,
                    "--shm",
                    self._shm.name,
                    "--nq",
                    str(nq),
                    "--parent-pid",
                    str(os.getpid()),
                    "--lookat",
                    lookat,
                ],
                stdin=subprocess.DEVNULL,
            )
        except Exception as exc:
            logging.warning("MuJoCo GUI spectator unavailable: %s", exc)
            self._cleanup()
            return
        print(
            "[spectator] MuJoCo GUI window opening in a child process "
            "([ / ] cycles cameras incl. the first-person 'head' view)"
        )

    def sync(self) -> None:
        """Publish the live qpos to the child (rate-limited seqlock write)."""
        if self._proc is None:
            return
        now = time.monotonic()
        if now - self._last_sync < 1.0 / self._MAX_SYNC_HZ:
            return
        self._last_sync = now
        if self._proc.poll() is not None:
            print("[spectator] MuJoCo GUI window closed")
            self._cleanup()
            return
        # seqlock write: odd = in progress, even = consistent
        self._head[0] += 1
        np.copyto(self._qpos, self._live_data.qpos)
        self._head[0] += 1

    def close(self) -> None:
        """Ask the child to exit, wait for it, and release the shared memory."""
        if self._shm is not None:
            try:
                self._head[1] = 1  # ask the child to exit
            except Exception:
                pass
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.wait(timeout=2.0)
            except Exception:
                self._proc.terminate()
        self._cleanup()

    def _cleanup(self) -> None:
        self._proc = None
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass
            self._shm = None
        path = self._model_path
        if path:
            shutil.rmtree(os.path.dirname(path), ignore_errors=True)
            self._model_path = None


@dataclass
class _MujocoViewerProcessArgs:
    """The MuJoCo GUI child process's command line (set by the parent)."""

    model: str
    """Compiled model file (.mjb)."""
    shm: str
    """Shared-memory block the parent writes qpos into."""
    nq: int
    """Length of qpos."""
    parent_pid: int
    """Exit when this process is gone."""
    lookat: str = "0,0,0.8"
    """Initial camera look-at point, x,y,z."""


def run_mujoco_viewer_process(argv: list[str]) -> int:
    """Child-process entry: passive viewer fed by the parent's seqlock shm."""
    import mujoco.viewer

    args = tyro.cli(
        _MujocoViewerProcessArgs, args=argv, prog="bigym.vr.viewer.spectator"
    )

    model = mujoco.MjModel.from_binary_path(args.model)
    data = mujoco.MjData(model)
    shm = shared_memory.SharedMemory(name=args.shm)
    try:
        # The attaching side must not own the block: Python's resource
        # tracker otherwise "cleans up" (and warns about) the parent's shm
        # at child exit (cpython bpo-39959).
        from multiprocessing import resource_tracker

        resource_tracker.unregister(
            shm._name,  # ty: ignore[unresolved-attribute]
            "shared_memory",
        )
    except Exception:
        pass
    head = np.ndarray((2,), dtype=np.int64, buffer=shm.buf)
    qpos_shared = np.ndarray((args.nq,), dtype=np.float64, buffer=shm.buf, offset=16)
    qpos = np.array(qpos_shared)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    def parent_alive() -> bool:
        # We are the collector's direct child, so its death reparents us and
        # getppid() stops matching. os.kill(pid, 0) is NOT a valid liveness
        # probe here: once the pid is recycled by any same-user process it
        # reports alive forever.
        return os.getppid() == args.parent_pid

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as handle:
        try:
            handle.cam.lookat[:] = [float(v) for v in args.lookat.split(",")]
            handle.cam.distance = 2.5
            handle.cam.elevation = -15
        except Exception:
            pass
        seen = -1
        while handle.is_running() and not head[1] and parent_alive():
            seq = int(head[0])
            if seq != seen and seq % 2 == 0:
                np.copyto(qpos, qpos_shared)
                if int(head[0]) == seq:  # seqlock read validated
                    seen = seq
                    data.qpos[:] = qpos
                    # positions + collision so the GUI's contact
                    # visualization stays meaningful; no dynamics solve.
                    mujoco.mj_fwdPosition(model, data)
                    handle.sync()
            time.sleep(1.0 / 60.0)
    shm.close()
    return 0


class ViserSpectator:
    """Web spectator isolated in a child process."""

    _HEADER = 3  # int64 slots: [seq, stop, status_len]
    _STATUS_BYTES = 16 * 1024

    def __init__(
        self,
        inner_env: Any,
        *,
        camera_names: tuple[str, ...] = ("head", "left_wrist", "right_wrist"),
        follow_view: bool = True,
        width: int = 480,
        height: int = 360,
        hz: float = 15.0,
        status_fn: Callable[[], dict[str, Any]] | None = None,
        port: int = 8080,
        egl_device_id: int | None = None,
    ) -> None:
        """Allocate the shared state buffers and spawn the viser child."""
        self._proc = None
        self._shm = None
        self._status_fn = status_fn
        self._model_path: str | None = None
        model, data = inner_env.model, inner_env.data
        self._live_model, self._live_data = model, data
        nq, nbody = int(model.nq), int(model.nbody)
        try:
            self._model_path = os.path.join(
                tempfile.mkdtemp(prefix="bigym_viser_"), "scene.mjb"
            )
            mujoco.mj_saveModel(model, self._model_path, None)
            state_values = nq + nbody * 7
            self._shm = shared_memory.SharedMemory(
                create=True,
                size=8 * (self._HEADER + state_values) + self._STATUS_BYTES,
            )
            self._head = np.ndarray(
                (self._HEADER,), dtype=np.int64, buffer=self._shm.buf
            )
            offset = 8 * self._HEADER
            self._qpos = np.ndarray(
                (nq,), dtype=np.float64, buffer=self._shm.buf, offset=offset
            )
            offset += 8 * nq
            self._body_pos = np.ndarray(
                model.body_pos.shape,
                dtype=np.float64,
                buffer=self._shm.buf,
                offset=offset,
            )
            offset += model.body_pos.nbytes
            self._body_quat = np.ndarray(
                model.body_quat.shape,
                dtype=np.float64,
                buffer=self._shm.buf,
                offset=offset,
            )
            offset += model.body_quat.nbytes
            self._status_bytes = np.ndarray(
                (self._STATUS_BYTES,),
                dtype=np.uint8,
                buffer=self._shm.buf,
                offset=offset,
            )
            self._head[:] = 0
            self._qpos[:] = data.qpos
            self._body_pos[:] = model.body_pos
            self._body_quat[:] = model.body_quat

            pelvis_bid = model.bind(inner_env.robot.pelvis).id
            cmd = [
                sys.executable,
                "-m",
                "bigym.vr.viewer.spectator",
                "viser",
                "--model",
                self._model_path,
                "--shm",
                self._shm.name,
                "--nq",
                str(nq),
                "--nbody",
                str(nbody),
                "--status-bytes",
                str(self._STATUS_BYTES),
                "--parent-pid",
                str(os.getpid()),
                "--port",
                str(int(port)),
                "--hz",
                str(max(float(hz), 0.5)),
                "--width",
                str(int(width)),
                "--height",
                str(int(height)),
                "--pelvis-bid",
                str(pelvis_bid),
            ]
            if camera_names:
                cmd += ["--camera", *camera_names]
            if follow_view:
                cmd.append("--follow")
            if status_fn is not None:
                cmd.append("--status")
            dynamic_body_ids = find_dynamic_body_ids(model, inner_env)
            if dynamic_body_ids:
                cmd.extend(
                    (
                        "--dynamic-body-ids",
                        ",".join(map(str, dynamic_body_ids)),
                    )
                )
            child_env = None
            if egl_device_id is not None:
                # Pin the child's offscreen camera renders to another GPU so
                # they stop competing with the XR compositor / stream encoder
                # on the display GPU. Only the EGL backend can select a
                # device, so force it alongside the id. The index is the EGL
                # device enumeration order (typically matches nvidia-smi).
                child_env = dict(os.environ)
                child_env["MUJOCO_GL"] = "egl"
                child_env["MUJOCO_EGL_DEVICE_ID"] = str(int(egl_device_id))
            self._proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, env=child_env)
        except Exception as exc:
            logging.warning("viser spectator unavailable: %s", exc)
            self._cleanup()
            return
        print(
            f"[spectator] viser child starting at http://localhost:{port} "
            "(simultaneous camera streams with WebUI toggles)"
        )

    def sync(self) -> None:
        """Publish qpos, movable body poses and the status dict to the child."""
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            print("[spectator] viser child stopped")
            self._cleanup()
            return
        status = self._status_fn() if self._status_fn is not None else {}
        status_raw = json.dumps(status).encode("utf-8")[: self._STATUS_BYTES]
        self._head[0] += 1
        np.copyto(self._qpos, self._live_data.qpos)
        np.copyto(self._body_pos, self._live_model.body_pos)
        np.copyto(self._body_quat, self._live_model.body_quat)
        self._status_bytes[: len(status_raw)] = np.frombuffer(
            status_raw, dtype=np.uint8
        )
        self._head[2] = len(status_raw)
        self._head[0] += 1

    def close(self) -> None:
        """Ask the child to exit, wait for it, and release the shared memory."""
        if self._shm is not None:
            self._head[1] = 1
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.wait(timeout=2.0)
            except Exception:
                self._proc.terminate()
        self._cleanup()

    def _cleanup(self) -> None:
        self._proc = None
        for name in ("_head", "_qpos", "_body_pos", "_body_quat", "_status_bytes"):
            setattr(self, name, None)
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass
            self._shm = None
        path = self._model_path
        if path:
            shutil.rmtree(os.path.dirname(path), ignore_errors=True)
            self._model_path = None


@dataclass
class _ViserViewerProcessArgs:
    """The Viser child process's command line (set by the parent)."""

    model: str
    """Compiled model file (.mjb)."""
    shm: str
    """Shared-memory block the parent writes qpos, body poses and status into."""
    nq: int
    """Length of qpos."""
    nbody: int
    """Number of bodies."""
    status_bytes: int
    """Size of the status text slot."""
    parent_pid: int
    """Exit when this process is gone."""
    port: int
    """Viser HTTP port."""
    hz: float
    """Update rate."""
    width: int
    """Camera view width."""
    height: int
    """Camera view height."""
    pelvis_bid: int = -1
    """Body the follow view tracks; -1 for none."""
    camera: list[str] = field(default_factory=list)
    """Cameras to stream to the browser."""
    follow: bool = False
    """Add a view that follows the pelvis."""
    status: bool = False
    """Show the parent's status text."""
    dynamic_body_ids: str = ""
    """Comma-separated ids of bodies whose poses the parent streams."""


def run_viser_viewer_process(argv: list[str]) -> int:
    """Child-process entry for the Viser scene and browser camera views."""
    args = tyro.cli(
        _ViserViewerProcessArgs, args=argv, prog="bigym.vr.viewer.spectator viser"
    )

    model = mujoco.MjModel.from_binary_path(args.model)
    data = mujoco.MjData(model)
    shm = shared_memory.SharedMemory(name=args.shm)
    try:
        from multiprocessing import resource_tracker

        resource_tracker.unregister(
            shm._name,  # ty: ignore[unresolved-attribute]
            "shared_memory",
        )
    except Exception:
        pass

    head = np.ndarray((3,), dtype=np.int64, buffer=shm.buf)
    offset = 24
    qpos_shared = np.ndarray(
        (args.nq,), dtype=np.float64, buffer=shm.buf, offset=offset
    )
    offset += 8 * args.nq
    body_pos_shared = np.ndarray(
        (args.nbody, 3), dtype=np.float64, buffer=shm.buf, offset=offset
    )
    offset += 8 * args.nbody * 3
    body_quat_shared = np.ndarray(
        (args.nbody, 4), dtype=np.float64, buffer=shm.buf, offset=offset
    )
    offset += 8 * args.nbody * 4
    status_shared = np.ndarray(
        (args.status_bytes,), dtype=np.uint8, buffer=shm.buf, offset=offset
    )

    dynamic_body_ids = tuple(
        int(value) for value in args.dynamic_body_ids.split(",") if value
    )
    server, scene = open_viser_scene(
        model,
        None,
        port=args.port,
        label="vr-spectator",
        dynamic_body_ids=dynamic_body_ids,
    )

    views: list[tuple[str, int]] = []
    for name in args.camera:
        cam_id = _find_camera_id(model, name)
        if cam_id >= 0:
            views.append((name.replace("_", " ").title(), cam_id))
        else:
            logging.warning("viser spectator: no camera named %r", name)
    if args.follow:
        views.append(("Follow", -1))

    blank = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    toggles: dict[str, Any] = {}
    images: dict[str, Any] = {}
    with server.gui.add_folder("Cameras"):
        for label, _ in views:
            toggle = server.gui.add_checkbox(label, initial_value=True)
            image = server.gui.add_image(blank, label=None, format="jpeg")
            toggles[label] = toggle
            images[label] = image

            def set_visible(event, *, view_label=label) -> None:
                images[view_label].visible = bool(event.target.value)

            toggle.on_update(set_visible)
    if args.status:
        with server.gui.add_folder("Session"):
            status_handle = server.gui.add_markdown("(waiting for first sync)")
    else:
        status_handle = None

    renderer = mujoco.Renderer(model, args.height, args.width) if views else None
    follow_cam = mujoco.MjvCamera()
    follow_cam.distance = 2.5
    follow_cam.elevation = -15
    follow_cam.azimuth = 160

    qpos = np.array(qpos_shared)
    body_pos = np.array(body_pos_shared)
    body_quat = np.array(body_quat_shared)
    period = 1.0 / args.hz
    seen = -1
    print(
        f"[spectator] viser at http://localhost:{args.port} "
        f"(isolated process, {len(views)} MuJoCo camera streams)",
        flush=True,
    )
    try:
        while not head[1] and os.getppid() == args.parent_pid:
            started = time.monotonic()
            seq = int(head[0])
            if seq != seen and seq % 2 == 0:
                np.copyto(qpos, qpos_shared)
                np.copyto(body_pos, body_pos_shared)
                np.copyto(body_quat, body_quat_shared)
                status_len = int(head[2])
                status_raw = bytes(status_shared[:status_len])
                if int(head[0]) == seq:
                    seen = seq
                    data.qpos[:] = qpos
                    model.body_pos[:] = body_pos
                    model.body_quat[:] = body_quat
                    mujoco.mj_kinematics(model, data)
                    mujoco.mj_camlight(model, data)
                    for label, cam_id in views:
                        if not toggles[label].value:
                            continue
                        assert renderer is not None
                        if cam_id >= 0:
                            renderer.update_scene(data, camera=cam_id)
                        else:
                            if args.pelvis_bid >= 0:
                                follow_cam.lookat[:] = data.xpos[args.pelvis_bid]
                            renderer.update_scene(data, camera=follow_cam)
                        images[label].image = renderer.render()
                    scene.update_from_mjdata(data)
                    if status_handle is not None and status_raw:
                        status = json.loads(status_raw)
                        status_handle.content = "\n".join(
                            f"- **{key}**: {value}" for key, value in status.items()
                        )
            delay = period - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
    finally:
        if renderer is not None:
            renderer.close()
        server.stop()
        del head, qpos_shared, body_pos_shared, body_quat_shared, status_shared
        shm.close()
    return 0


class SpectatorSet:
    """The collection loop's single handle over any number of spectators."""

    def __init__(self, spectators: list[Any]) -> None:
        """Hold the spectator objects the collection loop drives."""
        self._spectators = spectators

    def sync(self) -> None:
        """Sync every spectator once."""
        for s in self._spectators:
            s.sync()

    def close(self) -> None:
        """Close every spectator."""
        for s in self._spectators:
            s.close()


def make_spectators(
    inner_env: Any,
    mode: str,
    *,
    camera_names: tuple[str, ...] = ("head", "left_wrist", "right_wrist"),
    hz: float = 15.0,
    status_fn: Callable[[], dict[str, Any]] | None = None,
    viser_port: int = 8080,
    viser_egl_device_id: int | None = None,
) -> SpectatorSet:
    """Build spectators for ``mode`` in {none, mujoco, viser, both}."""
    mode = (mode or "none").lower()
    spectators: list[Any] = []
    if mode in ("mujoco", "both"):
        if viser_egl_device_id is not None:
            # The interactive window renders through GLX on whichever GPU
            # drives the X screen; there is no per-process device override.
            print(
                "[spectator] note: the MuJoCo GUI window always renders on "
                "the X display GPU; the GPU pin only moves viser's camera "
                "renders"
            )
        spectators.append(MujocoGuiSpectator(inner_env))
    if mode in ("viser", "both"):
        spectators.append(
            ViserSpectator(
                inner_env,
                camera_names=camera_names,
                hz=hz,
                status_fn=status_fn,
                port=viser_port,
                egl_device_id=viser_egl_device_id,
            )
        )
    return SpectatorSet(spectators)


if __name__ == "__main__":
    if sys.argv[1:2] == ["viser"]:
        sys.exit(run_viser_viewer_process(sys.argv[2:]))
    sys.exit(run_mujoco_viewer_process(sys.argv[1:]))
