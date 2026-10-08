"""Recording a viser page: a headless browser renders, ffmpeg encodes.

viser has no server-side renderer: ``ClientHandle.get_render`` asks a
connected browser's canvas for a picture. A recording therefore starts a
headless Chrome on the page (unless somebody's browser is already there),
asks it for one picture per video frame and pipes the frames into ffmpeg.
Compare mode's ``--record`` and ``scripts/demo_grid.py`` both record this way.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from bigym.loco.agent.demo_video import ffmpeg_binary, parse_size

# A capture is rendered offscreen at whatever size it is asked for; this is
# the promotional default.
RECORD_SIZE = "1920x1080"
RECORD_CRF = "18"
# ``--browser``: look one of these up, wait for a human, or a given path.
BROWSER_HEADLESS = "headless"
BROWSER_NONE = "none"
BROWSER_NAMES = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
)
# How long a recording waits for somebody's browser before starting its
# own, and how long it then waits for that one to connect.
BROWSER_GRACE_SECONDS = 3.0
BROWSER_START_SECONDS = 60.0


def hide_gui_chrome(server) -> None:
    """Collapse viser's own chrome so a screen recording is clean.

    The logo and the share button go, the page turns dark (a black sky
    wants a dark page behind it) and the control panel is minimised to its
    title bar — the Playback folder is still one click away, it is simply
    not in the shot.

    Args:
        server: The viser server.
    """
    try:
        server.gui.configure_theme(
            show_logo=False, show_share_button=False, dark_mode=True
        )
    except Exception as exc:  # chrome is cosmetic
        print(f"[grid] cannot restyle the viser chrome: {exc}", flush=True)
    try:
        server.gui.main_panel.minimize()
    except Exception as exc:  # older viser has no main panel
        print(f"[grid] cannot minimise the control panel: {exc}", flush=True)


def parse_record_size(spec: str) -> tuple[int, int]:
    """Parse ``--record-size`` (``3840x2160``) into ``(width, height)``.

    Args:
        spec: The ``WxH`` string.

    Returns:
        ``(width, height)``.

    Raises:
        SystemExit: It is not ``WxH`` with two positive integers.
    """
    try:
        return parse_size(str(spec))
    except ValueError as exc:
        raise SystemExit(f"--record-size {spec!r}: {exc}") from None


def ffmpeg_command(
    path, size: tuple[int, int], fps: float, crf: str = RECORD_CRF, binary: str = ""
) -> list[str]:
    """The argv that turns a pipe of raw RGB frames into an H.264 mp4.

    Args:
        path: Destination file.
        size: ``(width, height)`` of every frame.
        fps: Frame rate written into the container.
        crf: x264 quality (18 is visually lossless enough for a slide).
        binary: The ffmpeg to run, or "" to look one up the way
            :mod:`bigym.loco.agent.demo_video` does (the ``imageio-ffmpeg``
            wheel's copy, else an ``ffmpeg`` on ``PATH``).

    Returns:
        The command line.
    """
    if not binary:
        binary = ffmpeg_binary()
    width, height = int(size[0]), int(size[1])
    return [
        binary, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(int(round(float(fps)))), "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]  # fmt: skip


class VideoSink:
    """An ffmpeg subprocess that raw RGB frames are piped into.

    Frames are written as they are rendered rather than held in memory: a
    3840x2160 pass over the longest demonstration is 3853 frames, which is
    90 GB of uint8 if it is kept.
    """

    def __init__(
        self,
        path,
        size: tuple[int, int],
        fps: float,
        crf: str = RECORD_CRF,
        binary: str = "",
    ):
        """Start ffmpeg on ``path``.

        Args:
            path: Destination file (its directory is created).
            size: ``(width, height)`` of every frame.
            fps: Frame rate of the file.
            crf: x264 quality.
            binary: The ffmpeg to run, or "" to look one up.

        Raises:
            SystemExit: yuv420p cannot encode an odd frame size.
        """
        width, height = int(size[0]), int(size[1])
        if width % 2 or height % 2:
            raise SystemExit(
                f"--record-size {width}x{height} has an odd side; H.264 "
                "(yuv420p) needs even width and height"
            )
        self.path = Path(path)
        self.size = (width, height)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.command = ffmpeg_command(self.path, self.size, fps, crf, binary)
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE)
        assert self.process.stdin is not None
        self.pipe = self.process.stdin

    def write(self, frame) -> None:
        """Write one ``(H, W, 3)`` uint8 frame."""
        self.pipe.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        """Close the pipe and wait for ffmpeg.

        Raises:
            RuntimeError: ffmpeg exited non-zero.
        """
        with contextlib.suppress(Exception):
            self.pipe.close()
        status = self.process.wait(timeout=600)
        if status != 0:
            raise RuntimeError(
                f"ffmpeg exited with status {status} writing {self.path}"
            )


def browser_binary(spec: str = BROWSER_HEADLESS) -> str:
    """Which browser a recording should drive, if any.

    Args:
        spec: ``headless`` to look one up, ``none`` to wait for a human, or
            the path of a Chrome/Chromium binary.

    Returns:
        The binary to run, or "" for ``none``.

    Raises:
        SystemExit: ``headless`` was asked for and no Chrome/Chromium is on
            ``PATH``, or the given path is not executable.
    """
    name = str(spec or BROWSER_HEADLESS).strip()
    if name == BROWSER_NONE:
        return ""
    if name != BROWSER_HEADLESS:
        found = shutil.which(name)
        if not found:
            raise SystemExit(f"--browser {name!r}: not an executable on PATH")
        return found
    for candidate in BROWSER_NAMES:
        found = shutil.which(candidate)
        if found:
            return found
    raise SystemExit(
        "--browser headless: none of "
        + ", ".join(BROWSER_NAMES)
        + " is on PATH; install one, pass --browser <path>, or use "
        "--browser none and open the URL yourself"
    )


def chrome_command(binary: str, url: str, size: tuple[int, int]) -> list[str]:
    """The argv that opens a headless Chrome on the viser page.

    SwiftShader rather than a real GPU: the machine that runs the grid is
    already holding twenty MuJoCo scenes, the page is a handful of meshes,
    and a software rasteriser needs no display, no X server and no GPU
    sandbox. The window does NOT cap the capture — ``get_render`` renders
    offscreen at whatever size it is asked for, measured here: a 1280x720
    headless window returns a true 3840x2160 frame — so the window is
    :func:`browser_window`, the recording's aspect ratio at a size a
    software rasteriser is happy with.

    Args:
        binary: The Chrome/Chromium executable.
        url: The viser page.
        size: ``(width, height)`` of the window.

    Returns:
        The command line.
    """
    width, height = int(size[0]), int(size[1])
    return [
        binary, "--headless=new", "--no-sandbox", "--disable-gpu-sandbox",
        "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
        f"--window-size={width},{height}", "--remote-debugging-port=0", url,
    ]  # fmt: skip


def browser_window(size: tuple[int, int], cap=(1920, 1080)) -> tuple[int, int]:
    """The headless window for a recording of ``size``.

    ``get_render`` is not limited by the window (a 1280x720 window returns
    a 3840x2160 frame), so the window only has to carry the *aspect ratio*
    of the recording — the page lays out and the camera frames against it.
    It is halved until it fits ``cap``, which keeps the ratio exact and
    keeps a software rasteriser out of pointless 4K compositing.

    Args:
        size: The recording ``(width, height)``.
        cap: The largest window to open.

    Returns:
        The window ``(width, height)``.
    """
    width, height = max(2, int(size[0])), max(2, int(size[1]))
    while width > int(cap[0]) or height > int(cap[1]):
        width, height = max(2, width // 2), max(2, height // 2)
    return (width, height)


def launch_browser(binary: str, url: str, size: tuple[int, int]):
    """Start a headless browser on the page; it is the renderer.

    Args:
        binary: The Chrome/Chromium executable.
        url: The viser page.
        size: ``(width, height)`` of the window.

    Returns:
        The process, or None when it could not be started.
    """
    command = chrome_command(binary, url, size)
    print(f"[grid] starting a headless browser: {Path(binary).name} on {url}")
    try:
        return subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except OSError as exc:
        print(f"[grid] cannot start {binary}: {exc}", flush=True)
        return None


def stop_browser(process) -> None:
    """Close the headless browser once the recording is written."""
    if process is None or process.poll() is not None:
        return
    with contextlib.suppress(OSError):
        process.terminate()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(OSError):
            process.kill()
        with contextlib.suppress(subprocess.TimeoutExpired, OSError):
            process.wait(timeout=5.0)
    print("[grid] headless browser closed", flush=True)


def rgb_frame(image) -> np.ndarray:
    """One render as ``(H, W, 3)`` uint8.

    ``ClientHandle.get_render`` returns ``(H, W, 3)`` RGB for the default
    jpeg transport and ``(H, W, 4)`` RGBA for png; rawvideo takes rgb24
    either way, so an alpha channel is dropped here.

    Args:
        image: What the client returned.

    Returns:
        The RGB frame.
    """
    frame = np.asarray(image)
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]
    return np.ascontiguousarray(frame, dtype=np.uint8)


def wait_for_client(server, *, timeout: float = 0.0, poll: float = 0.25):
    """Block until a browser is connected; it is what renders the frames.

    viser has no server-side renderer: ``get_render`` asks a connected
    client's WebGL canvas for a picture. A recording therefore cannot start
    before somebody opens the page.

    Args:
        server: The viser server.
        timeout: Give up after this many seconds (0 = wait for ever).
        poll: Seconds between checks.

    Returns:
        The most recently connected client, or None when ``timeout``
        passed with nobody there.
    """
    print("[grid] waiting for a browser to connect for recording…", flush=True)
    started = time.time()
    # The first "still waiting" line is due ten seconds in, not at once.
    said = started
    while True:
        try:
            clients = list(server.get_clients().values())
        except Exception:  # a closed server is "no clients"
            clients = []
        if clients:
            print(
                f"[grid] browser connected: recording from client {clients[-1].client_id}"
            )
            return clients[-1]
        now = time.time()
        if timeout > 0.0 and now - started > timeout:
            print(f"[grid] no browser connected within {timeout:g}s", flush=True)
            return None
        if now - said >= 10.0:
            said = now
            print("[grid] still waiting for a browser…", flush=True)
        time.sleep(poll)
