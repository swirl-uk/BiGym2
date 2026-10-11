"""Recording a compare view from the browser."""

from __future__ import annotations

import time
from typing import cast

import numpy as np

from bigym.vr.viewer import video
from bigym.vr.viewer.layout import compare_camera_pose

RECORD_LOG_EVERY = 50


COMPARE_PAD = 1.3
"""Metres a compare slot's content (a table, a robot, the plates) reaches
from its origin, for the recording lens."""


COMPARE_BOX_CENTRE = (0.4, 0.0, 0.9)
"""Where the middle of one slot's content is, relative to its origin."""


COMPARE_BOX_HEIGHT = 1.0
"""Half height of that content."""


COMPARE_PAD_TIGHT = 0.9
"""The pad for the oblique view, which is about the robots and their tables
rather than the whole grid."""


SETTLE_TIMEOUT = 30.0


RECORD_VIEWS = ("overview", "oblique")


RECORD_LABEL_SCALE = 1.5
"""How much bigger the billboards are drawn for a recording than live: a
video is watched from further away than a browser tab."""


OBLIQUE_AZIMUTH = 35.0
"""Degrees the oblique camera stands off the robots' front (+x) towards +y."""


OBLIQUE_ELEVATION = 24.0
"""Degrees above the horizon the oblique camera looks down from."""


RECORD_FOV_MIN = 0.18


RECORD_FOV_MAX = 1.40


def compare_box(
    offsets, pad: float = COMPARE_PAD
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """The box a compare grid's content lives in: ``(centre, half extents)``.

    Args:
        offsets: Every slot's offset.
        pad: Metres a slot's content reaches from its origin.

    Returns:
        ``((cx, cy, cz), (hx, hy, hz))`` in world metres.
    """
    places = [tuple(float(v) for v in o) for o in offsets] or [(0.0, 0.0, 0.0)]
    xs = [o[0] for o in places]
    ys = [o[1] for o in places]
    centre = (
        0.5 * (min(xs) + max(xs)) + COMPARE_BOX_CENTRE[0],
        0.5 * (min(ys) + max(ys)) + COMPARE_BOX_CENTRE[1],
        COMPARE_BOX_CENTRE[2],
    )
    half = (
        0.5 * (max(xs) - min(xs)) + float(pad),
        0.5 * (max(ys) - min(ys)) + float(pad),
        COMPARE_BOX_HEIGHT,
    )
    return centre, half


def compare_record_pose(
    offsets, view: str = "overview", *, home=None
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Where a recording camera stands for ``view``.

    ``overview`` is the live compare's starting camera: behind the robots,
    high, every scene in view. ``oblique`` stands in front of them and to
    one side, lower and closer — the three-quarter view that shows the
    hands and what is on the tables rather than four backs.

    Args:
        offsets: Every slot's offset.
        view: One of :data:`RECORD_VIEWS`.
        home: The live overview pose, when the caller already has it.

    Returns:
        ``(position, look_at)`` in world metres.
    """
    if view not in RECORD_VIEWS:
        raise ValueError(f"record view must be one of {RECORD_VIEWS}, not {view!r}")
    if view == "overview":
        return home if home is not None else compare_camera_pose(offsets)
    centre, half = compare_box(offsets, COMPARE_PAD_TIGHT)
    azimuth = np.radians(OBLIQUE_AZIMUTH)
    elevation = np.radians(OBLIQUE_ELEVATION)
    direction = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    distance = 2.0 * max(half[0], half[1]) + 2.0
    position = np.asarray(centre, dtype=float) + distance * direction
    return cast(
        tuple[tuple[float, float, float], tuple[float, float, float]],
        (tuple(float(v) for v in position), tuple(float(v) for v in centre)),
    )


def box_fov(
    position, look_at, box, aspect: float = 16 / 9, margin: float = 1.05
) -> float:
    """The vertical field of view that fits ``box`` in the frame.

    The box's eight corners are projected into the camera's frame and the
    lens opened just enough, vertically and horizontally (through the
    aspect ratio), to hold the widest of them, so the framing is exact for
    any camera position rather than for one side of the grid.

    Args:
        position: Where the camera is.
        look_at: What it looks at.
        box: ``(centre, half extents)`` in world metres.
        aspect: Frame width over height.
        margin: Room to leave around the box (1.05 = 5%).

    Returns:
        The vertical field of view in radians, clamped.
    """
    eye = np.asarray(position, dtype=float)
    forward = np.asarray(look_at, dtype=float) - eye
    norm = float(np.linalg.norm(forward))
    if norm < 1e-6:
        return RECORD_FOV_MAX
    forward = forward / norm
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ world_up)) > 0.99:
        world_up = np.array([1.0, 0.0, 0.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    centre, half = box
    centre = np.asarray(centre, dtype=float)
    half = np.asarray(half, dtype=float)
    vertical = 0.0
    horizontal = 0.0
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corner = centre + half * np.array([sx, sy, sz]) - eye
                depth = max(0.2, float(corner @ forward))
                vertical = max(vertical, abs(float(corner @ up)) / depth)
                horizontal = max(horizontal, abs(float(corner @ right)) / depth)
    fov_v = 2.0 * np.arctan(margin * vertical)
    fov_h = 2.0 * np.arctan(margin * horizontal)
    from_width = 2.0 * np.arctan(np.tan(0.5 * fov_h) / max(0.1, float(aspect)))
    return float(np.clip(max(fov_v, from_width), RECORD_FOV_MIN, RECORD_FOV_MAX))


def settle_render(
    client, size: tuple[int, int], *, timeout: float = SETTLE_TIMEOUT
) -> int:
    """Render until two pictures in a row are identical; the page has loaded.

    A browser that has just connected is still downloading meshes, and a
    frame asked for too soon shows robots with no tables. The scene is
    static until the playhead moves, so once two consecutive renders match
    everything that is going to appear has appeared.

    Args:
        client: Anything with ``get_render(height, width)``.
        size: ``(width, height)`` to render at.
        timeout: Give up (and record anyway) after this many seconds.

    Returns:
        How many renders it took.
    """
    width, height = int(size[0]), int(size[1])
    previous = None
    renders = 0
    started = time.time()
    while True:
        frame = np.asarray(client.get_render(height, width))
        renders += 1
        if previous is not None and previous.shape == frame.shape:
            if np.array_equal(previous, frame):
                return renders
        previous = frame
        if time.time() - started > timeout:
            print(
                f"[viewer] compare: the page kept changing for {timeout:g}s; "
                "recording anyway",
                flush=True,
            )
            return renders


def compare_record_playheads(
    seconds: float, scene_fps: float, file_fps: float, length: int, speed: float = 1.0
) -> list[int]:
    """The playhead positions a compare recording steps through.

    One video frame per file frame; the playhead moves ``scene_fps /
    file_fps * speed`` episode steps between frames, so the file plays the
    episodes at ``speed`` times their recorded speed whatever rate it is
    written at, and wraps around when the episodes end.

    Args:
        seconds: Length of the video.
        scene_fps: The rate the episodes were recorded at.
        file_fps: The rate the file is written at.
        length: Frames in the longest slot.
        speed: Playback speed (2 = twice real time).

    Returns:
        The playheads, one per video frame.
    """
    count = max(1, int(round(float(seconds) * float(file_fps))))
    step = float(scene_fps) / max(1e-6, float(file_fps)) * max(1e-6, float(speed))
    span = max(1, int(length))
    return [int(round(i * step)) % span for i in range(count)]


def record_compare(
    show,
    client,
    make_sink,
    *,
    playheads,
    size: tuple[int, int],
    log_every: int = RECORD_LOG_EVERY,
) -> tuple[int, tuple[int, int]]:
    """Pose the scene at each playhead, render it in the browser, write it.

    No wall clock: the scene is posed, only then is the client asked for a
    picture, so the file is smooth at its frame rate however slowly the
    software rasteriser draws. The first render decides the real size (a
    human's browser may not return what was asked for) and ``make_sink``
    is called with that.

    Args:
        show: Called with a playhead to pose every slot at it.
        client: Anything with ``get_render(height, width)``.
        make_sink: Called once with ``(width, height)``; returns an object
            with ``write(frame)`` and ``close()``.
        playheads: The playheads to record, in order.
        size: The requested ``(width, height)``.
        log_every: Print progress every this many frames (0 = never).

    Returns:
        ``(frames written, the size they were written at)``.

    Raises:
        RuntimeError: The browser changed the frame size mid-recording.
    """
    width, height = int(size[0]), int(size[1])
    playheads = list(playheads)
    sink = None
    actual = (width, height)
    written = 0
    started = time.time()
    try:
        for index, playhead in enumerate(playheads):
            show(int(playhead))
            frame = np.asarray(client.get_render(height, width))
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[..., :3]
            frame = np.ascontiguousarray(frame.astype(np.uint8, copy=False))
            got = (int(frame.shape[1]), int(frame.shape[0]))
            if sink is None:
                actual = got
                if got != (width, height):
                    print(
                        f"[viewer] the browser returned {got[0]}x{got[1]}, not "
                        f"the requested {width}x{height}; recording at that",
                        flush=True,
                    )
                sink = make_sink(actual)
            elif got != actual:
                raise RuntimeError(
                    f"frame {index} came back {got[0]}x{got[1]} after "
                    f"{actual[0]}x{actual[1]}: do not resize the browser "
                    "window while recording"
                )
            sink.write(frame)
            written += 1
            if log_every and written % int(log_every) == 0:
                rate = written / max(1e-6, time.time() - started)
                print(
                    f"[viewer] recorded {written}/{len(playheads)} frames "
                    f"({rate:.1f} fps captured)",
                    flush=True,
                )
    finally:
        if sink is not None:
            sink.close()
    return written, actual


def record_compare_session(
    server,
    port: int,
    show,
    home,
    *,
    scene_fps: float,
    length: int,
    path,
    offsets=(),
    seconds: float = 10.0,
    size=(1920, 1080),
    fps: float = 0.0,
    browser: str = "headless",
    stay: bool = False,
    view: str = "overview",
    zoom: float = 1.0,
    speed: float = 1.0,
    start_timeout: float = 60.0,
) -> int:
    """Write a video of a built compare scene through a browser.

    viser has no server-side renderer: a headless Chrome is started on the
    page (or a browser somebody already has open is used), parked at the
    overview camera, and asked for one picture per video frame while the
    playhead is stepped by hand (:mod:`bigym.vr.viewer.video`).

    Args:
        server: The viser server the scene is in.
        port: Its port.
        show: Called with a playhead to pose every slot at it.
        home: The overview ``(position, look_at)`` the camera records from.
        scene_fps: The rate the episodes were recorded at.
        length: Frames in the longest slot.
        path: Destination mp4.
        offsets: Every slot's offset; the lens is narrowed to what they
            subtend from ``home`` (viser's default 80 degrees leaves four
            scenes in a small band in the middle of a 16:9 frame).
        seconds: Length of the video.
        size: ``(width, height)`` of the frames.
        fps: Frame rate of the file (0 = ``scene_fps``, real time).
        browser: ``headless``, ``none`` (wait for a person's browser) or a
            browser executable.
        stay: Unused here; :func:`~bigym.vr.viewer.agent.compare.run.run_compare`
            reads it.
        view: ``overview`` or ``oblique`` (:func:`compare_record_pose`).
        zoom: Narrows the lens (2 = twice as close-looking).
        speed: Playback speed (2 = twice real time).
        start_timeout: Seconds to wait for a browser to connect.

    Returns:
        The number of frames written (0 when no browser turned up).
    """
    del stay
    file_fps = float(fps) if float(fps) > 0.0 else float(scene_fps)
    width, height = int(size[0]), int(size[1])
    playheads = compare_record_playheads(seconds, scene_fps, file_fps, length, speed)
    video.hide_gui_chrome(server)
    binary = video.browser_binary(browser)
    url = f"http://localhost:{port}"
    process = None
    if binary:
        client = video.wait_for_client(server, timeout=video.BROWSER_GRACE_SECONDS)
        if client is None:
            process = video.launch_browser(binary, url, video.browser_window(size))
            client = (
                video.wait_for_client(server, timeout=start_timeout)
                if process is not None
                else None
            )
    else:
        client = video.wait_for_client(server, timeout=start_timeout)
    if client is None:
        print("[viewer] compare: no browser connected, nothing recorded", flush=True)
        video.stop_browser(process)
        return 0
    pose = compare_record_pose(offsets, view, home=home)
    client.camera.position = pose[0]
    client.camera.look_at = pose[1]
    if len(offsets):
        pad = COMPARE_PAD_TIGHT if view == "oblique" else COMPARE_PAD
        fov = box_fov(
            pose[0], pose[1], compare_box(offsets, pad), width / max(1, height)
        )
        fov = float(
            np.clip(fov / max(0.1, float(zoom)), RECORD_FOV_MIN, RECORD_FOV_MAX)
        )
        client.camera.fov = fov
        print(
            f"[viewer] compare: recording {view} view from ({pose[0][0]:.2f}, "
            f"{pose[0][1]:.2f}, {pose[0][2]:.2f}) looking at ({pose[1][0]:.2f}, "
            f"{pose[1][1]:.2f}, {pose[1][2]:.2f}), lens {np.degrees(fov):.0f} deg, "
            f"speed x{float(speed):g}",
            flush=True,
        )
    show(int(playheads[0]))
    renders = settle_render(client, (width, height))
    print(
        f"[viewer] compare: page settled after {renders} render(s); recording",
        flush=True,
    )
    print(
        f"[viewer] compare: recording {len(playheads)} frames at {width}x{height} "
        f"to {path} ({file_fps:g} fps, {len(playheads) / file_fps:.1f}s of video)",
        flush=True,
    )
    started = time.time()
    try:
        written, actual = record_compare(
            show,
            client,
            lambda real: video.VideoSink(path, real, file_fps),
            playheads=playheads,
            size=(width, height),
        )
    finally:
        video.stop_browser(process)
    elapsed = time.time() - started
    print(
        f"[viewer] compare: wrote {written} frames at {actual[0]}x{actual[1]} "
        f"to {path} in {elapsed:.1f}s ({written / max(1e-6, elapsed):.1f} fps "
        "captured)",
        flush=True,
    )
    return written
