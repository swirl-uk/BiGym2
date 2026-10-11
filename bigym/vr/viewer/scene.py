"""The scenes of a multi-scene viser view: models, robot tints, lights, labels.

Each slot is one MuJoCo model under its own node-name prefix, its robot
tinted for display only, with a label billboard above the robot.
"""

from __future__ import annotations

import contextlib
import time
from typing import cast

import mjviser.scene as mjscene
import mujoco
import numpy as np
import viser
from PIL import Image, ImageDraw, ImageFont

from bigym.vr.viewer.layout import SLOT_COLOURS, css_rgb

# How far above the pelvis a slot's billboard hangs, in metres, and how
# wide the plate is drawn.
BILLBOARD_Z = 1.15


BILLBOARD_WIDTH = 0.9


# The billboard is a flat image, so it has one face: it looks back along -x,
# which is the side both the single-rollout viewer and the compare camera
# watch the robots from.
PLATE_FACE = (-1.0, 0.0, 0.0)


PLATE_UP = (0.0, 0.0, 1.0)


# The plate is drawn on a canvas of this fixed size so the image node's
# geometry never has to change when the text does.
PLATE_SIZE = (720, 176)


PLATE_FONT_SIZE = 36


# A long label is drawn smaller rather than clipped, down to this size.
PLATE_FONT_MIN = 14


# DejaVu carries the glyphs the labels use (· ✓ ✗); PIL's own default face
# is the fallback when no DejaVu is installed.
PLATE_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSans.ttf",
)


def fly_clients_to(server, position, look_at) -> int:
    """Point every connected client's camera at one pose (main loop only).

    Args:
        server: The viser server.
        position: Where the camera goes.
        look_at: What it looks at.

    Returns:
        How many clients were moved.
    """
    moved = 0
    try:
        clients = list(server.get_clients().values())
    except Exception:  # a disconnect mid-iteration is not an error
        return 0
    for client in clients:
        with contextlib.suppress(Exception):
            client.camera.position = tuple(float(v) for v in position)
            client.camera.look_at = tuple(float(v) for v in look_at)
            moved += 1
    return moved


@contextlib.contextmanager
def prefixed_body_names(prefix: str):
    """Make mjviser name this scene's nodes ``/bodies/<prefix>/<body>/...``.

    mjviser hard-codes the viser node paths it builds (``/bodies/<body>`` and
    ``/fixed_bodies/<body>``), so N scenes on one server would collide. Every
    path goes through ``scene.get_body_name``, so prefixing that with a path
    segment gives each slot its own subtree — which is also how the slot is
    moved: a frame at ``/bodies/<prefix>`` offsets everything below it.

    Args:
        prefix: The path segment for this slot.

    Yields:
        None, with the patch in place for the duration of the block.
    """
    original = mjscene.get_body_name

    def named(mj_model, body_id) -> str:
        return f"{prefix}/{original(mj_model, body_id)}"

    mjscene.get_body_name = named  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        mjscene.get_body_name = original


@contextlib.contextmanager
def slot_scene_build(server, prefix: str, *, keep_grid: bool, fixed_frame):
    """Everything one compare slot needs mjviser to do differently.

    Three patches, live only while ``ViserMujocoScene`` is constructing:

    * node names get the slot's prefix (see :func:`prefixed_body_names`);
    * the ``/fixed_bodies`` root frame is created once by the caller and
      handed to every slot — mjviser adds it per scene, and adding a viser
      node twice at the same path removes the first one, taking the
      previous slot's static geometry with it;
    * the ground grid is captured so only the first slot keeps one (every
      slot draws its own infinite plane at z = 0, and coplanar copies
      z-fight).

    Args:
        server: The viser server.
        prefix: This slot's path segment.
        keep_grid: True for the slot whose ground plane stays visible.
        fixed_frame: The shared ``/fixed_bodies`` frame handle.

    Yields:
        None, with the patches in place for the duration of the block.
    """
    api = type(server.scene)
    original_grid = api.add_grid
    original_frame = api.add_frame
    created = []

    def add_grid(self, name, *args, **kwargs):
        handle = original_grid(self, name, *args, **kwargs)
        created.append(handle)
        return handle

    def add_frame(self, name, *args, **kwargs):
        if str(name) == "/fixed_bodies":
            return fixed_frame
        return original_frame(self, name, *args, **kwargs)

    api.add_grid = add_grid
    api.add_frame = add_frame
    try:
        with prefixed_body_names(prefix):
            yield
    finally:
        api.add_grid = original_grid
        api.add_frame = original_frame
        if not keep_grid:
            for handle in created:
                handle.visible = False


def tint_slot_robot(model, colour: tuple[float, float, float]) -> int:
    """Tint one slot's robot so the copies read apart (display only).

    Multiplies the robot's geom rgb by ``colour`` in this process's copy of
    the model, exactly as the viewer's ``--robot-tint`` does: nothing is
    written to disk and no observation changes (compare mode replays stored
    ``full_qpos``, it does not step a policy).

    Args:
        model: The slot's ``mujoco.MjModel``.
        colour: Per-channel multiplier.

    Returns:
        The number of geoms tinted.
    """
    roots = set()
    for body in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        if name.endswith("/pelvis"):
            roots.add(int(model.body_rootid[body]))
    if not roots:
        return 0
    mask = np.isin(model.body_rootid[model.geom_bodyid], list(roots))
    model.geom_rgba[mask, :3] = np.clip(
        model.geom_rgba[mask, :3] * np.asarray(colour, dtype=float), 0.0, 1.0
    )
    return int(mask.sum())


def apply_lighting(server, prefs: dict) -> None:
    """Apply the viewer's lighting preferences to a freshly built scene."""
    sky = str(prefs.get("sky", "black"))
    try:
        server.scene.configure_environment_map(
            environment_intensity=float(prefs.get("environment", 0.7)),
            background=sky != "transparent",
            background_blurriness=1.0,
            background_intensity=0.5 if sky == "grey" else 0.0,
        )
        server.scene.configure_default_lights(
            enabled=bool(prefs.get("default_lights", True)),
            cast_shadow=bool(prefs.get("shadows", True)),
        )
    except Exception as exc:  # lighting is cosmetic
        print(f"[viewer] lighting unavailable: {exc}", flush=True)


def pelvis_body_id(model) -> int:
    """The id of the robot's pelvis body, or -1 when the model has none.

    Args:
        model: A ``mujoco.MjModel``.

    Returns:
        The body id whose name ends in ``/pelvis``, else -1.
    """
    for body in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        if name.endswith("/pelvis"):
            return int(body)
    return -1


def pelvis_position(data, body: int) -> tuple[float, float, float]:
    """The pelvis's world position after ``mj_forward``.

    Reading ``data.xpos`` rather than decoding qpos means the anchor is
    right whatever the robot's root joint looks like, and it is already the
    position mjviser drew the body at.

    Args:
        data: A ``mujoco.MjData`` that has been forwarded.
        body: The pelvis body id, or -1.

    Returns:
        ``(x, y, z)``; ``(0, 0, 0.8)`` when there is no pelvis body.
    """
    if body < 0:
        return (0.0, 0.0, 0.8)
    x, y, z = (float(v) for v in data.xpos[body])
    return (x, y, z)


def billboard_anchor(
    pelvis, offset, z_offset: float = BILLBOARD_Z
) -> tuple[float, float, float]:
    """Where one slot's billboard hangs in the compare scene.

    Every slot's environment is modelled at the origin and moved into place
    by a frame that carries its grid offset, so a pelvis position read out
    of that slot's ``MjData`` has to be offset the same way before it can be
    used as a world anchor for a label, which hangs off the scene root.

    Args:
        pelvis: The pelvis world position inside the slot's own model.
        offset: The slot's grid offset (see
            :func:`~bigym.vr.viewer.layout.slot_offset`).
        z_offset: How far above the pelvis the billboard hangs.

    Returns:
        The pelvis plus the slot's offset plus ``z_offset`` in z.
    """
    return (
        float(pelvis[0]) + float(offset[0]),
        float(pelvis[1]) + float(offset[1]),
        float(pelvis[2]) + float(offset[2]) + float(z_offset),
    )


def slot_outcome(info: dict) -> str:
    """``✓`` / ``✗ · timeout · fell`` for the label under a finished slot."""
    parts = []
    success = info.get("success")
    if success is not None:
        parts.append("✓" if float(success) > 0 else "✗")
    end = str(info.get("termination") or "")
    if end and end != "success":
        parts.append(end)
    if info.get("fell"):
        parts.append("fell")
    return " · ".join(parts)


def quaternion_from_axes(x, y, z) -> tuple[float, float, float, float]:
    """The ``(w, x, y, z)`` of the rotation whose axes are the three given.

    Args:
        x: Where the local +x axis points, in world coordinates.
        y: Where the local +y axis points.
        z: Where the local +z axis points.

    Returns:
        The quaternion, with a non-negative ``w`` so the same rotation
        always reads the same way.
    """
    m = np.column_stack([np.asarray(v, dtype=float).reshape(3) for v in (x, y, z)])
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = [
            0.25 * scale,
            (m[2, 1] - m[1, 2]) / scale,
            (m[0, 2] - m[2, 0]) / scale,
            (m[1, 0] - m[0, 1]) / scale,
        ]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        scale = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        quat = [
            (m[2, 1] - m[1, 2]) / scale,
            0.25 * scale,
            (m[0, 1] + m[1, 0]) / scale,
            (m[0, 2] + m[2, 0]) / scale,
        ]
    elif m[1, 1] > m[2, 2]:
        scale = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        quat = [
            (m[0, 2] - m[2, 0]) / scale,
            (m[0, 1] + m[1, 0]) / scale,
            0.25 * scale,
            (m[1, 2] + m[2, 1]) / scale,
        ]
    else:
        scale = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        quat = [
            (m[1, 0] - m[0, 1]) / scale,
            (m[0, 2] + m[2, 0]) / scale,
            (m[1, 2] + m[2, 1]) / scale,
            0.25 * scale,
        ]
    if quat[0] < 0.0:
        quat = [-v for v in quat]
    return cast(tuple[float, float, float, float], tuple(float(v) for v in quat))


def plate_wxyz(face=PLATE_FACE, up=PLATE_UP) -> tuple[float, float, float, float]:
    """The orientation that shows an image plane's picture along ``face``.

    viser's image node is a plane that the client pre-rotates by half a turn
    about its x axis: the picture is on the side its local -z points to and
    the top of the picture is along its local -y (viewing the plane down its
    local +z shows the picture mirrored; keeping local +y up shows it upside
    down). So the local -z is sent along ``face``, from the plate towards
    whoever reads it, and the local -y is sent along ``up``. Looking along -x
    with +z up is what makes the billboards readable from the side the
    compare camera starts on.

    Args:
        face: The world direction the picture looks along (plate to reader).
        up: Which way is up on the picture.

    Returns:
        The ``(w, x, y, z)`` for ``scene.add_image``.
    """
    normal = np.asarray(face, dtype=float)
    normal = normal / np.linalg.norm(normal)
    vertical = np.asarray(up, dtype=float)
    vertical = vertical - normal * float(vertical @ normal)
    vertical = vertical / np.linalg.norm(vertical)
    local_z = -normal
    local_y = -vertical
    return quaternion_from_axes(np.cross(local_y, local_z), local_y, local_z)


def plate_font(size: int):
    """A face for the billboard text, DejaVu when the system has one."""
    for path in PLATE_FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:  # Pillow >= 10.1 can scale its own bundled face
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def plate_status(status: str) -> str:
    """The part of a slot's status that is worth drawing on its plate.

    The step counter changes every frame and the plate is an image, which
    has to be re-drawn and re-encoded to say anything new; the outcome
    changes once. The counter stays in the sidebar legend, which is text
    and costs nothing to rewrite.

    Args:
        status: A line from :func:`slot_status_text`.

    Returns:
        The same line with the ``step a/b`` part removed.
    """
    parts = [p for p in str(status).split(" · ") if not p.startswith("step ")]
    return " · ".join(parts)


def plate_image(
    label: str,
    status: str,
    colour: str,
    *,
    size: tuple[int, int] = PLATE_SIZE,
    font_size: int = PLATE_FONT_SIZE,
):
    """Draw one billboard: white text on an opaque dark plate.

    The plate is laid out on a ``size`` canvas (the text is drawn smaller
    when it would not fit that width) and the image returned is cropped to
    the plate itself, so the only transparent pixels are the rounded
    corners. That matters in the scene: a transparent pixel still writes
    depth, so a wide clear margin around the plate blanks out whatever is
    drawn after it and behind it — the semi-transparent floor grid, when
    the plate is farther from the camera than the grid's origin — into a
    black box. :func:`plate_extent` sizes the image node to match.

    Args:
        label: The slot title (session, version, scores).
        status: The line under it, or "" for a single-line plate.
        colour: The slot's CSS colour, drawn as a stripe down the side.
        size: ``(width, height)`` of the layout canvas in pixels.
        font_size: Text size in pixels.

    Returns:
        An ``(h, w, 4)`` uint8 RGBA array no larger than ``size``; a 2x2
        transparent one when there is nothing to say.
    """
    width, height = int(size[0]), int(size[1])
    lines = [line for line in (str(label), str(status)) if line]
    if not lines:
        return np.zeros((2, 2, 4), dtype=np.uint8)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    pad_x, pad_y, stripe = 14, 10, 7
    room = width - 2 * pad_x - stripe - 8
    font_px = int(font_size)
    font = plate_font(font_px)
    boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    text_w = max(box[2] - box[0] for box in boxes)
    # A label as long as "s2 · v001 (template) · no dev episodes · not
    # evaluated" is drawn smaller rather than run off the side of the plate.
    for _ in range(3):
        if text_w <= room or font_px <= PLATE_FONT_MIN:
            break
        font_px = max(PLATE_FONT_MIN, int(font_px * room / text_w))
        font = plate_font(font_px)
        boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
        text_w = max(box[2] - box[0] for box in boxes)
    spacing = int(round(0.35 * font_px))
    line_h = max(box[3] - box[1] for box in boxes)
    plate_w = min(width, text_w + 2 * pad_x + stripe + 8)
    plate_h = min(height, len(lines) * line_h + (len(lines) - 1) * spacing + 2 * pad_y)
    left = (width - plate_w) // 2
    top = (height - plate_h) // 2
    draw.rounded_rectangle(
        (left, top, left + plate_w - 1, top + plate_h - 1),
        radius=10,
        fill=(15, 15, 18, 255),
    )
    draw.rectangle(
        (left + 4, top + 4, left + 4 + stripe, top + plate_h - 5),
        fill=css_rgb(colour) + (255,),
    )
    y = top + pad_y
    for index, (line, box) in enumerate(zip(lines, boxes, strict=True)):
        draw.text(
            (left + stripe + pad_x, y - box[1]),
            line,
            font=font,
            fill=(255, 255, 255, 255) if index == 0 else (214, 214, 222, 255),
        )
        y += line_h + spacing
    return np.ascontiguousarray(
        np.asarray(image)[top : top + plate_h, left : left + plate_w]
    )


def plate_extent(image, width: float) -> tuple[float, float]:
    """The metres a plate image should span in the scene.

    ``width`` is what a plate as wide as the whole :data:`PLATE_SIZE`
    canvas would span; a narrower plate (the image is cropped to its text)
    spans proportionally less, so the text is the same size on every plate.

    Args:
        image: The ``(h, w, 4)`` array from :func:`plate_image`.
        width: The full-canvas width in metres.

    Returns:
        ``(width, height)`` in metres.
    """
    shown = float(width) * float(image.shape[1]) / float(PLATE_SIZE[0])
    return shown, shown * float(image.shape[0]) / float(image.shape[1])


def viewer_camera_position(server) -> tuple[float, float, float] | None:
    """The camera position of the most recently connected client, or None."""
    try:
        clients = list(server.get_clients().values())
    except Exception:  # a closed server
        return None
    if not clients:
        return None
    try:
        return cast(
            tuple[float, float, float],
            tuple(float(v) for v in clients[-1].camera.position),
        )
    except Exception:  # camera not reported yet
        return None


def facing_wxyz(plate, camera) -> tuple[float, float, float, float]:
    """Orientation that turns a plate at ``plate`` towards ``camera``.

    The picture looks along the line from the plate to the camera and its
    text keeps world +z up; when the camera is straight above or below, +x
    is used as the up hint instead so the orientation stays defined.

    Args:
        plate: The plate's world position.
        camera: The camera's world position.

    Returns:
        The ``(w, x, y, z)`` quaternion for the image node.
    """
    face = np.asarray(camera, dtype=float) - np.asarray(plate, dtype=float)
    norm = float(np.linalg.norm(face))
    if norm < 1e-6:
        return plate_wxyz()
    face = face / norm
    up = (
        PLATE_UP
        if abs(float(face @ np.asarray(PLATE_UP, dtype=float))) < 0.99
        else (1.0, 0.0, 0.0)
    )
    return plate_wxyz(face=tuple(face), up=up)


def add_plate_image(server, name: str, image, width: float, *, position, visible):
    """Add one billboard plane, without shadows when viser allows it."""
    width, height = plate_extent(image, width)
    common = {
        "format": "png",
        "wxyz": plate_wxyz(),
        "position": tuple(float(v) for v in position),
        "visible": bool(visible),
    }
    return server.scene.add_image(
        name,
        image,
        width,
        height,
        cast_shadow=False,
        receive_shadow=False,
        **common,
    )


class SlotBillboard:
    """The plate of text that hangs above one compare slot's pelvis.

    It is a PIL-drawn RGBA image, not an HTML node. viser 1.1 has no
    ``scene.add_html``; its only HTML-in-the-scene node is
    ``scene.add_3d_gui_container``, whose contents the client wraps in a
    Mantine ``Paper`` styled ``background-color: var(--mantine-color-body)``
    — a white card behind any label drawn in it. The container
    takes ``name``, ``wxyz``, ``position`` and ``visible`` and nothing else,
    and ``gui.configure_theme`` carries no custom CSS, so there is no
    supported way to switch that card off.

    An image is a plane, so the plate is turned towards the viewer by hand:
    every time it is moved (once per playback frame) it reads the camera
    position of the connected client and orients its face along the line
    from the plate to that camera, with world +z kept up, so the text is
    read head-on from wherever the user looks. With no client connected it
    faces -x, the side the compare camera starts on. The sidebar legend is
    the copy that carries the live step counter — an image has to be
    re-encoded to change, so the plate is re-drawn only when its text
    actually differs (see :func:`plate_status`).
    """

    def __init__(
        self,
        server,
        name: str,
        *,
        position,
        visible: bool = True,
        width: float = BILLBOARD_WIDTH,
    ):
        """Create the plate, empty, at ``position``.

        Args:
            server: The viser server.
            name: The scene node path.
            position: World anchor (see :func:`billboard_anchor`).
            visible: False while the compare scene is still being built.
            width: How wide the plate is drawn, in metres.
        """
        self.kind = "image"
        self.width = float(width)
        self.drawn = ""
        self.server = server
        self._faced: tuple | None = None
        self.node = add_plate_image(
            server,
            name,
            plate_image("", "", SLOT_COLOURS[0]),
            self.width,
            position=position,
            visible=visible,
        )

    def write(self, label: str, status: str, colour: str) -> None:
        """Set what the plate says, re-drawing it only when that changes."""
        short = plate_status(status)
        key = f"{label}\n{short}\n{colour}"
        if key == self.drawn:
            return
        self.drawn = key
        image = plate_image(label, short, colour)
        self.node.image = image
        shown_w, shown_h = plate_extent(image, self.width)
        try:
            self.node.render_width = shown_w
            self.node.render_height = shown_h
        except AttributeError:  # a fake node in tests
            pass

    def move(self, position) -> None:
        """Re-anchor the plate (the pelvis walked) and turn it to the viewer."""
        self.node.position = tuple(float(v) for v in position)
        self.face()

    def face(self) -> bool:
        """Turn the plate towards the connected client's camera.

        Uses the most recently connected client (one viewer is the common
        case). The orientation is recomputed only when the camera or the
        plate has moved by more than a centimetre since the last time.

        Returns:
            True when the orientation was updated.
        """
        camera = viewer_camera_position(self.server)
        if camera is None:
            return False
        key = tuple(round(v, 2) for v in (*camera, *self.position))
        if key == self._faced:
            return False
        self._faced = key
        self.node.wxyz = facing_wxyz(self.position, camera)
        return True

    @property
    def position(self) -> tuple:
        """Where the plate currently hangs."""
        return tuple(float(v) for v in self.node.position)

    def reveal(self) -> None:
        """Show the plate once its slot has been posed."""
        self.node.visible = True


def pose_slot(built: dict, frame: int, mujoco) -> int:
    """Pose one slot at ``frame`` of its replay and push it to the browser.

    The environment was reset on the compare seed, which puts the props
    where that seed wants them but leaves the robot in the reset pose --
    which is not frame 0 of the replay. Every slot is posed from the stored
    ``full_qpos`` before it is ever shown, so the reset pose never reaches
    the screen.

    Args:
        built: One entry of the scene's ``slots`` list.
        frame: The playhead, clamped to this slot's episode.
        mujoco: The ``mujoco`` module.

    Returns:
        The frame actually used.
    """
    qpos = built["qpos"]
    here = max(0, min(int(frame), int(qpos.shape[0]) - 1))
    built["data"].qpos[: qpos.shape[1]] = qpos[here]
    mujoco.mj_forward(built["model"], built["data"])
    built["scene"].update_from_mjdata(built["data"])
    return here


def slot_status_text(built: dict, here: int) -> str:
    """``step 812/1701``, plus the outcome once the playhead is at the end."""
    total = int(built["qpos"].shape[0])
    text = f"step {here + 1}/{total}"
    if here >= total - 1:
        outcome = slot_outcome(built["info"])
        if outcome:
            text = f"{text} · {outcome}"
    return text


def place_billboard(built: dict) -> tuple[float, float, float]:
    """Move one slot's billboard onto its pelvis and return the anchor."""
    anchor = billboard_anchor(
        pelvis_position(built["data"], built["pelvis"]), built["offset"]
    )
    built["billboard"].move(anchor)
    return anchor


def compare_server(port: int):
    """Open the viser server compare mode draws into.

    The port is retried: a just-stopped single-rollout server needs a moment
    to release the socket.

    Args:
        port: The viser port.

    Returns:
        The running server.
    """
    for attempt in range(20):
        try:
            return viser.ViserServer(port=port, label="demo-viewer", verbose=False)
        except OSError:
            if attempt == 19:
                raise
            time.sleep(0.25)
    raise OSError(f"cannot open a viser server on port {port}")
