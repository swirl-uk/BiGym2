"""The grid of a multi-scene view, the camera that frames it, a colour per scene."""

from __future__ import annotations

# Up to six environments in a grid. A column steps along +y and a row steps
# along -x, so several rows stand behind one another rather than side by
# side for ever.
COMPARE_SLOTS = 6


COMPARE_GAP = 2.5


# What the layout dropdown offers, and what each one means in columns.
LAYOUT_AUTO = "auto"


LAYOUT_ROW = "row"


COMPARE_LAYOUTS = (LAYOUT_AUTO, LAYOUT_ROW, "2 columns", "3 columns")


# ``auto`` keeps a single row while it stays readable and folds into two
# columns from the fourth slot on.
AUTO_ROW_MAX = 3


# One tint per slot so the robots read apart at a glance. A tint is a
# per-channel multiplier on the robot's geom colours and is display only: it
# is written to the model only while mjviser bakes that slot's meshes, and
# the original ``geom_rgba`` is put back before anything renders from the
# model again -- the camera panels go through MuJoCo's renderer, which reads
# ``geom_rgba`` live, and they must show the policy's own view.
SLOT_TINTS = (
    (0.45, 0.68, 1.00),
    (1.00, 0.62, 0.32),
    (0.48, 0.88, 0.52),
    (0.90, 0.56, 0.95),
    (1.00, 0.85, 0.35),
    (0.40, 0.85, 0.92),
)


# The same six colours as CSS, for the legend's swatches and the stripe down
# the side of each 3D billboard.
SLOT_COLOURS = (
    "#73aeff",
    "#ff9e52",
    "#7ae085",
    "#e68ef2",
    "#ffd95a",
    "#66d9eb",
)


FOCUS_FREE = "(free view)"


def layout_columns(layout: str, count: int) -> int:
    """How many columns a layout puts ``count`` slots in.

    Args:
        layout: One of :data:`COMPARE_LAYOUTS` (anything else reads as
            ``auto``).
        count: How many slots there are.

    Returns:
        The number of columns, at least one.
    """
    count = max(1, int(count))
    name = str(layout or LAYOUT_AUTO).strip().lower()
    if name == LAYOUT_ROW:
        return count
    digits = name.split(" ", 1)[0]
    if digits.isdigit() and int(digits) > 0:
        return min(count, int(digits))
    return count if count <= AUTO_ROW_MAX else 2


def slot_grid(count: int, layout: str = LAYOUT_AUTO) -> list[tuple[int, int]]:
    """The ``(row, column)`` of every slot, filling each row left to right."""
    columns = layout_columns(layout, count)
    return [(i // columns, i % columns) for i in range(max(0, int(count)))]


def slot_offset(
    row: int, column: int, gap: float, gap_x: float | None = None
) -> tuple[float, float, float]:
    """Where the scene of the slot at ``(row, column)`` is moved to.

    A column steps along +y, so one row reads left to right; a row steps
    along -x, so the second row stands behind the first rather than
    continuing the line off the side of the screen.

    Args:
        row: Its row index.
        column: Its column index.
        gap: Metres between neighbouring columns.
        gap_x: Metres between rows, or None for ``gap``.

    Returns:
        The ``(x, y, z)`` offset of that slot's frames.
    """
    depth = float(gap if gap_x is None else gap_x)
    # ``+ 0.0`` so row 0 reads as 0.0 rather than -0.0 in the logs.
    return (-depth * int(row) + 0.0, float(gap) * int(column), 0.0)


def grid_offsets(
    count: int, gap: float, layout: str = LAYOUT_AUTO
) -> list[tuple[float, float, float]]:
    """One offset per slot, in slot order."""
    return [slot_offset(r, c, gap) for r, c in slot_grid(count, layout)]


def grid_shape(count: int, layout: str = LAYOUT_AUTO) -> tuple[int, int]:
    """``(rows, columns)`` of the grid a layout puts ``count`` slots in."""
    cells = slot_grid(count, layout)
    if not cells:
        return (0, 0)
    return (cells[-1][0] + 1, layout_columns(layout, count))


def grid_note(
    count: int, gap: float, layout: str = LAYOUT_AUTO, noun: str = "slot"
) -> str:
    """``4 slots as 2 columns x 2 rows, gap 2.5 m`` for the log and header."""
    rows, columns = grid_shape(count, layout)
    return (
        f"{count} {noun}{'' if count == 1 else 's'} as {columns} column"
        f"{'' if columns == 1 else 's'} x {rows} row{'' if rows == 1 else 's'}"
        f", gap {gap:g} m"
    )


def compare_camera_pose(
    offsets, height: float = 1.6, distance: float = 2.8
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Where a client's camera starts so the whole grid is in view.

    The camera stands on the -x side, which is the side the single-rollout
    viewer watches a robot from, far enough out to hold the width of the
    grid (+y) and — once there is more than one row — high enough to look
    over the heads of the nearer row at the one behind it. Row 0 is the
    far row: rows recede along -x, towards the camera's own side, so the
    grid is read from the front of the last row backwards.

    Args:
        offsets: Every slot's offset, from :func:`grid_offsets`.
        height: Eye height above the floor for a single row.
        distance: How far out a single slot is watched from.

    Returns:
        ``(position, look_at)`` in world metres.
    """
    places = [tuple(float(v) for v in o) for o in offsets] or [(0.0, 0.0, 0.0)]
    xs = [o[0] for o in places]
    ys = [o[1] for o in places]
    span = max(ys) - min(ys)
    depth = max(xs) - min(xs)
    mid_y = 0.5 * (min(ys) + max(ys))
    # 0.8 m of eye height per metre of depth clears a 1.7 m robot standing
    # in the near row on the line of sight to the far one.
    return (
        (min(xs) - distance - 0.7 * span, mid_y, height + 0.8 * depth),
        (max(xs) - 0.5 * depth, mid_y, 0.8),
    )


def focus_pose(
    offset, pelvis, distance: float = 2.0, height: float = 1.35
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Where a camera stands to watch one slot on its own.

    On the -x side of that robot, looking at its pelvis a little above the
    hips.

    Args:
        offset: The slot's grid offset.
        pelvis: Its pelvis position inside its own model.
        distance: Metres from the pelvis along -x.
        height: Eye height above the floor.

    Returns:
        ``(position, look_at)`` in world metres.
    """
    anchor = (
        float(pelvis[0]) + float(offset[0]),
        float(pelvis[1]) + float(offset[1]),
        float(pelvis[2]),
    )
    return (
        (anchor[0] - float(distance), anchor[1], float(height)),
        (anchor[0], anchor[1], anchor[2] + 0.15),
    )


def slot_colour(index: int) -> str:
    """The CSS colour of slot ``index`` (the palette repeats)."""
    return SLOT_COLOURS[int(index) % len(SLOT_COLOURS)]


def css_rgb(colour: str) -> tuple[int, int, int]:
    """``#73aeff`` as ``(115, 174, 255)``; grey when it is not a hex colour."""
    text = str(colour).strip().lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    try:
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    except (ValueError, IndexError):
        return (160, 160, 160)
