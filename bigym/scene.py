"""Scene assembly with ``mujoco.MjSpec``.

A scene is one spec: models are parsed with :func:`load` and combined with
:func:`attach`, then the whole scene is compiled once. Every attached model
hangs below a frame body named after its namespace (``"table/"``, ``"table_1/"``
for a second table) and its element names carry that prefix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco

DEFAULTS = mujoco.MjSpec()
OPTION_FIELDS = tuple(
    name
    for name in dir(DEFAULTS.option)
    if not name.startswith("_") and not callable(getattr(DEFAULTS.option, name))
)
SIZE_FIELDS = ("njmax", "nconmax", "nstack", "memory")
TREE_ELEMENTS = ("joints", "geoms", "sites", "cameras", "lights", "bodies")


def load(path: Path) -> mujoco.MjSpec:
    """Parse an MJCF file; an unnamed mesh, texture or height field is named after its file."""
    spec = mujoco.MjSpec.from_file(str(path))
    for asset in (*spec.meshes, *spec.textures, *spec.hfields):
        if asset.file and not asset.name:
            asset.name = Path(asset.file).stem
    return spec


def namespace(
    parent: mujoco.MjSpec, model: mujoco.MjSpec, site: mujoco.MjsSite | None = None
) -> str:
    """The name prefix ``model`` gets when attached to ``parent`` (at ``site``).

    It is the model name, suffixed ``_1``, ``_2``, ... when taken, inside the
    namespace of the site's model.
    """
    scope = "" if site is None else site.name[: site.name.rfind("/") + 1]
    taken = {body.name for body in parent.bodies}
    name = model.modelname
    suffix = 0
    while f"{scope}{name}/" in taken:
        suffix += 1
        name = f"{model.modelname}_{suffix}"
    return f"{scope}{name}/"


def attach(
    parent: mujoco.MjSpec, model: mujoco.MjSpec, site: mujoco.MjsSite | None = None
) -> mujoco.MjsBody:
    """Attach ``model`` below a new frame body of ``parent`` and return that frame.

    The frame is named after the model's :func:`namespace` and is added to the
    world body, or next to ``site`` at the site's pose. Options and sizes that
    ``model`` sets away from MuJoCo's defaults become the scene's.
    """
    prefix = namespace(parent, model, site)
    for field in OPTION_FIELDS:
        value = getattr(model.option, field)
        if str(value) != str(getattr(DEFAULTS.option, field)):
            setattr(parent.option, field, value)
    for field in SIZE_FIELDS:
        if getattr(model, field) != getattr(DEFAULTS, field):
            setattr(parent, field, getattr(model, field))

    if site is None:
        frame = parent.worldbody.add_body(name=prefix)
    else:
        frame = site.parent.add_body(
            name=prefix, pos=list(site.pos), quat=list(site.quat)
        )
    parent.attach(model, prefix=prefix, frame=frame.add_frame())
    return frame


def descendants(body: mujoco.MjsBody, elements: str) -> list[Any]:
    """The ``elements`` (e.g. ``"geoms"``) below ``body``, depth first in document order.

    Parsed elements keep their order in the MJCF file. An element added later
    takes the source line of the ``<default>`` it inherits from, or follows the
    parsed elements when the file has none.
    """
    found = []
    for kind, child in document_order(body):
        if kind == elements:
            found.append(child)
        if kind == "bodies":
            found.extend(descendants(child, elements))
    return found


def document_order(body: mujoco.MjsBody) -> list[tuple[str, Any]]:
    """The direct children of ``body`` with their kind, in document order."""
    parsed, added = [], []
    for kind in TREE_ELEMENTS:
        for index, child in enumerate(getattr(body, kind)):
            if child.info:
                line = int(child.info.removeprefix("line "))
                parsed.append(((line, kind == "bodies", index), (kind, child)))
            else:
                added.append((kind, child))
    parsed.sort(key=lambda item: item[0])
    return [entry for _, entry in parsed] + added
