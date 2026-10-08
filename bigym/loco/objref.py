"""``"pkg.module:ATTR"`` references to objects defined outside bigym.

A task or backend name with a colon names an object in an importable
module (``controller={"backend": "my_lab.wbc:BINDING"}``). The string is
the whole description: it goes into config and metadata as is, and a
fresh process (the bigym-agent subprocess, a replay) imports the same
object from it. Where a name becomes a file or directory name,
:func:`path_name` replaces the colon.
"""

from __future__ import annotations

import functools
import importlib
from typing import Any


def is_object_ref(name: str) -> bool:
    """Whether ``name`` is a ``"pkg.module:ATTR"`` reference."""
    return ":" in name


def path_name(name: str) -> str:
    """``name`` as a file or directory name: ``"pkg.module:ATTR"`` -> ``"pkg.module-ATTR"``.

    Names without a colon come back unchanged.
    """
    return name.replace(":", "-")


@functools.cache
def import_object(ref: str) -> Any:
    """Import the object ``"pkg.module:ATTR"`` names (once per process).

    ``ATTR`` may be dotted (``"pkg.module:Class.attr"``). An unimportable
    module raises ImportError naming it; a missing attribute raises
    AttributeError.
    """
    module_name, _, attr = ref.partition(":")
    if not module_name or not attr or ":" in attr:
        raise ValueError(f"{ref!r} is not a 'pkg.module:ATTR' reference")
    try:
        obj = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"cannot import module {module_name!r} (from {ref!r}): {exc}"
        ) from exc
    for part in attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            raise AttributeError(
                f"module {module_name!r} has no attribute {attr!r} (from {ref!r})"
            ) from None
    return obj
