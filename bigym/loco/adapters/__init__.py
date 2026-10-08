"""Lower-body backends: the registry and the built-in ``groot_wbc_g1``.

``BACKENDS`` maps each registered backend name to its
:class:`BackendBinding`; :func:`register_backend` adds one. A backend name
with a colon (``"pkg.module:ATTR"``) instead names a binding in an
importable module, imported on first use.

The vendored policy runtime needs only onnxruntime (a core dependency),
imported when a controller is built, so the bigym core stays torch-free.
"""

from __future__ import annotations

from bigym.loco.adapters.binding import BackendBinding
from bigym.loco.adapters.groot_wbc import GROOT_WBC_G1
from bigym.loco.objref import import_object, is_object_ref

BACKENDS: dict[str, BackendBinding] = {}


def register_backend(name: str, binding: BackendBinding) -> None:
    """Register ``binding`` under ``name`` (``controller={"backend": name}``).

    A name can be registered once; registering it again (``groot_wbc_g1``
    included) raises ValueError. ``controller={"backend": "pkg.module:ATTR"}``
    uses a binding without registering it.
    """
    if not isinstance(binding, BackendBinding):
        raise TypeError(
            f"backend {name!r} must be a BackendBinding, got {type(binding).__name__}"
        )
    if is_object_ref(name):
        raise ValueError(
            f"backend name {name!r} contains ':', which marks a 'pkg.module:ATTR' "
            "reference; register a plain name or pass the reference itself"
        )
    if name in BACKENDS:
        raise ValueError(f"lowerbody backend {name!r} is already registered")
    BACKENDS[name] = binding


register_backend("groot_wbc_g1", GROOT_WBC_G1)


def resolve_backend_name(name: str) -> str:
    """Validate a backend name: registered, or an importable binding reference."""
    name = str(name)
    backend_binding(name)
    return name


def backend_binding(name: str) -> BackendBinding:
    """The binding a backend name refers to."""
    if is_object_ref(name):
        binding = import_object(name)
        if not isinstance(binding, BackendBinding):
            raise TypeError(
                f"lowerbody backend {name!r} is a {type(binding).__name__}, "
                "not a BackendBinding"
            )
        return binding
    if name not in BACKENDS:
        raise ValueError(
            f"Unknown lowerbody backend {name!r}; expected one of {tuple(BACKENDS)}"
        )
    return BACKENDS[name]
