"""OpenXR / WiVRn runtime plumbing (abort on foreign servers, logging)."""

from __future__ import annotations

import getpass
import os
import subprocess
from pathlib import Path


def _find_wivrn_runtime_json() -> Path | None:
    candidates: list[Path] = []
    direct_paths = [
        Path("/usr/share/openxr/1/openxr_wivrn.json"),
        Path("/usr/local/share/openxr/1/openxr_wivrn.json"),
    ]
    for path in direct_paths:
        if path.exists():
            candidates.append(path)

    flatpak_roots = [
        Path.home() / ".local/share/flatpak/app/io.github.wivrn.wivrn",
        Path("/var/lib/flatpak/app/io.github.wivrn.wivrn"),
    ]
    for root in flatpak_roots:
        if root.exists():
            candidates.extend(root.glob("**/files/share/openxr/1/openxr_wivrn.json"))

    if not candidates:
        return None
    return sorted(candidates)[-1]


def abort_on_foreign_vr_server() -> None:
    """Abort when a wivrn-server owned by another account is running.

    One machine, one headset: a second wivrn-server cannot bind the
    port, and the Quest only talks to one server — a foreign server makes
    the XR connect step hang or fail with nothing actionable in the output.
    Name the owner instead so people can coordinate. VR_CONFLICT_OK=1
    bypasses (sharing a server across accounts is not a supported setup).
    """
    if os.environ.get("VR_CONFLICT_OK"):
        return
    try:
        out = subprocess.run(
            ["ps", "-C", "wivrn-server", "-o", "user=,pid="],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except Exception:
        return
    me = getpass.getuser()
    foreign = [line.split() for line in out.splitlines() if line.split()]
    foreign = [(u, p) for u, p, *_ in foreign if u != me]
    if foreign:
        owners = ", ".join(f"{user} (pid {pid})" for user, pid in foreign)
        raise SystemExit(
            f"[vr] wivrn-server is already running under another account: {owners}.\n"
            "One headset, one server: ask them to quit WiVRn (dashboard + server), "
            "then rerun. VR_CONFLICT_OK=1 skips this check."
        )


def configure_openxr_runtime() -> None:
    """Point OpenXR at WiVRn when no runtime is set or installed as active."""
    if os.environ.get("XR_RUNTIME_JSON"):
        print(f"[info] using XR_RUNTIME_JSON={os.environ['XR_RUNTIME_JSON']}")
        return

    active_runtime_paths = [
        Path.home() / ".config/openxr/1/active_runtime.json",
        Path("/etc/xdg/openxr/1/active_runtime.json"),
    ]
    if any(path.exists() for path in active_runtime_paths):
        return

    runtime_json = _find_wivrn_runtime_json()
    if runtime_json is None:
        return
    os.environ["XR_RUNTIME_JSON"] = str(runtime_json)
    print(f"[info] auto-selected WiVRn OpenXR runtime: {runtime_json}")


def configure_openxr_logging(log_level: str) -> None:
    """Set the Monado/WiVRn log levels unless the environment already does."""
    level = str(log_level)
    os.environ.setdefault("XRT_LOG", level)
    os.environ.setdefault("XRT_COMPOSITOR_LOG", level)
