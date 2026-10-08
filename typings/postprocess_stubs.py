"""Make the generated MuJoCo stubs platform-independent.

pybind11-stubgen introspects the installed mujoco, so its output bakes in:

1. The active OpenGL backend (``mujoco.cgl`` on macOS, ``mujoco.glfw`` or
   ``mujoco.egl`` on Linux). The backend stubs are dropped and ``GLContext``
   is typed as ``Any``; ``mujoco.Renderer`` itself is kept.
2. Install paths, memory addresses, the host platform name and MUJOCO_GL in
   default values and ``# value = ...`` comments. These are stripped.
3. The order of import lines, which differs between platforms. Each run of
   import lines is sorted.

It also types ``MjModel.bind`` and ``MjData.bind`` as methods: mujoco assigns
them in Python, so the generator sees plain functions and marks them static.

Adapted from mjlab (typings/postprocess_stubs.py, Apache-2.0).
"""

import re
import shutil
from pathlib import Path

STUB_DIR = Path(__file__).parent / "mujoco"

BACKENDS = ("cgl", "egl", "glfw", "glx", "osmesa")
_BACKEND = "|".join(BACKENDS)

# "from mujoco.cgl import GLContext" or "... import GLContext as _GLContext".
BACKEND_IMPORT = re.compile(
    rf"^from mujoco\.(?:{_BACKEND}) import GLContext(?: as (\w+))?$"
)
# "from . import cgl" in __init__.pyi.
BACKEND_SUBMODULE = re.compile(rf"^from \. import (?:{_BACKEND})$")
# macOS-only Rosetta detection constants.
PLATFORM_ONLY = re.compile(r"^(?:is_rosetta|proc_translated)\b")
DROP_FROM_ALL = (*BACKENDS, "is_rosetta", "proc_translated")

# Top-level import lines (``from __future__`` stays first on its own).
IMPORT = re.compile(r"^(?:import \S|from (?!__future__)\S+ import )")
# Values that depend on the host platform or on MUJOCO_GL.
PLATFORM_DEFAULT = re.compile(r"^((?:_SYSTEM|_MUJOCO_GL|_VALID_MUJOCO_GL): \w+) = .*$")
VALUE_COMMENT = re.compile(r" {2,}# value = .*$")
BIND = re.compile(r"^\s+def bind\(")
PATH_DEFAULT = re.compile(
    r"^(\s*\w+: str) = '[^']*(?:site-packages|/Users/|/home/)[^']*'\s*$"
)


def remove_backend_stubs() -> None:
    for name in BACKENDS:
        shutil.rmtree(STUB_DIR / name, ignore_errors=True)


def patch_line(line: str) -> str | None:
    match = BACKEND_IMPORT.match(line)
    if match:
        return f"from typing import Any as {match.group(1) or 'GLContext'}"
    if BACKEND_SUBMODULE.match(line) or PLATFORM_ONLY.match(line):
        return None
    if line.startswith("__all__"):
        for name in DROP_FROM_ALL:
            line = line.replace(f"'{name}', ", "")
    if "# value = " in line and ("site-packages" in line or "0x" in line):
        line = VALUE_COMMENT.sub("", line)
    line = PLATFORM_DEFAULT.sub(r"\1", line)
    match = PATH_DEFAULT.match(line)
    return match.group(1) if match else line


def sort_import_runs(lines: list[str]) -> list[str]:
    out: list[str] = []
    run: list[str] = []
    for line in lines:
        if IMPORT.match(line):
            run.append(line)
            continue
        out.extend(sorted(run))
        run = []
        out.append(line)
    return out + sorted(run)


def bind_as_method(lines: list[str]) -> list[str]:
    return [
        line
        for line, following in zip(lines, [*lines[1:], ""], strict=True)
        if not (line.strip() == "@staticmethod" and BIND.match(following))
    ]


def patch_stubs() -> None:
    for path in sorted(STUB_DIR.rglob("*.pyi")):
        patched = [patch_line(line) for line in path.read_text().splitlines()]
        lines = sort_import_runs([line for line in patched if line is not None])
        lines = bind_as_method(lines)
        path.write_text("\n".join(lines) + "\n")


def main() -> None:
    remove_backend_stubs()
    patch_stubs()


if __name__ == "__main__":
    main()
