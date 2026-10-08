"""The GPU a session renders on, and its EGL device index.

The environment renders with EGL. A session takes the GPU with the most free
memory (``nvidia-smi``) and refuses to start when the best one has less than
``--min-free-mib`` free: a full GPU makes the EGL workers crash rather than
queue. ``MUJOCO_EGL_DEVICE_ID`` is an index into the EGL device list, which
need not agree with the ``nvidia-smi`` order; ``BIGYM_AGENT_EGL_MAP`` gives the
host's ``cuda:egl`` pairs (for example ``0:4,1:5,2:2,3:3,4:0,5:1``), and without
it the map is read from the driver.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys


def gpu_free_mib() -> dict[int, int]:
    """Read free memory per GPU from ``nvidia-smi``.

    Returns:
        ``{nvidia-smi index: free MiB}``, empty when there is no ``nvidia-smi``.
    """
    try:
        done = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
    except OSError:
        return {}
    free = {}
    for line in done.stdout.splitlines():
        try:
            index, mib = line.split(",")
            free[int(index)] = int(mib)
        except ValueError:
            continue
    return free


EGL_PROBE = r"""
import ctypes, importlib, json, sys
from OpenGL import EGL
devs = importlib.import_module("mujoco.egl.egl_ext").eglQueryDevicesEXT()
fn = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int,
                      ctypes.POINTER(ctypes.c_ssize_t))(
    EGL.eglGetProcAddress("eglQueryDeviceAttribEXT"))
out = {}
for i, d in enumerate(devs):
    h = ctypes.c_void_p(d) if isinstance(d, int) else ctypes.cast(d, ctypes.c_void_p)
    val = ctypes.c_ssize_t(-1)
    if fn(h, 0x323A, ctypes.byref(val)):  # EGL_CUDA_DEVICE_NV
        out[int(val.value)] = i
json.dump(out, sys.stdout)
"""


@functools.cache
def probe_egl_map() -> dict[int, int]:
    """Run :data:`EGL_PROBE` once per process (see :func:`detect_egl_map`)."""
    try:
        done = subprocess.run(
            [sys.executable, "-c", EGL_PROBE],
            capture_output=True,
            text=True,
            timeout=60,
            env=dict(os.environ, MUJOCO_GL="egl"),
        )
        found = json.loads(done.stdout) if done.returncode == 0 else {}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        found = {}
    return {int(k): int(v) for k, v in found.items()}


def detect_egl_map() -> dict[int, int]:
    """Ask EGL which CUDA device each EGL device is.

    EGL enumerates GPUs in its own order, which on multi-GPU hosts is not
    ``nvidia-smi``'s; rendering on "EGL device 5" may land on a GPU that is
    full. Every NVIDIA EGL device carries ``EGL_CUDA_DEVICE_NV``, so the map
    is read from the driver in a subprocess (EGL is never initialised in the
    launcher itself) and cached for the process.

    Returns:
        ``{nvidia-smi index: EGL index}``; empty when the probe fails (no
        NVIDIA EGL, no GPU), which means the identity map.
    """
    return dict(probe_egl_map())


def egl_map(raw: str | None = None) -> dict[int, int]:
    """The ``cuda:egl`` device map: ``$BIGYM_AGENT_EGL_MAP``, else detected.

    Args:
        raw: The map text; None reads ``$BIGYM_AGENT_EGL_MAP`` and, when it
            is unset, :func:`detect_egl_map`.

    Returns:
        ``{nvidia-smi index: EGL index}``; empty means the EGL index is the
        ``nvidia-smi`` index.

    Raises:
        ValueError: The map is not a comma list of ``int:int`` pairs.
    """
    if raw is None:
        text = os.environ.get("BIGYM_AGENT_EGL_MAP")
        if text is None:
            return detect_egl_map()
    else:
        text = raw
    mapping: dict[int, int] = {}
    for pair in text.split(","):
        pair = pair.strip()
        if not pair:
            continue
        cuda, _, egl = pair.partition(":")
        if not egl:
            raise ValueError(f"bad EGL map entry {pair!r}: expected cuda:egl")
        mapping[int(cuda)] = int(egl)
    return mapping


def pick_egl(min_free_mib: int = 6000) -> tuple[int, int]:
    """Choose the EGL device of the GPU with the most free memory.

    Args:
        min_free_mib: Refuse to run when the best GPU has less free than this.

    Returns:
        ``(EGL index, nvidia-smi index)``.

    Raises:
        SystemExit: No GPU is visible, or none has enough free memory.
    """
    free = gpu_free_mib()
    mapping = egl_map()
    candidates = [
        (mib, gpu) for gpu, mib in free.items() if not mapping or gpu in mapping
    ]
    if not candidates:
        raise SystemExit("no GPU visible (nvidia-smi listed none)")
    mib, gpu = max(candidates)
    if mib < min_free_mib:
        raise SystemExit(
            f"no GPU with {min_free_mib} MiB free (best: gpu{gpu} {mib} MiB); "
            "a full GPU makes the EGL workers crash"
        )
    return mapping.get(gpu, gpu), gpu
