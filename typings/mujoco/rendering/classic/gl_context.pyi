"""
Exports GLContext for MuJoCo Python bindings.
"""
from __future__ import annotations
from typing import Any as GLContext
from typing import Any as _GLContext
import ctypes as ctypes
import os as os
import platform as platform
__all__: list[str] = ['GLContext', 'ctypes', 'os', 'platform']
_MUJOCO_GL: str
_SYSTEM: str
_VALID_MUJOCO_GL: tuple
