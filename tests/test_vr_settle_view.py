"""Headset settle curtain stays isolated from simulation and recording."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from bigym.vr.collect.config import CollectConfig
from bigym.vr.collect.session import CollectorSession, CollectorStats

# vr_mujoco_renderer imports pyopenxr at module level (the `vr` extra).
# Not importorskip: on macOS importing pyopenxr raises NotImplementedError.
try:
    import xr  # noqa: F401
except Exception:
    pytest.skip("needs the bigym[vr] extra (pyopenxr)", allow_module_level=True)
from bigym.vr.viewer.vr_mujoco_renderer import VRMujocoRenderer


class _SessionRendererStub:
    def __init__(self):
        self.solid_calls = []
        self.live_calls = []

    def render_solid(self, frame_state, color):
        self.solid_calls.append((frame_state, color))

    def render(self, *args, **kwargs):
        self.live_calls.append((args, kwargs))


def test_collector_uses_headset_only_curtain_during_settle():
    session = object.__new__(CollectorSession)
    session.config = CollectConfig(task="move_plate", settle_view="curtain")
    session.env = SimpleNamespace(  # ty: ignore[invalid-assignment]
        pending_reset_warmup_steps=4
    )
    stub = _SessionRendererStub()
    session._renderer = stub  # ty: ignore[invalid-assignment]
    session._space_offset = object()  # ty: ignore[invalid-assignment]

    frame_state = object()
    session._render(frame_state)

    assert len(stub.solid_calls) == 1
    assert stub.live_calls == []
    assert session.env.pending_reset_warmup_steps == 4


def test_live_debug_mode_keeps_the_existing_scene_render():
    session = object.__new__(CollectorSession)
    session.config = CollectConfig(task="move_plate", settle_view="live", hud="off")
    session.env = SimpleNamespace(  # ty: ignore[invalid-assignment]
        pending_reset_warmup_steps=4
    )
    session.stats = CollectorStats(rgb="OK")
    stub = _SessionRendererStub()
    session._renderer = stub  # ty: ignore[invalid-assignment]
    session._space_offset = object()  # ty: ignore[invalid-assignment]
    session._space_yaw_offset = 0.0

    frame_state = object()
    session._render(frame_state)

    assert stub.solid_calls == []
    assert len(stub.live_calls) == 1


class _XRContextStub:
    @staticmethod
    def view_loop(_frame_state):
        return (object(), object())


class _HeadsetRendererStub:
    def __init__(self):
        self.calls = []

    def render(self, side, pixels):
        self.calls.append((side, pixels.copy()))


def test_solid_render_submits_both_eyes_without_touching_mujoco():
    renderer = object.__new__(VRMujocoRenderer)
    renderer._context = _XRContextStub()  # ty: ignore[invalid-assignment]
    headset = _HeadsetRendererStub()
    renderer._headset_renderer = headset  # ty: ignore[invalid-assignment]
    renderer._pixel_buf = np.empty((2, 4, 3), dtype=np.uint8)

    frame_state: Any = object()
    renderer.render_solid(frame_state, color=(3, 5, 7))

    assert len(headset.calls) == 2
    for _side, pixels in headset.calls:
        np.testing.assert_array_equal(pixels, np.full((2, 4, 3), (3, 5, 7)))
