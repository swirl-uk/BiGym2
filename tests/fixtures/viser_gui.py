"""A stand-in for the part of a viser server the viewer's panels use."""

from __future__ import annotations

from typing import Any

import numpy as np


class FakeGuiFolder:
    """Context manager stand-in for ``server.gui.add_folder``.

    Real viser folders can be re-entered to append more widgets, relabelled
    and removed, which is how the Compare panel adds and drops slots; the
    fake records all three.
    """

    def __init__(self, label="", expand_by_default=True, visible=True):
        """Remember the folder's label and that it is still there."""
        self.label = label
        self.expand_by_default = expand_by_default
        self.visible = visible
        self.removed = False

    def __enter__(self):
        """Enter the folder scope."""
        return self

    def __exit__(self, *exc):
        """Leave the folder scope."""
        return False

    def remove(self):
        """Drop the folder from the sidebar."""
        self.removed = True


class FakeHandle:
    """A viser GUI handle that remembers what was assigned to it.

    The fake never fires a callback by itself; a test calls :meth:`fire`
    where the browser would, which is also the contract the panels rely on
    (callbacks may only set a flag).
    """

    value: Any
    options: list
    content: str
    image: np.ndarray

    def __init__(self, **kwargs):
        """Store the widget's initial attributes."""
        self.__dict__.update(kwargs)
        self.visible = True
        self.disabled = False
        self.removed = False
        self.callbacks: list = []

    def remove(self):
        """Drop the widget from the sidebar."""
        self.removed = True

    def on_update(self, callback):
        """Record an update callback."""
        self.callbacks.append(callback)

    def on_click(self, callback):
        """Record a click callback."""
        self.callbacks.append(callback)

    def fire(self, value=None):
        """Pretend the browser changed this widget."""
        if value is not None:
            self.value = value
        for callback in self.callbacks:
            callback(self)


class FakeGui:
    """The subset of ``server.gui`` the viewer's panels use.

    Labelled widgets are kept in ``handles`` by label and images in
    ``images``; HTML blocks and progress bars in ``htmls`` and ``bars``, in
    the order they were added.
    """

    def __init__(self):
        """Start with no widgets recorded."""
        self.handles: dict[str, FakeHandle] = {}
        self.images: dict[str, FakeHandle] = {}
        self.htmls: list[FakeHandle] = []
        self.folders: list[FakeGuiFolder] = []
        self.bars: list[FakeHandle] = []

    def add(self, label, **attributes) -> FakeHandle:
        """Create a handle and record it under its label."""
        handle = FakeHandle(**attributes)
        self.handles[str(label)] = handle
        return handle

    def add_folder(self, label, expand_by_default=True, visible=True):
        """Open a (fake) folder."""
        folder = FakeGuiFolder(label, expand_by_default, visible)
        self.folders.append(folder)
        return folder

    def add_progress_bar(self, value=0.0, animated=False, color=None):
        """Add a progress-bar handle (value is 0-100, as viser's is)."""
        handle = FakeHandle(value=value, animated=animated, color=color)
        self.bars.append(handle)
        return handle

    def add_checkbox(self, label, initial_value=True):
        """Add a checkbox handle."""
        return self.add(label, value=initial_value)

    def add_html(self, content=""):
        """Add an HTML handle."""
        handle = FakeHandle(content=content)
        self.htmls.append(handle)
        return handle

    def add_dropdown(self, label, options=(), initial_value=None):
        """Add a dropdown handle."""
        options = list(options)
        value = initial_value if initial_value is not None else options[0]
        return self.add(label, options=options, value=value)

    def add_text(self, label, initial_value=""):
        """Add a text handle."""
        return self.add(label, value=initial_value)

    def add_number(self, label, initial_value=0.0, min=None, step=None):
        """Add a number handle."""
        return self.add(label, value=initial_value)

    def add_button(self, label):
        """Add a button handle."""
        return self.add(label, value=None)

    def add_image(self, image, label=None, format=None):
        """Add an image handle and record it under its label."""
        handle = FakeHandle(image=image, label=label, format=format)
        self.images[str(label)] = handle
        return handle


class FakeServer:
    """A viser server stand-in exposing only ``gui``."""

    def __init__(self):
        """Create the fake GUI."""
        self.gui = FakeGui()
