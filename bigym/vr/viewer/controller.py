"""State of a VR controller, as read from OpenXR each frame."""

from __future__ import annotations

from xr import Posef


class ControllerState:
    """State of the VR Controller."""

    def __init__(self):
        """Init."""
        # Inputs
        self.is_active: bool = False
        self.pose: Posef = Posef()
        self.pose_aim: Posef = Posef()
        self.trigger_click: bool = False
        self.trigger_changed: bool = False
        self.trigger_value: float = 0.0
        self.grip_value: float = 0.0
        self.a_click: bool = False
        self.a_changed: bool = False
        self.b_click: bool = False
        self.b_changed: bool = False
        self.thumbstick_x: float = 0.0
        self.thumbstick_y: float = 0.0
        self.thumbstick_click: bool = False
        self.thumbstick_click_changed: bool = False
        # Outputs
        self.vibration: bool = False

    @property
    def a_clicked(self):
        """True if A was clicked during this frame."""
        return self.a_click and self.a_changed

    @property
    def thumbstick_clicked(self):
        """True if the thumbstick was pressed during this frame."""
        return self.thumbstick_click and self.thumbstick_click_changed

    @property
    def b_clicked(self):
        """True if B was clicked during this frame."""
        return self.b_click and self.b_changed

    @property
    def trigger_clicked(self):
        """True if Trigger was clicked during this frame."""
        return self.trigger_click and self.trigger_changed
