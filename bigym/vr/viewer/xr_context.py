"""Customized pyopenxr ContextObject."""

import platform

import xr
from xr import (
    EnvironmentBlendMode,
    FormFactor,
    InstanceCreateInfo,
    ReferenceSpaceCreateInfo,
    SessionCreateInfo,
    ViewConfigurationType,
)
from xr.utils.gl import ContextObject
from xr.utils.gl.glfw_util import GLFWOffscreenContextProvider

from bigym.vr.viewer.xr_input import XRInput

ALWAYS_DESTROY_INSTANCE_ON_EXIT = True


class XRContextObject(ContextObject):
    """Customized pyopenxr ContextObject.

    Notes:
        - Handles update loop of the XRInput object.
        - Fixes the issue of "hanging" when calling `xr.destroy_instance`.
    """

    # ContextObject declares these Optional: they are None outside the
    # context. XRInput and the renderer only run inside it, so they are
    # typed by what they hold there; __exit__ resets them.
    instance: xr.Instance
    session: xr.Session
    space: xr.Space
    default_action_set: xr.ActionSet
    input: XRInput

    def __init__(
        self,
        context_provider=None,
        instance_create_info=None,
        session_create_info=None,
        reference_space_create_info=None,
        view_configuration_type=ViewConfigurationType.PRIMARY_STEREO,
        environment_blend_mode=EnvironmentBlendMode.OPAQUE,
        form_factor=FormFactor.HEAD_MOUNTED_DISPLAY,
    ):
        """Init."""
        super().__init__(
            context_provider=context_provider or GLFWOffscreenContextProvider(),
            instance_create_info=instance_create_info or InstanceCreateInfo(),
            session_create_info=session_create_info or SessionCreateInfo(),
            reference_space_create_info=(
                reference_space_create_info or ReferenceSpaceCreateInfo()
            ),
            view_configuration_type=view_configuration_type,
            environment_blend_mode=environment_blend_mode,
            form_factor=form_factor,
        )

    def __enter__(self):
        """Initializes XRInput upon entering the context."""
        enter_result = super().__enter__()
        self.input = XRInput(self)
        return enter_result

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleans up XR resources upon exiting the context.

        Contains fix to prevent application hang on Linux: https://github.com/ValveSoftware/SteamVR-for-Linux/issues/422.
        """
        if self.default_action_set is not None:
            xr.destroy_action_set(self.default_action_set)
            self.default_action_set = None  # ty: ignore[invalid-assignment]
        if self.space is not None:
            xr.destroy_space(self.space)
            self.space = None  # ty: ignore[invalid-assignment]
        if self.session is not None:
            xr.destroy_session(self.session)
            self.session = None  # ty: ignore[invalid-assignment]
        if self.graphics is not None:
            self.graphics.destroy()
            self.graphics = None
        if self.instance is not None:
            # Workaround to prevent hang
            if ALWAYS_DESTROY_INSTANCE_ON_EXIT or platform.system() != "Linux":
                xr.destroy_instance(self.instance)
            self.instance = None  # ty: ignore[invalid-assignment]

    def frame_loop(self):
        """Runs the frame loop and updates XR input."""
        for frame_state in super().frame_loop():
            self.input.update(frame_state.predicted_display_time)
            yield frame_state
