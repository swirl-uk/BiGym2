"""Module for handling pyopenxr input."""

from __future__ import annotations

import ctypes
from _ctypes import POINTER, byref
from typing import TYPE_CHECKING

import xr
from xr import Posef, ReferenceSpaceType, Time, View

from bigym.vr.viewer import Side
from bigym.vr.viewer.controller import ControllerState

if TYPE_CHECKING:
    from bigym.vr.viewer.xr_context import XRContextObject


class XRInput:
    """Class for handling pyopenxr input interactions.

    XRInput processes pyopenxr events and maps it to `ControllerState` objects.
    """

    def __init__(self, context: XRContextObject):
        """Init."""
        self._context = context

        self._state: list[ControllerState] = [ControllerState(), ControllerState()]
        self._views: list[View] = [View(), View()]
        self._hmd_pose: Posef = Posef()

        self.hand_subaction_paths: dict[int, xr.Path] = {
            Side.LEFT: xr.string_to_path(self._context.instance, "/user/hand/left"),
            Side.RIGHT: xr.string_to_path(self._context.instance, "/user/hand/right"),
        }

        # Create actions
        self.action_pose = xr.create_action(
            action_set=self._context.default_action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="hand_pose",
                localized_action_name="Hand Pose",
                count_subaction_paths=len(self.hand_subaction_paths),
                subaction_paths=self.hand_subaction_paths.values(),
            ),
        )
        self.action_pose_aim = xr.create_action(
            action_set=self._context.default_action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="hand_pose_aim",
                localized_action_name="Hand Pose Aim",
                count_subaction_paths=len(self.hand_subaction_paths),
                subaction_paths=self.hand_subaction_paths.values(),
            ),
        )
        self.action_vibrate = xr.create_action(
            action_set=self._context.default_action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.VIBRATION_OUTPUT,
                action_name="hand_vibrate",
                localized_action_name="Hand Vibrate",
                count_subaction_paths=len(self.hand_subaction_paths),
                subaction_paths=self.hand_subaction_paths.values(),
            ),
        )
        self.action_trigger_click = xr.create_action(
            action_set=self._context.default_action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.BOOLEAN_INPUT,
                action_name="hand_trigger_click",
                localized_action_name="Hand Trigger Click",
                count_subaction_paths=len(self.hand_subaction_paths),
                subaction_paths=self.hand_subaction_paths.values(),
            ),
        )
        self.action_trigger_value = xr.create_action(
            action_set=self._context.default_action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.FLOAT_INPUT,
                action_name="hand_trigger_value",
                localized_action_name="Hand Trigger Value",
                count_subaction_paths=len(self.hand_subaction_paths),
                subaction_paths=self.hand_subaction_paths.values(),
            ),
        )
        self.action_grip_value = xr.create_action(
            action_set=self._context.default_action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.FLOAT_INPUT,
                action_name="hand_grip_value",
                localized_action_name="Hand Grip Value",
                count_subaction_paths=len(self.hand_subaction_paths),
                subaction_paths=self.hand_subaction_paths.values(),
            ),
        )
        self.action_a = xr.create_action(
            action_set=self._context.default_action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.BOOLEAN_INPUT,
                action_name="hand_a",
                localized_action_name="Hand A",
                count_subaction_paths=len(self.hand_subaction_paths),
                subaction_paths=self.hand_subaction_paths.values(),
            ),
        )
        self.action_b = xr.create_action(
            action_set=self._context.default_action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.BOOLEAN_INPUT,
                action_name="hand_b",
                localized_action_name="Hand B",
                count_subaction_paths=len(self.hand_subaction_paths),
                subaction_paths=self.hand_subaction_paths.values(),
            ),
        )
        self.action_thumbstick_x = xr.create_action(
            action_set=self._context.default_action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.FLOAT_INPUT,
                action_name="hand_thumbstick_x",
                localized_action_name="Hand Thumbstick X",
                count_subaction_paths=len(self.hand_subaction_paths),
                subaction_paths=self.hand_subaction_paths.values(),
            ),
        )
        self.action_thumbstick_y = xr.create_action(
            action_set=self._context.default_action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.FLOAT_INPUT,
                action_name="hand_thumbstick_y",
                localized_action_name="Hand Thumbstick Y",
                count_subaction_paths=len(self.hand_subaction_paths),
                subaction_paths=self.hand_subaction_paths.values(),
            ),
        )
        self.action_thumbstick_click = xr.create_action(
            action_set=self._context.default_action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.BOOLEAN_INPUT,
                action_name="hand_thumbstick_click",
                localized_action_name="Hand Thumbstick Click",
                count_subaction_paths=len(self.hand_subaction_paths),
                subaction_paths=self.hand_subaction_paths.values(),
            ),
        )

        # Interaction paths from Khronos documentation:
        # https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#semantic-path-interaction-profiles
        khr_select_path = [
            xr.string_to_path(
                self._context.instance, "/user/hand/left/input/select/click"
            ),
            xr.string_to_path(
                self._context.instance, "/user/hand/right/input/select/click"
            ),
        ]
        pose_path = [
            xr.string_to_path(
                self._context.instance, "/user/hand/left/input/grip/pose"
            ),
            xr.string_to_path(
                self._context.instance, "/user/hand/right/input/grip/pose"
            ),
        ]
        pose_aim_path = [
            xr.string_to_path(self._context.instance, "/user/hand/left/input/aim/pose"),
            xr.string_to_path(
                self._context.instance, "/user/hand/right/input/aim/pose"
            ),
        ]
        haptic_path = [
            xr.string_to_path(self._context.instance, "/user/hand/left/output/haptic"),
            xr.string_to_path(self._context.instance, "/user/hand/right/output/haptic"),
        ]
        trigger_value_path = [
            xr.string_to_path(
                self._context.instance, "/user/hand/left/input/trigger/value"
            ),
            xr.string_to_path(
                self._context.instance, "/user/hand/right/input/trigger/value"
            ),
        ]
        trigger_click_path = [
            xr.string_to_path(
                self._context.instance, "/user/hand/left/input/trigger/click"
            ),
            xr.string_to_path(
                self._context.instance, "/user/hand/right/input/trigger/click"
            ),
        ]
        grip_value_path = [
            xr.string_to_path(
                self._context.instance, "/user/hand/left/input/squeeze/value"
            ),
            xr.string_to_path(
                self._context.instance, "/user/hand/right/input/squeeze/value"
            ),
        ]
        a_click_path = [
            xr.string_to_path(self._context.instance, "/user/hand/left/input/a/click"),
            xr.string_to_path(self._context.instance, "/user/hand/right/input/a/click"),
        ]
        b_click_path = [
            xr.string_to_path(self._context.instance, "/user/hand/left/input/b/click"),
            xr.string_to_path(self._context.instance, "/user/hand/right/input/b/click"),
        ]
        thumbstick_x_path = [
            xr.string_to_path(
                self._context.instance, "/user/hand/left/input/thumbstick/x"
            ),
            xr.string_to_path(
                self._context.instance, "/user/hand/right/input/thumbstick/x"
            ),
        ]
        thumbstick_y_path = [
            xr.string_to_path(
                self._context.instance, "/user/hand/left/input/thumbstick/y"
            ),
            xr.string_to_path(
                self._context.instance, "/user/hand/right/input/thumbstick/y"
            ),
        ]
        thumbstick_click_path = [
            xr.string_to_path(
                self._context.instance, "/user/hand/left/input/thumbstick/click"
            ),
            xr.string_to_path(
                self._context.instance, "/user/hand/right/input/thumbstick/click"
            ),
        ]

        # Binding simple KHR controller
        khr_bindings = [
            xr.ActionSuggestedBinding(self.action_pose, pose_path[Side.LEFT]),
            xr.ActionSuggestedBinding(self.action_pose, pose_path[Side.RIGHT]),
            xr.ActionSuggestedBinding(self.action_pose_aim, pose_aim_path[Side.LEFT]),
            xr.ActionSuggestedBinding(self.action_pose_aim, pose_aim_path[Side.RIGHT]),
            xr.ActionSuggestedBinding(self.action_vibrate, haptic_path[Side.LEFT]),
            xr.ActionSuggestedBinding(self.action_vibrate, haptic_path[Side.RIGHT]),
            xr.ActionSuggestedBinding(
                self.action_trigger_click, khr_select_path[Side.LEFT]
            ),
            xr.ActionSuggestedBinding(
                self.action_trigger_click, khr_select_path[Side.RIGHT]
            ),
        ]
        xr.suggest_interaction_profile_bindings(
            instance=self._context.instance,
            suggested_bindings=xr.InteractionProfileSuggestedBinding(
                interaction_profile=xr.string_to_path(
                    self._context.instance,
                    "/interaction_profiles/khr/simple_controller",
                ),
                count_suggested_bindings=len(khr_bindings),
                suggested_bindings=(xr.ActionSuggestedBinding * len(khr_bindings))(
                    *khr_bindings
                ),
            ),
        )
        # Bindings for Valve Index
        valve_bindings = [
            xr.ActionSuggestedBinding(self.action_pose, pose_path[Side.LEFT]),
            xr.ActionSuggestedBinding(self.action_pose, pose_path[Side.RIGHT]),
            xr.ActionSuggestedBinding(self.action_pose_aim, pose_aim_path[Side.LEFT]),
            xr.ActionSuggestedBinding(self.action_pose_aim, pose_aim_path[Side.RIGHT]),
            xr.ActionSuggestedBinding(self.action_vibrate, haptic_path[Side.LEFT]),
            xr.ActionSuggestedBinding(self.action_vibrate, haptic_path[Side.RIGHT]),
            xr.ActionSuggestedBinding(
                self.action_trigger_click, trigger_click_path[Side.LEFT]
            ),
            xr.ActionSuggestedBinding(
                self.action_trigger_click, trigger_click_path[Side.RIGHT]
            ),
            xr.ActionSuggestedBinding(
                self.action_trigger_value, trigger_value_path[Side.LEFT]
            ),
            xr.ActionSuggestedBinding(
                self.action_trigger_value, trigger_value_path[Side.RIGHT]
            ),
            xr.ActionSuggestedBinding(
                self.action_grip_value, grip_value_path[Side.LEFT]
            ),
            xr.ActionSuggestedBinding(
                self.action_grip_value, grip_value_path[Side.RIGHT]
            ),
            xr.ActionSuggestedBinding(self.action_a, a_click_path[Side.LEFT]),
            xr.ActionSuggestedBinding(self.action_a, a_click_path[Side.RIGHT]),
            xr.ActionSuggestedBinding(self.action_b, b_click_path[Side.LEFT]),
            xr.ActionSuggestedBinding(self.action_b, b_click_path[Side.RIGHT]),
            xr.ActionSuggestedBinding(
                self.action_thumbstick_x, thumbstick_x_path[Side.LEFT]
            ),
            xr.ActionSuggestedBinding(
                self.action_thumbstick_x, thumbstick_x_path[Side.RIGHT]
            ),
            xr.ActionSuggestedBinding(
                self.action_thumbstick_y, thumbstick_y_path[Side.LEFT]
            ),
            xr.ActionSuggestedBinding(
                self.action_thumbstick_y, thumbstick_y_path[Side.RIGHT]
            ),
            xr.ActionSuggestedBinding(
                self.action_thumbstick_click, thumbstick_click_path[Side.LEFT]
            ),
            xr.ActionSuggestedBinding(
                self.action_thumbstick_click, thumbstick_click_path[Side.RIGHT]
            ),
        ]
        xr.suggest_interaction_profile_bindings(
            instance=self._context.instance,
            suggested_bindings=xr.InteractionProfileSuggestedBinding(
                interaction_profile=xr.string_to_path(
                    self._context.instance,
                    "/interaction_profiles/valve/index_controller",
                ),
                count_suggested_bindings=len(valve_bindings),
                suggested_bindings=(xr.ActionSuggestedBinding * len(valve_bindings))(
                    *valve_bindings
                ),
            ),
        )
        # Bindings for Meta Quest Touch Controller (Quest 2/3/Pro)
        # Note: Oculus Touch has no trigger/click, left hand has x/y instead of a/b
        meta_touch_x_click_path = xr.string_to_path(
            self._context.instance, "/user/hand/left/input/x/click"
        )
        meta_touch_y_click_path = xr.string_to_path(
            self._context.instance, "/user/hand/left/input/y/click"
        )
        meta_touch_bindings = [
            xr.ActionSuggestedBinding(self.action_pose, pose_path[Side.LEFT]),
            xr.ActionSuggestedBinding(self.action_pose, pose_path[Side.RIGHT]),
            xr.ActionSuggestedBinding(self.action_pose_aim, pose_aim_path[Side.LEFT]),
            xr.ActionSuggestedBinding(self.action_pose_aim, pose_aim_path[Side.RIGHT]),
            xr.ActionSuggestedBinding(self.action_vibrate, haptic_path[Side.LEFT]),
            xr.ActionSuggestedBinding(self.action_vibrate, haptic_path[Side.RIGHT]),
            xr.ActionSuggestedBinding(
                self.action_trigger_value, trigger_value_path[Side.LEFT]
            ),
            xr.ActionSuggestedBinding(
                self.action_trigger_value, trigger_value_path[Side.RIGHT]
            ),
            xr.ActionSuggestedBinding(
                self.action_grip_value, grip_value_path[Side.LEFT]
            ),
            xr.ActionSuggestedBinding(
                self.action_grip_value, grip_value_path[Side.RIGHT]
            ),
            # Left hand: x->a, y->b; Right hand: a, b
            xr.ActionSuggestedBinding(self.action_a, meta_touch_x_click_path),
            xr.ActionSuggestedBinding(self.action_a, a_click_path[Side.RIGHT]),
            xr.ActionSuggestedBinding(self.action_b, meta_touch_y_click_path),
            xr.ActionSuggestedBinding(self.action_b, b_click_path[Side.RIGHT]),
            xr.ActionSuggestedBinding(
                self.action_thumbstick_x, thumbstick_x_path[Side.LEFT]
            ),
            xr.ActionSuggestedBinding(
                self.action_thumbstick_x, thumbstick_x_path[Side.RIGHT]
            ),
            xr.ActionSuggestedBinding(
                self.action_thumbstick_y, thumbstick_y_path[Side.LEFT]
            ),
            xr.ActionSuggestedBinding(
                self.action_thumbstick_y, thumbstick_y_path[Side.RIGHT]
            ),
            xr.ActionSuggestedBinding(
                self.action_thumbstick_click, thumbstick_click_path[Side.LEFT]
            ),
            xr.ActionSuggestedBinding(
                self.action_thumbstick_click, thumbstick_click_path[Side.RIGHT]
            ),
        ]
        xr.suggest_interaction_profile_bindings(
            instance=self._context.instance,
            suggested_bindings=xr.InteractionProfileSuggestedBinding(
                interaction_profile=xr.string_to_path(
                    self._context.instance,
                    "/interaction_profiles/oculus/touch_controller",
                ),
                count_suggested_bindings=len(meta_touch_bindings),
                suggested_bindings=(
                    xr.ActionSuggestedBinding * len(meta_touch_bindings)
                )(*meta_touch_bindings),
            ),
        )

        self.grip_spaces: dict[int, xr.Space] = {
            Side.LEFT: xr.create_action_space(
                session=self._context.session,
                create_info=xr.ActionSpaceCreateInfo(
                    action=self.action_pose,
                    subaction_path=self.hand_subaction_paths[Side.LEFT],
                ),
            ),
            Side.RIGHT: xr.create_action_space(
                session=self._context.session,
                create_info=xr.ActionSpaceCreateInfo(
                    action=self.action_pose,
                    subaction_path=self.hand_subaction_paths[Side.RIGHT],
                ),
            ),
        }

        self.aim_spaces: dict[int, xr.Space] = {
            Side.LEFT: xr.create_action_space(
                session=self._context.session,
                create_info=xr.ActionSpaceCreateInfo(
                    action=self.action_pose_aim,
                    subaction_path=self.hand_subaction_paths[Side.LEFT],
                ),
            ),
            Side.RIGHT: xr.create_action_space(
                session=self._context.session,
                create_info=xr.ActionSpaceCreateInfo(
                    action=self.action_pose_aim,
                    subaction_path=self.hand_subaction_paths[Side.RIGHT],
                ),
            ),
        }

        self.hmd_space = xr.create_reference_space(
            session=self._context.session,
            create_info=xr.ReferenceSpaceCreateInfo(
                reference_space_type=ReferenceSpaceType.VIEW
            ),
        )

    @property
    def state(self) -> list[ControllerState]:
        """Current state of controllers."""
        return self._state

    @property
    def views(self) -> list[View]:
        """Current state of views."""
        return self._views

    @property
    def hmd_pose(self) -> Posef:
        """Current HMD pose."""
        return self._hmd_pose

    def update(self, time: Time):
        """Update process active input and update `ControllerState`."""
        self._hmd_pose = xr.locate_space(
            space=self.hmd_space, base_space=self._context.space, time=time
        ).pose

        _, self._views = xr.locate_views(
            session=self._context.session,
            view_locate_info=xr.ViewLocateInfo(
                view_configuration_type=self._context.view_configuration_type,
                display_time=time,
                space=self._context.space,
            ),
        )

        active_action_set = xr.ActiveActionSet(
            action_set=self._context.default_action_set,
            subaction_path=xr.NULL_PATH,  # ty: ignore[invalid-argument-type]
        )

        # Silencing this exception similarly to pyopenxr_examples repo:
        # https://github.com/cmbruns/pyopenxr_examples/blob/3149bc0853e9306063f4185b3c9d82683518669d/xr_examples/hello_xr/main.py#L96
        try:
            xr.sync_actions(
                session=self._context.session,
                sync_info=xr.ActionsSyncInfo(
                    count_active_action_sets=1,
                    active_action_sets=ctypes.pointer(active_action_set),
                ),
            )
        except xr.exception.SessionNotFocused:
            pass

        for side in Side:
            state = self._state[side]

            # Check if controller is available
            is_active = self._get_action_state_pose(self.action_pose, side)
            state.is_active = is_active
            if not is_active:
                continue

            # Update pose
            state.pose = self._get_space_pose(time, self.grip_spaces[side])
            state.pose_aim = self._get_space_pose(time, self.aim_spaces[side])

            # Update trigger click
            trigger_click_current = self._get_action_state_bool(
                self.action_trigger_click, side
            )
            state.trigger_changed = trigger_click_current != state.trigger_click
            state.trigger_click = trigger_click_current
            # Update trigger pressure
            state.trigger_value = self._get_action_state_float(
                self.action_trigger_value, side
            )
            # Update side-grip/squeeze pressure
            state.grip_value = self._get_action_state_float(
                self.action_grip_value, side
            )
            # Update A-button click
            a_click_current = self._get_action_state_bool(self.action_a, side)
            state.a_changed = a_click_current != state.a_click
            state.a_click = a_click_current
            # Update B-button click
            b_click_current = self._get_action_state_bool(self.action_b, side)
            state.b_changed = b_click_current != state.b_click
            state.b_click = b_click_current
            # Update Thumbstick X
            state.thumbstick_x = self._get_action_state_float(
                self.action_thumbstick_x, side
            )
            # Update Thumbstick Y
            state.thumbstick_y = self._get_action_state_float(
                self.action_thumbstick_y, side
            )
            # Update Thumbstick click
            stick_click_current = self._get_action_state_bool(
                self.action_thumbstick_click, side
            )
            state.thumbstick_click_changed = (
                stick_click_current != state.thumbstick_click
            )
            state.thumbstick_click = stick_click_current

            # Apply vibration
            if state.vibration:
                self._apply_vibration(self.action_vibrate, side)
                state.vibration = False

    def _get_action_state_float(self, action: xr.Action, hand: Side) -> float:
        state = xr.get_action_state_float(
            self._context.session,
            xr.ActionStateGetInfo(
                action=action,
                subaction_path=self.hand_subaction_paths[hand],
            ),
        )
        return state.current_state if state.is_active else 0

    def _get_action_state_bool(self, action: xr.Action, hand: Side) -> bool:
        state = xr.get_action_state_boolean(
            self._context.session,
            xr.ActionStateGetInfo(
                action=action,
                subaction_path=self.hand_subaction_paths[hand],
            ),
        )
        return state.current_state if state.is_active else False

    def _get_action_state_pose(self, action: xr.Action, hand: Side) -> bool:
        state = xr.get_action_state_pose(
            self._context.session,
            xr.ActionStateGetInfo(
                action=action,
                subaction_path=self.hand_subaction_paths[hand],
            ),
        )
        return state.is_active

    def _get_space_pose(self, time: Time, space: xr.Space) -> Posef:
        hand_space = xr.locate_space(
            space=space,
            base_space=self._context.space,
            time=time,
        )
        location_flags = hand_space.location_flags
        if (
            location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT != 0
            and location_flags & xr.SPACE_LOCATION_ORIENTATION_VALID_BIT != 0
        ):
            return hand_space.pose
        return Posef()

    def _apply_vibration(self, action: xr.Action, hand: Side):
        vibration = xr.HapticVibration(
            amplitude=0.5,
            duration=xr.MIN_HAPTIC_DURATION,  # ty: ignore[invalid-argument-type]
            frequency=xr.FREQUENCY_UNSPECIFIED,
        )
        xr.apply_haptic_feedback(
            session=self._context.session,
            haptic_action_info=xr.HapticActionInfo(
                action=action,
                subaction_path=self.hand_subaction_paths[hand],
            ),
            haptic_feedback=ctypes.cast(
                byref(vibration), POINTER(xr.HapticBaseHeader)
            ).contents,
        )
