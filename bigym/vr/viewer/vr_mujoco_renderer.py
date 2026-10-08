"""VR Mujoco renderer class rendering mujoco environment to VR headset."""

from typing import Any, Optional

import mujoco
import numpy as np
from xr import FrameState, Posef, Quaternionf, Vector3f, View
from xr.utils import Matrix4x4f

from bigym.vr.viewer import Side
from bigym.vr.viewer.full_screen_renderer import VRFullScreenRenderer
from bigym.vr.viewer.pyopenxr_to_mujoco_converter import (
    matrix_as_flat16,
    rotate_vector_yaw,
    vector_from_pyopenxr,
)
from bigym.vr.viewer.xr_context import XRContextObject

RENDER_REFLECTIONS = False
RENDER_SHADOWS = False
RENDER_FOG = False


class Renderer(mujoco.Renderer):
    """Customized mujoco.Renderer with decreased font size."""

    _gl_context: Any
    _rect: mujoco.MjrRect
    _scene: mujoco.MjvScene

    def __init__(
        self,
        model: mujoco.MjModel,
        height: int = 240,
        width: int = 320,
        max_geom: int = 10000,
    ) -> None:
        """Init."""
        super().__init__(model, height, width, max_geom)
        self._mjr_context = mujoco.MjrContext(
            model, mujoco.mjtFontScale.mjFONTSCALE_50.value
        )
        self._mjr_context.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR
        mujoco.mjr_setBuffer(
            mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self._mjr_context
        )

    def render_rgb_raw(self, out: np.ndarray) -> np.ndarray:
        """RGB render into `out` WITHOUT the vertical flip.

        The stock render() flips the readback top-down in place; the eye quad
        samples the raw bottom-up mjr_readPixels row order as-is, so skipping
        the flip saves two full-frame CPU copies per frame.
        """
        if self._gl_context:
            self._gl_context.make_current()
        mujoco.mjr_render(self._rect, self._scene, self._mjr_context)
        mujoco.mjr_readPixels(out, None, self._rect, self._mjr_context)
        return out


class VRMujocoRenderer:
    """VR Mujoco renderer class rendering mujoco environment to VR headset."""

    def __init__(
        self, model: mujoco.MjModel, data: mujoco.MjData, height: int, width: int
    ):
        """Render ``data`` of ``model`` at ``width`` x ``height`` (both eyes)."""
        self._model = model
        self._data = data
        self._width = width
        self._height = height

        self._markers = []

        self._renderer = Renderer(self._model, self._height, self._width)
        self._renderer.scene.stereo = mujoco.mjtStereo.mjSTEREO_QUADBUFFERED
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = int(
            RENDER_REFLECTIONS
        )
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(RENDER_SHADOWS)
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = int(RENDER_FOG)
        self._vr_camera = mujoco.MjvCamera()
        # Geoms hidden from the FIRST-PERSON eye render only (e.g. the
        # robot's own head shell, which otherwise surrounds the viewpoint
        # whenever the operator drifts a few cm behind the recenter pose).
        # Recorded observations use the env's own offscreen renderer and
        # are untouched.
        self._first_person_hidden_geoms: set[int] = set()

        # Preallocated mjr_readPixels target for render_rgb_raw: no
        # per-frame allocation, no CPU flips (see Renderer.render_rgb_raw).
        self._pixel_buf = np.empty((self._height, self._width, 3), dtype=np.uint8)

        # Will be initialized after creation of the VR session
        self._context: Optional[XRContextObject] = None
        self._headset_renderer: Optional[VRFullScreenRenderer] = None

    @property
    def _scene(self) -> mujoco.MjvScene:
        return self._renderer.scene

    def set_context(self, context: XRContextObject):
        """Set context of VR application."""
        self._context = context
        self._headset_renderer = VRFullScreenRenderer(self._width // 2, self._height)

    def show_stats(
        self,
        info: dict[str, Any],
        pos: np.ndarray,
        spacing: np.ndarray | None = None,
    ):
        """Show label with information from dictionary."""
        if spacing is None:
            spacing = np.array([0, 0, -0.1])
        label_pos = pos.copy()
        for key, value in info.items():
            if isinstance(value, float):
                value = f"{value:.2f}"
            else:
                value = str(value)
            label = f"{key}: {value}"
            self.add_marker(pos=label_pos.copy(), label=label)
            label_pos += spacing

    def hide_first_person_meshes(self, mesh_names) -> int:
        """Hide every geom built from the named meshes in the eye render.

        Names match the unscoped mesh name (scope prefixes are stripped).
        Returns the number of geoms hidden.
        """
        model = self._model
        names = {str(n) for n in mesh_names}
        hidden = set()
        for g in range(model.ngeom):
            mid = int(model.geom_dataid[g])
            if mid < 0:
                continue
            mesh = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid) or ""
            if mesh.split("/")[-1] in names:
                hidden.add(g)
        self._first_person_hidden_geoms = hidden
        return len(hidden)

    def add_marker(self, **marker_params):
        """Add marker to scene."""
        self._markers.append(marker_params)

    def render(self, frame_state: FrameState, offset: Posef, yaw_offset: float = 0.0):
        """Render current state of the environment to VR headset."""
        assert self._context is not None and self._headset_renderer is not None
        self._renderer.update_scene(self._data, self._vr_camera)
        self._sync_mujoco_vr_cameras_with_views(
            self._context.input.views, offset, yaw_offset
        )
        pixels = self._render_mujoco_env()
        for view_index, _ in enumerate(self._context.view_loop(frame_state)):
            self._headset_renderer.render(Side(view_index), pixels)

    def render_solid(
        self,
        frame_state: FrameState,
        color: tuple[int, int, int] = (12, 12, 16),
    ) -> None:
        """Submit a motionless solid stereo frame without rendering MuJoCo.

        This is used as a headset-only comfort curtain while reset physics
        settles. It deliberately does not touch the MuJoCo scene or any of
        the environment's recorded observation renderers.
        """
        if self._context is None or self._headset_renderer is None:
            raise RuntimeError("VR renderer context is not initialized")
        rgb = np.asarray(color, dtype=np.uint8)
        if rgb.shape != (3,):
            raise ValueError(f"solid render color must be RGB, got {color!r}")
        self._pixel_buf[...] = rgb
        for view_index, _ in enumerate(self._context.view_loop(frame_state)):
            self._headset_renderer.render(Side(view_index), self._pixel_buf)

    def _render_mujoco_env(self) -> np.ndarray:
        if self._first_person_hidden_geoms:
            scn = self._renderer.scene
            for i in range(scn.ngeom):
                g = scn.geoms[i]
                if (
                    g.objtype == mujoco.mjtObj.mjOBJ_GEOM
                    and int(g.objid) in self._first_person_hidden_geoms
                ):
                    g.rgba[3] = 0.0
        for marker in self._markers:
            self._add_marker_to_scene(marker)
        pixels = self._renderer.render_rgb_raw(self._pixel_buf)
        self._markers.clear()
        return pixels

    def _sync_mujoco_vr_cameras_with_views(
        self, views: list[View], offset: Posef, yaw_offset: float = 0.0
    ):
        for camera_id, camera in enumerate(self._scene.camera):
            view = views[camera_id]
            z_near, z_far = 0.01, 50.0
            tan_left, tan_right, tan_down, tan_up = np.tan(view.fov.as_numpy())

            # Setup camera frustum
            camera.frustum_bottom = -tan_down * z_near
            camera.frustum_top = -tan_up * z_near
            camera.frustum_center = 0.5 * (tan_left + tan_right) * z_near
            camera.frustum_near = z_near
            camera.frustum_far = z_far

            # Column-major view matrix
            orientation = Matrix4x4f.create_from_quaternion(view.pose.orientation)
            if offset.orientation != Quaternionf():
                orientation_offset = Matrix4x4f.create_from_quaternion(
                    offset.orientation
                )
                orientation = orientation @ orientation_offset
            orientation = matrix_as_flat16(orientation)
            # Forward is the 3rd column of the view matrix - elements [8], [9], [10]
            # Up is the 2nd column of the view matrix - elements [4], [6], [5]
            # Also we have to invert forward axis, according to the documentation:
            # https://mujoco.readthedocs.io/en/stable/programming/visualization.html
            camera.forward = rotate_vector_yaw(
                -vector_from_pyopenxr(orientation[8:11]), yaw_offset
            )
            camera.up = rotate_vector_yaw(
                vector_from_pyopenxr(orientation[4:7]), yaw_offset
            )
            camera.pos = rotate_vector_yaw(
                vector_from_pyopenxr(view.pose.position), yaw_offset
            )
            if offset.position != Vector3f():
                camera.pos += offset.position.as_numpy()

    def _add_marker_to_scene(self, marker: dict):
        if self._scene.ngeom >= self._scene.maxgeom:
            raise RuntimeError("Ran out of geoms. maxgeom: %d" % self._scene.maxgeom)

        g = self._scene.geoms[self._scene.ngeom]
        # default values.
        g.dataid = -1
        g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        # MuJoCo >= 3.2 replaced mjvGeom texid/texuniform/texrepeat with matid.
        g.matid = -1
        g.emission = 0
        g.specular = 0.5
        g.shininess = 0.5
        g.reflectance = 0
        g.type = mujoco.mjtGeom.mjGEOM_LABEL
        g.size[:] = np.ones(3) * 0.1
        g.mat[:] = np.eye(3)
        g.rgba[:] = np.ones(4)

        for key, value in marker.items():
            if isinstance(value, (int, float, mujoco.mjtGeom)):
                setattr(g, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(g, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, str):
                assert key == "label", "Only label is a string in mjtGeom."
                if value is None:
                    g.label[0] = 0
                else:
                    g.label = value
            elif hasattr(g, key):
                raise ValueError(
                    "mjtGeom has attr {} but type {} is invalid".format(
                        key, type(value)
                    )
                )
            else:
                raise ValueError("mjtGeom doesn't have field %s" % key)

        self._scene.ngeom += 1
