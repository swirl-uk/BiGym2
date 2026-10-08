"""Shared mjviser workarounds for BiGym scenes.

The live VR spectator and the demo viewer render through the same fixed
pipeline. Apply the patches before constructing ``ViserMujocoScene``.
"""

from __future__ import annotations

import time
from typing import Any

import mjviser.conversions as conversions
import mjviser.scene as mjscene
import mujoco
import numpy as np
import trimesh
import trimesh.visual
import viser
from mjviser import ViserMujocoScene
from PIL import Image

from bigym.bigym_env import BiGymEnv
from bigym.envs.reach_target import Target, _ReachTargetEnv


def _geom_alpha(mj_model: Any, geom_id: int) -> float:
    """Alpha mjviser resolves for a geom: material rgba first, else geom rgba."""
    matid = int(mj_model.geom_matid[geom_id])
    if 0 <= matid < mj_model.nmat:
        return float(mj_model.mat_rgba[matid][3])
    return float(mj_model.geom_rgba[geom_id][3])


def patch_mjviser_uv_safe_merge() -> None:
    """Work around a texture-scrambling bug in mjviser's geom merging.

    mjviser meshes carry per-face-vertex UVs. Its fixed-body merge path
    (a) concatenates textured meshes that share a texture but have different
    materials — trimesh's material merge re-packs the UVs wrongly — and
    (b) runs merge_vertices(), which collapses co-located vertices down to
    one arbitrary UV each. Hard-edged atlas-mapped meshes (the BiGym table:
    every corner vertex is shared by faces on different atlas islands) come
    out covered in rainbow speckle; organically unwrapped meshes only break
    along UV seams, which is why props/robots look fine. Each textured geom
    in its own handle + no vertex merge renders clean.
    """
    if getattr(conversions, "_uv_safe_merge_patched", False):
        return

    orig_can_merge = conversions._can_merge_vertices
    conversions._can_merge_vertices = lambda mesh: (  # ty: ignore[invalid-assignment]
        not isinstance(mesh.visual, trimesh.visual.TextureVisuals)
        and orig_can_merge(mesh)
    )

    def group_textured_singletons(mj_model, geom_ids):
        # Textured meshes: one handle each (the atlas bug above). Textured
        # primitives (patch_mjviser_box_textures) carry generated UVs and an
        # identical material per texture, so trimesh's pack() returns that
        # material untouched; grouping them per texture means the image is
        # PNG-encoded once per handle instead of once per box (15 counter
        # boxes x 1756^2 otherwise take a drawer scene from 1.8 s to 11 s).
        subgroups, untextured = [], []
        by_texture: dict[int, list[int]] = {}
        for gid in geom_ids:
            texid = conversions.get_geom_texture_id(mj_model, gid)
            if 0.0 < _geom_alpha(mj_model, gid) < 1.0:
                # Translucent (patch_mjviser_translucency): its own handle,
                # so the blended material is not merged into opaque geoms.
                subgroups.append([gid])
            elif texid < 0:
                untextured.append(gid)
            elif int(mj_model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_MESH):
                subgroups.append([gid])
            else:
                by_texture.setdefault(int(texid), []).append(gid)
        subgroups.extend(by_texture.values())
        if untextured:
            subgroups.append(untextured)
        return subgroups

    conversions.group_geoms_by_visual_compat = (  # ty: ignore[invalid-assignment]
        group_textured_singletons
    )
    # scene.py binds the name at import time; patch its reference too.
    mjscene.group_geoms_by_visual_compat = (  # ty: ignore[invalid-assignment]
        group_textured_singletons
    )
    conversions._uv_safe_merge_patched = True  # ty: ignore[unresolved-attribute]


def patch_mjviser_translucency() -> None:
    """Draw geoms with 0 < alpha < 1 blended, as MuJoCo does.

    mjviser writes a geom's rgba into vertex colours, alpha included, but a
    glTF mesh with vertex alpha is drawn opaque unless its material says
    ``alphaMode: BLEND``. BiGym's glass cabinet doors (wall_cabinet_600
    ``glass`` material, alpha 0.5; the put_cups task) came out as solid
    grey panes. Give such geoms a blended PBR material carrying the rgba;
    the UV-safe grouping keeps each one in its own handle. Display only.
    """
    if getattr(conversions, "_translucency_patched", False):
        return

    orig_primitive = conversions.create_primitive_mesh
    orig_mesh = conversions.mujoco_mesh_to_trimesh

    def blended(mesh: Any, mj_model: Any, geom_id: int) -> Any:
        if isinstance(mesh.visual, trimesh.visual.TextureVisuals):
            return mesh  # textured: leave as is
        matid = int(mj_model.geom_matid[geom_id])
        rgba = (
            mj_model.mat_rgba[matid]
            if 0 <= matid < mj_model.nmat
            else mj_model.geom_rgba[geom_id]
        )
        material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=np.clip(np.asarray(rgba, dtype=np.float64), 0, 1),
            metallicFactor=0.0,
            roughnessFactor=1.0,
            alphaMode="BLEND",
            doubleSided=True,
        )
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.zeros((len(mesh.vertices), 2)), material=material
        )
        return mesh

    def create_primitive_mesh(mj_model: Any, geom_id: int) -> Any:
        mesh = orig_primitive(mj_model, geom_id)
        if 0.0 < _geom_alpha(mj_model, geom_id) < 1.0:
            return blended(mesh, mj_model, geom_id)
        return mesh

    def mujoco_mesh_to_trimesh(mj_model: Any, geom_idx: int) -> Any:
        mesh = orig_mesh(mj_model, geom_idx)
        if 0.0 < _geom_alpha(mj_model, geom_idx) < 1.0:
            return blended(mesh, mj_model, geom_idx)
        return mesh

    conversions.create_primitive_mesh = (  # ty: ignore[invalid-assignment]
        create_primitive_mesh
    )
    conversions.mujoco_mesh_to_trimesh = (  # ty: ignore[invalid-assignment]
        mujoco_mesh_to_trimesh
    )
    conversions._translucency_patched = True  # ty: ignore[unresolved-attribute]


def patch_mjviser_floor(
    plane_color: tuple[int, int, int] = (24, 24, 27),
    cell_color: tuple[int, int, int] = (52, 52, 58),
    section_color: tuple[int, int, int] = (74, 74, 82),
    cell_size: float = 0.30,
    section_size: float = 1.20,
) -> None:
    """Give the ground plane the paper's dark tiled look inside viser.

    mjviser replaces every ``mjGEOM_PLANE`` with ``scene.add_grid`` (scene.py,
    fixed-body pass) and passes no ``plane_color``, so the floor renders white
    at ``plane_opacity=0.4`` whatever the MJCF material says: MuJoCo's
    ``builtin="checker"`` texture is generated inside its own renderer and has
    no image a mesh viewer could read. Forcing the grid's own colours makes
    what you drag around in the browser match what the offscreen MuJoCo render
    puts in the figure.

    Display only — the model is untouched, so policy observations are
    unaffected.
    """
    if getattr(mjscene, "_floor_patched", False):
        return

    orig_add_grid = viser.SceneApi.add_grid

    def add_grid_dark(self, name, *args, **kwargs):
        if name.startswith("/fixed_bodies/"):
            kwargs.setdefault("plane_color", plane_color)
            kwargs.setdefault("cell_color", cell_color)
            kwargs.setdefault("section_color", section_color)
            kwargs.setdefault("cell_size", cell_size)
            kwargs.setdefault("section_size", section_size)
            kwargs["plane_opacity"] = 1.0
        return orig_add_grid(self, name, *args, **kwargs)

    viser.SceneApi.add_grid = add_grid_dark
    mjscene._floor_patched = True  # ty: ignore[unresolved-attribute]


def patch_mjviser_box_textures() -> None:
    """Texture boxes the way MuJoCo's own renderer does.

    mjviser's ``get_geom_texture_id`` returns -1 for every non-mesh geom, so
    a box whose material carries a 2D texture falls back to the material's
    flat rgba -- and BiGym leaves that at white. Across the 20 G1 tasks that
    is 136 boxes: every counter top and its four edge strips, the hobs, the
    wall cabinet's vent, the pick_box carton. The drawer
    fronts are meshes with UVs and were never affected.

    MuJoCo has no UVs for primitives either. Its classic renderer
    (``render_gl3.c``, the ``mjTEXTURE_2D`` branch of the texture setup)
    draws a box as the unit cube scaled by the half-sizes and lets OpenGL
    generate coordinates from object space::

        s =  0.5 * scl[0] * x - 0.5
        t = -0.5 * scl[1] * y - 0.5

    with ``scl = texrepeat``, multiplied by the x/y half-sizes when the
    material sets ``texuniform``. Only x and y enter, so the side faces of
    a box show the texture as streaks -- that is what MuJoCo shows too, and
    matching it is the point. Same ``(s, t)`` plus the same image (flipped
    to top-left origin, as mjviser already does for meshes) reproduces the
    offscreen render. Coordinates run outside [0, 1]; the glTF sampler
    default is REPEAT, which is what MuJoCo's ``GL_REPEAT`` does.

    Cube textures (the pick_box carton: ``type="cube"`` from one square
    image, so all six faces are the same picture) go through the same
    renderer's cube branch instead: the lookup direction is the object-space
    point, scaled by the half-sizes when ``texuniform``, and OpenGL's cube
    map rule picks the face by major axis and maps ``(sc, tc) / |ma|`` onto
    it. Each box face gets its own four vertices so that mapping can be
    baked as plain UVs into the face image; a cube whose faces differ is
    unrolled into one 6-high atlas with the v range split per face.

    Planes are untouched: mjviser replaces them with a grid and
    ``patch_mjviser_floor`` styles that on purpose.

    Also caches the PIL image per texture and asks Pillow for its fastest
    PNG level: trimesh's glTF export re-encodes the image for every handle
    that embeds it, and the 1756^2 counter texture costs 0.7 s per encode at
    the default level against 0.15 s at level 1 (same pixels; the file is
    a quarter larger). Fully opaque RGBA textures are sent as RGB.
    """
    if getattr(conversions, "_box_textures_patched", False):
        return

    orig_texture_id = conversions.get_geom_texture_id
    orig_primitive = conversions.create_primitive_mesh
    orig_extract = conversions._extract_texture_image
    image_cache: dict[tuple[int, int] | tuple[int, int, str], Any] = {}

    def extract_texture_image(mj_model: Any, texid: int) -> Any:
        key = (id(mj_model), int(texid))
        if key not in image_cache:
            image = orig_extract(mj_model, texid)
            if image is not None:
                if image.mode == "RGBA" and np.asarray(image)[..., 3].min() == 255:
                    image = image.convert("RGB")
                image.encoderinfo = {"compress_level": 1}
            image_cache[key] = image
        return image_cache[key]

    def box_2d_texture_id(mj_model: Any, geom_id: int) -> int:
        if int(mj_model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            return -1
        matid = int(mj_model.geom_matid[geom_id])
        if matid < 0 or matid >= mj_model.nmat:
            return -1
        texid = conversions._get_texture_id(mj_model, matid)
        if texid < 0:
            return -1
        tex_type = int(mj_model.tex_type[texid])
        if tex_type == int(mujoco.mjtTexture.mjTEXTURE_2D):
            return int(texid)
        if tex_type == int(mujoco.mjtTexture.mjTEXTURE_CUBE):
            # One square image (used for all six faces) or a 6-high stack.
            w, h = int(mj_model.tex_width[texid]), int(mj_model.tex_height[texid])
            if h in (w, 6 * w):
                return int(texid)
        return -1

    def cube_face_images(mj_model: Any, texid: int) -> tuple[Any, bool]:
        """Return (image, unrolled): one face if all six match, else the atlas."""
        key = (id(mj_model), int(texid), "cube")
        if key not in image_cache:
            w = int(mj_model.tex_width[texid])
            h = int(mj_model.tex_height[texid])
            nc = int(mj_model.tex_nchannel[texid])
            adr = int(mj_model.tex_adr[texid])
            if h == w:
                same, block = (
                    True,
                    mj_model.tex_data[adr : adr + w * w * nc].reshape(w, w, nc),
                )
            else:
                data = mj_model.tex_data[adr : adr + 6 * w * w * nc].reshape(
                    6, w, w, nc
                )
                same = all(np.array_equal(data[0], data[i]) for i in range(1, 6))
                block = data[0] if same else data.reshape(6 * w, w, nc)
            # GL cube faces are uploaded as stored, row 0 at t=0; glTF puts
            # v=0 on the first image row as well, so no flip here.
            arr = block.astype(np.uint8)
            image = (
                Image.fromarray(arr[..., 0], mode="L")
                if nc == 1
                else Image.fromarray(arr)
            )
            if image.mode == "RGBA" and arr[..., 3].min() == 255:
                image = image.convert("RGB")
            image.encoderinfo = {"compress_level": 1}
            image_cache[key] = (image, not same)
        return image_cache[key]

    # OpenGL cube-map face selection: (axis, sign) -> (sc, tc) as index/sign
    # pairs over the lookup direction r, with ma the major component.
    cube_faces = (
        # face, normal, sc=(idx, sign), tc=(idx, sign)
        (0, np.array([1.0, 0, 0]), (2, -1.0), (1, -1.0)),
        (1, np.array([-1.0, 0, 0]), (2, 1.0), (1, -1.0)),
        (2, np.array([0, 1.0, 0]), (0, 1.0), (2, 1.0)),
        (3, np.array([0, -1.0, 0]), (0, 1.0), (2, -1.0)),
        (4, np.array([0, 0, 1.0]), (0, 1.0), (1, -1.0)),
        (5, np.array([0, 0, -1.0]), (0, -1.0), (1, -1.0)),
    )

    def cube_textured_box(mj_model: Any, geom_id: int, texid: int) -> Any:
        matid = int(mj_model.geom_matid[geom_id])
        size = np.asarray(mj_model.geom_size[geom_id], dtype=np.float64)
        scale = size if mj_model.mat_texuniform[matid] else np.ones(3)
        image, unrolled = cube_face_images(mj_model, texid)
        verts, uvs, tris = [], [], []
        for face, normal, (si, ss), (ti, ts) in cube_faces:
            axis = int(np.argmax(np.abs(normal)))
            others = [a for a in range(3) if a != axis]
            for cu, cv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                unit = normal.copy()
                unit[others[0]], unit[others[1]] = cu, cv
                r = unit * scale
                ma = abs(r[axis])
                sc, tc = ss * r[si], ts * r[ti]
                u, v = 0.5 * (sc / ma + 1.0), 0.5 * (tc / ma + 1.0)
                if unrolled:
                    v = (face + v) / 6.0
                verts.append(unit * size)
                uvs.append((u, v))
            b = 4 * face
            # Wind so the triangle normal points outward along ``normal``.
            quad = [(b, b + 1, b + 2), (b, b + 2, b + 3)]
            p0, p1, p2 = (np.asarray(verts[i]) for i in quad[0])
            if np.dot(np.cross(p1 - p0, p2 - p0), normal) < 0:
                quad = [(b, b + 2, b + 1), (b, b + 3, b + 2)]
            tris.extend(quad)
        mesh = trimesh.Trimesh(
            vertices=np.asarray(verts), faces=np.asarray(tris), process=False
        )
        material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=mj_model.mat_rgba[matid],
            baseColorTexture=image,
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.asarray(uvs), material=material
        )
        return mesh

    def get_geom_texture_id(mj_model: Any, geom_idx: int) -> int:
        texid = orig_texture_id(mj_model, geom_idx)
        if texid >= 0:
            return texid
        return box_2d_texture_id(mj_model, geom_idx)

    def create_primitive_mesh(mj_model: Any, geom_id: int) -> trimesh.Trimesh:
        texid = box_2d_texture_id(mj_model, geom_id)
        if texid < 0:
            return orig_primitive(mj_model, geom_id)
        if int(mj_model.tex_type[texid]) == int(mujoco.mjtTexture.mjTEXTURE_CUBE):
            return cube_textured_box(mj_model, geom_id, texid)
        image = extract_texture_image(mj_model, texid)
        if image is None:
            return orig_primitive(mj_model, geom_id)

        matid = int(mj_model.geom_matid[geom_id])
        size = np.asarray(mj_model.geom_size[geom_id], dtype=np.float64)
        mesh = trimesh.creation.box(extents=2.0 * size)
        # Object-space coordinates of MuJoCo's unit cube, as fed to texgen.
        unit = mesh.vertices / np.where(size > 0, size, 1.0)
        scl = np.array(mj_model.mat_texrepeat[matid], dtype=np.float64)
        if mj_model.mat_texuniform[matid]:
            scl = scl * np.where(size[:2] > 0, size[:2], 1.0)
        uv = np.column_stack(
            (0.5 * scl[0] * unit[:, 0] - 0.5, -0.5 * scl[1] * unit[:, 1] - 0.5)
        )
        material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=mj_model.mat_rgba[matid],
            baseColorTexture=image,
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
        return mesh

    conversions._extract_texture_image = (  # ty: ignore[invalid-assignment]
        extract_texture_image
    )
    conversions.get_geom_texture_id = (  # ty: ignore[invalid-assignment]
        get_geom_texture_id
    )
    conversions.create_primitive_mesh = (  # ty: ignore[invalid-assignment]
        create_primitive_mesh
    )
    conversions._box_textures_patched = True  # ty: ignore[unresolved-attribute]


def tint_robot_geoms(model: Any, factor: float = 0.35) -> int:
    """Darken the robot's geoms so it separates from white kitchen fixtures.

    The G1 meshes are near-white and so are the cabinets, counters and
    crockery: 298 of a task's 339 geoms are plain grey rgba (0.5 or 0.7),
    and with viser's flat environment lighting the robot and the furniture
    read as one white mass; the tiles only work where something dark (a
    dishwasher basket's wire) happens to carry the contrast.

    Multiplies rgb by ``factor`` rather than assigning a flat colour, so the
    robot keeps its internal shading -- darker joints stay darker than the
    shells. Geoms are selected by kinematic root: every body under the
    robot's root shares ``body_rootid``, so no fixture is ever caught.

    Display only, and only inside this process: it edits the in-memory
    ``model.geom_rgba`` of the viewer's own env before mjviser converts the
    meshes. Nothing is written to disk, no recorded demonstration changes,
    and training and evaluation build their own envs.

    Returns the number of geoms tinted.
    """
    if factor >= 1.0:
        return 0

    def body_name(i: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""

    # A robot root is the kinematic root of the subtree holding the G1 pelvis.
    roots = set()
    for b in range(model.nbody):
        name = body_name(b)
        if name.endswith("/pelvis"):
            roots.add(int(model.body_rootid[b]))
    if not roots:
        return 0

    mask = np.isin(model.body_rootid[model.geom_bodyid], list(roots))
    model.geom_rgba[mask, :3] = np.clip(
        model.geom_rgba[mask, :3] * float(factor), 0.0, 1.0
    )
    return int(mask.sum())


def reach_targets(inner_env: BiGymEnv) -> list[Target]:
    """The target spheres of a reach task; empty for every other task."""
    if isinstance(inner_env, _ReachTargetEnv):
        return list(inner_env.targets)
    return []


def hide_reach_targets(inner_env: BiGymEnv) -> list[Target]:
    """Make a reach task's target geoms transparent; return the targets.

    mjviser bakes its meshes once, so a target moved by a later reset would
    leave a stale ball behind, and it skips transparent geoms at bake time.
    Viewers that show targets draw their own spheres from the live bodies.
    """
    targets = reach_targets(inner_env)
    for target in targets:
        colour = np.asarray(target._config.color_default, dtype=float)
        inner_env.model.bind(target.geom).rgba = np.append(colour[:3], 0.0)
    return targets


def find_dynamic_body_ids(model: Any, inner_env: BiGymEnv) -> tuple[int, ...]:
    """Return task-prop/target bodies that mjviser must update every frame."""
    roots = [prop.body for prop in inner_env._preset.get_props()]
    roots += [target.body for target in reach_targets(inner_env)]
    prop_roots = {inner_env.model.bind(body).id for body in roots}
    prop_bodies = set()
    for b in range(model.nbody):
        i = b
        while i > 0:
            if i in prop_roots:
                prop_bodies.add(b)
                break
            i = int(model.body_parentid[i])
    return tuple(sorted(prop_bodies))


def patch_mjviser() -> None:
    """Apply every scene-wide workaround; call before building a scene."""
    patch_mjviser_box_textures()
    patch_mjviser_translucency()
    patch_mjviser_uv_safe_merge()
    patch_mjviser_floor()


def patch_mjviser_track_body_ids(model: Any, dynamic_body_ids: tuple[int, ...]) -> None:
    """Make selected bodies dynamic in mjviser's scene conversion."""
    dynamic_bodies = set(map(int, dynamic_body_ids))
    orig = getattr(conversions, "_orig_is_fixed_body", conversions.is_fixed_body)
    conversions._orig_is_fixed_body = orig  # ty: ignore[unresolved-attribute]

    def is_fixed_body_except_props(mj_model, body_id):
        if int(body_id) in dynamic_bodies:
            return False
        return orig(mj_model, body_id)

    conversions.is_fixed_body = (  # ty: ignore[invalid-assignment]
        is_fixed_body_except_props
    )
    mjscene.is_fixed_body = is_fixed_body_except_props  # ty: ignore[invalid-assignment]


def patch_mjviser_track_props(model: Any, inner_env: Any) -> None:
    """Draw task props at their per-episode positions, not build-time ones.

    Props (racks, tables ...) are welded to the world, so mjviser's
    is_fixed_body() bakes them into the static world mesh at scene build.
    But the env's reset(seed) RANDOMIZES prop placement at the model level,
    so after the first episode load every static prop renders at a stale
    position. Un-fix all prop bodies — and reach
    targets, which randomize the same way — so mjviser gives them per-body
    dynamic nodes, updated from data.xpos on every frame like the robot.
    """
    patch_mjviser_track_body_ids(model, find_dynamic_body_ids(model, inner_env))


def open_viser_scene(
    model: Any,
    inner_env: Any | None,
    *,
    port: int,
    label: str | None = None,
    verbose: bool = False,
    port_retries: int = 20,
    dynamic_body_ids: tuple[int, ...] | None = None,
) -> tuple[Any, Any]:
    """Viser server + patched ViserMujocoScene, the shared bootstrap.

    Applies the compat patches (box textures, translucency, UV-safe merge,
    dark floor), retries the
    port briefly (a just-stopped
    server on the same port needs a moment to release the socket — the
    demo viewer hits this on every batch switch), and returns
    ``(server, scene)`` with camera tracking off. Callers that need to hide
    geoms from the baked static mesh (e.g. reach targets) must do so BEFORE
    calling this.
    """
    patch_mjviser()
    if dynamic_body_ids is not None:
        patch_mjviser_track_body_ids(model, dynamic_body_ids)
    elif inner_env is not None:
        patch_mjviser_track_props(model, inner_env)
    server = None
    for attempt in range(max(int(port_retries), 1)):
        try:
            server = viser.ViserServer(port=port, label=label, verbose=verbose)
            break
        except OSError:
            if attempt == port_retries - 1:
                raise
            time.sleep(0.25)
    assert server is not None
    scene = ViserMujocoScene(server=server, mj_model=model, num_envs=1)
    scene.camera_tracking_enabled = False
    return server, scene
