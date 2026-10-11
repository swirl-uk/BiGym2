"""Scene overlays that follow the episode on screen: velocity arrows, targets."""

from __future__ import annotations

import numpy as np
import trimesh
import viser.transforms as vtf

from bigym.envs.reach_target import Target
from bigym.vr.viewer.stores import Store


def rotation_z_to(direction: np.ndarray) -> vtf.SO3:
    """The rotation taking +z onto the unit vector ``direction``."""
    c = float(direction[2])
    if c > 0.9999:
        return vtf.SO3.identity()
    if c < -0.9999:
        return vtf.SO3.from_x_radians(np.pi)
    axis = np.cross([0.0, 0.0, 1.0], direction)
    axis /= np.linalg.norm(axis)
    return vtf.SO3.exp(axis * np.arccos(c))


def rgb255(colour) -> tuple[int, ...]:
    """A MuJoCo rgba (0-1 floats) as viser's 0-255 integers."""
    values = np.asarray(colour, dtype=float).reshape(-1)[:3]
    return tuple(int(round(255 * v)) for v in values)


class VelocityArrows:
    """Commanded and measured base velocity, drawn above the pelvis.

    Four arrows from the anchor A = pelvis + 0.575 m up: commanded and
    measured planar velocity (horizontal, 0.5 m per m/s) and commanded and
    measured yaw rate (vertical, 0.5 m per rad/s). Each is a shaft (80 % of
    the length, radius w) under a cone head (20 %, base radius 2w).
    """

    WIDTH = 0.015
    # commanded linear, measured linear, commanded yaw, measured yaw
    COLOURS = np.array(
        [(51, 51, 153), (0, 153, 255), (51, 153, 51), (0, 255, 102)], dtype=np.uint8
    )

    def __init__(self, server, model) -> None:
        """Find the floating-base joints and add the two batched meshes."""
        self.base_adr: dict[str, tuple[int, int]] = {}
        for j in range(model.njnt):
            name = model.jnt(j).name
            for suffix in ("pelvis_x", "pelvis_y", "pelvis_z", "pelvis_rz"):
                if name.endswith(suffix):
                    self.base_adr[suffix] = (
                        int(model.jnt_qposadr[j]),
                        int(model.jnt_dofadr[j]),
                    )
        self.available = {"pelvis_x", "pelvis_y"} <= self.base_adr.keys()
        shaft = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=16)
        shaft.apply_translation([0.0, 0.0, 0.5])  # spans z in [0, 1]
        head = trimesh.creation.cone(radius=2.0, height=1.0, sections=16)
        count = len(self.COLOURS)
        self.identity = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (count, 1))
        self.shafts = server.scene.add_batched_meshes_simple(
            "/velocity/shafts",
            shaft.vertices,
            shaft.faces,
            batched_wxyzs=self.identity,
            batched_positions=np.zeros((count, 3)),
            batched_scales=np.ones((count, 3)),
            batched_colors=self.COLOURS,
        )
        self.heads = server.scene.add_batched_meshes_simple(
            "/velocity/heads",
            head.vertices,
            head.faces,
            batched_wxyzs=self.identity,
            batched_positions=np.zeros((count, 3)),
            batched_scales=np.ones((count, 3)),
            batched_colors=self.COLOURS,
        )

    def pelvis_lookat(self, qpos: np.ndarray) -> tuple[float, float, float] | None:
        """A point above the pelvis for the free camera, or None without one."""
        if not self.available:
            return None
        adr = self.base_adr
        return (qpos[adr["pelvis_x"][0]], qpos[adr["pelvis_y"][0]], 0.9)

    def update(self, store: Store, frame: int, visible: bool) -> None:
        """Pose the arrows for ``frame``; hidden when asked or nothing is stored."""
        shown = (
            visible
            and self.available
            and store.cmd is not None
            and store.qvel is not None
        )
        self.shafts.visible = shown
        self.heads.visible = shown
        if not shown:
            return
        assert store.cmd is not None and store.qvel is not None
        adr = self.base_adr
        q = store.qpos[frame]
        pelvis_z = q[adr["pelvis_z"][0]] if "pelvis_z" in adr else 0.8
        yaw = q[adr["pelvis_rz"][0]] if "pelvis_rz" in adr else 0.0
        anchor = np.array(
            [q[adr["pelvis_x"][0]], q[adr["pelvis_y"][0]], pelvis_z + 0.575]
        )
        cmd = store.cmd[min(frame, len(store.cmd) - 1)]
        qv = store.qvel[min(frame, len(store.qvel) - 1)]
        c, s = np.cos(yaw), np.sin(yaw)
        yaw_rate = qv[adr["pelvis_rz"][1]] if "pelvis_rz" in adr else 0.0
        vectors = [
            0.5 * np.array([c * cmd[0] - s * cmd[1], s * cmd[0] + c * cmd[1], 0.0]),
            0.5 * np.array([qv[adr["pelvis_x"][1]], qv[adr["pelvis_y"][1]], 0.0]),
            0.5 * np.array([0.0, 0.0, cmd[2]]),
            0.5 * np.array([0.0, 0.0, yaw_rate]),
        ]
        count = len(self.COLOURS)
        positions = np.zeros((count, 3))
        wxyz = self.identity.copy()
        shaft_scales = np.full((count, 3), 1e-4)
        head_positions = np.tile(anchor, (count, 1))
        head_scales = np.full((count, 3), 1e-4)
        width = self.WIDTH
        for k, vector in enumerate(vectors):
            length = float(np.linalg.norm(vector))
            positions[k] = anchor
            if length < 0.01:  # collapse near-zero arrows instead of jittering
                continue
            direction = vector / length
            wxyz[k] = rotation_z_to(direction).wxyz
            shaft_scales[k] = (width, width, 0.8 * length)
            head_positions[k] = anchor + direction * 0.8 * length
            head_scales[k] = (width, width, 0.2 * length)
        self.shafts.batched_positions = positions
        self.shafts.batched_wxyzs = wxyz
        self.shafts.batched_scales = shaft_scales
        self.heads.batched_positions = head_positions
        self.heads.batched_wxyzs = wxyz
        self.heads.batched_scales = head_scales


class TargetSpheres:
    """Reach-target balls drawn by the viewer, with a glow when reached.

    They stand in for the model's own target geoms, which are hidden from
    mjviser: its meshes are baked once at scene build, so a reset moving a
    target would leave a stale ball behind.
    """

    def __init__(self, server, targets: list[Target]) -> None:
        """Add a ball and a hidden glow per target."""
        self.targets = targets
        self.balls, self.glows = [], []
        for i, target in enumerate(targets):
            radius = float(np.asarray(target._config.size).reshape(-1)[0])
            self.balls.append(
                server.scene.add_icosphere(
                    f"/targets/{i}/ball",
                    radius=radius,
                    color=rgb255(target._config.color_default),
                )
            )
            self.glows.append(
                server.scene.add_icosphere(
                    f"/targets/{i}/glow",
                    radius=radius * 1.35,
                    color=rgb255(target._config.color_highlight),
                    opacity=0.5,
                    visible=False,
                )
            )

    def measure(self, inner) -> list[tuple[bool, np.ndarray]]:
        """Each target's reached flag and position in the posed frame."""
        reached = [target.is_reached(inner.reach_tolerance) for target in self.targets]
        positions = [target.get_position() for target in self.targets]
        return list(zip(reached, positions, strict=True))

    def show(self, measured: list[tuple[bool, np.ndarray]]) -> None:
        """Move the balls and light up the reached ones."""
        for ball, glow, (hit, position) in zip(
            self.balls, self.glows, measured, strict=True
        ):
            ball.position = tuple(position)
            glow.position = tuple(position)
            glow.visible = bool(hit)
