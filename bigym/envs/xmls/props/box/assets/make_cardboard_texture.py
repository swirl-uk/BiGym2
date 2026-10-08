"""Generate the procedural kraft-cardboard texture used by ``props/box/box.xml``.

The texture is generated rather than photographed so that ``pick_box`` /
``store_box`` ship under the repository license.

Run from anywhere; the PNG is written next to this file::

    uv run --no-sync python bigym/envs/xmls/props/box/assets/make_cardboard_texture.py

The output is deterministic (fixed seed) and tileable, so a material with
``texuniform="true"`` can repeat it across the box faces without visible seams
other than the intentional fold lines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 1024
SEED = 20260902

# Kraft-paper base colour.
BASE_RGB = np.array([190.0, 150.0, 105.0])
# Packing-tape strip colour (darker, slightly less saturated).
TAPE_RGB = np.array([158.0, 126.0, 92.0])


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def _tileable_value_noise(size: int, grid: int, rng: np.random.Generator) -> np.ndarray:
    """Bilinear value noise on a ``grid x grid`` lattice, wrapping at the edges."""
    lattice = rng.random((grid, grid))
    # Wrap one row/column around so the interpolation is periodic.
    lattice = np.concatenate([lattice, lattice[:1]], axis=0)
    lattice = np.concatenate([lattice, lattice[:, :1]], axis=1)

    coords = np.arange(size) * (grid / size)
    idx = np.floor(coords).astype(int)
    frac = _smoothstep(coords - idx)[:, None]

    rows = lattice[idx] * (1.0 - frac) + lattice[idx + 1] * frac
    frac_c = _smoothstep(coords - idx)[None, :]
    return rows[:, idx] * (1.0 - frac_c) + rows[:, idx + 1] * frac_c


def _fbm(
    size: int, rng: np.random.Generator, octaves=(4, 8, 16, 32, 64, 128, 256)
) -> np.ndarray:
    """Multi-octave noise in ``[-1, 1]``, zero mean."""
    total = np.zeros((size, size))
    norm = 0.0
    amplitude = 1.0
    for grid in octaves:
        total += amplitude * _tileable_value_noise(size, grid, rng)
        norm += amplitude
        amplitude *= 0.55
    total /= norm
    return (total - total.mean()) * 2.0


def build_texture(size: int = SIZE, seed: int = SEED) -> np.ndarray:
    """Return the tileable kraft-cardboard texture as an HxWx3 uint8 array."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / float(size)

    # --- kraft paper: low-amplitude multi-octave noise on a flat base --------
    fibre = _fbm(size, rng)
    mottle = _tileable_value_noise(size, 3, rng) - 0.5  # broad tonal variation
    shade = 1.0 + 0.055 * fibre + 0.045 * mottle

    # Fine paper grain (per-pixel), kept small so it does not alias at 84x84.
    shade += rng.normal(0.0, 0.004, (size, size))

    # --- corrugation: faint horizontal flute lines --------------------------
    flutes = 24.0  # ~1 flute per 10 mm at the material's texrepeat
    wobble = 0.012 * _tileable_value_noise(size, 8, rng)
    corrugation = np.sin(2.0 * np.pi * (yy + wobble) * flutes)
    shade += 0.022 * corrugation

    rgb = BASE_RGB[None, None, :] * shade[..., None]

    # --- packing tape across the middle -------------------------------------
    tape_centre = 0.5 + 0.010 * (_tileable_value_noise(size, 6, rng) - 0.5)
    tape_half_width = 0.042
    dist = np.abs(yy - tape_centre)
    tape = (dist < tape_half_width).astype(float)
    # Soften the torn edges by a couple of pixels.
    edge = np.clip((tape_half_width - dist) / (2.0 / size), 0.0, 1.0)
    tape *= edge

    tape_shade = 1.0 + 0.035 * fibre
    tape_rgb = TAPE_RGB[None, None, :] * tape_shade[..., None]
    # A brighter specular streak down the middle of the tape.
    tape_rgb = tape_rgb + 22.0 * np.exp(-((dist / 0.011) ** 2))[..., None]
    rgb = rgb * (1.0 - tape[..., None]) + tape_rgb * tape[..., None]

    # Darker lines where the tape edges lift off the cardboard.
    tape_edge_line = np.exp(-(((dist - tape_half_width) / (1.3 / size)) ** 2))
    rgb *= (1.0 - 0.20 * tape_edge_line)[..., None]

    # --- thin fold / edge seams near the tile border ------------------------
    inset = 0.012
    for axis in (xx, yy):
        for pos in (inset, 1.0 - inset):
            seam = np.exp(-(((axis - pos) / (1.4 / size)) ** 2))
            rgb *= (1.0 - 0.20 * seam)[..., None]
            # A slightly lighter highlight just inside each seam.
            hi = np.exp(
                -(
                    ((axis - (pos + (0.004 if pos < 0.5 else -0.004))) / (2.5 / size))
                    ** 2
                )
            )
            rgb *= (1.0 + 0.06 * hi)[..., None]

    return np.clip(rgb, 0, 255).astype(np.uint8)


def main() -> None:
    """Write ``cardboard_procedural.png`` next to this script."""
    out = Path(__file__).resolve().parent / "cardboard_procedural.png"
    Image.fromarray(build_texture()).save(out, optimize=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
