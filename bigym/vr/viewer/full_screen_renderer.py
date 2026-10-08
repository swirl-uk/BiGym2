"""Renders a full-screen image from NumPy array data using OpenGL."""

import inspect

import numpy as np
from OpenGL import GL
from OpenGL.GL.shaders import compileProgram, compileShader

from bigym.vr.viewer import Side

VERTEX_SHADER = """
#version 150 core
out vec2 v_tex;

const vec2 pos[4]=vec2[4](vec2(-1.0, 1.0),
                          vec2(-1.0,-1.0),
                          vec2( 1.0, 1.0),
                          vec2( 1.0,-1.0));

void main()
{
    v_tex=0.5*pos[gl_VertexID] + vec2(0.5);
    gl_Position=vec4(pos[gl_VertexID], 0.0, 1.0);
}
"""
FRAGMENT_SHADER = """
#version 150 core
in vec2 v_tex;
uniform sampler2D texSampler;
out vec4 color;
void main()
{
    color=texture(texSampler, v_tex);
}
"""


class VRFullScreenRenderer:
    """Renders NumPy array to OpenGL texture."""

    def __init__(self, width: int, height: int):
        """Init.

        :param width (int): The width of the VR screen for one eye.
        :param height (int): The height of the VR screen for one eye.
        """
        self._width = width
        self._height = height
        self._shader = self._create_shader()
        self._vertex_array = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vertex_array)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0, 0, 0, 1)
        GL.glClearDepth(1.0)
        self._tex_ids = {
            Side.LEFT: self._create_texture(),
            Side.RIGHT: self._create_texture(),
        }

    @staticmethod
    def _create_shader() -> int:
        vertex_shader = compileShader(
            inspect.cleandoc(VERTEX_SHADER),
            GL.GL_VERTEX_SHADER,
        )
        fragment_shader = compileShader(
            inspect.cleandoc(FRAGMENT_SHADER),
            GL.GL_FRAGMENT_SHADER,
        )

        return compileProgram(vertex_shader, fragment_shader)

    def _create_texture(self) -> int:
        texid = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texid)
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP)
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP)
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        # Immutable-size storage allocated once; per-frame updates go through
        # glTexSubImage2D.
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGB,
            self._width,
            self._height,
            0,
            GL.GL_RGB,
            GL.GL_UNSIGNED_BYTE,
            None,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return texid

    def render(self, side: Side, pixels: np.ndarray):
        """Render pixels to the active buffer.

        Args:
            side: The side of the headset to render the pixels to.
            pixels: The FULL stereo pixel buffer (height x 2*eye_width x 3,
                bottom-up rows as returned by mjr_readPixels, NOT flipped).
        """
        GL.glClear(
            GL.GL_COLOR_BUFFER_BIT  # ty: ignore[unsupported-operator]
            | GL.GL_DEPTH_BUFFER_BIT
        )

        # `pixels` is the FULL stereo buffer straight from mjr_readPixels
        # (bottom-up rows, contiguous). The eye half is selected with
        # GL_UNPACK_ROW_LENGTH/SKIP_PIXELS and uploaded via glTexSubImage2D,
        # so no CPU slice/flip copies happen on this path.
        full_height, full_width = pixels.shape[0], pixels.shape[1]
        if full_height != self._height or full_width != 2 * self._width:
            raise ValueError(
                f"Expected stereo buffer {self._height}x{2 * self._width}, "
                f"got {full_height}x{full_width}"
            )
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_ids[side])
        # Orphan the storage before the upload (classic streaming pattern):
        # the previous frame's composite may still be reading this texture,
        # and glTexSubImage2D into in-use storage makes the driver stall the
        # queue (intermittent ~15 ms render spikes).
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGB,
            self._width,
            self._height,
            0,
            GL.GL_RGB,
            GL.GL_UNSIGNED_BYTE,
            None,
        )
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glPixelStorei(GL.GL_UNPACK_ROW_LENGTH, full_width)
        GL.glPixelStorei(
            GL.GL_UNPACK_SKIP_PIXELS, 0 if side == Side.LEFT else self._width
        )
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D,
            0,
            0,
            0,
            self._width,
            self._height,
            GL.GL_RGB,
            GL.GL_UNSIGNED_BYTE,
            pixels,
        )
        GL.glPixelStorei(GL.GL_UNPACK_ROW_LENGTH, 0)
        GL.glPixelStorei(GL.GL_UNPACK_SKIP_PIXELS, 0)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)

        # Render full-screen quad
        GL.glUseProgram(self._shader)
        GL.glBindVertexArray(self._vertex_array)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

        # Unbind texture
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
