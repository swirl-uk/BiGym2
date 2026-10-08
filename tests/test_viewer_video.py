"""Recording through a browser: the ffmpeg and headless-Chrome command lines."""

from __future__ import annotations

import numpy as np
import pytest

from bigym.vr.viewer import video


def test_the_ffmpeg_command_pipes_raw_rgb_at_the_recording_size():
    """Raw rgb24 in, H.264 yuv420p out, at the size and rate asked for."""
    argv = video.ffmpeg_command("/tmp/out.mp4", (1920, 1080), 50, binary="ffmpeg")
    assert argv[0] == "ffmpeg"
    assert argv[-1] == "/tmp/out.mp4"
    for pair in (
        ("-f", "rawvideo"),
        ("-pix_fmt", "rgb24"),
        ("-s", "1920x1080"),
        ("-r", "50"),
        ("-i", "-"),
        ("-c:v", "libx264"),
        ("-crf", "18"),
        ("-pix_fmt", "yuv420p"),
    ):
        assert pair[1] in argv and argv[argv.index(pair[1]) - 1] == pair[0]
    # The input size is declared before the input, the output format after.
    assert argv.index("rawvideo") < argv.index("-i") < argv.index("libx264")
    # A fractional rate is rounded to the integer ffmpeg wants.
    other = video.ffmpeg_command("o.mp4", (64, 48), 25.4, binary="ff")
    assert other[other.index("-r") + 1] == "25"
    assert other[other.index("-s") + 1] == "64x48"


def test_the_recording_size_is_parsed_as_width_by_height():
    """``--record-size 3840x2160`` is ``(width, height)``."""
    assert video.parse_record_size("3840x2160") == (3840, 2160)
    with pytest.raises(SystemExit):
        video.parse_record_size("4k")


def test_an_odd_frame_size_is_refused_before_ffmpeg_starts():
    """yuv420p cannot encode an odd side, and the message says so."""
    with pytest.raises(SystemExit):
        video.VideoSink("/tmp/x.mp4", (1921, 1080), 50)


def test_the_headless_browser_command_is_the_swiftshader_one():
    """Chrome renders the page with no display, no GPU and no sandbox."""
    argv = video.chrome_command("/usr/bin/google-chrome", "http://h:1", (800, 600))
    assert argv[0] == "/usr/bin/google-chrome"
    assert argv[-1] == "http://h:1"
    for flag in (
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu-sandbox",
        "--use-gl=angle",
        "--use-angle=swiftshader",
        "--enable-unsafe-swiftshader",
        "--window-size=800,600",
        "--remote-debugging-port=0",
    ):
        assert flag in argv


def test_the_headless_window_carries_the_recordings_aspect_ratio():
    """get_render is not capped by the window, so the window stays modest."""
    assert video.browser_window((1920, 1080)) == (1920, 1080)
    assert video.browser_window((3840, 2160)) == (1920, 1080)
    assert video.browser_window((640, 360)) == (640, 360)
    # Whatever the cap does, the ratio survives it.
    wide = video.browser_window((7680, 4320))
    assert wide[0] / wide[1] == pytest.approx(16 / 9)


def test_browser_none_waits_for_a_human():
    """``--browser none`` starts nothing; a bad path is refused."""
    assert video.browser_binary("none") == ""
    with pytest.raises(SystemExit):
        video.browser_binary("/definitely/not/a/browser")


def test_a_render_with_an_alpha_channel_becomes_rgb():
    """png transport returns RGBA; rawvideo takes rgb24."""
    rgba = np.zeros((4, 6, 4), dtype=np.uint8)
    assert video.rgb_frame(rgba).shape == (4, 6, 3)
    assert video.rgb_frame(np.zeros((4, 6, 3), np.uint8)).shape == (4, 6, 3)
