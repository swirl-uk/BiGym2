"""The settings of ``bigym-view``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bigym.vr.viewer import video
from bigym.vr.viewer.agent.compare.record import RECORD_LABEL_SCALE
from bigym.vr.viewer.agent.runs import EVAL_SEED_LO
from bigym.vr.viewer.layout import COMPARE_GAP

# --follow rescans the cell this often by default (seconds).
FOLLOW_SECONDS = 5.0


RECORD_SECONDS = 10.0


@dataclass
class ViewConfig:
    """bigym-view settings."""

    demo_dir: list[Path] | None = None
    """One or more paths: a demo batch dir, any ancestor folder (batches are
    discovered recursively), a local copy of the dataset (LeRobot task
    folders), an agent cell, a session root of cells, or a directory holding
    such roots (default: the published demonstrations from the Hugging Face
    Hub)."""
    viewer: Literal["mujoco", "viser"] = "viser"
    """viser: browser UI, works over ssh -L; mujoco: native window on a local
    display."""
    port: int = 8080
    """viser only: HTTP port."""
    robot_tint: Literal["none", "dark"] = "none"
    """viser only: darken the robot's geoms so it separates from white
    fixtures (display only; the model and observations are untouched)."""
    figure_dir: str = "figure_captures"
    """viser only: where the Figure capture button writes its PNGs."""
    batch: str | None = None
    """Which discovered batch to open first: an index into the printed list,
    or a substring of its label (default: the first)."""
    task: str | None = None
    """Which dataset task to open first (published or a local copy; default:
    the first)."""
    episode: int = 0
    """Episode index to open first."""
    follow: bool = False
    """viser only: rescan the cell every --follow-seconds so a live session's
    new batches, policy versions and run.json show up."""
    follow_seconds: float = FOLLOW_SECONDS
    """How often --follow rescans."""
    compare: str | None = None
    """viser only: start in compare mode, e.g. --compare "s1:v008,s2:v007,s3:last"
    — a comma-separated list of up to 6 <session>:<version> slots (version:
    vNNN, NNN, last or submission; session matched by name or a unique
    substring). The task is the one the viewer would open, so point
    --demo-dir at it."""
    seed: int = EVAL_SEED_LO
    """The seed compare mode poses every slot on."""
    compare_gap: float = COMPARE_GAP
    """Metres between neighbouring compare slots along +y."""
    layout: Literal["auto", "row", "2 columns", "3 columns"] = "auto"
    """How compare mode places its slots: a column steps along +y and a row
    steps back along -x (auto: one row up to 3 slots, two columns from four)."""
    exit_after_seconds: float = 0.0
    """viser only: serve for this many seconds and exit (0 = until Ctrl-C).
    Headless smoke tests use it to check a cell renders."""
    record: str | None = None
    """Compare mode only: once the scene is built, record it from the overview
    camera to this H.264 mp4 and exit. A headless Chrome renders the frames
    unless --browser says otherwise."""
    record_seconds: float = RECORD_SECONDS
    """Length of the recording; the episodes play at their recorded speed and
    wrap around."""
    record_size: str = video.RECORD_SIZE
    """Frame size of the recording, WxH."""
    record_fps: float = 0.0
    """Frame rate written into the file (0: the episodes' own rate, so one
    video frame per episode step)."""
    browser: str = "headless"
    """Who renders a recording: headless starts a headless Chrome/Chromium
    when nobody is connected after 3s, none waits for you to open the URL,
    anything else is a browser executable."""
    record_then_stay: bool = False
    """Keep serving the scene after the recording instead of exiting."""
    record_view: Literal["overview", "oblique"] = "overview"
    """Where the recording camera stands: overview (the live compare's camera,
    behind the robots) or oblique (in front and to one side, lower and
    closer: hands and tables, not backs)."""
    record_zoom: float = 1.0
    """Narrow the recording lens by this factor (2 = twice as close)."""
    record_speed: float = 1.0
    """Playback speed of the recording (2 = twice real time)."""
    record_label_scale: float = RECORD_LABEL_SCALE
    """How much bigger the billboards are drawn in a recording than live."""

    def __post_init__(self) -> None:
        """Reject values the viewer cannot act on.

        Raises:
            ValueError: A flag is out of range or needs another one.
        """
        if self.record and not self.compare:
            raise ValueError("--record needs --compare")
        if self.episode < 0:
            raise ValueError(f"--episode must be >= 0, got {self.episode}")
        if self.follow_seconds <= 0.0:
            raise ValueError(
                f"--follow-seconds must be > 0, got {self.follow_seconds:g}"
            )
        for flag, value in (
            ("--exit-after-seconds", self.exit_after_seconds),
            ("--record-fps", self.record_fps),
        ):
            if value < 0.0:
                raise ValueError(f"{flag} must be >= 0, got {value:g}")
        for flag, value in (
            ("--record-seconds", self.record_seconds),
            ("--record-zoom", self.record_zoom),
            ("--record-speed", self.record_speed),
            ("--record-label-scale", self.record_label_scale),
        ):
            if value <= 0.0:
                raise ValueError(f"{flag} must be > 0, got {value:g}")
