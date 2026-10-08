"""Render human demonstrations of a task to mp4 from the public dataset.

    bigym-agent demo-video --task move_plate --out DIR [--size 84x84]

The demonstrations ship as a Hugging Face dataset (a LeRobot v3 lossless
export per task, see :mod:`bigym.loco.demos.hub`). Every frame stores the full
simulator state (``full_qpos``), so the robot's own cameras can be re-rendered
at any resolution: :class:`bigym.loco.demos.rerender.BatchRenderer` builds the
recorded environment, writes each frame's ``full_qpos`` and renders the three
onboard cameras exactly the way the recording path did.

Two audiences, and they get different views:

- ``head``, ``left_wrist``, ``right_wrist`` are the robot's own cameras. They
  are the only views a demonstration handed to a coding agent may contain: an
  outside view would show it something no learned baseline (and no policy at
  evaluation time) ever sees.
- ``third`` is a free outside view for humans reading the results. The sandbox
  builder never writes it.

Encoding goes through ffmpeg. The ``imageio-ffmpeg`` wheel ships one; failing
that an ``ffmpeg`` binary on ``PATH`` is used.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cli import DemoVideoConfig, parse_command, parse_size

# Free outside cameras a scene may define, most specific first. Same list and
# the same free-camera fallback as the demo viewer (bigym/vr/viewer/view_demos.py),
# so "third" means the same view in both tools.
THIRD_PERSON_CAMERAS = ("rec_third", "third_person", "external")
ONBOARD_VIEWS = ("head", "left_wrist", "right_wrist")
DEFAULT_VIEWS = ("head", "left_wrist", "right_wrist")

# What each view shows, in words, for the documents built on top of a render.
VIEW_TEXT = {
    "head": "the robot's own head camera, first person",
    "left_wrist": "the robot's left wrist camera",
    "right_wrist": "the robot's right wrist camera",
    "third": "a third-person view from outside the robot",
}


@dataclass(frozen=True)
class DemoEpisode:
    """One demonstration episode of a task.

    Attributes:
        index: Episode index inside the dataset export.
        source_file: Name of the recording the episode came from.
        seed: The seed the episode was collected on (the object placement).
        length: Number of recorded control steps.
    """

    index: int
    source_file: str
    seed: int
    length: int


class TaskDemos:
    """The demonstrations of one task, read straight from the dataset export.

    Only the columns a re-render needs are touched (``episode_index`` and
    ``full_qpos``), so no camera PNG is ever decoded: reading the whole export
    back with :func:`bigym.loco.demos.dataset.load_episodes` would decode tens
    of thousands of images to render one video.
    """

    def __init__(self, task: str, task_dir: Path | str | None = None):
        """Locate the task's export, downloading it on first use.

        Args:
            task: Task name.
            task_dir: An already downloaded task folder, or None to fetch it
                from the Hub.

        Raises:
            bigym.loco.demos.hub.DemosUnavailableError: The dataset has no
                demonstrations for this task yet.
        """
        from bigym.loco.demos import hub

        self.task = task
        self.dir = Path(task_dir) if task_dir is not None else hub.task_dir(task)
        self._rows: dict[int, tuple[Path, int, int]] | None = None
        self._episodes: tuple[DemoEpisode, ...] | None = None
        self._metadata: dict | None = None
        self._init_states: dict[int, dict] = {}

    @property
    def metadata(self) -> dict:
        """The collector metadata of the export (task block, controller, stats)."""
        if self._metadata is None:
            from bigym.loco.demos.dataset import load_task_metadata

            self._metadata = load_task_metadata(self.dir)
        return self._metadata

    def _row_index(self) -> dict[int, tuple[Path, int, int]]:
        """Map each episode index to ``(parquet file, first row, row count)``."""
        if self._rows is None:
            import pyarrow.parquet as pq

            rows: dict[int, tuple[Path, int, int]] = {}
            files = sorted(self.dir.glob("data/chunk-*/file-*.parquet"))
            if not files:
                raise FileNotFoundError(f"no data parquet files under {self.dir}")
            for path in files:
                column = pq.read_table(path, columns=["episode_index"])
                index = column.column("episode_index").to_numpy()
                for episode in np.unique(index):
                    start = int(np.searchsorted(index, episode))
                    count = int((index == episode).sum())
                    rows[int(episode)] = (path, start, count)
            self._rows = rows
        return self._rows

    @property
    def episodes(self) -> tuple[DemoEpisode, ...]:
        """Every demonstration of the task, in dataset order."""
        if self._episodes is None:
            init_file = self.dir / "meta" / "episode_init_states.json"
            if not init_file.is_file():
                raise FileNotFoundError(
                    f"{self.dir} has no meta/episode_init_states.json; the seeds "
                    "the demonstrations were collected on are unknown"
                )
            states = json.loads(init_file.read_text())["episodes"]
            rows = self._row_index()
            episodes = []
            for key, entry in states.items():
                record = entry.get("arrays", {}).get("seed")
                if record is None:
                    raise ValueError(
                        f"{self.dir}: episode {key} carries no seed; cannot derive "
                        "the development seeds"
                    )
                index = int(key) if str(key).isdigit() else -1
                if index not in rows:
                    matches = [
                        i
                        for i, (path, _, _) in rows.items()
                        if path.name == entry.get("source_file")
                    ]
                    index = matches[0] if matches else index
                if index not in rows:
                    continue
                self._init_states[index] = entry
                episodes.append(
                    DemoEpisode(
                        index=index,
                        source_file=str(entry.get("source_file", f"episode_{index}")),
                        seed=int(np.asarray(record["data"]).ravel()[0]),
                        length=rows[index][2],
                    )
                )
            if not episodes:
                raise ValueError(f"{self.dir}: no demonstration episodes found")
            self._episodes = tuple(sorted(episodes, key=lambda e: e.index))
        return self._episodes

    @property
    def seeds(self) -> list[int]:
        """The seeds the demonstrations were collected on, sorted and unique.

        These are exactly the object placements the learned baselines train
        on, so they are the development seeds an agent is allowed to reset to.
        """
        return sorted({episode.seed for episode in self.episodes})

    def qpos(self, episode: DemoEpisode) -> np.ndarray:
        """Return one episode's ``full_qpos`` array, ``(T, nq)``.

        Args:
            episode: The episode to read.

        Returns:
            The per-frame simulator state.
        """
        qpos = self.columns(episode, ("full_qpos",))["full_qpos"]
        return np.asarray(qpos, dtype=np.float64)

    def columns(self, episode: DemoEpisode, names) -> dict[str, np.ndarray]:
        """Return some per-frame columns of one episode, ``(T, ...)`` each.

        Transition-scoped columns (``reward``, ``lowerbody_command``, ...)
        come back in the collector's convention, row t holding the transition
        that produced state t, exactly as
        :func:`bigym.loco.demos.dataset.load_episodes` returns them. Columns
        the export does not have are left out.

        Args:
            episode: The episode to read.
            names: The column names wanted.

        Returns:
            Column name to array, with the dtype and per-frame shape
            ``meta/info.json`` declares.
        """
        import pyarrow.parquet as pq

        from bigym.loco.demos.dataset import (
            load_info,
            load_transition_keys,
            undo_transition_shift,
        )

        features = load_info(self.dir).get("features", {})
        path, start, count = self._row_index()[episode.index]
        present = [n for n in names if n in pq.read_schema(path).names]
        table = pq.read_table(path, columns=present).slice(start, count)
        out = {}
        for name in present:
            array = np.stack([np.asarray(v) for v in table.column(name).to_pylist()])
            spec = features.get(name)
            if spec is not None:
                array = array.astype(spec["dtype"]).reshape((count, *spec["shape"]))
            out[name] = array
        _ = self.episodes  # reads meta/episode_init_states.json
        entry = self._init_states.get(episode.index, {})
        undo_transition_shift(
            out,
            entry.get("first_transition", {}),
            load_transition_keys(self.dir),
            where=f"{self.dir} episode {episode.index}",
        )
        return out


def pick_episodes(
    episodes: tuple[DemoEpisode, ...] | list[DemoEpisode],
    which: str = "median",
    count: int = 1,
) -> list[DemoEpisode]:
    """Choose which demonstrations to render.

    Args:
        episodes: The task's episodes.
        which: ``median`` for the episodes of median length (deterministic and
            representative), or an episode index to start from.
        count: How many episodes to return.

    Returns:
        The chosen episodes, in dataset order.

    Raises:
        ValueError: An explicit episode index that the task does not have.
    """
    episodes = list(episodes)
    count = max(1, min(int(count), len(episodes)))
    if str(which) == "median":
        order = sorted(range(len(episodes)), key=lambda i: (episodes[i].length, i))
        middle = len(order) // 2
        start = max(0, min(middle - count // 2, len(order) - count))
        chosen = [episodes[i] for i in order[start : start + count]]
    else:
        index = int(which)
        by_index = {
            episode.index: position for position, episode in enumerate(episodes)
        }
        if index not in by_index:
            raise ValueError(
                f"episode {index} is not in this task's demonstrations "
                f"(indices {episodes[0].index}..{episodes[-1].index})"
            )
        start = min(by_index[index], len(episodes) - count)
        chosen = episodes[start : start + count]
    return sorted(chosen, key=lambda e: e.index)


class _EpisodeArrays:
    """An ``NpzFile`` stand-in over in-memory arrays.

    :meth:`bigym.loco.demos.rerender.BatchRenderer.render_episode` reads an
    episode through the ``files`` / ``__getitem__`` npz interface; a dataset
    episode is handed to it through this wrapper instead of being written to
    disk first.
    """

    def __init__(self, arrays: dict):
        """Wrap a mapping of array name to array.

        Args:
            arrays: The arrays to serve.
        """
        self._arrays = dict(arrays)

    @property
    def files(self) -> list:
        """The array names, as ``NpzFile.files`` reports them."""
        return list(self._arrays)

    def __contains__(self, key) -> bool:
        """Whether an array is present."""
        return key in self._arrays

    def __getitem__(self, key):
        """Return one array."""
        return self._arrays[key]


def ffmpeg_binary() -> str:
    """Return the ffmpeg executable to encode with.

    Returns:
        The path of the ffmpeg binary: the one bundled with the
        ``imageio-ffmpeg`` wheel when it is installed, else the first
        ``ffmpeg`` on ``PATH``.

    Raises:
        RuntimeError: Neither is available.
    """
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:  # any import/lookup failure falls back
        found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError(
        "no ffmpeg available: install the imageio-ffmpeg wheel "
        "(uv pip install imageio-ffmpeg) or an ffmpeg binary on PATH"
    )


def write_video(
    path: Path,
    frames: np.ndarray,
    fps: int = 25,
    crf: str = "24",
    pix_fmt: str = "yuv420p",
) -> Path:
    """Encode RGB frames to an H.264 mp4.

    Args:
        path: Destination file.
        frames: ``(T, H, W, 3)`` uint8 RGB.
        fps: Frame rate written into the container.
        crf: x264 quality (0 is lossless).
        pix_fmt: Pixel format; ``yuv444p`` keeps full chroma, which matters at
            small resolutions where subsampling shifts colour edges.

    Returns:
        The written path.

    Raises:
        ValueError: An empty frame stack, or odd dimensions with a subsampled
            pixel format that cannot encode them.
        RuntimeError: ffmpeg failed.
    """
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[0] == 0:
        raise ValueError("frames must be a non-empty (T, H, W, 3) uint8 array")
    height, width = int(frames.shape[1]), int(frames.shape[2])
    if pix_fmt == "yuv420p" and (width % 2 or height % 2):
        raise ValueError(
            f"{width}x{height} has an odd side; yuv420p needs even dimensions "
            "(use --pix-fmt yuv444p)"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # rgb24 needs libx264rgb: plain libx264 converts to YUV and back, which
    # rounds by up to 2/255 even at crf 0.
    codec = "libx264rgb" if pix_fmt == "rgb24" else "libx264"
    command = [
        ffmpeg_binary(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(int(fps)), "-i", "-",
        "-c:v", codec, "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", pix_fmt, "-movflags", "+faststart", str(path),
    ]  # fmt: skip
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
        status = process.wait(timeout=600)
    if status != 0:
        raise RuntimeError(f"ffmpeg exited with status {status} writing {path}")
    return path


def _third_person_frames(renderer, qpos: np.ndarray, size: tuple[int, int]):
    """Render an outside view of one episode (humans only, never the agent)."""
    import mujoco

    from bigym.loco.demos.kinematic import pose

    width, height = size
    model, data = renderer.model, renderer.data
    camera: object = -1
    for name in THIRD_PERSON_CAMERAS:
        found = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if found >= 0:
            camera = int(found)
            break
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    if camera == -1:
        free = mujoco.MjvCamera()
        free.distance, free.elevation, free.azimuth = 3.0, -15.0, 160.0
        camera = free
    view = mujoco.Renderer(model, height, width)
    out = np.empty((len(qpos), height, width, 3), dtype=np.uint8)
    try:
        for t in range(len(qpos)):
            pose(renderer.inner, qpos[t])
            view.update_scene(data, camera)
            out[t] = view.render()
    finally:
        view.close()
    return out


def render_demo(
    task: str,
    out_dir: Path | str,
    *,
    size: tuple[int, int] = (84, 84),
    views: tuple[str, ...] | list[str] = DEFAULT_VIEWS,
    which: str = "median",
    episodes: int = 1,
    every: int = 2,
    fps: int = 25,
    crf: str = "24",
    pix_fmt: str = "yuv420p",
    prefix: str = "demo_",
    task_dir: Path | str | None = None,
    demos: TaskDemos | None = None,
) -> dict:
    """Render demonstrations of a task to one mp4 per view.

    Args:
        task: Task name.
        out_dir: Directory the mp4 files are written to.
        size: Onboard-camera resolution ``(width, height)``.
        views: Views to render (see :data:`ONBOARD_VIEWS` and ``third``).
        which: ``median`` or an episode index; see :func:`pick_episodes`.
        episodes: How many demonstrations to concatenate.
        every: Record one frame per this many control steps.
        fps: Frame rate of the mp4 files.
        crf: x264 quality.
        pix_fmt: Pixel format.
        prefix: File name prefix (``demo_head.mp4`` by default).
        task_dir: An already downloaded task folder, or None to fetch it.
        demos: An open :class:`TaskDemos` to reuse.

    Returns:
        A record of what was rendered: the files, the episodes, the seeds,
        the frame count and the encoding settings.

    Raises:
        ValueError: An unknown view name.
    """
    from bigym.loco.demos.rerender import BatchRenderer

    views = [str(v).strip() for v in views if str(v).strip()]
    unknown = [v for v in views if v not in ONBOARD_VIEWS and v != "third"]
    if unknown:
        raise ValueError(
            f"unknown view(s) {unknown}: choose from {list(ONBOARD_VIEWS) + ['third']}"
        )
    demos = demos or TaskDemos(task, task_dir)
    chosen = pick_episodes(demos.episodes, which, episodes)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    every = max(1, int(every))

    metadata = demos.metadata
    camera_keys = tuple(metadata["task"].get("camera_keys", ("head",)))
    renderer = BatchRenderer(metadata, (size[1], size[0]), verbose=False)
    panels: dict[str, list[np.ndarray]] = {view: [] for view in views}
    frames_per_episode = []
    try:
        for episode in chosen:
            qpos = demos.qpos(episode)[::every]
            frames_per_episode.append(int(len(qpos)))
            onboard = [v for v in views if v in ONBOARD_VIEWS]
            if onboard:
                rendered = renderer.render_episode(
                    _EpisodeArrays({"full_qpos": qpos, "seed": np.int64(episode.seed)})
                )
                for view in onboard:
                    index = camera_keys.index(view)
                    panels[view].append(rendered[:, index].transpose(0, 2, 3, 1))
            if "third" in views:
                panels["third"].append(_third_person_frames(renderer, qpos, size))
    finally:
        renderer.close()

    written = []
    for view in views:
        stack = np.concatenate(panels[view], axis=0)
        path = out_dir / f"{prefix}{view}.mp4"
        write_video(path, stack, fps=fps, crf=crf, pix_fmt=pix_fmt)
        written.append(path)
    from bigym.loco.demos import hub

    record = {
        "task": task,
        "dataset_repo": hub.dataset_repo(),
        "dataset_revision": hub.dataset_revision(),
        "dataset_dir": str(demos.dir),
        "views": list(views),
        "files": [path.name for path in written],
        "episodes": [
            {
                "index": episode.index,
                "source_file": episode.source_file,
                "seed": episode.seed,
                "steps": episode.length,
            }
            for episode in chosen
        ],
        "seeds": [episode.seed for episode in chosen],
        "frames": int(sum(frames_per_episode)),
        "frames_per_episode": frames_per_episode,
        "every": every,
        "fps": int(fps),
        "size": f"{size[0]}x{size[1]}",
        "crf": str(crf),
        "pix_fmt": pix_fmt,
        "source": "re-rendered from full_qpos",
    }
    for path in written:
        # The local cache path stays out of the sidecar: these files are read
        # inside the sandbox, which is told nothing about the host it runs on.
        sidecar = {k: v for k, v in record.items() if k != "dataset_dir"}
        sidecar["view"] = path.stem[len(prefix) :] if prefix else path.stem
        path.with_suffix(".json").write_text(json.dumps(sidecar, indent=1))
    return record


def main(argv: list[str] | DemoVideoConfig | None = None) -> int:
    """Render a task's demonstrations to mp4 from the command line.

    Args:
        argv: The parsed settings, or command line arguments (None for
            ``sys.argv[1:]``).

    Returns:
        The process exit status.
    """
    args = parse_command(DemoVideoConfig, argv)
    try:
        record = render_demo(
            args.task,
            args.out,
            size=parse_size(args.size),
            views=args.views.split(","),
            which=args.episode,
            episodes=args.episodes,
            every=args.every,
            fps=args.fps,
            crf=args.crf,
            pix_fmt=args.pix_fmt,
            prefix=args.prefix,
        )
    except Exception as error:
        print(f"bigym-agent demo-video: {error}", file=sys.stderr)
        return 1
    seeds = ", ".join(str(s) for s in record["seeds"])
    print(
        f"{args.task}: seeds {seeds}, {record['frames']} frames -> "
        + ", ".join(str(Path(args.out) / name) for name in record["files"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
