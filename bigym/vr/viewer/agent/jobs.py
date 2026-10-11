"""The ``bigym-agent`` replays and evaluations the viewer starts, and their progress."""

from __future__ import annotations

import contextlib
import subprocess
import sys
import time
from pathlib import Path

from bigym.loco.agent.snapshots import version_dir
from bigym.vr.viewer.agent.runs import as_int, file_digest, read_json
from bigym.vr.viewer.episodes import episode_files

AGENT_CLI_MODULE = "bigym.loco.agent"


# Subprocess output goes to a file under the cell so the main loop never
# blocks reading a pipe (and so a failure can be quoted back into the panel).
JOB_LOG_DIR = "viewer_jobs"


# A running replay or evaluation writes this file in its output directory
# (bigym.loco.agent.batch.RunProgress); it is what the progress bars read.
PROGRESS_NAME = ".progress.json"


# What ``bigym-agent evaluate`` runs when nothing says otherwise, so a bar
# has a denominator before the first episode lands in episodes.csv.
DEFAULT_EVAL_EPISODES = 100


# ---------------------------------------------------------------------------
# Progress of a running rollout (the ``.progress.json`` its process writes)
# ---------------------------------------------------------------------------


def read_progress(out_dir: Path) -> dict | None:
    """Read the ``.progress.json`` of a running replay or evaluation.

    Args:
        out_dir: The job's output directory (``replays/vNNN``, ``eval/vNNN``).

    Returns:
        The state dict written by ``bigym.loco.agent.batch.RunProgress``
        (``seed``, ``step``, ``max_steps``, ``episode``, ``episodes``,
        ``done``, ``pid``), or None when the job has not written one yet —
        which is also what a job that has just started looks like.
    """
    return read_json(Path(out_dir) / PROGRESS_NAME)


def progress_fraction(state: dict | None) -> float:
    """How far a run has got, 0 to 1; 0 when nothing is known yet.

    One episode is ``step / max_steps``. A run of several episodes counts
    the finished ones too, so the bar crosses the whole run once rather
    than snapping back to the left at every episode. An evaluation reports
    no steps at all (its episodes run in worker processes), and then the
    fraction is the finished episodes over the requested ones.

    Args:
        state: A dict from :func:`read_progress`, or None.

    Returns:
        A fraction in ``[0, 1]``.
    """
    if not state:
        return 0.0
    step = as_int(state.get("step")) or 0
    total = as_int(state.get("max_steps")) or 0
    episodes = as_int(state.get("episodes")) or 0
    finished = as_int(state.get("episode")) or 0
    inside = min(1.0, step / total) if total > 0 else 0.0
    if episodes > 1:
        return max(0.0, min(1.0, (finished + inside) / episodes))
    if total > 0:
        return inside
    return max(0.0, min(1.0, finished / episodes)) if episodes > 0 else 0.0


def progress_label(head: str, state: dict | None) -> str:
    """Name one running job and say where it is.

    ``astra_high_s2 · v002 · seed 500 · 812/3523`` — ``head`` is whatever
    names the job (a compare slot, a version), and what follows is the step
    inside the episode, plus ``k/n episodes`` when the run is more than one.
    A job that has not written its progress file yet reads ``starting…``.

    Args:
        head: The job's name.
        state: A dict from :func:`read_progress`, or None.

    Returns:
        The label line.
    """
    if not state:
        return f"{head} · starting…"
    step = as_int(state.get("step")) or 0
    total = as_int(state.get("max_steps")) or 0
    episodes = as_int(state.get("episodes")) or 0
    finished = as_int(state.get("episode")) or 0
    parts = [head]
    if total > 0:
        parts.append(f"{step}/{total}")
    if episodes > 1 or total <= 0:
        parts.append(f"{finished}/{episodes} episodes")
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# bigym-agent subprocesses
# ---------------------------------------------------------------------------


class AgentJob:
    """One ``bigym-agent`` subcommand run for the viser panels.

    The subprocess writes its output to a log file under the cell, so the
    viewer's main loop never blocks on a pipe; progress is read instead from
    the files the subcommand produces (replay npz count, evaluation CSV
    rows). Start/poll are main-loop operations: GUI callbacks only request.

    The same class serves the Policy versions panel (one cell, the cell on
    screen) and compare mode, which fills a missing seed in any cell of any
    session — the cell is a constructor argument, so nothing about the job
    is tied to the rollout being viewed.
    """

    def __init__(
        self,
        kind: str,
        cell: Path,
        version: int,
        seeds: str = "",
        policy: str | Path = "",
        extra: tuple[str, ...] = (),
    ):
        """Describe a run of one policy version, or of a loose policy file.

        Args:
            kind: ``replay`` or ``evaluate``.
            cell: The cell directory. Any cell: compare mode passes the cell
                of the slot that is missing an episode.
            version: The policy version, ignored when ``policy`` is given.
            seeds: Seed specification for ``replay``.
            policy: A policy file to replay instead of a recorded version;
                its batch lands in ``<cell>/replays/policy_<sha8>``.
            extra: Further command-line arguments (compare mode passes
                ``--no-video``: it needs the batch, not the mp4).
        """
        self.kind = str(kind)
        self.cell = Path(cell)
        self.version = int(version)
        self.seeds = str(seeds or "")
        self.policy = str(policy or "")
        self.extra = tuple(extra)
        self.digest = file_digest(Path(self.policy)) if self.policy else ""
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"policy_{self.digest[:8]}" if self.policy else f"v{self.version:03d}"
        self.log_path = self.cell / JOB_LOG_DIR / f"{self.kind}_{name}_{stamp}.log"
        self.proc: subprocess.Popen | None = None
        self.error: str | None = None

    @property
    def label(self) -> str:
        """What this job is called in the panel and the log lines."""
        return f"policy_{self.digest[:8]}" if self.policy else f"v{self.version:03d}"

    @property
    def command(self) -> list[str]:
        """The argv this job runs (the ``bigym-agent`` module entry point)."""
        cmd = [sys.executable, "-m", AGENT_CLI_MODULE, self.kind, str(self.cell)]
        if self.policy:
            cmd += ["--policy", self.policy]
        else:
            cmd += ["--version", str(self.version)]
        if self.kind == "replay" and self.seeds:
            cmd += ["--seeds", self.seeds]
        return cmd + list(self.extra)

    @property
    def out_dir(self) -> Path:
        """Where the subcommand writes this job's outputs."""
        kind_dir = "replays" if self.kind == "replay" else "eval"
        if self.policy:
            # Matches bigym-agent replay --policy: the file's content hash
            # names the directory, so replaying the same file twice reuses it.
            return self.cell / kind_dir / f"policy_{self.digest[:8]}"
        return version_dir(self.cell, self.version, kind_dir)

    def start(self) -> None:
        """Launch the subprocess; a launch failure lands in ``error``."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            log = self.log_path.open("w")
        except OSError as exc:
            self.error = f"cannot write {self.log_path}: {exc}"
            return
        try:
            self.proc = subprocess.Popen(
                self.command, stdout=log, stderr=subprocess.STDOUT
            )
        except OSError as exc:
            log.close()
            self.error = f"cannot run {' '.join(self.command)}: {exc}"

    @property
    def episodes(self) -> int:
        """How many episodes an evaluation was asked for (``--episodes``)."""
        argv = self.command
        for i, arg in enumerate(argv):
            if arg == "--episodes" and i + 1 < len(argv):
                return as_int(argv[i + 1]) or DEFAULT_EVAL_EPISODES
            if arg.startswith("--episodes="):
                return as_int(arg.split("=", 1)[1]) or DEFAULT_EVAL_EPISODES
        return DEFAULT_EVAL_EPISODES

    def csv_rows(self) -> int:
        """Episodes already scored, counted in ``episodes.csv``."""
        episodes_csv = self.out_dir / "episodes.csv"
        if not episodes_csv.exists():
            return 0
        try:
            with episodes_csv.open() as f:
                return max(0, sum(1 for _ in f) - 1)
        except OSError:
            return 0

    def progress(self) -> str:
        """One line of progress, polled from the output directory."""
        if self.kind == "replay":
            written = len(episode_files(self.out_dir / "batch"))
            return f"replay {self.label}: {written} episodes written"
        return f"evaluate {self.label}: {self.csv_rows()} episodes scored"

    def progress_state(self) -> tuple[float, str, bool]:
        """``(fraction, label, animated)`` for this job's progress bar.

        Prefers the ``.progress.json`` the subprocess writes (step level for
        a replay, episode level for an evaluation) and falls back to the
        files it has finished: an evaluation's ``episodes.csv`` rows over
        ``--episodes``, a replay's episode count. Before either exists the
        bar is animated and says ``starting…``, because a rollout can take a
        minute to build its environment and a bar stuck at zero with no
        motion reads as a hang.

        Returns:
            The fraction in ``[0, 1]``, the label line, and whether the bar
            should animate (nothing is known yet).
        """
        state = read_progress(self.out_dir)
        head = f"{self.kind} {self.label}"
        if state:
            return progress_fraction(state), progress_label(head, state), False
        if self.kind == "evaluate":
            rows, total = self.csv_rows(), max(1, self.episodes)
            if rows:
                return (
                    min(1.0, rows / total),
                    f"{head} · {rows}/{total} episodes",
                    False,
                )
        return 0.0, f"{head} · starting…", True

    def poll(self) -> bool:
        """True once the job is over; a non-zero exit sets ``error``."""
        if self.error is not None or self.proc is None:
            return True
        code = self.proc.poll()
        if code is None:
            return False
        if code != 0:
            self.error = self.diagnose(code)
        return True

    def stop(self) -> None:
        """Kill the subprocess: nobody is waiting for its output any more.

        Pressing Leave while compare mode is still producing its rollouts
        must not leave a replay running for another twenty minutes, writing
        into a cell the viewer has stopped looking at.
        """
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        with contextlib.suppress(OSError):
            proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                proc.wait(timeout=5.0)

    def diagnose(self, code: int) -> str:
        """Turn a non-zero exit into a message the panel can show."""
        tail = ""
        try:
            tail = self.log_path.read_text(errors="replace")[-4000:]
        except OSError:
            pass
        # Only the dispatcher's own refusals mean "no such subcommand"; a
        # missing module inside the agent's policy is a job failure and is
        # reported with its own last line below.
        unknown = any(
            marker in tail
            for marker in (
                f"No module named '{AGENT_CLI_MODULE}'",
                "No module named 'bigym'",
                "cannot be directly executed",
                "Unrecognized options",
                "invalid choice",
                "unknown command",
                "unknown subcommand",
                "unrecognized arguments",
            )
        )
        if unknown:
            return (
                f"this install has no working 'bigym-agent {self.kind}' "
                f"subcommand ({AGENT_CLI_MODULE} refused it); install "
                "bigym[agent] and try again"
            )
        lines = [line for line in tail.splitlines() if line.strip()]
        detail = lines[-1] if lines else f"see {self.log_path}"
        return f"{self.kind} failed (exit {code}): {detail}"
